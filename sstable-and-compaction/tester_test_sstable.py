"""Tests for SSTable implementation."""
import os
import sys
import tempfile

from sstable import (
    SSTableWriter, SSTableReader, SSTableEntry, SSTableMetadata,
    CompactionManager, merge_sstables, MAGIC
)


def make_sstable(tmpdir, name, entries, block_size=4):
    """Helper: write entries as (key, value, timestamp) to an SSTable."""
    path = os.path.join(tmpdir, name)
    w = SSTableWriter(path, block_size=block_size)
    for k, v, ts in entries:
        w.add(k, v, ts)
    meta = w.finish()
    return path, meta


class TestWriteReadRoundTrip:
    """Test 1: SSTable write/read round-trip."""

    def test_basic_round_trip(self, tmp_path):
        path, meta = make_sstable(str(tmp_path), "t1.sst", [
            ("apple", "red", 1.0),
            ("banana", "yellow", 1.0),
            ("cherry", "dark red", 1.0),
            ("date", "brown", 1.0),
            ("elderberry", "purple", 1.0),
        ])
        assert meta.entry_count == 5
        assert meta.min_key == "apple"
        assert meta.max_key == "elderberry"

        reader = SSTableReader(path)
        entries = list(reader.scan())
        assert len(entries) == 5
        assert entries[0].key == "apple" and entries[0].value == "red"
        assert entries[4].key == "elderberry" and entries[4].value == "purple"

    def test_point_lookup(self, tmp_path):
        path, _ = make_sstable(str(tmp_path), "t2.sst", [
            ("apple", "red", 1.0),
            ("banana", "yellow", 1.0),
            ("cherry", "dark red", 1.0),
        ])
        reader = SSTableReader(path)
        e = reader.get("cherry")
        assert e is not None and e.value == "dark red"
        assert reader.get("fig") is None
        assert reader.get("aaa") is None


class TestRangeScan:
    """Test 3: Range scan boundary handling."""

    def test_range_scan(self, tmp_path):
        path, _ = make_sstable(str(tmp_path), "r.sst", [
            ("apple", "1", 1.0),
            ("banana", "2", 1.0),
            ("cherry", "3", 1.0),
            ("date", "4", 1.0),
            ("elderberry", "5", 1.0),
        ])
        reader = SSTableReader(path)
        results = list(reader.range_scan("banana", "elderberry"))
        assert len(results) == 3  # banana, cherry, date
        assert results[0].key == "banana"
        assert results[-1].key == "date"

    def test_range_scan_no_match(self, tmp_path):
        path, _ = make_sstable(str(tmp_path), "r2.sst", [
            ("apple", "1", 1.0), ("banana", "2", 1.0),
        ])
        reader = SSTableReader(path)
        assert list(reader.range_scan("cat", "dog")) == []


class TestTombstones:
    """Test 4: Tombstone handling."""

    def test_tombstone_write_read(self, tmp_path):
        path, _ = make_sstable(str(tmp_path), "ts.sst", [
            ("key1", None, 1.0),
            ("key2", "val", 1.0),
        ])
        reader = SSTableReader(path)
        e = reader.get("key1")
        assert e is not None and e.value is None
        e2 = reader.get("key2")
        assert e2 is not None and e2.value == "val"


class TestMerge:
    """Tests 5-7: K-way merge, conflict resolution, tombstone removal."""

    def test_two_way_merge_conflict_resolution(self, tmp_path):
        d = str(tmp_path)
        p1, _ = make_sstable(d, "m1.sst", [
            ("apple", "red", 1.0), ("banana", "yellow", 1.0),
        ])
        p2, _ = make_sstable(d, "m2.sst", [
            ("apple", "green", 2.0), ("cherry", "dark", 2.0),
        ])
        r1, r2 = SSTableReader(p1), SSTableReader(p2)
        merged = merge_sstables([r2, r1], os.path.join(d, "merged.sst"))
        entries = list(merged.scan())
        assert len(entries) == 3
        assert entries[0].key == "apple" and entries[0].value == "green"
        assert entries[0].timestamp == 2.0

    def test_tombstone_removal(self, tmp_path):
        d = str(tmp_path)
        p1, _ = make_sstable(d, "t1.sst", [
            ("apple", "red", 1.0), ("banana", "yellow", 1.0),
        ])
        p2, _ = make_sstable(d, "t2.sst", [
            ("banana", None, 2.0),  # tombstone
        ])
        r1, r2 = SSTableReader(p1), SSTableReader(p2)
        merged = merge_sstables([r2, r1], os.path.join(d, "merged.sst"),
                                remove_tombstones=True)
        entries = list(merged.scan())
        assert len(entries) == 1
        assert entries[0].key == "apple"

    def test_five_way_merge(self, tmp_path):
        d = str(tmp_path)
        readers = []
        for i in range(5):
            p, _ = make_sstable(d, f"m{i}.sst", [
                (f"key{j:03d}", f"val_{i}_{j}", float(i))
                for j in range(3)
            ])
            readers.append(SSTableReader(p))
        merged = merge_sstables(readers, os.path.join(d, "merged5.sst"))
        entries = list(merged.scan())
        assert len(entries) == 3
        # Latest timestamp (4.0) should win for each key
        for e in entries:
            assert e.timestamp == 4.0


class TestCompaction:
    """Tests 8-9: Size-tiered and leveled compaction."""

    def test_size_tiered_compaction(self, tmp_path):
        d = str(tmp_path)
        manager = CompactionManager(d, strategy="size_tiered", min_threshold=2)
        for i in range(3):
            p, _ = make_sstable(d, f"stcs{i}.sst", [
                (f"key{i}_{j}", f"val{j}", 1.0) for j in range(5)
            ])
            manager.add_sstable(SSTableReader(p))
        assert manager.needs_compaction()
        result = manager.run_compaction()
        assert len(result) >= 1
        # All keys should be present in merged result
        all_entries = list(result[0].scan())
        assert len(all_entries) == 15

    def test_leveled_compaction(self, tmp_path):
        d = str(tmp_path)
        manager = CompactionManager(d, strategy="leveled",
                                     l0_compaction_trigger=2,
                                     level_base_size=100)
        for i in range(3):
            p, _ = make_sstable(d, f"lcs{i}.sst", [
                (f"key{i}_{j}", f"val{j}", float(i)) for j in range(3)
            ])
            manager.add_sstable(SSTableReader(p))
        assert manager.needs_compaction()
        result = manager.run_compaction()
        assert len(result) >= 1
        assert result[0].level == 1


class TestEdgeCases:
    """Test 10: Edge cases."""

    def test_empty_sstable(self, tmp_path):
        path, meta = make_sstable(str(tmp_path), "empty.sst", [])
        assert meta.entry_count == 0
        reader = SSTableReader(path)
        assert reader.get("anything") is None
        assert list(reader.scan()) == []

    def test_single_entry(self, tmp_path):
        path, meta = make_sstable(str(tmp_path), "single.sst", [
            ("only", "one", 1.0),
        ])
        assert meta.entry_count == 1
        assert meta.min_key == "only" and meta.max_key == "only"
        reader = SSTableReader(path)
        assert reader.get("only").value == "one"

    def test_all_tombstones_merge(self, tmp_path):
        d = str(tmp_path)
        p, _ = make_sstable(d, "tomb.sst", [
            ("a", None, 1.0), ("b", None, 1.0), ("c", None, 1.0),
        ])
        r = SSTableReader(p)
        merged = merge_sstables([r], os.path.join(d, "merged_tomb.sst"),
                                remove_tombstones=True)
        assert list(merged.scan()) == []


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
