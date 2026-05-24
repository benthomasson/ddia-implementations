"""Tests for LSM Tree storage engine."""

import tempfile

from lsm import LSMTree


def test_basic_crud():
    """Test put, get, update, delete from the spec example."""
    with tempfile.TemporaryDirectory() as d:
        db = LSMTree(d, memtable_threshold=3)
        db.put("apple", "red")
        db.put("banana", "yellow")
        assert db.get("apple") == "red"
        assert db.get("cherry") is None

        # Trigger flush (threshold=3) and continue
        db.put("cherry", "dark red")
        db.put("date", "brown")

        assert db.get("apple") == "red"
        assert db.get("date") == "brown"

        # Update
        db.put("apple", "green")
        assert db.get("apple") == "green"

        # Delete
        db.delete("banana")
        assert db.get("banana") is None

        # Range scan
        results = db.range_scan("apple", "date")
        assert results == [("apple", "green"), ("cherry", "dark red")]

        db.close()


def test_flush_and_sstable_reads():
    """Verify data persists after flush and reads work from SSTable."""
    with tempfile.TemporaryDirectory() as d:
        db = LSMTree(d, memtable_threshold=3)
        db.put("a", "1")
        db.put("b", "2")
        db.put("c", "3")
        # Should auto-flush, memtable now empty
        db.put("d", "4")
        # a,b,c should be in SSTable
        assert db.get("a") == "1"
        assert db.get("b") == "2"
        assert db.get("c") == "3"
        assert db.get("d") == "4"
        db.close()


def test_multiple_sstables_newest_wins():
    """Newest value wins across multiple SSTables."""
    with tempfile.TemporaryDirectory() as d:
        db = LSMTree(d, memtable_threshold=2, compaction_threshold=100)
        db.put("k", "v1")
        db.put("x", "x1")  # flush: k=v1, x=x1
        db.put("k", "v2")
        db.put("y", "y1")  # flush: k=v2, y=y1
        assert db.get("k") == "v2"
        db.close()


def test_compaction():
    """Verify compaction merges SSTables and removes tombstones."""
    with tempfile.TemporaryDirectory() as d:
        db = LSMTree(d, memtable_threshold=2, compaction_threshold=100)
        # Create several SSTables
        db.put("a", "1"); db.put("b", "2")  # flush
        db.put("c", "3"); db.put("d", "4")  # flush
        db.delete("a"); db.put("e", "5")     # flush (tombstone for a)

        sst_count_before = len(db._sstables)
        assert sst_count_before == 3

        db.compact()
        assert len(db._sstables) == 1

        # a was deleted, should be gone after compaction
        assert db.get("a") is None
        assert db.get("b") == "2"
        assert db.get("c") == "3"
        assert db.get("d") == "4"
        assert db.get("e") == "5"
        db.close()


def test_crash_recovery():
    """Simulate crash and verify WAL replay restores data."""
    with tempfile.TemporaryDirectory() as d:
        db = LSMTree(d, memtable_threshold=5)
        db.put("x", "1")
        db.put("y", "2")
        # Simulate crash: don't call close(), just reopen
        db2 = LSMTree(d, memtable_threshold=5)
        assert db2.get("x") == "1"
        assert db2.get("y") == "2"
        db2.close()


def test_large_dataset():
    """Insert 10,000+ keys and verify correctness."""
    with tempfile.TemporaryDirectory() as d:
        db = LSMTree(d, memtable_threshold=500, compaction_threshold=10)
        n = 10000
        for i in range(n):
            db.put(f"key_{i:06d}", f"val_{i}")
        # Spot check
        assert db.get("key_000000") == "val_0"
        assert db.get("key_005000") == "val_5000"
        assert db.get("key_009999") == "val_9999"
        assert db.get("key_999999") is None

        # Range scan subset
        results = db.range_scan("key_000100", "key_000105")
        assert len(results) == 5
        assert results[0] == ("key_000100", "val_100")
        db.close()


def test_edge_cases():
    """Empty db, single key, delete non-existent, empty range scan, overwrite same key."""
    with tempfile.TemporaryDirectory() as d:
        db = LSMTree(d, memtable_threshold=1000)

        # Empty db
        assert db.get("anything") is None
        assert db.range_scan("a", "z") == []

        # Single key
        db.put("only", "one")
        assert db.get("only") == "one"

        # Delete non-existent
        db.delete("ghost")
        assert db.get("ghost") is None

        # Overwrite same key many times
        for i in range(100):
            db.put("repeat", str(i))
        assert db.get("repeat") == "99"

        # Range scan with no results
        assert db.range_scan("zzz", "zzzz") == []

        db.close()


def test_range_scan_across_sources():
    """Range scan merges memtable and SSTables correctly."""
    with tempfile.TemporaryDirectory() as d:
        db = LSMTree(d, memtable_threshold=3, compaction_threshold=100)
        db.put("a", "1")
        db.put("b", "2")
        db.put("c", "3")  # flush
        db.put("b", "updated")  # in memtable, should override SSTable
        db.put("d", "4")        # in memtable

        results = db.range_scan("a", "e")
        assert results == [("a", "1"), ("b", "updated"), ("c", "3"), ("d", "4")]
        db.close()


if __name__ == "__main__":
    tests = [
        test_basic_crud,
        test_flush_and_sstable_reads,
        test_multiple_sstables_newest_wins,
        test_compaction,
        test_crash_recovery,
        test_large_dataset,
        test_edge_cases,
        test_range_scan_across_sources,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nRan {len(tests)} tests")
