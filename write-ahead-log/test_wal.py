"""Tests for WAL implementation."""
import tempfile, os, sys, glob as glob_mod
from wal import WriteAheadLog

def test_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        seq1 = wal.append("PUT", "user:1", "alice")
        seq2 = wal.append("PUT", "user:2", "bob")
        seq3 = wal.append("DELETE", "user:1")
        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3
        seq_commit = wal.append_batch([
            ("PUT", "order:1", "item_a"),
            ("PUT", "order:2", "item_b"),
            ("DELETE", "order:0", ""),
        ])
        assert wal.current_seq_num() == 8
        cp_seq = wal.checkpoint()
        assert cp_seq == 9
        records = wal.replay()
        assert len(records) == 6, f"expected 6, got {len(records)}"
        records = wal.replay(after_seq=cp_seq)
        assert len(records) == 0
        wal.close()
    print("PASSED: basic append, batch, checkpoint, replay")

def test_crash_recovery():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "k1", "v1")
        wal.append("PUT", "k2", "v2")
        wal2 = WriteAheadLog(tmpdir, sync_mode="sync")
        records = wal2.replay()
        assert len(records) == 2
        assert records[0].key == "k1"
        assert records[1].key == "k2"
        seq = wal2.append("PUT", "k3", "v3")
        assert seq == 3
        wal2.close()
    print("PASSED: crash recovery")

def test_corruption():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "a", "1")
        wal.append("PUT", "b", "2")
        wal.close()
        wal_files = glob_mod.glob(os.path.join(tmpdir, "*.wal"))
        with open(wal_files[0], "r+b") as f:
            f.seek(-5, 2)
            f.write(b"\xff\xff\xff\xff\xff")
        wal2 = WriteAheadLog(tmpdir, sync_mode="sync")
        records = wal2.replay()
        assert len(records) == 1, f"expected 1, got {len(records)}"
        wal2.close()
    print("PASSED: corruption detection")

def test_truncation():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "a", "1")
        wal.append("PUT", "b", "2")
        wal.append("PUT", "c", "3")
        wal.truncate(2)
        records = wal.replay()
        assert len(records) == 1, f"expected 1, got {len(records)}"
        assert records[0].key == "c"
        wal.close()
    print("PASSED: truncation")

def test_rotation():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync", max_file_size=100)
        for i in range(20):
            wal.append("PUT", f"key{i}", f"value{i}")
        files = [f for f in os.listdir(tmpdir) if f.endswith(".wal")]
        assert len(files) > 1, f"expected multiple files, got {len(files)}"
        records = wal.replay()
        assert len(records) == 20, f"expected 20, got {len(records)}"
        wal.close()
    print("PASSED: log rotation")

def test_iterate():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "x", "1")
        wal.append_batch([("PUT", "y", "2"), ("PUT", "z", "3")])
        all_recs = list(wal.iterate())
        assert len(all_recs) == 5, f"expected 5, got {len(all_recs)}"
        assert all_recs[1].op_type == "BEGIN"
        assert all_recs[4].op_type == "COMMIT"
        wal.close()
    print("PASSED: iterate")

def test_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        records = wal.replay()
        assert len(records) == 0
        assert wal.current_seq_num() == 0
        wal.close()
    print("PASSED: empty WAL")

def test_sync_modes():
    for mode in ["sync", "batch", "none"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WriteAheadLog(tmpdir, sync_mode=mode)
            wal.append("PUT", "k", "v")
            wal.close()
    print("PASSED: sync modes")

def test_large_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        big_val = "x" * 100000
        wal.append("PUT", "big", big_val)
        records = wal.replay()
        assert len(records) == 1
        assert records[0].value == big_val
        wal.close()
    print("PASSED: large values")

def test_torn_write():
    """Simulate a torn write (partial record at end of WAL) and verify replay skips it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "good1", "v1")
        wal.append("PUT", "good2", "v2")
        wal.close()

        # Append a partial record (torn write) to the WAL file
        wal_files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".wal"))
        wal_path = os.path.join(tmpdir, wal_files[-1])
        with open(wal_path, "ab") as f:
            # Write a valid length prefix but truncated record body
            import struct
            f.write(struct.pack("<I", 999))  # claims 999 bytes follow
            f.write(b"\x00" * 10)            # but only 10 bytes written

        wal2 = WriteAheadLog(tmpdir, sync_mode="sync")
        records = wal2.replay()
        assert len(records) == 2, f"expected 2 good records, got {len(records)}"
        assert records[0].key == "good1"
        assert records[1].key == "good2"
        wal2.close()
    print("PASSED: torn write recovery")

def test_incomplete_batch():
    """Simulate crash mid-batch: BEGIN + records but no COMMIT."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from wal import _encode_record, OP_BEGIN, OP_BYTES
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "solo", "v1")
        wal.append_batch([("PUT", "batch1", "v1"), ("PUT", "batch2", "v2")])
        # Manually write an incomplete batch (BEGIN + PUT, no COMMIT)
        wal_path = wal._current_file
        seq = wal.current_seq_num()
        with open(wal_path, "ab") as f:
            f.write(_encode_record(seq + 1, OP_BEGIN, b"", b""))
            f.write(_encode_record(seq + 2, OP_BYTES["PUT"],
                                   b"orphan", b"should_not_appear"))
            f.flush()
            os.fsync(f.fileno())

        wal2 = WriteAheadLog(tmpdir, sync_mode="sync")
        records = wal2.replay()
        keys = [r.key for r in records]
        assert "solo" in keys, "individual record should survive"
        assert "batch1" in keys, "complete batch should survive"
        assert "batch2" in keys, "complete batch should survive"
        assert "orphan" not in keys, "incomplete batch should be discarded"
        assert len(records) == 3, f"expected 3, got {len(records)}"
        wal2.close()
    print("PASSED: incomplete batch discarded")

if __name__ == "__main__":
    test_basic()
    test_crash_recovery()
    test_corruption()
    test_truncation()
    test_rotation()
    test_iterate()
    test_empty()
    test_sync_modes()
    test_large_values()
    test_torn_write()
    test_incomplete_batch()
    print("\nALL TESTS PASSED")
