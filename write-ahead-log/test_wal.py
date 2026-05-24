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
        assert wal.current_seq_num() == 7
        cp_seq = wal.checkpoint()
        assert cp_seq == 8
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
        assert len(all_recs) == 4, f"expected 4, got {len(all_recs)}"
        assert all_recs[3].op_type == "COMMIT"
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
    print("\nALL TESTS PASSED")
