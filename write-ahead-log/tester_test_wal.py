"""Tests for WAL implementation."""
import tempfile, os, sys, glob as glob_mod
from wal import WriteAheadLog


def test_basic_append_and_replay():
    """Spec example: individual writes round-trip correctly."""
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
    """Reopen WAL without close, verify fsynced data recovered."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "k1", "v1")
        wal.append("PUT", "k2", "v2")
        # Don't close - simulate crash
        wal2 = WriteAheadLog(tmpdir, sync_mode="sync")
        records = wal2.replay()
        assert len(records) == 2
        assert records[0].key == "k1"
        assert records[1].key == "k2"
        # Sequence numbers persist across restarts
        seq = wal2.append("PUT", "k3", "v3")
        assert seq == 3
        wal2.close()
    print("PASSED: crash recovery")


def test_corruption_stops_replay():
    """Corrupted records cause replay to stop; prior records still returned."""
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
        assert records[0].key == "a"
        wal2.close()
    print("PASSED: corruption stops replay")


def test_truncation():
    """Records before truncation point are removed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "a", "1")
        wal.append("PUT", "b", "2")
        wal.append("PUT", "c", "3")
        wal.truncate(2)
        records = wal.replay()
        assert len(records) == 1, f"expected 1, got {len(records)}"
        assert records[0].key == "c"
        assert records[0].seq_num == 3
        wal.close()
    print("PASSED: truncation")


def test_log_rotation():
    """Records span multiple WAL files when size limit exceeded."""
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


def test_iterate_includes_commit():
    """iterate() yields all raw records including COMMIT."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "x", "1")
        wal.append_batch([("PUT", "y", "2"), ("PUT", "z", "3")])
        all_recs = list(wal.iterate())
        # 1 standalone + 2 batch ops + 1 COMMIT = 4
        assert len(all_recs) == 4, f"expected 4, got {len(all_recs)}"
        assert all_recs[3].op_type == "COMMIT"
        wal.close()
    print("PASSED: iterate includes COMMIT")


def test_empty_wal():
    """Empty WAL replays nothing, seq starts at 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        assert wal.current_seq_num() == 0
        records = wal.replay()
        assert len(records) == 0
        wal.close()
    print("PASSED: empty WAL")


def test_large_values():
    """Large values round-trip correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        big_val = "x" * 100000
        wal.append("PUT", "big", big_val)
        records = wal.replay()
        assert len(records) == 1
        assert records[0].value == big_val
        assert records[0].key == "big"
        wal.close()
    print("PASSED: large values")


def test_checkpoint_replay_after():
    """Replay with after_seq only returns records after checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "before1", "v1")
        wal.append("PUT", "before2", "v2")
        cp = wal.checkpoint()
        wal.append("PUT", "after1", "v3")
        wal.append("PUT", "after2", "v4")
        records = wal.replay(after_seq=cp)
        assert len(records) == 2, f"expected 2, got {len(records)}"
        assert records[0].key == "after1"
        assert records[1].key == "after2"
        wal.close()
    print("PASSED: checkpoint replay after")


def test_sequence_monotonic_across_restart():
    """Sequence numbers never reset, even across restarts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal = WriteAheadLog(tmpdir, sync_mode="sync")
        wal.append("PUT", "a", "1")
        wal.append("PUT", "b", "2")
        wal.close()
        wal2 = WriteAheadLog(tmpdir, sync_mode="sync")
        seq = wal2.append("PUT", "c", "3")
        assert seq == 3, f"expected seq 3, got {seq}"
        wal2.close()
    print("PASSED: sequence monotonic across restart")


if __name__ == "__main__":
    test_basic_append_and_replay()
    test_crash_recovery()
    test_corruption_stops_replay()
    test_truncation()
    test_log_rotation()
    test_iterate_includes_commit()
    test_empty_wal()
    test_large_values()
    test_checkpoint_replay_after()
    test_sequence_monotonic_across_restart()
    print("\nALL TESTS PASSED")
