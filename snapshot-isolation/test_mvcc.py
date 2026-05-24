"""Tests for MVCC snapshot isolation database."""
import pytest
from mvcc_database import MVCCDatabase, TransactionError


def test_example_usage():
    """Test the example from the spec."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "account_A", 1000)
    db.write(tx1, "account_B", 2000)
    db.commit(tx1)

    tx2 = db.begin_transaction(read_only=True)
    assert db.read(tx2, "account_A") == 1000

    tx3 = db.begin_transaction()
    db.write(tx3, "account_A", 900)
    db.commit(tx3)

    assert db.read(tx2, "account_A") == 1000

    tx4 = db.begin_transaction()
    assert db.read(tx4, "account_A") == 900

    tx5 = db.begin_transaction()
    tx6 = db.begin_transaction()
    db.write(tx5, "account_A", 800)
    db.write(tx6, "account_A", 850)
    assert db.commit(tx5) == True
    assert db.commit(tx6) == False

    tx7 = db.begin_transaction()
    assert db.read(tx7, "account_A") == 800


def test_1_read_after_write():
    """Test basic read-after-write within a single transaction."""
    db = MVCCDatabase()
    tx = db.begin_transaction()
    db.write(tx, "x", 42)
    assert db.read(tx, "x") == 42


def test_2_no_uncommitted_writes():
    """Concurrent transaction doesn't see uncommitted writes."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    tx2 = db.begin_transaction()
    assert db.read(tx2, "x") is None


def test_3_no_writes_after_start():
    """Concurrent transaction doesn't see writes committed after its start."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    assert db.read(tx2, "x") == 10

    tx3 = db.begin_transaction()
    db.write(tx3, "x", 20)
    db.commit(tx3)

    # tx2 still sees old value
    assert db.read(tx2, "x") == 10


def test_4_write_write_conflict():
    """Write-write conflict detection with first-committer-wins."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    tx3 = db.begin_transaction()
    db.write(tx2, "x", 2)
    db.write(tx3, "x", 3)
    assert db.commit(tx2) == True
    assert db.commit(tx3) == False


def test_5_aborted_invisible():
    """Aborted transactions' writes are invisible."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    db.abort(tx1)

    tx2 = db.begin_transaction()
    assert db.read(tx2, "x") is None


def test_6_delete_visibility():
    """Delete operation and its visibility rules."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    db.commit(tx1)

    tx2 = db.begin_transaction()  # snapshot before delete
    tx3 = db.begin_transaction()
    assert db.delete(tx3, "x") == True
    db.commit(tx3)

    # tx2 still sees the value
    assert db.read(tx2, "x") == 10

    # new tx sees deletion
    tx4 = db.begin_transaction()
    assert db.read(tx4, "x") is None


def test_7_readonly_never_aborts():
    """Read-only transactions never abort."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    db.commit(tx1)

    tx_ro = db.begin_transaction(read_only=True)
    assert db.commit(tx_ro) == True

    with pytest.raises(TransactionError):
        db.write(tx_ro, "y", 20)

    with pytest.raises(TransactionError):
        db.delete(tx_ro, "x")


def test_8_scan():
    """Scan operation returns correct snapshot."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "user:1", "alice")
    db.write(tx1, "user:2", "bob")
    db.write(tx1, "item:1", "widget")
    db.commit(tx1)

    tx2 = db.begin_transaction()
    result = db.scan(tx2, "user:")
    assert result == {"user:1": "alice", "user:2": "bob"}

    all_items = db.scan(tx2)
    assert len(all_items) == 3


def test_9_gc_removes_unreachable():
    """Garbage collection removes unreachable versions."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    db.write(tx2, "x", 2)
    db.commit(tx2)

    tx3 = db.begin_transaction()
    db.write(tx3, "x", 3)
    db.commit(tx3)

    assert db.get_version_count("x") >= 3
    removed = db.garbage_collect()
    assert removed > 0
    # Should keep only the latest version
    assert db.get_version_count("x") == 1


def test_10_gc_preserves_active():
    """GC does not remove versions needed by active transactions."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx_long = db.begin_transaction(read_only=True)  # holds snapshot

    tx2 = db.begin_transaction()
    db.write(tx2, "x", 2)
    db.commit(tx2)

    db.garbage_collect()

    # Long-running tx still sees old value
    assert db.read(tx_long, "x") == 1


def test_11_multiple_versions():
    """Multiple versions of the same key coexist correctly."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    db.write(tx2, "x", 2)
    db.commit(tx2)

    assert db.get_version_count("x") >= 2

    tx3 = db.begin_transaction()
    assert db.read(tx3, "x") == 2


def test_12_read_own_writes():
    """Transaction reads its own writes before commit."""
    db = MVCCDatabase()
    tx = db.begin_transaction()
    assert db.read(tx, "x") is None
    db.write(tx, "x", 42)
    assert db.read(tx, "x") == 42
    db.write(tx, "x", 99)
    assert db.read(tx, "x") == 99


def test_13_sequential_transactions():
    """Later transaction sees earlier committed writes."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    assert db.read(tx2, "x") == 1
    db.write(tx2, "x", 2)
    db.commit(tx2)

    tx3 = db.begin_transaction()
    assert db.read(tx3, "x") == 2


def test_14_long_running_readonly():
    """Long-running read-only tx maintains snapshot through many writes."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 0)
    db.commit(tx1)

    tx_ro = db.begin_transaction(read_only=True)
    assert db.read(tx_ro, "x") == 0

    for i in range(1, 20):
        tx = db.begin_transaction()
        db.write(tx, "x", i)
        db.commit(tx)

    assert db.read(tx_ro, "x") == 0


def test_15_error_handling():
    """Operations on committed/aborted transactions raise errors."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    with pytest.raises(TransactionError):
        db.read(tx1, "x")

    with pytest.raises(TransactionError):
        db.write(tx1, "y", 2)

    with pytest.raises(TransactionError):
        db.commit(tx1)

    tx2 = db.begin_transaction()
    db.abort(tx2)

    with pytest.raises(TransactionError):
        db.read(tx2, "x")

    with pytest.raises(TransactionError):
        db.abort(tx2)

    with pytest.raises(TransactionError):
        db.delete(tx2, "x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
