"""Tests for MVCC snapshot isolation database."""
import sys, os
from mvcc_database import MVCCDatabase, TransactionError
import pytest


def test_example_usage():
    """The full example from the spec."""
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

    assert db.read(tx2, "account_A") == 1000  # snapshot isolation

    tx4 = db.begin_transaction()
    assert db.read(tx4, "account_A") == 900

    tx5 = db.begin_transaction()
    tx6 = db.begin_transaction()
    db.write(tx5, "account_A", 800)
    db.write(tx6, "account_A", 850)
    assert db.commit(tx5) == True
    assert db.commit(tx6) == False  # conflict

    tx7 = db.begin_transaction()
    assert db.read(tx7, "account_A") == 800


def test_read_own_writes():
    """A transaction can read its own uncommitted writes."""
    db = MVCCDatabase()
    tx = db.begin_transaction()
    db.write(tx, "x", 42)
    assert db.read(tx, "x") == 42
    # Overwrite own write
    db.write(tx, "x", 99)
    assert db.read(tx, "x") == 99


def test_snapshot_isolation_uncommitted():
    """A transaction doesn't see another's uncommitted writes."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    tx2 = db.begin_transaction()
    assert db.read(tx2, "x") is None


def test_snapshot_isolation_committed_after_start():
    """A transaction doesn't see writes committed after its start."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    assert db.read(tx2, "x") == 10

    tx3 = db.begin_transaction()
    db.write(tx3, "x", 20)
    db.commit(tx3)
    assert db.read(tx2, "x") == 10  # still sees old value


def test_write_write_conflict():
    """First-committer-wins on write-write conflict."""
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

    tx4 = db.begin_transaction()
    assert db.read(tx4, "x") == 2


def test_aborted_invisible():
    """Aborted transactions' writes are invisible."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    db.abort(tx1)

    tx2 = db.begin_transaction()
    assert db.read(tx2, "x") is None


def test_delete_visibility():
    """Delete marks versions and respects snapshot isolation."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    db.commit(tx1)

    tx2 = db.begin_transaction()  # snapshot before delete
    tx3 = db.begin_transaction()
    assert db.delete(tx3, "x") == True
    db.commit(tx3)

    assert db.read(tx2, "x") == 10  # tx2 still sees it
    tx4 = db.begin_transaction()
    assert db.read(tx4, "x") is None  # tx4 sees deletion


def test_read_only_never_aborts():
    """Read-only transactions always commit and can't write."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx_ro = db.begin_transaction(read_only=True)
    assert db.read(tx_ro, "x") == 1
    assert db.commit(tx_ro) == True

    with pytest.raises(TransactionError):
        tx_ro2 = db.begin_transaction(read_only=True)
        db.write(tx_ro2, "y", 20)


def test_scan():
    """Scan returns correct snapshot filtered by prefix."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "user:1", "Alice")
    db.write(tx1, "user:2", "Bob")
    db.write(tx1, "order:1", "Pizza")
    db.commit(tx1)

    tx2 = db.begin_transaction()
    result = db.scan(tx2, prefix="user:")
    assert result == {"user:1": "Alice", "user:2": "Bob"}
    assert "order:1" not in result


def test_error_on_committed_aborted():
    """Operations on committed/aborted transactions raise TransactionError."""
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


def test_garbage_collection():
    """GC removes old versions, preserves those needed by active txs."""
    db = MVCCDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    db.write(tx2, "x", 2)
    db.commit(tx2)

    assert db.get_version_count("x") == 2

    # No active txs — GC should clean up
    removed = db.garbage_collect()
    assert removed >= 1
    assert db.get_version_count("x") == 1

    # Verify the remaining version is the latest
    tx3 = db.begin_transaction()
    assert db.read(tx3, "x") == 2
