"""Tests for SSI database - write skew detection."""
import sys

from ssi_database import SSIDatabase


def test_write_skew_doctors():
    """Classic write skew: two doctors go off call concurrently."""
    db = SSIDatabase()
    tx_setup = db.begin_transaction()
    db.write(tx_setup, "doctor:alice", {"name": "Alice", "on_call": True})
    db.write(tx_setup, "doctor:bob", {"name": "Bob", "on_call": True})
    db.commit(tx_setup)

    tx_alice = db.begin_transaction()
    on_call = db.read_predicate(tx_alice, lambda k, v: v.get("on_call", False))
    assert len(on_call) == 2

    tx_bob = db.begin_transaction()
    on_call_bob = db.read_predicate(tx_bob, lambda k, v: v.get("on_call", False))
    assert len(on_call_bob) == 2

    db.write(tx_alice, "doctor:alice", {"name": "Alice", "on_call": False})
    db.write(tx_bob, "doctor:bob", {"name": "Bob", "on_call": False})

    result_alice = db.commit(tx_alice)
    assert result_alice["committed"] is True

    result_bob = db.commit(tx_bob)
    assert result_bob["committed"] is False
    assert "write skew" in result_bob["reason"].lower() or "conflict" in result_bob["reason"].lower()

    # Verify invariant: at least one doctor still on call
    tx_check = db.begin_transaction()
    alice = db.read(tx_check, "doctor:alice")
    bob = db.read(tx_check, "doctor:bob")
    on_call_count = sum(1 for d in [alice, bob] if d and d["on_call"])
    assert on_call_count >= 1


def test_phantom_insert():
    """Phantom detection: new row inserted matching a predicate."""
    db = SSIDatabase()

    tx1 = db.begin_transaction()
    meetings = db.read_predicate(tx1, lambda k, v: k.startswith("meeting:"))
    assert len(meetings) == 0

    tx2 = db.begin_transaction()
    db.write(tx2, "meeting:1", {"room": "A", "time": "10:00"})
    db.commit(tx2)

    db.write(tx1, "booking:room_A", {"time": "10:00"})
    result = db.commit(tx1)
    assert result["committed"] is False
    assert any(c["type"] == "phantom" for c in result["conflicts"])


def test_phantom_delete():
    """Phantom detection: row deleted that matched a predicate."""
    db = SSIDatabase()
    tx_setup = db.begin_transaction()
    db.write(tx_setup, "slot:1", {"available": True})
    db.write(tx_setup, "slot:2", {"available": True})
    db.commit(tx_setup)

    tx1 = db.begin_transaction()
    slots = db.read_predicate(tx1, lambda k, v: k.startswith("slot:"))
    assert len(slots) == 2

    tx2 = db.begin_transaction()
    db.delete(tx2, "slot:2")
    db.commit(tx2)

    db.write(tx1, "reservation:1", {"slot": "slot:1"})
    result = db.commit(tx1)
    assert result["committed"] is False
    assert any(c["type"] == "phantom" for c in result["conflicts"])


def test_non_conflicting_concurrent():
    """Non-conflicting concurrent transactions both commit."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "a", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    tx3 = db.begin_transaction()
    db.read(tx2, "a")
    db.write(tx2, "b", 2)
    db.write(tx3, "c", 3)
    assert db.commit(tx2)["committed"] is True
    assert db.commit(tx3)["committed"] is True


def test_write_write_conflict():
    """Write-write conflict on same key."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    tx2 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.write(tx2, "x", 2)
    assert db.commit(tx1)["committed"] is True
    result = db.commit(tx2)
    assert result["committed"] is False
    assert any(c["type"] == "write-write" for c in result["conflicts"])


def test_sequential_transactions_no_conflict():
    """Sequential (non-overlapping) transactions never conflict."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "k", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    val = db.read(tx2, "k")
    assert val == 1
    db.write(tx2, "k", 2)
    assert db.commit(tx2)["committed"] is True


def test_read_only_never_aborts():
    """Read-only transactions always commit."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "a", 1)
    db.commit(tx1)

    tx_ro = db.begin_transaction()
    db.read(tx_ro, "a")

    # Concurrent write to the same key
    tx2 = db.begin_transaction()
    db.write(tx2, "a", 2)
    db.commit(tx2)

    # Read-only should still commit
    assert db.commit(tx_ro)["committed"] is True


def test_constraint_validation():
    """Constraint checked at commit time."""
    db = SSIDatabase()
    db.add_constraint("positive_balance", lambda snap: all(
        v >= 0 for k, v in snap.items() if k.startswith("balance:")
    ))

    tx1 = db.begin_transaction()
    db.write(tx1, "balance:alice", 100)
    assert db.commit(tx1)["committed"] is True

    tx2 = db.begin_transaction()
    db.write(tx2, "balance:bob", -50)
    result = db.commit(tx2)
    assert result["committed"] is False
    assert any(c["type"] == "constraint" for c in result["conflicts"])


def test_dependency_graph():
    """Dependency tracking when tx reads another tx's writes."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 10)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    db.read(tx2, "x")  # reads tx1's write
    db.write(tx2, "y", 20)
    db.commit(tx2)

    graph = db.get_dependency_graph()
    assert tx1.tx_id in graph[tx2.tx_id]


def test_aborted_writes_invisible():
    """Aborted transaction's writes don't affect others."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "k", "from_tx1")
    db.abort(tx1)
    assert tx1.status == "aborted"

    tx2 = db.begin_transaction()
    val = db.read(tx2, "k")
    assert val is None  # tx1 was aborted, write not visible


if __name__ == "__main__":
    tests = [
        test_write_skew_doctors,
        test_phantom_insert,
        test_phantom_delete,
        test_non_conflicting_concurrent,
        test_write_write_conflict,
        test_sequential_transactions_no_conflict,
        test_read_only_never_aborts,
        test_constraint_validation,
        test_dependency_graph,
        test_aborted_writes_invisible,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        sys.exit(1)
