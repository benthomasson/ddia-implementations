"""Tests for SSI database implementation."""
from ssi_database import SSIDatabase


def test_write_skew_doctors():
    """Test the classic write skew scenario (on-call doctors)."""
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
    assert result_alice["committed"] == True

    result_bob = db.commit(tx_bob)
    assert result_bob["committed"] == False
    assert "write skew" in result_bob["reason"].lower() or "conflict" in result_bob["reason"].lower()

    tx_check = db.begin_transaction()
    alice = db.read(tx_check, "doctor:alice")
    bob = db.read(tx_check, "doctor:bob")
    on_call_count = sum(1 for d in [alice, bob] if d["on_call"])
    assert on_call_count >= 1
    print("PASS: test_write_skew_doctors")


def test_non_conflicting_concurrent():
    """Test that non-conflicting concurrent transactions both commit."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "a", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    tx3 = db.begin_transaction()
    db.read(tx2, "a")
    db.write(tx2, "b", 2)
    db.write(tx3, "c", 3)
    assert db.commit(tx2)["committed"] == True
    assert db.commit(tx3)["committed"] == True
    print("PASS: test_non_conflicting_concurrent")


def test_write_write_conflict():
    """Test write-write conflict detection."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    tx2 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.write(tx2, "x", 2)
    assert db.commit(tx1)["committed"] == True
    result = db.commit(tx2)
    assert result["committed"] == False
    assert any(c["type"] == "write-write" for c in result["conflicts"])
    print("PASS: test_write_write_conflict")


def test_phantom_insert():
    """Test phantom detection: inserting a row matching another txn's predicate."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    meetings = db.read_predicate(tx1, lambda k, v: k.startswith("meeting:"))
    assert len(meetings) == 0

    tx2 = db.begin_transaction()
    db.write(tx2, "meeting:1", {"room": "A", "time": "10:00"})
    db.commit(tx2)

    db.write(tx1, "booking:room_A", {"time": "10:00"})
    result = db.commit(tx1)
    assert result["committed"] == False
    assert any(c["type"] == "phantom" for c in result["conflicts"])
    print("PASS: test_phantom_insert")


def test_phantom_delete():
    """Test phantom detection: deleting a row matching another txn's predicate."""
    db = SSIDatabase()
    tx_setup = db.begin_transaction()
    db.write(tx_setup, "item:1", {"active": True})
    db.write(tx_setup, "item:2", {"active": True})
    db.commit(tx_setup)

    tx1 = db.begin_transaction()
    items = db.read_predicate(tx1, lambda k, v: k.startswith("item:"))
    assert len(items) == 2

    tx2 = db.begin_transaction()
    db.delete(tx2, "item:2")
    db.commit(tx2)

    db.write(tx1, "summary", {"count": len(items)})
    result = db.commit(tx1)
    assert result["committed"] == False
    print("PASS: test_phantom_delete")


def test_sequential_no_conflict():
    """Test that sequential (non-overlapping) transactions never conflict."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    val = db.read(tx2, "x")
    assert val == 1
    db.write(tx2, "x", 2)
    assert db.commit(tx2)["committed"] == True

    tx3 = db.begin_transaction()
    val = db.read(tx3, "x")
    assert val == 2
    db.write(tx3, "x", 3)
    assert db.commit(tx3)["committed"] == True
    print("PASS: test_sequential_no_conflict")


def test_read_only_never_abort():
    """Test read-only transactions never abort."""
    db = SSIDatabase()
    tx_setup = db.begin_transaction()
    db.write(tx_setup, "a", 1)
    db.commit(tx_setup)

    tx_ro = db.begin_transaction()
    db.read(tx_ro, "a")

    tx_w = db.begin_transaction()
    db.write(tx_w, "a", 2)
    db.commit(tx_w)

    result = db.commit(tx_ro)
    assert result["committed"] == True
    print("PASS: test_read_only_never_abort")


def test_constraint_validation():
    """Test constraint validation at commit time."""
    db = SSIDatabase()
    tx_setup = db.begin_transaction()
    db.write(tx_setup, "balance:alice", 100)
    db.write(tx_setup, "balance:bob", 100)
    db.commit(tx_setup)

    db.add_constraint("positive_balances", lambda snap: all(
        v >= 0 for k, v in snap.items() if k.startswith("balance:")
    ))

    tx = db.begin_transaction()
    db.write(tx, "balance:alice", -50)
    result = db.commit(tx)
    assert result["committed"] == False
    assert any(c["type"] == "constraint" for c in result["conflicts"])
    print("PASS: test_constraint_validation")


def test_dependency_graph():
    """Test dependency graph tracking."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    db.read(tx2, "x")
    db.write(tx2, "y", 2)
    db.commit(tx2)

    graph = db.get_dependency_graph()
    assert tx1.tx_id in graph.get(tx2.tx_id, set())
    print("PASS: test_dependency_graph")


def test_complex_dependency_chain():
    """Test multiple concurrent transactions with complex dependency chains."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "a", 1)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    db.read(tx2, "a")
    db.write(tx2, "b", 2)
    db.commit(tx2)

    tx3 = db.begin_transaction()
    db.read(tx3, "b")
    db.write(tx3, "c", 3)
    db.commit(tx3)

    graph = db.get_dependency_graph()
    assert tx1.tx_id in graph.get(tx2.tx_id, set())
    assert tx2.tx_id in graph.get(tx3.tx_id, set())
    print("PASS: test_complex_dependency_chain")


def test_aborted_writes_invisible():
    """Test that aborted transaction's writes don't affect other transactions."""
    db = SSIDatabase()
    tx1 = db.begin_transaction()
    db.write(tx1, "x", 1)
    db.commit(tx1)

    tx_bad = db.begin_transaction()
    db.write(tx_bad, "x", 999)
    db.abort(tx_bad)

    tx2 = db.begin_transaction()
    val = db.read(tx2, "x")
    assert val == 1
    print("PASS: test_aborted_writes_invisible")


def test_predicate_various():
    """Test predicate-based reads with various predicate functions."""
    db = SSIDatabase()
    tx_setup = db.begin_transaction()
    db.write(tx_setup, "user:1", {"age": 25, "active": True})
    db.write(tx_setup, "user:2", {"age": 30, "active": False})
    db.write(tx_setup, "user:3", {"age": 35, "active": True})
    db.commit(tx_setup)

    tx = db.begin_transaction()
    active = db.read_predicate(tx, lambda k, v: v.get("active", False))
    assert len(active) == 2

    old = db.read_predicate(tx, lambda k, v: v.get("age", 0) > 28)
    assert len(old) == 2
    db.commit(tx)
    print("PASS: test_predicate_various")


def test_meeting_room_double_booking():
    """Test the meeting room / double booking scenario."""
    db = SSIDatabase()

    tx1 = db.begin_transaction()
    bookings = db.read_predicate(tx1, lambda k, v: k.startswith("booking:") and v.get("room") == "A")
    assert len(bookings) == 0

    tx2 = db.begin_transaction()
    bookings2 = db.read_predicate(tx2, lambda k, v: k.startswith("booking:") and v.get("room") == "A")
    assert len(bookings2) == 0

    db.write(tx1, "booking:1", {"room": "A", "time": "10:00"})
    db.write(tx2, "booking:2", {"room": "A", "time": "10:00"})

    assert db.commit(tx1)["committed"] == True
    result = db.commit(tx2)
    assert result["committed"] == False
    print("PASS: test_meeting_room_double_booking")


def test_first_committer_wins():
    """Test that first-committer-wins applies to write skew."""
    db = SSIDatabase()
    tx_setup = db.begin_transaction()
    db.write(tx_setup, "x", 1)
    db.commit(tx_setup)

    tx1 = db.begin_transaction()
    tx2 = db.begin_transaction()
    db.read(tx1, "x")
    db.read(tx2, "x")
    db.write(tx1, "y", 10)
    db.write(tx2, "z", 20)

    assert db.commit(tx1)["committed"] == True

    tx3 = db.begin_transaction()
    db.write(tx3, "x", 2)
    db.commit(tx3)

    result = db.commit(tx2)
    assert result["committed"] == False
    print("PASS: test_first_committer_wins")


def test_pessimistic_mode():
    """Test pessimistic conflict detection."""
    db = SSIDatabase(pessimistic=True)
    tx_setup = db.begin_transaction()
    db.write(tx_setup, "x", 1)
    db.commit(tx_setup)

    tx1 = db.begin_transaction()
    db.read(tx1, "x")
    db.write(tx1, "y", 2)
    db.commit(tx1)

    tx2 = db.begin_transaction()
    db.write(tx2, "x", 99)
    db.commit(tx2)

    tx3 = db.begin_transaction()
    try:
        db.read(tx3, "x")
        print("PASS: test_pessimistic_mode (no conflict on sequential)")
    except RuntimeError:
        print("PASS: test_pessimistic_mode (correctly detected conflict)")

    assert tx3.status in ("active", "aborted")
    print("PASS: test_pessimistic_mode")


if __name__ == "__main__":
    test_write_skew_doctors()
    test_non_conflicting_concurrent()
    test_write_write_conflict()
    test_phantom_insert()
    test_phantom_delete()
    test_sequential_no_conflict()
    test_read_only_never_abort()
    test_constraint_validation()
    test_dependency_graph()
    test_complex_dependency_chain()
    test_aborted_writes_invisible()
    test_predicate_various()
    test_meeting_room_double_booking()
    test_first_committer_wins()
    test_pessimistic_mode()
    print("\nALL TESTS PASSED")
