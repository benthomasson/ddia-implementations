"""Tests for Total Order Broadcast protocol — QA tester suite."""

import sys

from total_order_broadcast import ConsensusInstance, TOBNode, TOBCluster, LinearizableRegister


# --- Core Protocol Tests ---

def test_single_broadcast_all_nodes_agree():
    """All nodes deliver the same single message."""
    cluster = TOBCluster(3)
    cluster.broadcast(0, "hello")
    assert cluster.run_until_delivered(1)
    assert cluster.verify_total_order()
    for i in range(3):
        assert cluster.get_delivery_order(i) == ["hello"]


def test_concurrent_proposals_total_order():
    """Three nodes broadcast concurrently; total order is maintained."""
    cluster = TOBCluster(3)
    cluster.broadcast(0, "A")
    cluster.broadcast(1, "B")
    cluster.broadcast(2, "C")
    assert cluster.run_until_delivered(3)
    assert cluster.verify_total_order()
    order = cluster.get_delivery_order(0)
    assert set(order) == {"A", "B", "C"}
    # All nodes see identical order
    assert cluster.get_delivery_order(1) == order
    assert cluster.get_delivery_order(2) == order


def test_crash_majority_survives():
    """After one crash, the remaining majority continues delivering."""
    cluster = TOBCluster(3)
    cluster.broadcast(0, "before")
    assert cluster.run_until_delivered(1)
    cluster.crash_node(2)
    cluster.broadcast(0, "after")
    assert cluster.run_until_delivered(2)
    assert "after" in cluster.get_delivery_order(0)
    assert "after" in cluster.get_delivery_order(1)
    # Crashed node should not have delivered the new message
    assert len(cluster.get_delivery_order(2)) == 1


def test_recovery_catches_up():
    """Recovered node gets all missed messages in correct order."""
    cluster = TOBCluster(3)
    cluster.broadcast(0, "m1")
    assert cluster.run_until_delivered(1)
    cluster.crash_node(2)
    cluster.broadcast(0, "m2")
    cluster.broadcast(1, "m3")
    assert cluster.run_until_delivered(3)
    cluster.recover_node(2)
    assert cluster.get_delivery_order(2) == cluster.get_delivery_order(0)


def test_minority_blocked():
    """A single node (minority) cannot make progress."""
    cluster = TOBCluster(3)
    cluster.crash_node(1)
    cluster.crash_node(2)
    cluster.broadcast(0, "lonely")
    assert not cluster.run_until_delivered(1, max_rounds=50)


# --- Consensus Instance Tests ---

def test_paxos_prepare_accept_semantics():
    """Higher proposal preempts lower; accept with stale proposal fails."""
    inst = ConsensusInstance(slot=0, num_nodes=3)
    r1 = inst.prepare(1, proposer_id=0)
    assert r1['promised']
    r2 = inst.prepare(5, proposer_id=1)
    assert r2['promised']
    # Stale accept rejected
    assert not inst.accept(1, "old", proposer_id=0)['accepted']
    # Current accept succeeds
    assert inst.accept(5, "new", proposer_id=1)['accepted']


def test_decided_slot_immutable():
    """Once decided, a slot's value cannot change."""
    cluster = TOBCluster(3)
    cluster.broadcast(0, "fixed")
    assert cluster.run_until_delivered(1)
    for node in cluster.nodes.values():
        inst = node._get_instance(0)
        assert inst.is_decided
        assert inst.decided_value == "fixed"


# --- Linearizable Register Tests ---

def test_linearizable_cas_success_and_failure():
    """CAS succeeds when expected matches, fails otherwise."""
    cluster = TOBCluster(3)
    reg = LinearizableRegister(cluster, 0)
    reg.write("k", 100)
    assert reg.read("k") == 100
    assert reg.compare_and_set("k", 100, 200)
    assert reg.read("k") == 200
    assert not reg.compare_and_set("k", 100, 300)
    assert reg.read("k") == 200


def test_read_nonexistent_key():
    """Reading a key that was never written returns None."""
    cluster = TOBCluster(3)
    reg = LinearizableRegister(cluster, 0)
    assert reg.read("missing") is None


# --- Scale & Stress ---

def test_25_sequential_broadcasts():
    """System handles 25+ messages across all nodes."""
    cluster = TOBCluster(5)
    for i in range(25):
        cluster.broadcast(i % 5, f"msg_{i}")
    assert cluster.run_until_delivered(25, max_rounds=5000)
    assert cluster.verify_total_order()
    order = cluster.get_delivery_order(0)
    assert len(order) == 25
    assert set(order) == {f"msg_{i}" for i in range(25)}


# ---------------------
# Test runner
# ---------------------

def run(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


tests = [
    ("single_broadcast_all_nodes_agree", test_single_broadcast_all_nodes_agree),
    ("concurrent_proposals_total_order", test_concurrent_proposals_total_order),
    ("crash_majority_survives", test_crash_majority_survives),
    ("recovery_catches_up", test_recovery_catches_up),
    ("minority_blocked", test_minority_blocked),
    ("paxos_prepare_accept_semantics", test_paxos_prepare_accept_semantics),
    ("decided_slot_immutable", test_decided_slot_immutable),
    ("linearizable_cas_success_and_failure", test_linearizable_cas_success_and_failure),
    ("read_nonexistent_key", test_read_nonexistent_key),
    ("25_sequential_broadcasts", test_25_sequential_broadcasts),
]

if __name__ == "__main__":
    passed = sum(run(n, f) for n, f in tests)
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
