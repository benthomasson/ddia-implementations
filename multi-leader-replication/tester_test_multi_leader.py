"""Tests for multi-leader replication system."""
import sys

from multi_leader import (
    ReplicaNode, MultiLeaderCluster, ConflictStrategy, Topology, ConflictRecord
)


def test_basic_replication_no_conflict():
    """Non-conflicting writes at different nodes replicate correctly."""
    cluster = MultiLeaderCluster(["a", "b", "c"])
    cluster.node("a").put("x", 1)
    cluster.node("b").put("y", 2)
    cluster.node("c").put("z", 3)
    cluster.sync()
    for nid in ["a", "b", "c"]:
        assert cluster.node(nid).get("x") == 1, f"{nid} missing x"
        assert cluster.node(nid).get("y") == 2, f"{nid} missing y"
        assert cluster.node(nid).get("z") == 3, f"{nid} missing z"
    assert cluster.all_converged()
    # No conflicts expected
    for nid in ["a", "b", "c"]:
        assert len(cluster.node(nid).conflict_log) == 0, f"{nid} has unexpected conflicts"


def test_lww_conflict_higher_timestamp_wins():
    """LWW: write with higher timestamp wins."""
    cluster = MultiLeaderCluster(["a", "b"])
    cluster.node("a").put("k", "first")
    cluster.node("a").put("k", "second")  # ts=2
    cluster.node("b").put("k", "from_b")   # ts=1
    cluster.sync()
    # a has ts=2, b has ts=1, so a's value wins
    assert cluster.all_converged()
    assert cluster.node("a").get("k") == "second"
    assert cluster.node("b").get("k") == "second"


def test_lww_tiebreak_by_node_id():
    """LWW tiebreak: when timestamps equal, higher node_id wins."""
    cluster = MultiLeaderCluster(["a", "b"])
    cluster.node("a").put("k", "from_a")  # ts=1
    cluster.node("b").put("k", "from_b")  # ts=1
    cluster.sync()
    assert cluster.all_converged()
    # (1, "b") > (1, "a"), so b wins
    assert cluster.node("a").get("k") == "from_b"
    assert cluster.node("b").get("k") == "from_b"


def test_custom_merge():
    """Custom merge function resolves conflicts (counter addition)."""
    def counter_merge(key, local_val, remote_val, local_ts, remote_ts):
        return local_val + remote_val

    cluster = MultiLeaderCluster(
        ["a", "b"],
        strategy=ConflictStrategy.CUSTOM_MERGE,
        merge_fn=counter_merge,
    )
    cluster.node("a").put("counter", 5)
    cluster.node("b").put("counter", 3)
    cluster.sync()
    assert cluster.node("a").get("counter") == 8, f'a={cluster.node("a").get("counter")}'
    assert cluster.node("b").get("counter") == 8, f'b={cluster.node("b").get("counter")}'


def test_tombstone_delete():
    """Tombstone deletes replicate correctly."""
    cluster = MultiLeaderCluster(["a", "b"])
    cluster.node("a").put("k", "val")
    cluster.sync()
    assert cluster.node("b").get("k") == "val"
    cluster.node("b").delete("k")
    cluster.sync()
    assert cluster.node("a").get("k") is None
    assert cluster.node("b").get("k") is None


def test_ring_topology():
    """Ring topology requires multiple sync rounds for full propagation."""
    cluster = MultiLeaderCluster(["n1", "n2", "n3"], topology=Topology.RING)
    cluster.node("n1").put("x", 1)
    rounds = cluster.sync_until_converged()
    assert rounds >= 2, f"Expected >=2 rounds, got {rounds}"
    assert cluster.all_converged()
    for nid in ["n1", "n2", "n3"]:
        assert cluster.node(nid).get("x") == 1


def test_conflict_logging():
    """ConflictRecords contain correct values."""
    cluster = MultiLeaderCluster(["a", "b"])
    cluster.node("a").put("k", "alice")
    cluster.node("b").put("k", "bob")
    cluster.sync()
    # At least one node should have a conflict record
    all_conflicts = cluster.node("a").conflict_log + cluster.node("b").conflict_log
    assert len(all_conflicts) >= 1, "No conflicts logged"
    c = all_conflicts[0]
    assert c.key == "k"
    assert isinstance(c, ConflictRecord)
    assert c.resolved_by == ConflictStrategy.LAST_WRITE_WINS
    assert {c.local_value, c.remote_value} == {"alice", "bob"}


def test_lamport_clock_ordering():
    """Lamport clocks advance correctly on local writes and remote applies."""
    node = ReplicaNode("test")
    ts1 = node.put("a", 1)
    ts2 = node.put("b", 2)
    ts3 = node.put("c", 3)
    assert ts1 < ts2 < ts3, f"Clocks not monotonic: {ts1}, {ts2}, {ts3}"

    # Remote apply should advance clock
    node.apply_remote_change(
        {"key": "d", "value": 99, "timestamp": 100, "node_id": "remote", "is_tombstone": False},
        ConflictStrategy.LAST_WRITE_WINS,
    )
    ts4 = node.put("e", 5)
    assert ts4 > 100, f"Clock not updated after remote apply: {ts4}"


def test_idempotency():
    """Applying the same change twice should not create duplicates."""
    cluster = MultiLeaderCluster(["a", "b"])
    cluster.node("a").put("key", "val")
    cluster.sync()
    cluster.sync()  # second sync should be a no-op
    assert cluster.node("b").get("key") == "val"
    assert len(cluster.node("b").conflict_log) == 0


def test_convergence_many_keys():
    """After sync, all nodes converge with many keys across many nodes."""
    cluster = MultiLeaderCluster([f"n{i}" for i in range(5)])
    for i in range(5):
        for j in range(20):
            cluster.node(f"n{i}").put(f"key_{i}_{j}", f"val_{i}_{j}")
    cluster.sync()
    assert cluster.all_converged()


def test_custom_merge_repeated_sync_converges():
    """CUSTOM_MERGE must converge after repeated syncs, not explode."""
    def counter_merge(key, local_val, remote_val, local_ts, remote_ts):
        return local_val + remote_val

    cluster = MultiLeaderCluster(
        ["a", "b", "c"],
        strategy=ConflictStrategy.CUSTOM_MERGE,
        merge_fn=counter_merge,
    )
    cluster.node("a").put("counter", 5)
    cluster.node("b").put("counter", 3)
    cluster.sync()
    val_after_first = cluster.node("a").get("counter")

    for _ in range(5):
        cluster.sync()

    for nid in ["a", "b", "c"]:
        assert cluster.node(nid).get("counter") == val_after_first, \
            f"Node {nid} value changed after extra syncs: {cluster.node(nid).get('counter')} != {val_after_first}"
    assert cluster.all_converged()


if __name__ == "__main__":
    tests = [
        test_basic_replication_no_conflict,
        test_lww_conflict_higher_timestamp_wins,
        test_lww_tiebreak_by_node_id,
        test_custom_merge,
        test_tombstone_delete,
        test_ring_topology,
        test_conflict_logging,
        test_lamport_clock_ordering,
        test_idempotency,
        test_convergence_many_keys,
        test_custom_merge_repeated_sync_converges,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
