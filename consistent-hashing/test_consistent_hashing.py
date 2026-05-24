"""Tests for consistent hashing ring."""

import pytest
from consistent_hashing import ConsistentHashRing


def test_determinism():
    ring = ConsistentHashRing(num_vnodes=150)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    results = [ring.get_node("key:42") for _ in range(100)]
    assert len(set(results)) == 1


def test_basic_routing():
    ring = ConsistentHashRing(num_vnodes=150)
    ring.add_node("A")
    ring.add_node("B")
    node = ring.get_node("hello")
    assert node in ("A", "B")


def test_replication_distinct():
    ring = ConsistentHashRing(num_vnodes=150, replication_factor=3)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    nodes = ring.get_nodes("test-key")
    assert len(nodes) == 3
    assert len(set(nodes)) == 3
    assert nodes[0] == ring.get_node("test-key")


def test_replication_insufficient_nodes():
    ring = ConsistentHashRing(num_vnodes=10, replication_factor=3)
    ring.add_node("A")
    ring.add_node("B")
    with pytest.raises(ValueError):
        ring.get_nodes("key")


def test_load_balance():
    ring = ConsistentHashRing(num_vnodes=150)
    for name in ["A", "B", "C"]:
        ring.add_node(name)
    assert ring.load_imbalance() < 1.5


def test_load_distribution_sums_to_one():
    ring = ConsistentHashRing(num_vnodes=150)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    dist = ring.get_load_distribution()
    assert abs(sum(dist.values()) - 1.0) < 0.001


def test_minimal_redistribution():
    ring = ConsistentHashRing(num_vnodes=150)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    keys = [f"key:{i}" for i in range(10000)]
    before = {k: ring.get_node(k) for k in keys}
    ring.add_node("D")
    after = {k: ring.get_node(k) for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k])
    # Expect ~1/4 of keys to move (to the new node)
    assert moved < 4000  # well under 40%
    assert moved > 1000  # but at least some moved


def test_node_removal():
    ring = ConsistentHashRing(num_vnodes=150)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    ring.remove_node("B")
    assert ring.get_node_count() == 2
    assert "B" not in ring.get_all_nodes()
    # All keys still resolve
    for i in range(100):
        node = ring.get_node(f"key:{i}")
        assert node in ("A", "C")


def test_weighted_nodes():
    ring = ConsistentHashRing(num_vnodes=100, replication_factor=1)
    ring.add_node("small", weight=1.0)
    ring.add_node("large", weight=3.0)
    keys = [f"k:{i}" for i in range(10000)]
    dist = ring.get_key_distribution(keys)
    ratio = dist["large"] / dist["small"]
    assert 2.0 < ratio < 4.5


def test_empty_ring():
    ring = ConsistentHashRing()
    with pytest.raises(ValueError):
        ring.get_node("key")


def test_single_node():
    ring = ConsistentHashRing(num_vnodes=50)
    ring.add_node("only")
    assert ring.get_node("any-key") == "only"
    dist = ring.get_load_distribution()
    assert abs(dist["only"] - 1.0) < 0.001


def test_duplicate_add_idempotent():
    ring = ConsistentHashRing(num_vnodes=10)
    ring.add_node("A")
    ring.add_node("A")  # should be idempotent
    assert ring.get_node_count() == 1


def test_ring_position_valid_range():
    ring = ConsistentHashRing()
    for i in range(100):
        pos = ring.get_ring_position(f"key:{i}")
        assert 0 <= pos < 2**32


def test_scalability():
    ring = ConsistentHashRing(num_vnodes=150)
    for i in range(100):
        ring.add_node(f"node-{i}")
    assert ring.get_node_count() == 100
    # Lookups should work
    for i in range(1000):
        node = ring.get_node(f"key:{i}")
        assert node.startswith("node-")


def test_ring_info():
    ring = ConsistentHashRing(num_vnodes=50, replication_factor=2)
    ring.add_node("A")
    ring.add_node("B")
    info = ring.ring_info()
    assert "2 nodes" in info
    assert "RF=2" in info


def test_add_node_returns_transfers():
    ring = ConsistentHashRing(num_vnodes=10)
    ring.add_node("A")
    transfers = ring.add_node("B")
    assert len(transfers) > 0
    for (start, end), (from_node, to_node) in transfers.items():
        assert from_node == "A"
        assert to_node == "B"


def test_remove_node_returns_transfers():
    ring = ConsistentHashRing(num_vnodes=10)
    ring.add_node("A")
    ring.add_node("B")
    transfers = ring.remove_node("B")
    assert len(transfers) > 0
    for (start, end), (from_node, to_node) in transfers.items():
        assert from_node == "B"
        assert to_node == "A"


def test_vnode_count_affects_balance():
    imbalances = []
    for vnodes in [1, 10, 50, 150, 500]:
        r = ConsistentHashRing(num_vnodes=vnodes)
        r.add_node("A")
        r.add_node("B")
        r.add_node("C")
        imbalances.append(r.load_imbalance())
    # More vnodes should generally improve balance
    assert imbalances[-1] < imbalances[0]
