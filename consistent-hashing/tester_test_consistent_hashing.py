"""Tests for consistent hashing ring."""

import pytest
from consistent_hashing import ConsistentHashRing


def test_determinism():
    """Same key always maps to same node."""
    ring = ConsistentHashRing(num_vnodes=150)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    results = [ring.get_node("key:42") for _ in range(100)]
    assert len(set(results)) == 1


def test_replication_returns_distinct_physical_nodes():
    """get_nodes returns RF distinct physical nodes, primary first."""
    ring = ConsistentHashRing(num_vnodes=150, replication_factor=3)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    nodes = ring.get_nodes("test-key")
    assert len(nodes) == 3
    assert len(set(nodes)) == 3
    assert nodes[0] == ring.get_node("test-key")


def test_replication_insufficient_nodes_raises():
    """RF > physical nodes raises ValueError."""
    ring = ConsistentHashRing(num_vnodes=10, replication_factor=3)
    ring.add_node("A")
    ring.add_node("B")
    with pytest.raises(ValueError):
        ring.get_nodes("key")


def test_empty_ring_raises():
    """get_node on empty ring raises ValueError."""
    ring = ConsistentHashRing()
    with pytest.raises(ValueError):
        ring.get_node("key")


def test_load_balance_with_vnodes():
    """With 150 vnodes and 3 nodes, imbalance < 1.5."""
    ring = ConsistentHashRing(num_vnodes=150)
    for name in ["A", "B", "C"]:
        ring.add_node(name)
    assert ring.load_imbalance() < 1.5
    dist = ring.get_load_distribution()
    assert abs(sum(dist.values()) - 1.0) < 0.001


def test_minimal_redistribution_on_add():
    """Adding a 4th node moves ~1/4 of keys, not all."""
    ring = ConsistentHashRing(num_vnodes=150)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    keys = [f"key:{i}" for i in range(10000)]
    before = {k: ring.get_node(k) for k in keys}
    ring.add_node("D")
    after = {k: ring.get_node(k) for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k])
    assert moved < 4000  # well under 40%
    assert moved > 1000  # but meaningful amount moved


def test_node_removal():
    """After removing a node, keys resolve to remaining nodes only."""
    ring = ConsistentHashRing(num_vnodes=150)
    ring.add_node("A")
    ring.add_node("B")
    ring.add_node("C")
    ring.remove_node("B")
    assert ring.get_node_count() == 2
    assert "B" not in ring.get_all_nodes()
    for i in range(100):
        assert ring.get_node(f"key:{i}") in ("A", "C")


def test_weighted_nodes():
    """Node with 3x weight gets ~3x keys."""
    ring = ConsistentHashRing(num_vnodes=100)
    ring.add_node("small", weight=1.0)
    ring.add_node("large", weight=3.0)
    keys = [f"k:{i}" for i in range(10000)]
    dist = ring.get_key_distribution(keys)
    ratio = dist["large"] / dist["small"]
    assert 2.0 < ratio < 4.5


def test_add_and_remove_return_transfers():
    """add_node and remove_node return transfer dicts."""
    ring = ConsistentHashRing(num_vnodes=10)
    ring.add_node("A")
    transfers = ring.add_node("B")
    assert len(transfers) > 0
    for (start, end), (from_node, to_node) in transfers.items():
        assert from_node == "A"
        assert to_node == "B"
        assert 0 <= start < 2**32
        assert 0 <= end < 2**32

    transfers = ring.remove_node("B")
    assert len(transfers) > 0
    for (start, end), (from_node, to_node) in transfers.items():
        assert from_node == "B"
        assert to_node == "A"


def test_single_node_owns_everything():
    """Single node gets all keys and 100% load."""
    ring = ConsistentHashRing(num_vnodes=50)
    ring.add_node("only")
    assert ring.get_node("any-key") == "only"
    dist = ring.get_load_distribution()
    assert abs(dist["only"] - 1.0) < 0.001
