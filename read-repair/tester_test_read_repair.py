"""Tests for read repair implementation."""

import warnings
import pytest
from read_repair import Replica, ReadRepairStore, InsufficientReplicasError


# --- Test 1: Normal read/write ---
def test_normal_read_write():
    """All replicas healthy: write then read returns correct value, no repairs."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    result = store.put("k1", "hello")
    assert result["success"]
    assert result["version"] == 1
    assert len(result["replicas_written"]) == 2

    read = store.get("k1")
    assert read["value"] == "hello"
    assert read["version"] == 1
    assert read["consistent"]
    assert read["repairs_triggered"] == []


# --- Test 2: Read repair with stale replica ---
def test_read_repair_stale_replica():
    """Replica down during write, comes back, read repair fixes it."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("k1", "v1")

    store.set_replica_available(1, False)
    store.put("k1", "v2")  # writes to 0, 2
    store.set_replica_available(1, True)

    # Read from 0, 1 — replica 1 has v1, triggers repair
    read = store.get("k1")
    assert read["value"] == "v2"
    assert 1 in read["repairs_triggered"]

    # After repair, replica 1 should be current
    read2 = store.get("k1")
    assert read2["consistent"]


# --- Test 3: Quorum failure ---
def test_quorum_failure_read_and_write():
    """Too many replicas down raises InsufficientReplicasError."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.set_replica_available(0, False)
    store.set_replica_available(1, False)

    with pytest.raises(InsufficientReplicasError):
        store.put("k1", "v1")

    with pytest.raises(InsufficientReplicasError):
        store.get("k1")


# --- Test 4: Anti-entropy repair ---
def test_anti_entropy_repair():
    """Anti-entropy syncs all available replicas to latest version."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("k1", "v1")

    store.set_replica_available(2, False)
    store.put("k1", "v2")
    store.set_replica_available(2, True)

    result = store.anti_entropy_repair("k1")
    assert 2 in result["replicas_repaired"]
    assert result["final_version"] == 2

    states = store.get_replica_states("k1")
    versions = [s["version"] for s in states]
    assert len(set(versions)) == 1


# --- Test 5: Monotonic versions ---
def test_monotonic_versions():
    """Version numbers increase monotonically per key."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    versions = []
    for i in range(5):
        result = store.put("k1", f"v{i}")
        versions.append(result["version"])
    assert versions == [1, 2, 3, 4, 5]


# --- Test 6: Multiple keys independent ---
def test_multiple_keys_independent():
    """Different keys have independent version counters."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("a", "a1")
    store.put("a", "a2")
    store.put("b", "b1")

    assert store.get("a")["version"] == 2
    assert store.get("b")["version"] == 1


# --- Test 7: Repair stats ---
def test_repair_stats():
    """Repair statistics accurately track reads and repairs."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("k1", "v1")
    store.get("k1")  # no repair needed

    store.set_replica_available(1, False)
    store.put("k1", "v2")  # writes to 0, 2
    store.set_replica_available(1, True)

    store.get("k1")  # repairs replica 1

    stats = store.get_repair_stats()
    assert stats["total_reads"] == 2
    assert stats["reads_triggering_repair"] == 1
    assert stats["total_replicas_repaired"] == 1
    assert stats["repair_rate"] == 0.5


# --- Test 8: Nonexistent key ---
def test_read_nonexistent_key():
    """Reading a missing key returns None/0 and is consistent."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    read = store.get("nonexistent")
    assert read["value"] is None
    assert read["version"] == 0
    assert read["consistent"]


# --- Test 9: Quorum warning ---
def test_quorum_warning():
    """Weak quorum (R + W <= N) emits a warning but doesn't raise."""
    with pytest.warns(UserWarning, match="Quorum condition not met"):
        ReadRepairStore(num_replicas=5, read_quorum=2, write_quorum=2)


# --- Test 10: Spec example usage ---
def test_spec_example():
    """End-to-end test matching the spec's example usage."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)

    store.put("user:1", {"name": "Alice", "age": 30})
    result = store.get("user:1")
    assert result["value"] == {"name": "Alice", "age": 30}
    assert result["consistent"]

    store.set_replica_available(2, False)
    store.put("user:1", {"name": "Alice", "age": 31})
    store.set_replica_available(2, True)

    result = store.get("user:1")
    assert result["value"] == {"name": "Alice", "age": 31}
    # With R=2, replicas 0,1 are queried — both have v2, so no read repair here.
    # Replica 2 is stale but wasn't in the read quorum.

    # Anti-entropy fixes replica 2
    store.anti_entropy_repair("user:1")
    states = store.get_replica_states("user:1")
    versions = [s["version"] for s in states if s["available"]]
    assert len(set(versions)) == 1  # all replicas now consistent
