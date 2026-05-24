"""Tests for read repair implementation."""

import pytest
from read_repair import Replica, ReadRepairStore, InsufficientReplicasError


def test_normal_read_write():
    """Test normal read/write with all replicas healthy."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    result = store.put("k1", "hello")
    assert result["success"]
    assert result["version"] == 1

    read = store.get("k1")
    assert read["value"] == "hello"
    assert read["version"] == 1
    assert read["consistent"]
    assert read["repairs_triggered"] == []


def test_read_repair_one_stale():
    """Test read repair when one replica is stale."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("k1", "v1")

    # Take down replica 2, write again, bring it back
    store.set_replica_available(2, False)
    store.put("k1", "v2")
    store.set_replica_available(2, True)

    # Read from replicas 0,1 — both have v2, no repair needed among those
    read = store.get("k1")
    assert read["value"] == "v2"

    # But replica 2 is still stale. Use anti-entropy or read with R=3 to fix.
    # With R=2, replicas 0 and 1 are queried (both up to date), so no repair.
    # Let's verify replica 2 is stale via get_replica_states.
    states = store.get_replica_states("k1")
    assert states[2]["version"] < states[0]["version"]

    # Now do anti-entropy to repair
    ae = store.anti_entropy_repair("k1")
    assert 2 in ae["replicas_repaired"]


def test_read_repair_multiple_stale():
    """Test read repair when multiple replicas are stale."""
    store = ReadRepairStore(num_replicas=5, read_quorum=3, write_quorum=3)
    store.put("k1", "v1")  # writes to 0,1,2

    # Take down 1 and 2, write again
    store.set_replica_available(1, False)
    store.set_replica_available(2, False)
    store.put("k1", "v2")  # writes to 0,3,4
    store.set_replica_available(1, True)
    store.set_replica_available(2, True)

    # Read from 0,1,2 — 1 and 2 are stale
    read = store.get("k1")
    assert read["value"] == "v2"
    assert 1 in read["repairs_triggered"] or 2 in read["repairs_triggered"]


def test_replica_down_during_write_then_back():
    """Test behavior when a replica is down during write and later comes back."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("k1", "v1")

    store.set_replica_available(1, False)
    store.put("k1", "v2")  # writes to 0, 2
    store.set_replica_available(1, True)

    # Read from 0, 1 — replica 1 has v1, triggers repair
    read = store.get("k1")
    assert read["value"] == "v2"
    assert 1 in read["repairs_triggered"]

    # Subsequent read should be consistent
    read2 = store.get("k1")
    assert read2["consistent"]


def test_quorum_failure():
    """Test quorum failure when too many replicas are down."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.set_replica_available(0, False)
    store.set_replica_available(1, False)

    with pytest.raises(InsufficientReplicasError):
        store.put("k1", "v1")

    with pytest.raises(InsufficientReplicasError):
        store.get("k1")


def test_anti_entropy_repair():
    """Test anti-entropy repair brings all replicas to the latest version."""
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


def test_monotonic_versions():
    """Test that version numbers are monotonically increasing per key."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    versions = []
    for i in range(5):
        result = store.put("k1", f"v{i}")
        versions.append(result["version"])
    assert versions == [1, 2, 3, 4, 5]


def test_multiple_keys_independent():
    """Test multiple keys with independent version tracking."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("a", "a1")
    store.put("a", "a2")
    store.put("b", "b1")

    ra = store.get("a")
    rb = store.get("b")
    assert ra["version"] == 2
    assert rb["version"] == 1


def test_repair_stats():
    """Test read repair statistics are accurate."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("k1", "v1")
    store.get("k1")  # no repair

    store.set_replica_available(1, False)
    store.put("k1", "v2")  # writes to 0, 2
    store.set_replica_available(1, True)

    store.get("k1")  # should repair replica 1

    stats = store.get_repair_stats()
    assert stats["total_reads"] == 2
    assert stats["reads_triggering_repair"] == 1
    assert stats["total_replicas_repaired"] == 1
    assert stats["repair_rate"] == 0.5


def test_repaired_replica_subsequent_read():
    """Test that a repaired replica returns the correct value on subsequent reads."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("k1", "v1")

    store.set_replica_available(1, False)
    store.put("k1", "v2")
    store.set_replica_available(1, True)

    # First read repairs replica 1
    store.get("k1")

    # Now read again — should be consistent
    read = store.get("k1")
    assert read["value"] == "v2"
    assert read["consistent"]


def test_concurrent_writes_version_divergence():
    """Test concurrent writes to different replicas creating version divergence."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    store.put("k1", "v1")  # v1 on replicas 0,1

    # Simulate concurrent writes by directly writing to individual replicas
    store.replicas[0].put("k1", "v_from_0", 2)
    store.replicas[2].put("k1", "v_from_2", 2)

    # Read repair should pick one (highest version — tied, so first encountered)
    # and repair stale replicas
    read = store.get("k1")
    assert read["version"] == 2
    # The value should be from one of the concurrent writes
    assert read["value"] in ("v_from_0", "v_from_2", "v1")


def test_read_nonexistent_key():
    """Test reading a key that doesn't exist on any replica."""
    store = ReadRepairStore(num_replicas=3, read_quorum=2, write_quorum=2)
    read = store.get("nonexistent")
    assert read["value"] is None
    assert read["version"] == 0
    assert read["consistent"]


def test_quorum_warning():
    """Test that weak quorum configuration emits a warning."""
    with pytest.warns(UserWarning, match="Quorum condition not met"):
        ReadRepairStore(num_replicas=5, read_quorum=2, write_quorum=2)


def test_example_usage():
    """Test the example from the spec."""
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

    states = store.get_replica_states("user:1")
    versions = [s["version"] for s in states if s["available"]]
    assert len(set(versions)) == 1

    stats = store.get_repair_stats()
    assert stats["reads_triggering_repair"] >= 1
