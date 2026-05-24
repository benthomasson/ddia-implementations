"""QA tester tests for Dynamo-style leaderless replication."""


import pytest
from dynamo import DynamoCluster, QuorumNotMet, ReplicaNode, VersionedValue, ReadResult


# 1. Spec example: end-to-end from the task description
def test_spec_example_full():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)

    cluster.put("user:1", {"name": "Alice", "age": 30})
    result = cluster.get("user:1")
    assert result.value == {"name": "Alice", "age": 30}
    assert result.version == 1
    assert not result.is_conflict

    # Write succeeds with one node down (W=2, 2 of 3 available)
    cluster.set_node_available("node_2", False)
    cluster.put("user:1", {"name": "Alice", "age": 31})
    result = cluster.get("user:1")
    assert result.value["age"] == 31

    # Read repair when node comes back
    cluster.set_node_available("node_2", True)
    result = cluster.get("user:1")
    assert result.replicas_repaired >= 1

    # Write fails when too many nodes are down
    cluster.set_node_available("node_0", False)
    cluster.set_node_available("node_1", False)
    with pytest.raises(QuorumNotMet):
        cluster.put("user:2", "value")

    # Anti-entropy repair
    cluster.set_node_available("node_0", True)
    cluster.set_node_available("node_1", True)
    repairs = cluster.anti_entropy_repair()
    assert repairs >= 0


# 2. Spec example: sloppy quorum with hinted handoff
def test_spec_sloppy_quorum_example():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2,
                            sloppy_quorum=True)
    cluster.set_node_available("node_2", False)
    cluster.put("key1", "value1")  # hint stored on another node
    cluster.set_node_available("node_2", True)
    delivered = cluster.deliver_hints()
    assert delivered >= 1


# 3. Read nonexistent key returns None/version 0
def test_read_nonexistent_key():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)
    result = cluster.get("nonexistent")
    assert result.value is None
    assert result.version == 0
    assert not result.is_conflict


# 4. Version rollback on failed write
def test_version_rollback_on_failed_write():
    cluster = DynamoCluster(num_replicas=3, write_quorum=3, read_quorum=1)
    v1 = cluster.put("k", "first")
    assert v1 == 1

    # Take one node down so W=3 can't be met
    cluster.set_node_available("node_2", False)
    with pytest.raises(QuorumNotMet):
        cluster.put("k", "second")

    # Bring node back, next successful write should be version 2, not 3
    cluster.set_node_available("node_2", True)
    v2 = cluster.put("k", "third")
    assert v2 == 2


# 5. ReplicaNode refuses writes when unavailable
def test_node_unavailable_rejects_operations():
    node = ReplicaNode("test_node")
    assert node.write("k", "v", 1) is True
    assert node.read("k").value == "v"

    node.set_available(False)
    assert node.is_available is False
    assert node.write("k", "v2", 2) is False
    assert node.read("k") is None

    node.set_available(True)
    # Original value should still be there
    assert node.read("k").value == "v"
    assert node.read("k").version == 1


# 6. Multiple keys are independent
def test_multiple_keys_independent():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)
    cluster.put("a", 1)
    cluster.put("b", 2)
    cluster.put("a", 10)

    ra = cluster.get("a")
    rb = cluster.get("b")
    assert ra.value == 10
    assert ra.version == 2  # second write to "a"
    assert rb.value == 2
    assert rb.version == 1  # first write to "b"


# 7. Conflict detection with manually set divergent state
def test_conflict_detection():
    cluster = DynamoCluster(num_replicas=3, write_quorum=1, read_quorum=3)

    # Manually inject conflicting state: same version, different values
    cluster.get_node("node_0")._store["k"] = VersionedValue("X", 5, "node_0")
    cluster.get_node("node_1")._store["k"] = VersionedValue("Y", 5, "node_1")
    cluster.get_node("node_2")._store["k"] = VersionedValue("Z", 3, "node_2")

    result = cluster.get("k")
    assert result.is_conflict is True
    assert result.version == 5
    assert isinstance(result.value, list)
    assert "X" in result.value
    assert "Y" in result.value


# 8. N=7 cluster with 10,000 operations (performance constraint)
def test_large_cluster_10k_ops():
    cluster = DynamoCluster(num_replicas=7, write_quorum=4, read_quorum=4)
    for i in range(10_000):
        cluster.put(f"key_{i % 50}", f"val_{i}")

    # Read all 50 keys
    for i in range(50):
        r = cluster.get(f"key_{i}")
        assert r.value is not None
        assert r.version > 0


# 9. Anti-entropy with a node that missed everything
def test_anti_entropy_full_sync():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)
    cluster.set_node_available("node_2", False)

    # Write several keys while node_2 is down
    for i in range(5):
        cluster.put(f"k{i}", f"v{i}")

    cluster.set_node_available("node_2", True)
    repairs = cluster.anti_entropy_repair()
    assert repairs >= 5  # at least 5 keys need repair on node_2

    # Verify node_2 has all keys now
    for i in range(5):
        val = cluster.get_node("node_2").read(f"k{i}")
        assert val is not None
        assert val.value == f"v{i}"


# 10. Hints are not stored when sloppy_quorum is disabled
def test_no_hints_without_sloppy_quorum():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2,
                            sloppy_quorum=False)
    cluster.set_node_available("node_2", False)
    cluster.put("k", "v")

    cluster.set_node_available("node_2", True)
    delivered = cluster.deliver_hints()
    assert delivered == 0

    # node_2 should NOT have the data
    val = cluster.get_node("node_2").read("k")
    assert val is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
