"""Tests for Dynamo-style leaderless replication."""

import pytest
from dynamo import DynamoCluster, QuorumNotMet, ReplicaNode, VersionedValue


# 1. Basic put/get with all nodes available
def test_basic_put_get():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)
    v = cluster.put("key1", "value1")
    assert v == 1
    result = cluster.get("key1")
    assert result.value == "value1"
    assert result.version == 1
    assert not result.is_conflict


# 2. Quorum writes: W=2, succeeds with 2 up, fails with 1
def test_quorum_writes():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)

    # 2 of 3 up -> succeeds
    cluster.set_node_available("node_2", False)
    v = cluster.put("key1", "val")
    assert v == 1

    # Only 1 up -> fails
    cluster.set_node_available("node_1", False)
    with pytest.raises(QuorumNotMet):
        cluster.put("key2", "val")


# 3. Quorum reads: R=2, succeeds with 2 up, fails with 1
def test_quorum_reads():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)
    cluster.put("key1", "val")

    # 2 of 3 up -> succeeds
    cluster.set_node_available("node_2", False)
    result = cluster.get("key1")
    assert result.value == "val"

    # Only 1 up -> fails
    cluster.set_node_available("node_1", False)
    with pytest.raises(QuorumNotMet):
        cluster.get("key1")


# 4. Read repair
def test_read_repair():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)
    cluster.put("key1", "v1")

    # Take node down, write new value
    cluster.set_node_available("node_2", False)
    cluster.put("key1", "v2")

    # Bring node back - it still has v1
    cluster.set_node_available("node_2", True)
    stale = cluster.get_node("node_2").read("key1")
    assert stale.version == 1

    # Read triggers repair
    result = cluster.get("key1")
    assert result.value == "v2"
    assert result.replicas_repaired >= 1

    # Verify node_2 is now up to date
    repaired = cluster.get_node("node_2").read("key1")
    assert repaired.version == 2
    assert repaired.value == "v2"


# 5. Different quorum configurations
def test_quorum_configurations():
    # W=1, R=3: write-one-read-all
    c1 = DynamoCluster(num_replicas=3, write_quorum=1, read_quorum=3)
    c1.put("k", "v")
    result = c1.get("k")
    assert result.value == "v"

    # W=3, R=1: write-all-read-one
    c2 = DynamoCluster(num_replicas=3, write_quorum=3, read_quorum=1)
    c2.put("k", "v")
    result = c2.get("k")
    assert result.value == "v"

    # W=2, R=2: balanced
    c3 = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)
    c3.put("k", "v")
    result = c3.get("k")
    assert result.value == "v"


# 6. W + R > N guarantees reading latest write
def test_quorum_overlap_guarantees_latest():
    cluster = DynamoCluster(num_replicas=5, write_quorum=3, read_quorum=3)
    # W+R=6 > N=5, so quorum overlap guarantees freshness

    cluster.put("key", "old")
    cluster.put("key", "new")

    # Even with some nodes down (as long as quorum met)
    cluster.set_node_available("node_4", False)
    cluster.set_node_available("node_3", False)
    result = cluster.get("key")
    assert result.value == "new"
    assert result.version == 2


# 7. Sloppy quorum and hinted handoff
def test_sloppy_quorum():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2,
                            sloppy_quorum=True)

    cluster.set_node_available("node_2", False)
    cluster.put("key1", "value1")

    # Bring node back and deliver hints
    cluster.set_node_available("node_2", True)
    delivered = cluster.deliver_hints()
    assert delivered >= 1

    # Verify data is now on the previously-down node
    val = cluster.get_node("node_2").read("key1")
    assert val is not None
    assert val.value == "value1"


# 8. Anti-entropy repair
def test_anti_entropy_repair():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)
    cluster.put("key1", "v1")
    cluster.put("key1", "v2")

    # Manually stale a node by directly manipulating its store
    node = cluster.get_node("node_0")
    node._store["key1"] = VersionedValue(value="v1", version=1, node_id="node_0")

    # Run anti-entropy
    repairs = cluster.anti_entropy_repair()
    assert repairs >= 1

    # Verify convergence
    for nid in ["node_0", "node_1", "node_2"]:
        val = cluster.get_node(nid).read("key1")
        assert val.version == 2
        assert val.value == "v2"


# 9. Concurrent writes / conflict detection
def test_conflict_detection():
    cluster = DynamoCluster(num_replicas=3, write_quorum=1, read_quorum=3)

    # Simulate concurrent writes: different values at the same version on different nodes
    # Write "A" to node_0 only
    cluster.set_node_available("node_1", False)
    cluster.set_node_available("node_2", False)
    cluster.put("key1", "A")  # version 1, only on node_0

    # Write "B" to node_1 only (simulate concurrent write)
    cluster.set_node_available("node_0", False)
    cluster.set_node_available("node_1", True)
    cluster.put("key1", "B")  # version 2, only on node_1

    # Write "C" to node_2 only at the same version to create a conflict
    cluster.set_node_available("node_1", False)
    cluster.set_node_available("node_2", True)
    cluster.put("key1", "C")  # version 3, only on node_2

    # Now manually set node_1 and node_2 to same version but different values
    cluster.get_node("node_1")._store["key1"] = VersionedValue("X", 5, "node_1")
    cluster.get_node("node_2")._store["key1"] = VersionedValue("Y", 5, "node_2")
    cluster.get_node("node_0")._store["key1"] = VersionedValue("Z", 4, "node_0")

    # Bring all nodes up
    cluster.set_node_available("node_0", True)
    cluster.set_node_available("node_1", True)
    cluster.set_node_available("node_2", True)

    result = cluster.get("key1")
    assert result.is_conflict
    assert result.version == 5
    assert "X" in result.value
    assert "Y" in result.value


# 10. Monotonically increasing versions
def test_monotonic_versions():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)
    versions = []
    for i in range(20):
        v = cluster.put("key1", f"value_{i}")
        versions.append(v)

    # Versions should be strictly increasing
    for i in range(1, len(versions)):
        assert versions[i] > versions[i - 1]

    # Verify final version
    result = cluster.get("key1")
    assert result.version == 20


# Bonus: test with N=7 and many operations
def test_large_cluster_performance():
    cluster = DynamoCluster(num_replicas=7, write_quorum=4, read_quorum=4)
    for i in range(10_000):
        cluster.put(f"key_{i % 100}", f"value_{i}")

    for i in range(100):
        result = cluster.get(f"key_{i}")
        assert result.value is not None


# Bonus: test example from spec
def test_spec_example():
    cluster = DynamoCluster(num_replicas=3, write_quorum=2, read_quorum=2)

    cluster.put("user:1", {"name": "Alice", "age": 30})
    result = cluster.get("user:1")
    assert result.value == {"name": "Alice", "age": 30}
    assert result.version == 1
    assert not result.is_conflict

    cluster.set_node_available("node_2", False)
    cluster.put("user:1", {"name": "Alice", "age": 31})
    result = cluster.get("user:1")
    assert result.value["age"] == 31

    cluster.set_node_available("node_2", True)
    result = cluster.get("user:1")
    assert result.replicas_repaired >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
