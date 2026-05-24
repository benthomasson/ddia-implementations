"""Tests for Raft consensus algorithm implementation."""


import pytest
from raft import LogEntry, RaftNode, RaftCluster


def test_leader_elected_from_fresh_cluster():
    """Test 1: A leader is elected from a fresh cluster."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader = cluster.run_until_leader()
    assert leader is not None
    assert cluster.nodes[leader].state == "leader"


def test_all_nodes_start_as_followers():
    """Test 2: All nodes start as followers."""
    cluster = RaftCluster(["n1", "n2", "n3"])
    for node in cluster.nodes.values():
        assert node.state == "follower"
        assert node.current_term == 0


def test_log_replication_and_commit():
    """Test 3: Log entries are replicated and committed."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader_id = cluster.run_until_leader()
    assert leader_id is not None

    cluster.submit("SET x = 1")
    cluster.submit("SET y = 2")
    assert cluster.run_until_committed(2)

    committed = cluster.get_committed_log()
    assert committed == ["SET x = 1", "SET y = 2"]


def test_new_leader_after_partition():
    """Test 4: New leader elected after current leader is partitioned."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader_id = cluster.run_until_leader()
    assert leader_id is not None

    cluster.partition_node(leader_id)
    # Tick enough for election timeout
    for _ in range(50):
        cluster.tick(10)
    new_leader = cluster.run_until_leader()
    assert new_leader is not None
    assert new_leader != leader_id


def test_committed_entries_survive_leader_change():
    """Test 5: New leader's log contains all previously committed entries."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader_id = cluster.run_until_leader()

    cluster.submit("cmd1")
    cluster.submit("cmd2")
    assert cluster.run_until_committed(2)

    cluster.partition_node(leader_id)
    for _ in range(50):
        cluster.tick(10)
    new_leader = cluster.run_until_leader()
    assert new_leader is not None

    # New leader must have committed entries
    new_log = cluster.nodes[new_leader].get_committed_entries()
    commands = [e.command for e in new_log]
    assert "cmd1" in commands
    assert "cmd2" in commands


def test_client_request_rejected_by_follower():
    """Test 9: Client requests are rejected by non-leader nodes."""
    cluster = RaftCluster(["n1", "n2", "n3"])
    # Before any election, all are followers
    result = cluster.nodes["n1"].client_request("cmd")
    assert result["success"] is False
    assert result["error"] == "not leader"


def test_heartbeat_prevents_elections():
    """Test 10: Heartbeats prevent unnecessary elections."""
    cluster = RaftCluster(["n1", "n2", "n3"])
    leader_id = cluster.run_until_leader()
    term_after_election = cluster.nodes[leader_id].current_term

    # Run for a while - heartbeats should keep things stable
    for _ in range(100):
        cluster.tick(10)

    assert cluster.get_leader() == leader_id
    assert cluster.nodes[leader_id].current_term == term_after_election


def test_minority_partition_cannot_elect_leader():
    """Test 12: A minority partition cannot elect a leader."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader_id = cluster.run_until_leader()

    # Partition 3 nodes (majority) - the remaining 2 cannot elect
    non_leader = [nid for nid in cluster.nodes if nid != leader_id]
    cluster.partition_node(leader_id)
    cluster.partition_node(non_leader[0])
    cluster.partition_node(non_leader[1])

    # The 2 remaining nodes should not be able to elect a leader
    for _ in range(100):
        cluster.tick(10)
    assert cluster.get_leader() is None


def test_heal_and_log_consistency():
    """Test 6: Log consistency after healing a partitioned node."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader_id = cluster.run_until_leader()

    cluster.submit("a")
    cluster.submit("b")
    assert cluster.run_until_committed(2)

    # Partition a follower, add more entries, then heal
    follower = [nid for nid in cluster.nodes if nid != leader_id][0]
    cluster.partition_node(follower)

    cluster.submit("c")
    assert cluster.run_until_committed(3)

    cluster.heal_node(follower)
    # Run ticks so the leader replicates to healed node
    for _ in range(100):
        cluster.tick(10)

    # The healed follower should have all entries
    log = cluster.nodes[follower].get_log()
    commands = [e.command for e in log[1:]]  # skip sentinel
    assert "a" in commands
    assert "b" in commands
    assert "c" in commands


def test_example_from_spec():
    """Test the example usage from the spec."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])

    leader_id = cluster.run_until_leader()
    assert leader_id is not None

    cluster.submit("SET x = 1")
    cluster.submit("SET y = 2")
    cluster.run_until_committed(2)

    committed = cluster.get_committed_log()
    assert committed == ["SET x = 1", "SET y = 2"]

    cluster.partition_node(leader_id)
    cluster.tick(500)
    new_leader = cluster.run_until_leader()
    assert new_leader is not None
    assert new_leader != leader_id

    cluster.submit("SET z = 3")
    cluster.run_until_committed(3)
    assert "SET z = 3" in cluster.get_committed_log()

    cluster.heal_node(leader_id)
    cluster.tick(300)
    node = cluster.nodes[leader_id]
    assert node.state == "follower"
