"""Tests for Raft consensus algorithm implementation."""

import random

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


def test_stale_log_candidate_loses_election():
    """Test 7: A candidate with a stale log cannot win election."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader_id = cluster.run_until_leader()

    cluster.submit("cmd1")
    cluster.submit("cmd2")
    assert cluster.run_until_committed(2)

    # Partition one follower so it misses future entries
    stale = [nid for nid in cluster.nodes if nid != leader_id][0]
    cluster.partition_node(stale)

    cluster.submit("cmd3")
    cluster.submit("cmd4")
    assert cluster.run_until_committed(4)

    # Now partition the leader and heal the stale node
    cluster.partition_node(leader_id)
    cluster.heal_node(stale)

    # Let elections happen
    for _ in range(200):
        cluster.tick(10)
    new_leader = cluster.run_until_leader()
    assert new_leader is not None
    # The stale node should not have been elected — it's missing entries
    # that the majority has
    assert new_leader != stale


def test_split_vote_resolves():
    """Test 8: Split vote scenario resolves in a subsequent election."""
    random.seed(42)
    # Use tight timeout range to increase chance of simultaneous candidates
    cluster = RaftCluster(["n1", "n2", "n3"], election_timeout_range=(150, 155))
    # Even if split votes occur, a leader must eventually emerge
    leader = cluster.run_until_leader(max_ticks=2000)
    assert leader is not None


def test_terms_monotonically_increasing():
    """Test 11: Terms are monotonically increasing across leader changes."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    terms = []

    leader_id = cluster.run_until_leader()
    terms.append(cluster.nodes[leader_id].current_term)
    prev_leader = leader_id

    for _ in range(3):
        cluster.partition_node(leader_id)
        for _ in range(50):
            cluster.tick(10)
        leader_id = cluster.run_until_leader()
        assert leader_id is not None
        terms.append(cluster.nodes[leader_id].current_term)
        cluster.heal_node(prev_leader)
        for _ in range(50):
            cluster.tick(10)
        prev_leader = leader_id

    for i in range(1, len(terms)):
        assert terms[i] > terms[i - 1]


def test_sequential_leader_failures():
    """Test 13: Multiple sequential leader failures and recoveries."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader_id = cluster.run_until_leader()

    cluster.submit("a")
    assert cluster.run_until_committed(1)

    partitioned = []
    for i in range(2):
        cluster.partition_node(leader_id)
        partitioned.append(leader_id)
        for _ in range(50):
            cluster.tick(10)
        leader_id = cluster.run_until_leader()
        assert leader_id is not None
        cluster.submit(f"fail-{i}")

    # Heal all and verify consistency
    for nid in partitioned:
        cluster.heal_node(nid)
    for _ in range(200):
        cluster.tick(10)

    committed = cluster.get_committed_log()
    assert "a" in committed


def test_uncommitted_entries_overwritten():
    """Test 14: Uncommitted entries from a deposed leader are overwritten."""
    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader1 = cluster.run_until_leader()

    cluster.submit("committed")
    assert cluster.run_until_committed(1)

    # Partition the leader immediately after appending an uncommitted entry
    # so it can't replicate to a majority
    followers = [nid for nid in cluster.nodes if nid != leader1]
    for f in followers:
        cluster.partition_node(f)
    cluster.nodes[leader1].client_request("doomed")
    for f in followers:
        cluster.heal_node(f)
    cluster.partition_node(leader1)

    # A new leader is elected among the followers
    for _ in range(100):
        cluster.tick(10)
    leader2 = cluster.run_until_leader()
    assert leader2 is not None
    assert leader2 != leader1

    # New leader writes a different entry at the same index
    cluster.submit("replacement")
    assert cluster.run_until_committed(2)

    # Heal old leader — it must accept the new leader's log
    cluster.heal_node(leader1)
    for _ in range(200):
        cluster.tick(10)

    old_log = cluster.nodes[leader1].get_log()
    old_commands = [e.command for e in old_log[1:]]
    assert "doomed" not in old_commands
    assert "committed" in old_commands
    assert "replacement" in old_commands


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
