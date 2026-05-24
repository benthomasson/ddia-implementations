"""Tests for Bully Algorithm leader election implementation."""
import sys

from leader_election import Message, BullyNode, BullyElectionCluster


def test_initial_election_highest_wins():
    """Test 1: Initial election selects highest-ID node."""
    cluster = BullyElectionCluster(node_ids=[1, 2, 3, 4, 5], heartbeat_interval=3, election_timeout=10)
    leader = cluster.run_until_leader()
    assert leader == 5, f"Expected leader 5, got {leader}"


def test_all_nodes_agree_on_leader():
    """Test 2: All nodes agree on the leader after election."""
    cluster = BullyElectionCluster(node_ids=[1, 2, 3, 4, 5], heartbeat_interval=3, election_timeout=10)
    leader = cluster.run_until_leader()
    state = cluster.get_cluster_state()
    for nid in [1, 2, 3, 4]:
        assert state[nid]["leader_id"] == 5, f"Node {nid} disagrees: {state[nid]}"
        assert state[nid]["state"] == "follower"
    assert state[5]["state"] == "leader"


def test_leader_failure_triggers_new_election():
    """Test 3-4: Leader failure triggers new election, next highest wins."""
    cluster = BullyElectionCluster(node_ids=[1, 2, 3, 4, 5], heartbeat_interval=3, election_timeout=10)
    cluster.run_until_leader()
    cluster.fail_node(5)
    leader = cluster.run_until_leader(start_time=20)
    assert leader == 4, f"Expected leader 4 after failing 5, got {leader}"


def test_recovered_node_takes_over():
    """Test 5: Recovered high-ID node takes over leadership (bully)."""
    cluster = BullyElectionCluster(node_ids=[1, 2, 3, 4, 5], heartbeat_interval=3, election_timeout=10)
    cluster.run_until_leader()
    cluster.fail_node(5)
    cluster.run_until_leader(start_time=20)
    cluster.recover_node(5)
    leader = cluster.run_until_leader(start_time=40)
    assert leader == 5, f"Expected leader 5 after recovery, got {leader}"


def test_multiple_failures():
    """Test 6: Simultaneous failure of multiple nodes."""
    cluster = BullyElectionCluster(node_ids=[1, 2, 3, 4, 5], heartbeat_interval=3, election_timeout=10)
    cluster.run_until_leader()
    cluster.fail_node(5)
    cluster.fail_node(4)
    leader = cluster.run_until_leader(start_time=60)
    assert leader == 3, f"Expected leader 3 after failing 4 and 5, got {leader}"


def test_single_survivor():
    """Test 7: All nodes except one fail."""
    cluster = BullyElectionCluster(node_ids=[1, 2, 3], heartbeat_interval=3, election_timeout=10)
    cluster.run_until_leader()
    cluster.fail_node(3)
    cluster.fail_node(2)
    leader = cluster.run_until_leader(start_time=20)
    assert leader == 1, f"Expected leader 1 as sole survivor, got {leader}"


def test_single_node_cluster():
    """Test 8: Single-node cluster: that node is always leader."""
    cluster = BullyElectionCluster(node_ids=[1], heartbeat_interval=3, election_timeout=10)
    leader = cluster.run_until_leader()
    assert leader == 1
    assert cluster.get_cluster_state()[1]["state"] == "leader"


def test_election_history():
    """Test 12: Election history records leadership changes."""
    cluster = BullyElectionCluster(node_ids=[1, 2, 3, 4, 5], heartbeat_interval=3, election_timeout=10)
    cluster.run_until_leader()
    cluster.fail_node(5)
    cluster.run_until_leader(start_time=20)
    cluster.recover_node(5)
    cluster.run_until_leader(start_time=40)
    history = cluster.get_election_history()
    assert len(history) >= 3, f"Expected at least 3 history entries, got {len(history)}"
    assert history[0]["leader_id"] == 5


def test_terms_increase_monotonically():
    """Test 15: Election terms increase monotonically."""
    cluster = BullyElectionCluster(node_ids=[1, 2, 3, 4, 5], heartbeat_interval=3, election_timeout=10)
    cluster.run_until_leader()
    cluster.fail_node(5)
    cluster.run_until_leader(start_time=20)
    cluster.recover_node(5)
    cluster.run_until_leader(start_time=40)
    history = cluster.get_election_history()
    for i in range(1, len(history)):
        assert history[i]["term"] >= history[i - 1]["term"], \
            f"Terms not monotonic: {history[i-1]['term']} -> {history[i]['term']}"


def test_example_from_spec():
    """Full example from the task specification."""
    cluster = BullyElectionCluster(node_ids=[1, 2, 3, 4, 5], heartbeat_interval=3, election_timeout=10)

    leader = cluster.run_until_leader()
    assert leader == 5

    state = cluster.get_cluster_state()
    for nid in [1, 2, 3, 4]:
        assert state[nid]["leader_id"] == 5
        assert state[nid]["state"] == "follower"
    assert state[5]["state"] == "leader"

    cluster.fail_node(5)
    leader = cluster.run_until_leader(start_time=20)
    assert leader == 4

    cluster.recover_node(5)
    leader = cluster.run_until_leader(start_time=40)
    assert leader == 5

    cluster.fail_node(5)
    cluster.fail_node(4)
    leader = cluster.run_until_leader(start_time=60)
    assert leader == 3

    history = cluster.get_election_history()
    assert len(history) >= 3
    assert history[0]["leader_id"] == 5


if __name__ == "__main__":
    tests = [
        test_initial_election_highest_wins,
        test_all_nodes_agree_on_leader,
        test_leader_failure_triggers_new_election,
        test_recovered_node_takes_over,
        test_multiple_failures,
        test_single_survivor,
        test_single_node_cluster,
        test_election_history,
        test_terms_increase_monotonically,
        test_example_from_spec,
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
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        sys.exit(1)
