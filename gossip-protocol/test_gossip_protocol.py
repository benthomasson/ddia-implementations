"""Tests for gossip protocol implementation."""

import math
from gossip_protocol import GossipNode, GossipCluster


def test_convergence_same_membership():
    """1. All nodes converge to the same membership list after several rounds."""
    cluster = GossipCluster(t_suspect=5, t_dead=10, t_cleanup=20, seed=42)
    nodes = [cluster.add_node(f"node-{i}") for i in range(5)]
    cluster.run_rounds(10, start_time=0)

    alive_sets = [set(n.get_alive_members()) for n in nodes]
    expected = {f"node-{i}" for i in range(5)}
    for s in alive_sets:
        assert s == expected, f"Expected {expected}, got {s}"


def test_crash_suspect_dead_cleanup():
    """2. A crashed node is detected as suspected, then dead, then cleaned up."""
    cluster = GossipCluster(t_suspect=3, t_dead=6, t_cleanup=12, seed=42)
    for i in range(4):
        cluster.add_node(f"n{i}")

    # Run a few rounds so everyone knows everyone (times 0-2)
    cluster.run_rounds(3, start_time=0)

    # Crash n2 - its last heartbeat was at time 2 at the latest
    cluster.remove_node("n2")

    # Run 6 rounds (times 3-8). At time 8, elapsed >= 6 > t_suspect=3
    cluster.run_rounds(6, start_time=3)
    n0 = cluster.nodes["n0"]
    ml = n0.get_membership_list()
    assert ml["n2"]["status"] in ("suspected", "dead"), f"Expected suspected/dead, got {ml['n2']['status']}"

    # Run more rounds (times 9-12). At time 12, elapsed >= 10 > t_dead=6
    cluster.run_rounds(4, start_time=9)
    ml = n0.get_membership_list()
    assert ml["n2"]["status"] == "dead", f"Expected dead, got {ml['n2']['status']}"

    # Run until cleanup (times 13-22). At time 16+, elapsed > t_cleanup=12
    cluster.run_rounds(10, start_time=13)
    ml = n0.get_membership_list()
    assert "n2" not in ml, "n2 should have been cleaned up"


def test_voluntary_leave():
    """3. Voluntary leave is propagated to all nodes."""
    cluster = GossipCluster(t_suspect=5, t_dead=10, t_cleanup=20, seed=42)
    nodes = [cluster.add_node(f"n{i}") for i in range(4)]

    cluster.run_rounds(5, start_time=0)

    # n1 leaves voluntarily
    nodes[1].leave()

    # Gossip a few rounds to propagate
    cluster.run_rounds(5, start_time=5)

    for n in [nodes[0], nodes[2], nodes[3]]:
        assert "n1" not in n.get_alive_members(), f"{n.node_id} still sees n1 as alive"


def test_new_node_discovered():
    """4. A new node joining mid-cluster is discovered by all existing nodes."""
    cluster = GossipCluster(t_suspect=5, t_dead=10, t_cleanup=20, seed=42)
    for i in range(4):
        cluster.add_node(f"n{i}")

    cluster.run_rounds(5, start_time=0)

    # Add new node
    cluster.add_node("n4")
    cluster.run_rounds(5, start_time=5)

    for nid in ["n0", "n1", "n2", "n3"]:
        assert "n4" in cluster.nodes[nid].get_alive_members()


def test_convergence_speed_olog_n():
    """5. Convergence speed is O(log N) for membership information."""
    for n_nodes in [8, 16, 32, 64]:
        cluster = GossipCluster(t_suspect=100, t_dead=200, t_cleanup=400, seed=42)
        # Add first node
        cluster.add_node("seed")
        cluster.run_rounds(1, start_time=0)

        # Add remaining nodes
        for i in range(1, n_nodes):
            cluster.add_node(f"node-{i}")

        # Run rounds and check when all nodes know about everyone
        max_rounds = int(5 * math.log2(n_nodes)) + 10
        all_node_ids = set(cluster.nodes.keys())

        converged_round = None
        for r in range(max_rounds):
            cluster.gossip_round(r + 1)
            all_converged = all(
                set(node.get_alive_members()) == all_node_ids
                for node in cluster.nodes.values()
            )
            if all_converged:
                converged_round = r + 1
                break

        assert converged_round is not None, f"Failed to converge with {n_nodes} nodes in {max_rounds} rounds"
        # Should converge in O(log N) rounds
        assert converged_round <= 5 * math.log2(n_nodes) + 5, \
            f"Convergence took {converged_round} rounds for {n_nodes} nodes (expected ~{math.log2(n_nodes):.0f})"


def test_heartbeat_monotonically_increasing():
    """6. Heartbeat counters are monotonically increasing."""
    node = GossipNode("test", t_suspect=5, t_dead=10, t_cleanup=20)
    prev = 0
    for t in range(20):
        node.heartbeat(t)
        current = node.membership["test"]["heartbeat_counter"]
        assert current > prev, f"Heartbeat not increasing: {prev} -> {current}"
        prev = current


def test_merge_conflicting_info():
    """7. Merging membership lists with conflicting information."""
    n1 = GossipNode("n1")
    n2 = GossipNode("n2")
    n1.join(n2)

    # Advance n1's heartbeat significantly
    for t in range(10):
        n1.heartbeat(t)

    # n2 has stale info about n1
    assert n2.membership["n1"]["heartbeat_counter"] == 0

    # After receiving gossip, n2 should have updated counter
    gossip = n1.send_gossip()
    n2.receive_gossip(gossip, 10)
    assert n2.membership["n1"]["heartbeat_counter"] == 10

    # Heartbeat should never decrease
    stale_gossip = {"n1": {"heartbeat_counter": 5, "timestamp_last_updated": 5, "status": "alive"}}
    n2.receive_gossip(stale_gossip, 11)
    assert n2.membership["n1"]["heartbeat_counter"] == 10, "Heartbeat counter decreased!"


def test_single_node_cluster():
    """8. Single-node cluster behavior."""
    cluster = GossipCluster(t_suspect=3, t_dead=6, t_cleanup=12, seed=42)
    n = cluster.add_node("solo")

    cluster.run_rounds(10, start_time=0)

    assert n.get_alive_members() == ["solo"]
    assert n.membership["solo"]["heartbeat_counter"] == 10


def test_simultaneous_failures():
    """9. Simultaneous failure of multiple nodes."""
    cluster = GossipCluster(t_suspect=3, t_dead=6, t_cleanup=12, seed=42)
    for i in range(6):
        cluster.add_node(f"n{i}")

    cluster.run_rounds(5, start_time=0)

    # Crash 3 nodes at once
    for i in [1, 3, 5]:
        cluster.remove_node(f"n{i}")

    # Run enough rounds for detection
    cluster.run_rounds(10, start_time=5)

    # Surviving nodes should detect all failures
    for survivor_id in ["n0", "n2", "n4"]:
        alive = cluster.nodes[survivor_id].get_alive_members()
        for failed in ["n1", "n3", "n5"]:
            assert failed not in alive, f"{survivor_id} still sees {failed} as alive"


def test_dead_node_rejoin():
    """10. A node declared dead can rejoin with a fresh state."""
    cluster = GossipCluster(t_suspect=3, t_dead=6, t_cleanup=12, seed=42)
    for i in range(4):
        cluster.add_node(f"n{i}")

    cluster.run_rounds(3, start_time=0)

    # Crash n2
    cluster.remove_node("n2")

    # Wait for cleanup
    cluster.run_rounds(15, start_time=3)

    # Verify n2 is cleaned up from n0's list
    ml = cluster.nodes["n0"].get_membership_list()
    assert "n2" not in ml, "n2 should be cleaned up before rejoin"

    # Rejoin: remove from failed set and re-add
    cluster.failed_nodes.discard("n2")
    del cluster.nodes["n2"]
    cluster.add_node("n2")

    cluster.run_rounds(5, start_time=18)

    # All surviving nodes should see n2 as alive again
    for nid in ["n0", "n1", "n3"]:
        assert "n2" in cluster.nodes[nid].get_alive_members(), \
            f"{nid} doesn't see rejoined n2 as alive"


if __name__ == "__main__":
    tests = [
        test_convergence_same_membership,
        test_crash_suspect_dead_cleanup,
        test_voluntary_leave,
        test_new_node_discovered,
        test_convergence_speed_olog_n,
        test_heartbeat_monotonically_increasing,
        test_merge_conflicting_info,
        test_single_node_cluster,
        test_simultaneous_failures,
        test_dead_node_rejoin,
    ]
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
