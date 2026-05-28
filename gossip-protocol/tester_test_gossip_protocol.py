"""Tests for gossip protocol implementation."""

import math
from gossip_protocol import GossipNode, GossipCluster


def test_convergence_same_membership():
    """1. All nodes converge to the same membership list after several rounds."""
    cluster = GossipCluster(t_suspect=5, t_dead=10, t_cleanup=20, seed=42)
    nodes = [cluster.add_node(f"node-{i}") for i in range(5)]
    cluster.run_rounds(10, start_time=0)

    expected = {f"node-{i}" for i in range(5)}
    for n in nodes:
        assert set(n.get_alive_members()) == expected


def test_crash_suspect_dead_cleanup():
    """2. A crashed node transitions: suspected -> dead -> cleaned up."""
    cluster = GossipCluster(t_suspect=3, t_dead=6, t_cleanup=12, seed=42)
    for i in range(4):
        cluster.add_node(f"n{i}")

    cluster.run_rounds(3, start_time=0)
    cluster.remove_node("n2")

    # After enough time, n2 should be suspected or dead
    cluster.run_rounds(6, start_time=3)
    n0 = cluster.nodes["n0"]
    ml = n0.get_membership_list()
    assert ml["n2"]["status"] in ("suspected", "dead")

    # After more time, n2 should be dead
    cluster.run_rounds(4, start_time=9)
    ml = n0.get_membership_list()
    assert ml["n2"]["status"] == "dead"

    # After cleanup threshold, n2 should be removed
    cluster.run_rounds(10, start_time=13)
    ml = n0.get_membership_list()
    assert "n2" not in ml


def test_voluntary_leave():
    """3. Voluntary leave is propagated to all nodes."""
    cluster = GossipCluster(t_suspect=5, t_dead=10, t_cleanup=20, seed=42)
    nodes = [cluster.add_node(f"n{i}") for i in range(4)]
    cluster.run_rounds(5, start_time=0)

    nodes[1].leave()
    cluster.run_rounds(5, start_time=5)

    for n in [nodes[0], nodes[2], nodes[3]]:
        assert "n1" not in n.get_alive_members()


def test_new_node_discovered():
    """4. A new node joining mid-cluster is discovered by all existing nodes."""
    cluster = GossipCluster(t_suspect=5, t_dead=10, t_cleanup=20, seed=42)
    for i in range(4):
        cluster.add_node(f"n{i}")
    cluster.run_rounds(5, start_time=0)

    cluster.add_node("n4")
    cluster.run_rounds(5, start_time=5)

    for nid in ["n0", "n1", "n2", "n3"]:
        assert "n4" in cluster.nodes[nid].get_alive_members()


def test_convergence_speed_olog_n():
    """5. Convergence speed is O(log N) for membership information."""
    for n_nodes in [8, 16, 32]:
        cluster = GossipCluster(t_suspect=100, t_dead=200, t_cleanup=400, seed=42)
        cluster.add_node("seed")
        cluster.run_rounds(1, start_time=0)

        for i in range(1, n_nodes):
            cluster.add_node(f"node-{i}")

        max_rounds = int(5 * math.log2(n_nodes)) + 10
        all_node_ids = set(cluster.nodes.keys())

        converged_round = None
        for r in range(max_rounds):
            cluster.gossip_round(r + 1)
            if all(set(node.get_alive_members()) == all_node_ids
                   for node in cluster.nodes.values()):
                converged_round = r + 1
                break

        assert converged_round is not None, \
            f"Failed to converge with {n_nodes} nodes in {max_rounds} rounds"


def test_heartbeat_monotonically_increasing():
    """6. Heartbeat counters are monotonically increasing."""
    node = GossipNode("test", t_suspect=5, t_dead=10, t_cleanup=20)
    prev = 0
    for t in range(20):
        node.heartbeat(t)
        current = node.membership["test"]["heartbeat_counter"]
        assert current > prev
        prev = current


def test_merge_conflicting_info():
    """7. Merging membership lists: higher heartbeat wins, never decrease."""
    n1 = GossipNode("n1")
    n2 = GossipNode("n2")
    n1.join(n2)

    for t in range(10):
        n1.heartbeat(t)

    # n2 has stale info
    assert n2.membership["n1"]["heartbeat_counter"] == 0

    # After gossip, n2 gets updated counter
    n2.receive_gossip(n1.send_gossip(), 10)
    assert n2.membership["n1"]["heartbeat_counter"] == 10

    # Stale gossip should not decrease counter
    stale = {"n1": {"heartbeat_counter": 5, "timestamp_last_updated": 5, "status": "alive"}}
    n2.receive_gossip(stale, 11)
    assert n2.membership["n1"]["heartbeat_counter"] == 10


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

    for i in [1, 3, 5]:
        cluster.remove_node(f"n{i}")

    cluster.run_rounds(10, start_time=5)

    for survivor in ["n0", "n2", "n4"]:
        alive = cluster.nodes[survivor].get_alive_members()
        for failed in ["n1", "n3", "n5"]:
            assert failed not in alive


def test_dead_node_rejoin():
    """10. A node declared dead can rejoin with a fresh state."""
    cluster = GossipCluster(t_suspect=3, t_dead=6, t_cleanup=12, seed=42)
    for i in range(4):
        cluster.add_node(f"n{i}")
    cluster.run_rounds(3, start_time=0)

    cluster.remove_node("n2")
    cluster.run_rounds(15, start_time=3)

    # n2 should be cleaned up
    assert "n2" not in cluster.nodes["n0"].get_membership_list()

    # Rejoin
    cluster.failed_nodes.discard("n2")
    del cluster.nodes["n2"]
    cluster.add_node("n2")
    cluster.run_rounds(5, start_time=18)

    for nid in ["n0", "n1", "n3"]:
        assert "n2" in cluster.nodes[nid].get_alive_members()


def test_join_does_not_use_stale_timestamp():
    """Node joining at a later time must not be immediately marked dead by the seed."""
    cluster = GossipCluster(t_suspect=3, t_dead=6, t_cleanup=12, seed=42)
    for i in range(3):
        cluster.add_node(f"n{i}")
    cluster.run_rounds(20, start_time=0)

    cluster.add_node("late_joiner")
    cluster.gossip_round(20)

    seed = None
    for nid, node in cluster.nodes.items():
        if nid != "late_joiner" and "late_joiner" in node.get_membership_list():
            seed = node
            break
    assert seed is not None
    ml = seed.get_membership_list()
    assert ml["late_joiner"]["status"] == "alive", \
        f"Newly joined node immediately marked {ml['late_joiner']['status']}"


def test_new_node_via_gossip_gets_current_timestamp():
    """When a node learns about a new peer through gossip, it should use current_time."""
    n1 = GossipNode("n1")
    n2 = GossipNode("n2")
    n1.join(n2)

    for t in range(20):
        n1.heartbeat(t)
        n2.heartbeat(t)

    n3 = GossipNode("n3")
    n3.join(n1)

    n2.receive_gossip(n1.send_gossip(), current_time=20)
    assert "n3" in n2.membership
    assert n2.membership["n3"]["status"] == "alive"

    n2.detect_failures(current_time=20)
    assert n2.membership["n3"]["status"] == "alive", \
        "Node learned via gossip should not be immediately suspected"
