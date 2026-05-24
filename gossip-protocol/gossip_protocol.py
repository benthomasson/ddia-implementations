"""Gossip protocol for failure detection and membership."""

import copy
import random


class GossipNode:
    """A node in a gossip-based failure detection cluster."""

    def __init__(self, node_id: str, t_suspect: int = 5, t_dead: int = 10, t_cleanup: int = 20):
        self.node_id = node_id
        self.t_suspect = t_suspect
        self.t_dead = t_dead
        self.t_cleanup = t_cleanup
        self.membership = {
            node_id: {"heartbeat_counter": 0, "timestamp_last_updated": 0, "status": "alive"}
        }
        self._alive = True  # Whether this node is actively participating
        self._leaving = False  # Set during voluntary leave for broadcast

    def join(self, seed_node: 'GossipNode') -> None:
        """Join the cluster by contacting a seed node."""
        # Copy seed's membership list
        for nid, info in seed_node.membership.items():
            if nid != self.node_id:
                self.membership[nid] = copy.deepcopy(info)
        # Add self to seed's list
        seed_node.membership[self.node_id] = copy.deepcopy(self.membership[self.node_id])

    def leave(self) -> None:
        """Voluntarily leave the cluster. Marks self as dead.
        Node stays active briefly to broadcast its death via gossip.
        """
        self.membership[self.node_id]["status"] = "dead"
        self._leaving = True

    def heartbeat(self, current_time: int) -> None:
        """Increment own heartbeat counter and record the current time."""
        if self.membership[self.node_id]["status"] == "dead":
            return
        entry = self.membership[self.node_id]
        entry["heartbeat_counter"] += 1
        entry["timestamp_last_updated"] = current_time

    def send_gossip(self) -> dict:
        """Return this node's membership list for sending to a peer."""
        return copy.deepcopy(self.membership)

    def receive_gossip(self, membership_list: dict, current_time: int) -> None:
        """Merge received membership list with local list."""
        for nid, remote in membership_list.items():
            if nid not in self.membership:
                # Don't re-add nodes we've already cleaned up
                if remote["status"] == "dead":
                    continue
                self.membership[nid] = copy.deepcopy(remote)
            else:
                local = self.membership[nid]
                if remote["heartbeat_counter"] > local["heartbeat_counter"]:
                    local["heartbeat_counter"] = remote["heartbeat_counter"]
                    local["timestamp_last_updated"] = current_time
                    # Propagate status from remote if it has a higher counter
                    if remote["status"] == "dead":
                        local["status"] = "dead"
                    elif remote["status"] == "alive" and local["status"] != "dead":
                        local["status"] = "alive"
                elif (remote["status"] == "dead"
                      and remote["heartbeat_counter"] == local["heartbeat_counter"]
                      and local["status"] != "dead"):
                    # Accept death notification with equal heartbeat counter
                    local["status"] = "dead"

    def detect_failures(self, current_time: int) -> dict:
        """Check all nodes for failures based on timeout thresholds.

        Returns:
            Dict mapping node_id -> new status for nodes whose status changed.
        """
        changes = {}
        to_remove = []
        for nid, info in self.membership.items():
            if nid == self.node_id:
                continue
            elapsed = current_time - info["timestamp_last_updated"]
            old_status = info["status"]

            if info["status"] == "dead" and elapsed > self.t_cleanup:
                to_remove.append(nid)
                changes[nid] = "removed"
            elif elapsed > self.t_dead and info["status"] != "dead":
                info["status"] = "dead"
                if old_status != "dead":
                    changes[nid] = "dead"
            elif elapsed > self.t_suspect and info["status"] == "alive":
                info["status"] = "suspected"
                changes[nid] = "suspected"

        for nid in to_remove:
            del self.membership[nid]

        return changes

    def get_alive_members(self) -> list:
        """Return list of node_ids that are currently alive."""
        return [nid for nid, info in self.membership.items() if info["status"] == "alive"]

    def get_membership_list(self) -> dict:
        """Return the full membership list."""
        return copy.deepcopy(self.membership)


class GossipCluster:
    """Orchestrates gossip protocol simulation across multiple nodes."""

    def __init__(self, t_suspect: int = 5, t_dead: int = 10, t_cleanup: int = 20, seed: int = None):
        self.t_suspect = t_suspect
        self.t_dead = t_dead
        self.t_cleanup = t_cleanup
        self.nodes: dict[str, GossipNode] = {}
        self.failed_nodes: set[str] = set()  # Nodes that have "crashed"
        self.rng = random.Random(seed)

    def add_node(self, node_id: str) -> GossipNode:
        """Add a new node to the cluster."""
        node = GossipNode(node_id, self.t_suspect, self.t_dead, self.t_cleanup)
        # Join via any existing alive node
        for nid, existing in self.nodes.items():
            if nid not in self.failed_nodes and existing._alive:
                node.join(existing)
                break
        self.nodes[node_id] = node
        return node

    def remove_node(self, node_id: str) -> None:
        """Simulate a node crash (stops heartbeating, no leave message)."""
        self.failed_nodes.add(node_id)

    def gossip_round(self, current_time: int) -> None:
        """Execute one round of gossip."""
        active_ids = [nid for nid in self.nodes
                      if nid not in self.failed_nodes and self.nodes[nid]._alive]

        # Handle leaving nodes: broadcast to all peers, then deactivate
        leaving = [nid for nid in active_ids if self.nodes[nid]._leaving]
        for nid in leaving:
            node = self.nodes[nid]
            gossip = node.send_gossip()
            for peer_id in active_ids:
                if peer_id != nid:
                    self.nodes[peer_id].receive_gossip(gossip, current_time)
            node._alive = False
            node._leaving = False

        # Recompute active list after removing leaving nodes
        active_ids = [nid for nid in self.nodes
                      if nid not in self.failed_nodes and self.nodes[nid]._alive]

        # Each active node: heartbeat, pick random peer, exchange
        for nid in active_ids:
            node = self.nodes[nid]
            node.heartbeat(current_time)

            # Pick a random alive peer (excluding self)
            peers = [p for p in active_ids if p != nid]
            if peers:
                peer_id = self.rng.choice(peers)
                peer = self.nodes[peer_id]
                # Bidirectional exchange
                gossip_from_node = node.send_gossip()
                gossip_from_peer = peer.send_gossip()
                node.receive_gossip(gossip_from_peer, current_time)
                peer.receive_gossip(gossip_from_node, current_time)

        # All nodes run failure detection
        for nid in active_ids:
            self.nodes[nid].detect_failures(current_time)

    def run_rounds(self, num_rounds: int, start_time: int = 0) -> list:
        """Run multiple gossip rounds.

        Returns:
            List of dicts, each mapping node_id -> {members, suspected, dead}.
        """
        results = []
        for i in range(num_rounds):
            current_time = start_time + i
            self.gossip_round(current_time)

            round_state = {}
            for nid, node in self.nodes.items():
                ml = node.get_membership_list()
                round_state[nid] = {
                    "members": [n for n, info in ml.items() if info["status"] == "alive"],
                    "suspected": [n for n, info in ml.items() if info["status"] == "suspected"],
                    "dead": [n for n, info in ml.items() if info["status"] == "dead"],
                }
            results.append(round_state)

        return results
