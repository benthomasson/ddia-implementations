"""Consistent hashing ring with virtual nodes."""

import hashlib
import bisect
import threading
from typing import Dict, List, Tuple

RING_SIZE = 2**32


def _hash(key: str) -> int:
    """Hash a string to a 32-bit integer using MD5."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


class ConsistentHashRing:
    """Consistent hash ring with virtual nodes, replication, and weighted nodes."""

    def __init__(self, num_vnodes: int = 150, replication_factor: int = 1):
        self.num_vnodes = num_vnodes
        self.replication_factor = replication_factor
        self._ring_positions: List[int] = []  # sorted hash positions
        self._ring_nodes: List[str] = []      # node id at each position
        self._nodes: Dict[str, float] = {}    # node_id -> weight
        self._lock = threading.Lock()

    def add_node(self, node_id: str, weight: float = 1.0) -> Dict:
        """Add a physical node. Returns {(start, end): (from_node, to_node)} transfers."""
        with self._lock:
            return self._add_node(node_id, weight)

    def _add_node(self, node_id: str, weight: float = 1.0) -> Dict:
        if node_id in self._nodes:
            return {}
        self._nodes[node_id] = weight
        transfers = {}
        vnode_count = int(self.num_vnodes * weight)
        for i in range(vnode_count):
            pos = _hash(f"{node_id}:{i}")
            idx = bisect.bisect_left(self._ring_positions, pos)
            # Find which node previously owned this position
            if self._ring_positions:
                succ_idx = idx % len(self._ring_positions)
                old_owner = self._ring_nodes[succ_idx]
                # The arc [prev_vnode+1, pos] transfers from old_owner to node_id
                if old_owner != node_id:
                    prev_idx = (idx - 1) % len(self._ring_positions)
                    arc_start = (self._ring_positions[prev_idx] + 1) % RING_SIZE
                    transfers[(arc_start, pos)] = (old_owner, node_id)
            self._ring_positions.insert(idx, pos)
            self._ring_nodes.insert(idx, node_id)
        return transfers

    def remove_node(self, node_id: str) -> Dict:
        """Remove a physical node. Returns {(start, end): (from_node, to_node)} transfers."""
        with self._lock:
            return self._remove_node(node_id)

    def _remove_node(self, node_id: str) -> Dict:
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not in ring")
        del self._nodes[node_id]
        transfers = {}
        # Collect positions to remove (iterate in reverse to preserve indices)
        indices = [i for i, n in enumerate(self._ring_nodes) if n == node_id]
        for idx in reversed(indices):
            pos = self._ring_positions[idx]
            self._ring_positions.pop(idx)
            self._ring_nodes.pop(idx)
            if self._ring_positions:
                succ_idx = bisect.bisect_left(self._ring_positions, pos) % len(self._ring_positions)
                new_owner = self._ring_nodes[succ_idx]
                prev_idx = (succ_idx - 1) % len(self._ring_positions)
                arc_start = (self._ring_positions[prev_idx] + 1) % RING_SIZE
                transfers[(arc_start, pos)] = (node_id, new_owner)
        return transfers

    def get_node(self, key: str) -> str:
        """Return the primary node for the given key."""
        with self._lock:
            return self._get_node(key)

    def _get_node(self, key: str) -> str:
        if not self._ring_positions:
            raise ValueError("Empty ring")
        pos = _hash(key)
        idx = bisect.bisect(self._ring_positions, pos)
        if idx == len(self._ring_positions):
            idx = 0
        return self._ring_nodes[idx]

    def get_nodes(self, key: str) -> List[str]:
        """Return RF distinct physical nodes for the key (preference list)."""
        with self._lock:
            return self._get_nodes(key)

    def _get_nodes(self, key: str) -> List[str]:
        if len(self._nodes) < self.replication_factor:
            raise ValueError(
                f"Not enough nodes ({len(self._nodes)}) for replication factor {self.replication_factor}"
            )
        if not self._ring_positions:
            raise ValueError("Empty ring")
        pos = _hash(key)
        idx = bisect.bisect(self._ring_positions, pos)
        result = []
        seen = set()
        n = len(self._ring_positions)
        for i in range(n):
            node = self._ring_nodes[(idx + i) % n]
            if node not in seen:
                seen.add(node)
                result.append(node)
                if len(result) == self.replication_factor:
                    break
        return result

    def get_all_nodes(self) -> List[str]:
        """Return all physical node IDs."""
        return list(self._nodes.keys())

    def get_node_count(self) -> int:
        """Return the number of physical nodes."""
        return len(self._nodes)

    def get_ring_position(self, key: str) -> int:
        """Return the hash ring position for a key."""
        return _hash(key)

    def get_load_distribution(self) -> Dict[str, float]:
        """Return fraction of ring owned by each node. Values sum to 1.0."""
        if not self._ring_positions:
            return {}
        ownership: Dict[str, int] = {n: 0 for n in self._nodes}
        n = len(self._ring_positions)
        for i in range(n):
            # Arc from previous position to this position
            prev_pos = self._ring_positions[(i - 1) % n]
            cur_pos = self._ring_positions[i]
            if cur_pos > prev_pos:
                arc = cur_pos - prev_pos
            else:
                arc = RING_SIZE - prev_pos + cur_pos
            ownership[self._ring_nodes[i]] += arc
        total = sum(ownership.values())
        return {node: count / total for node, count in ownership.items()}

    def get_key_distribution(self, keys: List[str]) -> Dict[str, int]:
        """Return how many keys are assigned to each node."""
        dist: Dict[str, int] = {n: 0 for n in self._nodes}
        for key in keys:
            node = self.get_node(key)
            dist[node] += 1
        return dist

    def load_imbalance(self) -> float:
        """Return max_load / average_load. Perfect balance = 1.0."""
        if not self._nodes:
            return 1.0
        dist = self.get_load_distribution()
        if not dist:
            return 1.0
        avg = 1.0 / len(dist)
        max_load = max(dist.values())
        return max_load / avg

    def ring_info(self) -> str:
        """Return a human-readable summary of the ring."""
        lines = [
            f"ConsistentHashRing: {len(self._nodes)} nodes, "
            f"{len(self._ring_positions)} virtual nodes, RF={self.replication_factor}",
        ]
        if self._nodes:
            lines.append("Nodes:")
            dist = self.get_load_distribution()
            for node, weight in sorted(self._nodes.items()):
                vnodes = int(self.num_vnodes * weight)
                pct = dist.get(node, 0) * 100
                lines.append(f"  {node}: weight={weight}, vnodes={vnodes}, load={pct:.1f}%")
            lines.append(f"Load imbalance: {self.load_imbalance():.3f}")
        return "\n".join(lines)
