"""Leaderless (Dynamo-style) replication with quorum reads/writes."""

from dataclasses import dataclass
from typing import Any, Optional


class QuorumNotMet(Exception):
    """Raised when insufficient replicas are available for a quorum."""
    pass


@dataclass
class VersionedValue:
    """A value with its version number and source node."""
    value: Any
    version: int
    node_id: str


@dataclass
class ReadResult:
    """Result of a quorum read."""
    value: Any
    version: int
    is_conflict: bool = False
    replicas_repaired: int = 0


class ReplicaNode:
    """A single replica node storing key-value pairs with versions."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._store: dict[str, VersionedValue] = {}
        self._available = True
        self._hints: list[tuple[str, Any, int, str]] = []  # (key, value, version, target_node_id)

    def write(self, key: str, value: Any, version: int) -> bool:
        """Store a value if version >= current. Returns True if accepted."""
        if not self._available:
            return False
        current = self._store.get(key)
        if current is None or version >= current.version:
            self._store[key] = VersionedValue(value=value, version=version, node_id=self.node_id)
            return True
        return False

    def read(self, key: str) -> Optional[VersionedValue]:
        """Read current value for a key. Returns None if unavailable or key missing."""
        if not self._available:
            return None
        return self._store.get(key)

    def set_available(self, available: bool):
        self._available = available

    @property
    def is_available(self) -> bool:
        return self._available

    def add_hint(self, key: str, value: Any, version: int, target_node_id: str):
        """Store a hinted handoff for a down node."""
        self._hints.append((key, value, version, target_node_id))

    def pop_hints(self) -> list[tuple[str, Any, int, str]]:
        """Return and clear all stored hints."""
        hints = self._hints
        self._hints = []
        return hints


class DynamoCluster:
    """Coordinator for a Dynamo-style leaderless replication cluster."""

    def __init__(self, num_replicas: int, write_quorum: int, read_quorum: int,
                 sloppy_quorum: bool = False):
        self.n = num_replicas
        self.w = write_quorum
        self.r = read_quorum
        self.sloppy_quorum = sloppy_quorum
        self.nodes: dict[str, ReplicaNode] = {}
        self._version_counters: dict[str, int] = {}  # global per-key version counter

        for i in range(num_replicas):
            node_id = f"node_{i}"
            self.nodes[node_id] = ReplicaNode(node_id)

    def put(self, key: str, value: Any) -> int:
        """Write a value to the cluster. Returns assigned version. Raises QuorumNotMet."""
        version = self._version_counters.get(key, 0) + 1
        self._version_counters[key] = version

        ack_count = 0

        for node in self.nodes.values():
            if node.is_available:
                if node.write(key, value, version):
                    ack_count += 1

        if ack_count < self.w:
            # Roll back version counter since write failed
            self._version_counters[key] = version - 1
            raise QuorumNotMet(
                f"Write quorum not met: got {ack_count} acks, need {self.w}"
            )

        # Store hints on available nodes for unavailable ones (only after successful write)
        if self.sloppy_quorum:
            unavailable_ids = [nid for nid, n in self.nodes.items() if not n.is_available]
            available_nodes = [n for n in self.nodes.values() if n.is_available]
            for target_id in unavailable_ids:
                if available_nodes:
                    available_nodes[0].add_hint(key, value, version, target_id)

        return version

    def get(self, key: str) -> ReadResult:
        """Read a value using quorum reads with read repair. Raises QuorumNotMet."""
        responses: list[VersionedValue] = []

        for node in self.nodes.values():
            if node.is_available:
                result = node.read(key)
                if result is not None:
                    responses.append(result)

        available_count = sum(1 for n in self.nodes.values() if n.is_available)
        if available_count < self.r:
            raise QuorumNotMet(
                f"Read quorum not met: {available_count} available, need {self.r}"
            )

        if not responses:
            return ReadResult(value=None, version=0)

        # Find highest version
        max_version = max(r.version for r in responses)

        # Check for conflicts: multiple distinct values at the max version
        max_version_values = [r for r in responses if r.version == max_version]
        distinct_values = []
        for r in max_version_values:
            if r.value not in distinct_values:
                distinct_values.append(r.value)

        is_conflict = len(distinct_values) > 1

        if is_conflict:
            value = distinct_values
        else:
            value = distinct_values[0]

        # Read repair: push latest value to stale replicas
        repaired = 0
        for node in self.nodes.values():
            if not node.is_available:
                continue
            current = node.read(key)
            if current is None or current.version < max_version:
                # Use the first max-version value for repair
                repair_value = max_version_values[0].value
                node.write(key, repair_value, max_version)
                repaired += 1

        return ReadResult(
            value=value,
            version=max_version,
            is_conflict=is_conflict,
            replicas_repaired=repaired,
        )

    def get_node(self, node_id: str) -> ReplicaNode:
        """Access a specific replica node."""
        return self.nodes[node_id]

    def set_node_available(self, node_id: str, available: bool):
        """Simulate a node going up or down."""
        self.nodes[node_id].set_available(available)

    def anti_entropy_repair(self) -> int:
        """Sync all replicas to highest version for each key. Returns repair count."""
        all_keys: set[str] = set()
        for node in self.nodes.values():
            if node.is_available:
                all_keys.update(node._store.keys())

        repairs = 0
        for key in all_keys:
            # Find max version across all available nodes
            best: Optional[VersionedValue] = None
            for node in self.nodes.values():
                if not node.is_available:
                    continue
                v = node._store.get(key)
                if v is not None and (best is None or v.version > best.version):
                    best = v

            if best is None:
                continue

            # Push to any node that's behind
            for node in self.nodes.values():
                if not node.is_available:
                    continue
                current = node._store.get(key)
                if current is None or current.version < best.version:
                    node.write(key, best.value, best.version)
                    repairs += 1

        return repairs

    def deliver_hints(self) -> int:
        """Deliver pending hinted handoffs. Returns count delivered."""
        delivered = 0
        for node in self.nodes.values():
            hints = node.pop_hints()
            for key, value, version, target_id in hints:
                target = self.nodes[target_id]
                if target.is_available:
                    if target.write(key, value, version):
                        delivered += 1
                else:
                    # Put hint back if target still unavailable
                    node.add_hint(key, value, version, target_id)
        return delivered
