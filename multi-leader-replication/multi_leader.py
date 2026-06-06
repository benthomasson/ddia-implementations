"""Multi-leader replication with conflict resolution."""

from enum import Enum
from typing import Callable, Optional, Any
from dataclasses import dataclass

_TOMBSTONE = object()


class ConflictStrategy(Enum):
    LAST_WRITE_WINS = "lww"
    CUSTOM_MERGE = "custom"


class Topology(Enum):
    ALL_TO_ALL = "all_to_all"
    RING = "ring"


@dataclass
class ConflictRecord:
    key: str
    local_value: Any
    remote_value: Any
    local_timestamp: int
    remote_timestamp: int
    resolved_value: Any
    resolved_by: ConflictStrategy


class ReplicaNode:
    """A replica node in a multi-leader replication cluster."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._clock = 0
        # store[key] = (value, timestamp, origin_node_id, is_tombstone)
        self._store: dict[str, tuple] = {}
        self._pending: list[dict] = []
        self._conflict_log: list[ConflictRecord] = []
        # Track which (ts, node_id) pairs we've already seen per key to avoid dupes
        self._seen: dict[str, set[tuple[int, str]]] = {}

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _record_seen(self, key: str, ts: int, origin: str):
        if key not in self._seen:
            self._seen[key] = set()
        self._seen[key].add((ts, origin))

    def put(self, key: str, value: Any) -> int:
        """Write a key-value pair. Returns the Lamport timestamp."""
        ts = self._tick()
        self._store[key] = (value, ts, self.node_id, False)
        self._record_seen(key, ts, self.node_id)
        self._pending.append({
            "key": key,
            "value": value,
            "timestamp": ts,
            "node_id": self.node_id,
            "is_tombstone": False,
        })
        return ts

    def get(self, key: str) -> Optional[Any]:
        """Read a value by key. Returns None if not found or tombstoned."""
        entry = self._store.get(key)
        if entry is None or entry[3]:  # missing or tombstone
            return None
        return entry[0]

    def delete(self, key: str) -> int:
        """Delete a key by writing a tombstone. Returns the timestamp."""
        ts = self._tick()
        self._store[key] = (_TOMBSTONE, ts, self.node_id, True)
        self._record_seen(key, ts, self.node_id)
        self._pending.append({
            "key": key,
            "value": None,
            "timestamp": ts,
            "node_id": self.node_id,
            "is_tombstone": True,
        })
        return ts

    def get_pending_changes(self) -> list:
        """Return and clear the outgoing replication log."""
        changes = self._pending
        self._pending = []
        return changes

    def apply_remote_change(self, change: dict, strategy: ConflictStrategy,
                            merge_fn: Optional[Callable] = None) -> Optional[ConflictRecord]:
        """Apply a replicated change from another node. Returns ConflictRecord if conflict."""
        key = change["key"]
        remote_ts = change["timestamp"]
        remote_node = change["node_id"]
        remote_val = change["value"]
        is_tombstone = change.get("is_tombstone", False)

        # Idempotency: skip if we've already seen this exact (ts, node) for this key
        if key in self._seen and (remote_ts, remote_node) in self._seen[key]:
            return None

        # Update Lamport clock
        self._clock = max(self._clock, remote_ts) + 1
        self._record_seen(key, remote_ts, remote_node)

        local_entry = self._store.get(key)

        if local_entry is None:
            # No local value — just accept
            actual_val = _TOMBSTONE if is_tombstone else remote_val
            self._store[key] = (actual_val, remote_ts, remote_node, is_tombstone)
            # Queue for further propagation (ring topology)
            self._pending.append(change)
            return None

        local_val, local_ts, local_origin, local_is_tomb = local_entry

        # If local value came from the same origin with the same timestamp, no conflict
        if local_origin == remote_node and local_ts == remote_ts:
            return None

        # Check if this is truly a conflict: local was written by a different origin
        # than the remote change, meaning concurrent writes happened
        is_conflict = (local_origin != remote_node)

        if not is_conflict:
            # Same origin but different timestamp — this is an update, take the newer one
            if (remote_ts, remote_node) > (local_ts, local_origin):
                actual_val = _TOMBSTONE if is_tombstone else remote_val
                self._store[key] = (actual_val, remote_ts, remote_node, is_tombstone)
            self._pending.append(change)
            return None

        # Conflict detected — resolve it
        # Get display values for the conflict record
        display_local = None if local_is_tomb else local_val
        display_remote = None if is_tombstone else remote_val

        if strategy == ConflictStrategy.LAST_WRITE_WINS:
            if (remote_ts, remote_node) > (local_ts, local_origin):
                # Remote wins
                actual_val = _TOMBSTONE if is_tombstone else remote_val
                self._store[key] = (actual_val, remote_ts, remote_node, is_tombstone)
                resolved = display_remote
            else:
                # Local wins
                resolved = display_local
            record = ConflictRecord(
                key=key,
                local_value=display_local,
                remote_value=display_remote,
                local_timestamp=local_ts,
                remote_timestamp=remote_ts,
                resolved_value=resolved,
                resolved_by=strategy,
            )

        elif strategy == ConflictStrategy.CUSTOM_MERGE:
            merged = merge_fn(key, display_local, display_remote, local_ts, remote_ts)
            new_ts = max(local_ts, remote_ts) + 1
            canonical_origin = max(local_origin, remote_node)
            self._clock = max(self._clock, new_ts)
            self._store[key] = (merged, new_ts, canonical_origin, False)
            self._record_seen(key, new_ts, canonical_origin)
            self._pending.append({
                "key": key,
                "value": merged,
                "timestamp": new_ts,
                "node_id": canonical_origin,
                "is_tombstone": False,
            })
            record = ConflictRecord(
                key=key,
                local_value=display_local,
                remote_value=display_remote,
                local_timestamp=local_ts,
                remote_timestamp=remote_ts,
                resolved_value=merged,
                resolved_by=strategy,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self._conflict_log.append(record)
        if strategy != ConflictStrategy.CUSTOM_MERGE:
            self._pending.append(change)
        return record

    @property
    def conflict_log(self) -> list[ConflictRecord]:
        """All conflicts detected at this node."""
        return self._conflict_log


class MultiLeaderCluster:
    """Manages a set of ReplicaNodes with configurable replication."""

    def __init__(self, node_ids: list[str],
                 strategy: ConflictStrategy = ConflictStrategy.LAST_WRITE_WINS,
                 merge_fn: Optional[Callable] = None,
                 topology: Topology = Topology.ALL_TO_ALL):
        if strategy == ConflictStrategy.CUSTOM_MERGE and merge_fn is None:
            raise ValueError("CUSTOM_MERGE strategy requires a merge_fn")
        self._strategy = strategy
        self._merge_fn = merge_fn
        self._topology = topology
        self._nodes: dict[str, ReplicaNode] = {}
        self._node_order: list[str] = list(node_ids)
        for nid in node_ids:
            self._nodes[nid] = ReplicaNode(nid)

    def node(self, node_id: str) -> ReplicaNode:
        """Get a node by ID."""
        return self._nodes[node_id]

    def sync(self) -> int:
        """Perform one round of replication. Returns number of changes propagated."""
        count = 0
        # Collect pending changes from all nodes first
        pending_by_node: dict[str, list[dict]] = {}
        for nid in self._node_order:
            pending_by_node[nid] = self._nodes[nid].get_pending_changes()

        if self._topology == Topology.ALL_TO_ALL:
            for src_id in self._node_order:
                changes = pending_by_node[src_id]
                for change in changes:
                    for dst_id in self._node_order:
                        if dst_id == src_id:
                            continue
                        self._nodes[dst_id].apply_remote_change(
                            change, self._strategy, self._merge_fn)
                        count += 1

        elif self._topology == Topology.RING:
            for i, src_id in enumerate(self._node_order):
                dst_id = self._node_order[(i + 1) % len(self._node_order)]
                changes = pending_by_node[src_id]
                for change in changes:
                    self._nodes[dst_id].apply_remote_change(
                        change, self._strategy, self._merge_fn)
                    count += 1

        return count

    def all_converged(self) -> bool:
        """Check if all nodes have identical state."""
        if len(self._nodes) <= 1:
            return True
        nodes = list(self._nodes.values())
        ref_keys = set(nodes[0]._store.keys())
        for n in nodes[1:]:
            if set(n._store.keys()) != ref_keys:
                return False
        for key in ref_keys:
            ref_entry = nodes[0]._store[key]
            # Compare value, timestamp, and tombstone flag
            for n in nodes[1:]:
                entry = n._store[key]
                if (entry[0], entry[1], entry[3]) != (ref_entry[0], ref_entry[1], ref_entry[3]):
                    return False
        return True

    def sync_until_converged(self, max_rounds: int = 100) -> int:
        """Sync repeatedly until all nodes converge. Returns rounds taken."""
        for r in range(1, max_rounds + 1):
            self.sync()
            if self.all_converged():
                return r
        raise RuntimeError(f"Failed to converge after {max_rounds} rounds")
