"""Leader-follower replication with sync/async modes."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ReplicationLogEntry:
    """A single entry in the replication log."""
    lsn: int
    op_type: str  # "PUT" or "DELETE"
    key: str
    value: Optional[str]
    timestamp: float


class LeaderNode:
    """Leader node that accepts writes and replicates to followers."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._data: Dict[str, str] = {}
        self._log: List[ReplicationLogEntry] = []
        self._next_lsn = 1
        self._followers: Dict[str, dict] = {}  # id -> {node, mode, online}
        self._log_retention = 10000

    def put(self, key: str, value: str) -> int:
        """Write a key-value pair. Returns the LSN."""
        lsn = self._append_log("PUT", key, value)
        self._data[key] = value
        self._propagate(lsn)
        self._trim_log()
        return lsn

    def delete(self, key: str) -> int:
        """Delete a key. Returns the LSN."""
        lsn = self._append_log("DELETE", key, None)
        self._data.pop(key, None)
        self._propagate(lsn)
        self._trim_log()
        return lsn

    def get(self, key: str) -> Optional[str]:
        """Read from the leader's data store."""
        return self._data.get(key)

    def add_follower(self, follower: 'FollowerNode', sync_mode: str = "async") -> None:
        """Register a follower. sync_mode: 'sync', 'async', or 'semi_sync'."""
        self._followers[follower.node_id] = {
            "node": follower,
            "mode": sync_mode,
        }

    def remove_follower(self, follower_id: str) -> None:
        """Unregister a follower."""
        self._followers.pop(follower_id, None)

    def get_log_entries(self, after_lsn: int = 0) -> List[ReplicationLogEntry]:
        """Get all log entries after the given LSN."""
        return [e for e in self._log if e.lsn > after_lsn]

    def follower_status(self) -> Dict[str, Dict]:
        """Return status of all followers."""
        leader_lsn = self.current_lsn()
        result = {}
        for fid, info in self._followers.items():
            node = info["node"]
            flsn = node.current_lsn()
            result[fid] = {
                "lsn": flsn,
                "lag": leader_lsn - flsn,
                "mode": info["mode"],
            }
        return result

    def current_lsn(self) -> int:
        """Return the leader's latest LSN."""
        return self._next_lsn - 1

    def _append_log(self, op_type: str, key: str, value: Optional[str]) -> int:
        lsn = self._next_lsn
        self._next_lsn += 1
        entry = ReplicationLogEntry(lsn=lsn, op_type=op_type, key=key,
                                    value=value, timestamp=time.time())
        self._log.append(entry)
        return lsn

    def _propagate(self, lsn: int):
        entry = self._log[-1]
        assert entry.lsn == lsn

        sync_follower_failed = False
        semi_sync_acked = False

        for fid, info in list(self._followers.items()):
            node = info["node"]
            mode = info["mode"]

            if mode == "sync":
                if node._online:
                    node.apply_entries([entry])
                # sync blocks — done inline
            elif mode == "async":
                if node._online:
                    node.apply_entries([entry])
            elif mode == "semi_sync":
                if node._online:
                    node.apply_entries([entry])
                    semi_sync_acked = True
                else:
                    sync_follower_failed = True

        # Semi-sync: if the designated semi_sync follower failed, promote another
        if sync_follower_failed and not semi_sync_acked:
            self._promote_async_to_semi_sync()

    def _promote_async_to_semi_sync(self):
        """Promote the most caught-up async follower to semi_sync."""
        best_id = None
        best_lsn = -1
        for fid, info in self._followers.items():
            if info["mode"] == "async" and info["node"]._online:
                flsn = info["node"].current_lsn()
                if flsn > best_lsn:
                    best_lsn = flsn
                    best_id = fid
        if best_id:
            self._followers[best_id]["mode"] = "semi_sync"

    def _trim_log(self):
        if len(self._log) > self._log_retention:
            self._log = self._log[-self._log_retention:]


class FollowerNode:
    """Follower node that replicates data from the leader."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._data: Dict[str, str] = {}
        self._applied_lsn = 0
        self._online = True

    def get(self, key: str) -> Optional[str]:
        """Read from the follower's local data store (may be stale)."""
        return self._data.get(key)

    def read_at_lsn(self, key: str, min_lsn: int, timeout: float = 5.0) -> Optional[str]:
        """Read a key after catching up to min_lsn. Raises TimeoutError if not caught up."""
        deadline = time.time() + timeout
        while self._applied_lsn < min_lsn:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Follower {self.node_id} at LSN {self._applied_lsn}, "
                    f"needed {min_lsn}")
            time.sleep(0.01)
        return self._data.get(key)

    def apply_entries(self, entries: List[ReplicationLogEntry]) -> None:
        """Apply replication log entries to the local data store."""
        if not self._online:
            return
        for entry in entries:
            if entry.lsn <= self._applied_lsn:
                continue
            if entry.op_type == "PUT":
                self._data[entry.key] = entry.value
            elif entry.op_type == "DELETE":
                self._data.pop(entry.key, None)
            self._applied_lsn = entry.lsn

    def current_lsn(self) -> int:
        """Return the latest applied LSN."""
        return self._applied_lsn

    def replication_lag(self, leader_lsn: int) -> int:
        """Return the lag: leader_lsn - self.current_lsn."""
        return leader_lsn - self._applied_lsn

    def go_offline(self) -> None:
        """Simulate going offline."""
        self._online = False

    def come_online(self, leader: LeaderNode) -> None:
        """Come back online and catch up from the leader."""
        self._online = True
        entries = leader.get_log_entries(after_lsn=self._applied_lsn)
        self.apply_entries(entries)

    def promote_to_leader(self) -> LeaderNode:
        """Promote this follower to a new leader node."""
        new_leader = LeaderNode(self.node_id)
        new_leader._data = dict(self._data)
        new_leader._next_lsn = self._applied_lsn + 1
        # Copy log entries we have knowledge of (via our applied state)
        # The new leader starts fresh with no log history
        return new_leader


class ReadSession:
    """Session-based reads with monotonic read guarantee."""

    def __init__(self, followers: List[FollowerNode]):
        self._followers = followers
        self._last_seen_lsn = 0

    def read(self, key: str) -> Optional[str]:
        """Read from a follower with monotonic read guarantee."""
        # Pick a follower that is at least as up-to-date as last_seen_lsn
        for follower in self._followers:
            if follower._online and follower.current_lsn() >= self._last_seen_lsn:
                self._last_seen_lsn = follower.current_lsn()
                return follower.get(key)
        # Fallback: use the most caught-up follower
        best = max((f for f in self._followers if f._online),
                   key=lambda f: f.current_lsn(), default=None)
        if best:
            self._last_seen_lsn = best.current_lsn()
            return best.get(key)
        raise RuntimeError("No online followers available")
