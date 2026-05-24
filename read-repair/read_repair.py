"""Read repair for eventual consistency in a leaderless replication system."""

import warnings


class InsufficientReplicasError(Exception):
    """Raised when not enough replicas are available for a quorum operation."""
    pass


class Replica:
    """A single replica node with a versioned key-value store."""

    def __init__(self, replica_id: int):
        self.replica_id = replica_id
        self._store = {}  # key -> (value, version)
        self._available = True

    def put(self, key: str, value, version: int) -> bool:
        """Store a key-value pair if version >= current version."""
        if key in self._store:
            _, current_version = self._store[key]
            if version < current_version:
                return False
        self._store[key] = (value, version)
        return True

    def get(self, key: str):
        """Retrieve (value, version) for a key, or None if not found."""
        if key in self._store:
            return self._store[key]
        return None

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available

    def keys(self) -> set:
        return set(self._store.keys())


class ReadRepairStore:
    """Leaderless store with quorum reads/writes and read repair."""

    def __init__(self, num_replicas: int = 3, read_quorum: int = 2, write_quorum: int = 2):
        if read_quorum + write_quorum <= num_replicas:
            warnings.warn(
                f"Quorum condition not met: R({read_quorum}) + W({write_quorum}) <= N({num_replicas}). "
                "Strong consistency is not guaranteed."
            )
        self.num_replicas = num_replicas
        self.read_quorum = read_quorum
        self.write_quorum = write_quorum
        self.replicas = [Replica(i) for i in range(num_replicas)]
        self._total_reads = 0
        self._reads_triggering_repair = 0
        self._total_replicas_repaired = 0

    def _available_replicas(self):
        return [r for r in self.replicas if r.is_available()]

    def put(self, key: str, value) -> dict:
        """Write a value to W replicas."""
        available = self._available_replicas()
        if len(available) < self.write_quorum:
            raise InsufficientReplicasError(
                f"Need {self.write_quorum} replicas for write, only {len(available)} available"
            )

        # Determine next version by reading max across ALL replicas
        max_version = 0
        for r in self.replicas:
            result = r.get(key)
            if result is not None:
                _, v = result
                max_version = max(max_version, v)
        new_version = max_version + 1

        # Write to first W available replicas
        targets = available[:self.write_quorum]
        written = []
        for r in targets:
            r.put(key, value, new_version)
            written.append(r.replica_id)

        return {"success": True, "replicas_written": written, "version": new_version}

    def get(self, key: str) -> dict:
        """Read from R replicas with read repair."""
        available = self._available_replicas()
        if len(available) < self.read_quorum:
            raise InsufficientReplicasError(
                f"Need {self.read_quorum} replicas for read, only {len(available)} available"
            )

        self._total_reads += 1
        targets = available[:self.read_quorum]

        # Gather responses
        responses = []
        for r in targets:
            result = r.get(key)
            responses.append((r.replica_id, result))

        # Find the newest version
        best_value = None
        best_version = 0
        for rid, result in responses:
            if result is not None:
                val, ver = result
                if ver > best_version:
                    best_version = ver
                    best_value = val

        # If key doesn't exist anywhere
        if best_version == 0:
            return {
                "value": None,
                "version": 0,
                "replicas_read": [rid for rid, _ in responses],
                "repairs_triggered": [],
                "consistent": True,
            }

        # Read repair: push newest value to stale replicas among the R queried
        stale = []
        for rid, result in responses:
            if result is None or result[1] < best_version:
                stale.append(rid)

        if stale:
            self._reads_triggering_repair += 1
            for rid in stale:
                self.replicas[rid].put(key, best_value, best_version)
                self._total_replicas_repaired += 1

        return {
            "value": best_value,
            "version": best_version,
            "replicas_read": [rid for rid, _ in responses],
            "repairs_triggered": stale,
            "consistent": len(stale) == 0,
        }

    def anti_entropy_repair(self, key: str) -> dict:
        """Sync all available replicas to the latest version for a key."""
        available = self._available_replicas()

        # Find max version across all available replicas
        best_value = None
        best_version = 0
        for r in available:
            result = r.get(key)
            if result is not None:
                val, ver = result
                if ver > best_version:
                    best_version = ver
                    best_value = val

        # Repair stale replicas
        repaired = []
        for r in available:
            result = r.get(key)
            if result is None or result[1] < best_version:
                r.put(key, best_value, best_version)
                repaired.append(r.replica_id)

        return {"key": key, "replicas_repaired": repaired, "final_version": best_version}

    def get_repair_stats(self) -> dict:
        rate = (self._reads_triggering_repair / self._total_reads) if self._total_reads > 0 else 0.0
        return {
            "total_reads": self._total_reads,
            "reads_triggering_repair": self._reads_triggering_repair,
            "total_replicas_repaired": self._total_replicas_repaired,
            "repair_rate": rate,
        }

    def set_replica_available(self, replica_id: int, available: bool) -> None:
        self.replicas[replica_id].set_available(available)

    def get_replica_states(self, key: str) -> list:
        """Return the state of a key across all replicas."""
        states = []
        for r in self.replicas:
            result = r.get(key)
            states.append({
                "replica_id": r.replica_id,
                "value": result[0] if result else None,
                "version": result[1] if result else 0,
                "available": r.is_available(),
            })
        return states
