"""Hinted handoff for temporary failure handling in leaderless replication."""


class Hint:
    """A hinted write record for later handoff to an unavailable node."""

    def __init__(self, target_node_id: str, key: str, value, version: int,
                 created_at: int, ttl: int):
        self.target_node_id = target_node_id
        self.key = key
        self.value = value
        self.version = version
        self.created_at = created_at
        self.ttl = ttl

    def is_expired(self, current_time: int) -> bool:
        """Check if this hint has exceeded its TTL."""
        return current_time >= self.created_at + self.ttl


class Node:
    """A storage node with main store and hint log."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.store = {}  # key -> (value, version)
        self.hints = []  # list of Hint objects
        self._available = True

    def put(self, key: str, value, version: int) -> bool:
        """Store a key-value pair in the main store."""
        existing = self.store.get(key)
        if existing is None or version >= existing[1]:
            self.store[key] = (value, version)
        return True

    def get(self, key: str):
        """Retrieve a key-value pair from the main store."""
        return self.store.get(key)

    def store_hint(self, hint: Hint) -> None:
        """Store a hinted write for later handoff."""
        self.hints.append(hint)

    def get_hints_for(self, target_node_id: str) -> list:
        """Return all hints intended for a specific target node."""
        return [h for h in self.hints if h.target_node_id == target_node_id]

    def remove_hints_for(self, target_node_id: str) -> int:
        """Remove all hints for a target node after successful handoff."""
        before = len(self.hints)
        self.hints = [h for h in self.hints if h.target_node_id != target_node_id]
        return before - len(self.hints)

    def expire_hints(self, current_time: int) -> int:
        """Remove hints that have exceeded their TTL."""
        before = len(self.hints)
        self.hints = [h for h in self.hints if not h.is_expired(current_time)]
        return before - len(self.hints)

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available

    def hint_count(self) -> int:
        return len(self.hints)


class HintedHandoffStore:
    """Coordinator that routes requests and manages hinted handoff logic."""

    def __init__(self, node_ids: list, replication_factor: int = 3,
                 write_quorum: int = 2, read_quorum: int = 2,
                 hint_ttl: int = 100, sloppy_quorum: bool = True):
        self.sorted_node_ids = sorted(node_ids)
        self.nodes = {nid: Node(nid) for nid in node_ids}
        self.replication_factor = replication_factor
        self.write_quorum = write_quorum
        self.read_quorum = read_quorum
        self.hint_ttl = hint_ttl
        self.sloppy_quorum = sloppy_quorum
        self.versions = {}  # key -> current version

    def get_preferred_nodes(self, key: str) -> list:
        """Return preferred replica node IDs for a key using hash-mod-N ring."""
        n = len(self.sorted_node_ids)
        start = hash(key) % n
        return [self.sorted_node_ids[(start + i) % n]
                for i in range(self.replication_factor)]

    def put(self, key: str, value, current_time: int) -> dict:
        """Write a value with hinted handoff support."""
        version = self.versions.get(key, 0) + 1
        self.versions[key] = version

        preferred = self.get_preferred_nodes(key)
        replicas_written = []
        hints_stored = []
        unavailable_preferred = []

        # Try preferred replicas first
        for nid in preferred:
            node = self.nodes[nid]
            if node.is_available():
                node.put(key, value, version)
                replicas_written.append(nid)
            else:
                unavailable_preferred.append(nid)

        sloppy = False

        # If strict quorum and not enough preferred replicas, fail immediately
        if not self.sloppy_quorum and len(replicas_written) < self.write_quorum:
            return {
                'success': False,
                'replicas_written': replicas_written,
                'hints_stored': hints_stored,
                'version': version,
                'sloppy': False,
            }

        # Store hints for ALL unavailable preferred replicas on non-preferred nodes
        if unavailable_preferred:
            non_preferred = [nid for nid in self.sorted_node_ids
                             if nid not in preferred]
            for target_nid in unavailable_preferred:
                for nid in non_preferred:
                    node = self.nodes[nid]
                    if node.is_available():
                        hint = Hint(target_nid, key, value, version,
                                    current_time, self.hint_ttl)
                        node.store_hint(hint)
                        hints_stored.append({
                            'hint_node': nid,
                            'target_node': target_nid,
                        })
                        sloppy = True
                        break

        total_acks = len(replicas_written) + len(hints_stored)
        success = total_acks >= self.write_quorum

        return {
            'success': success,
            'replicas_written': replicas_written,
            'hints_stored': hints_stored,
            'version': version,
            'sloppy': sloppy,
        }

    def get(self, key: str) -> dict:
        """Read a value from available preferred replicas."""
        preferred = self.get_preferred_nodes(key)
        results = []
        replicas_read = []

        for nid in preferred:
            node = self.nodes[nid]
            if node.is_available():
                result = node.get(key)
                if result is not None:
                    results.append(result)
                    replicas_read.append(nid)

        if not results:
            return {'value': None, 'version': None, 'replicas_read': replicas_read}

        # Return highest version
        best = max(results, key=lambda r: r[1])
        return {
            'value': best[0],
            'version': best[1],
            'replicas_read': replicas_read,
        }

    def trigger_handoff(self, recovered_node_id: str, current_time: int) -> dict:
        """Trigger handoff to a recovered node."""
        recovered_node = self.nodes[recovered_node_id]
        hints_delivered = 0
        hints_expired = 0
        keys_recovered = set()

        for node in self.nodes.values():
            if node.node_id == recovered_node_id:
                continue
            hints = node.get_hints_for(recovered_node_id)
            for hint in hints:
                if hint.is_expired(current_time):
                    hints_expired += 1
                else:
                    recovered_node.put(hint.key, hint.value, hint.version)
                    hints_delivered += 1
                    keys_recovered.add(hint.key)
            node.remove_hints_for(recovered_node_id)

        return {
            'node': recovered_node_id,
            'hints_delivered': hints_delivered,
            'hints_expired': hints_expired,
            'keys_recovered': list(keys_recovered),
        }

    def set_node_available(self, node_id: str, available: bool) -> None:
        self.nodes[node_id].set_available(available)

    def expire_all_hints(self, current_time: int) -> int:
        """Run hint expiry across all nodes."""
        total = 0
        for node in self.nodes.values():
            total += node.expire_hints(current_time)
        return total

    def get_cluster_status(self) -> dict:
        """Return status of all nodes and hint counts."""
        return {
            nid: {
                'available': node.is_available(),
                'hint_count': node.hint_count(),
                'keys_stored': len(node.store),
            }
            for nid, node in self.nodes.items()
        }
