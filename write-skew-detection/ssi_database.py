"""SSI (Serializable Snapshot Isolation) database with write skew detection."""


class SSITransaction:
    """A transaction tracked by SSI with read set, write set, and predicate locks."""

    def __init__(self, tx_id, start_timestamp):
        self.tx_id = tx_id
        self.start_timestamp = start_timestamp
        self._read_set = set()
        self._write_set = set()
        self._writes = {}  # key -> value (buffered writes)
        self._deletes = set()
        self._predicate_locks = []  # list of (predicate_fn, snapshot_result_dict)
        self._status = "active"
        self.commit_timestamp = None

    @property
    def read_set(self):
        return set(self._read_set)

    @property
    def write_set(self):
        return set(self._write_set)

    @property
    def predicate_locks(self):
        return [p[0] for p in self._predicate_locks]

    @property
    def status(self):
        return self._status


class SSIDatabase:
    """MVCC database with SSI write skew detection."""

    def __init__(self, pessimistic=False):
        # MVCC store: key -> list of (commit_timestamp, value, tx_id)
        # value is _DELETED sentinel for deletions
        self._store = {}
        self._next_tx_id = 1
        self._next_timestamp = 1
        self._committed_txns = []  # list of committed SSITransaction objects
        self._constraints = {}  # name -> check_fn
        self._dependency_graph = {}  # tx_id -> set of tx_ids
        self._pessimistic = pessimistic

    def begin_transaction(self):
        """Start a new SSI transaction."""
        tx_id = self._next_tx_id
        self._next_tx_id += 1
        ts = self._next_timestamp
        self._next_timestamp += 1
        tx = SSITransaction(tx_id, ts)
        self._dependency_graph.setdefault(tx_id, set())
        return tx

    def _visible_value(self, key, snapshot_ts):
        """Get the value visible at snapshot_ts for a key. Returns (value, writer_tx_id) or (None, None)."""
        if key not in self._store:
            return None, None
        versions = self._store[key]
        # Find the latest version with commit_ts <= snapshot_ts
        best = None
        for commit_ts, value, tx_id in versions:
            if commit_ts <= snapshot_ts and (best is None or commit_ts > best[0]):
                best = (commit_ts, value, tx_id)
        if best is None:
            return None, None
        if best[1] is _DELETED:
            return None, best[2]
        return best[1], best[2]

    def _snapshot(self, snapshot_ts):
        """Build a full snapshot dict at the given timestamp."""
        result = {}
        for key in self._store:
            val, _ = self._visible_value(key, snapshot_ts)
            if val is not None:
                result[key] = val
        return result

    def read(self, tx, key):
        """Read a key, adding it to the transaction's read set."""
        if tx._status != "active":
            raise RuntimeError(f"Transaction {tx.tx_id} is {tx._status}")
        tx._read_set.add(key)

        # Check buffered writes first
        if key in tx._deletes:
            val = None
        elif key in tx._writes:
            val = tx._writes[key]
        else:
            val, writer_tx_id = self._visible_value(key, tx.start_timestamp)
            # Track causal dependency
            if writer_tx_id is not None:
                self._dependency_graph.setdefault(tx.tx_id, set()).add(writer_tx_id)

        if self._pessimistic:
            self._check_read_conflict(tx, key)

        return val

    def read_predicate(self, tx, predicate):
        """Read all key-value pairs matching a predicate, storing it for phantom detection."""
        if tx._status != "active":
            raise RuntimeError(f"Transaction {tx.tx_id} is {tx._status}")

        # Build snapshot and evaluate predicate
        snap = self._snapshot(tx.start_timestamp)
        # Also include buffered writes/deletes
        merged = dict(snap)
        for k, v in tx._writes.items():
            merged[k] = v
        for k in tx._deletes:
            merged.pop(k, None)

        result = {}
        for k, v in merged.items():
            try:
                if predicate(k, v):
                    result[k] = v
                    tx._read_set.add(k)
            except Exception:
                pass

        # Track causal dependencies for keys read from committed data
        for k in result:
            if k in snap:
                _, writer_tx_id = self._visible_value(k, tx.start_timestamp)
                if writer_tx_id is not None:
                    self._dependency_graph.setdefault(tx.tx_id, set()).add(writer_tx_id)

        # Store predicate + original result for phantom detection
        tx._predicate_locks.append((predicate, dict(result)))

        if self._pessimistic:
            for k in result:
                self._check_read_conflict(tx, k)

        return result

    def write(self, tx, key, value):
        """Write a key, adding it to the transaction's write set."""
        if tx._status != "active":
            raise RuntimeError(f"Transaction {tx.tx_id} is {tx._status}")
        tx._write_set.add(key)
        tx._writes[key] = value
        tx._deletes.discard(key)

    def delete(self, tx, key):
        """Delete a key, adding it to the transaction's write set."""
        if tx._status != "active":
            raise RuntimeError(f"Transaction {tx.tx_id} is {tx._status}")
        tx._write_set.add(key)
        tx._deletes.add(key)
        tx._writes.pop(key, None)
        return True

    def commit(self, tx):
        """Attempt to commit with SSI validation."""
        if tx._status != "active":
            return {"committed": False, "reason": f"Transaction already {tx._status}", "conflicts": []}

        conflicts = []

        # Read-only optimization: skip conflict checks
        if not tx._write_set:
            tx._status = "committed"
            tx.commit_timestamp = self._next_timestamp
            self._next_timestamp += 1
            self._committed_txns.append(tx)
            return {"committed": True, "reason": None, "conflicts": []}

        # Find concurrent committed transactions (committed after our start)
        concurrent = [
            ctxn for ctxn in self._committed_txns
            if ctxn.commit_timestamp > tx.start_timestamp
        ]

        # 1. Write-write conflict detection
        for ctxn in concurrent:
            overlap = tx._write_set & ctxn._write_set
            if overlap:
                for k in overlap:
                    conflicts.append({
                        "type": "write-write",
                        "key": k,
                        "conflicting_tx": ctxn.tx_id
                    })

        # 2. Read-write conflict detection (write skew)
        for ctxn in concurrent:
            # Keys we read that were written by a concurrent txn
            rw_overlap = tx._read_set & ctxn._write_set
            if rw_overlap:
                for k in rw_overlap:
                    conflicts.append({
                        "type": "read-write",
                        "key": k,
                        "conflicting_tx": ctxn.tx_id
                    })

        # 3. Phantom detection: re-evaluate predicates against current state
        if tx._predicate_locks:
            current_snap = self._snapshot(self._next_timestamp)
            # Apply our own writes to see if predicates match differently
            # But we compare against the COMMITTED state (without our writes)
            for pred_fn, original_result in tx._predicate_locks:
                current_result = {}
                for k, v in current_snap.items():
                    try:
                        if pred_fn(k, v):
                            current_result[k] = v
                    except Exception:
                        pass
                # Compare keys: if the set of matching keys changed, phantom detected
                if set(current_result.keys()) != set(original_result.keys()):
                    new_keys = set(current_result.keys()) - set(original_result.keys())
                    removed_keys = set(original_result.keys()) - set(current_result.keys())
                    for k in new_keys:
                        conflicts.append({
                            "type": "phantom",
                            "key": k,
                            "detail": "new key matches predicate"
                        })
                    for k in removed_keys:
                        conflicts.append({
                            "type": "phantom",
                            "key": k,
                            "detail": "key no longer matches predicate"
                        })
                else:
                    # Same keys but values might have changed
                    for k in current_result:
                        if current_result[k] != original_result.get(k):
                            conflicts.append({
                                "type": "phantom",
                                "key": k,
                                "detail": "value changed for predicate match"
                            })

        # 4. Constraint validation
        if not conflicts and self._constraints:
            # Build hypothetical snapshot with our writes applied
            hypo_snap = self._snapshot(self._next_timestamp)
            for k, v in tx._writes.items():
                hypo_snap[k] = v
            for k in tx._deletes:
                hypo_snap.pop(k, None)
            for name, check_fn in self._constraints.items():
                if not check_fn(hypo_snap):
                    conflicts.append({
                        "type": "constraint",
                        "constraint": name,
                        "detail": f"Constraint '{name}' violated"
                    })

        if conflicts:
            tx._status = "aborted"
            return {
                "committed": False,
                "reason": self._conflict_reason(conflicts),
                "conflicts": conflicts
            }

        # Commit: assign timestamp and install writes
        tx.commit_timestamp = self._next_timestamp
        self._next_timestamp += 1
        tx._status = "committed"

        for key, value in tx._writes.items():
            self._store.setdefault(key, []).append(
                (tx.commit_timestamp, value, tx.tx_id)
            )
        for key in tx._deletes:
            self._store.setdefault(key, []).append(
                (tx.commit_timestamp, _DELETED, tx.tx_id)
            )

        self._committed_txns.append(tx)
        return {"committed": True, "reason": None, "conflicts": []}

    def _conflict_reason(self, conflicts):
        types = {c["type"] for c in conflicts}
        if "write-write" in types:
            return "Write-write conflict detected"
        if "phantom" in types:
            return "Phantom detected: write skew via predicate conflict"
        if "read-write" in types:
            return "Write skew: read-write conflict detected"
        if "constraint" in types:
            return "Constraint violation: " + conflicts[0].get("detail", "")
        return "Conflict detected"

    def abort(self, tx):
        """Abort a transaction."""
        tx._status = "aborted"

    def add_constraint(self, name, check):
        """Register a database-wide constraint."""
        self._constraints[name] = check

    def get_dependency_graph(self):
        """Return the transaction dependency graph."""
        return {tx_id: set(deps) for tx_id, deps in self._dependency_graph.items()}

    def _check_read_conflict(self, tx, key):
        """Pessimistic check: abort immediately if a concurrent txn wrote this key."""
        for ctxn in self._committed_txns:
            if ctxn.commit_timestamp > tx.start_timestamp and key in ctxn._write_set:
                tx._status = "aborted"
                raise RuntimeError(
                    f"Pessimistic abort: key '{key}' was modified by tx {ctxn.tx_id}"
                )


class _DeletedSentinel:
    """Sentinel value representing a deleted key in MVCC store."""
    def __repr__(self):
        return "<DELETED>"

_DELETED = _DeletedSentinel()
