"""Snapshot Isolation (MVCC) database implementation."""


class TransactionError(Exception):
    """Raised for invalid transaction operations."""
    pass


class Version:
    """A single version of a key-value pair."""

    def __init__(self, key, value, created_by, deleted_by=None):
        self.key = key
        self.value = value
        self.created_by = created_by
        self.deleted_by = deleted_by


class Transaction:
    """Represents a database transaction."""

    def __init__(self, tx_id, start_timestamp, read_only=False):
        self._tx_id = tx_id
        self.start_timestamp = start_timestamp
        self.read_only = read_only
        self._status = "active"
        self.write_set = set()  # keys written by this tx
        # Set of tx IDs that were active when this tx started
        self.active_at_start = set()

    @property
    def tx_id(self):
        return self._tx_id

    @property
    def is_active(self):
        return self._status == "active"

    @property
    def status(self):
        return self._status


class MVCCDatabase:
    """MVCC database with snapshot isolation."""

    def __init__(self):
        self._next_tx_id = 1
        self._next_timestamp = 1
        self._versions = {}  # key -> list[Version]
        self._transactions = {}  # tx_id -> Transaction
        self._committed = set()  # committed tx IDs
        self._aborted = set()  # aborted tx IDs
        # Map tx_id -> commit_timestamp for committed transactions
        self._commit_timestamps = {}

    def begin_transaction(self, read_only=False):
        """Start a new transaction."""
        tx_id = self._next_tx_id
        self._next_tx_id += 1
        ts = self._next_timestamp
        self._next_timestamp += 1

        tx = Transaction(tx_id, ts, read_only)
        # Snapshot: record which transactions are currently active
        tx.active_at_start = {
            tid for tid, t in self._transactions.items() if t.is_active
        }
        self._transactions[tx_id] = tx
        return tx

    def _is_visible(self, tx, version):
        """Check if a version is visible to transaction tx."""
        created_by = version.created_by

        # Aborted transactions' versions are never visible
        if created_by in self._aborted:
            return False

        # Check if the creating tx is visible
        created_visible = False
        if created_by == tx.tx_id:
            # Own writes are visible
            created_visible = True
        elif created_by in self._committed:
            # Committed before we started? Check it wasn't active at our start
            if created_by not in tx.active_at_start:
                # And its tx_id must be less than ours (started before us)
                if created_by < tx.tx_id:
                    created_visible = True

        if not created_visible:
            return False

        # Check deletion
        deleted_by = version.deleted_by
        if deleted_by is None:
            return True

        # If deleted by self, not visible
        if deleted_by == tx.tx_id:
            return False

        # If deleted by an aborted tx, deletion doesn't count
        if deleted_by in self._aborted:
            return True

        # If deleted by a committed tx that was not active at start and started before us
        if deleted_by in self._committed and deleted_by not in tx.active_at_start and deleted_by < tx.tx_id:
            return False  # deletion is visible, so version is not visible

        # Otherwise deletion not yet visible
        return True

    def _check_active(self, tx):
        if not tx.is_active:
            raise TransactionError(f"Transaction {tx.tx_id} is not active (status: {tx.status})")

    def _check_writable(self, tx):
        self._check_active(tx)
        if tx.read_only:
            raise TransactionError(f"Transaction {tx.tx_id} is read-only")

    def read(self, tx, key):
        """Read a key within a transaction's snapshot."""
        self._check_active(tx)

        if key not in self._versions:
            return None

        # Find the latest visible version
        visible = None
        for v in self._versions[key]:
            if self._is_visible(tx, v):
                visible = v

        return visible.value if visible is not None else None

    def write(self, tx, key, value):
        """Write a key within a transaction."""
        self._check_writable(tx)

        # If this tx already wrote to this key, update that version in place
        if key in self._versions:
            for v in self._versions[key]:
                if v.created_by == tx.tx_id and v.deleted_by is None:
                    v.value = value
                    return

        # Create new version (don't modify old versions — read returns last visible)
        new_version = Version(key, value, tx.tx_id)
        if key not in self._versions:
            self._versions[key] = []
        self._versions[key].append(new_version)
        tx.write_set.add(key)

    def delete(self, tx, key):
        """Delete a key within a transaction."""
        self._check_writable(tx)

        if key not in self._versions:
            return False

        # Mark ALL visible versions as deleted by this tx
        found = False
        for v in self._versions[key]:
            if self._is_visible(tx, v):
                v.deleted_by = tx.tx_id
                found = True

        if not found:
            return False

        tx.write_set.add(key)
        return True

    def commit(self, tx):
        """Attempt to commit a transaction. Returns True if successful."""
        self._check_active(tx)

        if tx.read_only:
            tx._status = "committed"
            self._committed.add(tx.tx_id)
            return True

        # Check for write-write conflicts (first-committer-wins)
        for key in tx.write_set:
            if key in self._versions:
                for v in self._versions[key]:
                    # Another tx created this version
                    if v.created_by != tx.tx_id and v.created_by not in self._aborted:
                        # Was it committed after we started?
                        if v.created_by in self._committed and (
                            v.created_by in tx.active_at_start or v.created_by >= tx.tx_id
                        ):
                            # Conflict: this tx committed after our start
                            self.abort(tx)
                            return False

        # Commit
        tx._status = "committed"
        self._committed.add(tx.tx_id)
        commit_ts = self._next_timestamp
        self._next_timestamp += 1
        self._commit_timestamps[tx.tx_id] = commit_ts
        return True

    def abort(self, tx):
        """Abort a transaction."""
        self._check_active(tx)
        tx._status = "aborted"
        self._aborted.add(tx.tx_id)

        # Clean up: unmark deletions made by this tx
        for key in tx.write_set:
            if key in self._versions:
                for v in self._versions[key]:
                    if v.deleted_by == tx.tx_id:
                        v.deleted_by = None

    def scan(self, tx, prefix=""):
        """Scan all keys visible to this transaction, optionally filtered by prefix."""
        self._check_active(tx)
        result = {}
        for key in self._versions:
            if key.startswith(prefix):
                val = self.read(tx, key)
                if val is not None:
                    result[key] = val
        return result

    def garbage_collect(self):
        """Remove versions no longer visible to any active transaction."""
        active_txs = [t for t in self._transactions.values() if t.is_active]
        removed = 0

        for key in list(self._versions.keys()):
            versions = self._versions[key]

            # Step 1: remove versions from aborted transactions
            versions = [v for v in versions if v.created_by not in self._aborted]
            removed += len(self._versions[key]) - len(versions)

            if not versions:
                del self._versions[key]
                continue

            if not active_txs:
                # No active transactions: keep only the latest committed version
                # Find latest committed non-deleted version
                latest = None
                for v in versions:
                    if v.created_by in self._committed and (
                        v.deleted_by is None or v.deleted_by not in self._committed
                    ):
                        latest = v

                if latest is not None:
                    before = len(versions)
                    versions = [latest]
                    removed += before - 1
                else:
                    # All versions are deleted — remove them all
                    removed += len(versions)
                    versions = []
            else:
                # With active transactions: remove versions that no active tx
                # could ever need. A version is safe to remove if it's been
                # superseded by a newer committed version that ALL active txs
                # can see.
                min_tx_id = min(t.tx_id for t in active_txs)

                # Find the newest committed version visible to the oldest active tx
                # (i.e., committed before the oldest active tx started)
                committed_before_oldest = [
                    v for v in versions
                    if v.created_by in self._committed
                    and v.created_by < min_tx_id
                ]

                # Versions superseded before the oldest active tx can be removed
                to_keep = []
                for v in versions:
                    if (v.deleted_by is not None
                            and v.deleted_by in self._committed
                            and v.deleted_by < min_tx_id):
                        removed += 1
                    elif (v.deleted_by is None
                          and v.created_by in self._committed
                          and v.created_by < min_tx_id
                          and any(v2.created_by > v.created_by
                                  and v2.created_by in self._committed
                                  and v2.created_by < min_tx_id
                                  for v2 in committed_before_oldest)):
                        # Zombie: superseded by a newer committed version
                        # that all active txs can see
                        removed += 1
                    else:
                        to_keep.append(v)
                versions = to_keep

            if versions:
                self._versions[key] = versions
            else:
                if key in self._versions:
                    del self._versions[key]

        return removed

    def get_version_count(self, key):
        """Return the number of stored versions for a key."""
        return len(self._versions.get(key, []))

    def get_active_transactions(self):
        """Return list of active transaction IDs."""
        return [t.tx_id for t in self._transactions.values() if t.is_active]
