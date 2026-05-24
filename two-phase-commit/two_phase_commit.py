"""Two-Phase Commit (2PC) protocol implementation."""

import itertools


class Participant:
    """A participant node in the 2PC protocol."""

    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.store = {}
        self.log = []
        self.locks = {}  # key -> tx_id
        self._available = True
        self._pending = {}  # tx_id -> list of operations

    def prepare(self, tx_id: str, operations: list[dict]) -> dict:
        """Handle PREPARE request. Returns vote dict."""
        if not self._available:
            return {"vote": "no", "reason": "participant unavailable"}

        # Check for lock conflicts
        for op in operations:
            key = op["key"]
            if key in self.locks and self.locks[key] != tx_id:
                self.log.append({"tx_id": tx_id, "state": "aborted", "reason": "lock conflict"})
                return {"vote": "no", "reason": f"key '{key}' locked by {self.locks[key]}"}

        # Acquire locks and store pending ops
        for op in operations:
            self.locks[op["key"]] = tx_id
        self._pending[tx_id] = operations
        self.log.append({"tx_id": tx_id, "state": "prepared", "operations": operations})
        return {"vote": "yes", "reason": None}

    def commit(self, tx_id: str) -> dict:
        """Handle COMMIT decision. Apply the transaction."""
        if not self._available:
            return {"success": False, "applied": []}

        operations = self._pending.pop(tx_id, [])
        applied = []
        for op in operations:
            if op["op"] == "set":
                self.store[op["key"]] = op["value"]
                applied.append(op)
            elif op["op"] == "delete":
                self.store.pop(op["key"], None)
                applied.append(op)
            # Release lock
            self.locks.pop(op["key"], None)

        self.log.append({"tx_id": tx_id, "state": "committed"})
        return {"success": True, "applied": applied}

    def abort(self, tx_id: str) -> dict:
        """Handle ABORT decision. Discard the transaction."""
        if not self._available:
            return {"success": False}

        operations = self._pending.pop(tx_id, [])
        for op in operations:
            if self.locks.get(op["key"]) == tx_id:
                del self.locks[op["key"]]

        self.log.append({"tx_id": tx_id, "state": "aborted"})
        return {"success": True}

    def get(self, key: str):
        """Read a value from the local store."""
        return self.store.get(key)

    def get_transaction_state(self, tx_id: str):
        """Return the latest state of a transaction from the log."""
        for entry in reversed(self.log):
            if entry["tx_id"] == tx_id:
                return entry["state"]
        return None

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available

    def recover(self) -> list[str]:
        """Return list of in-doubt tx_ids (prepared but no commit/abort)."""
        tx_states = {}
        for entry in self.log:
            tx_states[entry["tx_id"]] = entry["state"]
        return [tx_id for tx_id, state in tx_states.items() if state == "prepared"]


class Coordinator:
    """The 2PC coordinator."""

    def __init__(self, participant_ids: list[str], timeout: int = 5):
        self.participants = {pid: Participant(pid) for pid in participant_ids}
        self.timeout = timeout
        self.log = []
        self._available = True
        self._tx_counter = itertools.count(1)

    def begin_transaction(self) -> str:
        """Start a new transaction and return its ID."""
        tx_id = f"tx-{next(self._tx_counter):03d}"
        self.log.append({"tx_id": tx_id, "state": "initiated"})
        return tx_id

    def execute_transaction(self, tx_id: str, participant_operations: dict[str, list[dict]]) -> dict:
        """Execute a distributed 2PC transaction."""
        if not self._available:
            return {"tx_id": tx_id, "outcome": "aborted", "votes": {}, "reason": "coordinator unavailable"}

        votes = {}
        abort_reason = None

        # Phase 1: Prepare
        self.log.append({"tx_id": tx_id, "state": "preparing", "participants": list(participant_operations.keys())})

        for pid, ops in participant_operations.items():
            participant = self.participants[pid]
            if not participant.is_available():
                votes[pid] = "no"
                abort_reason = f"participant {pid} unavailable (timeout)"
                continue
            result = participant.prepare(tx_id, ops)
            votes[pid] = result["vote"]
            if result["vote"] == "no":
                abort_reason = result["reason"]

        self.log.append({"tx_id": tx_id, "state": "prepared", "votes": votes})

        # Phase 2: Commit or Abort
        all_yes = all(v == "yes" for v in votes.values())

        if all_yes:
            self.log.append({"tx_id": tx_id, "state": "committing"})
            for pid in participant_operations:
                p = self.participants[pid]
                if p.is_available():
                    p.commit(tx_id)
            self.log.append({"tx_id": tx_id, "state": "committed"})
            return {"tx_id": tx_id, "outcome": "committed", "votes": votes, "reason": None}
        else:
            self.log.append({"tx_id": tx_id, "state": "aborting"})
            for pid in participant_operations:
                p = self.participants[pid]
                if p.is_available():
                    p.abort(tx_id)
            self.log.append({"tx_id": tx_id, "state": "aborted"})
            return {"tx_id": tx_id, "outcome": "aborted", "votes": votes, "reason": abort_reason}

    def get_transaction_state(self, tx_id: str) -> str:
        """Query the coordinator's log for a transaction's latest state."""
        for entry in reversed(self.log):
            if entry["tx_id"] == tx_id:
                return entry["state"]
        return None

    def recover(self) -> dict:
        """Recover after coordinator crash. Re-send decisions for incomplete transactions."""
        tx_states = {}
        tx_participants = {}
        for entry in self.log:
            tx_states[entry["tx_id"]] = entry["state"]
            if "participants" in entry:
                tx_participants[entry["tx_id"]] = entry["participants"]

        recovered = []
        decisions_resent = 0

        for tx_id, state in tx_states.items():
            pids = tx_participants.get(tx_id, [])
            if state == "committing":
                for pid in pids:
                    p = self.participants[pid]
                    if p.is_available() and p.get_transaction_state(tx_id) == "prepared":
                        p.commit(tx_id)
                        decisions_resent += 1
                self.log.append({"tx_id": tx_id, "state": "committed"})
                recovered.append(tx_id)
            elif state == "aborting":
                for pid in pids:
                    p = self.participants[pid]
                    if p.is_available() and p.get_transaction_state(tx_id) == "prepared":
                        p.abort(tx_id)
                        decisions_resent += 1
                self.log.append({"tx_id": tx_id, "state": "aborted"})
                recovered.append(tx_id)

        return {"recovered_transactions": recovered, "decisions_resent": decisions_resent}

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available


class TwoPhaseCommitSystem:
    """Wrapper that creates a coordinator and participants for 2PC."""

    def __init__(self, participant_ids: list[str], timeout: int = 5):
        self._coordinator = Coordinator(participant_ids, timeout)

    @property
    def coordinator(self) -> Coordinator:
        return self._coordinator

    @property
    def participants(self) -> dict[str, Participant]:
        return self._coordinator.participants

    def execute(self, operations: dict[str, list[dict]]) -> dict:
        """Execute a distributed transaction."""
        tx_id = self._coordinator.begin_transaction()
        return self._coordinator.execute_transaction(tx_id, operations)

    def get_all_states(self) -> dict:
        """Return the state of all participants' stores."""
        return {pid: dict(p.store) for pid, p in self.participants.items()}
