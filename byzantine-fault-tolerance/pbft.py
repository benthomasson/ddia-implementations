"""Simplified PBFT consensus simulation."""

import hashlib
import json


class MessageType:
    REQUEST = "request"
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"
    NEW_VIEW = "new_view"
    REPLY = "reply"


class ByzantineMode:
    HONEST = "honest"
    SILENT = "silent"
    EQUIVOCATING = "equivocating"
    WRONG_SEQUENCE = "wrong_sequence"
    WRONG_DIGEST = "wrong_digest"


class Message:
    """A PBFT protocol message."""

    def __init__(self, msg_type: str, view: int, sequence: int,
                 digest: str, sender: int, data: dict | None = None,
                 recipient: int | None = None):
        self.msg_type = msg_type
        self.view = view
        self.sequence = sequence
        self.digest = digest
        self.sender = sender
        self.data = data or {}
        self.recipient = recipient  # None means broadcast

    def __repr__(self):
        return (f"Message({self.msg_type}, v={self.view}, s={self.sequence}, "
                f"from={self.sender}, to={self.recipient})")


def compute_digest(request) -> str:
    """Compute SHA-256 digest of a request."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()


class PBFTNode:
    """A single PBFT node."""

    TIMEOUT_MS = 3000

    def __init__(self, node_id: int, total_nodes: int, f: int,
                 byzantine_mode: str = ByzantineMode.HONEST):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = f
        self.byzantine_mode = byzantine_mode

        self.current_view = 0
        self.next_sequence = 1
        self.message_log: dict[tuple[int, int], dict[str, list[Message]]] = {}
        self._executed_log: list[tuple[int, any]] = []
        self.prepared_requests: set[tuple[int, int]] = set()
        self.committed_requests: set[tuple[int, int]] = set()
        self.next_execute_seq = 1

        # View change state
        self.view_change_msgs: dict[int, list[Message]] = {}
        self.timer_ms = 0
        self.waiting_for_request = False

        # Track accepted pre-prepares to prevent conflicts
        self.accepted_preprepare: dict[tuple[int, int]] = {}  # (view, seq) -> digest

        # Track senders per phase to reject duplicates
        self.phase_senders: dict[tuple[int, int, str], set[int]] = {}  # (view, seq, phase) -> {sender_ids}

    @property
    def view(self) -> int:
        return self.current_view

    @property
    def is_primary(self) -> bool:
        return self.current_view % self.total_nodes == self.node_id

    @property
    def executed_log(self) -> list[tuple[int, any]]:
        return list(self._executed_log)

    def _get_log(self, view: int, seq: int) -> dict[str, list[Message]]:
        key = (view, seq)
        if key not in self.message_log:
            self.message_log[key] = {
                MessageType.PRE_PREPARE: [],
                MessageType.PREPARE: [],
                MessageType.COMMIT: [],
            }
        return self.message_log[key]

    def _is_duplicate(self, msg: Message) -> bool:
        key = (msg.view, msg.sequence, msg.msg_type)
        if key not in self.phase_senders:
            self.phase_senders[key] = set()
        if msg.sender in self.phase_senders[key]:
            return True
        self.phase_senders[key].add(msg.sender)
        return False

    def _apply_byzantine(self, messages: list[Message]) -> list[Message]:
        """Apply Byzantine behavior to outgoing messages."""
        if self.byzantine_mode == ByzantineMode.HONEST:
            return messages
        if self.byzantine_mode == ByzantineMode.SILENT:
            return []
        if self.byzantine_mode == ByzantineMode.WRONG_SEQUENCE:
            for m in messages:
                m.sequence = m.sequence + 1000
            return messages
        if self.byzantine_mode == ByzantineMode.WRONG_DIGEST:
            for m in messages:
                m.digest = "bad_digest_" + str(self.node_id)
            return messages
        if self.byzantine_mode == ByzantineMode.EQUIVOCATING:
            result = []
            for m in messages:
                if m.recipient is not None:
                    result.append(m)
                else:
                    for peer in range(self.total_nodes):
                        if peer != self.node_id:
                            equivocated = Message(
                                m.msg_type, m.view, m.sequence,
                                compute_digest(f"equivoc_{peer}_{m.sequence}"),
                                m.sender, dict(m.data), recipient=peer
                            )
                            result.append(equivocated)
            return result
        return messages

    def receive_message(self, message: Message) -> list[Message]:
        """Process an incoming message and return messages to send."""
        # Validate sender
        if message.sender < 0 or message.sender >= self.total_nodes:
            return []

        if message.msg_type == MessageType.VIEW_CHANGE:
            return self._handle_view_change(message)
        if message.msg_type == MessageType.NEW_VIEW:
            return self._handle_new_view(message)

        # Reject messages from wrong view (except view change related)
        if message.view != self.current_view:
            return []

        if message.msg_type == MessageType.PRE_PREPARE:
            return self._apply_byzantine(self._handle_pre_prepare(message))
        elif message.msg_type == MessageType.PREPARE:
            return self._apply_byzantine(self._handle_prepare(message))
        elif message.msg_type == MessageType.COMMIT:
            return self._apply_byzantine(self._handle_commit(message))
        return []

    def _handle_pre_prepare(self, msg: Message) -> list[Message]:
        # Only accept from the current primary
        primary_id = self.current_view % self.total_nodes
        if msg.sender != primary_id:
            return []

        # Check for duplicate
        if self._is_duplicate(msg):
            return []

        # Check digest matches request
        request = msg.data.get("request")
        expected_digest = compute_digest(request)
        if msg.digest != expected_digest:
            return []

        # Check for conflicting pre-prepare
        key = (msg.view, msg.sequence)
        if key in self.accepted_preprepare:
            if self.accepted_preprepare[key] != msg.digest:
                return []
        self.accepted_preprepare[key] = msg.digest

        # Log the pre-prepare
        log = self._get_log(msg.view, msg.sequence)
        log[MessageType.PRE_PREPARE].append(msg)

        self.timer_ms = 0
        self.waiting_for_request = False

        # If we are the primary, we don't send PREPARE (we sent PRE-PREPARE)
        if self.is_primary:
            return self._check_prepared(msg.view, msg.sequence, msg.digest)

        # Send PREPARE
        prepare = Message(
            MessageType.PREPARE, msg.view, msg.sequence,
            msg.digest, self.node_id, {"request": request}
        )
        # Also log our own prepare
        self._get_log(msg.view, msg.sequence)[MessageType.PREPARE].append(prepare)
        sender_key = (msg.view, msg.sequence, MessageType.PREPARE)
        if sender_key not in self.phase_senders:
            self.phase_senders[sender_key] = set()
        self.phase_senders[sender_key].add(self.node_id)

        result = [prepare]
        result.extend(self._check_prepared(msg.view, msg.sequence, msg.digest))
        return result

    def _handle_prepare(self, msg: Message) -> list[Message]:
        if msg.sender == msg.view % self.total_nodes:
            return []
        if self._is_duplicate(msg):
            return []

        # Verify digest matches accepted pre-prepare
        key = (msg.view, msg.sequence)
        if key in self.accepted_preprepare:
            if self.accepted_preprepare[key] != msg.digest:
                return []

        log = self._get_log(msg.view, msg.sequence)
        log[MessageType.PREPARE].append(msg)

        return self._check_prepared(msg.view, msg.sequence, msg.digest)

    def _check_prepared(self, view: int, seq: int, digest: str) -> list[Message]:
        key = (view, seq)
        if key in self.prepared_requests:
            return []

        log = self._get_log(view, seq)

        # Need PRE-PREPARE
        if not log[MessageType.PRE_PREPARE]:
            return []

        # Count matching PREPARE messages from distinct senders
        prepare_senders = set()
        for p in log[MessageType.PREPARE]:
            if p.digest == digest:
                prepare_senders.add(p.sender)

        # Need 2f matching PREPAREs
        if len(prepare_senders) < 2 * self.f:
            return []

        self.prepared_requests.add(key)

        # Broadcast COMMIT
        commit = Message(
            MessageType.COMMIT, view, seq, digest, self.node_id
        )
        # Log our own commit
        self._get_log(view, seq)[MessageType.COMMIT].append(commit)
        sender_key = (view, seq, MessageType.COMMIT)
        if sender_key not in self.phase_senders:
            self.phase_senders[sender_key] = set()
        self.phase_senders[sender_key].add(self.node_id)

        result = [commit]
        result.extend(self._check_committed(view, seq, digest))
        return result

    def _handle_commit(self, msg: Message) -> list[Message]:
        if self._is_duplicate(msg):
            return []

        log = self._get_log(msg.view, msg.sequence)
        log[MessageType.COMMIT].append(msg)

        return self._check_committed(msg.view, msg.sequence, msg.digest)

    def _check_committed(self, view: int, seq: int, digest: str) -> list[Message]:
        key = (view, seq)
        if key in self.committed_requests:
            return []

        log = self._get_log(view, seq)

        # Count matching COMMIT messages from distinct senders
        commit_senders = set()
        for c in log[MessageType.COMMIT]:
            if c.digest == digest:
                commit_senders.add(c.sender)

        # Need 2f+1 matching COMMITs
        if len(commit_senders) < 2 * self.f + 1:
            return []

        self.committed_requests.add(key)

        # Try to execute in order
        return self._try_execute()

    def _try_execute(self) -> list[Message]:
        """Execute committed requests in sequence order."""
        results = []
        while True:
            # Find the committed request for next_execute_seq
            found = False
            for (v, s) in self.committed_requests:
                if s == self.next_execute_seq:
                    log = self._get_log(v, s)
                    if log[MessageType.PRE_PREPARE]:
                        pp = log[MessageType.PRE_PREPARE][0]
                        request = pp.data.get("request")
                        self._executed_log.append((s, request))
                        self.next_execute_seq += 1
                        # Send REPLY
                        reply = Message(
                            MessageType.REPLY, v, s,
                            pp.digest, self.node_id,
                            {"request": request, "result": f"executed:{request}"}
                        )
                        results.append(reply)
                        found = True
                        break
            if not found:
                break
        return results

    def submit_request(self, request) -> list[Message]:
        """Submit a client request to this node."""
        if not self.is_primary:
            return []

        if self.byzantine_mode == ByzantineMode.SILENT:
            return []

        digest = compute_digest(request)
        seq = self.next_sequence
        self.next_sequence += 1

        pp = Message(
            MessageType.PRE_PREPARE, self.current_view, seq,
            digest, self.node_id, {"request": request}
        )

        # Log our own pre-prepare
        log = self._get_log(self.current_view, seq)
        log[MessageType.PRE_PREPARE].append(pp)
        self.accepted_preprepare[(self.current_view, seq)] = digest
        sender_key = (self.current_view, seq, MessageType.PRE_PREPARE)
        if sender_key not in self.phase_senders:
            self.phase_senders[sender_key] = set()
        self.phase_senders[sender_key].add(self.node_id)

        result = self._apply_byzantine([pp])
        return result

    def tick(self, elapsed_ms: int) -> list[Message]:
        """Advance timer. Trigger view change on timeout."""
        if self.is_primary:
            return []

        self.timer_ms += elapsed_ms
        if self.timer_ms >= self.TIMEOUT_MS:
            self.timer_ms = 0
            return self._apply_byzantine(self._initiate_view_change())
        return []

    def _collect_prepared_data(self) -> list[dict]:
        """Collect prepared-but-not-committed requests for view change messages."""
        prepared_data = []
        for (v, s) in self.prepared_requests:
            if (v, s) in self.committed_requests:
                continue
            log = self._get_log(v, s)
            if log[MessageType.PRE_PREPARE]:
                pp = log[MessageType.PRE_PREPARE][0]
                prepared_data.append({
                    "view": v, "sequence": s,
                    "digest": pp.digest, "request": pp.data.get("request")
                })
        return prepared_data

    def _initiate_view_change(self) -> list[Message]:
        new_view = self.current_view + 1

        vc = Message(
            MessageType.VIEW_CHANGE, new_view, 0,
            "", self.node_id,
            {"prepared": self._collect_prepared_data()}
        )

        # Store our own VC so the new primary counts it
        if new_view not in self.view_change_msgs:
            self.view_change_msgs[new_view] = []
        self.view_change_msgs[new_view].append(vc)

        return [vc]

    def _handle_view_change(self, msg: Message) -> list[Message]:
        if msg.view <= self.current_view:
            return []

        target_view = msg.view
        if target_view not in self.view_change_msgs:
            self.view_change_msgs[target_view] = []

        # Check for duplicate
        for existing in self.view_change_msgs[target_view]:
            if existing.sender == msg.sender:
                return []
        self.view_change_msgs[target_view].append(msg)

        # Check if we're the new primary and have enough VC messages
        new_primary = target_view % self.total_nodes
        if self.node_id != new_primary:
            # If we haven't sent our own VC for this view, do so
            sent_own = any(m.sender == self.node_id for m in self.view_change_msgs[target_view])
            if not sent_own:
                vc = Message(
                    MessageType.VIEW_CHANGE, target_view, 0,
                    "", self.node_id,
                    {"prepared": self._collect_prepared_data()}
                )
                self.view_change_msgs[target_view].append(vc)
                return self._apply_byzantine([vc])
            return []

        if len(self.view_change_msgs[target_view]) < 2 * self.f + 1:
            return []

        # We are the new primary with enough VC messages - send NEW-VIEW
        self.current_view = target_view
        self.timer_ms = 0

        # Collect all prepared requests from VC messages
        repropose = {}
        for vc in self.view_change_msgs[target_view]:
            for p in vc.data.get("prepared", []):
                s = p["sequence"]
                if s not in repropose:
                    repropose[s] = p

        nv = Message(
            MessageType.NEW_VIEW, target_view, 0,
            "", self.node_id,
            {"view_changes": len(self.view_change_msgs[target_view]),
             "repropose": list(repropose.values())}
        )

        results = self._apply_byzantine([nv])

        # Re-propose prepared but uncommitted requests
        for p in repropose.values():
            seq = p["sequence"]
            request = p["request"]
            digest = compute_digest(request)
            if seq >= self.next_sequence:
                self.next_sequence = seq + 1
            pp = Message(
                MessageType.PRE_PREPARE, target_view, seq,
                digest, self.node_id, {"request": request}
            )
            log = self._get_log(target_view, seq)
            log[MessageType.PRE_PREPARE].append(pp)
            self.accepted_preprepare[(target_view, seq)] = digest
            results.extend(self._apply_byzantine([pp]))

        return results

    def _handle_new_view(self, msg: Message) -> list[Message]:
        new_primary = msg.view % self.total_nodes
        if msg.sender != new_primary:
            return []
        if msg.view <= self.current_view:
            return []

        self.current_view = msg.view
        self.timer_ms = 0
        self.waiting_for_request = False
        return []

    def get_state(self) -> dict:
        """Return node state summary."""
        return {
            "node_id": self.node_id,
            "view": self.current_view,
            "is_primary": self.is_primary,
            "byzantine_mode": self.byzantine_mode,
            "executed": [r for _, r in self._executed_log],
            "prepared": list(self.prepared_requests),
            "committed": list(self.committed_requests),
            "next_sequence": self.next_sequence,
        }


class PBFTCluster:
    """A cluster of PBFT nodes with a message bus."""

    def __init__(self, n: int, f: int, byzantine_nodes: dict[int, str] | None = None):
        if n != 3 * f + 1:
            raise ValueError(f"N must equal 3f+1. Got n={n}, f={f}")
        byzantine_nodes = byzantine_nodes or {}
        if len(byzantine_nodes) > f:
            raise ValueError(f"At most {f} Byzantine nodes allowed, got {len(byzantine_nodes)}")

        self.n = n
        self.f = f
        self.nodes: list[PBFTNode] = []
        self.pending_messages: list[Message] = []

        for i in range(n):
            mode = byzantine_nodes.get(i, ByzantineMode.HONEST)
            self.nodes.append(PBFTNode(i, n, f, mode))

    def submit_request(self, request) -> bool:
        """Submit a request and run protocol to completion."""
        # Use an honest node's view to find the current primary
        current_view = 0
        for node in self.nodes:
            if node.byzantine_mode == ByzantineMode.HONEST:
                current_view = node.current_view
                break
        primary_id = current_view % self.n
        primary = self.nodes[primary_id]
        msgs = primary.submit_request(request)
        self.pending_messages.extend(msgs)
        self.run_protocol()

        # Check if honest nodes executed it
        for node in self.nodes:
            if node.byzantine_mode == ByzantineMode.HONEST:
                if any(r == request for _, r in node._executed_log):
                    return True
        return False

    def run_protocol(self, max_rounds: int = 100) -> None:
        """Deliver messages in rounds until quiescent."""
        for _ in range(max_rounds):
            if not self.pending_messages:
                break
            current_batch = self.pending_messages
            self.pending_messages = []

            for msg in current_batch:
                if msg.recipient is not None:
                    # Targeted message
                    target = self.nodes[msg.recipient]
                    responses = target.receive_message(msg)
                    self.pending_messages.extend(responses)
                else:
                    # Broadcast to all except sender
                    for node in self.nodes:
                        if node.node_id != msg.sender:
                            responses = node.receive_message(msg)
                            self.pending_messages.extend(responses)

    def get_executed_log(self) -> list:
        """Return executed log from honest nodes."""
        for node in self.nodes:
            if node.byzantine_mode == ByzantineMode.HONEST:
                return [r for _, r in node._executed_log]
        return []

    def verify_agreement(self) -> bool:
        """Check that all honest nodes have the same executed log."""
        honest_logs = []
        for node in self.nodes:
            if node.byzantine_mode == ByzantineMode.HONEST:
                honest_logs.append([r for _, r in node._executed_log])

        if not honest_logs:
            return True
        return all(log == honest_logs[0] for log in honest_logs)

    def trigger_view_change(self) -> int:
        """Force a view change by ticking all non-primary nodes."""
        for node in self.nodes:
            msgs = node.tick(PBFTNode.TIMEOUT_MS + 1000)
            self.pending_messages.extend(msgs)
        self.run_protocol()
        # Return the new view from any honest node
        for node in self.nodes:
            if node.byzantine_mode == ByzantineMode.HONEST:
                return node.current_view
        return 0

    def get_node(self, node_id: int) -> PBFTNode:
        return self.nodes[node_id]
