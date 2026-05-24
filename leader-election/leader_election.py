"""Bully Algorithm for leader election in distributed systems."""


class Message:
    """A message in the election protocol."""

    def __init__(self, msg_type: str, sender_id: int, receiver_id: int,
                 term: int, timestamp: int):
        self.msg_type = msg_type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.term = term
        self.timestamp = timestamp

    def __repr__(self):
        return (f"Message({self.msg_type}, {self.sender_id}->{self.receiver_id}, "
                f"term={self.term}, t={self.timestamp})")


class BullyNode:
    """A single node participating in Bully leader election."""

    def __init__(self, node_id: int, all_node_ids: list[int],
                 heartbeat_interval: int = 3, election_timeout: int = 10):
        self.node_id = node_id
        self.all_node_ids = sorted(all_node_ids)
        self.heartbeat_interval = heartbeat_interval
        self.election_timeout = election_timeout

        self._state = "follower"
        self._leader_id = None
        self._current_term = 0
        self._available = True
        self._last_heartbeat_time = 0
        self._last_heartbeat_sent = 0
        self._election_start_time = None
        self._got_alive = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def leader_id(self) -> int | None:
        return self._leader_id

    @property
    def current_term(self) -> int:
        return self._current_term

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available
        if not available:
            self._state = "follower"
            self._leader_id = None
            self._election_start_time = None

    def receive_message(self, message: Message) -> list[Message]:
        """Process an incoming message and return response messages."""
        if not self._available:
            return []

        responses = []

        if message.msg_type == "ELECTION":
            if self.node_id > message.sender_id:
                # Respond with ALIVE and start own election
                responses.append(Message(
                    "ALIVE", self.node_id, message.sender_id,
                    self._current_term, message.timestamp
                ))
                # Start our own election if not already in one
                if self._state != "candidate":
                    responses.extend(self.start_election(message.timestamp))
            # Ignore ELECTION from higher-ID nodes (shouldn't happen per protocol)

        elif message.msg_type == "ALIVE":
            # A higher-ID node is alive, back down
            self._got_alive = True
            self._state = "follower"
            self._leader_id = None
            self._election_start_time = None

        elif message.msg_type == "COORDINATOR":
            if message.sender_id > self.node_id:
                # Accept the new leader
                self._state = "follower"
                self._leader_id = message.sender_id
                self._current_term = message.term
                self._last_heartbeat_time = message.timestamp
                self._election_start_time = None
            elif message.sender_id < self.node_id:
                # We have higher ID, start election to take over
                responses.extend(self.start_election(message.timestamp))

        elif message.msg_type == "HEARTBEAT":
            if message.sender_id == self._leader_id:
                self._last_heartbeat_time = message.timestamp
            elif self._state != "leader":
                # Accept heartbeat from any node claiming leadership with higher ID
                if message.sender_id > self.node_id:
                    self._leader_id = message.sender_id
                    self._state = "follower"
                    self._last_heartbeat_time = message.timestamp
                    self._current_term = message.term

        return responses

    def tick(self, current_time: int) -> list[Message]:
        """Advance the node's timer."""
        if not self._available:
            return []

        messages = []

        if self._state == "leader":
            if current_time - self._last_heartbeat_sent >= self.heartbeat_interval:
                self._last_heartbeat_sent = current_time
                for nid in self.all_node_ids:
                    if nid != self.node_id:
                        messages.append(Message(
                            "HEARTBEAT", self.node_id, nid,
                            self._current_term, current_time
                        ))

        elif self._state == "follower":
            if current_time - self._last_heartbeat_time >= self.election_timeout:
                messages.extend(self.start_election(current_time))

        elif self._state == "candidate":
            if self._election_start_time is not None:
                # Wait for ALIVE responses; if timeout, declare victory
                if current_time - self._election_start_time >= self.election_timeout // 2:
                    if not self._got_alive:
                        messages.extend(self.declare_victory(current_time))
                    else:
                        # Got ALIVE, go back to follower and wait
                        self._state = "follower"
                        self._election_start_time = None
                        self._last_heartbeat_time = current_time

        return messages

    def start_election(self, current_time: int) -> list[Message]:
        """Initiate an election."""
        self._current_term += 1
        self._state = "candidate"
        self._leader_id = None
        self._election_start_time = current_time
        self._got_alive = False

        higher_nodes = [nid for nid in self.all_node_ids if nid > self.node_id]

        if not higher_nodes:
            # No higher nodes, declare victory immediately
            return self.declare_victory(current_time)

        messages = []
        for nid in higher_nodes:
            messages.append(Message(
                "ELECTION", self.node_id, nid,
                self._current_term, current_time
            ))
        return messages

    def declare_victory(self, current_time: int) -> list[Message]:
        """Declare self as leader and send COORDINATOR messages."""
        self._state = "leader"
        self._leader_id = self.node_id
        self._last_heartbeat_sent = current_time

        messages = []
        for nid in self.all_node_ids:
            if nid != self.node_id:
                messages.append(Message(
                    "COORDINATOR", self.node_id, nid,
                    self._current_term, current_time
                ))
        return messages


class BullyElectionCluster:
    """Simulation harness for Bully leader election."""

    def __init__(self, node_ids: list[int], heartbeat_interval: int = 3,
                 election_timeout: int = 10):
        self.nodes = {}
        for nid in node_ids:
            self.nodes[nid] = BullyNode(nid, node_ids, heartbeat_interval, election_timeout)
        self.election_history = []
        self._current_time = 0

    def tick(self, current_time: int) -> None:
        """Advance all nodes' timers and deliver messages."""
        self._current_time = current_time
        all_messages = []

        # Collect tick messages from all available nodes
        for node in self.nodes.values():
            msgs = node.tick(current_time)
            all_messages.extend(msgs)

        # Deliver messages and collect responses, iterate until no new messages
        while all_messages:
            next_messages = []
            for msg in all_messages:
                receiver = self.nodes.get(msg.receiver_id)
                if receiver and receiver.is_available():
                    responses = receiver.receive_message(msg)
                    next_messages.extend(responses)

                    # Track COORDINATOR messages for election history
                    if msg.msg_type == "COORDINATOR":
                        self._record_leader(msg.sender_id, msg.term, current_time)

            all_messages = next_messages

        # Split-brain detection and resolution
        self._resolve_split_brain(current_time)

    def _record_leader(self, leader_id: int, term: int, timestamp: int):
        """Record a leadership change if it's new."""
        if (not self.election_history or
                self.election_history[-1]["leader_id"] != leader_id or
                self.election_history[-1]["term"] != term):
            self.election_history.append({
                "term": term,
                "leader_id": leader_id,
                "timestamp": timestamp,
            })

    def _resolve_split_brain(self, current_time: int):
        """If multiple nodes think they're leader, lower-ID ones step down."""
        leaders = [nid for nid, node in self.nodes.items()
                   if node.is_available() and node.state == "leader"]
        if len(leaders) > 1:
            # Highest ID wins, others step down and start elections
            highest = max(leaders)
            for lid in leaders:
                if lid != highest:
                    node = self.nodes[lid]
                    msgs = node.start_election(current_time)
                    # Deliver these messages
                    for msg in msgs:
                        receiver = self.nodes.get(msg.receiver_id)
                        if receiver and receiver.is_available():
                            responses = receiver.receive_message(msg)
                            for resp in responses:
                                resp_receiver = self.nodes.get(resp.receiver_id)
                                if resp_receiver and resp_receiver.is_available():
                                    resp_receiver.receive_message(resp)

    def run_until_leader(self, start_time: int = 0, max_ticks: int = 100) -> int | None:
        """Run simulation until a leader is elected."""
        for t in range(start_time, start_time + max_ticks):
            self.tick(t)
            leader = self.get_leader()
            if leader is not None:
                # Check that all available nodes agree
                agreed = all(
                    node.leader_id == leader
                    for node in self.nodes.values()
                    if node.is_available()
                )
                if agreed:
                    return leader
        return None

    def get_leader(self) -> int | None:
        """Return the current leader's ID."""
        leaders = [nid for nid, node in self.nodes.items()
                   if node.is_available() and node.state == "leader"]
        if len(leaders) == 1:
            return leaders[0]
        if len(leaders) > 1:
            return max(leaders)  # Bully: highest wins
        # No node claims leader state; check follower consensus
        leader_votes = {}
        for node in self.nodes.values():
            if node.is_available() and node.leader_id is not None:
                lid = node.leader_id
                leader_votes[lid] = leader_votes.get(lid, 0) + 1
        if leader_votes:
            return max(leader_votes, key=leader_votes.get)
        return None

    def fail_node(self, node_id: int) -> None:
        """Simulate a node crash."""
        self.nodes[node_id].set_available(False)

    def recover_node(self, node_id: int) -> None:
        """Recover a failed node (triggers election)."""
        node = self.nodes[node_id]
        node.set_available(True)
        node._last_heartbeat_time = self._current_time
        # Recovered node starts an election immediately
        msgs = node.start_election(self._current_time)
        # Deliver these messages
        for msg in msgs:
            receiver = self.nodes.get(msg.receiver_id)
            if receiver and receiver.is_available():
                responses = receiver.receive_message(msg)
                for resp in responses:
                    resp_receiver = self.nodes.get(resp.receiver_id)
                    if resp_receiver and resp_receiver.is_available():
                        resp_receiver.receive_message(resp)

    def get_cluster_state(self) -> dict:
        """Return the state of all nodes."""
        return {
            nid: {
                "state": node.state,
                "leader_id": node.leader_id,
                "term": node.current_term,
                "available": node.is_available(),
            }
            for nid, node in self.nodes.items()
        }

    def get_election_history(self) -> list[dict]:
        """Return history of leader changes."""
        return list(self.election_history)
