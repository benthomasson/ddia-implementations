"""Raft consensus algorithm simulation."""

import random


class LogEntry:
    """A single entry in the replicated log."""

    def __init__(self, term: int, index: int, command=None):
        self.term = term
        self.index = index
        self.command = command


class RaftNode:
    """A single Raft node."""

    def __init__(self, node_id, peer_ids, election_timeout_range=(150, 300)):
        self._node_id = node_id
        self._peer_ids = list(peer_ids)
        self._election_timeout_range = election_timeout_range

        # Persistent state
        self._current_term = 0
        self._voted_for = None
        self._log = [LogEntry(term=0, index=0, command=None)]  # sentinel

        # Volatile state
        self._state = "follower"
        self._commit_index = 0
        self._last_applied = 0

        # Leader state
        self._next_index = {}
        self._match_index = {}

        # Timers
        self._election_timer = 0
        self._heartbeat_timer = 0
        self._heartbeat_interval = 50
        self._reset_election_timer()

        # Election
        self._votes_received = set()

    def _reset_election_timer(self):
        lo, hi = self._election_timeout_range
        self._election_timeout = random.randint(lo, hi)
        self._election_timer = 0

    @property
    def state(self):
        return self._state

    @property
    def current_term(self):
        return self._current_term

    @property
    def commit_index(self):
        return self._commit_index

    def get_log(self):
        return list(self._log)

    def _last_log_index(self):
        return self._log[-1].index

    def _last_log_term(self):
        return self._log[-1].term

    def _become_follower(self, term):
        self._state = "follower"
        self._current_term = term
        self._voted_for = None
        self._reset_election_timer()

    def _become_candidate(self):
        self._state = "candidate"
        self._current_term += 1
        self._voted_for = self._node_id
        self._votes_received = {self._node_id}
        self._reset_election_timer()

    def _become_leader(self):
        self._state = "leader"
        last = self._last_log_index() + 1
        for peer in self._peer_ids:
            self._next_index[peer] = last
            self._match_index[peer] = 0
        self._heartbeat_timer = self._heartbeat_interval  # send immediately

    def _is_log_up_to_date(self, last_log_term, last_log_index):
        my_term = self._last_log_term()
        my_index = self._last_log_index()
        if last_log_term != my_term:
            return last_log_term > my_term
        return last_log_index >= my_index

    def handle_request_vote(self, candidate_id, candidate_term, last_log_index, last_log_term):
        if candidate_term > self._current_term:
            self._become_follower(candidate_term)

        if candidate_term < self._current_term:
            return {"term": self._current_term, "vote_granted": False}

        can_vote = (self._voted_for is None or self._voted_for == candidate_id)
        log_ok = self._is_log_up_to_date(last_log_term, last_log_index)

        if can_vote and log_ok:
            self._voted_for = candidate_id
            self._reset_election_timer()
            return {"term": self._current_term, "vote_granted": True}

        return {"term": self._current_term, "vote_granted": False}

    def handle_append_entries(self, leader_id, leader_term, prev_log_index, prev_log_term, entries, leader_commit):
        if leader_term > self._current_term:
            self._become_follower(leader_term)
        elif leader_term == self._current_term and self._state == "candidate":
            self._become_follower(leader_term)

        if leader_term < self._current_term:
            return {"term": self._current_term, "success": False, "match_index": 0}

        # Valid leader heartbeat/append — reset election timer
        self._reset_election_timer()

        # Check log consistency
        if prev_log_index > self._last_log_index():
            return {"term": self._current_term, "success": False, "match_index": self._last_log_index()}

        if self._log[prev_log_index].term != prev_log_term:
            # Delete conflicting entry and everything after
            self._log = self._log[:prev_log_index]
            return {"term": self._current_term, "success": False, "match_index": self._last_log_index()}

        # Append new entries (overwrite conflicts)
        insert_idx = prev_log_index + 1
        for i, entry in enumerate(entries):
            log_pos = insert_idx + i
            if log_pos < len(self._log):
                if self._log[log_pos].term != entry.term:
                    self._log = self._log[:log_pos]
                    self._log.append(entry)
                # else: already have this entry, skip
            else:
                self._log.append(entry)

        # Update commit index
        if leader_commit > self._commit_index:
            self._commit_index = min(leader_commit, self._last_log_index())

        return {"term": self._current_term, "success": True, "match_index": self._last_log_index()}

    def client_request(self, command):
        if self._state != "leader":
            return {"success": False, "entry": None, "error": "not leader"}

        entry = LogEntry(term=self._current_term, index=self._last_log_index() + 1, command=command)
        self._log.append(entry)
        return {"success": True, "entry": entry, "error": None}

    def tick(self, elapsed_ms):
        messages = []

        if self._state == "leader":
            self._heartbeat_timer += elapsed_ms
            if self._heartbeat_timer >= self._heartbeat_interval:
                self._heartbeat_timer = 0
                for peer in self._peer_ids:
                    messages.append(self._make_append_entries(peer))
                self._advance_commit_index()
        else:
            self._election_timer += elapsed_ms
            if self._election_timer >= self._election_timeout:
                self._become_candidate()
                for peer in self._peer_ids:
                    messages.append({
                        "type": "request_vote",
                        "to": peer,
                        "from": self._node_id,
                        "term": self._current_term,
                        "last_log_index": self._last_log_index(),
                        "last_log_term": self._last_log_term(),
                    })

        return messages

    def _make_append_entries(self, peer):
        ni = self._next_index.get(peer, 1)
        prev_index = ni - 1
        prev_term = self._log[prev_index].term
        entries = self._log[ni:]
        return {
            "type": "append_entries",
            "to": peer,
            "from": self._node_id,
            "term": self._current_term,
            "prev_log_index": prev_index,
            "prev_log_term": prev_term,
            "entries": entries,
            "leader_commit": self._commit_index,
        }

    def _advance_commit_index(self):
        # Only commit entries from the current term by replica count
        for n in range(self._commit_index + 1, self._last_log_index() + 1):
            if self._log[n].term != self._current_term:
                continue
            count = 1  # leader itself
            for peer in self._peer_ids:
                if self._match_index.get(peer, 0) >= n:
                    count += 1
            total = len(self._peer_ids) + 1
            if count > total // 2:
                self._commit_index = n

    def handle_vote_response(self, from_id, term, vote_granted):
        if term > self._current_term:
            self._become_follower(term)
            return

        if self._state != "candidate" or term != self._current_term:
            return

        if vote_granted:
            self._votes_received.add(from_id)
            total = len(self._peer_ids) + 1
            if len(self._votes_received) > total // 2:
                self._become_leader()

    def handle_append_response(self, from_id, term, success, match_index):
        if term > self._current_term:
            self._become_follower(term)
            return

        if self._state != "leader":
            return

        if success:
            self._next_index[from_id] = match_index + 1
            self._match_index[from_id] = match_index
            self._advance_commit_index()
        else:
            self._next_index[from_id] = max(1, self._next_index.get(from_id, 1) - 1)

    def get_committed_entries(self):
        return [e for e in self._log[1:] if e.index <= self._commit_index]


class RaftCluster:
    """Simulated Raft cluster with message passing."""

    def __init__(self, node_ids, election_timeout_range=(150, 300)):
        self.nodes = {}
        for nid in node_ids:
            peers = [p for p in node_ids if p != nid]
            self.nodes[nid] = RaftNode(nid, peers, election_timeout_range)
        self._partitioned = set()

    def tick(self, elapsed_ms):
        all_messages = []
        for nid, node in self.nodes.items():
            if nid not in self._partitioned:
                msgs = node.tick(elapsed_ms)
                all_messages.extend(msgs)

        # Deliver messages
        for msg in all_messages:
            sender = msg["from"]
            receiver = msg["to"]
            if sender in self._partitioned or receiver in self._partitioned:
                continue
            node = self.nodes[receiver]
            if msg["type"] == "request_vote":
                resp = node.handle_request_vote(
                    sender, msg["term"], msg["last_log_index"], msg["last_log_term"]
                )
                # Deliver response back to sender
                if sender not in self._partitioned:
                    self.nodes[sender].handle_vote_response(receiver, resp["term"], resp["vote_granted"])
            elif msg["type"] == "append_entries":
                resp = node.handle_append_entries(
                    sender, msg["term"], msg["prev_log_index"], msg["prev_log_term"],
                    msg["entries"], msg["leader_commit"]
                )
                if sender not in self._partitioned:
                    self.nodes[sender].handle_append_response(receiver, resp["term"], resp["success"], resp["match_index"])

    def run_until_leader(self, max_ticks=1000):
        for _ in range(max_ticks):
            self.tick(10)
            leader = self.get_leader()
            if leader is not None:
                return leader
        return None

    def submit(self, command):
        leader_id = self.get_leader()
        if leader_id is None:
            return {"success": False, "leader": None, "entry": None}
        result = self.nodes[leader_id].client_request(command)
        return {"success": result["success"], "leader": leader_id, "entry": result["entry"]}

    def run_until_committed(self, index, max_ticks=1000):
        for _ in range(max_ticks):
            self.tick(10)
            count = 0
            total = len(self.nodes) - len(self._partitioned)
            for nid, node in self.nodes.items():
                if nid not in self._partitioned and node.commit_index >= index:
                    count += 1
            if count > total // 2:
                return True
        return False

    def partition_node(self, node_id):
        self._partitioned.add(node_id)

    def heal_node(self, node_id):
        self._partitioned.discard(node_id)

    def get_leader(self):
        leaders = [nid for nid, n in self.nodes.items()
                   if n.state == "leader" and nid not in self._partitioned]
        if len(leaders) == 1:
            return leaders[0]
        return None

    def get_committed_log(self):
        leader_id = self.get_leader()
        if leader_id is None:
            return []
        return [e.command for e in self.nodes[leader_id].get_committed_entries()]
