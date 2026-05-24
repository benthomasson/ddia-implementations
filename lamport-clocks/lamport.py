"""Lamport logical clocks for event ordering in distributed systems."""

from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque
import itertools

_msg_counter = itertools.count(1)


@dataclass
class Event:
    """A single event in the distributed system."""
    node_id: str
    event_type: str  # "LOCAL", "SEND", "RECEIVE"
    timestamp: int
    description: str
    message_id: Optional[str] = None
    _parent: Optional['Event'] = field(default=None, repr=False)
    _cause: Optional['Event'] = field(default=None, repr=False)


@dataclass
class Message:
    """A message passed between nodes."""
    sender_id: str
    sender_timestamp: int
    message_id: str
    payload: str


class LamportClock:
    """Lamport logical clock counter."""

    def __init__(self):
        self._counter = 0

    def tick(self) -> int:
        """Increment and return new timestamp."""
        self._counter += 1
        return self._counter

    def send_tick(self) -> int:
        """Increment and return timestamp for send events."""
        self._counter += 1
        return self._counter

    def receive_tick(self, received_timestamp: int) -> int:
        """Update clock on message receipt. Returns new timestamp."""
        self._counter = max(self._counter, received_timestamp) + 1
        return self._counter

    @property
    def current_time(self) -> int:
        return self._counter


class Node:
    """A node in the distributed system with a Lamport clock."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._clock = LamportClock()
        self._event_log: List[Event] = []

    def local_event(self, description: str) -> Event:
        """Record a local event."""
        ts = self._clock.tick()
        parent = self._event_log[-1] if self._event_log else None
        event = Event(self.node_id, "LOCAL", ts, description, _parent=parent)
        self._event_log.append(event)
        return event

    def send_message(self, to_node: 'Node', payload: str) -> Event:
        """Send a message to another node. Delivers synchronously."""
        ts = self._clock.send_tick()
        msg_id = f"msg-{next(_msg_counter)}"
        parent = self._event_log[-1] if self._event_log else None
        send_event = Event(self.node_id, "SEND", ts,
                           f"send to {to_node.node_id}: {payload}",
                           message_id=msg_id, _parent=parent)
        self._event_log.append(send_event)
        message = Message(self.node_id, ts, msg_id, payload)
        to_node.receive_message(message, _send_event=send_event)
        return send_event

    def receive_message(self, message: Message, _send_event: Optional[Event] = None) -> Event:
        """Process a received message."""
        ts = self._clock.receive_tick(message.sender_timestamp)
        parent = self._event_log[-1] if self._event_log else None
        event = Event(self.node_id, "RECEIVE", ts,
                      f"receive from {message.sender_id}: {message.payload}",
                      message_id=message.message_id,
                      _parent=parent, _cause=_send_event)
        self._event_log.append(event)
        return event

    def get_event_log(self) -> List[Event]:
        return list(self._event_log)

    @property
    def clock(self) -> LamportClock:
        return self._clock


class LamportMutex:
    """Distributed mutual exclusion using Lamport's algorithm."""

    def __init__(self, nodes: List[Node]):
        self._nodes = {n.node_id: n for n in nodes}
        self._request_queue: List[tuple] = []  # (timestamp, node_id)
        self._acks: dict = {}  # node_id -> set of node_ids that have acked

    def request(self, node: Node) -> None:
        """Node requests the lock. Broadcasts REQUEST to all others."""
        ts = node.clock.tick()
        parent = node._event_log[-1] if node._event_log else None
        req_event = Event(node.node_id, "LOCAL", ts,
                          "mutex request", _parent=parent)
        node._event_log.append(req_event)

        self._request_queue.append((ts, node.node_id))
        self._request_queue.sort()
        self._acks[node.node_id] = set()

        for nid, other in self._nodes.items():
            if nid != node.node_id:
                msg_id = f"msg-{next(_msg_counter)}"
                send_ts = node.clock.send_tick()
                parent = node._event_log[-1] if node._event_log else None
                send_evt = Event(node.node_id, "SEND", send_ts,
                                 f"REQUEST to {nid}", message_id=msg_id,
                                 _parent=parent)
                node._event_log.append(send_evt)

                msg = Message(node.node_id, send_ts, msg_id, "REQUEST")
                other.receive_message(msg, _send_event=send_evt)

                # Other node sends ACK back
                ack_msg_id = f"msg-{next(_msg_counter)}"
                ack_ts = other.clock.send_tick()
                ack_parent = other._event_log[-1] if other._event_log else None
                ack_send = Event(nid, "SEND", ack_ts,
                                 f"ACK to {node.node_id}", message_id=ack_msg_id,
                                 _parent=ack_parent)
                other._event_log.append(ack_send)

                ack_msg = Message(nid, ack_ts, ack_msg_id, "ACK")
                node.receive_message(ack_msg, _send_event=ack_send)

                self._acks[node.node_id].add(nid)

    def release(self, node: Node) -> None:
        """Node releases the lock. Broadcasts RELEASE to all others."""
        self._request_queue = [(t, n) for t, n in self._request_queue
                               if n != node.node_id]

        for nid, other in self._nodes.items():
            if nid != node.node_id:
                msg_id = f"msg-{next(_msg_counter)}"
                send_ts = node.clock.send_tick()
                parent = node._event_log[-1] if node._event_log else None
                send_evt = Event(node.node_id, "SEND", send_ts,
                                 f"RELEASE to {nid}", message_id=msg_id,
                                 _parent=parent)
                node._event_log.append(send_evt)

                msg = Message(node.node_id, send_ts, msg_id, "RELEASE")
                other.receive_message(msg, _send_event=send_evt)

        if node.node_id in self._acks:
            del self._acks[node.node_id]

    def can_enter(self, node: Node) -> bool:
        """Check if node can enter the critical section."""
        if not self._request_queue:
            return False
        own_requests = [(t, n) for t, n in self._request_queue if n == node.node_id]
        if not own_requests:
            return False
        lowest = self._request_queue[0]
        if lowest[1] != node.node_id:
            return False
        others = set(self._nodes.keys()) - {node.node_id}
        return others.issubset(self._acks.get(node.node_id, set()))


def total_order(events: List[Event]) -> List[Event]:
    """Sort events into total order by (timestamp, node_id)."""
    return sorted(events, key=lambda e: (e.timestamp, e.node_id))


def happens_before(event_a: Event, event_b: Event,
                   all_events: List[Event]) -> Optional[bool]:
    """Determine if event_a happens before event_b.

    Returns True if a -> b, False if b -> a, None if concurrent.
    """
    if event_a is event_b:
        return None
    if _reaches(event_a, event_b):
        return True
    if _reaches(event_b, event_a):
        return False
    return None


def _reaches(source: Event, target: Event) -> bool:
    """BFS backward from target to see if source is reachable."""
    visited = set()
    queue = deque([target])
    while queue:
        current = queue.popleft()
        if current is source:
            return True
        cid = id(current)
        if cid in visited:
            continue
        visited.add(cid)
        if current._parent is not None:
            queue.append(current._parent)
        if current._cause is not None:
            queue.append(current._cause)
    return False
