"""Tests for Lamport logical clocks implementation."""


from lamport import LamportClock, Node, Event, Message, LamportMutex, total_order, happens_before


def test_clock_basic():
    """Clock tick increments correctly; receive_tick uses max rule."""
    clock = LamportClock()
    assert clock.current_time == 0
    assert clock.tick() == 1
    assert clock.tick() == 2
    assert clock.receive_tick(5) == 6  # max(2,5)+1
    assert clock.receive_tick(3) == 7  # max(6,3)+1 — local is higher


def test_send_receive_timestamps():
    """Receiver's clock adjusts to max(local, received) + 1."""
    a = Node("A")
    b = Node("B")
    e1 = a.local_event("write x=1")
    assert e1.timestamp == 1
    e2 = a.send_message(b, "update x=1")
    assert e2.timestamp == 2
    # B had clock=0, receives ts=2 → max(0,2)+1 = 3
    b_recv = b.get_event_log()[0]
    assert b_recv.event_type == "RECEIVE"
    assert b_recv.timestamp == 3
    # B local after receive
    e4 = b.local_event("read x")
    assert e4.timestamp == 4


def test_total_order():
    """Events from multiple nodes sorted by (timestamp, node_id)."""
    a, b, c = Node("A"), Node("B"), Node("C")
    a.local_event("a1")
    a.send_message(b, "msg")
    b.local_event("b1")
    c.local_event("c1")
    c.local_event("c2")
    all_events = a.get_event_log() + b.get_event_log() + c.get_event_log()
    ordered = total_order(all_events)
    for i in range(len(ordered) - 1):
        assert (ordered[i].timestamp, ordered[i].node_id) <= (ordered[i+1].timestamp, ordered[i+1].node_id)


def test_happens_before_same_node():
    """Sequential events on same node have happens-before relationship."""
    a = Node("A")
    e1 = a.local_event("first")
    e2 = a.local_event("second")
    all_ev = a.get_event_log()
    assert happens_before(e1, e2, all_ev) == True
    assert happens_before(e2, e1, all_ev) == False


def test_happens_before_send_receive():
    """SEND happens-before corresponding RECEIVE."""
    a, b = Node("A"), Node("B")
    send_evt = a.send_message(b, "hello")
    recv_evt = b.get_event_log()[0]
    all_ev = a.get_event_log() + b.get_event_log()
    assert happens_before(send_evt, recv_evt, all_ev) == True


def test_concurrent_events():
    """Independent events on different nodes are concurrent (None)."""
    a, b = Node("A"), Node("B")
    ea = a.local_event("x")
    eb = b.local_event("y")
    all_ev = a.get_event_log() + b.get_event_log()
    assert happens_before(ea, eb, all_ev) is None


def test_transitivity():
    """If A->B and B->C then A->C (across nodes via send/receive)."""
    a, b = Node("A"), Node("B")
    e1 = a.local_event("write x")
    e2 = a.send_message(b, "update")
    recv = b.get_event_log()[0]
    e3 = b.local_event("after receive")
    all_ev = a.get_event_log() + b.get_event_log()
    # e1 -> e2 -> recv -> e3, so e1 -> e3
    assert happens_before(e1, e3, all_ev) == True


def test_mutex_basic():
    """Lower-timestamp requester enters first; second waits then enters after release."""
    a, b, c = Node("A"), Node("B"), Node("C")
    mutex = LamportMutex([a, b, c])
    mutex.request(a)
    mutex.request(b)
    assert mutex.can_enter(a) == True
    assert mutex.can_enter(b) == False
    mutex.release(a)
    assert mutex.can_enter(b) == True


def test_five_node_chain():
    """5 nodes passing messages in a chain; total order is consistent."""
    nodes = [Node(f"N{i}") for i in range(5)]
    nodes[0].local_event("init")
    nodes[0].send_message(nodes[1], "hello")
    nodes[1].send_message(nodes[2], "forward")
    nodes[2].send_message(nodes[3], "chain")
    nodes[3].send_message(nodes[4], "end")
    nodes[4].local_event("done")
    all_ev = []
    for n in nodes:
        all_ev.extend(n.get_event_log())
    ordered = total_order(all_ev)
    for i in range(len(ordered) - 1):
        assert (ordered[i].timestamp, ordered[i].node_id) <= (ordered[i+1].timestamp, ordered[i+1].node_id)
    # First event on N0 should happen-before last event on N4
    assert happens_before(nodes[0].get_event_log()[0], nodes[4].get_event_log()[-1], all_ev) == True


def test_single_node_edge_case():
    """Single node with one event; self-comparison is concurrent (None)."""
    solo = Node("solo")
    e = solo.local_event("alone")
    assert e.timestamp == 1
    assert happens_before(e, e, [e]) is None


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  PASS: {name}")
    print("\nAll tests passed!")
