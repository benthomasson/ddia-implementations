"""Tests for Total Order Broadcast protocol."""

import pytest
from total_order_broadcast import ConsensusInstance, TOBNode, TOBCluster, LinearizableRegister


# 1. Single message broadcast
def test_single_message_broadcast():
    cluster = TOBCluster(3)
    cluster.broadcast(0, "hello")
    assert cluster.run_until_delivered(1)
    assert cluster.verify_total_order()
    for i in range(3):
        assert cluster.get_delivery_order(i) == ["hello"]


# 2. Multiple messages in same order
def test_multiple_messages_same_order():
    cluster = TOBCluster(3)
    cluster.broadcast(0, "msg_A")
    assert cluster.run_until_delivered(1)
    cluster.broadcast(0, "msg_B")
    assert cluster.run_until_delivered(2)
    cluster.broadcast(0, "msg_C")
    assert cluster.run_until_delivered(3)
    assert cluster.verify_total_order()
    order = cluster.get_delivery_order(0)
    assert "msg_A" in order
    assert "msg_B" in order
    assert "msg_C" in order
    assert len(order) == 3


# 3. Concurrent proposals from different nodes
def test_concurrent_proposals():
    cluster = TOBCluster(3)
    cluster.broadcast(0, "msg_A")
    cluster.broadcast(1, "msg_B")
    cluster.broadcast(2, "msg_C")
    assert cluster.run_until_delivered(3)
    assert cluster.verify_total_order()
    order = cluster.get_delivery_order(0)
    assert set(order) == {"msg_A", "msg_B", "msg_C"}


# 4. Node crash - remaining majority continues
def test_node_crash():
    cluster = TOBCluster(3)
    cluster.broadcast(0, "before_crash")
    assert cluster.run_until_delivered(1)
    cluster.crash_node(2)
    cluster.broadcast(0, "after_crash")
    assert cluster.run_until_delivered(2)
    assert "after_crash" in cluster.get_delivery_order(0)
    assert "after_crash" in cluster.get_delivery_order(1)


# 5. Node recovery catches up
def test_node_recovery():
    cluster = TOBCluster(3)
    cluster.broadcast(0, "msg_1")
    assert cluster.run_until_delivered(1)
    cluster.crash_node(2)
    cluster.broadcast(0, "msg_2")
    cluster.broadcast(1, "msg_3")
    assert cluster.run_until_delivered(3)
    cluster.recover_node(2)
    assert cluster.get_delivery_order(2) == cluster.get_delivery_order(0)


# 6. Minority cannot make progress
def test_minority_cannot_progress():
    cluster = TOBCluster(3)
    cluster.crash_node(1)
    cluster.crash_node(2)
    cluster.broadcast(0, "lonely")
    result = cluster.run_until_delivered(1, max_rounds=100)
    assert not result


# 7. Consensus instance with competing proposers
def test_consensus_competing_proposers():
    inst = ConsensusInstance(slot=0, num_nodes=3)
    # Two proposers prepare with different numbers
    r1 = inst.prepare(1, proposer_id=0)
    assert r1['promised']
    r2 = inst.prepare(2, proposer_id=1)
    assert r2['promised']
    # First proposer's accept should fail (lower number)
    r3 = inst.accept(1, "val_A", proposer_id=0)
    assert not r3['accepted']
    # Second proposer's accept should succeed
    r4 = inst.accept(2, "val_B", proposer_id=1)
    assert r4['accepted']


# 8. Decided values are immutable
def test_decided_values_immutable():
    cluster = TOBCluster(3)
    cluster.broadcast(0, "immutable_msg")
    assert cluster.run_until_delivered(1)
    decided_val = cluster.get_delivery_order(0)[0]
    # The slot 0 instance on each node should be decided
    for node in cluster.nodes.values():
        inst = node._get_instance(0)
        assert inst.is_decided
        assert inst.decided_value == decided_val


# 9. Linearizable register write and read
def test_linearizable_read_write():
    cluster = TOBCluster(3)
    reg = LinearizableRegister(cluster, node_id=0)
    reg.write("x", 42)
    val = reg.read("x")
    assert val == 42


# 10. Linearizable CAS
def test_linearizable_cas():
    cluster = TOBCluster(3)
    reg = LinearizableRegister(cluster, node_id=0)
    reg.write("x", 10)
    # Successful CAS
    assert reg.compare_and_set("x", 10, 20)
    assert reg.read("x") == 20
    # Failed CAS (wrong expected)
    assert not reg.compare_and_set("x", 10, 30)
    assert reg.read("x") == 20


# 11. Delivery callbacks fire in order
def test_delivery_callbacks_in_order():
    cluster = TOBCluster(3)
    delivered = []
    cluster.get_node(0).on_deliver(lambda slot, msg: delivered.append((slot, msg)))
    cluster.broadcast(0, "a")
    cluster.run_until_delivered(1)
    cluster.broadcast(1, "b")
    cluster.run_until_delivered(2)
    cluster.broadcast(2, "c")
    cluster.run_until_delivered(3)
    assert len(delivered) == 3
    # Slots should be monotonically increasing
    slots = [s for s, _ in delivered]
    assert slots == sorted(slots)
    assert slots == [0, 1, 2]


# 12. FIFO per-sender ordering
def test_fifo_per_sender():
    cluster = TOBCluster(3)
    # Node 0 sends messages in order
    cluster.broadcast(0, "s0_1")
    cluster.run_until_delivered(1)
    cluster.broadcast(0, "s0_2")
    cluster.run_until_delivered(2)
    cluster.broadcast(0, "s0_3")
    cluster.run_until_delivered(3)
    order = cluster.get_delivery_order(0)
    # s0_1 must come before s0_2, which must come before s0_3
    assert order.index("s0_1") < order.index("s0_2") < order.index("s0_3")


# 13. Many sequential broadcasts (at least 20)
def test_many_sequential_broadcasts():
    cluster = TOBCluster(3)
    for i in range(25):
        cluster.broadcast(i % 3, f"msg_{i}")
    assert cluster.run_until_delivered(25)
    assert cluster.verify_total_order()
    order = cluster.get_delivery_order(0)
    assert len(order) == 25
    assert set(order) == {f"msg_{i}" for i in range(25)}


# 14. Replicated state machine (counter)
def test_replicated_state_machine():
    cluster = TOBCluster(3)
    counters = {i: [0] for i in range(3)}

    for i in range(3):
        def make_cb(node_id):
            def cb(slot, msg):
                if isinstance(msg, str) and msg.startswith("inc"):
                    counters[node_id][0] += 1
            return cb
        cluster.get_node(i).on_deliver(make_cb(i))

    # Broadcast increment operations from different nodes
    for i in range(10):
        cluster.broadcast(i % 3, f"inc_{i}")

    assert cluster.run_until_delivered(10)
    assert cluster.verify_total_order()

    # All nodes should have the same counter value
    assert counters[0][0] == counters[1][0] == counters[2][0] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
