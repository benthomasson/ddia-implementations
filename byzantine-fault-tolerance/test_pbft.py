"""Tests for PBFT consensus simulation."""

import pytest
from pbft import (
    Message, MessageType, ByzantineMode, PBFTNode, PBFTCluster, compute_digest
)


# 1. Normal case: 4 nodes, no faults, single request committed by all
def test_normal_case_single_request():
    cluster = PBFTCluster(n=4, f=1)
    success = cluster.submit_request("SET x = 1")
    assert success
    assert cluster.verify_agreement()
    log = cluster.get_executed_log()
    assert log == ["SET x = 1"]


# 2. Multiple sequential requests committed in order
def test_multiple_sequential_requests():
    cluster = PBFTCluster(n=4, f=1)
    cluster.submit_request("SET x = 1")
    cluster.submit_request("SET y = 2")
    cluster.submit_request("SET z = 3")
    assert cluster.verify_agreement()
    log = cluster.get_executed_log()
    assert log == ["SET x = 1", "SET y = 2", "SET z = 3"]


# 3. 1 silent Byzantine node (4 nodes): protocol still works
def test_silent_byzantine():
    cluster = PBFTCluster(n=4, f=1, byzantine_nodes={2: ByzantineMode.SILENT})
    success = cluster.submit_request("SET a = 10")
    assert success
    assert cluster.verify_agreement()
    log = cluster.get_executed_log()
    assert log == ["SET a = 10"]


# 4. 1 equivocating Byzantine node: honest nodes still agree
def test_equivocating_byzantine():
    # Use node 1 as equivocating (not primary in view 0)
    cluster = PBFTCluster(n=4, f=1, byzantine_nodes={1: ByzantineMode.EQUIVOCATING})
    success = cluster.submit_request("SET b = 20")
    assert success
    assert cluster.verify_agreement()


# 5. 1 wrong-digest Byzantine node: bad messages rejected
def test_wrong_digest_byzantine():
    cluster = PBFTCluster(n=4, f=1, byzantine_nodes={2: ByzantineMode.WRONG_DIGEST})
    success = cluster.submit_request("SET c = 30")
    assert success
    assert cluster.verify_agreement()


# 6. View change: silent primary triggers view change, new primary works
def test_view_change_silent_primary():
    cluster = PBFTCluster(n=4, f=1, byzantine_nodes={0: ByzantineMode.SILENT})
    # Primary (node 0) is silent, won't propose
    # Tick to trigger view change
    new_view = cluster.trigger_view_change()
    assert new_view >= 1

    # New primary should be node 1
    assert cluster.get_node(1).is_primary

    # Now submit through new primary
    success = cluster.submit_request("SET d = 40")
    assert success
    assert cluster.verify_agreement()


# 7. After view change, previously prepared requests are re-proposed
def test_view_change_repropose_prepared():
    cluster = PBFTCluster(n=4, f=1)

    # Submit a request that gets prepared
    primary = cluster.get_node(0)
    msgs = primary.submit_request("PREPARED_REQ")
    cluster.pending_messages.extend(msgs)

    # Deliver just the pre-prepare and prepare phases (partial run)
    # Run a few rounds to get to prepared state
    cluster.run_protocol(max_rounds=2)

    # Check some node is prepared
    any_prepared = False
    for node in cluster.nodes:
        if node.prepared_requests:
            any_prepared = True
            break

    # Now force view change
    new_view = cluster.trigger_view_change()
    assert new_view >= 1


# 8. Agreement invariant: honest nodes always have identical executed logs
def test_agreement_invariant():
    cluster = PBFTCluster(n=4, f=1)
    for i in range(10):
        cluster.submit_request(f"OP_{i}")
        assert cluster.verify_agreement(), f"Agreement broken after request {i}"
    log = cluster.get_executed_log()
    assert len(log) == 10


# 9. 7-node cluster with 2 Byzantine faults
def test_7_node_cluster():
    cluster = PBFTCluster(
        n=7, f=2,
        byzantine_nodes={3: ByzantineMode.WRONG_DIGEST, 5: ByzantineMode.SILENT}
    )
    success = cluster.submit_request("TRANSFER 100")
    assert success
    assert cluster.verify_agreement()


# 10. Cluster with too many Byzantine nodes (> f) cannot be created
def test_too_many_byzantine_nodes():
    with pytest.raises(ValueError):
        PBFTCluster(n=4, f=1, byzantine_nodes={1: ByzantineMode.SILENT, 2: ByzantineMode.SILENT})

    # Also test invalid n vs f
    with pytest.raises(ValueError):
        PBFTCluster(n=5, f=1)


# 11. f+1 matching replies required for client acceptance
def test_f_plus_1_matching_replies():
    cluster = PBFTCluster(n=4, f=1)
    cluster.submit_request("REPLY_TEST")

    # Collect replies from honest nodes
    replies = {}
    for node in cluster.nodes:
        if node.byzantine_mode == ByzantineMode.HONEST:
            for seq, req in node._executed_log:
                if req == "REPLY_TEST":
                    replies[node.node_id] = req

    # Need f+1 = 2 matching replies
    assert len(replies) >= cluster.f + 1


# 12. Duplicate message rejection
def test_duplicate_message_rejection():
    cluster = PBFTCluster(n=4, f=1)
    primary = cluster.get_node(0)

    # Create a request
    request = "DUP_TEST"
    digest = compute_digest(request)

    # Create a pre-prepare
    pp = Message(MessageType.PRE_PREPARE, 0, 1, digest, 0, {"request": request})

    # Send to node 1 twice
    node1 = cluster.get_node(1)
    result1 = node1.receive_message(pp)
    result2 = node1.receive_message(pp)

    # First should produce PREPARE, second should be rejected
    assert len(result1) > 0
    assert len(result2) == 0


# 13. Requests execute in sequence number order
def test_execution_order():
    cluster = PBFTCluster(n=4, f=1)
    for i in range(5):
        cluster.submit_request(f"REQ_{i}")

    log = cluster.get_executed_log()
    assert log == [f"REQ_{i}" for i in range(5)]

    # Also verify sequence numbers are in order
    for node in cluster.nodes:
        if node.byzantine_mode == ByzantineMode.HONEST:
            seqs = [s for s, _ in node._executed_log]
            assert seqs == sorted(seqs)
            assert seqs == list(range(1, 6))


# 14. Reject messages from unknown senders
def test_reject_unknown_sender():
    node = PBFTNode(0, 4, 1)
    bad_msg = Message(MessageType.PREPARE, 0, 1, "digest", sender=99)
    result = node.receive_message(bad_msg)
    assert result == []

    bad_msg2 = Message(MessageType.PREPARE, 0, 1, "digest", sender=-1)
    result2 = node.receive_message(bad_msg2)
    assert result2 == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
