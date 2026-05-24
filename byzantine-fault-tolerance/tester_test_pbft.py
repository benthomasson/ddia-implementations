"""Tests for PBFT consensus simulation."""

import pytest
from pbft import (
    Message, MessageType, ByzantineMode, PBFTNode, PBFTCluster, compute_digest
)


# --- Core protocol tests ---

def test_normal_single_request():
    """4 nodes, no faults, single request committed by all."""
    cluster = PBFTCluster(n=4, f=1)
    assert cluster.submit_request("SET x = 1")
    assert cluster.verify_agreement()
    assert cluster.get_executed_log() == ["SET x = 1"]


def test_multiple_sequential_requests():
    """Multiple requests committed in order."""
    cluster = PBFTCluster(n=4, f=1)
    for cmd in ["SET x = 1", "SET y = 2", "SET z = 3"]:
        cluster.submit_request(cmd)
    assert cluster.verify_agreement()
    assert cluster.get_executed_log() == ["SET x = 1", "SET y = 2", "SET z = 3"]


def test_sequence_numbers_are_consecutive():
    """Executed log has consecutive sequence numbers starting at 1."""
    cluster = PBFTCluster(n=4, f=1)
    for i in range(5):
        cluster.submit_request(f"REQ_{i}")
    for node in cluster.nodes:
        if node.byzantine_mode == ByzantineMode.HONEST:
            seqs = [s for s, _ in node._executed_log]
            assert seqs == list(range(1, 6))


# --- Byzantine fault tolerance ---

def test_silent_byzantine_node():
    """Protocol works with 1 silent node out of 4."""
    cluster = PBFTCluster(n=4, f=1, byzantine_nodes={2: ByzantineMode.SILENT})
    assert cluster.submit_request("SET a = 10")
    assert cluster.verify_agreement()
    assert cluster.get_executed_log() == ["SET a = 10"]


def test_equivocating_byzantine_node():
    """Equivocating non-primary doesn't break agreement."""
    cluster = PBFTCluster(n=4, f=1, byzantine_nodes={1: ByzantineMode.EQUIVOCATING})
    assert cluster.submit_request("SET b = 20")
    assert cluster.verify_agreement()


def test_wrong_digest_byzantine_node():
    """Wrong-digest messages are rejected, protocol still works."""
    cluster = PBFTCluster(n=4, f=1, byzantine_nodes={2: ByzantineMode.WRONG_DIGEST})
    assert cluster.submit_request("SET c = 30")
    assert cluster.verify_agreement()


def test_7_node_cluster_2_byzantine():
    """7-node cluster tolerates 2 Byzantine faults."""
    cluster = PBFTCluster(
        n=7, f=2,
        byzantine_nodes={3: ByzantineMode.WRONG_DIGEST, 5: ByzantineMode.SILENT}
    )
    assert cluster.submit_request("TRANSFER 100")
    assert cluster.verify_agreement()


# --- View change ---

def test_view_change_on_silent_primary():
    """Silent primary triggers view change; new primary handles requests."""
    cluster = PBFTCluster(n=4, f=1, byzantine_nodes={0: ByzantineMode.SILENT})
    new_view = cluster.trigger_view_change()
    assert new_view >= 1
    assert cluster.get_node(1).is_primary
    assert cluster.submit_request("SET d = 40")
    assert cluster.verify_agreement()


# --- Validation / edge cases ---

def test_invalid_cluster_n_vs_f():
    """N != 3f+1 raises ValueError."""
    with pytest.raises(ValueError):
        PBFTCluster(n=5, f=1)


def test_too_many_byzantine_nodes():
    """More than f Byzantine nodes raises ValueError."""
    with pytest.raises(ValueError):
        PBFTCluster(n=4, f=1, byzantine_nodes={1: ByzantineMode.SILENT, 2: ByzantineMode.SILENT})


def test_duplicate_message_rejected():
    """Same pre-prepare delivered twice; second is ignored."""
    cluster = PBFTCluster(n=4, f=1)
    digest = compute_digest("DUP_TEST")
    pp = Message(MessageType.PRE_PREPARE, 0, 1, digest, 0, {"request": "DUP_TEST"})
    node1 = cluster.get_node(1)
    r1 = node1.receive_message(pp)
    r2 = node1.receive_message(pp)
    assert len(r1) > 0
    assert len(r2) == 0


def test_reject_unknown_sender():
    """Messages from non-existent node IDs are dropped."""
    node = PBFTNode(0, 4, 1)
    bad = Message(MessageType.PREPARE, 0, 1, "digest", sender=99)
    assert node.receive_message(bad) == []
    bad2 = Message(MessageType.PREPARE, 0, 1, "digest", sender=-1)
    assert node.receive_message(bad2) == []


def test_digest_computation():
    """compute_digest is deterministic and uses SHA-256."""
    d1 = compute_digest("hello")
    d2 = compute_digest("hello")
    d3 = compute_digest("world")
    assert d1 == d2
    assert d1 != d3
    assert len(d1) == 64  # SHA-256 hex digest


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
