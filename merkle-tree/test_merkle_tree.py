"""Tests for Merkle tree implementation."""
import sys

from merkle_tree import MerkleTree, KeyRangeMerkleTree, MerkleTreeBuilder, MerkleProof, EMPTY_HASH


def test_construction_and_determinism():
    """Same data -> same root hash; correct leaf count and height."""
    data = [b"block0", b"block1", b"block2", b"block3"]
    t1 = MerkleTree(data)
    t2 = MerkleTree(data)
    assert t1.root_hash == t2.root_hash, "Determinism failed"
    assert t1.leaf_count == 4
    assert t1.height == 2
    assert len(t1.root_hash) == 64  # SHA-256 hex


def test_sensitivity():
    """Any change in data changes root hash."""
    t1 = MerkleTree([b"block0", b"block1", b"block2", b"block3"])
    t2 = MerkleTree([b"block0", b"CHANGED", b"block2", b"block3"])
    assert t1.root_hash != t2.root_hash


def test_diff_detection():
    """Diff finds exactly the differing leaves."""
    base = [b"block0", b"block1", b"block2", b"block3"]
    t1 = MerkleTree(base)
    # Single diff
    t2 = MerkleTree([b"block0", b"CHANGED", b"block2", b"block3"])
    assert t1.diff(t2) == [1]
    # Multiple diffs
    t3 = MerkleTree([b"block0", b"CHANGED", b"block2", b"ALSO_CHANGED"])
    assert sorted(t1.diff(t3)) == [1, 3]
    # All different
    t4 = MerkleTree([b"w", b"x", b"y", b"z"])
    assert sorted(t1.diff(t4)) == [0, 1, 2, 3]
    # No diff (identical)
    t5 = MerkleTree(base)
    assert t1.diff(t5) == []


def test_proof_generation_and_verification():
    """Valid proofs verify; invalid data rejected."""
    tree = MerkleTree([b"block0", b"block1", b"block2", b"block3"])
    proof = tree.get_proof(2)
    assert proof.leaf_index == 2
    assert proof.root_hash == tree.root_hash
    assert len(proof.siblings) == 2  # height=2
    assert MerkleTree.verify_proof(b"block2", proof) is True
    assert MerkleTree.verify_proof(b"fake_data", proof) is False


def test_incremental_update():
    """Update leaf changes root hash, only affects that leaf."""
    data = [b"block0", b"block1", b"block2", b"block3"]
    tree = MerkleTree(data)
    old_root = tree.root_hash
    tree.update_leaf(1, b"updated")
    assert tree.root_hash != old_root
    assert tree.get_leaf(1).data == b"updated"
    assert tree.get_leaf(0).data == b"block0"
    # Proof still works after update
    proof = tree.get_proof(1)
    assert MerkleTree.verify_proof(b"updated", proof) is True


def test_padding_non_power_of_2():
    """Non-power-of-2 leaves get padded; proofs still valid."""
    tree = MerkleTree([b"a", b"b", b"c"])
    assert tree.leaf_count == 3
    assert tree._padded_size == 4
    proof = tree.get_proof(0)
    assert MerkleTree.verify_proof(b"a", proof) is True
    proof2 = tree.get_proof(2)
    assert MerkleTree.verify_proof(b"c", proof2) is True


def test_key_range_merkle_tree():
    """KeyRangeMerkleTree detects differing keys."""
    r1 = [("apple", "red"), ("banana", "yellow"), ("cherry", "dark red"), ("date", "brown")]
    r2 = [("apple", "red"), ("banana", "green"), ("cherry", "dark red"), ("date", "brown")]
    t1 = KeyRangeMerkleTree(r1)
    t2 = KeyRangeMerkleTree(r2)
    assert t1.root_hash != t2.root_hash
    assert t1.diff_keys(t2) == ["banana"]
    # update_key
    t1.update_key("banana", "green")
    assert t1.root_hash == t2.root_hash


def test_serialization_roundtrip():
    """Serialize and deserialize preserves tree."""
    tree = MerkleTree([b"block0", b"block1", b"block2", b"block3"])
    tree.update_leaf(1, b"updated")
    d = tree.to_dict()
    restored = MerkleTree.from_dict(d)
    assert restored.root_hash == tree.root_hash
    assert restored.leaf_count == tree.leaf_count
    assert restored.get_leaf(1).data == b"updated"


def test_edge_cases():
    """Single leaf, two leaves, empty tree."""
    # Empty
    te = MerkleTree()
    assert te.leaf_count == 0
    assert te.root_hash == EMPTY_HASH
    # Single leaf
    t1 = MerkleTree([b"only"])
    assert t1.leaf_count == 1
    assert t1.height == 0
    p = t1.get_proof(0)
    assert MerkleTree.verify_proof(b"only", p) is True
    # Two leaves
    t2 = MerkleTree([b"a", b"b"])
    assert t2.leaf_count == 2
    assert t2.height == 1


def test_builder():
    """MerkleTreeBuilder accumulates and builds correctly."""
    builder = MerkleTreeBuilder()
    for i in range(8):
        builder.add_leaf(f"block{i}".encode())
    tree = builder.build()
    assert tree.leaf_count == 8
    # Should match direct construction
    direct = MerkleTree([f"block{i}".encode() for i in range(8)])
    assert tree.root_hash == direct.root_hash


if __name__ == "__main__":
    tests = [
        test_construction_and_determinism,
        test_sensitivity,
        test_diff_detection,
        test_proof_generation_and_verification,
        test_incremental_update,
        test_padding_non_power_of_2,
        test_key_range_merkle_tree,
        test_serialization_roundtrip,
        test_edge_cases,
        test_builder,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")
