"""Merkle tree for data verification and anti-entropy."""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


EMPTY_HASH = hashlib.sha256(b"").hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


@dataclass
class MerkleNode:
    """A node in the Merkle tree, constructed on-demand."""
    hash: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    data: Optional[bytes] = None
    index: int = -1


@dataclass
class MerkleProof:
    """Inclusion proof for a leaf in a Merkle tree."""
    leaf_hash: str
    leaf_index: int
    siblings: List[Tuple[str, str]]  # (hash, "left" or "right")
    root_hash: str


class MerkleTree:
    """Array-backed Merkle tree with SHA-256 hashing."""

    def __init__(self, data_blocks: Optional[List[bytes]] = None, _from_dict=False):
        """Build a Merkle tree from a list of data blocks."""
        if _from_dict:
            return  # initialized by from_dict
        if data_blocks is None or len(data_blocks) == 0:
            self._leaf_count = 0
            self._padded_size = 1
            self._hashes = [EMPTY_HASH]
            self._data = [None]
            return

        self._leaf_count = len(data_blocks)
        self._padded_size = _next_power_of_2(len(data_blocks))
        total_nodes = 2 * self._padded_size - 1
        self._hashes = [""] * total_nodes
        self._data = [None] * self._padded_size

        # Fill leaves
        leaf_start = self._padded_size - 1
        for i in range(self._padded_size):
            if i < len(data_blocks):
                self._data[i] = data_blocks[i]
                self._hashes[leaf_start + i] = _sha256(data_blocks[i])
            else:
                self._hashes[leaf_start + i] = EMPTY_HASH

        # Build internal nodes bottom-up
        for i in range(leaf_start - 1, -1, -1):
            left_hash = self._hashes[2 * i + 1]
            right_hash = self._hashes[2 * i + 2]
            self._hashes[i] = _sha256((left_hash + right_hash).encode())

    @property
    def root_hash(self) -> str:
        return self._hashes[0]

    @property
    def root(self) -> MerkleNode:
        return self._build_node(0)

    @property
    def leaf_count(self) -> int:
        return self._leaf_count

    @property
    def height(self) -> int:
        if self._padded_size <= 1:
            return 0
        h = 0
        n = self._padded_size
        while n > 1:
            n >>= 1
            h += 1
        return h

    def _build_node(self, idx: int) -> MerkleNode:
        """Construct a MerkleNode from array index (on-demand)."""
        leaf_start = self._padded_size - 1
        if idx >= leaf_start:
            leaf_idx = idx - leaf_start
            return MerkleNode(
                hash=self._hashes[idx],
                data=self._data[leaf_idx] if leaf_idx < len(self._data) else None,
                index=leaf_idx,
            )
        return MerkleNode(
            hash=self._hashes[idx],
            left=self._build_node(2 * idx + 1),
            right=self._build_node(2 * idx + 2),
        )

    def get_leaf(self, index: int) -> MerkleNode:
        """Return the leaf node at the given index."""
        if index < 0 or index >= self._leaf_count:
            raise IndexError(f"Leaf index {index} out of range [0, {self._leaf_count})")
        arr_idx = self._padded_size - 1 + index
        return MerkleNode(
            hash=self._hashes[arr_idx],
            data=self._data[index],
            index=index,
        )

    def update_leaf(self, index: int, new_data: bytes) -> None:
        """Update a leaf's data and recompute hashes up to root. O(log N)."""
        if index < 0 or index >= self._leaf_count:
            raise IndexError(f"Leaf index {index} out of range [0, {self._leaf_count})")
        self._data[index] = new_data
        arr_idx = self._padded_size - 1 + index
        self._hashes[arr_idx] = _sha256(new_data)
        # Walk up to root
        while arr_idx > 0:
            parent = (arr_idx - 1) // 2
            left = self._hashes[2 * parent + 1]
            right = self._hashes[2 * parent + 2]
            self._hashes[parent] = _sha256((left + right).encode())
            arr_idx = parent

    def get_proof(self, index: int) -> MerkleProof:
        """Generate an inclusion proof for the leaf at the given index."""
        if index < 0 or index >= self._leaf_count:
            raise IndexError(f"Leaf index {index} out of range [0, {self._leaf_count})")
        arr_idx = self._padded_size - 1 + index
        leaf_hash = self._hashes[arr_idx]
        siblings = []
        while arr_idx > 0:
            parent = (arr_idx - 1) // 2
            if arr_idx == 2 * parent + 1:
                # current is left child, sibling is right
                sibling_idx = 2 * parent + 2
                siblings.append((self._hashes[sibling_idx], "right"))
            else:
                # current is right child, sibling is left
                sibling_idx = 2 * parent + 1
                siblings.append((self._hashes[sibling_idx], "left"))
            arr_idx = parent
        return MerkleProof(
            leaf_hash=leaf_hash,
            leaf_index=index,
            siblings=siblings,
            root_hash=self.root_hash,
        )

    @staticmethod
    def verify_proof(data: bytes, proof: MerkleProof) -> bool:
        """Verify that data is included in a tree with the given root hash."""
        current = _sha256(data)
        if current != proof.leaf_hash:
            return False
        for sibling_hash, direction in proof.siblings:
            if direction == "left":
                current = _sha256((sibling_hash + current).encode())
            else:
                current = _sha256((current + sibling_hash).encode())
        return current == proof.root_hash

    def diff(self, other: 'MerkleTree') -> List[int]:
        """Find indices of leaves that differ between self and other."""
        if self._padded_size != other._padded_size:
            raise ValueError("Trees must have the same padded size to diff")
        result = []
        self._diff_recursive(other, 0, result)
        # Filter to only real leaves (not padding)
        max_idx = max(self._leaf_count, other._leaf_count)
        return [i for i in result if i < max_idx]

    def _diff_recursive(self, other: 'MerkleTree', idx: int, result: List[int]):
        if self._hashes[idx] == other._hashes[idx]:
            return  # subtrees match
        leaf_start = self._padded_size - 1
        if idx >= leaf_start:
            result.append(idx - leaf_start)
            return
        self._diff_recursive(other, 2 * idx + 1, result)
        self._diff_recursive(other, 2 * idx + 2, result)

    def to_dict(self) -> Dict:
        """Serialize the tree to a dictionary."""
        return {
            "hashes": self._hashes,
            "leaf_count": self._leaf_count,
            "padded_size": self._padded_size,
            "data": [d.hex() if d is not None else None for d in self._data],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'MerkleTree':
        """Reconstruct tree from serialized dictionary."""
        tree = cls(_from_dict=True)
        tree._hashes = d["hashes"]
        tree._leaf_count = d["leaf_count"]
        tree._padded_size = d["padded_size"]
        tree._data = [bytes.fromhex(x) if x is not None else None for x in d["data"]]
        return tree


class KeyRangeMerkleTree:
    """Merkle tree over sorted key-value pairs for anti-entropy."""

    def __init__(self, kv_pairs: List[Tuple[str, str]]):
        """Build a Merkle tree over sorted key-value pairs."""
        sorted_pairs = sorted(kv_pairs, key=lambda x: x[0])
        self._keys = [k for k, v in sorted_pairs]
        data_blocks = [f"{k}:{v}".encode() for k, v in sorted_pairs]
        self._tree = MerkleTree(data_blocks)

    @property
    def root_hash(self) -> str:
        return self._tree.root_hash

    def diff_keys(self, other: 'KeyRangeMerkleTree') -> List[str]:
        """Find keys that differ between two trees."""
        diff_indices = self._tree.diff(other._tree)
        result = []
        for i in diff_indices:
            if i < len(self._keys):
                result.append(self._keys[i])
            elif i < len(other._keys):
                result.append(other._keys[i])
        return result

    def get_key_proof(self, key: str) -> MerkleProof:
        """Generate a proof for a specific key."""
        idx = self._keys.index(key)
        return self._tree.get_proof(idx)

    def update_key(self, key: str, new_value: str) -> None:
        """Update a key's value and recompute hashes."""
        idx = self._keys.index(key)
        self._tree.update_leaf(idx, f"{key}:{new_value}".encode())


class MerkleTreeBuilder:
    """Streaming Merkle tree builder."""

    def __init__(self):
        self._leaves: List[bytes] = []

    def add_leaf(self, data: bytes) -> None:
        """Add a leaf to the tree."""
        self._leaves.append(data)

    def build(self) -> MerkleTree:
        """Finalize and return the constructed MerkleTree."""
        return MerkleTree(self._leaves)
