from merkle_tree import *

data = [b'block0', b'block1', b'block2', b'block3']
tree = MerkleTree(data)
print(f'Root hash: {tree.root_hash[:16]}...')
assert tree.leaf_count == 4
assert tree.height == 2

tree2 = MerkleTree(data)
assert tree.root_hash == tree2.root_hash
print('Determinism: OK')

data_modified = [b'block0', b'CHANGED', b'block2', b'block3']
tree3 = MerkleTree(data_modified)
assert tree.root_hash != tree3.root_hash
print('Sensitivity: OK')

diffs = tree.diff(tree3)
assert diffs == [1], f'Expected [1], got {diffs}'
print(f'Diff: {diffs} OK')

data_multi = [b'block0', b'CHANGED', b'block2', b'ALSO_CHANGED']
tree4 = MerkleTree(data_multi)
diffs = tree.diff(tree4)
assert sorted(diffs) == [1, 3]
print(f'Multi diff: {sorted(diffs)} OK')

proof = tree.get_proof(2)
assert MerkleTree.verify_proof(b'block2', proof) == True
assert MerkleTree.verify_proof(b'fake_data', proof) == False
assert proof.leaf_index == 2
assert len(proof.siblings) == 2
assert proof.root_hash == tree.root_hash
print('Proof: OK')

tree.update_leaf(1, b'updated_block1')
assert tree.root_hash != tree2.root_hash
assert tree.get_leaf(1).data == b'updated_block1'
assert tree.get_leaf(0).data == b'block0'
print('Update: OK')

tree5 = MerkleTree([b'a', b'b', b'c'])
assert tree5.leaf_count == 3
proof = tree5.get_proof(0)
assert MerkleTree.verify_proof(b'a', proof) == True
print('Padding: OK')

r1 = [('apple','red'),('banana','yellow'),('cherry','dark red'),('date','brown')]
r2 = [('apple','red'),('banana','green'),('cherry','dark red'),('date','brown')]
tr1 = KeyRangeMerkleTree(r1)
tr2 = KeyRangeMerkleTree(r2)
assert tr1.root_hash != tr2.root_hash
dk = tr1.diff_keys(tr2)
assert dk == ['banana'], f'Expected ["banana"], got {dk}'
print(f'Key-range diff: {dk} OK')

builder = MerkleTreeBuilder()
for i in range(8):
    builder.add_leaf(f'block{i}'.encode())
tree6 = builder.build()
assert tree6.leaf_count == 8
print('Builder: OK')

serialized = tree.to_dict()
restored = MerkleTree.from_dict(serialized)
assert restored.root_hash == tree.root_hash
print('Serialization: OK')

t1 = MerkleTree([b'only'])
assert t1.leaf_count == 1
assert t1.height == 0
p = t1.get_proof(0)
assert MerkleTree.verify_proof(b'only', p)
print('Single leaf: OK')

t2 = MerkleTree([b'a', b'b'])
assert t2.leaf_count == 2
assert t2.height == 1
print('Two leaves: OK')

te = MerkleTree()
assert te.leaf_count == 0
print('Empty tree: OK')

tsame = MerkleTree([b'x', b'x', b'x', b'x'])
assert tsame.leaf_count == 4
tdiff = MerkleTree([b'x', b'x', b'x', b'y'])
assert sorted(tsame.diff(tdiff)) == [3]
print('All identical / one diff: OK')

tall1 = MerkleTree([b'a', b'b', b'c', b'd'])
tall2 = MerkleTree([b'w', b'x', b'y', b'z'])
assert sorted(tall1.diff(tall2)) == [0, 1, 2, 3]
print('All different: OK')

print()
print('ALL TESTS PASSED')
