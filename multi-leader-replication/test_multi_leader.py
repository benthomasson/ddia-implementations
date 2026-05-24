"""Quick validation tests."""
from multi_leader import *

def test_lww():
    cluster = MultiLeaderCluster(['dc1', 'dc2', 'dc3'])
    cluster.node('dc1').put('user:1:name', 'Alice')
    cluster.node('dc2').put('user:1:name', 'Bob')

    assert cluster.node('dc1').get('user:1:name') == 'Alice'
    assert cluster.node('dc2').get('user:1:name') == 'Bob'
    assert cluster.node('dc3').get('user:1:name') is None

    cluster.sync()
    assert cluster.all_converged(), 'Not converged after sync'

    dc1_conflicts = cluster.node('dc1').conflict_log
    assert len(dc1_conflicts) >= 1, f'Expected conflicts, got {len(dc1_conflicts)}'
    assert dc1_conflicts[0].key == 'user:1:name'
    print('Test LWW passed')

def test_custom_merge():
    def counter_merge(key, local_val, remote_val, local_ts, remote_ts):
        return local_val + remote_val

    cluster = MultiLeaderCluster(
        ['a', 'b'],
        strategy=ConflictStrategy.CUSTOM_MERGE,
        merge_fn=counter_merge
    )
    cluster.node('a').put('counter', 5)
    cluster.node('b').put('counter', 3)
    cluster.sync()
    assert cluster.node('a').get('counter') == 8, f'a got {cluster.node("a").get("counter")}'
    assert cluster.node('b').get('counter') == 8, f'b got {cluster.node("b").get("counter")}'
    print('Test custom merge passed')

def test_ring():
    cluster = MultiLeaderCluster(['n1', 'n2', 'n3'], topology=Topology.RING)
    cluster.node('n1').put('x', 1)
    rounds = cluster.sync_until_converged()
    assert rounds >= 2, f'Expected >=2 rounds, got {rounds}'
    assert cluster.all_converged()
    print(f'Test ring passed, rounds={rounds}')

def test_tombstone():
    cluster = MultiLeaderCluster(['a', 'b'])
    cluster.node('a').put('k', 'val')
    cluster.sync()
    assert cluster.node('b').get('k') == 'val'
    cluster.node('b').delete('k')
    cluster.sync()
    assert cluster.node('a').get('k') is None
    assert cluster.node('b').get('k') is None
    print('Test tombstone passed')

def test_lamport_clock():
    node = ReplicaNode('test')
    ts1 = node.put('a', 1)
    ts2 = node.put('b', 2)
    ts3 = node.put('c', 3)
    assert ts1 < ts2 < ts3, f'Clocks not monotonic: {ts1}, {ts2}, {ts3}'
    print('Test Lamport clock passed')

def test_idempotency():
    cluster = MultiLeaderCluster(['a', 'b'])
    cluster.node('a').put('key', 'val')
    cluster.sync()
    cluster.sync()
    assert cluster.node('b').get('key') == 'val'
    assert len(cluster.node('b').conflict_log) == 0
    print('Test idempotency passed')

def test_lww_tiebreak():
    cluster = MultiLeaderCluster(['a', 'b'])
    cluster.node('a').put('k', 'from_a')
    cluster.node('b').put('k', 'from_b')
    # Both have ts=1, so node_id 'b' > 'a' should win
    cluster.sync()
    assert cluster.all_converged()
    assert cluster.node('a').get('k') == 'from_b', f'got {cluster.node("a").get("k")}'
    assert cluster.node('b').get('k') == 'from_b'
    print('Test LWW tiebreak passed')

def test_conflict_logging():
    cluster = MultiLeaderCluster(['a', 'b'])
    cluster.node('a').put('k', 'va')
    cluster.node('b').put('k', 'vb')
    cluster.sync()
    conflicts = cluster.node('a').conflict_log + cluster.node('b').conflict_log
    assert len(conflicts) >= 1
    c = conflicts[0]
    assert c.key == 'k'
    assert c.resolved_by == ConflictStrategy.LAST_WRITE_WINS
    print('Test conflict logging passed')

if __name__ == '__main__':
    test_lww()
    test_custom_merge()
    test_ring()
    test_tombstone()
    test_lamport_clock()
    test_idempotency()
    test_lww_tiebreak()
    test_conflict_logging()
    print('\nAll tests passed!')
