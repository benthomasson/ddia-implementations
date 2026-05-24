from partitioned_log import *

broker = Broker()
broker.create_topic('events', num_partitions=3)
producer = Producer(broker)

m1 = producer.send('events', value={'type': 'click'}, key='user1')
m2 = producer.send('events', value={'type': 'view'}, key='user1')
m3 = producer.send('events', value={'type': 'click'}, key='user2')
assert m1.partition == m2.partition, 'Same key should go to same partition'
print(f'Key partitioning OK: user1->p{m1.partition}, user2->p{m3.partition}')

c1 = Consumer(broker, group_id='g1', consumer_id='c1')
c2 = Consumer(broker, group_id='g1', consumer_id='c2')
c1.subscribe(['events'])
c2.subscribe(['events'])
assert len(c1.assignment) + len(c2.assignment) == 3
assert set(c1.assignment).isdisjoint(set(c2.assignment))
print(f'Consumer group OK: c1={c1.assignment}, c2={c2.assignment}')

msgs1 = c1.poll()
msgs2 = c2.poll()
print(f'Polled: c1={len(msgs1)}, c2={len(msgs2)}, total={len(msgs1)+len(msgs2)}')
assert len(msgs1) + len(msgs2) == 3

c3 = Consumer(broker, group_id='g2', consumer_id='c3')
c3.subscribe(['events'])
msgs3 = c3.poll()
assert len(msgs3) == 3
print(f'Independent group OK: {len(msgs3)} msgs')

for p in range(3):
    c3.seek('events', p, 0)
replayed = c3.poll()
assert len(replayed) == 3
print(f'Replay OK: {len(replayed)} msgs')

c1.commit()
committed = c1.committed()
print(f'Commit OK: {committed}')

producer.send('events', value={'b': 100}, key='acct1')
producer.send('events', value={'b': 200}, key='acct1')
producer.send('events', value={'b': 300}, key='acct1')
result = broker.compact('events')
print(f'Compaction OK: {result}')
assert result['messages_removed'] >= 2

broker.create_topic('metrics', num_partitions=1, max_log_size=5)
for i in range(10):
    producer.send('metrics', value=f'metric_{i}')
removed = broker.enforce_retention('metrics')
assert removed == 5
topic = broker.get_topic('metrics')
assert topic.earliest_offset(0) == 5
print(f'Retention OK: removed={removed}, earliest={topic.earliest_offset(0)}')

c4 = Consumer(broker, group_id='g3', consumer_id='c4', auto_commit=True)
c4.subscribe(['events'])
c4.poll()
assert len(c4.committed()) > 0
print('Auto-commit OK')

producer.send('events', value='new1')
c5 = Consumer(broker, group_id='g4', consumer_id='c5', auto_offset_reset='latest')
c5.subscribe(['events'])
msgs5 = c5.poll()
assert len(msgs5) == 0, f'Expected 0 msgs with latest reset, got {len(msgs5)}'
print('Latest offset reset OK')

broker.create_topic('rr', num_partitions=3)
for i in range(9):
    producer.send('rr', value=f'msg_{i}')
rr_topic = broker.get_topic('rr')
sizes = [rr_topic.partition_size(p) for p in range(3)]
assert sizes == [3, 3, 3], f'Expected even distribution, got {sizes}'
print(f'Round-robin OK: {sizes}')

broker.create_topic('small', num_partitions=1)
ca = Consumer(broker, group_id='g5', consumer_id='ca')
cb = Consumer(broker, group_id='g5', consumer_id='cb')
ca.subscribe(['small'])
cb.subscribe(['small'])
assert len(ca.assignment) + len(cb.assignment) == 1
print(f'Idle consumer OK: ca={ca.assignment}, cb={cb.assignment}')

metas = producer.send_batch('events', [{'value': 'b1'}, {'value': 'b2', 'key': 'k1'}])
assert len(metas) == 2
print(f'Batch OK: {len(metas)} messages')

c2.close()
assert len(c1.assignment) == 3
print('Close/rebalance OK')

c_empty = Consumer(broker, group_id='g6', consumer_id='c_empty')
c_empty.subscribe(['events'])
c_empty.poll()
msgs_empty = c_empty.poll()
assert len(msgs_empty) == 0
print('No new messages OK')

print()
print('ALL TESTS PASSED')
