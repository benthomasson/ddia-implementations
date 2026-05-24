"""Tests for the Kafka-style partitioned append-only log."""

import shutil
import tempfile

# Add implementer directory to path

from partitioned_log import Broker, Producer, Consumer, Message, RecordMetadata, Topic


def test_basic_produce_consume():
    """Test 1: Basic produce and consume - send messages, poll them, verify content and order."""
    broker = Broker()
    broker.create_topic('test', num_partitions=1)
    producer = Producer(broker)

    meta1 = producer.send('test', value='hello', partition=0)
    meta2 = producer.send('test', value='world', partition=0)

    assert isinstance(meta1, RecordMetadata)
    assert meta1.topic == 'test'
    assert meta1.partition == 0
    assert meta1.offset == 0
    assert meta2.offset == 1

    consumer = Consumer(broker, consumer_id='c1')
    consumer.assign([('test', 0)])
    msgs = consumer.poll()

    assert len(msgs) == 2
    assert msgs[0].value == 'hello'
    assert msgs[1].value == 'world'
    assert msgs[0].offset == 0
    assert msgs[1].offset == 1

    # Polling again returns empty (no new messages - test 13)
    msgs2 = consumer.poll()
    assert len(msgs2) == 0
    print("PASS: test_basic_produce_consume")


def test_key_partitioning():
    """Test 2: Messages with same key go to same partition."""
    broker = Broker()
    broker.create_topic('keyed', num_partitions=4)
    producer = Producer(broker)

    results = []
    for _ in range(10):
        results.append(producer.send('keyed', value='v', key='same-key'))

    partitions = {r.partition for r in results}
    assert len(partitions) == 1, f"Expected all same partition, got {partitions}"

    # Different keys may go to different partitions
    m1 = producer.send('keyed', value='v', key='key-a')
    m2 = producer.send('keyed', value='v', key='key-b')
    # At minimum, the hash is deterministic
    m1b = producer.send('keyed', value='v', key='key-a')
    assert m1.partition == m1b.partition
    print("PASS: test_key_partitioning")


def test_round_robin():
    """Test 3: Messages without keys distribute evenly via round-robin."""
    broker = Broker()
    broker.create_topic('rr', num_partitions=3)
    producer = Producer(broker)

    for i in range(9):
        producer.send('rr', value=f'msg_{i}')

    topic = broker.get_topic('rr')
    sizes = [topic.partition_size(p) for p in range(3)]
    assert sizes == [3, 3, 3], f"Expected [3,3,3], got {sizes}"
    print("PASS: test_round_robin")


def test_consumer_groups():
    """Test 4: Two consumers in a group each get a subset of partitions with no overlap."""
    broker = Broker()
    broker.create_topic('events', num_partitions=3)
    producer = Producer(broker)

    for i in range(6):
        producer.send('events', value=f'msg_{i}', partition=i % 3)

    c1 = Consumer(broker, group_id='g1', consumer_id='c1')
    c2 = Consumer(broker, group_id='g1', consumer_id='c2')
    c1.subscribe(['events'])
    c2.subscribe(['events'])

    assert len(c1.assignment) + len(c2.assignment) == 3
    assert set(c1.assignment).isdisjoint(set(c2.assignment))

    msgs1 = c1.poll()
    msgs2 = c2.poll()
    assert len(msgs1) + len(msgs2) == 6
    print("PASS: test_consumer_groups")


def test_rebalancing():
    """Test 5: Adding/removing a consumer triggers reassignment."""
    broker = Broker()
    broker.create_topic('t', num_partitions=4)

    c1 = Consumer(broker, group_id='g', consumer_id='c1')
    c1.subscribe(['t'])
    assert len(c1.assignment) == 4  # sole consumer gets all

    c2 = Consumer(broker, group_id='g', consumer_id='c2')
    c2.subscribe(['t'])
    assert len(c1.assignment) + len(c2.assignment) == 4
    assert len(c1.assignment) == 2
    assert len(c2.assignment) == 2

    c2.close()
    assert len(c1.assignment) == 4  # c1 gets everything back
    print("PASS: test_rebalancing")


def test_independent_groups():
    """Test 6: Two groups consume the same messages independently."""
    broker = Broker()
    broker.create_topic('shared', num_partitions=1)
    producer = Producer(broker)
    producer.send('shared', value='msg1', partition=0)
    producer.send('shared', value='msg2', partition=0)

    c1 = Consumer(broker, group_id='group-a', consumer_id='c1')
    c1.subscribe(['shared'])
    c2 = Consumer(broker, group_id='group-b', consumer_id='c2')
    c2.subscribe(['shared'])

    msgs1 = c1.poll()
    msgs2 = c2.poll()
    assert len(msgs1) == 2
    assert len(msgs2) == 2
    assert [m.value for m in msgs1] == [m.value for m in msgs2]
    print("PASS: test_independent_groups")


def test_offset_commit_restore():
    """Test 7: Commit offsets, create new consumer with same group, resumes from committed position."""
    broker = Broker()
    broker.create_topic('log', num_partitions=1)
    producer = Producer(broker)
    for i in range(5):
        producer.send('log', value=f'msg_{i}', partition=0)

    c1 = Consumer(broker, group_id='grp', consumer_id='c1')
    c1.subscribe(['log'])
    msgs = c1.poll(max_messages=3)  # read first 3
    assert len(msgs) == 3
    c1.commit()
    committed = c1.committed()
    assert committed[('log', 0)] == 3
    c1.close()

    # New consumer in same group should resume from offset 3
    c2 = Consumer(broker, group_id='grp', consumer_id='c2')
    c2.subscribe(['log'])
    msgs2 = c2.poll()
    assert len(msgs2) == 2
    assert msgs2[0].value == 'msg_3'
    assert msgs2[1].value == 'msg_4'
    print("PASS: test_offset_commit_restore")


def test_seek_replay():
    """Test 8: Rewind to offset 0 and re-consume all messages."""
    broker = Broker()
    broker.create_topic('replay', num_partitions=2)
    producer = Producer(broker)
    producer.send('replay', value='a', partition=0)
    producer.send('replay', value='b', partition=0)
    producer.send('replay', value='c', partition=1)

    consumer = Consumer(broker, consumer_id='c1')
    consumer.assign([('replay', 0), ('replay', 1)])
    msgs = consumer.poll()
    assert len(msgs) == 3

    # Seek back to beginning
    consumer.seek('replay', 0, 0)
    consumer.seek('replay', 1, 0)
    replayed = consumer.poll()
    assert len(replayed) == 3
    assert replayed[0].value == 'a'
    print("PASS: test_seek_replay")


def test_auto_commit_and_offset_reset():
    """Test 9+10: Auto-commit and auto_offset_reset."""
    broker = Broker()
    broker.create_topic('ac', num_partitions=1)
    producer = Producer(broker)
    producer.send('ac', value='x', partition=0)

    # auto_commit consumer
    c1 = Consumer(broker, group_id='ag', consumer_id='c1', auto_commit=True)
    c1.subscribe(['ac'])
    c1.poll()
    assert len(c1.committed()) > 0  # offsets auto-committed

    # auto_offset_reset='latest' - should see no existing messages
    producer.send('ac', value='y', partition=0)
    c2 = Consumer(broker, group_id='ag2', consumer_id='c2', auto_offset_reset='latest')
    c2.subscribe(['ac'])
    msgs = c2.poll()
    assert len(msgs) == 0, f"Expected 0 with latest reset, got {len(msgs)}"
    print("PASS: test_auto_commit_and_offset_reset")


def test_retention_and_compaction():
    """Test 11+12: Retention removes old messages; compaction keeps latest per key."""
    broker = Broker()
    # Retention
    broker.create_topic('metrics', num_partitions=1, max_log_size=5)
    producer = Producer(broker)
    for i in range(10):
        producer.send('metrics', value=f'metric_{i}', partition=0)
    removed = broker.enforce_retention('metrics')
    assert removed == 5
    topic = broker.get_topic('metrics')
    assert topic.earliest_offset(0) == 5
    assert topic.partition_size(0) == 5

    # Compaction
    broker.create_topic('compact', num_partitions=1)
    for i in range(5):
        producer.send('compact', value=f'val_{i}', key='same', partition=0)
    producer.send('compact', value='null-key', partition=0)  # no key - retained
    result = broker.compact('compact')
    assert result['messages_removed'] == 4  # only latest 'same' + null-key kept
    assert result['messages_retained'] == 2
    print("PASS: test_retention_and_compaction")


def test_batch_produce_and_multiple_topics():
    """Test 14+15: Batch produce and multiple topics."""
    broker = Broker()
    broker.create_topic('batch', num_partitions=1)
    producer = Producer(broker)

    batch = [{'value': f'b{i}', 'partition': 0} for i in range(5)]
    results = producer.send_batch('batch', batch)
    assert len(results) == 5
    assert all(isinstance(r, RecordMetadata) for r in results)
    assert [r.offset for r in results] == [0, 1, 2, 3, 4]

    # Multiple topics
    broker.create_topic('topicA', num_partitions=1)
    broker.create_topic('topicB', num_partitions=1)
    producer.send('topicA', value='a', partition=0)
    producer.send('topicB', value='b', partition=0)

    ca = Consumer(broker, consumer_id='ca')
    ca.assign([('topicA', 0)])
    cb = Consumer(broker, consumer_id='cb')
    cb.assign([('topicB', 0)])
    assert ca.poll()[0].value == 'a'
    assert cb.poll()[0].value == 'b'
    print("PASS: test_batch_produce_and_multiple_topics")


def test_more_consumers_than_partitions():
    """Test 17: More consumers than partitions - some idle."""
    broker = Broker()
    broker.create_topic('small', num_partitions=1)

    c1 = Consumer(broker, group_id='g', consumer_id='c1')
    c2 = Consumer(broker, group_id='g', consumer_id='c2')
    c3 = Consumer(broker, group_id='g', consumer_id='c3')
    c1.subscribe(['small'])
    c2.subscribe(['small'])
    c3.subscribe(['small'])

    total = len(c1.assignment) + len(c2.assignment) + len(c3.assignment)
    assert total == 1
    idle = sum(1 for c in [c1, c2, c3] if len(c.assignment) == 0)
    assert idle == 2
    print("PASS: test_more_consumers_than_partitions")


def test_disk_persistence():
    """Test 8 (optional): Disk persistence - write, reload, verify."""
    tmpdir = tempfile.mkdtemp()
    try:
        broker = Broker(persist_dir=tmpdir)
        broker.create_topic('persist', num_partitions=2)
        producer = Producer(broker)
        producer.send('persist', value='hello', partition=0)
        producer.send('persist', value='world', partition=1)

        c = Consumer(broker, group_id='pg', consumer_id='c1')
        c.subscribe(['persist'])
        c.poll()
        c.commit()

        # Reload from disk
        broker2 = Broker(persist_dir=tmpdir)
        topic = broker2.get_topic('persist')
        assert topic.partition_size(0) == 1
        assert topic.partition_size(1) == 1

        # Committed offsets restored
        c2 = Consumer(broker2, group_id='pg', consumer_id='c2')
        c2.subscribe(['persist'])
        msgs = c2.poll()
        assert len(msgs) == 0  # already consumed and committed
        print("PASS: test_disk_persistence")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    test_basic_produce_consume()
    test_key_partitioning()
    test_round_robin()
    test_consumer_groups()
    test_rebalancing()
    test_independent_groups()
    test_offset_commit_restore()
    test_seek_replay()
    test_auto_commit_and_offset_reset()
    test_retention_and_compaction()
    test_batch_produce_and_multiple_topics()
    test_more_consumers_than_partitions()
    test_disk_persistence()
    print("\n=== ALL TESTS PASSED ===")
