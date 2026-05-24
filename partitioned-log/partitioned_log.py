"""Kafka-style partitioned append-only log."""

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """A message in the log."""
    key: Optional[str]
    value: object
    timestamp: float
    headers: Optional[dict] = None
    topic: Optional[str] = None
    partition: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class RecordMetadata:
    """Acknowledgment returned after producing a message."""
    topic: str
    partition: int
    offset: int
    timestamp: float


class Topic:
    """A named group of partitions."""

    def __init__(self, name: str, num_partitions: int,
                 retention_ms: int = -1, max_log_size: int = -1):
        if num_partitions < 1 or num_partitions > 128:
            raise ValueError("num_partitions must be between 1 and 128")
        self._name = name
        self._num_partitions = num_partitions
        self.retention_ms = retention_ms
        self.max_log_size = max_log_size
        # Per-partition storage
        self._partitions: list[list[Message]] = [[] for _ in range(num_partitions)]
        self._base_offsets: list[int] = [0] * num_partitions

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def append(self, partition: int, message: Message) -> int:
        """Append a message to a partition. Returns the assigned offset."""
        offset = self._base_offsets[partition] + len(self._partitions[partition])
        message.topic = self._name
        message.partition = partition
        message.offset = offset
        self._partitions[partition].append(message)
        return offset

    def read(self, partition: int, offset: int, max_messages: int) -> list[Message]:
        """Read messages from a partition starting at offset."""
        base = self._base_offsets[partition]
        log = self._partitions[partition]
        if offset < base:
            offset = base
        start = offset - base
        if start >= len(log):
            return []
        end = min(start + max_messages, len(log))
        return log[start:end]

    def partition_size(self, partition: int) -> int:
        return len(self._partitions[partition])

    def earliest_offset(self, partition: int) -> int:
        return self._base_offsets[partition]

    def latest_offset(self, partition: int) -> int:
        return self._base_offsets[partition] + len(self._partitions[partition])

    def truncate(self, partition: int, count: int):
        """Remove count messages from the beginning of a partition."""
        if count <= 0:
            return
        log = self._partitions[partition]
        actual = min(count, len(log))
        self._partitions[partition] = log[actual:]
        self._base_offsets[partition] += actual

    def compact_partition(self, partition: int) -> int:
        """Keep only the latest message per key. Returns number removed."""
        log = self._partitions[partition]
        if not log:
            return 0
        # Find last occurrence of each key
        last_by_key: dict[str, int] = {}
        for i, msg in enumerate(log):
            if msg.key is not None:
                last_by_key[msg.key] = i
        keep_indices = set()
        for i, msg in enumerate(log):
            if msg.key is None:
                keep_indices.add(i)
            elif last_by_key[msg.key] == i:
                keep_indices.add(i)
        new_log = [log[i] for i in sorted(keep_indices)]
        removed = len(log) - len(new_log)
        if removed > 0:
            # Update base_offset: the new base is the offset of the first retained message
            if new_log:
                self._base_offsets[partition] = new_log[0].offset
            else:
                self._base_offsets[partition] = self._base_offsets[partition] + len(log)
            self._partitions[partition] = new_log
        return removed


class Producer:
    """Publishes messages to topics."""

    def __init__(self, broker: 'Broker'):
        self._broker = broker
        self._rr_counters: dict[str, int] = {}  # topic -> round-robin counter

    def send(self, topic: str, value: object, key: str | None = None,
             headers: dict | None = None, partition: int | None = None) -> RecordMetadata:
        topic_obj = self._broker.get_topic(topic)
        ts = time.time()

        if partition is not None:
            p = partition
        elif key is not None:
            h = int(hashlib.md5(key.encode()).hexdigest(), 16)
            p = h % topic_obj.num_partitions
        else:
            counter = self._rr_counters.get(topic, 0)
            p = counter % topic_obj.num_partitions
            self._rr_counters[topic] = counter + 1

        msg = Message(key=key, value=value, timestamp=ts, headers=headers)
        offset = topic_obj.append(p, msg)
        self._broker._persist_message(topic, p, msg)
        return RecordMetadata(topic=topic, partition=p, offset=offset, timestamp=ts)

    def send_batch(self, topic: str, messages: list[dict]) -> list[RecordMetadata]:
        results = []
        for m in messages:
            meta = self.send(
                topic,
                value=m['value'],
                key=m.get('key'),
                headers=m.get('headers'),
                partition=m.get('partition'),
            )
            results.append(meta)
        return results

    def flush(self) -> None:
        pass


class Consumer:
    """Reads messages from partitions."""

    def __init__(self, broker: 'Broker', group_id: str | None = None,
                 consumer_id: str | None = None,
                 auto_commit: bool = False,
                 auto_offset_reset: str = "earliest"):
        self._broker = broker
        self._group_id = group_id
        self._consumer_id = consumer_id or str(uuid.uuid4())
        self._auto_commit = auto_commit
        self._auto_offset_reset = auto_offset_reset
        # (topic, partition) -> current offset
        self._offsets: dict[tuple[str, int], int] = {}
        self._assignment: list[tuple[str, int]] = []
        self._subscribed_topics: list[str] = []

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def subscribe(self, topics: list[str]) -> None:
        self._subscribed_topics = list(topics)
        if self._group_id:
            group = self._broker._get_or_create_group(self._group_id)
            if self not in [c for c in group._consumers]:
                group.add_consumer(self)
            else:
                group.rebalance()
        else:
            # Standalone: assign all partitions
            self._assignment = []
            for t in topics:
                topic_obj = self._broker.get_topic(t)
                for p in range(topic_obj.num_partitions):
                    self._assignment.append((t, p))
            self._init_offsets()

    def _init_offsets(self):
        """Initialize offsets for assigned partitions."""
        for tp in self._assignment:
            if tp in self._offsets:
                continue
            # Check for committed offset
            if self._group_id:
                committed = self._broker._committed_offsets.get(
                    (self._group_id, tp[0], tp[1]))
                if committed is not None:
                    self._offsets[tp] = committed
                    continue
            topic_obj = self._broker.get_topic(tp[0])
            if self._auto_offset_reset == "latest":
                self._offsets[tp] = topic_obj.latest_offset(tp[1])
            else:
                self._offsets[tp] = topic_obj.earliest_offset(tp[1])

    def assign(self, partitions: list[tuple[str, int]]) -> None:
        self._assignment = list(partitions)
        self._init_offsets()

    def poll(self, max_messages: int = 100) -> list[Message]:
        result = []
        remaining = max_messages
        for tp in self._assignment:
            if remaining <= 0:
                break
            topic_obj = self._broker.get_topic(tp[0])
            # Reset offset if it's before earliest
            earliest = topic_obj.earliest_offset(tp[1])
            if self._offsets.get(tp, 0) < earliest:
                self._offsets[tp] = earliest
            offset = self._offsets.get(tp, 0)
            msgs = topic_obj.read(tp[1], offset, remaining)
            if msgs:
                result.extend(msgs)
                self._offsets[tp] = msgs[-1].offset + 1
                remaining -= len(msgs)
        if self._auto_commit and result:
            self.commit()
        return result

    def seek(self, topic: str, partition: int, offset: int) -> None:
        tp = (topic, partition)
        topic_obj = self._broker.get_topic(topic)
        earliest = topic_obj.earliest_offset(partition)
        if offset < earliest:
            offset = earliest
        self._offsets[tp] = offset

    def commit(self) -> dict:
        result = {}
        for tp, offset in self._offsets.items():
            if self._group_id:
                self._broker._committed_offsets[
                    (self._group_id, tp[0], tp[1])] = offset
            result[tp] = offset
        self._broker._persist_offsets()
        return result

    def committed(self) -> dict:
        result = {}
        for tp in self._assignment:
            if self._group_id:
                key = (self._group_id, tp[0], tp[1])
                if key in self._broker._committed_offsets:
                    result[tp] = self._broker._committed_offsets[key]
        return result

    @property
    def assignment(self) -> list[tuple[str, int]]:
        return list(self._assignment)

    def close(self) -> None:
        if self._group_id:
            group = self._broker._get_or_create_group(self._group_id)
            group.remove_consumer(self._consumer_id)


class ConsumerGroup:
    """Manages partition assignment for a group of consumers."""

    def __init__(self, group_id: str, broker: 'Broker'):
        self._group_id = group_id
        self._broker = broker
        self._consumers: list[Consumer] = []

    def add_consumer(self, consumer: Consumer) -> None:
        self._consumers.append(consumer)
        self.rebalance()

    def remove_consumer(self, consumer_id: str) -> None:
        self._consumers = [c for c in self._consumers if c.consumer_id != consumer_id]
        self.rebalance()

    def get_assignment(self) -> dict[str, list[tuple[str, int]]]:
        result = {}
        for c in self._consumers:
            result[c.consumer_id] = list(c._assignment)
        return result

    def rebalance(self) -> None:
        if not self._consumers:
            return
        # Collect all subscribed topics across consumers
        all_topics: set[str] = set()
        for c in self._consumers:
            all_topics.update(c._subscribed_topics)
        # Collect all partitions
        all_partitions: list[tuple[str, int]] = []
        for t in sorted(all_topics):
            topic_obj = self._broker.get_topic(t)
            for p in range(topic_obj.num_partitions):
                all_partitions.append((t, p))
        # Sort consumers by ID
        sorted_consumers = sorted(self._consumers, key=lambda c: c.consumer_id)
        # Clear assignments
        for c in sorted_consumers:
            c._assignment = []
        # Round-robin assignment
        for i, tp in enumerate(all_partitions):
            consumer = sorted_consumers[i % len(sorted_consumers)]
            consumer._assignment.append(tp)
        # Init offsets for each consumer
        for c in sorted_consumers:
            c._init_offsets()


class Broker:
    """Central broker managing topics, groups, and offsets."""

    def __init__(self, persist_dir: str | None = None):
        self._topics: dict[str, Topic] = {}
        self._groups: dict[str, ConsumerGroup] = {}
        # (group_id, topic, partition) -> offset
        self._committed_offsets: dict[tuple[str, str, int], int] = {}
        self._persist_dir = persist_dir
        if persist_dir:
            self._load_from_disk()

    def create_topic(self, name: str, num_partitions: int,
                     retention_ms: int = -1, max_log_size: int = -1) -> Topic:
        topic = Topic(name, num_partitions, retention_ms, max_log_size)
        self._topics[name] = topic
        return topic

    def get_topic(self, name: str) -> Topic:
        if name not in self._topics:
            raise KeyError(f"Topic '{name}' not found")
        return self._topics[name]

    def list_topics(self) -> list[str]:
        return list(self._topics.keys())

    def compact(self, topic: str) -> dict:
        topic_obj = self.get_topic(topic)
        removed = 0
        retained = 0
        for p in range(topic_obj.num_partitions):
            removed += topic_obj.compact_partition(p)
            retained += topic_obj.partition_size(p)
        return {"messages_removed": removed, "messages_retained": retained}

    def enforce_retention(self, topic: str) -> int:
        topic_obj = self.get_topic(topic)
        total_removed = 0
        now = time.time() * 1000  # ms

        for p in range(topic_obj.num_partitions):
            # Size-based retention
            if topic_obj.max_log_size > 0:
                excess = topic_obj.partition_size(p) - topic_obj.max_log_size
                if excess > 0:
                    topic_obj.truncate(p, excess)
                    total_removed += excess

            # Time-based retention
            if topic_obj.retention_ms > 0:
                count = 0
                log = topic_obj._partitions[p]
                cutoff = (now - topic_obj.retention_ms) / 1000.0
                for msg in log:
                    if msg.timestamp < cutoff:
                        count += 1
                    else:
                        break
                if count > 0:
                    topic_obj.truncate(p, count)
                    total_removed += count

        return total_removed

    def _get_or_create_group(self, group_id: str) -> ConsumerGroup:
        if group_id not in self._groups:
            self._groups[group_id] = ConsumerGroup(group_id, self)
        return self._groups[group_id]

    # Persistence methods
    def _persist_message(self, topic: str, partition: int, message: Message):
        if not self._persist_dir:
            return
        os.makedirs(self._persist_dir, exist_ok=True)
        path = os.path.join(self._persist_dir, f"{topic}_{partition}.jsonl")
        record = {
            "key": message.key,
            "value": message.value,
            "timestamp": message.timestamp,
            "headers": message.headers,
            "offset": message.offset,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _persist_offsets(self):
        if not self._persist_dir:
            return
        os.makedirs(self._persist_dir, exist_ok=True)
        path = os.path.join(self._persist_dir, "offsets.json")
        data = {f"{g}|{t}|{p}": o for (g, t, p), o in self._committed_offsets.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_from_disk(self):
        if not self._persist_dir or not os.path.isdir(self._persist_dir):
            return
        # Load offsets
        offsets_path = os.path.join(self._persist_dir, "offsets.json")
        if os.path.exists(offsets_path):
            with open(offsets_path) as f:
                data = json.load(f)
            for key, offset in data.items():
                g, t, p = key.split("|")
                self._committed_offsets[(g, t, int(p))] = offset
        # Load partition data
        for fname in os.listdir(self._persist_dir):
            if not fname.endswith(".jsonl"):
                continue
            parts = fname[:-6].rsplit("_", 1)
            if len(parts) != 2:
                continue
            topic_name, part_str = parts
            partition = int(part_str)
            # Ensure topic exists
            if topic_name not in self._topics:
                # Count partition files to determine num_partitions
                count = sum(1 for f in os.listdir(self._persist_dir)
                            if f.startswith(topic_name + "_") and f.endswith(".jsonl"))
                self.create_topic(topic_name, max(count, partition + 1))
            topic_obj = self._topics[topic_name]
            # Extend partitions if needed
            while topic_obj.num_partitions <= partition:
                topic_obj._partitions.append([])
                topic_obj._base_offsets.append(0)
                topic_obj._num_partitions += 1
            path = os.path.join(self._persist_dir, fname)
            with open(path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    msg = Message(
                        key=record["key"],
                        value=record["value"],
                        timestamp=record["timestamp"],
                        headers=record.get("headers"),
                    )
                    topic_obj.append(partition, msg)
