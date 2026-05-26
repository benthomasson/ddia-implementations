"""Batch processing pipeline with composable stages inspired by Unix philosophy."""

import heapq
import json
import os
import re
import tempfile
import time
from collections import defaultdict


class PipelineStats:
    """Statistics from a pipeline run."""

    def __init__(self):
        self.stages = []
        self.total_records_in = 0
        self.total_records_out = 0
        self.total_elapsed_seconds = 0.0


class Stage:
    """Base class for pipeline stages."""

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.records_in = 0
        self.records_out = 0
        self.elapsed_seconds = 0.0

    def process(self, input_iter):
        raise NotImplementedError

    def _tracked_process(self, input_iter):
        """Wrap process with stats tracking."""
        start = time.monotonic()
        downstream = 0.0
        try:
            for record in self.process(input_iter):
                self.records_out += 1
                t = time.monotonic()
                yield record
                downstream += time.monotonic() - t
        finally:
            self.elapsed_seconds = time.monotonic() - start - downstream

    def _count_input(self, input_iter):
        """Wrap input iterator to count records."""
        for record in input_iter:
            self.records_in += 1
            yield record


class ReadLines(Stage):
    """Read lines from a file path or list of strings."""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def process(self, input_iter):
        if isinstance(self.source, str):
            with open(self.source, 'r') as f:
                for i, line in enumerate(f, 1):
                    yield (i, line.rstrip('\n'))
        else:
            for i, line in enumerate(self.source, 1):
                yield (i, line.rstrip('\n') if isinstance(line, str) else line)


class Tokenize(Stage):
    """Tokenize text into words, yielding (word, 1) pairs."""

    def __init__(self, lowercase=True, strip_punctuation=True, min_length=1, delimiter=None):
        super().__init__()
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.min_length = min_length
        self.delimiter = delimiter

    def process(self, input_iter):
        for key, text in self._count_input(input_iter):
            if self.delimiter:
                words = text.split(self.delimiter)
            else:
                words = text.split()
            for word in words:
                if self.strip_punctuation:
                    word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
                if self.lowercase:
                    word = word.lower()
                if len(word) >= self.min_length:
                    yield (word, 1)


class Count(Stage):
    """Group by key and sum values."""

    def process(self, input_iter):
        counts = defaultdict(int)
        for key, value in self._count_input(input_iter):
            counts[key] += value
        for item in counts.items():
            yield item


class Sort(Stage):
    """Sort records by key or value with external sort support."""

    def __init__(self, by="key", descending=False, memory_limit=10000):
        super().__init__()
        self.by = by
        self.descending = descending
        self.memory_limit = memory_limit

    def _sort_key(self, record):
        return record[0] if self.by == "key" else record[1]

    def process(self, input_iter):
        buffer = []
        chunk_files = []
        tmp_dir = None

        try:
            for record in self._count_input(input_iter):
                buffer.append(record)
                if len(buffer) >= self.memory_limit:
                    if tmp_dir is None:
                        tmp_dir = tempfile.mkdtemp()
                    chunk_path = os.path.join(tmp_dir, f"chunk_{len(chunk_files)}.jsonl")
                    buffer.sort(key=self._sort_key, reverse=self.descending)
                    with open(chunk_path, 'w') as f:
                        for rec in buffer:
                            f.write(json.dumps(rec) + '\n')
                    chunk_files.append(chunk_path)
                    buffer.clear()

            if not chunk_files:
                buffer.sort(key=self._sort_key, reverse=self.descending)
                yield from buffer
            else:
                if buffer:
                    chunk_path = os.path.join(tmp_dir, f"chunk_{len(chunk_files)}.jsonl")
                    buffer.sort(key=self._sort_key, reverse=self.descending)
                    with open(chunk_path, 'w') as f:
                        for rec in buffer:
                            f.write(json.dumps(rec) + '\n')
                    chunk_files.append(chunk_path)
                    buffer.clear()

                key_idx = 0 if self.by == "key" else 1
                descending = self.descending

                class KeyedRecord:
                    __slots__ = ('key', 'seq', 'record')
                    def __init__(self, record, seq):
                        self.key = record[key_idx]
                        self.seq = seq
                        self.record = record
                    def __lt__(self, other):
                        if self.key != other.key:
                            if descending:
                                return self.key > other.key
                            return self.key < other.key
                        if descending:
                            return self.seq > other.seq
                        return self.seq < other.seq

                def keyed_chunk(path, start_seq):
                    seq = start_seq
                    with open(path, 'r') as f:
                        for line in f:
                            rec = tuple(json.loads(line))
                            yield KeyedRecord(rec, seq)
                            seq += 1

                keyed_iters = [keyed_chunk(p, i * self.memory_limit) for i, p in enumerate(chunk_files)]
                for kr in heapq.merge(*keyed_iters):
                    yield kr.record
        finally:
            if tmp_dir:
                for f in chunk_files:
                    if os.path.exists(f):
                        os.remove(f)
                if os.path.exists(tmp_dir):
                    os.rmdir(tmp_dir)


class Filter(Stage):
    """Keep only records matching the predicate."""

    def __init__(self, predicate):
        super().__init__()
        self.predicate = predicate

    def process(self, input_iter):
        for record in self._count_input(input_iter):
            if self.predicate(record):
                yield record


class TopN(Stage):
    """Keep only the top N records using a min-heap."""

    def __init__(self, n, by="value"):
        super().__init__()
        self.n = n
        self.by = by

    def process(self, input_iter):
        key_idx = 0 if self.by == "key" else 1
        heap = []
        for record in self._count_input(input_iter):
            item = (record[key_idx], record)
            if len(heap) < self.n:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

        results = [item[1] for item in heap]
        results.sort(key=lambda r: r[key_idx], reverse=True)
        yield from results


class Partition(Stage):
    """Partition output into files based on a partition function."""

    def __init__(self, partition_fn, output_dir):
        super().__init__()
        self.partition_fn = partition_fn
        self.output_dir = output_dir

    def process(self, input_iter):
        os.makedirs(self.output_dir, exist_ok=True)
        open_files = {}
        try:
            for record in self._count_input(input_iter):
                partition_name = self.partition_fn(record)
                if partition_name not in open_files:
                    path = os.path.join(self.output_dir, partition_name)
                    open_files[partition_name] = open(path, 'w')
                open_files[partition_name].write(json.dumps(record) + '\n')
                yield record
        finally:
            for f in open_files.values():
                f.close()


class FlatMap(Stage):
    """Apply fn to each record, yielding zero or more output records."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def process(self, input_iter):
        for record in self._count_input(input_iter):
            for output in self.fn(record):
                yield output


class Pipeline:
    """Chains together multiple processing stages."""

    def __init__(self):
        self._stages = []
        self._stats = None

    def add_stage(self, stage):
        """Add a stage. Returns self for chaining."""
        self._stages.append(stage)
        return self

    def _build_iterator(self, input_data=None):
        """Build the chained iterator from all stages."""
        if not self._stages:
            return iter([])

        # Reset stats
        for stage in self._stages:
            stage.records_in = 0
            stage.records_out = 0
            stage.elapsed_seconds = 0.0

        first = self._stages[0]

        # If first stage is ReadLines, it ignores input_iter
        if isinstance(first, ReadLines):
            if input_data is not None:
                first.source = input_data
            it = first._tracked_process(iter([]))
        elif input_data is not None:
            read_stage = ReadLines(input_data)
            it = read_stage.process(iter([]))
            it = first._tracked_process(it)
        else:
            it = first._tracked_process(iter([]))

        for stage in self._stages[1:]:
            it = stage._tracked_process(it)

        return it

    def run(self, input_data=None):
        """Execute the pipeline, returning output as a list."""
        start = time.monotonic()
        try:
            result = list(self._build_iterator(input_data))
        finally:
            elapsed = time.monotonic() - start
            self._build_stats(elapsed)
        return result

    def run_to_file(self, output_path, input_data=None):
        """Execute the pipeline and write output to a file."""
        start = time.monotonic()
        try:
            with open(output_path, 'w') as f:
                for record in self._build_iterator(input_data):
                    f.write(json.dumps(record) + '\n')
        finally:
            elapsed = time.monotonic() - start
            self._build_stats(elapsed)

    def run_lazy(self, input_data=None):
        """Execute the pipeline lazily, yielding output records."""
        start = time.monotonic()
        try:
            yield from self._build_iterator(input_data)
        finally:
            elapsed = time.monotonic() - start
            self._build_stats(elapsed)

    def _build_stats(self, total_elapsed):
        stats = PipelineStats()
        for stage in self._stages:
            stats.stages.append({
                'name': stage.name,
                'records_in': stage.records_in,
                'records_out': stage.records_out,
                'elapsed_seconds': stage.elapsed_seconds,
            })
        if stats.stages:
            first = stats.stages[0]
            stats.total_records_in = first['records_in'] or first['records_out']
            stats.total_records_out = stats.stages[-1]['records_out']
        stats.total_elapsed_seconds = total_elapsed
        self._stats = stats

    @property
    def stats(self):
        """Return statistics from the most recent run."""
        return self._stats
