"""Single-machine MapReduce framework."""

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Callable, Optional


@dataclass
class JobStats:
    """Statistics from a MapReduce job run."""
    map_input_records: int = 0
    map_output_records: int = 0
    reduce_output_records: int = 0
    num_map_workers: int = 0
    num_reduce_workers: int = 0
    elapsed_seconds: float = 0.0


class MapReduceJob:
    """A single MapReduce job with map, shuffle/sort, and reduce phases."""

    def __init__(self,
                 mapper: Callable[[Any, Any], list[tuple[Any, Any]]],
                 reducer: Callable[[Any, list[Any]], list[tuple[Any, Any]]],
                 num_mappers: int = 4,
                 num_reducers: int = 2,
                 combiner: Optional[Callable[[Any, list[Any]], list[tuple[Any, Any]]]] = None,
                 fault_tolerant: bool = False):
        self.mapper = mapper
        self.reducer = reducer
        self.num_mappers = num_mappers
        self.num_reducers = num_reducers
        self.combiner = combiner
        self.fault_tolerant = fault_tolerant
        self._stats = JobStats()

    @property
    def stats(self) -> JobStats:
        return self._stats

    def run(self, input_data) -> list[tuple[Any, Any]]:
        """Execute the MapReduce job. Input can be a list of (key, value) pairs or a file path."""
        start = time.time()
        self._stats = JobStats(num_map_workers=self.num_mappers, num_reduce_workers=self.num_reducers)

        # Handle file input
        if isinstance(input_data, str):
            with open(input_data, 'r') as f:
                input_data = [(i, line.rstrip('\n')) for i, line in enumerate(f, 1)]

        self._stats.map_input_records = len(input_data)

        # Split input into chunks for map workers
        chunks = self._split_input(input_data, self.num_mappers)

        tmp_dir = tempfile.mkdtemp(prefix='mapreduce_')
        try:
            # Map phase
            for mapper_id, chunk in enumerate(chunks):
                self._run_mapper(mapper_id, chunk, tmp_dir)

            # Shuffle/sort + Reduce phase
            results = []
            for partition in range(self.num_reducers):
                partition_results = self._run_reducer(partition, tmp_dir)
                results.extend(partition_results)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        results.sort(key=lambda x: x[0])
        self._stats.reduce_output_records = len(results)
        self._stats.elapsed_seconds = time.time() - start
        return results

    def _split_input(self, data, n):
        """Split data into n roughly equal chunks."""
        if n <= 0 or not data:
            return [data] if data else []
        chunks = []
        size = len(data)
        base = size // n
        remainder = size % n
        start = 0
        for i in range(n):
            end = start + base + (1 if i < remainder else 0)
            if start < size:
                chunks.append(data[start:end])
            start = end
        return chunks if chunks else [data] if data else []

    def _run_mapper(self, mapper_id, chunk, tmp_dir):
        """Run map function on a chunk and write intermediate files."""
        # Collect outputs per partition
        partitions = {}
        for key, value in chunk:
            try:
                pairs = self.mapper(key, value)
            except Exception as e:
                if self.fault_tolerant:
                    continue
                raise
            for k, v in pairs:
                self._stats.map_output_records += 1
                p = hash(k) % self.num_reducers
                partitions.setdefault(p, []).append((k, v))

        # Apply combiner if present
        if self.combiner:
            for p in partitions:
                partitions[p] = self._apply_combiner(partitions[p])

        # Write intermediate files
        for p, pairs in partitions.items():
            path = os.path.join(tmp_dir, f'map-{mapper_id}-part-{p}.json')
            with open(path, 'w') as f:
                json.dump(pairs, f)

    def _apply_combiner(self, pairs):
        """Apply combiner to group-by-key within a partition."""
        pairs.sort(key=lambda x: x[0])
        combined = []
        for key, group in groupby(pairs, key=lambda x: x[0]):
            values = [v for _, v in group]
            combined.extend(self.combiner(key, values))
        return combined

    def _run_reducer(self, partition, tmp_dir):
        """Shuffle/sort and reduce for one partition."""
        # Read all intermediate files for this partition
        all_pairs = []
        for fname in os.listdir(tmp_dir):
            if fname.endswith(f'-part-{partition}.json'):
                with open(os.path.join(tmp_dir, fname), 'r') as f:
                    all_pairs.extend(json.load(f))

        # Sort by key
        all_pairs.sort(key=lambda x: x[0])

        # Group by key and reduce
        results = []
        for key, group in groupby(all_pairs, key=lambda x: x[0]):
            values = [v for _, v in group]
            try:
                output = self.reducer(key, values)
            except Exception as e:
                if self.fault_tolerant:
                    continue
                raise
            results.extend(output)
        return results


class MapReducePipeline:
    """Chain multiple MapReduce jobs in sequence."""

    def __init__(self):
        self._stages: list[MapReduceJob] = []
        self._stage_stats: list[JobStats] = []

    def add_stage(self, job: MapReduceJob) -> 'MapReducePipeline':
        """Add a MapReduce stage. Returns self for chaining."""
        self._stages.append(job)
        return self

    def run(self, input_data) -> list[tuple[Any, Any]]:
        """Run all stages in sequence."""
        self._stage_stats = []
        data = input_data
        for job in self._stages:
            data = job.run(data)
            self._stage_stats.append(job.stats)
        return data

    @property
    def stage_stats(self) -> list[JobStats]:
        return self._stage_stats
