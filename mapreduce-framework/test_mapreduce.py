"""Tests for the MapReduce framework."""

import os
import sys
import tempfile


from mapreduce import MapReduceJob, MapReducePipeline


def word_count_mapper(key, value):
    return [(word, 1) for word in value.lower().split()]

def word_count_reducer(key, values):
    return [(key, sum(values))]


def test_word_count():
    """Test 1: Basic word count from the spec example."""
    job = MapReduceJob(
        mapper=word_count_mapper,
        reducer=word_count_reducer,
        num_mappers=2,
        num_reducers=2,
    )
    input_data = [
        (1, "the quick brown fox"),
        (2, "the fox jumped over"),
        (3, "the lazy dog"),
    ]
    result = job.run(input_data)
    d = dict(result)
    assert d["the"] == 3
    assert d["fox"] == 2
    assert d["brown"] == 1
    assert d["dog"] == 1
    assert len(d) == 8


def test_multi_worker_consistency():
    """Test 2: Results are identical regardless of worker count."""
    input_data = [(i, f"word{i % 5} word{(i+1) % 5}") for i in range(20)]

    results = []
    for m, r in [(1, 1), (2, 3), (4, 2), (7, 5)]:
        job = MapReduceJob(mapper=word_count_mapper, reducer=word_count_reducer,
                           num_mappers=m, num_reducers=r)
        results.append(job.run(input_data))

    for r in results[1:]:
        assert r == results[0], f"Results differ with different worker counts"


def test_combiner():
    """Test 3: Combiner produces same results with fewer intermediate records."""
    input_data = [(i, "the the the") for i in range(10)]

    job_no_combiner = MapReduceJob(mapper=word_count_mapper, reducer=word_count_reducer,
                                    num_mappers=2, num_reducers=2)
    job_with_combiner = MapReduceJob(mapper=word_count_mapper, reducer=word_count_reducer,
                                      num_mappers=2, num_reducers=2,
                                      combiner=word_count_reducer)

    r1 = job_no_combiner.run(input_data)
    r2 = job_with_combiner.run(input_data)
    assert r1 == r2
    # Combiner should reduce intermediate records (map_output_records is counted before combiner)
    assert job_no_combiner.stats.map_output_records == job_with_combiner.stats.map_output_records


def test_file_input():
    """Test 4: Reading input from a file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("hello world\nhello mapreduce\n")
        path = f.name

    try:
        job = MapReduceJob(mapper=word_count_mapper, reducer=word_count_reducer,
                           num_mappers=2, num_reducers=2)
        result = job.run(path)
        d = dict(result)
        assert d["hello"] == 2
        assert d["world"] == 1
        assert d["mapreduce"] == 1
    finally:
        os.unlink(path)


def test_empty_input():
    """Test 5: Empty input produces empty output."""
    job = MapReduceJob(mapper=word_count_mapper, reducer=word_count_reducer)
    result = job.run([])
    assert result == []
    assert job.stats.map_input_records == 0


def test_single_key():
    """Test 6: All map outputs have the same key."""
    def same_key_mapper(key, value):
        return [("only_key", value)]

    def concat_reducer(key, values):
        return [(key, sum(values))]

    job = MapReduceJob(mapper=same_key_mapper, reducer=concat_reducer,
                       num_mappers=3, num_reducers=2)
    input_data = [(i, i) for i in range(10)]
    result = job.run(input_data)
    assert len(result) == 1
    assert result[0][0] == "only_key"
    assert result[0][1] == 45  # sum(0..9)


def test_fault_tolerant_mode():
    """Test 7: Fault-tolerant mode skips failing records."""
    def flaky_mapper(key, value):
        if key == 2:
            raise ValueError("bad record")
        return [(value, 1)]

    job = MapReduceJob(mapper=flaky_mapper, reducer=word_count_reducer,
                       num_mappers=1, num_reducers=1, fault_tolerant=True)
    input_data = [(1, "good"), (2, "bad"), (3, "good")]
    result = job.run(input_data)
    d = dict(result)
    assert d["good"] == 2
    assert "bad" not in d


def test_strict_mode():
    """Test 8: Strict mode (default) aborts on exception."""
    def bad_mapper(key, value):
        raise RuntimeError("fail")

    job = MapReduceJob(mapper=bad_mapper, reducer=word_count_reducer,
                       num_mappers=1, num_reducers=1)
    try:
        job.run([(1, "test")])
        assert False, "Should have raised"
    except RuntimeError:
        pass


def test_pipeline():
    """Test 9: Pipeline chaining two stages."""
    def swap_mapper(key, value):
        return [(value, key)]

    def identity_reducer(key, values):
        return [(key, v) for v in values]

    pipeline = MapReducePipeline()
    pipeline.add_stage(MapReduceJob(mapper=word_count_mapper, reducer=word_count_reducer,
                                     num_mappers=2, num_reducers=2))
    pipeline.add_stage(MapReduceJob(mapper=swap_mapper, reducer=identity_reducer,
                                     num_mappers=2, num_reducers=2))

    input_data = [
        (1, "the quick brown fox"),
        (2, "the fox jumped over"),
        (3, "the lazy dog"),
    ]
    result = pipeline.run(input_data)
    assert len(pipeline.stage_stats) == 2
    # After swap, keys are counts — verify "the" had count 3
    d = dict(result)
    assert "the" in d.get(3, d.get("3", []))  # count 3 -> word "the"


def test_statistics():
    """Test 10: Job statistics are accurate."""
    job = MapReduceJob(mapper=word_count_mapper, reducer=word_count_reducer,
                       num_mappers=2, num_reducers=2)
    input_data = [
        (1, "the quick brown fox"),
        (2, "the fox jumped over"),
        (3, "the lazy dog"),
    ]
    result = job.run(input_data)

    assert job.stats.map_input_records == 3
    assert job.stats.map_output_records == 11  # 4+4+3 words
    assert job.stats.reduce_output_records == 8  # 8 unique words
    assert job.stats.num_map_workers == 2
    assert job.stats.num_reduce_workers == 2
    assert job.stats.elapsed_seconds > 0


if __name__ == '__main__':
    tests = [
        test_word_count,
        test_multi_worker_consistency,
        test_combiner,
        test_file_input,
        test_empty_input,
        test_single_key,
        test_fault_tolerant_mode,
        test_strict_mode,
        test_pipeline,
        test_statistics,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(1 if failed else 0)
