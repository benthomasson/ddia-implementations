"""Tests for the batch processing pipeline."""
import json
import os
import sys
import tempfile


from pipeline import (
    Pipeline, ReadLines, Tokenize, Count, Sort, Filter, TopN, Partition, FlatMap
)


TEXT = [
    "the quick brown fox jumps over the lazy dog",
    "the fox jumped over the lazy cat",
    "the dog barked at the cat",
]


def test_word_count_example():
    """Test the classic word count from the spec examples."""
    p = Pipeline()
    p.add_stage(ReadLines(TEXT))
    p.add_stage(Tokenize(lowercase=True, strip_punctuation=True))
    p.add_stage(Count())
    p.add_stage(Sort(by="value", descending=True))
    result = p.run()
    assert result[0] == ("the", 6), f"Expected ('the', 6), got {result[0]}"
    d = dict(result)
    assert d["fox"] == 2
    assert d["lazy"] == 2
    assert d["cat"] == 2
    assert d["dog"] == 2


def test_tokenize_options():
    """Test tokenization: lowercase, punctuation, min_length."""
    p = Pipeline()
    p.add_stage(ReadLines(["Hello, WORLD! I'm a test."]))
    p.add_stage(Tokenize(lowercase=True, strip_punctuation=True, min_length=2))
    result = p.run()
    words = [w for w, _ in result]
    assert "hello" in words
    assert "world" in words
    assert "i'm" in words or "im" in words  # depends on punctuation stripping
    # "a" should be filtered out by min_length=2
    assert "a" not in words


def test_sort_ascending_descending():
    """Test sort by key and value, ascending and descending."""
    data = [("b", 2), ("a", 3), ("c", 1)]

    # Sort by key ascending
    p = Pipeline()
    p.add_stage(ReadLines(["b 2", "a 3", "c 1"]))
    p.add_stage(FlatMap(lambda r: [(r[1].split()[0], int(r[1].split()[1]))]))
    p.add_stage(Sort(by="key", descending=False))
    result = p.run()
    keys = [k for k, v in result]
    assert keys == ["a", "b", "c"], f"Expected sorted keys, got {keys}"

    # Sort by value descending
    p2 = Pipeline()
    p2.add_stage(ReadLines(["b 2", "a 3", "c 1"]))
    p2.add_stage(FlatMap(lambda r: [(r[1].split()[0], int(r[1].split()[1]))]))
    p2.add_stage(Sort(by="value", descending=True))
    result2 = p2.run()
    vals = [v for k, v in result2]
    assert vals == [3, 2, 1], f"Expected [3,2,1], got {vals}"


def test_external_sort():
    """Test external sort with small memory_limit forces spill to disk."""
    p = Pipeline()
    p.add_stage(ReadLines(TEXT))
    p.add_stage(Tokenize())
    p.add_stage(Count())
    p.add_stage(Sort(by="key", memory_limit=3))
    result = p.run()
    keys = [k for k, v in result]
    assert keys == sorted(keys), f"Not sorted: {keys}"
    assert len(keys) > 3  # Ensure we had enough records to trigger external sort


def test_filter():
    """Test filter passes only matching records."""
    p = Pipeline()
    p.add_stage(ReadLines(TEXT))
    p.add_stage(Tokenize())
    p.add_stage(Count())
    p.add_stage(Filter(lambda r: r[1] >= 2))
    result = p.run()
    assert all(count >= 2 for _, count in result)
    assert any(k == "the" for k, _ in result)  # "the" appears 6 times


def test_topn():
    """Test TopN returns correct top results without full sort."""
    p = Pipeline()
    p.add_stage(ReadLines(TEXT))
    p.add_stage(Tokenize())
    p.add_stage(Count())
    p.add_stage(TopN(3, by="value"))
    result = p.run()
    assert len(result) == 3
    assert result[0][1] >= result[1][1] >= result[2][1]
    assert result[0] == ("the", 6)


def test_partition():
    """Test partition creates correct files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Pipeline()
        p.add_stage(ReadLines(TEXT))
        p.add_stage(Tokenize())
        p.add_stage(Count())
        p.add_stage(Partition(partition_fn=lambda r: r[0][0], output_dir=tmpdir))
        result = p.run()
        # "the" -> partition "t"
        assert os.path.exists(os.path.join(tmpdir, "t"))
        # Read partition and check content
        with open(os.path.join(tmpdir, "t")) as f:
            t_records = [tuple(json.loads(line)) for line in f]
        t_words = {r[0] for r in t_records}
        assert "the" in t_words
        # Records also yielded downstream
        assert len(result) > 0


def test_empty_input():
    """Test empty input produces empty output without errors."""
    p = Pipeline()
    p.add_stage(ReadLines([]))
    p.add_stage(Tokenize())
    p.add_stage(Count())
    result = p.run()
    assert result == []


def test_flatmap():
    """Test FlatMap one-to-many and one-to-zero transformations."""
    # One-to-many
    p = Pipeline()
    p.add_stage(ReadLines(["hello world"]))
    p.add_stage(FlatMap(lambda r: [(r[0], w) for w in r[1].split()]))
    result = p.run()
    assert len(result) == 2
    assert result[0] == (1, "hello")
    assert result[1] == (1, "world")

    # One-to-zero (filtering via FlatMap)
    p2 = Pipeline()
    p2.add_stage(ReadLines(["keep", "drop", "keep"]))
    p2.add_stage(FlatMap(lambda r: [r] if r[1] == "keep" else []))
    result2 = p2.run()
    assert len(result2) == 2


def test_pipeline_stats():
    """Test pipeline statistics are tracked."""
    p = Pipeline()
    p.add_stage(ReadLines(TEXT))
    p.add_stage(Tokenize())
    p.add_stage(Count())
    p.add_stage(Sort(by="value", descending=True))
    p.run()
    stats = p.stats
    assert stats is not None
    assert len(stats.stages) == 4
    assert stats.total_records_out > 0
    assert stats.total_elapsed_seconds > 0
    # ReadLines emits 3 lines
    assert stats.stages[0]['records_out'] == 3


def test_file_input():
    """Test reading from a file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("hello world\nfoo bar baz\n")
        tmp_path = f.name
    try:
        p = Pipeline()
        p.add_stage(ReadLines(tmp_path))
        p.add_stage(Tokenize())
        p.add_stage(Count())
        result = p.run()
        d = dict(result)
        assert d["hello"] == 1
        assert d["foo"] == 1
        assert len(d) == 5
    finally:
        os.unlink(tmp_path)


def test_records_in_counts():
    """records_in should match upstream stage's records_out, not be double-counted."""
    p = Pipeline()
    p.add_stage(ReadLines(TEXT))
    p.add_stage(Tokenize())
    p.add_stage(Count())
    p.add_stage(Sort(by="value", descending=True))
    p.run()
    s = p.stats.stages
    assert s[0]['records_in'] == 0
    assert s[1]['records_in'] == s[0]['records_out']
    assert s[2]['records_in'] == s[1]['records_out']
    assert s[3]['records_in'] == s[2]['records_out']


def test_external_sort_descending():
    """Descending external sort should produce correct order."""
    p = Pipeline()
    p.add_stage(ReadLines(TEXT))
    p.add_stage(Tokenize())
    p.add_stage(Count())
    p.add_stage(Sort(by="value", descending=True, memory_limit=3))
    result = p.run()
    vals = [v for k, v in result]
    assert vals == sorted(vals, reverse=True), f"Not descending: {vals}"
    assert result[0] == ("the", 6)
    # Also test descending by key
    p2 = Pipeline()
    p2.add_stage(ReadLines(TEXT))
    p2.add_stage(Tokenize())
    p2.add_stage(Count())
    p2.add_stage(Sort(by="key", descending=True, memory_limit=3))
    result2 = p2.run()
    keys = [k for k, v in result2]
    assert keys == sorted(keys, reverse=True), f"Not descending: {keys}"


def test_lazy_stats():
    """run_lazy() should populate stats after iterator is fully consumed."""
    p = Pipeline()
    p.add_stage(ReadLines(TEXT))
    p.add_stage(Tokenize())
    p.add_stage(Count())
    assert p.stats is None
    result = list(p.run_lazy())
    assert p.stats is not None
    assert p.stats.total_records_out == len(result)
    assert p.stats.total_records_out > 0


if __name__ == "__main__":
    tests = [
        test_word_count_example,
        test_tokenize_options,
        test_sort_ascending_descending,
        test_external_sort,
        test_filter,
        test_topn,
        test_partition,
        test_empty_input,
        test_flatmap,
        test_pipeline_stats,
        test_file_input,
        test_records_in_counts,
        test_external_sort_descending,
        test_lazy_stats,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
