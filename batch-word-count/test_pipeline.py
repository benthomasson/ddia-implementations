"""Quick smoke tests for the pipeline."""
from pipeline import *
import tempfile, os

text = [
    "the quick brown fox jumps over the lazy dog",
    "the fox jumped over the lazy cat",
    "the dog barked at the cat",
]

# Classic word count
pipeline = Pipeline()
pipeline.add_stage(ReadLines(text))
pipeline.add_stage(Tokenize(lowercase=True, strip_punctuation=True))
pipeline.add_stage(Count())
pipeline.add_stage(Sort(by="value", descending=True))

result = pipeline.run()
print("Word count result:", result)
assert result[0] == ("the", 6), f"Expected (the, 6), got {result[0]}"
assert dict(result)["fox"] == 2

# Top 5
pipeline2 = Pipeline()
pipeline2.add_stage(ReadLines(text))
pipeline2.add_stage(Tokenize())
pipeline2.add_stage(Count())
pipeline2.add_stage(TopN(5, by="value"))
top5 = pipeline2.run()
print("Top 5:", top5)
assert len(top5) == 5
assert top5[0][1] >= top5[4][1]

# Filter
pipeline3 = Pipeline()
pipeline3.add_stage(ReadLines(text))
pipeline3.add_stage(Tokenize())
pipeline3.add_stage(Count())
pipeline3.add_stage(Filter(lambda record: record[1] >= 2))
pipeline3.add_stage(Sort(by="key"))
filtered = pipeline3.run()
print("Filtered:", filtered)
assert all(count >= 2 for _, count in filtered)

# Partition
with tempfile.TemporaryDirectory() as tmpdir:
    pipeline4 = Pipeline()
    pipeline4.add_stage(ReadLines(text))
    pipeline4.add_stage(Tokenize())
    pipeline4.add_stage(Count())
    pipeline4.add_stage(Partition(partition_fn=lambda r: r[0][0], output_dir=tmpdir))
    pipeline4.run()
    assert os.path.exists(os.path.join(tmpdir, "t")), f"Missing partition t, found: {os.listdir(tmpdir)}"

# External sort
pipeline5 = Pipeline()
pipeline5.add_stage(ReadLines(text))
pipeline5.add_stage(Tokenize())
pipeline5.add_stage(Count())
pipeline5.add_stage(Sort(by="key", memory_limit=3))
result5 = pipeline5.run()
keys = [k for k, v in result5]
assert keys == sorted(keys), f"Not sorted: {keys}"
print("External sort result:", result5)

# Stats
stats = pipeline.stats
assert stats.total_records_in > 0
assert len(stats.stages) == 4
print("Stats stages:", stats.stages)

# Lazy
pipeline6 = Pipeline()
pipeline6.add_stage(ReadLines(text))
pipeline6.add_stage(Tokenize())
lazy_iter = pipeline6.run_lazy()
first = next(lazy_iter)
print("Lazy first:", first)

# FlatMap
pipeline7 = Pipeline()
pipeline7.add_stage(ReadLines(["hello world"]))
pipeline7.add_stage(FlatMap(lambda r: [(r[0], w) for w in r[1].split()]))
fm_result = pipeline7.run()
print("FlatMap:", fm_result)

# Empty input
pipeline8 = Pipeline()
pipeline8.add_stage(ReadLines([]))
pipeline8.add_stage(Tokenize())
pipeline8.add_stage(Count())
empty_result = pipeline8.run()
assert empty_result == []

# File input
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write("hello world\nfoo bar baz\n")
    tmp_path = f.name
try:
    pipeline9 = Pipeline()
    pipeline9.add_stage(ReadLines(tmp_path))
    pipeline9.add_stage(Tokenize())
    pipeline9.add_stage(Count())
    pipeline9.add_stage(Sort(by="key"))
    file_result = pipeline9.run()
    print("File input result:", file_result)
    assert ("bar", 1) in file_result
    assert ("hello", 1) in file_result
finally:
    os.unlink(tmp_path)

# run_to_file
with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
    out_path = f.name
try:
    pipeline10 = Pipeline()
    pipeline10.add_stage(ReadLines(text))
    pipeline10.add_stage(Tokenize())
    pipeline10.add_stage(Count())
    pipeline10.add_stage(Sort(by="value", descending=True))
    pipeline10.run_to_file(out_path)
    with open(out_path) as f:
        lines = f.readlines()
    print(f"run_to_file wrote {len(lines)} lines")
    assert len(lines) > 0
finally:
    os.unlink(out_path)

print()
print("ALL TESTS PASSED")
