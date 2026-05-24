"""Tests for Bloom filter implementations."""

import math
import pytest

from bloom_filter import BloomFilter, CountingBloomFilter, ScalableBloomFilter


# 1. No false negatives
def test_no_false_negatives():
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    items = [f"item-{i}" for i in range(1000)]
    for item in items:
        bf.add(item)
    for item in items:
        assert item in bf, f"False negative for {item}"
    assert len(bf) == 1000


# 2. FPR within 2x of target
def test_false_positive_rate():
    n, target = 5000, 0.01
    bf = BloomFilter(expected_items=n, false_positive_rate=target)
    for i in range(n):
        bf.add(f"member-{i}")
    fp = sum(1 for i in range(50000) if f"nonmember-{i}" in bf)
    actual_fpr = fp / 50000
    assert actual_fpr < target * 2, f"FPR {actual_fpr:.4f} exceeds 2x target {target}"


# 3. Optimal parameter calculation
def test_optimal_parameters():
    n, p = 10000, 0.01
    bf = BloomFilter(expected_items=n, false_positive_rate=p)
    ln2 = math.log(2)
    expected_m = math.ceil(-n * math.log(p) / (ln2 ** 2))
    expected_k = round((expected_m / n) * ln2)
    assert bf.bit_count == expected_m
    assert bf.hash_count == expected_k


# 4. Counting Bloom filter add/remove
def test_counting_bloom_filter():
    cbf = CountingBloomFilter(expected_items=100, false_positive_rate=0.01)
    cbf.add("apple")
    cbf.add("banana")
    assert "apple" in cbf
    assert "banana" in cbf
    assert len(cbf) == 2
    cbf.remove("apple")
    assert "apple" not in cbf
    assert "banana" in cbf
    assert len(cbf) == 1


# 5. Remove safety
def test_remove_nonexistent_raises():
    cbf = CountingBloomFilter(expected_items=100, false_positive_rate=0.01)
    with pytest.raises(ValueError):
        cbf.remove("never-added")


# 6. Serialization round-trip
def test_serialization_roundtrip():
    bf = BloomFilter(expected_items=500, false_positive_rate=0.01)
    items = [f"ser-{i}" for i in range(200)]
    for item in items:
        bf.add(item)
    data = bf.to_bytes()
    bf2 = BloomFilter.from_bytes(data)
    assert bf2.bit_count == bf.bit_count
    assert bf2.hash_count == bf.hash_count
    assert len(bf2) == 200
    for item in items:
        assert item in bf2


# 7. Union contains all items
def test_union():
    bfa = BloomFilter(expected_items=100, false_positive_rate=0.01)
    bfb = BloomFilter(expected_items=100, false_positive_rate=0.01)
    bfa.add("x")
    bfb.add("y")
    bfu = bfa.union(bfb)
    assert "x" in bfu
    assert "y" in bfu


# 8. estimate_count within 10%
def test_estimate_count():
    bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
    for i in range(5000):
        bf.add(f"est-{i}")
    est = bf.estimate_count()
    assert abs(est - 5000) / 5000 < 0.1, f"Estimate {est} too far from 5000"


# 9. Edge cases
def test_empty_filter():
    bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
    assert "anything" not in bf
    assert len(bf) == 0
    assert bf.estimate_count() == 0.0
    assert bf.bit_array_density() == 0.0


def test_determinism():
    bf1 = BloomFilter(expected_items=100, false_positive_rate=0.01)
    bf2 = BloomFilter(expected_items=100, false_positive_rate=0.01)
    for w in ["a", "b", "c"]:
        bf1.add(w)
        bf2.add(w)
    assert bf1.to_bytes() == bf2.to_bytes()
