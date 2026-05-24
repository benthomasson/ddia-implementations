"""Tests for Bloom filter implementations."""

import math
import random
import string
import pytest
from bloom_filter import BloomFilter, CountingBloomFilter, ScalableBloomFilter


# 1. Basic operations: no false negatives
def test_add_and_contains():
    bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
    items = ["hello", "world", "foo", "bar", "baz"]
    for item in items:
        bf.add(item)
    for item in items:
        assert item in bf
    assert len(bf) == 5


def test_no_false_negatives():
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    items = [f"item-{i}" for i in range(1000)]
    for item in items:
        bf.add(item)
    for item in items:
        assert item in bf, f"False negative for {item}"


# 2. False positive rate within 2x
def test_false_positive_rate():
    target_fpr = 0.01
    n = 5000
    bf = BloomFilter(expected_items=n, false_positive_rate=target_fpr)
    for i in range(n):
        bf.add(f"member-{i}")

    fp = 0
    tests = 50000
    for i in range(tests):
        if f"nonmember-{i}" in bf:
            fp += 1
    actual_fpr = fp / tests
    assert actual_fpr < target_fpr * 2, f"FPR {actual_fpr} exceeds 2x target {target_fpr}"


# 3. Optimal parameter calculation
def test_optimal_parameters():
    n, p = 10000, 0.01
    bf = BloomFilter(expected_items=n, false_positive_rate=p)
    ln2 = math.log(2)
    expected_m = math.ceil(-n * math.log(p) / (ln2 ** 2))
    expected_k = round((expected_m / n) * ln2)
    assert bf.bit_count == expected_m
    assert bf.hash_count == expected_k


def test_explicit_parameters():
    bf = BloomFilter(bit_size=1024, num_hashes=5)
    assert bf.bit_count == 1024
    assert bf.hash_count == 5


# 4. Counting Bloom filter add/remove/contains
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
def test_counting_remove_nonexistent():
    cbf = CountingBloomFilter(expected_items=100, false_positive_rate=0.01)
    with pytest.raises(ValueError):
        cbf.remove("never-added")


def test_counting_double_add_remove():
    cbf = CountingBloomFilter(expected_items=100, false_positive_rate=0.01)
    cbf.add("x")
    cbf.add("x")
    cbf.remove("x")
    assert "x" in cbf  # still present after one removal
    cbf.remove("x")
    assert "x" not in cbf


# 6. Serialization round-trip
def test_serialization():
    bf = BloomFilter(expected_items=500, false_positive_rate=0.01)
    items = [f"ser-{i}" for i in range(200)]
    for item in items:
        bf.add(item)

    data = bf.to_bytes()
    bf2 = BloomFilter.from_bytes(data)
    assert bf2.bit_count == bf.bit_count
    assert bf2.hash_count == bf.hash_count
    assert len(bf2) == len(bf)
    for item in items:
        assert item in bf2


# 7. Union
def test_union():
    bf_a = BloomFilter(expected_items=100, false_positive_rate=0.01)
    bf_b = BloomFilter(expected_items=100, false_positive_rate=0.01)
    bf_a.add("x")
    bf_a.add("y")
    bf_b.add("z")
    bf_b.add("w")

    bf_u = bf_a.union(bf_b)
    for item in ["x", "y", "z", "w"]:
        assert item in bf_u


def test_union_incompatible():
    bf_a = BloomFilter(bit_size=100, num_hashes=3)
    bf_b = BloomFilter(bit_size=200, num_hashes=3)
    with pytest.raises(ValueError):
        bf_a.union(bf_b)


# 8. estimate_count within 10%
def test_estimate_count():
    n = 5000
    bf = BloomFilter(expected_items=n * 2, false_positive_rate=0.01)
    for i in range(n):
        bf.add(f"est-{i}")
    est = bf.estimate_count()
    assert abs(est - n) / n < 0.1, f"Estimate {est} not within 10% of {n}"


# 9. Edge cases
def test_empty_filter():
    bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
    assert "anything" not in bf
    assert len(bf) == 0
    assert bf.bit_array_density() == 0.0
    assert bf.estimated_false_positive_rate() == 0.0
    assert bf.estimate_count() == 0.0


def test_single_item():
    bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
    bf.add("only")
    assert "only" in bf
    assert len(bf) == 1


def test_duplicate_add():
    bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
    bf.add("dup")
    bf.add("dup")
    assert "dup" in bf
    assert len(bf) == 2  # count increments even for duplicates


# 10. Determinism
def test_determinism():
    bf1 = BloomFilter(expected_items=100, false_positive_rate=0.01)
    bf2 = BloomFilter(expected_items=100, false_positive_rate=0.01)
    items = ["alpha", "beta", "gamma", "delta"]
    for item in items:
        bf1.add(item)
        bf2.add(item)
    assert bf1._bits == bf2._bits


# Bonus: Scalable Bloom filter
def test_scalable_bloom_filter():
    sbf = ScalableBloomFilter(initial_capacity=100, false_positive_rate=0.01)
    items = [f"scale-{i}" for i in range(500)]
    for item in items:
        sbf.add(item)
    for item in items:
        assert item in sbf
    assert len(sbf) == 500
    assert len(sbf._slices) > 1  # should have grown


# Statistics
def test_bit_array_density():
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    assert bf.bit_array_density() == 0.0
    for i in range(500):
        bf.add(f"d-{i}")
    density = bf.bit_array_density()
    assert 0 < density < 1


def test_estimated_fpr():
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    for i in range(1000):
        bf.add(f"fpr-{i}")
    efpr = bf.estimated_false_positive_rate()
    assert 0 < efpr < 0.05  # should be in reasonable range
