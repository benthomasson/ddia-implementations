"""Tests for B-Tree storage engine."""
import tempfile


from btree import BTree


def test_basic_put_get():
    """Test basic insert and lookup from the example usage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        tree.put("apple", b"red fruit")
        tree.put("banana", b"yellow fruit")
        tree.put("cherry", b"small red fruit")
        tree.put("date", b"sweet fruit")

        assert tree.get("banana") == b"yellow fruit"
        assert tree.get("grape") is None
        assert "cherry" in tree

        # 5th insert triggers split -> height grows
        tree.put("elderberry", b"dark berry")
        assert tree.stats.height == 2, f"height={tree.stats.height}"

        for fruit in ["fig", "grape", "honeydew", "kiwi", "lemon"]:
            tree.put(fruit, fruit.encode())
        assert len(tree) == 10

        # Update existing key
        tree.put("apple", b"green fruit")
        assert tree.get("apple") == b"green fruit"
        assert len(tree) == 10  # no duplicate

        # Delete
        assert tree.delete("banana")
        assert tree.get("banana") is None
        assert not tree.delete("nonexistent")
        assert len(tree) == 9

        # Min/max
        assert tree.min_key() == "apple"
        assert tree.max_key() == "lemon"

        tree.close()
        print("test_basic_put_get PASSED")


def test_range_scan():
    """Test range scan within and across leaf pages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(20):
            tree.put(f"k{i:02d}", f"v{i}".encode())

        # Bounded range
        results = tree.range_scan("k05", "k10")
        keys = [k for k, v in results]
        assert keys == ["k05", "k06", "k07", "k08", "k09"], f"got {keys}"

        # Unbounded range (to end)
        results = tree.range_scan("k18")
        keys = [k for k, v in results]
        assert keys == ["k18", "k19"], f"got {keys}"

        # Full iteration is sorted
        all_keys = [k for k, v in tree]
        assert all_keys == sorted(all_keys), "iteration not sorted"
        assert len(all_keys) == 20

        tree.close()
        print("test_range_scan PASSED")


def test_persistence():
    """Test crash recovery: close and reopen preserves data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(50):
            tree.put(f"key_{i:03d}", f"val_{i}".encode())
        tree.put("key_000", b"updated")
        tree.delete("key_025")
        tree.close()

        # Reopen
        tree2 = BTree(tmpdir, max_keys_per_page=4)
        assert tree2.get("key_000") == b"updated"
        assert tree2.get("key_025") is None
        assert len(tree2) == 49
        assert tree2.min_key() == "key_000"
        assert tree2.max_key() == "key_049"
        tree2.close()
        print("test_persistence PASSED")


def test_large_dataset():
    """Test with 1000+ keys: correctness and balance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        n = 1000
        for i in range(n):
            tree.put(f"key_{i:05d}", f"value_{i}".encode())
        assert len(tree) == n

        # Verify all lookups
        for i in range(n):
            val = tree.get(f"key_{i:05d}")
            assert val == f"value_{i}".encode(), f"key_{i:05d}: got {val}"

        # Sorted iteration
        all_keys = [k for k, v in tree]
        assert all_keys == sorted(all_keys)
        assert len(all_keys) == n

        # Tree must be balanced (height >= 2 for 1000 keys with max_keys=4)
        stats = tree.stats
        assert stats.height >= 2
        print(f"test_large_dataset PASSED (height={stats.height}, pages={stats.total_pages})")
        tree.close()


def test_page_io_counts():
    """Test that page reads are O(height) for lookups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(100):
            tree.put(f"key_{i:03d}", f"val_{i}".encode())

        height = tree.stats.height
        # A get should read at most height pages (plus metadata read)
        tree.pm.reset_counters()
        tree.get("key_050")
        # pages_read should be height (one per level) — reset_counters is called in get()
        # Actually get() calls reset_counters itself, so we read the counter after
        assert tree.pm.pages_read <= height + 1, \
            f"pages_read={tree.pm.pages_read}, height={height}"
        tree.close()
        print("test_page_io_counts PASSED")


def test_value_too_large():
    """Test error when key-value pair exceeds page size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, page_size=128, max_keys_per_page=4)
        try:
            tree.put("k", b"x" * 200)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        tree.close()
        print("test_value_too_large PASSED")


def test_empty_tree():
    """Test operations on an empty tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        assert tree.get("anything") is None
        assert len(tree) == 0
        assert tree.min_key() is None
        assert tree.max_key() is None
        assert not tree.delete("nothing")
        assert list(tree) == []
        assert tree.range_scan("a", "z") == []
        tree.close()
        print("test_empty_tree PASSED")


def test_delete_all_keys():
    """Test deleting all keys from the tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        keys = ["a", "b", "c", "d", "e", "f", "g", "h"]
        for k in keys:
            tree.put(k, k.encode())
        assert len(tree) == 8

        for k in keys:
            assert tree.delete(k), f"failed to delete {k}"
        assert len(tree) == 0
        assert tree.get("a") is None

        # Can still insert after deleting everything
        tree.put("new", b"value")
        assert tree.get("new") == b"value"
        assert len(tree) == 1
        tree.close()
        print("test_delete_all_keys PASSED")


if __name__ == '__main__':
    test_basic_put_get()
    test_range_scan()
    test_persistence()
    test_large_dataset()
    test_page_io_counts()
    test_value_too_large()
    test_empty_tree()
    test_delete_all_keys()
    print("\nAll tests PASSED")
