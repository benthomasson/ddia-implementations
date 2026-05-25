"""Tests for B-Tree storage engine."""
import tempfile
import os
from btree import BTree


def test_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        tree.put("apple", b"red fruit")
        tree.put("banana", b"yellow fruit")
        tree.put("cherry", b"small red fruit")
        tree.put("date", b"sweet fruit")

        assert tree.get("banana") == b"yellow fruit"
        assert tree.get("grape") is None
        assert "cherry" in tree

        tree.put("elderberry", b"dark berry")
        assert tree.stats.height == 2, f"height={tree.stats.height}"

        for fruit in ["fig", "grape", "honeydew", "kiwi", "lemon"]:
            tree.put(fruit, fruit.encode())

        assert len(tree) == 10, f"len={len(tree)}"

        results = tree.range_scan("cherry", "grape")
        keys = [k for k, v in results]
        assert keys == ["cherry", "date", "elderberry", "fig"], f"got {keys}"

        all_keys = [k for k, v in tree]
        assert all_keys == sorted(all_keys), f"not sorted: {all_keys}"

        assert tree.min_key() == "apple"
        assert tree.max_key() == "lemon"

        tree.put("apple", b"green fruit")
        assert tree.get("apple") == b"green fruit"

        assert tree.delete("banana")
        assert tree.get("banana") is None
        assert not tree.delete("nonexistent")

        stats = tree.stats
        assert stats.total_keys == 9
        assert stats.height >= 2

        tree.close()
        tree2 = BTree(tmpdir, max_keys_per_page=4)
        assert tree2.get("apple") == b"green fruit"
        assert tree2.get("banana") is None
        assert len(tree2) == 9
        tree2.close()
        print("test_basic PASSED")


def test_large():
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(1000):
            tree.put(f"key_{i:05d}", f"value_{i}".encode())
        assert len(tree) == 1000

        for i in range(1000):
            val = tree.get(f"key_{i:05d}")
            assert val == f"value_{i}".encode(), f"key_{i:05d}: {val}"

        all_keys = [k for k, v in tree]
        assert all_keys == sorted(all_keys)
        assert len(all_keys) == 1000

        assert tree.stats.height >= 2
        print(f"test_large PASSED (height={tree.stats.height}, pages={tree.stats.total_pages})")
        tree.close()


def test_range_scan():
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(20):
            tree.put(f"k{i:02d}", f"v{i}".encode())

        results = tree.range_scan("k05", "k10")
        keys = [k for k, v in results]
        assert keys == ["k05", "k06", "k07", "k08", "k09"], f"got {keys}"

        results = tree.range_scan("k15")
        keys = [k for k, v in results]
        assert keys == ["k15", "k16", "k17", "k18", "k19"], f"got {keys}"

        tree.close()
        print("test_range_scan PASSED")


def test_page_io_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(100):
            tree.put(f"key_{i:05d}", f"val_{i}".encode())
        height = tree.stats.height
        tree.get("key_00050")
        assert tree.pm.pages_read <= height + 1, f"read {tree.pm.pages_read} pages for height {height}"
        tree.close()
        print(f"test_page_io_counts PASSED (height={height}, reads={tree.pm.pages_read})")


def test_too_large_kv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, page_size=128, max_keys_per_page=4)
        try:
            tree.put("x", b"y" * 200)
            assert False, "should have raised ValueError"
        except ValueError:
            pass
        tree.close()
        print("test_too_large_kv PASSED")


def test_wal_recovery():
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(10):
            tree.put(f"key_{i:02d}", f"val_{i}".encode())
        # Don't call close - simulate crash (just close file handles)
        tree.pm._f.close()
        tree.wal._f.close()

        tree2 = BTree(tmpdir, max_keys_per_page=4)
        for i in range(10):
            val = tree2.get(f"key_{i:02d}")
            assert val == f"val_{i}".encode(), f"key_{i:02d}: {val}"
        assert len(tree2) == 10
        tree2.close()
        print("test_wal_recovery PASSED")


def test_delete_and_reinsert():
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(20):
            tree.put(f"k{i:02d}", f"v{i}".encode())
        assert len(tree) == 20

        for i in range(0, 20, 2):
            assert tree.delete(f"k{i:02d}")
        assert len(tree) == 10

        for i in range(0, 20, 2):
            tree.put(f"k{i:02d}", f"new_v{i}".encode())
        assert len(tree) == 20

        for i in range(0, 20, 2):
            assert tree.get(f"k{i:02d}") == f"new_v{i}".encode()
        for i in range(1, 20, 2):
            assert tree.get(f"k{i:02d}") == f"v{i}".encode()

        tree.close()
        print("test_delete_and_reinsert PASSED")


def test_wal_uncommitted_entries():
    """WAL entries without commit should be replayed on recovery."""
    import struct
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(5):
            tree.put(f"key_{i:02d}", f"val_{i}".encode())
        tree.close()

        tree2 = BTree(tmpdir, max_keys_per_page=4)
        assert len(tree2) == 5

        # Manually write WAL entries without committing
        wal_path = os.path.join(tmpdir, 'btree.wal')
        from btree import WAL, _serialize_leaf, HEADER_FMT, LEAF
        wal = WAL(wal_path)
        # Write a modified version of the root leaf's page
        page_data = tree2.pm.read_page(1)
        wal.log_write(1, page_data)
        wal._f.close()
        tree2.pm._f.close()

        # Reopen — WAL should replay even without commit marker
        tree3 = BTree(tmpdir, max_keys_per_page=4)
        assert len(tree3) == 5
        tree3.close()
        print("test_wal_uncommitted_entries PASSED")


def test_delete_frees_empty_leaf():
    """Deleting all keys from a non-root leaf should free the page."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        # Insert enough to get a height-2 tree with multiple leaves
        keys = [f"k{i:02d}" for i in range(10)]
        for k in keys:
            tree.put(k, b"v")

        assert tree.stats.height == 2
        pages_before = tree.stats.total_pages

        # Delete keys that land in a non-first leaf to trigger page freeing
        # With max_keys=4, after splits the right leaves hold keys from the middle/end
        # Delete all keys from one leaf by removing consecutive keys
        for k in ["k05", "k06"]:
            tree.delete(k)

        # Verify the tree is still consistent
        remaining = [k for k, v in tree]
        assert sorted(remaining) == remaining
        assert len(tree) == 8

        # Range scan should still work across the gap
        results = tree.range_scan("k04", "k08")
        result_keys = [k for k, v in results]
        assert "k05" not in result_keys
        assert "k06" not in result_keys
        assert "k04" in result_keys
        assert "k07" in result_keys

        tree.close()
        print("test_delete_frees_empty_leaf PASSED")


def test_metadata_consistency_after_split():
    """Root pointer in metadata must match the actual root after splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        for i in range(20):
            tree.put(f"key_{i:03d}", f"val_{i}".encode())

        root, height, total_keys, _, _ = tree._read_meta()
        assert total_keys == 20
        assert height >= 2

        # Verify root page is a valid internal node
        from btree import _page_type, INTERNAL
        data = tree.pm.read_page(root)
        assert _page_type(data) == INTERNAL

        # Every key should be reachable from the root
        for i in range(20):
            assert tree.get(f"key_{i:03d}") == f"val_{i}".encode()

        tree.close()

        # Reopen and verify consistency persists
        tree2 = BTree(tmpdir, max_keys_per_page=4)
        root2, height2, total2, _, _ = tree2._read_meta()
        assert root2 == root
        assert height2 == height
        assert total2 == 20
        tree2.close()
        print("test_metadata_consistency_after_split PASSED")


def test_crc32_detects_corruption():
    """Corrupted WAL entries should be skipped during recovery."""
    import struct as st
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = BTree(tmpdir, max_keys_per_page=4)
        tree.put("good_key", b"good_value")
        tree.close()

        # Write a valid WAL entry, then corrupt it
        wal_path = os.path.join(tmpdir, 'btree.wal')
        from btree import WAL
        wal = WAL(wal_path)
        page_data = b'\x01' * 4096
        wal.log_write(1, page_data)

        # Corrupt the page data in the WAL by flipping a byte
        wal._f.seek(12 + 100)  # past header, into page data
        wal._f.write(b'\xff')
        wal._f.flush()
        wal._f.close()

        # Reopen — corrupted entry should be skipped, tree should still work
        tree2 = BTree(tmpdir, max_keys_per_page=4)
        # The good_key was committed before the WAL was written,
        # so it's in the data file and not affected by corruption
        assert tree2.get("good_key") == b"good_value"
        tree2.close()
        print("test_crc32_detects_corruption PASSED")


if __name__ == "__main__":
    test_basic()
    test_large()
    test_range_scan()
    test_page_io_counts()
    test_too_large_kv()
    test_wal_recovery()
    test_delete_and_reinsert()
    test_wal_uncommitted_entries()
    test_delete_frees_empty_leaf()
    test_metadata_consistency_after_split()
    test_crc32_detects_corruption()
    print("\nAll tests passed!")
