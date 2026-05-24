"""Tests for Bitcask storage engine."""
import os
import tempfile

from bitcask import BitcaskStore


def test_basic_crud():
    """Test put, get, delete, keys, len, contains."""
    with tempfile.TemporaryDirectory() as d:
        s = BitcaskStore(d, sync_writes=False)
        assert s.get("x") is None
        assert len(s) == 0
        assert s.keys() == []

        s.put("a", "1")
        s.put("b", "2")
        assert s.get("a") == "1"
        assert s.get("b") == "2"
        assert s.get("nope") is None
        assert len(s) == 2
        assert "a" in s
        assert "z" not in s

        s.delete("a")
        assert s.get("a") is None
        assert len(s) == 1
        assert "a" not in s
        assert s.keys() == ["b"]
        s.close()
    print("PASS: basic_crud")


def test_overwrite():
    """Updating a key returns the latest value."""
    with tempfile.TemporaryDirectory() as d:
        s = BitcaskStore(d, sync_writes=False)
        s.put("k", "v1")
        s.put("k", "v2")
        s.put("k", "v3")
        assert s.get("k") == "v3"
        assert len(s) == 1
        s.close()
    print("PASS: overwrite")


def test_file_rotation():
    """Data files rotate when size limit is exceeded."""
    with tempfile.TemporaryDirectory() as d:
        s = BitcaskStore(d, max_file_size=256, sync_writes=False)
        for i in range(100):
            s.put(f"key{i}", f"value{i}")
        data_files = [f for f in os.listdir(d) if f.endswith(".data")]
        assert len(data_files) > 1
        for i in range(100):
            assert s.get(f"key{i}") == f"value{i}"
        s.close()
    print("PASS: file_rotation")


def test_compaction():
    """Old values and tombstones removed, live data preserved."""
    with tempfile.TemporaryDirectory() as d:
        s = BitcaskStore(d, max_file_size=1024, sync_writes=False)
        s.put("sensor:temp:1", "72.5")
        s.put("sensor:temp:2", "68.3")
        for i in range(100):
            s.put("sensor:temp:1", str(70.0 + i * 0.1))
        s.delete("sensor:temp:2")
        s.compact()
        assert s.get("sensor:temp:1") == str(70.0 + 99 * 0.1)
        assert s.get("sensor:temp:2") is None
        assert len(s) == 1
        s.close()
    print("PASS: compaction")


def test_hint_files_and_startup():
    """Hint files created during compaction; used for fast startup rebuild."""
    with tempfile.TemporaryDirectory() as d:
        s = BitcaskStore(d, max_file_size=256, sync_writes=False)
        for i in range(50):
            s.put(f"key:{i}", f"value:{i}")
        s.compact()
        hint_files = [f for f in os.listdir(d) if f.endswith(".hint")]
        assert len(hint_files) > 0, "No hint files created"
        s.close()

        s2 = BitcaskStore(d, max_file_size=256, sync_writes=False)
        for i in range(50):
            assert s2.get(f"key:{i}") == f"value:{i}"
        s2.close()
    print("PASS: hint_files_and_startup")


def test_startup_recovery_no_hints():
    """Index rebuilt from data files without hint files."""
    with tempfile.TemporaryDirectory() as d:
        s = BitcaskStore(d, max_file_size=512, sync_writes=False)
        s.put("a", "1")
        s.put("b", "2")
        s.put("c", "3")
        s.delete("b")
        s.close()

        # Remove any hint files to force data-file scan
        for f in os.listdir(d):
            if f.endswith(".hint"):
                os.remove(os.path.join(d, f))

        s2 = BitcaskStore(d, max_file_size=512, sync_writes=False)
        assert s2.get("a") == "1"
        assert s2.get("b") is None
        assert s2.get("c") == "3"
        assert len(s2) == 2
        s2.close()
    print("PASS: startup_recovery_no_hints")


def test_large_dataset():
    """10,000+ keys with repeated updates."""
    with tempfile.TemporaryDirectory() as d:
        s = BitcaskStore(d, max_file_size=64 * 1024, sync_writes=False)
        for i in range(10000):
            s.put(f"k{i}", f"v{i}")
        # Overwrite half
        for i in range(0, 10000, 2):
            s.put(f"k{i}", f"updated{i}")
        assert len(s) == 10000
        assert s.get("k0") == "updated0"
        assert s.get("k1") == "v1"
        assert s.get("k9999") == "v9999"
        assert s.get("k9998") == "updated9998"
        s.close()
    print("PASS: large_dataset")


def test_edge_cases():
    """Empty store, delete non-existent, put after delete."""
    with tempfile.TemporaryDirectory() as d:
        s = BitcaskStore(d, sync_writes=False)
        assert len(s) == 0
        assert s.keys() == []
        assert s.get("nothing") is None

        # Delete non-existent key should not error
        s.delete("nothing")
        assert len(s) == 0

        # Put after delete
        s.put("x", "1")
        s.delete("x")
        s.put("x", "2")
        assert s.get("x") == "2"
        assert len(s) == 1
        s.close()
    print("PASS: edge_cases")


def test_example_from_spec():
    """Run the example from the task specification."""
    with tempfile.TemporaryDirectory() as d:
        store = BitcaskStore(d, max_file_size=1024, sync_writes=False)
        store.put("sensor:temp:1", "72.5")
        store.put("sensor:temp:2", "68.3")
        assert store.get("sensor:temp:1") == "72.5"
        assert store.get("nonexistent") is None
        assert len(store) == 2

        for i in range(100):
            store.put("sensor:temp:1", str(70.0 + i * 0.1))
        assert store.get("sensor:temp:1") == str(70.0 + 99 * 0.1)

        store.delete("sensor:temp:2")
        assert store.get("sensor:temp:2") is None
        assert "sensor:temp:2" not in store
        assert store.keys() == ["sensor:temp:1"]

        store.compact()
        assert store.get("sensor:temp:1") == str(70.0 + 99 * 0.1)
        assert store.get("sensor:temp:2") is None
        store.close()
    print("PASS: example_from_spec")


if __name__ == "__main__":
    test_basic_crud()
    test_overwrite()
    test_file_rotation()
    test_compaction()
    test_hint_files_and_startup()
    test_startup_recovery_no_hints()
    test_large_dataset()
    test_edge_cases()
    test_example_from_spec()
    print("\nAll tests passed!")
