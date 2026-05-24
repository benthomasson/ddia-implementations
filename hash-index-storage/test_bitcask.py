"""Tests for Bitcask storage engine."""
import tempfile
import os
from bitcask import BitcaskStore

def test_basic_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BitcaskStore(tmpdir, sync_writes=False)
        store.put("a", "1")
        store.put("b", "2")
        assert store.get("a") == "1"
        assert store.get("b") == "2"
        assert store.get("nonexistent") is None
        assert len(store) == 2
        assert "a" in store
        assert "z" not in store
        store.delete("a")
        assert store.get("a") is None
        assert len(store) == 1
        store.close()
    print("PASS: basic_crud")

def test_overwrite():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BitcaskStore(tmpdir, sync_writes=False)
        store.put("k", "v1")
        store.put("k", "v2")
        store.put("k", "v3")
        assert store.get("k") == "v3"
        assert len(store) == 1
        store.close()
    print("PASS: overwrite")

def test_file_rotation():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BitcaskStore(tmpdir, max_file_size=256, sync_writes=False)
        for i in range(100):
            store.put(f"key{i}", f"value{i}")
        data_files = [f for f in os.listdir(tmpdir) if f.endswith(".data")]
        assert len(data_files) > 1, f"Expected multiple data files, got {len(data_files)}"
        for i in range(100):
            assert store.get(f"key{i}") == f"value{i}"
        store.close()
    print("PASS: file_rotation")

def test_compaction():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BitcaskStore(tmpdir, max_file_size=1024, sync_writes=False)
        store.put("sensor:temp:1", "72.5")
        store.put("sensor:temp:2", "68.3")
        for i in range(100):
            store.put("sensor:temp:1", str(70.0 + i * 0.1))
        store.delete("sensor:temp:2")
        store.compact()
        assert store.get("sensor:temp:1") == str(70.0 + 99 * 0.1)
        assert store.get("sensor:temp:2") is None
        assert len(store) == 1
        store.close()
    print("PASS: compaction")

def test_hint_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BitcaskStore(tmpdir, max_file_size=256, sync_writes=False)
        for i in range(50):
            store.put(f"key:{i}", f"value:{i}")
        store.compact()
        hint_files = [f for f in os.listdir(tmpdir) if f.endswith(".hint")]
        assert len(hint_files) > 0, "No hint files created"
        store.close()

        store2 = BitcaskStore(tmpdir, max_file_size=256, sync_writes=False)
        for i in range(50):
            assert store2.get(f"key:{i}") == f"value:{i}", f"Failed for key:{i}"
        store2.close()
    print("PASS: hint_files")

def test_startup_recovery():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BitcaskStore(tmpdir, max_file_size=512, sync_writes=False)
        store.put("a", "1")
        store.put("b", "2")
        store.put("c", "3")
        store.close()

        store2 = BitcaskStore(tmpdir, max_file_size=512, sync_writes=False)
        assert store2.get("a") == "1"
        assert store2.get("b") == "2"
        assert store2.get("c") == "3"
        store2.close()
    print("PASS: startup_recovery")

def test_crash_recovery():
    """Rebuild index from data files without hint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BitcaskStore(tmpdir, max_file_size=512, sync_writes=False)
        for i in range(50):
            store.put(f"k{i}", f"v{i}")
        store.put("k0", "updated0")
        store.delete("k1")
        store.close()

        for f in os.listdir(tmpdir):
            if f.endswith(".hint"):
                os.remove(os.path.join(tmpdir, f))

        store2 = BitcaskStore(tmpdir, max_file_size=512, sync_writes=False)
        assert store2.get("k0") == "updated0"
        assert store2.get("k1") is None
        assert store2.get("k2") == "v2"
        assert len(store2) == 49
        store2.close()
    print("PASS: crash_recovery")

def test_large_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BitcaskStore(tmpdir, max_file_size=4096, sync_writes=False)
        for i in range(10000):
            store.put(f"k{i}", f"v{i}")
        for i in range(0, 10000, 2):
            store.put(f"k{i}", f"updated{i}")
        assert len(store) == 10000
        assert store.get("k0") == "updated0"
        assert store.get("k1") == "v1"
        store.compact()
        assert store.get("k0") == "updated0"
        assert store.get("k1") == "v1"
        assert len(store) == 10000
        store.close()
    print("PASS: large_dataset")

def test_edge_cases():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BitcaskStore(tmpdir, sync_writes=False)
        assert len(store) == 0
        assert store.keys() == []
        assert store.get("x") is None
        store.delete("nonexistent")
        store.put("a", "1")
        store.delete("a")
        assert store.get("a") is None
        store.put("a", "2")
        assert store.get("a") == "2"
        store.close()
    print("PASS: edge_cases")

if __name__ == "__main__":
    test_basic_crud()
    test_overwrite()
    test_file_rotation()
    test_compaction()
    test_hint_files()
    test_startup_recovery()
    test_crash_recovery()
    test_large_dataset()
    test_edge_cases()
    print("\nAll tests passed!")
