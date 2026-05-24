"""Quick verification of the SSTable implementation."""
import tempfile, os, sys
from sstable import *

with tempfile.TemporaryDirectory() as tmpdir:
    path1 = os.path.join(tmpdir, '001.sst')
    writer = SSTableWriter(path1, block_size=4)
    writer.add('apple', 'red', 1.0)
    writer.add('banana', 'yellow', 1.0)
    writer.add('cherry', 'dark red', 1.0)
    writer.add('date', 'brown', 1.0)
    writer.add('elderberry', 'purple', 1.0)
    meta = writer.finish()
    assert meta.entry_count == 5
    assert meta.min_key == 'apple'
    assert meta.max_key == 'elderberry'
    print('Write OK')

    reader = SSTableReader(path1)
    entry = reader.get('cherry')
    assert entry is not None
    assert entry.value == 'dark red'
    assert reader.get('fig') is None
    print('Read OK')

    results = list(reader.range_scan('banana', 'elderberry'))
    assert len(results) == 3, f'range_scan len={len(results)}: {[e.key for e in results]}'
    assert results[0].key == 'banana'
    print('Range scan OK')

    path2 = os.path.join(tmpdir, '002.sst')
    writer2 = SSTableWriter(path2, block_size=4)
    writer2.add('apple', 'green', 2.0)
    writer2.add('banana', None, 2.0)
    writer2.add('fig', 'purple', 2.0)
    writer2.finish()
    print('Write2 OK')

    merged_path = os.path.join(tmpdir, 'merged.sst')
    reader2 = SSTableReader(path2)
    merged = merge_sstables([reader2, reader], merged_path, remove_tombstones=True)
    entries = list(merged.scan())
    assert len(entries) == 5, f'merged len={len(entries)}: {[(e.key,e.value) for e in entries]}'
    assert entries[0].key == 'apple' and entries[0].value == 'green'
    print('Merge OK')

    manager = CompactionManager(tmpdir, strategy='size_tiered', min_threshold=2)
    r1 = SSTableReader(path1)
    r2 = SSTableReader(path2)
    manager.add_sstable(r1)
    manager.add_sstable(r2)
    assert manager.needs_compaction() == True
    new_sstables = manager.run_compaction()
    assert len(new_sstables) >= 1
    print('Compaction OK')

    # Tombstone test
    path3 = os.path.join(tmpdir, '003.sst')
    w3 = SSTableWriter(path3, block_size=4)
    w3.add('key1', None, 3.0)
    w3.finish()
    r3 = SSTableReader(path3)
    e3 = r3.get('key1')
    assert e3 is not None and e3.value is None
    print('Tombstone OK')

    # Empty SSTable
    path4 = os.path.join(tmpdir, '004.sst')
    w4 = SSTableWriter(path4, block_size=4)
    m4 = w4.finish()
    assert m4.entry_count == 0
    r4 = SSTableReader(path4)
    assert r4.get('anything') is None
    assert list(r4.scan()) == []
    print('Empty SSTable OK')

    # Single entry
    path5 = os.path.join(tmpdir, '005.sst')
    w5 = SSTableWriter(path5, block_size=4)
    w5.add('only', 'one', 1.0)
    w5.finish()
    r5 = SSTableReader(path5)
    assert r5.get('only').value == 'one'
    assert r5.metadata().min_key == 'only'
    assert r5.metadata().max_key == 'only'
    print('Single entry OK')

    # Multi-way merge (5 SSTables)
    readers_multi = []
    for i in range(5):
        p = os.path.join(tmpdir, f'multi_{i}.sst')
        w = SSTableWriter(p, block_size=4)
        w.add(f'key_{i:02d}', f'val_{i}', float(i))
        w.add('shared', f'val_{i}', float(i))
        w.finish()
        readers_multi.append(SSTableReader(p))
    merged_multi_path = os.path.join(tmpdir, 'multi_merged.sst')
    merged_multi = merge_sstables(readers_multi, merged_multi_path)
    entries_m = list(merged_multi.scan())
    shared = [e for e in entries_m if e.key == 'shared']
    assert len(shared) == 1
    assert shared[0].value == 'val_4'  # highest timestamp wins
    print('Multi-way merge OK')

    # Leveled compaction
    lcs_dir = os.path.join(tmpdir, 'lcs')
    os.makedirs(lcs_dir)
    mgr = CompactionManager(lcs_dir, strategy='leveled', l0_compaction_trigger=2)
    for i in range(3):
        p = os.path.join(lcs_dir, f'l0_{i}.sst')
        w = SSTableWriter(p, block_size=4)
        w.add(f'k{i}', f'v{i}', float(i))
        w.finish()
        r = SSTableReader(p)
        mgr.add_sstable(r)
    assert mgr.needs_compaction() == True
    result = mgr.run_compaction()
    assert len(result) >= 1
    assert result[0].level == 1
    print('Leveled compaction OK')

    print('\nALL TESTS PASSED')
