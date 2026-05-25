# B-Tree Fix Plan

## Bug 1: WAL doesn't fsync data file

**Problem:** `PageManager.write_page` calls `flush()` but not `os.fsync()`. The WAL fsyncs its own entries, but the data file pages it protects may not be durable on disk when the WAL commits. On crash, WAL recovery replays entries into a data file that may have already received those writes (harmless) or may be missing unrelated writes that were flushed but not synced.

**Fix:** Add `os.fsync(self._f.fileno())` to `PageManager.write_page`. Also add a `sync()` method for explicit use during WAL commit.

**Files:** `btree.py` lines 71-79

## Bug 2: WAL commit/truncate race

**Problem:** `WAL.commit()` writes the commit marker, fsyncs, then immediately truncates the WAL to zero. If the process crashes after truncation but before the data file is fsynced, there's no WAL to replay. The commit marker check in `recover()` (`if self.COMMIT_MARKER in data`) also discards a committed WAL without verifying data file durability.

**Fix:** Change the commit sequence to:
1. Fsync the data file (new `PageManager.sync()` call)
2. Truncate the WAL
3. Fsync the WAL

The commit marker is unnecessary — if the WAL has entries and no crash occurred, they've already been applied to the data file. On recovery, replay is always safe (idempotent page writes). Remove the commit marker entirely.

**Files:** `btree.py` lines 133-142, 144-172

## Bug 3: WAL doesn't log metadata writes

**Problem:** `BTree._write_meta` calls `self.pm.write_meta()` directly, bypassing the WAL. If the process crashes after a metadata update (e.g., new root pointer after split) but before the corresponding data pages are written, the tree is corrupted.

**Fix:** Route metadata writes through `_wal_write_page(0, data)` so they're logged in the WAL alongside data page writes. Add a `_wal_write_meta` method that packs the metadata, logs it, and writes it.

**Files:** `btree.py` lines 291-292, used by `put()` and `delete()`

## Bug 4: Weak checksum

**Problem:** `WAL._checksum` sums bytes mod 2^32. Byte reordering and many multi-byte corruptions go undetected.

**Fix:** Replace with `zlib.crc32`:
```python
@staticmethod
def _checksum(data):
    return zlib.crc32(data) & 0xFFFFFFFF
```

**Files:** `btree.py` lines 174-180, add `import zlib`

## Bug 5: Delete doesn't free pages

**Problem:** `_delete` removes keys from leaf nodes but never calls `free_page` when a leaf becomes empty, and never rebalances. The free list mechanism exists but is unused by delete.

**Fix:** After deleting a key, if the leaf is empty AND it's not the root, free the page and remove the corresponding key/child pointer from the parent. This requires passing parent context through `_delete`.

Change `_delete` to return a status indicating whether the child page is now empty, and handle cleanup in the parent's internal node. Full merge/redistribute is out of scope — just handle the empty-leaf case.

**Files:** `btree.py` lines 442-458

## Bug 6: `put` reads metadata multiple times

**Problem:** `put()` reads metadata at line 330, then `_insert` may call `allocate_page` which modifies metadata, then `put` reads metadata again at lines 340 and 349. The `total_keys` from the first read is used alongside `next_free`/`free_head` from later reads. This works because metadata is always re-read from disk, but it's fragile.

**Fix:** After `_insert` returns, do a single `self._read_meta()` to get the current state, then write updated metadata once with `total_keys + 1` (or unchanged for updates).

**Files:** `btree.py` lines 330-351

## Execution Order

1. Bug 4 (weak checksum) — isolated, no dependencies
2. Bug 6 (metadata reads) — isolated refactor
3. Bug 1 (data file fsync) — prerequisite for bug 2
4. Bug 3 (WAL metadata logging) — prerequisite for bug 2
5. Bug 2 (WAL commit sequence) — depends on 1 and 3
6. Bug 5 (delete page freeing) — independent but most complex

## Tests to Add

- WAL recovery with uncommitted entries (write WAL entries manually, don't commit, reopen)
- Delete freeing empty leaf pages (delete all keys from one leaf, verify page count)
- Metadata consistency after split (verify root pointer matches actual root)
- CRC32 detects corruption (flip a byte in WAL entry, verify recovery skips it)
