"""B-Tree storage engine with page-based storage, splits, and WAL crash safety."""

import os
import struct
import zlib
from bisect import bisect_left, bisect_right

# Page types
INTERNAL = 0
LEAF = 1

# Sentinel for no sibling
NO_SIBLING = 0xFFFFFFFF

# Metadata page layout (page 0):
#   root_page(4B), height(4B), total_keys(4B), next_free_page(4B), free_list_head(4B)
META_FMT = '>5I'
META_SIZE = struct.calcsize(META_FMT)

# Page header: type(1B), num_keys(2B)
HEADER_FMT = '>BH'
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class PageManager:
    """Fixed-size page I/O to a single data file."""

    def __init__(self, file_path, page_size=4096):
        self.file_path = file_path
        self.page_size = page_size
        self.pages_read = 0
        self.pages_written = 0
        existed = os.path.exists(file_path)
        self._f = open(file_path, 'r+b' if existed else 'w+b')
        if not existed:
            # Write initial metadata page
            self._write_meta(1, 1, 0, 2, NO_SIBLING)
            # Write empty root leaf page (page 1)
            self._write_empty_leaf(1)

    def _write_meta(self, root, height, total_keys, next_free, free_head):
        data = struct.pack(META_FMT, root, height, total_keys, next_free, free_head)
        data = data.ljust(self.page_size, b'\x00')
        self._f.seek(0)
        self._f.write(data)
        self._f.flush()

    def _write_empty_leaf(self, page_num):
        data = struct.pack(HEADER_FMT, LEAF, 0)
        data += struct.pack('>I', NO_SIBLING)  # next sibling
        data = data.ljust(self.page_size, b'\x00')
        self._f.seek(page_num * self.page_size)
        self._f.write(data)
        self._f.flush()

    def read_meta(self):
        self._f.seek(0)
        raw = self._f.read(META_SIZE)
        return struct.unpack(META_FMT, raw)

    def write_meta(self, root, height, total_keys, next_free, free_head):
        self._write_meta(root, height, total_keys, next_free, free_head)

    def read_page(self, page_num):
        self.pages_read += 1
        self._f.seek(page_num * self.page_size)
        data = self._f.read(self.page_size)
        if len(data) < self.page_size:
            data = data.ljust(self.page_size, b'\x00')
        return data

    def write_page(self, page_num, data):
        self.pages_written += 1
        if len(data) < self.page_size:
            data = data.ljust(self.page_size, b'\x00')
        elif len(data) > self.page_size:
            data = data[:self.page_size]
        self._f.seek(page_num * self.page_size)
        self._f.write(data)
        self._f.flush()

    def allocate_page(self):
        root, height, total_keys, next_free, free_head = self.read_meta()
        if free_head != NO_SIBLING:
            # Reuse from free list
            page_num = free_head
            page_data = self.read_page(page_num)
            # First 4 bytes after header store next free pointer
            new_head = struct.unpack('>I', page_data[HEADER_SIZE:HEADER_SIZE+4])[0]
            self.write_meta(root, height, total_keys, next_free, new_head)
            return page_num
        page_num = next_free
        self.write_meta(root, height, total_keys, page_num + 1, free_head)
        return page_num

    def free_page(self, page_num):
        root, height, total_keys, next_free, free_head = self.read_meta()
        # Write a free-list node: header + pointer to old head
        data = struct.pack(HEADER_FMT, 0, 0) + struct.pack('>I', free_head)
        self.write_page(page_num, data)
        self.write_meta(root, height, total_keys, next_free, page_num)

    def sync(self):
        self._f.flush()
        os.fsync(self._f.fileno())

    def reset_counters(self):
        self.pages_read = 0
        self.pages_written = 0

    def close(self):
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()


class WAL:
    """Write-ahead log for crash safety."""

    # Entry format: seq(4B) + page_num(4B) + data_len(4B) + data + checksum(4B)
    ENTRY_HEADER = '>III'
    ENTRY_HEADER_SIZE = struct.calcsize('>III')

    def __init__(self, wal_path):
        self.wal_path = wal_path
        self._seq = 0
        existed = os.path.exists(wal_path) and os.path.getsize(wal_path) > 0
        self._f = open(wal_path, 'r+b' if existed else 'w+b')

    def log_write(self, page_num, page_data):
        self._seq += 1
        header = struct.pack(self.ENTRY_HEADER, self._seq, page_num, len(page_data))
        checksum = struct.pack('>I', self._checksum(page_data))
        self._f.seek(0, 2)  # seek to end
        self._f.write(header + page_data + checksum)
        self._f.flush()
        os.fsync(self._f.fileno())

    def commit(self, page_manager):
        page_manager.sync()
        self._f.seek(0)
        self._f.truncate(0)
        self._f.flush()
        os.fsync(self._f.fileno())
        self._seq = 0

    def recover(self, page_manager):
        self._f.seek(0)
        data = self._f.read()
        if not data:
            return 0
        offset = 0
        recovered = 0
        while offset + self.ENTRY_HEADER_SIZE <= len(data):
            seq, page_num, data_len = struct.unpack(
                self.ENTRY_HEADER, data[offset:offset + self.ENTRY_HEADER_SIZE])
            offset += self.ENTRY_HEADER_SIZE
            if offset + data_len + 4 > len(data):
                break
            page_data = data[offset:offset + data_len]
            offset += data_len
            checksum = struct.unpack('>I', data[offset:offset + 4])[0]
            offset += 4
            if self._checksum(page_data) == checksum:
                page_manager.write_page(page_num, page_data)
                recovered += 1
        page_manager.sync()
        self._f.seek(0)
        self._f.truncate(0)
        self._f.flush()
        os.fsync(self._f.fileno())
        return recovered

    @staticmethod
    def _checksum(data):
        return zlib.crc32(data) & 0xFFFFFFFF

    def close(self):
        self._f.close()


def _serialize_leaf(keys, values, next_sibling=NO_SIBLING):
    """Serialize a leaf page: header + next_sibling + entries."""
    buf = struct.pack(HEADER_FMT, LEAF, len(keys))
    buf += struct.pack('>I', next_sibling)
    for k, v in zip(keys, values):
        kb = k.encode('utf-8') if isinstance(k, str) else k
        buf += struct.pack('>H', len(kb)) + kb
        buf += struct.pack('>H', len(v)) + v
    return buf


def _deserialize_leaf(data):
    """Returns (keys, values, next_sibling)."""
    _, num_keys = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    offset = HEADER_SIZE
    next_sib = struct.unpack('>I', data[offset:offset+4])[0]
    offset += 4
    keys = []
    values = []
    for _ in range(num_keys):
        klen = struct.unpack('>H', data[offset:offset+2])[0]
        offset += 2
        k = data[offset:offset+klen].decode('utf-8')
        offset += klen
        vlen = struct.unpack('>H', data[offset:offset+2])[0]
        offset += 2
        v = data[offset:offset+vlen]
        offset += vlen
        keys.append(k)
        values.append(v)
    return keys, values, next_sib


def _serialize_internal(keys, children):
    """Serialize internal page: header + child_0 + (key, child)..."""
    buf = struct.pack(HEADER_FMT, INTERNAL, len(keys))
    buf += struct.pack('>I', children[0])
    for i, k in enumerate(keys):
        kb = k.encode('utf-8') if isinstance(k, str) else k
        buf += struct.pack('>H', len(kb)) + kb
        buf += struct.pack('>I', children[i + 1])
    return buf


def _deserialize_internal(data):
    """Returns (keys, children)."""
    _, num_keys = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    offset = HEADER_SIZE
    child0 = struct.unpack('>I', data[offset:offset+4])[0]
    offset += 4
    children = [child0]
    keys = []
    for _ in range(num_keys):
        klen = struct.unpack('>H', data[offset:offset+2])[0]
        offset += 2
        k = data[offset:offset+klen].decode('utf-8')
        offset += klen
        child = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        keys.append(k)
        children.append(child)
    return keys, children


def _page_type(data):
    return struct.unpack('>B', data[:1])[0]


class TreeStats:
    def __init__(self, height=1, total_keys=0, total_pages=0,
                 internal_pages=0, leaf_pages=0, pages_read=0, pages_written=0):
        self.height = height
        self.total_keys = total_keys
        self.total_pages = total_pages
        self.internal_pages = internal_pages
        self.leaf_pages = leaf_pages
        self.pages_read = pages_read
        self.pages_written = pages_written


class BTree:
    """Disk-backed B-tree with WAL crash safety."""

    def __init__(self, directory, page_size=4096, max_keys_per_page=None):
        os.makedirs(directory, exist_ok=True)
        data_path = os.path.join(directory, 'btree.dat')
        wal_path = os.path.join(directory, 'btree.wal')
        self.page_size = page_size
        self.pm = PageManager(data_path, page_size)
        self.wal = WAL(wal_path)
        # Recover from WAL if needed
        self.wal.recover(self.pm)
        # Calculate max keys if not given
        if max_keys_per_page is not None:
            self.max_keys = max_keys_per_page
        else:
            # Conservative estimate: each leaf entry needs ~2+key+2+val bytes
            # Assume avg key=32B, val=64B => ~100B per entry
            # Available space = page_size - header(3) - next_sib(4) = page_size - 7
            self.max_keys = max(4, (page_size - 7) // 100)
        self.pm.reset_counters()

    def _read_meta(self):
        return self.pm.read_meta()

    def _write_meta(self, root, height, total_keys, next_free, free_head):
        self.pm.write_meta(root, height, total_keys, next_free, free_head)

    def _wal_write_page(self, page_num, data):
        """Log to WAL then write page."""
        padded = data.ljust(self.page_size, b'\x00')
        self.wal.log_write(page_num, padded)
        self.pm.write_page(page_num, padded)

    def _wal_write_meta(self, root, height, total_keys, next_free, free_head):
        """Log metadata to WAL then write."""
        data = struct.pack(META_FMT, root, height, total_keys, next_free, free_head)
        padded = data.ljust(self.page_size, b'\x00')
        self.wal.log_write(0, padded)
        self.pm.write_meta(root, height, total_keys, next_free, free_head)

    def get(self, key):
        """Look up a value by key. Returns None if not found."""
        self.pm.reset_counters()
        root, height, _, _, _ = self._read_meta()
        return self._search(root, key, height)

    def _search(self, page_num, key, depth):
        data = self.pm.read_page(page_num)
        if depth == 1:
            # Leaf
            keys, values, _ = _deserialize_leaf(data)
            idx = bisect_left(keys, key)
            if idx < len(keys) and keys[idx] == key:
                return values[idx]
            return None
        else:
            # Internal
            ikeys, children = _deserialize_internal(data)
            idx = bisect_right(ikeys, key)
            return self._search(children[idx], key, depth - 1)

    def put(self, key, value):
        """Insert or update a key-value pair."""
        kb = key.encode('utf-8')
        entry_size = HEADER_SIZE + 4 + 2 + len(kb) + 2 + len(value)
        if entry_size > self.page_size:
            raise ValueError(f"Key-value pair too large for page size {self.page_size}")

        self.pm.reset_counters()
        root, height, total_keys, next_free, free_head = self._read_meta()
        result = self._insert(root, key, value, height)

        if result is None:
            self.wal.commit(self.pm)
            return

        if result == 'inserted':
            root, height, _, next_free, free_head = self._read_meta()
            self._wal_write_meta(root, height, total_keys + 1, next_free, free_head)
            self.wal.commit(self.pm)
            return

        mid_key, new_page = result
        new_root = self.pm.allocate_page()
        root_data = _serialize_internal([mid_key], [root, new_page])
        self._wal_write_page(new_root, root_data)
        _, _, _, next_free, free_head = self._read_meta()
        self._wal_write_meta(new_root, height + 1, total_keys + 1, next_free, free_head)
        self.wal.commit(self.pm)

    def _insert(self, page_num, key, value, depth):
        """Insert into subtree rooted at page_num.

        Returns:
            None - key existed and was updated
            'inserted' - key was inserted, no split needed
            (mid_key, new_page_num) - split occurred
        """
        data = self.pm.read_page(page_num)

        if depth == 1:
            # Leaf page
            keys, values, next_sib = _deserialize_leaf(data)
            idx = bisect_left(keys, key)
            if idx < len(keys) and keys[idx] == key:
                # Update existing
                values[idx] = value
                new_data = _serialize_leaf(keys, values, next_sib)
                self._wal_write_page(page_num, new_data)
                return None

            keys.insert(idx, key)
            values.insert(idx, value)

            if len(keys) <= self.max_keys:
                new_data = _serialize_leaf(keys, values, next_sib)
                self._wal_write_page(page_num, new_data)
                return 'inserted'

            # Split
            mid = len(keys) // 2
            left_keys, left_vals = keys[:mid], values[:mid]
            right_keys, right_vals = keys[mid:], values[mid:]
            mid_key = right_keys[0]

            new_page = self.pm.allocate_page()
            # Right page gets old next_sib
            right_data = _serialize_leaf(right_keys, right_vals, next_sib)
            self._wal_write_page(new_page, right_data)
            # Left page points to right page
            left_data = _serialize_leaf(left_keys, left_vals, new_page)
            self._wal_write_page(page_num, left_data)
            return (mid_key, new_page)

        else:
            # Internal page
            ikeys, children = _deserialize_internal(data)
            idx = bisect_right(ikeys, key)
            result = self._insert(children[idx], key, value, depth - 1)

            if result is None or result == 'inserted':
                return result

            # Child split: insert new key and child
            mid_key, new_child = result
            ikeys.insert(idx, mid_key)
            children.insert(idx + 1, new_child)

            if len(ikeys) <= self.max_keys:
                new_data = _serialize_internal(ikeys, children)
                self._wal_write_page(page_num, new_data)
                return 'inserted'

            # Split internal node
            mid = len(ikeys) // 2
            promote_key = ikeys[mid]
            left_keys = ikeys[:mid]
            left_children = children[:mid + 1]
            right_keys = ikeys[mid + 1:]
            right_children = children[mid + 1:]

            new_page = self.pm.allocate_page()
            right_data = _serialize_internal(right_keys, right_children)
            self._wal_write_page(new_page, right_data)
            left_data = _serialize_internal(left_keys, left_children)
            self._wal_write_page(page_num, left_data)
            return (promote_key, new_page)

    def delete(self, key):
        """Delete a key. Returns True if found and deleted."""
        self.pm.reset_counters()
        root, height, total_keys, next_free, free_head = self._read_meta()
        found = self._delete(root, key, height)
        if found:
            _, _, _, next_free, free_head = self._read_meta()
            self._wal_write_meta(root, height, total_keys - 1, next_free, free_head)
            self.wal.commit(self.pm)
        return bool(found)

    def _delete(self, page_num, key, depth):
        """Returns False (not found), True (deleted), or 'empty' (deleted, leaf now empty)."""
        data = self.pm.read_page(page_num)

        if depth == 1:
            keys, values, next_sib = _deserialize_leaf(data)
            idx = bisect_left(keys, key)
            if idx >= len(keys) or keys[idx] != key:
                return False
            keys.pop(idx)
            values.pop(idx)
            new_data = _serialize_leaf(keys, values, next_sib)
            self._wal_write_page(page_num, new_data)
            return 'empty' if not keys else True
        else:
            ikeys, children = _deserialize_internal(data)
            idx = bisect_right(ikeys, key)
            result = self._delete(children[idx], key, depth - 1)

            if result == 'empty' and depth == 2 and idx > 0:
                child_page = children[idx]
                empty_data = self.pm.read_page(child_page)
                _, _, empty_next_sib = _deserialize_leaf(empty_data)
                prev_data = self.pm.read_page(children[idx - 1])
                prev_keys, prev_vals, _ = _deserialize_leaf(prev_data)
                self._wal_write_page(
                    children[idx - 1],
                    _serialize_leaf(prev_keys, prev_vals, empty_next_sib))
                ikeys.pop(idx - 1)
                children.pop(idx)
                self.pm.free_page(child_page)
                self._wal_write_page(page_num, _serialize_internal(ikeys, children))
                return True

            if result == 'empty':
                return True
            return result

    def contains(self, key):
        return self.get(key) is not None

    def __contains__(self, key):
        return self.contains(key)

    def range_scan(self, start_key, end_key=None):
        """Return all (key, value) pairs where start_key <= key < end_key."""
        self.pm.reset_counters()
        root, height, _, _, _ = self._read_meta()
        # Find the leaf containing start_key
        leaf_num = self._find_leaf(root, start_key, height)
        results = []
        while leaf_num != NO_SIBLING:
            data = self.pm.read_page(leaf_num)
            keys, values, next_sib = _deserialize_leaf(data)
            for k, v in zip(keys, values):
                if k < start_key:
                    continue
                if end_key is not None and k >= end_key:
                    return results
                results.append((k, v))
            leaf_num = next_sib
        return results

    def _find_leaf(self, page_num, key, depth):
        """Find the leaf page number that would contain key."""
        if depth == 1:
            return page_num
        data = self.pm.read_page(page_num)
        ikeys, children = _deserialize_internal(data)
        idx = bisect_right(ikeys, key)
        return self._find_leaf(children[idx], key, depth - 1)

    def min_key(self):
        root, height, total_keys, _, _ = self._read_meta()
        if total_keys == 0:
            return None
        page_num = root
        for _ in range(height - 1):
            data = self.pm.read_page(page_num)
            _, children = _deserialize_internal(data)
            page_num = children[0]
        data = self.pm.read_page(page_num)
        keys, _, _ = _deserialize_leaf(data)
        return keys[0] if keys else None

    def max_key(self):
        root, height, total_keys, _, _ = self._read_meta()
        if total_keys == 0:
            return None
        page_num = root
        for _ in range(height - 1):
            data = self.pm.read_page(page_num)
            _, children = _deserialize_internal(data)
            page_num = children[-1]
        data = self.pm.read_page(page_num)
        keys, _, _ = _deserialize_leaf(data)
        return keys[-1] if keys else None

    def __len__(self):
        _, _, total_keys, _, _ = self._read_meta()
        return total_keys

    def __iter__(self):
        """Iterate over all (key, value) pairs in sorted order."""
        root, height, _, _, _ = self._read_meta()
        # Find leftmost leaf
        page_num = root
        for _ in range(height - 1):
            data = self.pm.read_page(page_num)
            _, children = _deserialize_internal(data)
            page_num = children[0]
        # Scan through leaves
        while page_num != NO_SIBLING:
            data = self.pm.read_page(page_num)
            keys, values, next_sib = _deserialize_leaf(data)
            for k, v in zip(keys, values):
                yield (k, v)
            page_num = next_sib

    @property
    def stats(self):
        root, height, total_keys, next_free, _ = self._read_meta()
        total_pages = next_free  # includes metadata page
        internal = 0
        leaves = 0
        self._count_pages(root, height, counts := [0, 0])
        internal, leaves = counts
        return TreeStats(
            height=height,
            total_keys=total_keys,
            total_pages=total_pages,
            internal_pages=internal,
            leaf_pages=leaves,
            pages_read=self.pm.pages_read,
            pages_written=self.pm.pages_written,
        )

    def _count_pages(self, page_num, depth, counts):
        data = self.pm.read_page(page_num)
        if depth == 1:
            counts[1] += 1
        else:
            counts[0] += 1
            _, children = _deserialize_internal(data)
            for c in children:
                self._count_pages(c, depth - 1, counts)

    def close(self):
        self.wal.commit(self.pm)
        self.wal.close()
        self.pm.close()

    def print_tree(self):
        """Return a string representation of the tree structure."""
        root, height, total_keys, _, _ = self._read_meta()
        lines = [f"BTree(height={height}, keys={total_keys})"]
        self._print_node(root, height, 0, lines)
        return '\n'.join(lines)

    def _print_node(self, page_num, depth, indent, lines):
        data = self.pm.read_page(page_num)
        prefix = '  ' * indent
        if depth == 1:
            keys, values, next_sib = _deserialize_leaf(data)
            lines.append(f"{prefix}Leaf[{page_num}]: {keys}")
        else:
            ikeys, children = _deserialize_internal(data)
            lines.append(f"{prefix}Internal[{page_num}]: {ikeys}")
            for c in children:
                self._print_node(c, depth - 1, indent + 1, lines)
