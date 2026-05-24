"""Map-side join implementations for batch processing (DDIA Ch.10)."""

from collections import defaultdict


class JoinResult:
    """Result of a join operation."""

    def __init__(self, records: list, stats: dict):
        self.records = records
        self.stats = stats

    @property
    def count(self):
        return len(self.records)


def _merge_records(left, right, left_key, right_key):
    """Merge two records, resolving field name conflicts with left_/right_ prefixes."""
    result = {left_key: left[left_key]}
    left_fields = {k for k in left if k != left_key}
    right_fields = {k for k in right if k != right_key}
    conflicts = left_fields & right_fields

    for k, v in left.items():
        if k == left_key:
            continue
        if k in conflicts:
            result[f"left_{k}"] = v
        else:
            result[k] = v

    for k, v in right.items():
        if k == right_key:
            continue
        if k in conflicts:
            result[f"right_{k}"] = v
        else:
            result[k] = v

    return result


def _none_record(record, key, prefix_conflicts, conflicts):
    """Create a record with None fills for unmatched join."""
    result = {}
    for k, v in record.items():
        if k in conflicts and k != key:
            result[f"{prefix_conflicts}_{k}"] = v
        else:
            result[k] = v
    return result


def _build_hash_table(dataset, key):
    """Build a hash table from dataset, skipping records missing the key."""
    ht = defaultdict(list)
    skipped = 0
    for rec in dataset:
        if key not in rec:
            skipped += 1
            continue
        ht[rec[key]].append(rec)
    return ht, skipped


class BroadcastHashJoin:
    """Broadcast hash join: load small dataset into hash table, probe with large."""

    def __init__(self, small_dataset, small_key, large_key=None, num_mappers=4):
        self.small_key = small_key
        self.large_key = large_key or small_key
        self.num_mappers = num_mappers
        self.hash_table, self.skipped_small = _build_hash_table(small_dataset, small_key)
        self.small_size = len(small_dataset)

    def join(self, large_dataset, join_type="inner"):
        records = []
        hash_lookups = 0
        skipped_large = 0
        records_read_right = 0

        # Chunk large dataset across mappers
        chunks = [[] for _ in range(self.num_mappers)]
        for i, rec in enumerate(large_dataset):
            chunks[i % self.num_mappers].append((i, rec))

        for mapper_id, chunk in enumerate(chunks):
            for _orig_idx, rec in chunk:
                records_read_right += 1
                if self.large_key not in rec:
                    skipped_large += 1
                    continue
                key_val = rec[self.large_key]
                hash_lookups += 1
                matches = self.hash_table.get(key_val, [])
                if matches:
                    for small_rec in matches:
                        merged = _merge_records(small_rec, rec, self.small_key, self.large_key)
                        merged["_mapper_id"] = mapper_id
                        records.append(merged)
                elif join_type == "left":
                    # Unmatched large-side record
                    # Determine conflicts for None fill
                    if self.hash_table:
                        sample = next(iter(self.hash_table.values()))[0]
                        small_fields = {k for k in sample if k != self.small_key}
                    else:
                        small_fields = set()
                    large_fields = {k for k in rec if k != self.large_key}
                    conflicts = small_fields & large_fields

                    out = {self.large_key: key_val}
                    for k, v in rec.items():
                        if k == self.large_key:
                            continue
                        if k in conflicts:
                            out[f"right_{k}"] = v
                        else:
                            out[k] = v
                    for k in small_fields:
                        if k in conflicts:
                            out[f"left_{k}"] = None
                        else:
                            out[k] = None
                    out["_mapper_id"] = mapper_id
                    records.append(out)

        stats = {
            "records_read_left": self.small_size,
            "records_read_right": records_read_right,
            "hash_lookups": hash_lookups,
            "output_records": len(records),
            "hash_table_size": len(self.hash_table),
            "mappers_used": self.num_mappers,
            "skipped_records": self.skipped_small + skipped_large,
        }
        return JoinResult(records, stats)


class PartitionedHashJoin:
    """Partitioned hash join: partition both datasets, hash-join per partition."""

    def __init__(self, num_partitions, left_key, right_key=None):
        self.num_partitions = num_partitions
        self.left_key = left_key
        self.right_key = right_key or left_key

    def join(self, left_dataset, right_dataset, join_type="inner"):
        left_parts = partition_dataset(left_dataset, self.left_key, self.num_partitions)
        right_parts = partition_dataset(right_dataset, self.right_key, self.num_partitions)

        records = []
        total_hash_lookups = 0
        total_skipped = 0
        total_ht_size = 0

        for part_id in range(self.num_partitions):
            ht, skipped_left = _build_hash_table(left_parts[part_id], self.left_key)
            total_skipped += skipped_left
            total_ht_size += len(ht)

            # Determine field conflicts once per partition
            if left_parts[part_id] and right_parts[part_id]:
                left_fields = {k for k in left_parts[part_id][0] if k != self.left_key}
                right_fields = {k for k in right_parts[part_id][0] if k != self.right_key}
                conflicts = left_fields & right_fields
            else:
                conflicts = set()
                left_fields = set()

            matched_left_keys = set() if join_type == "left" else None

            for rec in right_parts[part_id]:
                if self.right_key not in rec:
                    total_skipped += 1
                    continue
                key_val = rec[self.right_key]
                total_hash_lookups += 1
                matches = ht.get(key_val, [])
                if matches:
                    if join_type == "left":
                        matched_left_keys.add(key_val)
                    for left_rec in matches:
                        merged = _merge_records(left_rec, rec, self.left_key, self.right_key)
                        merged["_mapper_id"] = part_id
                        records.append(merged)

            if join_type == "left":
                # Emit unmatched left records
                for left_rec in left_parts[part_id]:
                    if self.left_key not in left_rec:
                        continue
                    if left_rec[self.left_key] not in matched_left_keys:
                        out = {self.left_key: left_rec[self.left_key]}
                        for k, v in left_rec.items():
                            if k == self.left_key:
                                continue
                            if k in conflicts:
                                out[f"left_{k}"] = v
                            else:
                                out[k] = v
                        # Get right-side fields for None fill
                        if right_parts[part_id]:
                            right_fields_actual = {k for k in right_parts[part_id][0] if k != self.right_key}
                        else:
                            right_fields_actual = set()
                        for k in right_fields_actual:
                            if k in conflicts:
                                out[f"right_{k}"] = None
                            else:
                                out[k] = None
                        out["_mapper_id"] = part_id
                        records.append(out)

        stats = {
            "records_read_left": len(left_dataset),
            "records_read_right": len(right_dataset),
            "hash_lookups": total_hash_lookups,
            "output_records": len(records),
            "hash_table_size": total_ht_size,
            "mappers_used": self.num_partitions,
            "skipped_records": total_skipped,
        }
        return JoinResult(records, stats)


class SortMergeJoin:
    """Sort-merge join: linear merge of two sorted datasets."""

    def __init__(self, left_key, right_key=None):
        self.left_key = left_key
        self.right_key = right_key or left_key

    def join(self, left_dataset, right_dataset, join_type="inner"):
        sorted_flag = False

        # Check if sorted, sort if needed
        left = left_dataset
        right = right_dataset
        if not _is_sorted(left, self.left_key):
            left = sort_dataset(left, self.left_key)
            sorted_flag = True
        if not _is_sorted(right, self.right_key):
            right = sort_dataset(right, self.right_key)
            sorted_flag = True

        # Filter out records missing key
        skipped = 0
        left_clean = []
        for r in left:
            if self.left_key in r:
                left_clean.append(r)
            else:
                skipped += 1
        right_clean = []
        for r in right:
            if self.right_key in r:
                right_clean.append(r)
            else:
                skipped += 1

        records = []
        comparisons = 0
        li, ri = 0, 0

        while li < len(left_clean) and ri < len(right_clean):
            lk = left_clean[li][self.left_key]
            rk = right_clean[ri][self.right_key]
            comparisons += 1

            if lk < rk:
                if join_type == "left":
                    records.append(self._left_unmatched(left_clean[li], right_clean))
                li += 1
            elif lk > rk:
                ri += 1
            else:
                # Collect all left records with this key
                left_group = []
                while li < len(left_clean) and left_clean[li][self.left_key] == lk:
                    left_group.append(left_clean[li])
                    li += 1
                # Collect all right records with this key
                right_group = []
                while ri < len(right_clean) and right_clean[ri][self.right_key] == lk:
                    right_group.append(right_clean[ri])
                    ri += 1
                # Cartesian product
                comparisons += len(left_group) + len(right_group) - 2
                for lr in left_group:
                    for rr in right_group:
                        merged = _merge_records(lr, rr, self.left_key, self.right_key)
                        merged["_mapper_id"] = 0
                        records.append(merged)

        # Remaining unmatched left records for left join
        if join_type == "left":
            while li < len(left_clean):
                records.append(self._left_unmatched(left_clean[li], right_clean))
                li += 1

        stats = {
            "records_read_left": len(left_dataset),
            "records_read_right": len(right_dataset),
            "comparisons": comparisons,
            "output_records": len(records),
            "mappers_used": 1,
            "skipped_records": skipped,
            "sorted_input": not sorted_flag,
        }
        return JoinResult(records, stats)

    def _left_unmatched(self, left_rec, right_sample):
        """Emit unmatched left record with None fills."""
        if right_sample:
            right_fields = {k for k in right_sample[0] if k != self.right_key}
        else:
            right_fields = set()
        left_fields = {k for k in left_rec if k != self.left_key}
        conflicts = left_fields & right_fields

        out = {self.left_key: left_rec[self.left_key]}
        for k, v in left_rec.items():
            if k == self.left_key:
                continue
            if k in conflicts:
                out[f"left_{k}"] = v
            else:
                out[k] = v
        for k in right_fields:
            if k in conflicts:
                out[f"right_{k}"] = None
            else:
                out[k] = None
        out["_mapper_id"] = 0
        return out


def _is_sorted(dataset, key):
    """Check if dataset is sorted by key."""
    prev = None
    for rec in dataset:
        if key not in rec:
            continue
        if prev is not None and rec[key] < prev:
            return False
        prev = rec[key]
    return True


def partition_dataset(dataset, key, num_partitions):
    """Partition a dataset by hashing a key field."""
    partitions = [[] for _ in range(num_partitions)]
    for rec in dataset:
        if key in rec:
            p = hash(rec[key]) % num_partitions
            partitions[p].append(rec)
    return partitions


def sort_dataset(dataset, key):
    """Sort a dataset by a key field (returns new list)."""
    has_key = [r for r in dataset if key in r]
    missing = [r for r in dataset if key not in r]
    return sorted(has_key, key=lambda r: r[key]) + missing


def compare_join_strategies(left, right, join_key):
    """Run all three join strategies and compare results."""
    bhj = BroadcastHashJoin(left, small_key=join_key, num_mappers=2)
    bhj_result = bhj.join(right, join_type="inner")

    phj = PartitionedHashJoin(num_partitions=3, left_key=join_key)
    phj_result = phj.join(left, right, join_type="inner")

    smj = SortMergeJoin(left_key=join_key)
    smj_result = smj.join(left, right, join_type="inner")

    # Verify all produce same records (ignoring _mapper_id and order)
    def normalize(records):
        normalized = []
        for r in records:
            nr = {k: v for k, v in r.items() if k != "_mapper_id"}
            normalized.append(tuple(sorted(nr.items())))
        return sorted(normalized)

    b_norm = normalize(bhj_result.records)
    p_norm = normalize(phj_result.records)
    s_norm = normalize(smj_result.records)

    verification = (b_norm == p_norm == s_norm)

    return {
        "broadcast": bhj_result.stats,
        "partitioned": phj_result.stats,
        "sort_merge": smj_result.stats,
        "verification": verification,
    }
