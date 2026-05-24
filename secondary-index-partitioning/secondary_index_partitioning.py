"""Secondary Index Partitioning: Document-Partitioned vs Term-Partitioned."""


class Document:
    """A document with a primary key and arbitrary fields."""

    def __init__(self, pk: str, fields: dict):
        self.pk = pk
        self.fields = dict(fields)


class Partition:
    """A single partition holding documents and optional local indexes."""

    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.documents: dict[str, dict] = {}
        self.local_index: dict[str, dict[object, set[str]]] = {}
        self.global_index: dict[str, dict[object, set[str]]] = {}

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def get_documents(self) -> dict[str, dict]:
        return dict(self.documents)


class OperationResult:
    """Result of a database operation with partition metrics."""

    def __init__(self, data, partitions_touched: int, operation: str):
        self.data = data
        self.partitions_touched = partitions_touched
        self.operation = operation


class DocumentPartitionedDB:
    """Database with document-partitioned (local) secondary indexes."""

    def __init__(self, num_partitions: int, indexed_fields: list[str]):
        self.num_partitions = num_partitions
        self.indexed_fields = indexed_fields
        self.partitions = [Partition(i) for i in range(num_partitions)]
        self._stats = {
            'total_reads': 0, 'total_writes': 0, 'total_queries': 0,
            'total_partitions_touched_reads': 0,
            'total_partitions_touched_writes': 0,
            'total_partitions_touched_queries': 0,
        }
        for p in self.partitions:
            for f in indexed_fields:
                p.local_index[f] = {}

    def _partition_for(self, pk: str) -> int:
        return hash(pk) % self.num_partitions

    def put(self, pk: str, fields: dict) -> OperationResult:
        pid = self._partition_for(pk)
        p = self.partitions[pid]
        # Remove old index entries if updating
        if pk in p.documents:
            old = p.documents[pk]
            for f in self.indexed_fields:
                if f in old:
                    s = p.local_index[f].get(old[f])
                    if s:
                        s.discard(pk)
                        if not s:
                            del p.local_index[f][old[f]]
        # Store document
        p.documents[pk] = dict(fields)
        # Add new index entries
        for f in self.indexed_fields:
            if f in fields:
                p.local_index[f].setdefault(fields[f], set()).add(pk)
        self._stats['total_writes'] += 1
        self._stats['total_partitions_touched_writes'] += 1
        return OperationResult(None, 1, 'write')

    def get(self, pk: str) -> OperationResult:
        pid = self._partition_for(pk)
        doc = self.partitions[pid].documents.get(pk)
        self._stats['total_reads'] += 1
        self._stats['total_partitions_touched_reads'] += 1
        return OperationResult(doc, 1, 'read')

    def delete(self, pk: str) -> OperationResult:
        pid = self._partition_for(pk)
        p = self.partitions[pid]
        if pk in p.documents:
            old = p.documents.pop(pk)
            for f in self.indexed_fields:
                if f in old:
                    s = p.local_index[f].get(old[f])
                    if s:
                        s.discard(pk)
                        if not s:
                            del p.local_index[f][old[f]]
        self._stats['total_writes'] += 1
        self._stats['total_partitions_touched_writes'] += 1
        return OperationResult(None, 1, 'write')

    def query_by_field(self, field: str, value) -> OperationResult:
        results = []
        for p in self.partitions:
            pks = p.local_index.get(field, {}).get(value, set())
            for pk in pks:
                results.append((pk, p.documents[pk]))
        self._stats['total_queries'] += 1
        self._stats['total_partitions_touched_queries'] += self.num_partitions
        return OperationResult(results, self.num_partitions, 'query')

    def get_partition(self, partition_id: int) -> Partition:
        return self.partitions[partition_id]

    def get_stats(self) -> dict:
        s = dict(self._stats)
        s['avg_partitions_per_read'] = (
            s['total_partitions_touched_reads'] / s['total_reads']
            if s['total_reads'] else 0.0
        )
        s['avg_partitions_per_write'] = (
            s['total_partitions_touched_writes'] / s['total_writes']
            if s['total_writes'] else 0.0
        )
        s['avg_partitions_per_query'] = (
            s['total_partitions_touched_queries'] / s['total_queries']
            if s['total_queries'] else 0.0
        )
        return s


class TermPartitionedDB:
    """Database with term-partitioned (global) secondary indexes."""

    def __init__(self, num_partitions: int, indexed_fields: list[str],
                 partition_by: str = "hash", async_index: bool = False):
        self.num_partitions = num_partitions
        self.indexed_fields = indexed_fields
        self.partition_by = partition_by
        self.async_index = async_index
        self.partitions = [Partition(i) for i in range(num_partitions)]
        self._pending: list[tuple] = []  # (action, field, value, pk)
        self._stats = {
            'total_reads': 0, 'total_writes': 0, 'total_queries': 0,
            'total_partitions_touched_reads': 0,
            'total_partitions_touched_writes': 0,
            'total_partitions_touched_queries': 0,
        }
        # Initialize global index structures on each partition
        for p in self.partitions:
            for f in indexed_fields:
                p.global_index[f] = {}
        # Range boundaries for range-based partitioning
        if partition_by == "range":
            self._range_boundaries = self._build_range_boundaries()

    def _build_range_boundaries(self) -> list[str]:
        # Split a-z into num_partitions ranges
        # boundaries[i] is the upper bound (exclusive) for partition i
        n = self.num_partitions
        letters = [chr(ord('a') + i) for i in range(26)]
        boundaries = []
        for i in range(1, n):
            idx = (26 * i) // n
            boundaries.append(letters[idx])
        return boundaries

    def _term_partition(self, value) -> int:
        if self.partition_by == "range":
            sv = str(value).lower()
            for i, bound in enumerate(self._range_boundaries):
                if sv < bound:
                    return i
            return self.num_partitions - 1
        return hash(value) % self.num_partitions

    def _partition_for(self, pk: str) -> int:
        return hash(pk) % self.num_partitions

    def _apply_index_op(self, action: str, field: str, value, pk: str):
        tid = self._term_partition(value)
        idx = self.partitions[tid].global_index[field]
        if action == 'add':
            idx.setdefault(value, set()).add(pk)
        elif action == 'remove':
            s = idx.get(value)
            if s:
                s.discard(pk)
                if not s:
                    del idx[value]

    def put(self, pk: str, fields: dict) -> OperationResult:
        pid = self._partition_for(pk)
        p = self.partitions[pid]
        touched_partitions = {pid}
        # Remove old index entries if updating
        if pk in p.documents:
            old = p.documents[pk]
            for f in self.indexed_fields:
                if f in old:
                    if self.async_index:
                        self._pending.append(('remove', f, old[f], pk))
                    else:
                        tid = self._term_partition(old[f])
                        touched_partitions.add(tid)
                        self._apply_index_op('remove', f, old[f], pk)
        # Store document
        p.documents[pk] = dict(fields)
        # Add new index entries
        for f in self.indexed_fields:
            if f in fields:
                if self.async_index:
                    self._pending.append(('add', f, fields[f], pk))
                else:
                    tid = self._term_partition(fields[f])
                    touched_partitions.add(tid)
                    self._apply_index_op('add', f, fields[f], pk)
        count = len(touched_partitions)
        self._stats['total_writes'] += 1
        self._stats['total_partitions_touched_writes'] += count
        return OperationResult(None, count, 'write')

    def get(self, pk: str) -> OperationResult:
        pid = self._partition_for(pk)
        doc = self.partitions[pid].documents.get(pk)
        self._stats['total_reads'] += 1
        self._stats['total_partitions_touched_reads'] += 1
        return OperationResult(doc, 1, 'read')

    def delete(self, pk: str) -> OperationResult:
        pid = self._partition_for(pk)
        p = self.partitions[pid]
        touched_partitions = {pid}
        if pk in p.documents:
            old = p.documents.pop(pk)
            for f in self.indexed_fields:
                if f in old:
                    if self.async_index:
                        self._pending.append(('remove', f, old[f], pk))
                    else:
                        tid = self._term_partition(old[f])
                        touched_partitions.add(tid)
                        self._apply_index_op('remove', f, old[f], pk)
        count = len(touched_partitions)
        self._stats['total_writes'] += 1
        self._stats['total_partitions_touched_writes'] += count
        return OperationResult(None, count, 'write')

    def query_by_field(self, field: str, value) -> OperationResult:
        tid = self._term_partition(value)
        idx = self.partitions[tid].global_index.get(field, {})
        pks = idx.get(value, set())
        touched_partitions = {tid}
        results = []
        for pk in pks:
            dpid = self._partition_for(pk)
            touched_partitions.add(dpid)
            doc = self.partitions[dpid].documents.get(pk)
            if doc is not None:
                results.append((pk, doc))
        count = len(touched_partitions)
        self._stats['total_queries'] += 1
        self._stats['total_partitions_touched_queries'] += count
        return OperationResult(results, count, 'query')

    def query_range(self, field: str, min_value, max_value) -> OperationResult:
        """Range query on a range-partitioned global index."""
        touched_partitions = set()
        pks_found = set()
        results = []
        for p in self.partitions:
            idx = p.global_index.get(field, {})
            found_any = False
            for val, pk_set in idx.items():
                if min_value <= str(val).lower() <= max_value:
                    found_any = True
                    for pk in pk_set:
                        if pk not in pks_found:
                            pks_found.add(pk)
                            dpid = self._partition_for(pk)
                            touched_partitions.add(dpid)
                            doc = self.partitions[dpid].documents.get(pk)
                            if doc is not None:
                                results.append((pk, doc))
            if found_any:
                touched_partitions.add(p.partition_id)
        count = len(touched_partitions)
        self._stats['total_queries'] += 1
        self._stats['total_partitions_touched_queries'] += count
        return OperationResult(results, count, 'query')

    def flush_index(self) -> int:
        """Apply all pending async index updates."""
        count = len(self._pending)
        for action, field, value, pk in self._pending:
            self._apply_index_op(action, field, value, pk)
        self._pending.clear()
        return count

    def get_partition(self, partition_id: int) -> Partition:
        return self.partitions[partition_id]

    def get_stats(self) -> dict:
        s = dict(self._stats)
        s['avg_partitions_per_read'] = (
            s['total_partitions_touched_reads'] / s['total_reads']
            if s['total_reads'] else 0.0
        )
        s['avg_partitions_per_write'] = (
            s['total_partitions_touched_writes'] / s['total_writes']
            if s['total_writes'] else 0.0
        )
        s['avg_partitions_per_query'] = (
            s['total_partitions_touched_queries'] / s['total_queries']
            if s['total_queries'] else 0.0
        )
        return s


def compare_strategies(documents: list[tuple[str, dict]],
                       queries: list[tuple[str, object]],
                       num_partitions: int,
                       indexed_fields: list[str]) -> dict:
    """Run the same workload on both strategies and compare metrics."""
    doc_db = DocumentPartitionedDB(num_partitions, indexed_fields)
    term_db = TermPartitionedDB(num_partitions, indexed_fields)

    for pk, fields in documents:
        doc_db.put(pk, fields)
        term_db.put(pk, fields)

    for field, value in queries:
        doc_db.query_by_field(field, value)
        term_db.query_by_field(field, value)

    doc_stats = doc_db.get_stats()
    term_stats = term_db.get_stats()

    return {
        'document_partitioned': doc_stats,
        'term_partitioned': term_stats,
        'summary': {
            'doc_avg_write_partitions': doc_stats['avg_partitions_per_write'],
            'doc_avg_query_partitions': doc_stats['avg_partitions_per_query'],
            'term_avg_write_partitions': term_stats['avg_partitions_per_write'],
            'term_avg_query_partitions': term_stats['avg_partitions_per_query'],
        }
    }
