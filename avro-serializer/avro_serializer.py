"""Avro-style schema-based binary serializer with schema evolution."""

import io
import struct

PRIMITIVES = {"null", "boolean", "int", "long", "float", "double", "string", "bytes"}
_NO_DEFAULT = object()
PROMOTIONS = {
    ("int", "long"), ("int", "float"), ("int", "double"),
    ("long", "float"), ("long", "double"),
    ("float", "double"),
}


class SchemaError(Exception):
    """Raised for invalid schema definitions."""
    pass


class SchemaCompatibilityError(Exception):
    """Raised when schemas are incompatible for resolution."""
    pass


# --- Zigzag / Varint ---

def zigzag_encode(n):
    """Encode signed int to unsigned via zigzag."""
    return (n << 1) ^ (n >> 63)


def zigzag_decode(n):
    """Decode unsigned zigzag back to signed."""
    return (n >> 1) ^ -(n & 1)


def write_varint(buf, n):
    """Write unsigned int as varint to buffer."""
    while (n & ~0x7F) != 0:
        buf.write(bytes([(n & 0x7F) | 0x80]))
        n >>= 7
    buf.write(bytes([n & 0x7F]))


def read_varint(buf):
    """Read varint from buffer, return unsigned int."""
    result = 0
    shift = 0
    while True:
        b = buf.read(1)
        if not b:
            raise ValueError("Unexpected end of data reading varint")
        byte = b[0]
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result


def write_long(buf, n):
    write_varint(buf, zigzag_encode(n))


def read_long(buf):
    return zigzag_decode(read_varint(buf))


# --- Schema ---

class Schema:
    """Parse and validate an Avro-like schema definition."""

    def __init__(self, definition):
        self._def = definition
        self._parsed = self._parse(definition)

    def _parse(self, defn):
        if isinstance(defn, str):
            if defn not in PRIMITIVES:
                raise SchemaError(f"Unknown primitive type: {defn}")
            return {"type": defn}
        if isinstance(defn, list):
            # Union
            if len(defn) == 0:
                raise SchemaError("Union must have at least one type")
            types = []
            seen = set()
            for t in defn:
                s = Schema(t)
                key = s.name if s.name else s.type_name
                if key in seen:
                    raise SchemaError(f"Duplicate type in union: {key}")
                seen.add(key)
                types.append(s)
            return {"type": "union", "types": types}
        if isinstance(defn, dict):
            t = defn.get("type")
            if t == "record":
                name = defn.get("name")
                if not name:
                    raise SchemaError("Record must have a name")
                fields = []
                seen = set()
                for f in defn.get("fields", []):
                    fname = f["name"]
                    if fname in seen:
                        raise SchemaError(f"Duplicate field name: {fname}")
                    seen.add(fname)
                    field = {"name": fname, "type": Schema(f["type"])}
                    if "default" in f:
                        field["default"] = f["default"]
                    fields.append(field)
                return {"type": "record", "name": name, "fields": fields}
            if t == "array":
                return {"type": "array", "items": Schema(defn["items"])}
            if t == "map":
                return {"type": "map", "values": Schema(defn["values"])}
            if t == "enum":
                name = defn.get("name")
                if not name:
                    raise SchemaError("Enum must have a name")
                symbols = defn.get("symbols", [])
                if not symbols:
                    raise SchemaError("Enum must have at least one symbol")
                parsed = {"type": "enum", "name": name, "symbols": list(symbols)}
                if "default" in defn:
                    parsed["default"] = defn["default"]
                return parsed
            if t == "fixed":
                name = defn.get("name")
                if not name:
                    raise SchemaError("Fixed must have a name")
                size = defn.get("size")
                if size is None or not isinstance(size, int) or size < 0:
                    raise SchemaError("Fixed must have a non-negative integer size")
                return {"type": "fixed", "name": name, "size": size}
            if isinstance(t, str) and t in PRIMITIVES:
                return {"type": t}
            raise SchemaError(f"Unknown schema type: {t}")
        raise SchemaError(f"Invalid schema definition: {defn}")

    @property
    def type_name(self):
        return self._parsed["type"]

    @property
    def name(self):
        return self._parsed.get("name")

    @property
    def fields(self):
        return self._parsed.get("fields", [])

    @property
    def items(self):
        return self._parsed.get("items")

    @property
    def values(self):
        return self._parsed.get("values")

    @property
    def symbols(self):
        return self._parsed.get("symbols", [])

    @property
    def size(self):
        return self._parsed.get("size")

    @property
    def union_types(self):
        return self._parsed.get("types", [])

    @property
    def default(self):
        return self._parsed.get("default")

    def has_default(self):
        return "default" in self._parsed

    def to_dict(self):
        return self._def

    def _canonical_form(self):
        return self._canonicalize(self._parsed)

    def _canonicalize(self, p):
        t = p["type"]
        if t in PRIMITIVES:
            return t
        if t == "union":
            return ("union", tuple(s._canonical_form() for s in p["types"]))
        if t == "record":
            fields = tuple(
                (f["name"], f["type"]._canonical_form(), f.get("default", _NO_DEFAULT))
                for f in p["fields"]
            )
            return ("record", p["name"], fields)
        if t == "array":
            return ("array", p["items"]._canonical_form())
        if t == "map":
            return ("map", p["values"]._canonical_form())
        if t == "enum":
            return ("enum", p["name"], tuple(p["symbols"]))
        if t == "fixed":
            return ("fixed", p["name"], p["size"])
        return t

    def __eq__(self, other):
        if not isinstance(other, Schema):
            return NotImplemented
        return self._canonical_form() == other._canonical_form()

    def __hash__(self):
        return hash(self._canonical_form())

    def __repr__(self):
        return f"Schema({self._def!r})"


# --- Encoder ---

class AvroEncoder:
    """Encode Python values to binary according to a writer schema."""

    def __init__(self, writer_schema):
        self.schema = writer_schema

    def encode(self, value):
        buf = io.BytesIO()
        self._encode(buf, self.schema, value)
        return buf.getvalue()

    def _encode(self, buf, schema, value):
        t = schema.type_name
        if t == "null":
            if value is not None:
                raise ValueError(f"Expected None for null schema, got {type(value)}")
        elif t == "boolean":
            buf.write(b'\x01' if value else b'\x00')
        elif t == "int":
            if value < -2147483648 or value > 2147483647:
                raise ValueError(f"Value {value} out of range for Avro int (32-bit signed)")
            write_long(buf, value)
        elif t == "long":
            if value < -9223372036854775808 or value > 9223372036854775807:
                raise ValueError(f"Value {value} out of range for Avro long (64-bit signed)")
            write_long(buf, value)
        elif t == "float":
            buf.write(struct.pack('<f', value))
        elif t == "double":
            buf.write(struct.pack('<d', value))
        elif t == "string":
            encoded = value.encode('utf-8')
            write_long(buf, len(encoded))
            buf.write(encoded)
        elif t == "bytes":
            write_long(buf, len(value))
            buf.write(value)
        elif t == "record":
            for field in schema.fields:
                self._encode(buf, field["type"], value[field["name"]])
        elif t == "array":
            items = schema.items
            if value:
                write_long(buf, len(value))
                for item in value:
                    self._encode(buf, items, item)
            write_long(buf, 0)  # terminator
        elif t == "map":
            vals_schema = schema.values
            if value:
                write_long(buf, len(value))
                for k, v in value.items():
                    # encode key as string
                    kb = k.encode('utf-8')
                    write_long(buf, len(kb))
                    buf.write(kb)
                    self._encode(buf, vals_schema, v)
            write_long(buf, 0)  # terminator
        elif t == "union":
            union_types = schema.union_types
            idx = self._match_union(union_types, value)
            write_long(buf, idx)
            self._encode(buf, union_types[idx], value)
        elif t == "enum":
            idx = schema.symbols.index(value)
            write_long(buf, idx)
        elif t == "fixed":
            if len(value) != schema.size:
                raise ValueError(f"Fixed requires exactly {schema.size} bytes, got {len(value)}")
            buf.write(value)
        else:
            raise ValueError(f"Unknown type: {t}")

    def _match_union(self, union_types, value):
        """Find which branch of a union matches the value."""
        for i, s in enumerate(union_types):
            t = s.type_name
            if value is None and t == "null":
                return i
            if isinstance(value, bool) and t == "boolean":
                return i
            if isinstance(value, int) and not isinstance(value, bool) and t in ("int", "long"):
                return i
            if isinstance(value, float) and t in ("float", "double"):
                return i
            if isinstance(value, str) and t == "string":
                return i
            if isinstance(value, (bytes, bytearray)) and t == "bytes":
                return i
            if isinstance(value, dict) and t == "record":
                field_names = {f["name"] for f in s.fields}
                if set(value.keys()).issubset(field_names):
                    return i
            if isinstance(value, list) and t == "array":
                return i
            if isinstance(value, dict) and t == "map":
                return i
            if isinstance(value, str) and t == "enum":
                if value in s.symbols:
                    return i
            if isinstance(value, (bytes, bytearray)) and t == "fixed":
                if len(value) == s.size:
                    return i
        raise ValueError(f"No matching union branch for value: {value!r}")


# --- Decoder ---

class AvroDecoder:
    """Decode binary data using writer schema, optionally resolving to reader schema."""

    def __init__(self, writer_schema, reader_schema=None):
        self.writer_schema = writer_schema
        self.reader_schema = reader_schema or writer_schema

    def decode(self, data):
        buf = io.BytesIO(data)
        return self._decode(buf, self.writer_schema, self.reader_schema)

    def _decode(self, buf, writer, reader):
        wt = writer.type_name
        rt = reader.type_name

        # Union handling
        if wt == "union":
            idx = read_long(buf)
            actual_writer = writer.union_types[idx]
            if rt == "union":
                # Find matching branch in reader union
                matched = self._match_reader_union(actual_writer, reader.union_types)
                return self._decode(buf, actual_writer, matched)
            else:
                return self._decode(buf, actual_writer, reader)

        if rt == "union" and wt != "union":
            # Writer is not union but reader is — find matching branch
            matched = self._match_reader_union(writer, reader.union_types)
            return self._decode(buf, writer, matched)

        # Primitives
        if wt in PRIMITIVES and rt in PRIMITIVES:
            val = self._read_primitive(buf, wt)
            if wt == rt:
                return val
            if (wt, rt) in PROMOTIONS:
                return self._promote(val, rt)
            raise SchemaCompatibilityError(f"Cannot resolve {wt} to {rt}")

        # Record
        if wt == "record" and rt == "record":
            if writer.name != reader.name:
                raise SchemaCompatibilityError(
                    f"Record name mismatch: {writer.name} vs {reader.name}")
            return self._resolve_record(buf, writer, reader)

        # Array
        if wt == "array" and rt == "array":
            return self._read_array(buf, writer.items, reader.items)

        # Map
        if wt == "map" and rt == "map":
            return self._read_map(buf, writer.values, reader.values)

        # Enum
        if wt == "enum" and rt == "enum":
            if writer.name != reader.name:
                raise SchemaCompatibilityError(
                    f"Enum name mismatch: {writer.name} vs {reader.name}")
            idx = read_long(buf)
            symbol = writer.symbols[idx]
            if symbol in reader.symbols:
                return symbol
            if reader.has_default():
                return reader.default
            raise SchemaCompatibilityError(
                f"Enum symbol '{symbol}' not in reader schema and no default")

        # Fixed
        if wt == "fixed" and rt == "fixed":
            if writer.name != reader.name:
                raise SchemaCompatibilityError(
                    f"Fixed name mismatch: {writer.name} vs {reader.name}")
            if writer.size != reader.size:
                raise SchemaCompatibilityError(
                    f"Fixed size mismatch: {writer.size} vs {reader.size}")
            return buf.read(writer.size)

        raise SchemaCompatibilityError(f"Cannot resolve {wt} to {rt}")

    def _read_primitive(self, buf, t):
        if t == "null":
            return None
        if t == "boolean":
            b = buf.read(1)
            if not b:
                raise ValueError("Unexpected end of data")
            return b[0] != 0
        if t in ("int", "long"):
            return read_long(buf)
        if t == "float":
            data = buf.read(4)
            return struct.unpack('<f', data)[0]
        if t == "double":
            data = buf.read(8)
            return struct.unpack('<d', data)[0]
        if t == "string":
            length = read_long(buf)
            return buf.read(length).decode('utf-8')
        if t == "bytes":
            length = read_long(buf)
            return buf.read(length)

    def _promote(self, val, target):
        if target in ("long", "float", "double"):
            return int(val) if target == "long" else float(val)
        raise SchemaCompatibilityError(f"Cannot promote to {target}")

    def _resolve_record(self, buf, writer, reader):
        # Read all writer fields into a dict keyed by name
        writer_values = {}
        reader_field_map = {f["name"]: f for f in reader.fields}
        writer_field_map = {f["name"]: f for f in writer.fields}

        for wf in writer.fields:
            fname = wf["name"]
            if fname in reader_field_map:
                rf = reader_field_map[fname]
                writer_values[fname] = self._decode(buf, wf["type"], rf["type"])
            else:
                # Skip: read and discard
                self._skip(buf, wf["type"])

        # Build result in reader field order
        result = {}
        for rf in reader.fields:
            fname = rf["name"]
            if fname in writer_values:
                result[fname] = writer_values[fname]
            elif "default" in rf:
                result[fname] = rf["default"]
            else:
                raise SchemaCompatibilityError(
                    f"Reader field '{fname}' has no default and is missing from writer")
        return result

    def _skip(self, buf, schema):
        """Read and discard a value according to schema."""
        t = schema.type_name
        if t == "null":
            pass
        elif t == "boolean":
            buf.read(1)
        elif t in ("int", "long"):
            read_long(buf)
        elif t == "float":
            buf.read(4)
        elif t == "double":
            buf.read(8)
        elif t == "string" or t == "bytes":
            length = read_long(buf)
            buf.read(length)
        elif t == "record":
            for f in schema.fields:
                self._skip(buf, f["type"])
        elif t == "array":
            while True:
                count = read_long(buf)
                if count == 0:
                    break
                if count < 0:
                    count = -count
                    read_long(buf)  # byte size of block
                for _ in range(count):
                    self._skip(buf, schema.items)
        elif t == "map":
            while True:
                count = read_long(buf)
                if count == 0:
                    break
                if count < 0:
                    count = -count
                    read_long(buf)  # byte size of block
                for _ in range(count):
                    klen = read_long(buf)
                    buf.read(klen)
                    self._skip(buf, schema.values)
        elif t == "union":
            idx = read_long(buf)
            self._skip(buf, schema.union_types[idx])
        elif t == "enum":
            read_long(buf)
        elif t == "fixed":
            buf.read(schema.size)

    def _read_array(self, buf, writer_items, reader_items):
        result = []
        while True:
            count = read_long(buf)
            if count == 0:
                break
            if count < 0:
                count = -count
                read_long(buf)  # byte size of block
            for _ in range(count):
                result.append(self._decode(buf, writer_items, reader_items))
        return result

    def _read_map(self, buf, writer_values, reader_values):
        result = {}
        while True:
            count = read_long(buf)
            if count == 0:
                break
            if count < 0:
                count = -count
                read_long(buf)  # byte size of block
            for _ in range(count):
                klen = read_long(buf)
                key = buf.read(klen).decode('utf-8')
                result[key] = self._decode(buf, writer_values, reader_values)
        return result

    def _match_reader_union(self, writer_schema, reader_union_types):
        wt = writer_schema.type_name
        wn = writer_schema.name
        for rs in reader_union_types:
            if wt == rs.type_name:
                if wt in PRIMITIVES or wt in ("array", "map"):
                    return rs
                if wt == "fixed" and wn == rs.name and writer_schema.size == rs.size:
                    return rs
                if wn and wn == rs.name:
                    return rs
            # Check promotions
            if wt in PRIMITIVES and rs.type_name in PRIMITIVES:
                if (wt, rs.type_name) in PROMOTIONS:
                    return rs
        raise SchemaCompatibilityError(
            f"No matching branch in reader union for writer type {wt}"
            + (f" ({wn})" if wn else ""))


# --- Compatibility ---

def check_compatibility(writer_schema, reader_schema):
    """Check backward/forward/full compatibility between schemas."""
    errors = []
    backward = _check_one_direction(writer_schema, reader_schema, errors, "backward")
    fwd_errors = []
    forward = _check_one_direction(reader_schema, writer_schema, fwd_errors, "forward")
    errors.extend(fwd_errors)
    return {
        "backward_compatible": backward,
        "forward_compatible": forward,
        "full_compatible": backward and forward,
        "errors": errors,
    }


def _check_one_direction(writer, reader, errors, direction):
    """Check if reader can read data from writer."""
    try:
        _resolve_check(writer, reader)
        return True
    except SchemaCompatibilityError as e:
        errors.append(f"{direction}: {e}")
        return False


def _resolve_check(writer, reader):
    """Dry-run schema resolution to check compatibility."""
    wt = writer.type_name
    rt = reader.type_name

    if wt == "union":
        for branch in writer.union_types:
            _resolve_check(branch, reader)
        return

    if rt == "union":
        # Writer type must match some reader branch
        for rs in reader.union_types:
            try:
                _resolve_check(writer, rs)
                return
            except SchemaCompatibilityError:
                continue
        # Check promotions
        raise SchemaCompatibilityError(
            f"Writer type {wt} has no match in reader union")

    if wt in PRIMITIVES and rt in PRIMITIVES:
        if wt == rt or (wt, rt) in PROMOTIONS:
            return
        raise SchemaCompatibilityError(f"Incompatible primitives: {wt} vs {rt}")

    if wt == "record" and rt == "record":
        if writer.name != reader.name:
            raise SchemaCompatibilityError(
                f"Record name mismatch: {writer.name} vs {reader.name}")
        writer_field_names = {f["name"] for f in writer.fields}
        writer_field_map = {f["name"]: f for f in writer.fields}
        for rf in reader.fields:
            if rf["name"] in writer_field_names:
                _resolve_check(writer_field_map[rf["name"]]["type"], rf["type"])
            elif "default" not in rf:
                raise SchemaCompatibilityError(
                    f"Reader field '{rf['name']}' missing from writer with no default")
        return

    if wt == "array" and rt == "array":
        _resolve_check(writer.items, reader.items)
        return

    if wt == "map" and rt == "map":
        _resolve_check(writer.values, reader.values)
        return

    if wt == "enum" and rt == "enum":
        if writer.name != reader.name:
            raise SchemaCompatibilityError(
                f"Enum name mismatch: {writer.name} vs {reader.name}")
        for sym in writer.symbols:
            if sym not in reader.symbols and not reader.has_default():
                raise SchemaCompatibilityError(
                    f"Writer enum symbol '{sym}' not in reader and no default")
        return

    if wt == "fixed" and rt == "fixed":
        if writer.name != reader.name:
            raise SchemaCompatibilityError(
                f"Fixed name mismatch: {writer.name} vs {reader.name}")
        if writer.size != reader.size:
            raise SchemaCompatibilityError(
                f"Fixed size mismatch: {writer.size} vs {reader.size}")
        return

    raise SchemaCompatibilityError(f"Incompatible types: {wt} vs {rt}")


# --- Schema Registry ---

class SchemaRegistry:
    """Simple in-memory schema registry with optional compatibility checking."""

    def __init__(self):
        self._schemas = {}
        self._next_id = 1
        self._subjects = {}  # subject -> [schema_id, ...]

    def register(self, schema, subject=None):
        if subject and subject in self._subjects:
            prev_id = self._subjects[subject][-1]
            prev_schema = self._schemas[prev_id]
            compat = check_compatibility(prev_schema, schema)
            if not compat["backward_compatible"]:
                raise ValueError(
                    f"Schema not backward compatible with {subject} v{len(self._subjects[subject])}: "
                    + "; ".join(compat["errors"])
                )
        sid = self._next_id
        self._schemas[sid] = schema
        self._next_id += 1
        if subject:
            self._subjects.setdefault(subject, []).append(sid)
        return sid

    def get(self, schema_id):
        if schema_id not in self._schemas:
            raise KeyError(f"Schema ID {schema_id} not found")
        return self._schemas[schema_id]

    def encode_with_id(self, schema_id, value):
        schema = self.get(schema_id)
        encoder = AvroEncoder(schema)
        data = encoder.encode(value)
        return struct.pack('>I', schema_id) + data

    def decode_with_id(self, data, reader_schema=None):
        schema_id = struct.unpack('>I', data[:4])[0]
        writer_schema = self.get(schema_id)
        decoder = AvroDecoder(writer_schema, reader_schema)
        return decoder.decode(data[4:])
