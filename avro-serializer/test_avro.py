"""Comprehensive tests for avro_serializer."""
import io
import time
from avro_serializer import (
    Schema, AvroEncoder, AvroDecoder, SchemaRegistry,
    SchemaError, SchemaCompatibilityError,
    check_compatibility, zigzag_encode, zigzag_decode,
)

def test_primitives():
    print("Primitives:")
    for typ, val in [
        ("null", None), ("boolean", True), ("boolean", False),
        ("int", 0), ("int", -1), ("int", 42), ("int", 2147483647), ("int", -2147483648),
        ("long", 2**40), ("long", -(2**40)),
        ("float", 3.14), ("double", 2.718281828),
        ("string", "hello"), ("string", ""),
        ("bytes", b"\x00\x01\x02"), ("bytes", b""),
    ]:
        s = Schema(typ)
        enc = AvroEncoder(s)
        dec = AvroDecoder(s)
        encoded = enc.encode(val)
        decoded = dec.decode(encoded)
        if typ == "float":
            assert abs(decoded - val) < 1e-6, f"{typ}: {decoded} != {val}"
        else:
            assert decoded == val, f"{typ}: {decoded} != {val}"
        print(f"  {typ}({val!r}): {len(encoded)} bytes OK")

def test_zigzag():
    print("Zigzag:")
    for n in [0, -1, 1, -2, 2, 2147483647, -2147483648]:
        assert zigzag_decode(zigzag_encode(n)) == n
        print(f"  {n} -> {zigzag_encode(n)} OK")

def test_record():
    print("Record:")
    schema = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"},
    ]})
    enc = AvroEncoder(schema)
    data = enc.encode({"id": 1, "name": "Alice", "email": "alice@example.com"})
    dec = AvroDecoder(schema)
    result = dec.decode(data)
    assert result == {"id": 1, "name": "Alice", "email": "alice@example.com"}
    assert b"email" not in data
    print(f"  Record: {len(data)} bytes, no field names OK")

def test_array():
    print("Array:")
    arr_schema = Schema({"type": "array", "items": "int"})
    enc = AvroEncoder(arr_schema)
    data = enc.encode([1, 2, 3, 4, 5])
    assert AvroDecoder(arr_schema).decode(data) == [1, 2, 3, 4, 5]
    assert AvroDecoder(arr_schema).decode(enc.encode([])) == []
    print("  Array OK")

def test_map():
    print("Map:")
    map_schema = Schema({"type": "map", "values": "string"})
    enc = AvroEncoder(map_schema)
    data = enc.encode({"key1": "val1", "key2": "val2"})
    assert AvroDecoder(map_schema).decode(data) == {"key1": "val1", "key2": "val2"}
    assert AvroDecoder(map_schema).decode(enc.encode({})) == {}
    print("  Map OK")

def test_union():
    print("Union:")
    union_schema = Schema(["null", "string"])
    enc = AvroEncoder(union_schema)
    assert AvroDecoder(union_schema).decode(enc.encode(None)) is None
    assert AvroDecoder(union_schema).decode(enc.encode("hello")) == "hello"
    print("  Union OK")

def test_enum():
    print("Enum:")
    enum_schema = Schema({"type": "enum", "name": "Color", "symbols": ["RED", "GREEN", "BLUE"]})
    enc = AvroEncoder(enum_schema)
    for sym in ["RED", "GREEN", "BLUE"]:
        assert AvroDecoder(enum_schema).decode(enc.encode(sym)) == sym
    print("  Enum OK")

def test_schema_evolution_add_field():
    print("Schema evolution (add field with default, remove field):")
    v1 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"},
    ]})
    v2 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int", "default": 0},
    ]})
    data = AvroEncoder(v1).encode({"id": 1, "name": "Alice", "email": "alice@example.com"})
    result = AvroDecoder(v1, v2).decode(data)
    assert result == {"id": 1, "name": "Alice", "age": 0}, f"Got {result}"
    print("  v1->v2 backward compat OK")

def test_type_promotion():
    print("Type promotion:")
    int_s = Schema("int")
    long_s = Schema("long")
    float_s = Schema("float")
    double_s = Schema("double")
    data = AvroEncoder(int_s).encode(42)
    assert AvroDecoder(int_s, long_s).decode(data) == 42
    assert AvroDecoder(int_s, float_s).decode(data) == 42.0
    assert AvroDecoder(int_s, double_s).decode(data) == 42.0
    data_long = AvroEncoder(long_s).encode(100)
    assert AvroDecoder(long_s, double_s).decode(data_long) == 100.0
    data_float = AvroEncoder(float_s).encode(1.5)
    assert abs(AvroDecoder(float_s, double_s).decode(data_float) - 1.5) < 1e-6
    print("  int->long, int->float, int->double, long->double, float->double OK")

def test_incompatibility():
    print("Incompatibility:")
    v1 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
    ]})
    v3 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "required_field", "type": "int"},
    ]})
    data = AvroEncoder(v1).encode({"id": 1, "name": "Alice"})
    try:
        AvroDecoder(v1, v3).decode(data)
        assert False, "Should have raised"
    except SchemaCompatibilityError:
        print("  Correctly raised SchemaCompatibilityError")

def test_compatibility_check():
    print("Compatibility check:")
    v1 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"},
    ]})
    v2 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int", "default": 0},
    ]})
    compat = check_compatibility(v1, v2)
    assert compat["backward_compatible"] == True
    print(f"  backward={compat['backward_compatible']}, forward={compat['forward_compatible']}, full={compat['full_compatible']}")

def test_schema_registry():
    print("Schema registry:")
    v1 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"},
    ]})
    v2 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int", "default": 0},
    ]})
    registry = SchemaRegistry()
    sid = registry.register(v1)
    encoded = registry.encode_with_id(sid, {"id": 1, "name": "Alice", "email": "a@b.com"})
    decoded = registry.decode_with_id(encoded, reader_schema=v2)
    assert decoded["age"] == 0
    assert decoded["id"] == 1
    print(f"  Registry OK (id={sid})")

def test_nested():
    print("Nested:")
    nested = Schema({"type": "record", "name": "Order", "fields": [
        {"name": "id", "type": "int"},
        {"name": "items", "type": {"type": "array", "items": "string"}},
        {"name": "meta", "type": {"type": "map", "values": "int"}},
    ]})
    val = {"id": 1, "items": ["a", "b"], "meta": {"x": 10, "y": 20}}
    assert AvroDecoder(nested).decode(AvroEncoder(nested).encode(val)) == val
    print("  Nested record with array+map OK")

    inner = {"type": "record", "name": "Address", "fields": [
        {"name": "city", "type": "string"},
    ]}
    outer = Schema({"type": "record", "name": "Person", "fields": [
        {"name": "name", "type": "string"},
        {"name": "addr", "type": inner},
    ]})
    val2 = {"name": "Bob", "addr": {"city": "NYC"}}
    assert AvroDecoder(outer).decode(AvroEncoder(outer).encode(val2)) == val2
    print("  Record in record OK")

def test_compact_encoding():
    print("Compact encoding:")
    schema = Schema({"type": "record", "name": "R", "fields": [
        {"name": "id", "type": "int"},
        {"name": "value", "type": "string"},
    ]})
    data = AvroEncoder(schema).encode({"id": 1, "value": "test"})
    assert b"id" not in data
    assert b"value" not in data
    print(f"  No field names in {len(data)} bytes OK")

def test_streaming():
    print("Streaming:")
    int_s = Schema("int")
    buf_data = AvroEncoder(int_s).encode(42) + AvroEncoder(int_s).encode(99)
    stream = io.BytesIO(buf_data)
    d = AvroDecoder(int_s)
    v1 = d._decode(stream, int_s, int_s)
    v2 = d._decode(stream, int_s, int_s)
    assert v1 == 42 and v2 == 99
    assert stream.read() == b""
    print("  Sequential decode, exact bytes consumed OK")

def test_enum_evolution():
    print("Enum evolution:")
    writer_enum = Schema({"type": "enum", "name": "Color", "symbols": ["RED", "GREEN", "BLUE", "YELLOW"]})
    reader_enum = Schema({"type": "enum", "name": "Color", "symbols": ["RED", "GREEN", "BLUE"], "default": "RED"})
    data = AvroEncoder(writer_enum).encode("YELLOW")
    result = AvroDecoder(writer_enum, reader_enum).decode(data)
    assert result == "RED"
    print("  Enum default fallback OK")

def test_performance():
    print("Performance:")
    perf_schema = Schema({"type": "record", "name": "R", "fields": [
        {"name": "a", "type": "int"},
        {"name": "b", "type": "string"},
        {"name": "c", "type": "double"},
    ]})
    enc = AvroEncoder(perf_schema)
    t0 = time.time()
    records = [enc.encode({"a": i, "b": f"val{i}", "c": float(i)}) for i in range(10000)]
    t_enc = time.time() - t0
    dec = AvroDecoder(perf_schema)
    t0 = time.time()
    for r in records:
        dec.decode(r)
    t_dec = time.time() - t0
    print(f"  10k records: encode={t_enc:.3f}s decode={t_dec:.3f}s")

if __name__ == "__main__":
    test_zigzag()
    test_primitives()
    test_record()
    test_array()
    test_map()
    test_union()
    test_enum()
    test_schema_evolution_add_field()
    test_type_promotion()
    test_incompatibility()
    test_compatibility_check()
    test_schema_registry()
    test_nested()
    test_compact_encoding()
    test_streaming()
    test_enum_evolution()
    test_performance()
    print()
    print("ALL TESTS PASSED")
