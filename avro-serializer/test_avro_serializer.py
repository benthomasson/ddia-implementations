"""Tests for avro_serializer — covers spec examples and key edge cases."""

import pytest
from avro_serializer import (
    Schema, AvroEncoder, AvroDecoder, SchemaRegistry,
    SchemaError, SchemaCompatibilityError,
    check_compatibility, zigzag_encode, zigzag_decode,
)


# --- Test 1: Primitive round-trips ---

@pytest.mark.parametrize("typ,val", [
    ("null", None),
    ("boolean", True), ("boolean", False),
    ("int", 0), ("int", -1), ("int", 42), ("int", 2147483647), ("int", -2147483648),
    ("long", 2**40), ("long", -(2**40)),
    ("double", 2.718281828),
    ("string", "hello"), ("string", ""), ("string", "emoji: \u2603"),
    ("bytes", b"\x00\x01\x02"), ("bytes", b""),
])
def test_primitive_roundtrip(typ, val):
    s = Schema(typ)
    encoded = AvroEncoder(s).encode(val)
    decoded = AvroDecoder(s).decode(encoded)
    assert decoded == val


def test_float_roundtrip():
    s = Schema("float")
    encoded = AvroEncoder(s).encode(3.14)
    decoded = AvroDecoder(s).decode(encoded)
    assert abs(decoded - 3.14) < 1e-6


# --- Test 2: Zigzag encoding ---

@pytest.mark.parametrize("n,expected", [
    (0, 0), (-1, 1), (1, 2), (-2, 3), (2, 4),
    (2147483647, 4294967294), (-2147483648, 4294967295),
])
def test_zigzag(n, expected):
    assert zigzag_encode(n) == expected
    assert zigzag_decode(expected) == n


# --- Test 3: Record, array, map, union, enum ---

def test_record():
    schema = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"},
    ]})
    val = {"id": 1, "name": "Alice", "email": "alice@example.com"}
    data = AvroEncoder(schema).encode(val)
    # No field names in binary output
    assert b"id" not in data
    assert b"email" not in data
    assert AvroDecoder(schema).decode(data) == val


def test_array_and_empty():
    s = Schema({"type": "array", "items": "int"})
    enc = AvroEncoder(s)
    assert AvroDecoder(s).decode(enc.encode([1, 2, 3])) == [1, 2, 3]
    assert AvroDecoder(s).decode(enc.encode([])) == []


def test_map():
    s = Schema({"type": "map", "values": "string"})
    val = {"a": "x", "b": "y"}
    assert AvroDecoder(s).decode(AvroEncoder(s).encode(val)) == val
    assert AvroDecoder(s).decode(AvroEncoder(s).encode({})) == {}


def test_union():
    s = Schema(["null", "string"])
    enc = AvroEncoder(s)
    dec = AvroDecoder(s)
    assert dec.decode(enc.encode(None)) is None
    assert dec.decode(enc.encode("hello")) == "hello"


def test_enum():
    s = Schema({"type": "enum", "name": "Color", "symbols": ["RED", "GREEN", "BLUE"]})
    enc = AvroEncoder(s)
    for sym in ["RED", "GREEN", "BLUE"]:
        assert AvroDecoder(s).decode(enc.encode(sym)) == sym


# --- Test 4: Schema evolution (the DDIA core concept) ---

def test_schema_evolution_add_field_with_default():
    """Reader adds a field with default — backward compatible."""
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
    # Encode with v1, decode with v2 (backward compat)
    data = AvroEncoder(v1).encode({"id": 1, "name": "Alice", "email": "alice@example.com"})
    result = AvroDecoder(v1, v2).decode(data)
    assert result == {"id": 1, "name": "Alice", "age": 0}


def test_schema_evolution_missing_required_field_fails():
    """Reader has required field without default that writer lacks."""
    v1 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
    ]})
    v2 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},  # no default
    ]})
    data = AvroEncoder(v1).encode({"id": 1})
    with pytest.raises(SchemaCompatibilityError):
        AvroDecoder(v1, v2).decode(data)


def test_type_promotion():
    """int -> long, int -> double promotions."""
    int_s = Schema("int")
    long_s = Schema("long")
    double_s = Schema("double")
    data = AvroEncoder(int_s).encode(42)
    assert AvroDecoder(int_s, long_s).decode(data) == 42
    assert AvroDecoder(int_s, double_s).decode(data) == 42.0


# --- Test 5: Compatibility checking ---

def test_compatibility_check():
    v1 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
    ]})
    v2 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int", "default": 0},
    ]})
    compat = check_compatibility(v1, v2)
    assert compat["backward_compatible"] is True


# --- Test 6: Schema registry ---

def test_schema_registry():
    v1 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
    ]})
    v2 = Schema({"type": "record", "name": "User", "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int", "default": 0},
    ]})
    reg = SchemaRegistry()
    sid = reg.register(v1)
    encoded = reg.encode_with_id(sid, {"id": 1, "name": "Alice"})
    # Decode with evolution
    decoded = reg.decode_with_id(encoded, reader_schema=v2)
    assert decoded == {"id": 1, "name": "Alice", "age": 0}


# --- Test 7: Nested schemas ---

def test_nested_record_with_array():
    schema = Schema({"type": "record", "name": "Doc", "fields": [
        {"name": "title", "type": "string"},
        {"name": "tags", "type": {"type": "array", "items": "string"}},
    ]})
    val = {"title": "hello", "tags": ["a", "b", "c"]}
    assert AvroDecoder(schema).decode(AvroEncoder(schema).encode(val)) == val


# --- Test 8: Schema validation errors ---

def test_invalid_schemas():
    with pytest.raises(SchemaError):
        Schema("invalid_type")
    with pytest.raises(SchemaError):
        Schema([])  # empty union
    with pytest.raises(SchemaError):
        Schema(["int", "int"])  # duplicate union types


# --- Test 9: Fixed type ---

def test_fixed_roundtrip():
    s = Schema({"type": "fixed", "name": "IPv4", "size": 4})
    val = b"\xc0\xa8\x01\x01"
    encoded = AvroEncoder(s).encode(val)
    assert len(encoded) == 4
    assert AvroDecoder(s).decode(encoded) == val


def test_fixed_wrong_size_rejected():
    s = Schema({"type": "fixed", "name": "IPv4", "size": 4})
    with pytest.raises(ValueError):
        AvroEncoder(s).encode(b"\x00\x01")


def test_fixed_schema_validation():
    with pytest.raises(SchemaError):
        Schema({"type": "fixed", "name": "F"})  # missing size
    with pytest.raises(SchemaError):
        Schema({"type": "fixed", "size": 4})  # missing name


def test_fixed_compatibility():
    s1 = Schema({"type": "fixed", "name": "Hash", "size": 16})
    s2 = Schema({"type": "fixed", "name": "Hash", "size": 16})
    compat = check_compatibility(s1, s2)
    assert compat["full_compatible"] is True

    s3 = Schema({"type": "fixed", "name": "Hash", "size": 32})
    compat2 = check_compatibility(s1, s3)
    assert compat2["backward_compatible"] is False


def test_fixed_in_record():
    schema = Schema({"type": "record", "name": "Msg", "fields": [
        {"name": "id", "type": "int"},
        {"name": "checksum", "type": {"type": "fixed", "name": "MD5", "size": 16}},
    ]})
    val = {"id": 42, "checksum": b"\x00" * 16}
    assert AvroDecoder(schema).decode(AvroEncoder(schema).encode(val)) == val


# --- Test 10: Schema canonical equality ---

def test_schema_equality_canonical():
    """Schema("int") and Schema({"type": "int"}) should be equal."""
    assert Schema("int") == Schema({"type": "int"})
    assert Schema("string") == Schema({"type": "string"})


def test_schema_equality_complex():
    s1 = Schema({"type": "record", "name": "R", "fields": [
        {"name": "x", "type": "int"},
    ]})
    s2 = Schema({"type": "record", "name": "R", "fields": [
        {"name": "x", "type": {"type": "int"}},
    ]})
    assert s1 == s2


def test_schema_hashable():
    s1 = Schema("int")
    s2 = Schema({"type": "int"})
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1


# --- Test 11: Record field reordering ---

def test_record_field_reorder():
    """Writer and reader have same fields in different order."""
    writer = Schema({"type": "record", "name": "R", "fields": [
        {"name": "a", "type": "int"},
        {"name": "b", "type": "string"},
        {"name": "c", "type": "double"},
    ]})
    reader = Schema({"type": "record", "name": "R", "fields": [
        {"name": "c", "type": "double"},
        {"name": "a", "type": "int"},
        {"name": "b", "type": "string"},
    ]})
    val = {"a": 1, "b": "hello", "c": 3.14}
    data = AvroEncoder(writer).encode(val)
    result = AvroDecoder(writer, reader).decode(data)
    assert result == val


# --- Test 12: Union dict ambiguity (record vs map) ---

def test_union_record_vs_map():
    """Union with record and map types should disambiguate dicts correctly."""
    record_schema = {"type": "record", "name": "Point", "fields": [
        {"name": "x", "type": "int"},
        {"name": "y", "type": "int"},
    ]}
    union = Schema([record_schema, {"type": "map", "values": "int"}])
    enc = AvroEncoder(union)
    dec = AvroDecoder(union)

    # Dict matching record fields -> record branch
    record_val = {"x": 1, "y": 2}
    assert dec.decode(enc.encode(record_val)) == record_val

    # Dict with non-record keys -> map branch
    map_val = {"arbitrary": 10, "keys": 20}
    assert dec.decode(enc.encode(map_val)) == map_val


# --- Test 13: Int range validation ---

def test_int_range_overflow():
    s = Schema("int")
    enc = AvroEncoder(s)
    enc.encode(2147483647)  # max int32, should work
    enc.encode(-2147483648)  # min int32, should work
    with pytest.raises(ValueError):
        enc.encode(2147483648)  # int32 + 1
    with pytest.raises(ValueError):
        enc.encode(-2147483649)


def test_long_accepts_large_values():
    s = Schema("long")
    enc = AvroEncoder(s)
    dec = AvroDecoder(s)
    val = 2**40
    assert dec.decode(enc.encode(val)) == val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
