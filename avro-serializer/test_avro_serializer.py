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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
