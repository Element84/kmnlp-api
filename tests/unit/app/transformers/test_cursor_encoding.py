from datetime import UTC, datetime

import pytest

from app.transformers.cursor_encoding import (
    CursorValueNotConvertibleToFloatError,
    InvalidCursorError,
    MalformedCursorError,
    decode_cursor,
    decode_datetime_cursor,
    encode_cursor,
    encode_datetime_cursor,
)
from app.types.internal import Cursor


def test_encode_cursor():
    encoded = encode_cursor("some_key", "42")
    assert encoded == "c29tZV9rZXk6NDI="


def test_decode_cursor():
    (key, value) = decode_cursor("c29tZV9rZXk6NDI=")
    assert key == "some_key"
    assert value == "42"


def test_decode_cursor_invalid_b64():
    with pytest.raises(InvalidCursorError):
        decode_cursor("lkjqweiosdijfnji123498")


def test_decode_cursor_no_separator():
    with pytest.raises(MalformedCursorError):
        decode_cursor("c3R1ZmYsNDI=")  # Decodes to `stuff,42`


def test_decode_cursor_with_2_separators():
    key, value = decode_cursor("c3R1ZmY6NDI6NDM=")  # Decodes to `stuff:42:43`
    assert key == "stuff"
    assert value == "42:43"


def test_encode_datetime():
    value = datetime(2026, 1, 1, tzinfo=UTC)
    encoded = encode_datetime_cursor(Cursor(key="created_at", value=value))
    assert encoded == "Y3JlYXRlZF9hdDoxNzY3MjI1NjAwLjA="


def test_decode_datetime():
    cursor = decode_datetime_cursor("Y3JlYXRlZF9hdDoxNzY3MjI1NjAwLjA=")

    assert cursor.key == "created_at"
    assert cursor.value == datetime(2026, 1, 1, tzinfo=UTC)


def test_decode_datetime_value_not_float():
    with pytest.raises(CursorValueNotConvertibleToFloatError):
        decode_datetime_cursor("c3R1ZmY6eW8=")  # Decodes to `stuff:yo`


def test_decode_datetime_strange_value():
    with pytest.raises(CursorValueNotConvertibleToFloatError):
        decode_datetime_cursor("c3R1ZmY6NDI6NDM=")  # Decodes to `stuff:42:43`


def test_datetime_encoded_value_is_timestamp_float():
    # Encode a datetime
    datetime_value = datetime(2026, 1, 1, tzinfo=UTC)
    encoded = encode_datetime_cursor(Cursor(key="created_at", value=datetime_value))

    # Assert encoded string is base64-encoded "created_at:1767225600.0"
    assert encoded == "Y3JlYXRlZF9hdDoxNzY3MjI1NjAwLjA="

    # Decode directly to string
    key, value = decode_cursor(str(encoded))
    assert key == "created_at"
    assert value == "1767225600.0"
