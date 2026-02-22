import base64
import binascii
from datetime import UTC, datetime

from app.types.internal import Cursor


class InvalidCursorError(ValueError):
    """Provided cursor does not base64 decode."""

    def __init__(self, cursor: str) -> None:
        """Initializer."""
        super().__init__("Cursor does not base64 decode: %s", cursor)


class MalformedCursorError(ValueError):
    """Provided cursor base64 decodes into a valid string with no `:`."""

    def __init__(self, encoded_cursor: str, decoded_cursor: str) -> None:
        """Initializer."""
        super().__init__(
            "Decoded cursor does not have the expected `:` separator. encoded: %s, decoded: %s",
            encoded_cursor,
            decoded_cursor,
        )


class UnexpectedCursorKeyError(ValueError):
    """Provided Base64 cursor's key is not expected value."""

    def __init__(self, encoded_cursor: str, decoded_cursor: str) -> None:
        """Initializer."""
        super().__init__(
            "Decoded cursor's key portion has unexpected value: encoded: %s, decoded: %s",
            encoded_cursor,
            decoded_cursor,
        )


class CursorValueNotConvertibleToFloatError(ValueError):
    """Provided Base64 cursor does not encode a float value."""

    def __init__(self, encoded_cursor: str, decoded_cursor: str) -> None:
        """Initializer."""
        super().__init__(
            "Decoded cursor's value portion does not convert to float: encoded: %s, decoded: %s",
            encoded_cursor,
            decoded_cursor,
        )


def encode_cursor(key: str, value: str) -> str:
    """Encode a cursor."""
    encodable_value = f"{key}:{value}"
    return base64.urlsafe_b64encode(encodable_value.encode("utf-8")).decode("utf-8")


def decode_cursor(cursor: str) -> tuple[str, str]:
    """Decode a cursor to its composing parts.

    Args:
        cursor (str): The cursor to decode

    Returns:
        A tuple of the key/value pair encoded in the cursor

    Raises:
        InvalidCursorError: If the decoded cursor is not of the form "key:value"

    """
    try:
        decoded_cursor = str(base64.urlsafe_b64decode(cursor), "utf-8")
    except binascii.Error as e:
        raise InvalidCursorError(cursor) from e

    split_string = decoded_cursor.split(":", 1)

    if len(split_string) == 1:
        raise MalformedCursorError(cursor, decoded_cursor)

    return (split_string[0], "".join(split_string[1:]))


def encode_datetime_cursor(cursor: Cursor) -> str:
    """Serialize a datetime as a Base64 string.

    `datetime` objects are not Base64-encodable, and the `timestamp()` method
    on `datetime` returns a `float`, which is not Base64 encodable because the
    byte-order is non-determinate.

    Encode a datetime by converting to float to a string. This might not work
    the same on each machine, but it should be consistent enough to use when
    deployed.

    Args:
        cursor (Cursor): The domain cursor to format and encode.

    Returns:
        A Base64-encoded string with the bytes representing:
        `key:value_packed_bytes`
    """
    return encode_cursor(cursor.key, f"{cursor.value.timestamp()}")


def decode_datetime_cursor(encoded_string: str) -> Cursor:
    """Deserialize a Base64 string into a key and datetime.

    Decode a `datetime` by building a float out of the value portion of the
    cursor string.

    Args:
        encoded_string (str): The value to decode.

    Returns:
        A tuple containing the key and datetime value encoded in the `value`.
    """
    key, value = decode_cursor(encoded_string)
    try:
        timestamp = float(value)
    except ValueError as e:
        raise CursorValueNotConvertibleToFloatError(encoded_string, f"{key}:{value}") from e

    if key == "created_at":
        return Cursor(key, datetime.fromtimestamp(timestamp, UTC))

    raise UnexpectedCursorKeyError(encoded_string, f"{key}:{value}")
