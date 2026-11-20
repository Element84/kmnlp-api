from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Parse an S3 URI into (bucket, key)."""
    bucket, key = s3_uri.removeprefix("s3://").split("/", 1)
    return bucket, key


def exists(s3_client: S3Client, s3_uri: str) -> bool:
    """Check if an S3 object exists."""
    bucket, key = parse_s3_uri(s3_uri)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    return True


def uri_to_presigned_url(s3_client: S3Client, s3_uri: str, expiry_seconds: int) -> str:
    """Convert an S3 URI to a pre-signed HTTP URL."""
    bucket, key = parse_s3_uri(s3_uri)
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry_seconds,
    )
    return url


def upload_file(s3_client: S3Client, file_path: str, s3_uri: str) -> None:
    """Upload a local file to S3."""
    bucket, key = parse_s3_uri(s3_uri)
    s3_client.upload_file(file_path, bucket, key)


def upload_object(s3_client: S3Client, data: bytes | str, s3_uri: str) -> None:
    """Upload bytes data to S3."""
    bucket, key = parse_s3_uri(s3_uri)
    s3_client.put_object(Body=data, Bucket=bucket, Key=key)


def get_object(s3_client: S3Client, s3_uri: str) -> bytes:
    """Get an object from S3 and return its content as bytes."""
    bucket, key = parse_s3_uri(s3_uri)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def make_data_key_path(user_id: str, chat_id: str, data_key: str) -> str:
    """Construct a relative S3 path to a data object."""
    return f"{user_id}/chats/{chat_id}/{data_key}"
