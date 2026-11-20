from mypy_boto3_s3 import S3Client

from app.utils import s3


def test_s3_uri_to_presigned_url(mock_s3_client: S3Client) -> None:
    mock_s3_client.put_object(Bucket="test-bucket", Key="test-object", Body="test data")

    s3_uri = "s3://test-bucket/test-object"
    presigned_url = s3.uri_to_presigned_url(mock_s3_client, s3_uri, expiry_seconds=3600)

    assert presigned_url.startswith("https://test-bucket.s3.amazonaws.com/test-object?")
    assert "Expires" in presigned_url
