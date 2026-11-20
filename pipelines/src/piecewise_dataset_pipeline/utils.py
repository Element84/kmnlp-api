"""Utils."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import boto3
import obstore
from botocore.credentials import Credentials
from virtualizarr.registry import ObjectStoreRegistry

if TYPE_CHECKING:
    from obstore.store import S3Credential


class NoBoto3CredentialsError(ValueError):
    """No Boto3 Credentials Error."""

    def __init__(self) -> None:
        """Initialize ValueError with string."""
        super().__init__(ValueError("Received None from session.get_credentials"))


class Boto3CredentialProvider:
    """Obstore Provider to provide same credentials Boto3 gets."""

    credentials: Credentials

    def __init__(self, session: boto3.session.Session) -> None:
        """Initializer."""
        credentials = session.get_credentials()
        if credentials is None:
            raise NoBoto3CredentialsError

        self.credentials = credentials

    def __call__(self) -> "S3Credential":
        """Return obstore S3 credentials from boto3 credentials.

        Why use this provider at all rather than using the default S3
        credentials that the obstore library would get from the Rust AWS SDK?
        The AWS SDK does not support the AWS_PROFILE environment variable,
        which would break the way that we run locally.

        Most of this logic is copied from the aws/aws-cli project. Botocore
        does not expose the credential expiration in its API because some
        credentials do not expire. However, the credentials we get in practice
        typically do expire. In that case, there is a private attribute we can
        read (and that the AWS CLI tool does -- or did -- read in practice) to
        get the time the credentials will expire.


        See:
            https://github.com/aws/aws-cli/blob/b578a39cec15b2c206d83e7575adc05068f0bebe/awscli/customizations/configure/exportcreds.py#L27

        Setting the expiry time on the returning obstore credentials results in
        obstore calling this function again once the clock has moved forward to
        an extent that the credentials would have expired.
        """
        credentials = self.credentials
        frozen_credentials = credentials.get_frozen_credentials()

        expiry_attribute: Any | None = getattr(credentials, "_expiry_time", None)
        expires_at: datetime | None = (
            expiry_attribute if isinstance(expiry_attribute, datetime) else None
        )

        # obstore requires datetime to use datetime.timezone.utc specifically
        # boto3 uses dateutil.tz.tzutc(), so we need to convert
        if expires_at and expires_at.tzinfo is not None:
            expires_at = expires_at.astimezone(UTC)

        return {
            "access_key_id": frozen_credentials.access_key or "",
            "secret_access_key": frozen_credentials.secret_key or "",
            "token": frozen_credentials.token,
            "expires_at": expires_at,
        }


def object_store_registry_for_bucket_names(bucket_names: list[str]) -> ObjectStoreRegistry:
    """Return object store registry given a list of bucket names."""

    def _url_for_bucket_name(bucket_name: str) -> str:
        return f"s3://{bucket_name}"

    return ObjectStoreRegistry(
        {
            _url_for_bucket_name(bucket_name): obstore.store.from_url(
                _url_for_bucket_name(bucket_name),
                credential_provider=Boto3CredentialProvider(boto3.session.Session()),
            )
            for bucket_name in bucket_names
        },
    )
