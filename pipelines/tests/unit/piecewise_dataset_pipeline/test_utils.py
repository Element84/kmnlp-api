# type: ignore[reportTypedDictNotRequiredAccess]

from datetime import datetime, timezone
from unittest.mock import Mock

import dateutil
import pytest

from piecewise_dataset_pipeline.utils import Boto3CredentialProvider, NoBoto3CredentialsError


def test_boto3_credential_provider_raises_error_when_no_credentials():
    """Test that provider raises error when boto3 session has no credentials."""
    mock_session = Mock()
    mock_session.get_credentials.return_value = None

    with pytest.raises(NoBoto3CredentialsError):
        Boto3CredentialProvider(mock_session)


def test_boto3_credential_provider_returns_correct_structure():
    """Verify provider returns expected credential structure."""
    # Create mock session and credentials
    mock_session = Mock()
    mock_creds = Mock()
    mock_frozen = Mock()
    mock_frozen.access_key = "access key"
    mock_frozen.secret_key = "secret key"  # noqa: S105
    mock_frozen.token = "token"  # noqa: S105

    mock_creds.get_frozen_credentials.return_value = mock_frozen
    mock_creds._expiry_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # noqa: UP017 (obstore requires it)

    mock_session.get_credentials.return_value = mock_creds

    # Build and call the provider
    provider = Boto3CredentialProvider(mock_session)
    result = provider()

    # Verify structure
    assert result["access_key_id"] == "access key"
    assert result["secret_access_key"] == "secret key"  # noqa: S105
    assert result["token"] == "token"  # noqa: S105
    assert result["expires_at"] == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # noqa: UP017


def test_boto3_credential_provider_handles_missing_expiry_time():
    """Test provider when credentials don't have _expiry_time (long-term creds)."""
    # Create mock session and credentials
    mock_session = Mock()
    mock_creds = Mock()
    mock_frozen = Mock()
    mock_frozen.access_key = "access key"
    mock_frozen.secret_key = "secret key"  # noqa: S105
    mock_frozen.token = None  # Long-term creds don't have token

    mock_creds.get_frozen_credentials.return_value = mock_frozen

    mock_session.get_credentials.return_value = mock_creds

    # Build and call the provider
    provider = Boto3CredentialProvider(mock_session)
    result = provider()

    # Verify expires_at is None for long-term credentials
    assert result["access_key_id"] == "access key"
    assert result["secret_access_key"] == "secret key"  # noqa: S105
    assert result["token"] is None
    assert result["expires_at"] is None


def test_boto3_credential_provider_handles_non_datetime_expiry():
    """Test that provider handles when _expiry_time is not a datetime."""
    # Create mock with invalid _expiry_time
    mock_session = Mock()
    mock_creds = Mock()
    mock_frozen = Mock()
    mock_frozen.access_key = "access key"
    mock_frozen.secret_key = "secret key"  # noqa: S105
    mock_frozen.token = "token"  # noqa: S105
    mock_creds.get_frozen_credentials.return_value = mock_frozen

    # Set _expiry_time as something weird (should be ignored)
    mock_creds._expiry_time = "not a datetime"

    # Finish setting up session
    mock_session.get_credentials.return_value = mock_creds

    # Build and call the provider
    provider = Boto3CredentialProvider(mock_session)
    result = provider()

    # Assert: expires_at should be None when _expiry_time is not datetime
    assert result["expires_at"] is None


def test_boto3_credential_provider_converts_to_utc_timezone():
    """Test that provider converts datetime to use datetime.utc specifically."""
    # Create mock with datetime using custom UTC timezone
    mock_session = Mock()
    mock_creds = Mock()
    mock_frozen = Mock()
    mock_frozen.access_key = "access key"
    mock_frozen.secret_key = "secret key"  # noqa: S105
    mock_frozen.token = "token"  # noqa: S105
    mock_creds.get_frozen_credentials.return_value = mock_frozen

    # Create datetime with custom UTC (not timezone.utc)
    expiry_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=dateutil.tz.tzutc())
    mock_creds._expiry_time = expiry_time

    mock_session.get_credentials.return_value = mock_creds

    # Build and call the provider
    provider = Boto3CredentialProvider(mock_session)
    result = provider()

    # `expires_at` should use datetime.utc specifically
    assert result["expires_at"] is not None
    assert result["expires_at"].tzinfo is timezone.utc  # noqa: UP017
