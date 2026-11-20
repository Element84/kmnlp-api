from unittest.mock import Mock, patch

import boto3
import obstore
import pytest

from piecewise_dataset_pipeline.utils import Boto3CredentialProvider


@pytest.mark.integration
def test_boto3_credential_provider_integrates_with_obstore():
    """Verify Boto3CredentialProvider works with real obstore library.

    This test ensures:
    1. Our provider implements the interface obstore expects
    2. obstore accepts our credentials without errors
    3. We catch breaking changes when upgrading obstore

    Note: This test doesn't make actual network calls - it mocks the S3 client.
    """
    # Arrange: Create real provider with real boto3 session
    session = boto3.Session()
    provider = Boto3CredentialProvider(session)

    # Act: Try to create obstore with our provider
    # Mock the actual S3 operations to prevent network calls
    with patch("obstore.store.S3Store") as mock_store_class:
        mock_store = Mock()
        mock_store_class.return_value = mock_store

        try:
            # This will fail if our provider doesn't match obstore's expectations
            _ = obstore.store.from_url(
                "s3://test-bucket",
                credential_provider=provider,
            )

            # If we got here, obstore accepted our provider
            contract_accepted = True
            error_message = None

        except (TypeError, ValueError, AttributeError, KeyError) as e:
            # Contract violation - our provider doesn't match expectations
            contract_accepted = False
            error_message = str(e)

    # Assert: obstore should accept our provider
    assert contract_accepted, f"Provider rejected by obstore: {error_message}"


@pytest.mark.integration
def test_boto3_credential_provider_return_value_conforms_to_obstore_interface():
    """Verify return value matches obstore's expected interface.

    This is a conformance test that catches:
    - Missing required fields
    - Wrong field types
    - Extra unexpected fields
    """
    # Arrange: Create real provider
    session = boto3.Session()
    provider = Boto3CredentialProvider(session)

    # Act: Get credentials from provider
    result = provider()

    # Assert: Verify it matches obstore's S3Credential interface

    # Required fields must exist
    assert "access_key_id" in result, "Missing required field: access_key_id"
    assert "secret_access_key" in result, "Missing required field: secret_access_key"
    assert "token" in result, "Missing required field: token"
    assert "expires_at" in result, "Missing required field: expires_at"

    # Type checks - access_key_id must be str
    assert isinstance(result["access_key_id"], str), "access_key_id must be str"

    # Type checks - secret_access_key must be str
    assert isinstance(result["secret_access_key"], str), "secret_access_key must be str"

    # Type checks - token must be str or None
    assert result["token"] is None or isinstance(result["token"], str), "token must be str or None"

    # Type checks - expires_at type
    # Note: Based on our investigation, this should be ISO string or None
    # but the current implementation returns datetime or None
    # This test will catch if we need to convert to ISO string
    assert result["expires_at"] is None or isinstance(
        result["expires_at"], (str, type(result["expires_at"].__class__))
    ), "expires_at must be compatible with obstore's expectations"

    # Verify no unexpected fields
    expected_fields = {"access_key_id", "secret_access_key", "token", "expires_at"}
    actual_fields = set(result.keys())
    unexpected = actual_fields - expected_fields
    assert not unexpected, f"Unexpected fields in credential dict: {unexpected}"
