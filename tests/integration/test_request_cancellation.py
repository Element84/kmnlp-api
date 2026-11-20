"""Integration test for request cancellation when clients disconnect.

This test verifies that when a client disconnects (via AbortController in React),
the server properly cancels ongoing agent execution and Dask computations.
"""

import asyncio
import logging
import os
import threading
import time
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from testcontainers.postgres import PostgresContainer  # type: ignore[reportMissingTypeStubs]

from app.main import app
from app.providers import sql_alchemy_engine_provider
from app.types.input import ChatRequest, UserCreateRequest

log = logging.getLogger(__name__)


@pytest.mark.skip(reason="Not really testing that cancellation actually happens.")
@pytest.mark.integration
class TestRequestCancellation:
    """Integration tests for request cancellation."""

    @pytest.fixture
    def client(self, dask_scheduler_address: str) -> Iterator[TestClient]:
        """Set up test client with PostgreSQL and Dask."""
        with (
            patch.dict(os.environ, {"DASK_SCHEDULER_ADDRESS": dask_scheduler_address}),
            TestClient(app, base_url="http://testserver") as client,
            PostgresContainer("postgres:17") as postgres,
        ):
            psql_url = postgres.get_connection_url()
            engine = create_engine(psql_url)
            app.dependency_overrides[sql_alchemy_engine_provider] = lambda: engine

            log.info("Using dask scheduler address: %s", dask_scheduler_address)

            yield client

    @classmethod
    def setup_class(cls) -> None:
        """Set up class-level test data."""
        cls.user_id = "cancellation-test-user"
        cls.headers = {"x-user-id": cls.user_id}

    def _init_user_and_chat(self, client: TestClient) -> str:
        """Initialize a user and chat, return chat ID."""
        # Create user
        user_data = UserCreateRequest(user_id=self.user_id)
        response = client.post("/users", json=user_data.model_dump())
        assert response.status_code == 200, response.text

        # Create chat
        response = client.post("/chats", headers=self.headers)
        assert response.status_code == 200, response.text
        chat_data = response.json()
        return chat_data["id"]

    def test_request_cancellation_on_client_disconnect(self, client: TestClient) -> None:
        """Test that agent execution is cancelled when client disconnects.

        This simulates what happens when a React client uses AbortController to
        cancel a request:
        1. Client starts a chat request (long-running operation)
        2. Client disconnects mid-execution
        3. Server should:
           - Detect the disconnection
           - Cancel the agent task
           - Clean up Dask computations
           - Save chat state to database
        """
        chat_id = self._init_user_and_chat(client)

        # Start a long-running request that will involve Dask computation
        message = ChatRequest(
            content="Analyze sea surface temperature data for the entire Atlantic Ocean over the past year"  # noqa: E501
        )

        # Track responses and whether stream was interrupted
        responses: list[str] = []
        stream_interrupted = False

        # Start streaming in a background thread
        def stream_request() -> None:
            nonlocal stream_interrupted
            try:
                with client.stream(
                    "post",
                    f"/chats/{chat_id}",
                    json=message.model_dump(mode="json"),
                    headers=self.headers,
                    timeout=5.0,  # Short timeout to simulate cancellation
                ) as response:
                    for line in response.iter_lines():
                        if line:
                            responses.append(line)
                            log.info("Received response: %s", line[:100])
            except asyncio.CancelledError:
                # Connection will be closed/timeout, causing an exception
                log.exception("Stream interrupted")
                stream_interrupted = True

        stream_thread = threading.Thread(target=stream_request)
        stream_thread.start()

        # Give the request time to start processing
        time.sleep(0.5)

        # Simulate client disconnection by closing the connection
        # (In practice, the timeout will cause the stream to close)
        log.info("Simulating client disconnection")

        # Wait for stream to complete/timeout
        stream_thread.join(timeout=10.0)

        # Give the server time to clean up
        time.sleep(0.5)

        # Verify that the chat was saved to the database despite cancellation
        response = client.get(f"/chats/{chat_id}", headers=self.headers)
        assert response.status_code == 200
