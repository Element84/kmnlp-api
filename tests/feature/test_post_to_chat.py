import logging
from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import Mock

import pytest
from e84_geoai_common.llm.core import (
    LLMAssistantMessage,
    LLMInferenceConfig,
    LLMMessage,
    LLMToolUseContent,
)
from fastapi.testclient import TestClient
from llm_agent.context import LLMAgentContext
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from llm_agent.tools.response_shaping_tools.text_response_tool import TextResponse
from llm_agent.utils.prompt_as_service import PromptService, SimplePromptServiceTransformer
from moto import mock_aws

from app.dependencies import get_app_state
from app.main import app
from app.providers import (
    aois_dao_provider,
    dask_client_provider,
    db_provider,
    s3_client_provider,
    users_dao_provider,
)
from app.types.input import ChatRequest, UserCreateRequest
from app.types.output import Chat, ChatHeader, ChatResponse, EndOfTurn, Text
from app.utils.env import DASK_SCHEDULER_ADDRESS
from app.utils.utils import AppState, make_agent
from tests.utils.providers import feature_test_db_provider

log = logging.getLogger(__name__)


class TestPostToChat:
    @classmethod
    def setup_class(cls) -> None:
        cls.headers = {"x-user-id": "test"}
        cls.created_chat_ids: list[str] = []

        log.info("Using dask scheduler address: %s", DASK_SCHEDULER_ADDRESS)

    @pytest.fixture(autouse=True)
    def clear_caches(self):
        """Manage dependencies.

        * Override the DB dependency with the test DB dependency.
        * Override the app fixture dependency with one specific to these feature tests.
        * Clear dependency caches between each test run so that the tests don't affect each other.
        """
        # Override dependencies
        app.dependency_overrides[get_app_state] = app_state_fixture
        app.dependency_overrides[db_provider] = feature_test_db_provider

        # Run test
        yield

        # Clear caches
        feature_test_db_provider.cache_clear()
        users_dao_provider.cache_clear()
        aois_dao_provider.cache_clear()

    @mock_aws
    def test_post_to_chat_basic(self):
        with TestClient(app) as client:
            chat = self._init_user_and_chat(client)

            msg = ChatRequest(content="List the names of your tools.")
            parsed_responses: list[ChatResponse] = []
            with client.stream(
                "post",
                f"/chats/{chat.id}",
                json=msg.model_dump(mode="json"),
                headers=self.headers,
            ) as response_stream:
                assert response_stream.status_code == 200
                for line in response_stream.iter_lines():
                    parsed = ChatResponse.model_validate_json(line)
                    parsed_responses.append(parsed)

        assert len(parsed_responses) == 2
        assert isinstance(parsed_responses[0].content, Text)
        assert isinstance(parsed_responses[1].content, EndOfTurn)

    @mock_aws
    def test_chat_history_contains_created_chats(self):
        with TestClient(app) as client:
            chat = self._init_user_and_chat(client)

            # Get the user's chat history
            response = client.get("/chats", headers=self.headers)
            assert response.status_code == 200
            response_json = cast("list[dict[str, Any]]", response.json())

            # Assert chat history starts with single chat (created in _init... function)
            assert isinstance(response_json, list)
            assert len(response_json) == 1

            chat_header = ChatHeader.model_validate(response_json[0])
            assert len(chat_header.title) > 0

            # Create chat and post a single message
            msg = ChatRequest(content="List the names of your tools.")
            parsed_responses: list[ChatResponse] = []
            with client.stream(
                "post",
                f"/chats/{chat.id}",
                json=msg.model_dump(mode="json"),
                headers=self.headers,
            ) as response_stream:
                assert response_stream.status_code == 200
                for line in response_stream.iter_lines():
                    parsed = ChatResponse.model_validate_json(line)
                    parsed_responses.append(parsed)

            # Get the user's chat history
            response = client.get("/chats", headers=self.headers)
            assert response.status_code == 200
            response_json = cast("list[dict[str, Any]]", response.json())

            # Assert chat history now has single element with updated title
            assert isinstance(response_json, list)
            assert len(response_json) == 1

            chat_header = ChatHeader.model_validate(response_json[0])
            assert chat_header.title == "List the names of your tools."

    def _init_user_and_chat(self, client: TestClient) -> Chat:
        user_data = UserCreateRequest(user_id="test")
        response = client.post("/users", json=user_data.model_dump())

        response = client.post("/chats", headers=self.headers)
        assert response.status_code == 200
        chat = Chat.model_validate(response.json())
        self.created_chat_ids.append(chat.id)
        return chat


def app_state_fixture() -> AppState:
    """Return app state for use in feature tests."""
    return AppState(
        context=LLMAgentContext(
            place_lookup=Mock(),
            s3_client=s3_client_provider(),
            knowledge_graph_dao=KnowledgeGraphDAO(),
            dask_client=dask_client_provider(),
        ),
        agent=make_agent(
            bedrock_runtime_client=Mock(),
            orchestration_prompt_simple_transformer=MockPromptServiceTransformer(),
        ),
    )


class MockPromptServiceTransformer(SimplePromptServiceTransformer):
    async def __call__(
        self, _: tuple[Sequence[LLMMessage], LLMInferenceConfig], __: PromptService
    ) -> LLMAssistantMessage:
        return LLMAssistantMessage(
            content=[
                LLMToolUseContent(
                    id="a",
                    name="TextResponseTool",
                    input=TextResponse(text="Hi.").model_dump(mode="json"),
                ),
                LLMToolUseContent(
                    id="b",
                    name="EndTurnTool",
                    input={},
                ),
            ]
        )
