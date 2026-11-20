import logging
from collections.abc import AsyncIterator, Callable, Generator
from typing import Any, cast
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from llm_agent.agents.orchestration_agent.orchestration_agent import OrchestrationAgent
from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool
from pydantic import BaseModel
from pydantic_ai import ModelMessage, RunContext, models
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.db.sqlalchemy_db import SQLAlchemyDB
from app.dependencies import get_app_state
from app.main import app
from app.providers import aois_dao_provider, db_provider, users_dao_provider
from app.types.input import ChatRequest, UserCreateRequest
from app.types.output import Chat, ChatHeader, ChatResponse, EndOfTurn, Text
from app.utils.app_state import AppState

log = logging.getLogger(__name__)

models.ALLOW_MODEL_REQUESTS = False


class MockToolInput(BaseModel):
    pass


class MockTool(AgentTool[MockToolInput, str]):
    name = "MockTool"
    description = "A mock tool"
    input_model = MockToolInput
    output_type = str

    async def __call__(self, ctx: RunContext[AgentDeps], inp: MockToolInput, /) -> str:  # noqa: ARG002
        return "Mock tool response"


@pytest.fixture
def mocked_llm() -> FunctionModel:
    async def _stream(
        messages: list[ModelMessage],  # noqa: ARG001
        info: AgentInfo,  # noqa: ARG001
    ) -> AsyncIterator[str]:
        yield "Mock agent response"

    return FunctionModel(stream_function=_stream)


@pytest.fixture
def orchestration_agent(mocked_llm: FunctionModel) -> OrchestrationAgent:
    """Instantiate and return orchestration agent."""
    return OrchestrationAgent(mocked_llm, system_prompt="Test system prompt", tools=[MockTool()])


@pytest.fixture
def app_state_feature_tests(
    app_state_custom_agent: Callable[..., AppState],
    orchestration_agent: OrchestrationAgent,
) -> AppState:
    """Return app state for use in feature tests."""
    return app_state_custom_agent(agent=orchestration_agent)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with patch("app.main.get_dask_client"), TestClient(app) as client:
        yield client


class TestPostToChat:
    @classmethod
    def setup_class(cls) -> None:
        cls.headers = {"x-user-id": "test"}
        cls.created_chat_ids: list[str] = []

    @pytest.fixture(autouse=True)
    def clear_caches(self, sqlalchemy_db: SQLAlchemyDB, app_state_feature_tests: AppState):
        """Manage dependencies.

        * Override the DB dependency with the test DB dependency.
        * Override the app fixture dependency with one specific to these feature tests.
        * Clear dependency caches between each test run so that the tests don't affect each other.
        """
        # Override dependencies
        app.dependency_overrides[get_app_state] = lambda: app_state_feature_tests
        # Get the DB from the fixture, which manages clearing it.
        app.dependency_overrides[db_provider] = lambda: sqlalchemy_db

        # Run test
        yield

        # Clear caches
        users_dao_provider.cache_clear()
        aois_dao_provider.cache_clear()

    def test_post_to_chat_basic(self, client: TestClient):
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

        assert len(parsed_responses) >= 2
        assert isinstance(parsed_responses[-2].content, Text)
        assert isinstance(parsed_responses[-1].content, EndOfTurn)

    def test_chat_history_contains_created_chats(self, client: TestClient):
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
