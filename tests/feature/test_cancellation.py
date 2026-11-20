"""Feature tests for cancellation recovery.

Verifies that when a request is cancelled mid-tool-execution, the conversation
history is left in a valid state and the next message can be sent successfully.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
from llm_agent.agents.orchestration_agent.interface import UserMessage
from llm_agent.agents.orchestration_agent.orchestration_agent import OrchestrationAgent
from llm_agent.context import AgentDeps
from llm_agent.tools.agent_tool import AgentTool, ErrorResponse
from pydantic import BaseModel
from pydantic_ai import ModelMessage, RunContext, models
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from app.context import KmnlpChatContext
from app.dao.users_dao import UsersDAO
from app.types.input import ChatRequest
from app.types.internal import InternalChat
from app.utils.orchestration.orchestration_agent import get_agent_response_content_stream

if TYPE_CHECKING:
    from app.types.output import chat_response_content_types

log = logging.getLogger(__name__)

models.ALLOW_MODEL_REQUESTS = False


class StallingToolInput(BaseModel):
    pass


class StallingTool(AgentTool[StallingToolInput, str]):
    name = "StallingTool"
    description = "A tool that stalls indefinitely to test cancellation recovery."
    input_model = StallingToolInput
    output_type = str

    async def __call__(self, ctx: RunContext[AgentDeps], inp: StallingToolInput, /) -> str:  # noqa: ARG002
        await asyncio.sleep(5)
        pytest.fail("Did not cancel call to invoke model.")


@pytest.fixture
def mocked_llm() -> FunctionModel:
    async def _stream(
        messages: list[ModelMessage],  # noqa: ARG001
        info: AgentInfo,  # noqa: ARG001
    ) -> AsyncIterator[DeltaToolCalls]:
        yield {0: DeltaToolCall(name="StallingTool", json_args="{}", tool_call_id="0")}

    return FunctionModel(stream_function=_stream)


@pytest.fixture
def orchestration_agent(mocked_llm: FunctionModel) -> OrchestrationAgent:
    """Instantiate and return orchestration agent."""
    return OrchestrationAgent(
        mocked_llm, system_prompt="Test system prompt", tools=[StallingTool()]
    )


class TestCancellation:
    async def test_orphaned_tool_calls_handled_on_cancellation(
        self,
        orchestration_agent: OrchestrationAgent,
        users_dao_with_user: UsersDAO,
        empty_chat: InternalChat,
        chat_context: KmnlpChatContext,
    ) -> None:
        chat = empty_chat

        # Set up: create the chat and insert the first user message
        users_dao_with_user.create_chat("test-user", chat)

        new_user_msg = UserMessage(user_message="Do stuff")
        chat.insert_chat_request(ChatRequest(content=new_user_msg.user_message), aoi=None)

        users_dao_with_user.update_chat("test-user", chat)

        emitted: list[chat_response_content_types] = []

        task = asyncio.create_task(
            get_agent_response_content_stream(
                orchestration_agent,
                new_user_msg=new_user_msg,
                chat=empty_chat,
                chat_context=chat_context,
                users_dao=users_dao_with_user,
                user_id="test-user",
                emit_response=lambda c: emitted.append(c),
            )
        )

        await asyncio.sleep(1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            log.info("Received cancellation")

        assert len(chat.agent_conversation) == 3
        tool_return_part = chat.agent_conversation[-1].parts[0]
        assert tool_return_part.part_kind == "tool-return"
        assert isinstance(tool_return_part.content, ErrorResponse)
        assert (
            tool_return_part.content.message
            == "Tool execution aborted due to client disconnection."
        )
