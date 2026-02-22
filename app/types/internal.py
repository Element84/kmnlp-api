from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from textwrap import shorten
from typing import Literal
from uuid import UUID

from e84_geoai_common.llm.core import LLMUserMessage
from e84_geoai_common.llm.core.llm import LLMMessage, TextContent
from pydantic import BaseModel, ConfigDict, Field

from app.types.geojson import GeoJSONGeometry
from app.types.input import ChatRequest
from app.types.output import Chat, ChatHeader, ChatResponse
from app.utils.plan_manager import PlanManager


class InternalChat(BaseModel):
    """Internal representation of chat data with agent conversation history."""

    id: str
    title: str
    messages: list[ChatRequest | ChatResponse] = Field(
        default_factory=list[ChatRequest | ChatResponse]
    )
    agent_conversation: list[LLMMessage] = Field(
        default_factory=list[LLMMessage],
        description="Internal agent conversation history.",
    )
    plan_manager: PlanManager = Field(default_factory=PlanManager)
    date_created: datetime = Field(default_factory=partial(datetime.now, tz=UTC))

    def insert_chat_request(self, message: ChatRequest, aoi: GeoJSONGeometry | None) -> None:
        """Add a message to the chat and agent conversation history."""
        # Set title from first message
        # FUTURE: allow LLM to set title
        is_first_message = len(self.messages) == 0
        if is_first_message:
            self.title = shorten(message.content.strip(), width=50, placeholder="...")

        self.messages.append(message)

        text_content = TextContent(text=message.content.strip())
        aoi_content = TextContent(text=f"Provided area: {aoi.model_dump_json()}") if aoi else None

        msg = LLMUserMessage(
            content=[content for content in [text_content, aoi_content] if content is not None],
        )
        self.agent_conversation.append(msg)

    def to_external(self) -> Chat:
        """Convert to external Chat representation."""
        return Chat(
            id=self.id,
            title=self.title,
            messages=self.messages,
            date_created=self.date_created,
        )


class User(BaseModel, **ConfigDict(arbitrary_types_allowed=True)):
    """User model."""

    id: str
    date_created: datetime = Field(default_factory=partial(datetime.now, tz=UTC))
    chats: dict[str, ChatHeader] = Field(default_factory=dict)


@dataclass
class AOI:
    """AOI Model."""

    uuid: UUID
    user_id: str
    name: str
    geojson: GeoJSONGeometry
    created_at: datetime


@dataclass
class AOISummary:
    """AOI Summary.

    Intended for use with collections to avoid overfetching large amounts of data.
    """

    uuid: UUID
    user_id: str
    name: str
    created_at: datetime


@dataclass
class Cursor:
    """Cursor to use to resume a search."""

    key: Literal["created_at"]
    value: datetime


@dataclass
class AOIListResponse:
    """AOI list with metadata."""

    aois: list[AOISummary]
    cursor: Cursor | None
    total_count: int
