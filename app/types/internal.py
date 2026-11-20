import json
from datetime import UTC, datetime
from functools import partial
from textwrap import shorten
from typing import Literal
from uuid import UUID

from llm_agent.agents.orchestration_agent.interface import UserMessage
from llm_agent.tools.data_analysis_tool.types import GeometryRef
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest
from pydantic_core import to_jsonable_python

from app.types.geojson import GeoJSONFeatureCollection, GeoJSONGeometry
from app.types.input import ChatRequest
from app.types.output import Chat, ChatHeader, ChatResponse
from app.utils.plan_manager import PlanManager


class InternalDataRef(BaseModel):
    """Reference to data persisted by the system."""

    key: str
    data_type: Literal["geometry", "image", "geotiff", "animation"]
    location: Literal["s3", "db"]
    date_created: datetime = Field(default_factory=partial(datetime.now, tz=UTC))


class InternalChat(BaseModel):
    """Internal representation of chat data with agent conversation history."""

    id: str
    title: str
    messages: list[ChatRequest | ChatResponse] = Field(
        default_factory=list[ChatRequest | ChatResponse]
    )
    agent_conversation: list[ModelMessage] = Field(
        default_factory=list[ModelMessage],
        description="Internal agent conversation history.",
    )
    plan_manager: PlanManager = Field(default_factory=PlanManager)
    data_refs: list[InternalDataRef] = Field(default_factory=list[InternalDataRef])
    date_created: datetime = Field(default_factory=partial(datetime.now, tz=UTC))

    def chat_request_to_user_message(self, message: ChatRequest, aoi: "AOI | None") -> UserMessage:
        """Convert a ChatRequest to a UserMessage, adding any attached data references."""
        if aoi is not None:
            description = "Area of interest"
            if isinstance(aoi.geojson, GeoJSONFeatureCollection):
                description = (
                    f"Area of interest with {len(aoi.geojson.features)} features. "
                    f"Unique geometry types: {', '.join(set(aoi.geojson.geom_types))}"
                )
            aoi_data_ref = GeometryRef(
                key=str(aoi.uuid),
                name=aoi.name,
                description=description,
                geometry_type=aoi.geojson.type,
                bbox=aoi.geojson.bbox,
            )
            aoi_data_internal_ref = InternalDataRef(
                key=str(aoi.uuid), data_type="geometry", location="db"
            )
            self.data_refs.append(aoi_data_internal_ref)
        else:
            aoi_data_ref = None

        return UserMessage(
            user_message=message.content.strip(),
            attached_data=[aoi_data_ref] if aoi_data_ref else [],
        )

    def insert_chat_request(self, message: ChatRequest, aoi: "AOI | None") -> UserMessage:
        """Add a message to the chat and agent conversation history."""
        # Set title from first message
        # FUTURE: allow LLM to set title
        is_first_message = len(self.messages) == 0
        if is_first_message:
            self.title = shorten(message.content.strip(), width=50, placeholder="...")

        self.messages.append(message)

        msg = self.chat_request_to_user_message(message, aoi)
        self.agent_conversation.append(ModelRequest.user_text_prompt(msg.model_dump_json()))
        return msg

    def insert_conversation_messages(self, messages: list[ModelMessage]) -> None:
        """Add messages to the agent conversation history."""
        self.agent_conversation.extend(messages)

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
    geojson: GeoJSONGeometry | GeoJSONFeatureCollection
    created_at: datetime = Field(default_factory=partial(datetime.now, tz=UTC))


@dataclass
class AOISummary:
    """AOI Summary.

    Intended for use with collections to avoid overfetching large amounts of data.
    """

    uuid: UUID
    user_id: str
    name: str
    created_at: datetime = Field(default_factory=partial(datetime.now, tz=UTC))


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


def serialize_agent_conversation(agent_conversation: list[ModelMessage]) -> str:
    """Serialize the agent conversation history to JSON."""
    return json.dumps(to_jsonable_python(agent_conversation))


def deserialize_agent_conversation(json_str: str) -> list[ModelMessage]:
    """Deserialize the agent conversation history from JSON."""
    return ModelMessagesTypeAdapter.validate_json(json_str)
