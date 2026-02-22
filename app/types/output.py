from datetime import UTC, datetime
from functools import partial
from typing import Literal

from llm_agent.tools.response_shaping_tools.end_turn_tool import EmptyInput
from llm_agent.tools.response_shaping_tools.text_response_tool import TextResponse
from pydantic import BaseModel, Field

from app.types.geojson import GeoJSONFeatureCollection, GeoJSONGeometry
from app.types.input import ChatRequest


class EndOfTurn(EmptyInput):
    """End of LLM turn."""

    content_type: Literal["EndOfTurn"] = "EndOfTurn"


class Text(TextResponse):
    """Text response from the LLM."""

    content_type: Literal["Text"] = "Text"


class GeocodedAOI(BaseModel):
    """Geocoded AOI."""

    content_type: Literal["GeocodedAOI"] = "GeocodedAOI"
    query: str
    geojson: GeoJSONFeatureCollection


class DatasetSearchQuery(BaseModel):
    """Dataset search query."""

    content_type: Literal["DatasetSearchQuery"] = "DatasetSearchQuery"
    query: str


class DatasetMetadata(BaseModel):
    """Dataset metadata."""

    content_type: Literal["DatasetMetadata"] = "DatasetMetadata"
    title: str
    description: str
    temporal_coverage: str
    spatial_coverage: str
    variables: list[str]
    landing_page: str | None


class DatasetSearchDetails(BaseModel):
    """Dataset search details."""

    content_type: Literal["DatasetSearchDetails"] = "DatasetSearchDetails"
    query: str
    results: list[DatasetMetadata] | None = None


class Variable(BaseModel):
    """A variable used in data analysis."""

    name: str
    type: str
    description: str | None = None


class DataAnalysisProblem(BaseModel):
    """A data analysis problem."""

    content_type: Literal["DataAnalysisProblem"] = "DataAnalysisProblem"
    statement: str
    inputs_provided: list[Variable]
    outputs_requested: list[Variable]


class DataAnalysisSolution(BaseModel):
    """A data analysis solution."""

    content_type: Literal["DataAnalysisSolution"] = "DataAnalysisSolution"
    thoughts: str
    pseudocode: str
    explanation: str


class DataAnalysisOutput(BaseModel):
    """A data analysis output."""

    name: str
    value: str


class DataAnalysisDetails(BaseModel):
    """Dataset analysis details."""

    content_type: Literal["DataAnalysisDetails"] = "DataAnalysisDetails"
    problem: DataAnalysisProblem
    solution: DataAnalysisSolution | None = None
    outputs: list[DataAnalysisOutput] | None = None


class ProgressUpdate(BaseModel):
    """Progress update."""

    content_type: Literal["ProgressUpdate"] = "ProgressUpdate"
    title: str


class StepDetails(BaseModel):
    """Details of a step."""

    content_type: Literal["StepDetails"] = "StepDetails"
    content: DatasetSearchDetails | DataAnalysisDetails = Field(discriminator="content_type")


class Step(BaseModel):
    """A step inside a plan."""

    id: str
    title: str
    status: Literal["pending", "in_progress", "success", "failure", "cancelled"]
    outcome_summary: str | None = None
    progress_updates: list[ProgressUpdate] | None = None
    details: StepDetails | None = None


class Plan(BaseModel):
    """Plan."""

    content_type: Literal["Plan"] = "Plan"
    id: str
    title: str
    steps: list[Step]


class Legend(BaseModel):
    """A Legend to show on a map."""

    content_type: Literal["Legend"] = "Legend"
    label: str
    units: str | None
    min_value: float | None
    max_value: float | None


class GeoTIFF(BaseModel):
    """A GeoTIFF to show on a map."""

    content_type: Literal["GeoTIFF"] = "GeoTIFF"
    url: str
    legend: Legend


class Image(BaseModel):
    """An image to display in chat."""

    content_type: Literal["Image"] = "Image"
    media_type: Literal["image/png"] = "image/png"
    url: str


class BackendError(BaseModel):
    """API error message."""

    content_type: Literal["BackendError"] = "BackendError"
    message: str


chat_response_content_types = BackendError | EndOfTurn | Text | Plan | GeoTIFF | GeocodedAOI | Image


class ChatResponse(BaseModel):
    """/chat/ response model."""

    content: chat_response_content_types = Field(discriminator="content_type")
    timestamp: datetime = Field(default_factory=partial(datetime.now, tz=UTC))


class Chat(BaseModel):
    """External representation of chat data exposed via API."""

    id: str
    title: str
    messages: list[ChatRequest | ChatResponse] = Field(
        default_factory=list[ChatRequest | ChatResponse]
    )
    date_created: datetime


class ChatHeader(BaseModel):
    """Chat header with minimal information."""

    id: str
    title: str
    date_created: datetime


# FUTURE: include date_created field and remove chat_ids
class UserResponse(BaseModel):
    """Response model for user with their chat IDs."""

    user_id: str
    chat_ids: list[str]


class PaginationInfo(BaseModel):
    """Generic pagination information."""

    total_count: int
    next_cursor: str | None


class AOISummary(BaseModel):
    """AOI Summary."""

    uuid: str
    name: str
    created_at: datetime


class AOIListResponse(BaseModel):
    """Response model for listing AOIs."""

    aois: list[AOISummary]
    pagination_info: PaginationInfo


class AOI(BaseModel):
    """AOI Model."""

    uuid: str
    name: str
    geojson: GeoJSONGeometry
    created_at: datetime


class UUIDDict(BaseModel):
    """Model to return a single UUID."""

    uuid: str
