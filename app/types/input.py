from datetime import UTC, datetime
from functools import partial
from uuid import UUID

from pydantic import BaseModel, Field

from app.types.geojson import GeoJSONFeatureCollection, GeoJSONGeometry


class UIState(BaseModel):
    """UI state."""

    aoi: UUID | None = Field(default=None, description="AOI UUID.")


class ChatRequest(BaseModel):
    """/chat/ input model."""

    content: str
    timestamp: datetime = Field(default_factory=partial(datetime.now, tz=UTC))
    ui_state: UIState = Field(default_factory=UIState)


class UserCreateRequest(BaseModel):
    """Request model for creating a new user."""

    user_id: str


class AOI(BaseModel):
    """Request model for creating a new AOI."""

    name: str
    geometry: GeoJSONGeometry | GeoJSONFeatureCollection
