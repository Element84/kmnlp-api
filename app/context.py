import json
from dataclasses import dataclass
from os.path import join
from typing import TYPE_CHECKING, Any, Literal, overload

import langfuse
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from llm_agent.context import (
    ChatContext,
    DataPersistenceError,
    DataRefNotResolvedError,
    LLMAgentContext,
)
from mypy_boto3_s3.client import S3Client
from pydantic import BaseModel
from shapely import from_geojson  # pyright: ignore[reportUnknownVariableType]
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from app.dao.aois_dao import AOIsDAO
from app.types.internal import InternalChat, InternalDataRef
from app.utils import s3

if TYPE_CHECKING:
    from llm_agent.tools.data_analysis_tool.types import (
        AnimationRef,
        GeometryRef,
        GeoTIFFRef,
        ImageRef,
    )


@dataclass(repr=False)
class KmnlpGlobalContext(LLMAgentContext):
    """Global context for KMNLP API, shared across all chats and users."""

    s3_client: S3Client
    s3_user_data_root: str


class KmnlpChatContext(ChatContext):
    """Custom chat context for KMNLP API.

    The main goal of this class is to manage the persistence and retrieval of data associated with
    a chat, such as geometries, images, and GeoTIFFs.
    """

    user_id: str
    chat: InternalChat
    aois_dao: AOIsDAO
    s3_client: S3Client
    data_refs: dict[str, InternalDataRef]
    base_url: str

    @property
    def global_context(self) -> KmnlpGlobalContext:
        """Global context for the entire application."""
        return self._global_context

    def __init__(
        self,
        *,
        user_id: str,
        chat: InternalChat,
        global_context: KmnlpGlobalContext,
        aois_dao: AOIsDAO,
        base_url: str,
    ) -> None:
        """Constructor."""
        self._global_context = global_context
        self.user_id = user_id
        self.chat = chat
        self.s3_client = global_context.s3_client
        self.aois_dao = aois_dao
        self.base_url = base_url

        self.data_refs = {ref.key: ref for ref in chat.data_refs}

    @observe(as_type="chain", name="KmnlpChatContext.persist_data", capture_input=False)
    def persist_data(
        self,
        data_or_path: Any,  # noqa: ANN401
        *,
        data_type: Literal["geometry", "image", "geotiff", "animation"],
    ) -> str:
        """Persist data and return a reference to it."""
        langfuse.get_client().update_current_span(
            input={
                "data_or_path": (
                    data_or_path if isinstance(data_or_path, str) else type(data_or_path).__name__
                ),
                "data_type": data_type,
            }
        )
        try:
            return self._persist_data(data_or_path, data_type=data_type)
        except Exception as e:
            raise DataPersistenceError(data_or_path, data_type) from e

    def _persist_data(
        self,
        data_or_path: Any,  # noqa: ANN401
        *,
        data_type: Literal["geometry", "image", "geotiff", "animation"],
    ) -> str:
        """Persist data to S3 and return a reference to it."""
        data_key = self._make_data_key(data_type=data_type)
        s3_uri = self._data_key_to_s3_uri(data_key)
        match data_type:
            case "geometry":
                self._persist_geometry(data_or_path, s3_uri)
            case "image":
                self._persist_image(data_or_path, s3_uri)
            case "geotiff":
                self._persist_geotiff(data_or_path, s3_uri)
            case "animation":
                self._persist_animation(data_or_path, s3_uri)

        data_ref = InternalDataRef(key=data_key, data_type=data_type, location="s3")
        self.data_refs[data_ref.key] = data_ref
        self.chat.data_refs.append(data_ref)
        return data_ref.key

    def _persist_geometry(self, data_or_path: Any, s3_uri: str) -> None:  # noqa: ANN401
        match data_or_path:
            case str() as path:
                s3.upload_file(self.s3_client, path, s3_uri)
            case BaseGeometry() as geometry:
                s3.upload_object(self.s3_client, json.dumps(mapping(geometry)), s3_uri)
            case BaseModel() as geometry:
                s3.upload_object(self.s3_client, geometry.model_dump_json(), s3_uri)
            case _:
                raise TypeError(
                    "Allowed types for geometry data: str, BaseGeometry, or BaseModel. "
                    f"Received type: {type(data_or_path)}"
                )

    def _persist_image(self, data_or_path: Any, s3_uri: str) -> None:  # noqa: ANN401
        match data_or_path:
            case str() as path:
                s3.upload_file(self.s3_client, path, s3_uri)
            case bytes() as data_bytes:
                s3.upload_object(self.s3_client, data_bytes, s3_uri)
            case _:
                raise TypeError(
                    "Allowed types for image data: str or bytes. "
                    f"Received type: {type(data_or_path)}"
                )

    def _persist_geotiff(self, data_or_path: Any, s3_uri: str) -> None:  # noqa: ANN401
        match data_or_path:
            case str() as path:
                s3.upload_file(self.s3_client, path, s3_uri)
            case _:
                raise TypeError(
                    "Allowed types for geotiff data: str (local path to a GeoTIFF). "
                    f"Received type: {type(data_or_path)}"
                )

    def _persist_animation(self, data_or_path: Any, s3_uri: str) -> None:  # noqa: ANN401
        match data_or_path:
            case str() as path:
                s3.upload_file(self.s3_client, path, s3_uri)
            case _:
                raise TypeError(
                    "Allowed types for animation data: str (local path to an animation). "
                    f"Received type: {type(data_or_path)}"
                )

    @overload
    def retrieve_data(self, data_ref: "GeometryRef") -> "BaseGeometry": ...

    @overload
    def retrieve_data(self, data_ref: "GeoTIFFRef") -> str: ...

    @overload
    def retrieve_data(self, data_ref: "ImageRef") -> bytes: ...

    @overload
    def retrieve_data(self, data_ref: "AnimationRef") -> str: ...

    @observe(as_type="chain", name="KmnlpChatContext.retrieve_data")
    def retrieve_data(self, data_ref: "GeometryRef | ImageRef | GeoTIFFRef | AnimationRef") -> Any:
        """Retrieve data for a given reference."""
        try:
            return self._retrieve_data(data_ref)
        except Exception as e:
            raise DataRefNotResolvedError(data_ref) from e

    def _retrieve_data(self, data_ref: "GeometryRef | ImageRef | GeoTIFFRef | AnimationRef") -> Any:  # noqa: ANN401
        """Retrieve data for a given reference."""
        internal_ref = self.data_refs[data_ref.key]
        match internal_ref.location:
            case "s3":
                return self._retrieve_data_from_s3(internal_ref)
            case "db":
                return self._retrieve_data_from_db(internal_ref)

    def retrieve_url(self, key: str) -> str:
        """Retrieve a URL for the given data reference."""
        try:
            internal_ref = self.data_refs[key]
            match internal_ref.location:
                case "s3":
                    return self._data_key_to_data_url(internal_ref.key)
                case "db":
                    raise NotImplementedError  # noqa: TRY301
        except Exception as e:
            raise DataRefNotResolvedError(key) from e

    def _retrieve_data_from_s3(self, data_ref: InternalDataRef) -> "BaseGeometry | bytes | str":
        s3_uri = self._data_key_to_s3_uri(data_ref.key)
        if not s3.exists(self.s3_client, s3_uri):
            raise FileNotFoundError(f"S3 object not found: {s3_uri}")

        match data_ref.data_type:
            case "geotiff" | "animation":
                return self._data_key_to_data_url(data_ref.key)
            case "geometry":
                obj = s3.get_object(self.s3_client, s3_uri)
                # Note: returns a GeometryCollection if the GeoJSON is a FeatureCollection.
                return from_geojson(obj)
            case "image":
                return s3.get_object(self.s3_client, s3_uri)

    def _retrieve_data_from_db(self, data_ref: InternalDataRef) -> Any:  # noqa: ANN401
        match data_ref.data_type:
            case "geometry":
                aoi = self.aois_dao.get_aoi(data_ref.key, self.user_id)
                if aoi is None:
                    raise DataRefNotResolvedError(data_ref.key)
                return from_geojson(aoi.geojson.model_dump_json())
            case _:
                raise NotImplementedError

    def _make_data_key(
        self, data_type: Literal["geotiff", "animation", "image", "geometry"]
    ) -> str:
        """Generate a unique data key for storing data in S3."""
        ext_map = {
            "geotiff": ".tif",
            "animation": ".mp4",
            "image": ".png",
            "geometry": ".json",
        }
        ext = ext_map.get(data_type, "")
        return f"{len(self.data_refs):06d}{ext}"

    def _data_key_to_data_url(self, data_key: str) -> str:
        return join(self.base_url, "chats", self.chat.id, "data", data_key)

    def _data_key_to_s3_uri(self, data_key: str) -> str:
        return join(
            self.global_context.s3_user_data_root,
            s3.make_data_key_path(self.user_id, self.chat.id, data_key),
        )
