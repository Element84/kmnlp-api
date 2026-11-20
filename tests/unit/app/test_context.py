from llm_agent.tools.data_analysis_tool.types import GeometryRef, GeoTIFFRef, ImageRef
from shapely.geometry.base import BaseGeometry

from app.context import KmnlpChatContext
from app.types.geojson import GeoJSONGeometry


class TestKmnlpChatContext:
    def test_persist_and_retrieve_geometry_path(
        self,
        chat_context: KmnlpChatContext,
        gulf_of_mexico_geojson_path: str,
        gulf_of_mexico_geometry: BaseGeometry,
    ) -> None:
        key = chat_context.persist_data(gulf_of_mexico_geojson_path, data_type="geometry")
        ref = GeometryRef(key=key, name="")
        retrieved_geometry = chat_context.retrieve_data(ref)
        assert isinstance(retrieved_geometry, BaseGeometry)
        assert retrieved_geometry.equals(gulf_of_mexico_geometry)

    def test_persist_and_retrieve_geometry_shape(
        self, chat_context: KmnlpChatContext, gulf_of_mexico_geometry: BaseGeometry
    ) -> None:
        key = chat_context.persist_data(gulf_of_mexico_geometry, data_type="geometry")
        ref = GeometryRef(key=key, name="")
        retrieved_geometry = chat_context.retrieve_data(ref)
        assert isinstance(retrieved_geometry, BaseGeometry)
        assert retrieved_geometry.equals(gulf_of_mexico_geometry)

    def test_persist_and_retrieve_geometry_model(
        self,
        chat_context: KmnlpChatContext,
        gulf_of_mexico_geojson_model: GeoJSONGeometry,
        gulf_of_mexico_geometry: BaseGeometry,
    ) -> None:
        key = chat_context.persist_data(gulf_of_mexico_geojson_model, data_type="geometry")
        ref = GeometryRef(key=key, name="")
        retrieved_geometry = chat_context.retrieve_data(ref)
        assert isinstance(retrieved_geometry, BaseGeometry)
        assert retrieved_geometry.equals(gulf_of_mexico_geometry)

    def test_persist_and_retrieve_image_path(
        self, chat_context: KmnlpChatContext, sample_image_path: str, sample_image_bytes: bytes
    ) -> None:
        key = chat_context.persist_data(sample_image_path, data_type="image")
        ref = ImageRef(key=key, name="")
        retrieved_bytes = chat_context.retrieve_data(ref)
        assert isinstance(retrieved_bytes, bytes)
        assert retrieved_bytes == sample_image_bytes

    def test_persist_and_retrieve_image_bytes(
        self, chat_context: KmnlpChatContext, sample_image_bytes: bytes
    ) -> None:
        key = chat_context.persist_data(sample_image_bytes, data_type="image")
        ref = ImageRef(key=key, name="")
        retrieved_bytes = chat_context.retrieve_data(ref)
        assert isinstance(retrieved_bytes, bytes)
        assert retrieved_bytes == sample_image_bytes

    def test_persist_and_retrieve_geotiff(
        self, chat_context: KmnlpChatContext, sample_geotiff_path: str
    ) -> None:
        key = chat_context.persist_data(sample_geotiff_path, data_type="geotiff")
        ref = GeoTIFFRef(key=key, name="")
        retrieved_url = chat_context.retrieve_data(ref)
        assert isinstance(retrieved_url, str)
