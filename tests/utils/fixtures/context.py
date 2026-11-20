from collections.abc import Callable, Generator
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from moto import mock_aws
from mypy_boto3_s3 import S3Client
from natural_language_geocoding.tests.canned_place_lookup import CannedPlaceLookup

from app.context import KmnlpChatContext, KmnlpGlobalContext
from app.dao.aois_dao import AOIsDAO
from app.providers import knowledge_graph_dao_provider, s3_client_provider
from app.types.internal import InternalChat

if TYPE_CHECKING:
    from dask.distributed import Client as DaskClient
    from natural_language_geocoding.place_lookup import PlaceLookup
    from shapely.geometry.base import BaseGeometry


@pytest.fixture(scope="session")
def default_knowledge_graph_dao() -> KnowledgeGraphDAO:
    return knowledge_graph_dao_provider()


@pytest.fixture(scope="session")
def canned_place_lookup(
    black_sea_geometry: "BaseGeometry", gulf_of_mexico_geometry: "BaseGeometry"
) -> CannedPlaceLookup:
    place_lookup = CannedPlaceLookup()
    place_lookup.name_to_geom["gulf of mexico"] = gulf_of_mexico_geometry
    place_lookup.name_to_geom["black sea"] = black_sea_geometry
    return place_lookup


@pytest.fixture
def mock_s3_client() -> Generator[S3Client, None, None]:
    """Fixture for a mocked S3 client."""
    with mock_aws():
        s3_client = s3_client_provider()
        s3_client.create_bucket(Bucket="test-bucket")
        yield s3_client


@pytest.fixture
def global_context(
    default_knowledge_graph_dao: KnowledgeGraphDAO,
    canned_place_lookup: CannedPlaceLookup,
    mock_s3_client: S3Client,
) -> KmnlpGlobalContext:
    """Fixture for a default global context."""
    return KmnlpGlobalContext(
        place_lookup=canned_place_lookup,
        knowledge_graph_dao=default_knowledge_graph_dao,
        embedding_repo=None,
        dask_client=Mock(),
        s3_client=mock_s3_client,
        s3_user_data_root="s3://test-bucket",
        bedrock_client=Mock(),
    )


@pytest.fixture
def global_context_dynamic(
    default_knowledge_graph_dao: KnowledgeGraphDAO,
    canned_place_lookup: CannedPlaceLookup,
    mock_s3_client: S3Client,
) -> Callable[..., KmnlpGlobalContext]:
    """Fixture for a default global context."""

    def _make_context(
        *,
        place_lookup: "PlaceLookup | None" = None,
        knowledge_graph_dao: "KnowledgeGraphDAO | None" = None,
        dask_client: "DaskClient | None" = None,
        s3_client: "S3Client | None" = None,
        s3_user_data_root: "str | None" = None,
    ) -> KmnlpGlobalContext:
        return KmnlpGlobalContext(
            place_lookup=place_lookup or canned_place_lookup,
            knowledge_graph_dao=knowledge_graph_dao or default_knowledge_graph_dao,
            embedding_repo=None,
            dask_client=dask_client or Mock(),
            s3_client=s3_client or mock_s3_client,
            s3_user_data_root=s3_user_data_root or "s3://test-bucket",
            bedrock_client=Mock(),
        )

    return _make_context


@pytest.fixture
def empty_chat() -> InternalChat:
    """Fixture for a default empty chat."""
    return InternalChat(
        id="test-chat-id",
        title="Test chat",
    )


@pytest.fixture
def chat_context(
    global_context: KmnlpGlobalContext, empty_chat: InternalChat, aois_dao: AOIsDAO
) -> KmnlpChatContext:
    """Fixture for a default chat context."""
    return KmnlpChatContext(
        user_id="test-user",
        chat=empty_chat,
        global_context=global_context,
        aois_dao=aois_dao,
        base_url="http://testserver",
    )
