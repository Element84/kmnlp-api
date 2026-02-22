from unittest.mock import MagicMock, patch

import pytest
from dask.distributed import Client
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from moto import mock_aws
from natural_language_geocoding.place_lookup import PlaceLookup
from sqlalchemy import MetaData, create_engine

from app.dao.aois_dao import AOIsDAO
from app.dao.users_dao import UsersDAO
from app.db.sqlalchemy_db import SQLAlchemyDB
from app.providers import (
    _create_dask_client,  # type: ignore[reportPrivateUsage]
    aois_dao_provider,  # type: ignore[reportPrivateUsage]
    bedrock_runtime_client_provider,
    dask_client_provider,
    db_provider,
    knowledge_graph_dao_provider,
    place_lookup_provider,
    s3_client_provider,
    users_dao_provider,
)


def test_dask_provider():
    client = dask_client_provider()
    assert isinstance(client, Client)


def test_dask_provider_returns_same_instance():
    client1 = dask_client_provider()
    client2 = dask_client_provider()
    assert client1 is client2


def test_dask_client_provider_with_scheduler_address():
    with (
        patch("app.providers.Client") as mock_client,
        patch("app.providers.LocalCluster") as mock_local_cluster,
    ):
        client = _create_dask_client("tcp://10.0.0.1:8786", allow_local_dask_cluster=False)

        mock_client.assert_called_once_with("tcp://10.0.0.1:8786")
        assert isinstance(client, MagicMock)
        mock_local_cluster.assert_not_called()


def test_dask_client_provider_no_scheduler_address():
    with pytest.raises(
        ValueError, match="DASK_SCHEDULER_ADDRESS is not set and ALLOW_LOCAL_DASK_CLUSTER is falsy"
    ):
        _create_dask_client(None, allow_local_dask_cluster=False)


def test_dask_client_provider_with_scheduler_address_and_allow_local():
    with (
        patch("app.providers.Client") as mock_client,
        patch("app.providers.LocalCluster") as mock_local_cluster,
    ):
        client = _create_dask_client("tcp://10.0.0.1:8786", allow_local_dask_cluster=True)

        mock_client.assert_called_once_with("tcp://10.0.0.1:8786")
        assert isinstance(client, MagicMock)
        mock_local_cluster.assert_not_called()


def test_dask_client_provider_with_no_scheduler_address_and_allow_local():
    with (
        patch("app.providers.Client") as mock_client,
        patch("app.providers.LocalCluster") as mock_local_cluster,
    ):
        client = _create_dask_client(None, allow_local_dask_cluster=True)

        mock_client.assert_called_once()
        assert isinstance(client, MagicMock)
        mock_local_cluster.assert_called_once()


@mock_aws
def test_bedrock_runtime_client_provider():
    client = bedrock_runtime_client_provider()
    assert hasattr(client, "invoke_model")


@mock_aws
def test_bedrock_runtime_client_provider_returns_same_instance():
    client1 = bedrock_runtime_client_provider()
    client2 = bedrock_runtime_client_provider()
    assert client1 is client2


@mock_aws
def test_s3_client_provider():
    client = s3_client_provider()
    assert hasattr(client, "get_object")


@mock_aws
def test_s3_client_provider_returns_same_instance():
    client1 = s3_client_provider()
    client2 = s3_client_provider()
    assert client1 is client2


def test_place_lookup_provider():
    place_lookup = place_lookup_provider()
    assert isinstance(place_lookup, PlaceLookup)


def test_place_lookup_provider_returns_same_instance():
    place_lookup1 = place_lookup_provider()
    place_lookup2 = place_lookup_provider()
    assert place_lookup1 is place_lookup2


def test_knowledge_graph_dao_provider():
    knowledge_graph_dao = knowledge_graph_dao_provider()
    assert isinstance(knowledge_graph_dao, KnowledgeGraphDAO)


def test_knowledge_graph_dao_provider_returns_same_instance():
    knowledge_graph_dao1 = knowledge_graph_dao_provider()
    knowledge_graph_dao2 = knowledge_graph_dao_provider()
    assert knowledge_graph_dao1 is knowledge_graph_dao2


def test_users_dao_provider():
    engine = create_engine("sqlite+pysqlite:///:memory:", echo=False)
    db = db_provider(engine)
    users_dao = users_dao_provider(db)
    assert isinstance(users_dao, UsersDAO)
    assert users_dao.db == db


def test_users_dao_provider_returns_same_instance():
    engine = create_engine("sqlite+pysqlite:///:memory:", echo=False)
    db = db_provider(engine)
    users_dao1 = users_dao_provider(db)
    users_dao2 = users_dao_provider(db)
    assert users_dao1 is users_dao2


def test_users_dao_provider_returns_different_instances_if_different_db_providers():
    db1 = SQLAlchemyDB(create_engine("sqlite+pysqlite:///:memory:", echo=True), MetaData())
    db2 = SQLAlchemyDB(create_engine("sqlite+pysqlite:///:memory:", echo=True), MetaData())

    # Assert that db1 and db2 are not the same object. (This test setup is a
    # little weird, so it's worth validating.)
    assert db1 is not db2

    # Create 2 DAOs
    users_dao1 = users_dao_provider(db1)
    users_dao2 = users_dao_provider(db2)

    # Assert DAOs are different
    assert users_dao1 is not users_dao2


def test_db_provider():
    engine = create_engine("sqlite+pysqlite:///:memory:", echo=False)
    db = db_provider(engine)
    assert isinstance(db, SQLAlchemyDB)


def test_sqlite_user_datatabase_provider_return_same_instance():
    engine = create_engine("sqlite+pysqlite:///:memory:", echo=False)
    db1 = db_provider(engine)
    db2 = db_provider(engine)
    assert db1 is db2


def test_aois_dao_provider():
    engine = create_engine("sqlite+pysqlite:///:memory:", echo=False)
    db = db_provider(engine)
    aois_dao = aois_dao_provider(db)
    assert isinstance(aois_dao, AOIsDAO)
    assert aois_dao.db == db


def test_aois_dao_provider_returns_same_instance():
    engine = create_engine("sqlite+pysqlite:///:memory:", echo=False)
    db = db_provider(engine)
    aois_dao1 = aois_dao_provider(db)
    aois_dao2 = aois_dao_provider(db)
    assert aois_dao1 is aois_dao2


def test_aois_dao_provider_returns_different_instances_if_different_db_providers():
    db1 = SQLAlchemyDB(create_engine("sqlite+pysqlite:///:memory:", echo=True), MetaData())
    db2 = SQLAlchemyDB(create_engine("sqlite+pysqlite:///:memory:", echo=True), MetaData())

    aois_dao1 = aois_dao_provider(db1)
    aois_dao2 = aois_dao_provider(db2)

    assert aois_dao1 is not aois_dao2


@pytest.mark.skip("Not implemented")
def test_sqlalchemy_engine_provider():
    pytest.fail("Not implemented.")


@pytest.mark.skip("Not implemented")
def test_sqlalchemy_metadata_provider():
    pytest.fail("Not implemented.")
