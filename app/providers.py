import logging
import os
from functools import cache
from typing import Annotated

import boto3
import botocore.config
from dask.distributed import Client
from distributed import LocalCluster
from fastapi import Depends
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_s3 import S3Client
from natural_language_geocoding.geocode_index.geocode_index_place_lookup import (
    GeocodeIndexPlaceLookup,
)
from natural_language_geocoding.place_lookup import NominatimAPI, PlaceLookup
from sqlalchemy import Engine, MetaData, create_engine

from app.dao.aois_dao import AOIsDAO
from app.dao.users_dao import UsersDAO
from app.db.postgresql_config import build_postgres_config
from app.db.sqlalchemy_db import SQLAlchemyDB
from app.utils.env import (
    ALLOW_LOCAL_DASK_CLUSTER,
    GEOCODE_INDEX_HOST,
    LITE_LLM_PROXY_URL,
    NOMINATIM_USER_AGENT,
    USE_SQLITE,
)

log = logging.getLogger(__name__)


def get_dask_client(
    dask_scheduler_address: str | None = None,
    *,
    allow_local_dask_cluster: bool = ALLOW_LOCAL_DASK_CLUSTER,
) -> Client:
    """Create Dask Client to use for the application.

    Return a client for the scheduler running running at the
    `DASK_SCHEDULER_ADDRESS`.

    If the `DASK_SCHEDULER_ADDRESS` is not set AND `ALLOW_LOCAL_DASK_CLUSTER`
    is set to `True`: create a local cluster and return a client referencing it.

    Do not create the local cluster if `ALLOW_LOCAL_DASK_CLUSTER` is not set.
    """
    dask_scheduler_address = dask_scheduler_address or os.getenv("DASK_SCHEDULER_ADDRESS")
    if dask_scheduler_address:
        log.info("Using Dask scheduler address from env var: %s", dask_scheduler_address)
        return Client(dask_scheduler_address, asynchronous=True)

    if allow_local_dask_cluster:
        local_cluster = LocalCluster(asynchronous=True)
        return Client(local_cluster)

    raise ValueError("DASK_SCHEDULER_ADDRESS is not set and ALLOW_LOCAL_DASK_CLUSTER is falsy.")


@cache
def bedrock_runtime_client_provider() -> BedrockRuntimeClient:
    """Return Bedrock Runtime Client for the application, caching the result."""
    return boto3.client(  # type: ignore[reportUnknownMemberType]
        "bedrock-runtime",
        config=botocore.config.Config(read_timeout=300, connect_timeout=2),
        endpoint_url=LITE_LLM_PROXY_URL,
    )


@cache
def s3_client_provider() -> S3Client:
    """Return S3 Client for the application, caching the result."""
    return boto3.client("s3")  # type: ignore[reportUnknownMemberType]


@cache
def place_lookup_provider() -> PlaceLookup:
    """Return PlaceLookup object for the application, caching the result."""
    # Prefer Geocode Index Place Lookup because it is more accurate
    if GEOCODE_INDEX_HOST is not None:
        return GeocodeIndexPlaceLookup()

    # Fallback to NominatimAPI
    if NOMINATIM_USER_AGENT is not None:
        return NominatimAPI()

    raise RuntimeError(
        "Set one of NOMINATIM_USER_AGENT or GEOCODE_INDEX_* env vars to choose a place lookup."
    )


@cache
def knowledge_graph_dao_provider() -> KnowledgeGraphDAO:
    """Return the Knowledge Graph DAO, caching the result."""
    return KnowledgeGraphDAO()


@cache
def sql_alchemy_engine_provider() -> Engine:
    """Return the Knowledge Graph DAO, caching the result."""
    if USE_SQLITE:
        return create_engine("sqlite:///temp/users.db", echo=False)

    config = build_postgres_config()
    return create_engine(
        "postgresql://",
        connect_args={
            "host": config.host,
            "port": config.port,
            "dbname": config.name,
            "user": config.user,
            "password": config.password,
            "sslmode": "require",
            "connect_timeout": 10,
        },
        echo=False,
    )


@cache
def sql_alchemy_metadata_provider() -> MetaData:
    """Return the Knowledge Graph DAO, caching the result."""
    return MetaData()


@cache
def db_provider(
    sql_alchemy_engine: Annotated[Engine, Depends(sql_alchemy_engine_provider)],
) -> SQLAlchemyDB:
    """Return the Knowledge Graph DAO, caching the result."""
    return SQLAlchemyDB(sql_alchemy_engine, sql_alchemy_metadata_provider())


@cache
def users_dao_provider(
    db: Annotated[SQLAlchemyDB, Depends(db_provider)],
) -> UsersDAO:
    """Return the Knowledge Graph DAO, caching the result."""
    return UsersDAO(db)


@cache
def aois_dao_provider(
    db: Annotated[SQLAlchemyDB, Depends(db_provider)],
) -> AOIsDAO:
    """Return the AOIs DAO, caching the result."""
    return AOIsDAO(db)
