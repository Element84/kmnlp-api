import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from e84_geoai_common.util import get_env_var
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langfuse import get_client

from app.context import KmnlpGlobalContext
from app.providers import (
    bedrock_runtime_client_provider,
    get_dask_client,
    knowledge_graph_dao_provider,
    place_lookup_provider,
    s3_client_provider,
)
from app.routers import aois, chats, system, users
from app.utils.app_state import AppState
from app.utils.orchestration.orchestration_agent import make_agent

log = logging.getLogger(__name__)

langfuse = get_client()

S3_USER_DATA_BUCKET = "kmnlp-api-user-data"


def configure_app(app: FastAPI) -> None:
    """Configure the app.

    Add middleware and routers in this function.

    This method allows feature tests to use the same configuration that
    production uses on an app that was created with a different lifespan
    function.
    """
    frontend_endpoint = get_env_var("FRONTEND_ENDPOINT")
    app.add_middleware(
        CORSMiddleware,
        # Localhost to always allow local testing
        allow_origins=["http://localhost:5173", frontend_endpoint],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "x-user-id"],
    )

    app.include_router(users.router)
    app.include_router(chats.router)
    app.include_router(aois.router)
    app.include_router(system.router)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan function to manage startup and shutdown events."""
    log.info("Starting up the application...")

    with get_dask_client() as dask_client:
        bedrock_client = bedrock_runtime_client_provider()
        app.state.app_state = AppState(
            context=KmnlpGlobalContext(
                place_lookup=place_lookup_provider(),
                knowledge_graph_dao=knowledge_graph_dao_provider(),
                embedding_repo=None,
                dask_client=dask_client,
                s3_client=s3_client_provider(),
                s3_user_data_root=f"s3://{S3_USER_DATA_BUCKET}",
                bedrock_client=bedrock_client,
            ),
            agent=make_agent(bedrock_runtime_client=bedrock_client),
        )
        yield

    log.info("Shutting down the application...")


app = FastAPI(lifespan=lifespan)
configure_app(app)


@app.get("/health")
async def health_check() -> str:
    """Use the check whether server is ready to receive requests."""
    return "OK"
