from e84_geoai_common.util import get_env_var
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langfuse import get_client  # pyright: ignore[reportUnknownVariableType]

from app.routers import aois, chats, users

langfuse = get_client()


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


app = FastAPI()
configure_app(app)


@app.get("/health")
async def health_check() -> str:
    """Use the check whether server is ready to receive requests."""
    return "OK"
