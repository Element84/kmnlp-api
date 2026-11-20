from typing import Annotated

from fastapi import Header, Request

from app.error_responses import MissingUserIdError
from app.utils.app_state import AppState


def get_app_state(request: Request) -> AppState:
    """Get the application state from the request."""
    return request.app.state.app_state


def get_current_user(x_user_id: Annotated[str, Header()]) -> str:
    """Dependency function to read current user id from header."""
    if not x_user_id:
        raise MissingUserIdError
    return x_user_id
