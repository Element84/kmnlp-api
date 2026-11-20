from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest

from app.context import KmnlpGlobalContext
from app.utils.app_state import AppState


@pytest.fixture
def app_state(global_context: KmnlpGlobalContext) -> AppState:
    """Get the application state from the request."""
    return AppState(context=global_context, agent=Mock())


@pytest.fixture
def app_state_custom_agent(
    global_context_dynamic: Callable[..., KmnlpGlobalContext],
) -> Callable[..., AppState]:
    """Return app state for use in feature tests."""

    def _make_app_state(*, agent: Any) -> AppState:  # noqa: ANN401
        return AppState(
            context=global_context_dynamic(),
            agent=agent,
        )

    return _make_app_state
