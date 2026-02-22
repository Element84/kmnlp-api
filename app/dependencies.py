from typing import Annotated

from fastapi import Header
from llm_agent.context import LLMAgentContext

from app.error_responses import MissingUserIdError
from app.providers import (
    bedrock_runtime_client_provider,
    dask_client_provider,
    knowledge_graph_dao_provider,
    place_lookup_provider,
    s3_client_provider,
)
from app.utils.utils import AppState, make_agent


def get_app_state() -> AppState:
    """Get the application state from the request."""
    return AppState(
        context=LLMAgentContext(
            place_lookup=place_lookup_provider(),
            s3_client=s3_client_provider(),
            knowledge_graph_dao=knowledge_graph_dao_provider(),
            dask_client=dask_client_provider(),
        ),
        agent=make_agent(
            bedrock_runtime_client=bedrock_runtime_client_provider(),
            orchestration_prompt_simple_transformer=None,
        ),
    )


def get_current_user(x_user_id: Annotated[str, Header()]) -> str:
    """Dependency function to read current user id from header."""
    if not x_user_id:
        raise MissingUserIdError
    return x_user_id
