from unittest.mock import Mock

from llm_agent.context import LLMAgentContext

from app.providers import knowledge_graph_dao_provider
from app.utils.utils import AppState, make_agent


def override_app_state() -> AppState:
    """Get the application state from the request."""
    return AppState(
        context=LLMAgentContext(
            place_lookup=Mock(),
            s3_client=Mock(),
            knowledge_graph_dao=knowledge_graph_dao_provider(),
            dask_client=Mock(),
        ),
        agent=make_agent(
            bedrock_runtime_client=Mock(),
            orchestration_prompt_simple_transformer=None,
        ),
    )
