from unittest.mock import Mock

from llm_agent.context import LLMAgentContext
from llm_agent.dao.knowledge_graph_dao import KnowledgeGraphDAO
from natural_language_geocoding.tests.canned_place_lookup import CannedPlaceLookup

from app.utils.utils import AppState, make_agent


def app_state_stub() -> AppState:
    return AppState(
        context=LLMAgentContext(
            place_lookup=CannedPlaceLookup(),
            s3_client=Mock(),
            knowledge_graph_dao=KnowledgeGraphDAO(),
            dask_client=Mock(),
        ),
        agent=make_agent(
            bedrock_runtime_client=Mock(), orchestration_prompt_simple_transformer=None
        ),
    )
