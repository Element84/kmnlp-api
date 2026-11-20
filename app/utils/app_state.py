from dataclasses import dataclass

from llm_agent.agents.orchestration_agent.orchestration_agent import OrchestrationAgent

from app.context import KmnlpGlobalContext


@dataclass()
class AppState:
    """Application state container."""

    context: KmnlpGlobalContext
    agent: OrchestrationAgent
