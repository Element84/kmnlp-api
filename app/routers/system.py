"""System information endpoints."""

from fastapi import APIRouter

from app.types.output import ModelInfo

router = APIRouter(
    prefix="/system",
    tags=["system"],
)


@router.get("/models")
async def get_models() -> list[ModelInfo]:
    """Get information about LLM models used in the system."""
    return [
        ModelInfo(
            name="Anthropic Claude Opus 4.6",
            description=(
                "Used for the orchestration agent, data search agent, and algorithm agent."
            ),
        ),
        ModelInfo(
            name="Anthropic Claude Haiku 4.5",
            description=("Used for natural language geocoding"),
        ),
    ]
