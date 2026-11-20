from collections.abc import AsyncGenerator

from pydantic import BaseModel


async def serialize_obj_stream(stream: AsyncGenerator[BaseModel]) -> AsyncGenerator[str]:
    """Serialize a pydantic object stream to ND-JSON."""
    async for obj in stream:
        yield f"{obj.model_dump_json()}\n"
