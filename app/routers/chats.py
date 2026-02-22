import asyncio
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langfuse import get_client, observe  # pyright: ignore[reportUnknownVariableType]

from app.dao.aois_dao import AOIsDAO
from app.dao.users_dao import (
    ChatNotFoundError as DAOChatNotFoundError,
)
from app.dao.users_dao import (
    UserNotFoundError as DAOUserNotFoundError,
)
from app.dao.users_dao import (
    UsersDAO,
)
from app.dependencies import get_app_state, get_current_user
from app.error_responses import ChatNotFoundError, UserNotFoundError
from app.providers import aois_dao_provider, users_dao_provider
from app.types.geojson import GeoJSONGeometry
from app.types.internal import InternalChat
from app.types.output import (
    Chat,
    ChatHeader,
    ChatRequest,
    ChatResponse,
    EndOfTurn,
    Image,
    chat_response_content_types,
)
from app.utils.utils import (
    AppState,
    get_agent_response_content_stream,
    save_responses_to_db,
    serialize_obj_stream,
)

langfuse = get_client()

router = APIRouter(
    prefix="/chats",
    tags=["chats"],
)


@router.get("")
async def get_user_chats(
    user_id: Annotated[str, Depends(get_current_user)],
    users_dao: Annotated[UsersDAO, Depends(users_dao_provider)],
) -> list[ChatHeader]:
    """Fetch user chats."""
    try:
        return users_dao.get_user_chats(user_id)
    except DAOUserNotFoundError:
        raise UserNotFoundError(user_id) from None


@router.post("")
async def start_chat(
    user_id: Annotated[str, Depends(get_current_user)],
    users_dao: Annotated[UsersDAO, Depends(users_dao_provider)],
) -> Chat:
    """Start chat session."""
    chat_id = str(uuid4())
    new_internal_chat = InternalChat(id=chat_id, title=f"Chat {chat_id[:8]}")
    try:
        users_dao.create_chat(user_id, new_internal_chat)
    except DAOUserNotFoundError:
        raise UserNotFoundError(user_id) from None
    else:
        return new_internal_chat.to_external()


@router.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    user_id: Annotated[str, Depends(get_current_user)],
    users_dao: Annotated[UsersDAO, Depends(users_dao_provider)],
) -> Chat:
    """Get a chat by ID."""
    try:
        internal_chat = users_dao.get_chat(user_id, chat_id)
        return internal_chat.to_external()
    except DAOChatNotFoundError:
        raise ChatNotFoundError(chat_id) from None
    except DAOUserNotFoundError:
        raise UserNotFoundError(user_id) from None


@router.post(
    "/{chat_id}",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {
                "application/x-ndjson": {
                    "schema": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/ChatResponse"},
                    }
                }
            },
            "description": "ND-JSON stream of ChatResponses",
        }
    },
)
@observe(name="api.post_to_chat")
async def post_to_chat(  # noqa: PLR0913
    chat_id: str,
    message: ChatRequest,
    app_state: Annotated[AppState, Depends(get_app_state)],
    user_id: Annotated[str, Depends(get_current_user)],
    users_dao: Annotated[UsersDAO, Depends(users_dao_provider)],
    aois_dao: Annotated[AOIsDAO, Depends(aois_dao_provider)],
    *,
    stub_image: bool = False,
) -> StreamingResponse:
    """Reply to a chat message."""
    try:
        chat = users_dao.get_chat(user_id, chat_id)
    except DAOChatNotFoundError:
        raise ChatNotFoundError(chat_id) from None
    except DAOUserNotFoundError:
        raise UserNotFoundError(user_id) from None

    langfuse.update_current_trace(user_id=user_id, session_id=chat_id)

    # Add the user message to the chat
    aoi = _aoi_geometry_from_chat_request(message, user_id, aois_dao)
    chat.insert_chat_request(message, aoi)
    users_dao.update_chat(user_id, chat)

    # Create response queue for agent communication
    response_queue: asyncio.Queue[chat_response_content_types] = asyncio.Queue()

    # Function to emit responses to the queue
    def emit_response(content: chat_response_content_types) -> None:
        response_queue.put_nowait(content)

    # Start the agent response generation as a background task
    # Use a set to prevent garbage collection
    background_tasks: set[asyncio.Task[None]] = set()
    agent_task = asyncio.create_task(
        get_agent_response_content_stream(
            app_state.agent,
            chat=chat,
            app_state=app_state,
            users_dao=users_dao,
            user_id=user_id,
            emit_response=emit_response,
        )
    )
    background_tasks.add(agent_task)
    agent_task.add_done_callback(background_tasks.discard)

    # Create a generator that yields ChatResponse objects from the response queue
    async def response_content_stream() -> AsyncGenerator[chat_response_content_types]:
        while True:
            content = await response_queue.get()
            yield content
            if isinstance(content, EndOfTurn):
                break

    response_stream = (ChatResponse(content=c) async for c in response_content_stream())
    response_stream = save_responses_to_db(
        response_stream, chat=chat, users_dao=users_dao, user_id=user_id
    )

    if stub_image:
        response_stream = _add_devel_image(response_stream)

    return StreamingResponse(
        serialize_obj_stream(response_stream),
        media_type="application/x-ndjson",
    )


async def _add_devel_image(
    response_stream: AsyncGenerator[ChatResponse],
) -> AsyncGenerator[ChatResponse]:
    """Insert a development image to the response stream (before EndOfTurn)."""
    async for response in response_stream:
        if isinstance(response.content, EndOfTurn):
            yield ChatResponse(
                content=Image(
                    url="https://test-geotiff-bucket.s3.us-east-1.amazonaws.com/devel_image.png"
                )
            )
        yield response


def _aoi_geometry_from_chat_request(
    chat_request: ChatRequest, user_id: str, aois_dao: AOIsDAO
) -> GeoJSONGeometry | None:
    requested_aoi_id = chat_request.ui_state.aoi

    if requested_aoi_id is None:
        return None

    try:
        requested_aoi_uuid = UUID(requested_aoi_id)
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provided AOI UUID is malformed.",
        ) from e

    aoi = aois_dao.get_aoi(requested_aoi_uuid, user_id)

    if aoi is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Provided AOI UUID not found."
        )

    return aoi.geojson
