import asyncio
import logging
from collections.abc import AsyncGenerator
from os.path import join
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langfuse import get_client, observe  # pyright: ignore[reportUnknownVariableType]

from app.context import KmnlpChatContext
from app.dao.aois_dao import AOIsDAO
from app.dao.users_dao import ChatNotFoundError as DAOChatNotFoundError
from app.dao.users_dao import UserNotFoundError as DAOUserNotFoundError
from app.dao.users_dao import UsersDAO
from app.dependencies import get_app_state, get_current_user
from app.error_responses import ChatNotFoundError, UserNotFoundError
from app.providers import aois_dao_provider, users_dao_provider
from app.transformers.responses import serialize_obj_stream
from app.types.internal import AOI, InternalChat
from app.types.output import (
    BackendError,
    Chat,
    ChatHeader,
    ChatRequest,
    ChatResponse,
    DataDownloadURL,
    EndOfTurn,
    EphemeralPlan,
    TextDelta,
    ThinkingDelta,
    chat_response_content_types,
)
from app.utils import s3
from app.utils.app_state import AppState
from app.utils.orchestration.orchestration_agent import get_agent_response_content_stream

log = logging.getLogger(__name__)

PRESIGNED_URL_EXPIRY_SECONDS = 60 * 60 * 24  # 24 hours


langfuse = get_client()

router = APIRouter(
    prefix="/chats",
    tags=["chats"],
)


@router.get("")
async def get_user_chats(
    user_id: Annotated[str, Depends(get_current_user)],
    users_dao: Annotated[UsersDAO, Depends(users_dao_provider)],
    offset: int = 0,
) -> list[ChatHeader]:
    """Fetch user chats, up to 25 at a time; use offset for pagination."""
    try:
        return users_dao.get_user_chats(user_id, limit=25, offset=offset)
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
async def post_to_chat(  # noqa: C901, PLR0913
    raw_request: Request,
    chat_id: str,
    message: ChatRequest,
    app_state: Annotated[AppState, Depends(get_app_state)],
    user_id: Annotated[str, Depends(get_current_user)],
    users_dao: Annotated[UsersDAO, Depends(users_dao_provider)],
    aois_dao: Annotated[AOIsDAO, Depends(aois_dao_provider)],
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
    aoi = _validate_aoi(message.ui_state.aoi, user_id, aois_dao)
    new_user_msg = chat.insert_chat_request(message, aoi)
    users_dao.update_chat(user_id, chat)

    chat_context = KmnlpChatContext(
        user_id=user_id,
        chat=chat,
        global_context=app_state.context,
        aois_dao=aois_dao,
        base_url=f"{raw_request.url.scheme}://{raw_request.url.netloc}",
    )

    # Create response queue for agent communication
    response_queue: asyncio.Queue[chat_response_content_types] = asyncio.Queue()

    # Emit a response: persist it first (so persistence is independent of client liveness), then
    # forward it to the client stream. Because this runs synchronously within the agent task,
    # `chat`/`users_dao` are always valid here — including during cancellation cleanup after the
    # client disconnects.
    #
    # `TextDelta`s are ephemeral streaming chunks used only for live rendering; they are forwarded
    # to the client but not persisted. The consolidated `Text` (emitted when a text part completes,
    # or flushed from a partial buffer on cancellation) is the durable record.
    def save_and_emit_response(content: chat_response_content_types) -> None:
        match content:
            case TextDelta() | ThinkingDelta() | EphemeralPlan():
                log.debug("Emitting ephemeral content: %s", type(content))
            case _:
                chat.messages.append(ChatResponse(content=content))
                users_dao.update_chat(user_id, chat)
        response_queue.put_nowait(content)

    # Start the agent response generation as a background task
    # Use a set to prevent garbage collection
    background_tasks: set[asyncio.Task[None]] = set()
    agent_task = asyncio.create_task(
        get_agent_response_content_stream(
            app_state.agent,
            new_user_msg=new_user_msg,
            chat=chat,
            chat_context=chat_context,
            users_dao=users_dao,
            user_id=user_id,
            emit_response=save_and_emit_response,
        )
    )
    background_tasks.add(agent_task)
    agent_task.add_done_callback(background_tasks.discard)

    # Watch for client disconnect and cancel the agent task directly.
    # This runs concurrently with Starlette's streaming loop (which is where real awaits happen),
    # so it can observe the disconnect and propagate cancellation to agent_task.
    async def watch_disconnect() -> None:
        while True:
            message = await raw_request.receive()
            if message["type"] == "http.disconnect":
                log.info("Client disconnected, cancelling agent task")
                agent_task.cancel()
                break

    disconnect_watcher = asyncio.create_task(watch_disconnect())
    background_tasks.add(disconnect_watcher)
    disconnect_watcher.add_done_callback(background_tasks.discard)
    agent_task.add_done_callback(lambda _: disconnect_watcher.cancel())

    # Create a generator that yields ChatResponse objects from the response queue
    async def response_content_stream() -> AsyncGenerator[chat_response_content_types]:
        while True:
            content = await response_queue.get()
            yield content
            if isinstance(content, EndOfTurn | BackendError):
                break

    response_stream = (ChatResponse(content=c) async for c in response_content_stream())

    return StreamingResponse(
        serialize_obj_stream(response_stream),
        media_type="application/x-ndjson",
    )


@router.get("/{chat_id}/data/{key}")
async def get_data_download_url(
    chat_id: str,
    key: str,
    app_state: Annotated[AppState, Depends(get_app_state)],
    user_id: Annotated[str, Depends(get_current_user)],
    users_dao: Annotated[UsersDAO, Depends(users_dao_provider)],
) -> DataDownloadURL:
    """Get a download URL for the specified chat data."""
    try:
        _ = users_dao.get_chat(user_id, chat_id)
    except DAOChatNotFoundError:
        raise ChatNotFoundError(chat_id) from None
    except DAOUserNotFoundError:
        raise UserNotFoundError(user_id) from None

    s3_uri = join(
        app_state.context.s3_user_data_root,
        s3.make_data_key_path(user_id, chat_id, key),
    )
    log.info("Generating pre-signed URL for S3 URI: %s", s3_uri)
    if not s3.exists(app_state.context.s3_client, s3_uri):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

    presigned_url = s3.uri_to_presigned_url(
        s3_client=app_state.context.s3_client,
        s3_uri=s3_uri,
        expiry_seconds=PRESIGNED_URL_EXPIRY_SECONDS,
    )
    log.info("Pre-signed URL: %s", presigned_url)
    return DataDownloadURL(url=presigned_url)


def _validate_aoi(aoi_id: UUID | None, user_id: str, aois_dao: AOIsDAO) -> AOI | None:
    if aoi_id is None:
        return None

    aoi = aois_dao.get_aoi(aoi_id, user_id)

    if aoi is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Provided AOI UUID not found."
        )

    return aoi
