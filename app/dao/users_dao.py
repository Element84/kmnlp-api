import json
from datetime import UTC
from typing import Any

from e84_geoai_common.llm.core import LLMAssistantMessage, LLMMessage, LLMUserMessage
from sqlalchemy import Table, desc, insert, select, update

from app.db.sqlalchemy_db import SQLAlchemyDB
from app.types.input import ChatRequest
from app.types.internal import InternalChat, User
from app.types.output import ChatHeader, ChatResponse
from app.utils.plan_manager import PlanManager


class UserAlreadyExistsError(Exception):
    """User already exists in database."""

    def __init__(self, user_id: str) -> None:
        """Initialize error."""
        super().__init__(f"User already exists: {user_id}")


class UserNotFoundError(Exception):
    """User not found in database."""

    def __init__(self, user_id: str) -> None:
        """Initialize error."""
        super().__init__(f"User not found: {user_id}")


class ChatNotFoundError(Exception):
    """Chat not found in database."""

    def __init__(self, chat_id: str) -> None:
        """Initialize error."""
        super().__init__(f"Chat not found: {chat_id}")


class UsersDAO:
    """DAO for users."""

    db: SQLAlchemyDB
    _users_table: Table
    _chats_table: Table

    def __init__(self, db: SQLAlchemyDB) -> None:
        """Initializer."""
        self.db = db
        self._users_table = self.db.users_table
        self._chats_table = self.db.chats_table

    def create_user(self, user_id: str) -> User:
        """Create a new user."""
        if self.user_exists(user_id):
            raise UserAlreadyExistsError(user_id)

        user = User(id=user_id, chats={})

        statement = insert(self._users_table).values(id=user_id, date_created=user.date_created)

        rowcount = self.db.insert_or_update_single_row(statement)

        if rowcount == 1:
            return user

        raise ValueError("Create User did not insert 1 row.")

    def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        statement = (
            select(self._users_table.c.id, self._users_table.c.date_created)
            .where(self._users_table.c.id == user_id)
            .limit(1)
        )

        user_row = self.db.select_one(statement)

        if user_row is None:
            return None

        chats = self.get_user_chats(user_id)

        return User(
            id=user_row.id,
            date_created=user_row.date_created.replace(tzinfo=UTC),
            chats={chat.id: chat for chat in chats},
        )

    def user_exists(self, user_id: str) -> bool:
        """Check if a user exists."""
        statement = select(self._users_table.c.id).where(self._users_table.c.id == user_id).limit(1)

        with self.db.get_connection() as conn:
            result = conn.execute(statement)
            for _ in result:
                return True

        return False

    def create_chat(self, user_id: str, chat: InternalChat) -> InternalChat:
        """Create a new chat for a user."""
        if not self.user_exists(user_id):
            raise UserNotFoundError(user_id)

        messages_json = json.dumps([msg.model_dump(mode="json") for msg in chat.messages])
        agent_conversation_json = json.dumps(
            [msg.model_dump(mode="json") for msg in chat.agent_conversation]
        )

        stmt = insert(self._chats_table).values(
            id=chat.id,
            title=chat.title,
            user_id=user_id,
            messages_json=messages_json,
            agent_conversation_json=agent_conversation_json,
            plan_manager_json=chat.plan_manager.model_dump_json(),
            created_at=chat.date_created,
        )

        self.db.insert_or_update_single_row(stmt)

        return chat

    def get_chat(self, user_id: str, chat_id: str) -> InternalChat:
        """Get a chat by user ID and chat ID."""
        if not self.user_exists(user_id):
            raise UserNotFoundError(user_id)

        return self._get_chat_from_db(user_id, chat_id)

    def get_user_chats(self, user_id: str) -> list[ChatHeader]:
        """Get all chat headers for a user."""
        if not self.user_exists(user_id):
            raise UserNotFoundError(user_id)

        columns = self._chats_table.c
        stmt = (
            select(columns.id, columns.title, columns.created_at)
            .where(columns.user_id == user_id)
            .order_by(desc(columns.created_at))
            .limit(15)
        )

        rows = self.db.select_many(stmt)

        return [
            ChatHeader(id=row.id, title=row.title, date_created=row.created_at.replace(tzinfo=UTC))
            for row in rows
        ]

    def update_chat(self, user_id: str, chat: InternalChat) -> InternalChat:
        """Update an existing chat."""
        if not self.user_exists(user_id):
            raise UserNotFoundError(user_id)

        columns = self._chats_table.c
        stmt = (
            update(self._chats_table)
            .values(
                title=chat.title,
                messages_json=json.dumps([msg.model_dump(mode="json") for msg in chat.messages]),
                agent_conversation_json=json.dumps(
                    [
                        msg.model_dump(mode="json")
                        for msg in chat.agent_conversation  # pyright: ignore[reportPrivateUsage]
                    ]
                ),
                plan_manager_json=chat.plan_manager.model_dump_json(),
            )
            .where(columns.id == chat.id)
            .where(columns.user_id == user_id)
        )

        did_update = self.db.insert_or_update_single_row(stmt)

        if not did_update:
            raise ChatNotFoundError(chat.id)

        return chat

    def _get_chat_from_db(self, user_id: str, chat_id: str) -> InternalChat:
        """Get a chat from the database cursor."""
        stmt = (
            select(self._chats_table)
            .where(self._chats_table.c.id == chat_id)
            .where(self._chats_table.c.user_id == user_id)
            .limit(1)
        )

        result = self.db.select_one(stmt)

        if result is None:
            raise ChatNotFoundError(chat_id)

        messages_data = json.loads(result.messages_json)
        agent_conversation_data = json.loads(result.agent_conversation_json)

        # Reconstruct messages from JSON
        messages: list[ChatRequest | ChatResponse] = []
        for msg_data in messages_data:
            if "content" in msg_data and isinstance(msg_data["content"], str):
                # This is a ChatRequest
                messages.append(ChatRequest.model_validate(msg_data))
            else:
                # This is a ChatResponse
                messages.append(ChatResponse.model_validate(msg_data))

        # Reconstruct agent conversation from JSON
        agent_conversation = [
            deserialize_llm_message(msg_data) for msg_data in agent_conversation_data
        ]

        plan_manager = PlanManager.model_validate_json(result.plan_manager_json)

        return InternalChat(
            id=result.id,
            title=result.title,
            messages=messages,
            agent_conversation=agent_conversation,
            plan_manager=plan_manager,
            date_created=result.created_at.replace(tzinfo=UTC),
        )


def deserialize_llm_message(data: dict[str, Any]) -> LLMMessage:
    """Deserialize a dictionary into an LLMMessage."""
    match data.get("role"):
        case "user":
            return LLMUserMessage.model_validate(data)
        case "assistant":
            return LLMAssistantMessage.model_validate(data)
        case _:
            raise ValueError
