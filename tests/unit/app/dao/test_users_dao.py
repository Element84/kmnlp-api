import os
import tempfile
from datetime import UTC, datetime

import pytest
from llm_agent.tools.agent_plan_tool import AgentPlan, AgentStep
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from app.dao.users_dao import ChatNotFoundError, UserAlreadyExistsError, UserNotFoundError, UsersDAO
from app.db.sqlalchemy_db import SQLAlchemyDB
from app.types.input import ChatRequest
from app.types.internal import InternalChat, User
from app.types.output import ChatHeader, ChatResponse, Text
from app.utils.plan_manager import PlanManager


class TestUsersDAO:
    def test_create_user(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test creating a new user."""
        dao = UsersDAO(sqlalchemy_db)

        user = dao.create_user("test_user")

        assert isinstance(user, User)
        assert user.id == "test_user"
        assert isinstance(user.date_created, datetime)
        assert user.chats == {}

    def test_create_user_already_exists(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test creating a user that already exists raises error."""
        dao = UsersDAO(sqlalchemy_db)

        dao.create_user("test_user")

        with pytest.raises(UserAlreadyExistsError) as exc_info:
            dao.create_user("test_user")

        assert "User already exists: test_user" in str(exc_info.value)

    def test_get_user(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting an existing user."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        created_user = dao.create_user("test_user")

        # Get user
        retrieved_user = dao.get_user("test_user")

        assert retrieved_user is not None

        assert retrieved_user.id == "test_user"
        assert retrieved_user.date_created == created_user.date_created
        assert retrieved_user.chats == {}

    def test_get_user_not_found(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting a non-existent user returns None."""
        dao = UsersDAO(sqlalchemy_db)

        result = dao.get_user("nonexistent")

        assert result is None

    def test_get_user_with_chats(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting a user that has chats."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        dao.create_user("test_user")

        # Create chat
        chat = InternalChat(id="chat1", title="Test Chat")
        dao.create_chat("test_user", chat)

        # Get user
        user = dao.get_user("test_user")

        assert user is not None

        assert len(user.chats) == 1
        assert "chat1" in user.chats
        assert user.chats["chat1"].id == "chat1"

    def test_user_exists(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test checking if user exists."""
        dao = UsersDAO(sqlalchemy_db)

        assert not dao.user_exists("test_user")

        dao.create_user("test_user")

        assert dao.user_exists("test_user")

    def test_create_chat(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test creating a new chat."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user first
        dao.create_user("test_user")

        # Create chat
        chat = InternalChat(id="chat1", title="Test Chat")
        result = dao.create_chat("test_user", chat)

        assert result.id == "chat1"
        assert result.messages == []

    def test_create_chat_user_not_found(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test creating chat for non-existent user raises error."""
        dao = UsersDAO(sqlalchemy_db)

        chat = InternalChat(id="chat1", title="Test Chat")

        with pytest.raises(UserNotFoundError) as exc_info:
            dao.create_chat("nonexistent", chat)

        assert "User not found: nonexistent" in str(exc_info.value)

    def test_create_chat_with_messages(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test creating chat with existing messages."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user first
        dao.create_user("test_user")

        # Create chat with messages
        request = ChatRequest(content="Hello")
        response = ChatResponse(content=Text(text="Hi there"))
        chat = InternalChat(id="chat1", title="Test Chat", messages=[request, response])

        # Add some agent conversation
        user_msg = ModelRequest(parts=[UserPromptPart(content="Hello")])
        assistant_msg = ModelResponse(parts=[TextPart(content="Hi there")])
        chat.agent_conversation = [user_msg, assistant_msg]  # pyright: ignore[reportPrivateUsage]

        result = dao.create_chat("test_user", chat)

        assert len(result.messages) == 2
        assert isinstance(result.messages[0], ChatRequest)
        assert isinstance(result.messages[1], ChatResponse)

    def test_get_chat(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting an existing chat."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user and chat
        dao.create_user("test_user")
        original_chat = InternalChat(id="chat1", title="Test Chat")
        dao.create_chat("test_user", original_chat)

        # Get chat
        retrieved_chat = dao.get_chat("test_user", "chat1")

        assert retrieved_chat.id == "chat1"
        assert retrieved_chat.messages == []

    def test_get_chat_user_not_found(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting chat for non-existent user raises error."""
        dao = UsersDAO(sqlalchemy_db)

        with pytest.raises(UserNotFoundError) as exc_info:
            dao.get_chat("nonexistent", "chat1")

        assert "User not found: nonexistent" in str(exc_info.value)

    def test_get_chat_not_found(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting non-existent chat raises error."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user but no chat
        dao.create_user("test_user")

        with pytest.raises(ChatNotFoundError) as exc_info:
            dao.get_chat("test_user", "nonexistent")

        assert "Chat not found: nonexistent" in str(exc_info.value)

    def test_get_chat_with_complex_messages(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting chat with complex message types."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        dao.create_user("test_user")

        # Create chat with various message types
        request = ChatRequest(content="Hello")
        response = ChatResponse(content=Text(text="Hi there"))
        chat = InternalChat(id="chat1", title="Test Chat", messages=[request, response])

        # Add agent conversation
        user_msg = ModelRequest(parts=[UserPromptPart(content="Hello")])
        assistant_msg = ModelResponse(parts=[TextPart(content="Hi there")])
        chat.agent_conversation = [user_msg, assistant_msg]  # pyright: ignore[reportPrivateUsage]

        dao.create_chat("test_user", chat)

        # Get chat
        retrieved_chat = dao.get_chat("test_user", "chat1")

        assert len(retrieved_chat.messages) == 2
        assert isinstance(retrieved_chat.messages[0], ChatRequest)
        assert retrieved_chat.messages[0].content == "Hello"
        assert isinstance(retrieved_chat.messages[1], ChatResponse)
        assert isinstance(retrieved_chat.messages[1].content, Text)
        assert retrieved_chat.messages[1].content.text == "Hi there"

        # Check agent conversation
        assert len(retrieved_chat.agent_conversation) == 2  # pyright: ignore[reportPrivateUsage]
        assert isinstance(retrieved_chat.agent_conversation[0], ModelRequest)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(retrieved_chat.agent_conversation[1], ModelResponse)  # pyright: ignore[reportPrivateUsage]

    def test_get_user_chats(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting all chat IDs for a user."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        dao.create_user("test_user")

        # Initially no chats
        assert dao.get_user_chats("test_user") == []

        # Create some chats
        chats = [InternalChat(id=f"chat{i}", title=f"Chat {i}") for i in range(1, 4)]
        for chat in chats:
            dao.create_chat("test_user", chat)

        timestamps = [chat.date_created for chat in chats]
        timestamps_expected = [t.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for t in timestamps]

        # Get chat headers
        chat_headers = dao.get_user_chats("test_user")
        assert TypeAdapter(list[ChatHeader]).dump_python(chat_headers, mode="json") == [
            {"id": "chat3", "title": "Chat 3", "date_created": timestamps_expected[2]},
            {"id": "chat2", "title": "Chat 2", "date_created": timestamps_expected[1]},
            {"id": "chat1", "title": "Chat 1", "date_created": timestamps_expected[0]},
        ]

    def test_get_user_chats_user_not_found(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting chats for non-existent user raises error."""
        dao = UsersDAO(sqlalchemy_db)

        with pytest.raises(UserNotFoundError) as exc_info:
            dao.get_user_chats("nonexistent")

        assert "User not found: nonexistent" in str(exc_info.value)

    def test_update_chat(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test updating an existing chat."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user and chat
        dao.create_user("test_user")
        original_chat = InternalChat(id="chat1", title="Original Chat")
        dao.create_chat("test_user", original_chat)

        # Update chat with new messages
        request = ChatRequest(content="Updated message")
        response = ChatResponse(content=Text(text="Updated response"))
        updated_chat = InternalChat(id="chat1", title="Updated Chat", messages=[request, response])

        # Add agent conversation
        user_msg = ModelRequest(parts=[UserPromptPart(content="Updated message")])
        updated_chat.agent_conversation = [user_msg]  # pyright: ignore[reportPrivateUsage]

        result = dao.update_chat("test_user", updated_chat)

        assert result.id == "chat1"
        assert len(result.messages) == 2

        # Verify update persisted
        retrieved_chat = dao.get_chat("test_user", "chat1")
        assert len(retrieved_chat.messages) == 2
        assert retrieved_chat.messages[0].content == "Updated message"

        # Verify agent conversation was also updated
        assert len(retrieved_chat.agent_conversation) == 1  # pyright: ignore[reportPrivateUsage]
        assert isinstance(retrieved_chat.agent_conversation[0], ModelRequest)  # pyright: ignore[reportPrivateUsage]
        agent_content = retrieved_chat.agent_conversation[0].parts[0]  # pyright: ignore[reportPrivateUsage]
        assert isinstance(agent_content, UserPromptPart)
        assert agent_content.content == "Updated message"

    def test_update_chat_user_not_found(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test updating chat for non-existent user raises error."""
        dao = UsersDAO(sqlalchemy_db)

        chat = InternalChat(id="chat1", title="Test Chat")

        with pytest.raises(UserNotFoundError) as exc_info:
            dao.update_chat("nonexistent", chat)

        assert "User not found: nonexistent" in str(exc_info.value)

    def test_update_chat_not_found(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test updating non-existent chat raises error."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user but no chat
        dao.create_user("test_user")

        chat = InternalChat(id="nonexistent", title="Test Chat")

        with pytest.raises(ChatNotFoundError) as exc_info:
            dao.update_chat("test_user", chat)

        assert "Chat not found: nonexistent" in str(exc_info.value)

    def test_chat_title_is_required_field(self) -> None:
        """Test that InternalChat requires title field."""
        # Title is now required, so we must provide it
        chat = InternalChat(id="test_chat", title="My Chat")
        assert chat.title == "My Chat"

    def test_chat_title_set_from_first_message(self) -> None:
        """Test that chat title gets updated from first message via insert_chat_request."""
        chat = InternalChat(id="test_chat", title="Initial Title")

        # Initially title is what we set
        assert chat.title == "Initial Title"

        # Add first message - title should be updated
        first_message = ChatRequest(
            content="What is the weather like in New York today and what should I wear?"
        )
        chat.insert_chat_request(first_message, None)

        # Title should be truncated to 50 characters with ellipsis
        expected_title = "What is the weather like in New York today and..."
        assert chat.title == expected_title

        # Adding another message should not change the title
        second_message = ChatRequest(content="Also, what about tomorrow?")
        chat.insert_chat_request(second_message, None)

        # Title should remain the same
        assert chat.title == expected_title

    def test_chat_title_with_short_message(self) -> None:
        """Test that chat title is set correctly for short messages."""
        chat = InternalChat(id="test_chat", title="Original")

        # Add a short message that won't need truncation
        short_message = ChatRequest(content="Hello")
        chat.insert_chat_request(short_message, None)

        # Title should be the full message content
        assert chat.title == "Hello"

    def test_chat_title_with_whitespace(self) -> None:
        """Test that chat title handles whitespace correctly."""
        chat = InternalChat(id="test_chat", title="Original")

        # Add message with surrounding whitespace
        message_with_whitespace = ChatRequest(content="  What is AI?   ")
        chat.insert_chat_request(message_with_whitespace, None)

        # Title should have whitespace stripped
        assert chat.title == "What is AI?"

    def test_chat_date_created_is_set_automatically(self) -> None:
        """Test that InternalChat date_created field gets set automatically."""
        before_creation = datetime.now(tz=UTC)
        chat = InternalChat(id="test_chat", title="Test Chat")
        after_creation = datetime.now(tz=UTC)

        # date_created should be set automatically and be between before and after
        assert before_creation <= chat.date_created <= after_creation
        assert chat.date_created.tzinfo == UTC

    def test_chat_date_created_can_be_set_explicitly(self) -> None:
        """Test that InternalChat date_created can be set explicitly."""
        specific_date = datetime(2023, 1, 15, 10, 30, 0, tzinfo=UTC)
        chat = InternalChat(id="test_chat", title="Test Chat", date_created=specific_date)

        assert chat.date_created == specific_date

    def test_chat_date_created_persistence_in_database(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test that date_created is correctly saved and retrieved from database."""
        dao = UsersDAO(sqlalchemy_db)
        dao.create_user("test_user")

        # Create chat with specific date
        specific_date = datetime(2023, 5, 20, 14, 45, 30, tzinfo=UTC)
        original_chat = InternalChat(id="chat1", title="Date Test", date_created=specific_date)

        # Save to database
        dao.create_chat("test_user", original_chat)

        # Retrieve from database
        retrieved_chat = dao.get_chat("test_user", "chat1")

        # Should preserve the exact date_created
        assert retrieved_chat.date_created == specific_date

    def test_chat_title_persistence_in_database(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test that title is correctly saved and retrieved from database operations."""
        dao = UsersDAO(sqlalchemy_db)
        dao.create_user("test_user")

        # Test create_chat preserves title
        original_chat = InternalChat(id="chat1", title="Original Title")
        dao.create_chat("test_user", original_chat)

        retrieved_chat = dao.get_chat("test_user", "chat1")
        assert retrieved_chat.title == "Original Title"

        # Test update_chat preserves title changes
        updated_chat = InternalChat(id="chat1", title="Updated Title")
        dao.update_chat("test_user", updated_chat)

        retrieved_updated_chat = dao.get_chat("test_user", "chat1")
        assert retrieved_updated_chat.title == "Updated Title"

    def test_chat_to_external_includes_new_fields(self) -> None:
        """Test that InternalChat.to_external() includes title and date_created."""
        specific_date = datetime(2023, 3, 10, 9, 15, 0, tzinfo=UTC)
        chat = InternalChat(
            id="test_chat",
            title="Test Title",
            date_created=specific_date,
            messages=[ChatRequest(content="Hello")],
        )

        external_chat = chat.to_external()

        assert external_chat.id == "test_chat"
        assert external_chat.title == "Test Title"
        assert external_chat.date_created == specific_date
        assert len(external_chat.messages) == 1

    def test_create_chat_with_plan_manager(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test creating chat with custom plan_manager data."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user first
        dao.create_user("test_user")

        # Create a plan and add it to plan manager
        agent_step = AgentStep(
            id="step1", title="Test Step", status="pending", outcome_summary="Not started yet"
        )
        agent_plan = AgentPlan(
            content_type="plan", id="plan1", title="Test Plan", goal="Test goal", steps=[agent_step]
        )

        plan_manager = PlanManager()
        plan_manager.add_plan(agent_plan)
        plan_manager.register_tool_call("tool1", "plan1", "step1")

        # Create chat with custom plan manager
        chat = InternalChat(id="chat1", title="Plan Manager Chat", plan_manager=plan_manager)
        result = dao.create_chat("test_user", chat)

        assert result.id == "chat1"
        assert result.plan_manager is not None
        assert "plan1" in result.plan_manager.plans
        assert "tool1" in result.plan_manager.tool_call_id_to_plan_step_id

    def test_get_chat_with_plan_manager(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test getting chat preserves plan_manager data."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        dao.create_user("test_user")

        # Create plan manager with data
        agent_step = AgentStep(
            id="step1", title="Test Step", status="in_progress", outcome_summary="Working on it"
        )
        agent_plan = AgentPlan(
            content_type="plan", id="plan1", title="Test Plan", goal="Test goal", steps=[agent_step]
        )

        plan_manager = PlanManager()
        plan_manager.add_plan(agent_plan)
        plan_manager.register_tool_call("tool1", "plan1", "step1")

        # Create and save chat
        original_chat = InternalChat(
            id="chat1", title="Original Plan Chat", plan_manager=plan_manager
        )
        dao.create_chat("test_user", original_chat)

        # Retrieve chat
        retrieved_chat = dao.get_chat("test_user", "chat1")

        assert retrieved_chat.id == "chat1"
        assert retrieved_chat.plan_manager is not None
        assert "plan1" in retrieved_chat.plan_manager.plans
        assert "tool1" in retrieved_chat.plan_manager.tool_call_id_to_plan_step_id

        # Verify plan data is preserved
        plan = retrieved_chat.plan_manager.get_plan("plan1")
        assert plan.id == "plan1"
        assert plan.title == "Test Plan"
        assert len(plan.steps) == 1
        assert plan.steps[0].id == "step1"
        assert plan.steps[0].status == "in_progress"

    def test_update_chat_with_plan_manager_changes(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test updating chat with plan_manager changes."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user and initial chat
        dao.create_user("test_user")
        original_chat = InternalChat(id="chat1", title="Original Chat")
        dao.create_chat("test_user", original_chat)

        # Create updated plan manager with new data
        agent_step = AgentStep(
            id="step1",
            title="Updated Step",
            status="success",
            outcome_summary="Completed successfully",
        )
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan1",
            title="Updated Plan",
            goal="Updated goal",
            steps=[agent_step],
        )

        updated_plan_manager = PlanManager()
        updated_plan_manager.add_plan(agent_plan)
        updated_plan_manager.register_tool_call("tool1", "plan1", "step1")

        # Update chat with new plan manager
        updated_chat = InternalChat(
            id="chat1", title="Updated Chat", plan_manager=updated_plan_manager
        )
        result = dao.update_chat("test_user", updated_chat)

        assert result.id == "chat1"
        assert result.plan_manager is not None

        # Verify update persisted
        retrieved_chat = dao.get_chat("test_user", "chat1")
        assert "plan1" in retrieved_chat.plan_manager.plans

        plan = retrieved_chat.plan_manager.get_plan("plan1")
        assert plan.title == "Updated Plan"
        assert plan.steps[0].status == "success"
        assert plan.steps[0].outcome_summary == "Completed successfully"

    def test_empty_plan_manager_serialization(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test handling of empty plan_manager."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        dao.create_user("test_user")

        # Create chat with default (empty) plan manager
        chat = InternalChat(id="chat1", title="Empty Plan Chat")  # Uses default_factory=PlanManager
        dao.create_chat("test_user", chat)

        # Retrieve and verify empty plan manager
        retrieved_chat = dao.get_chat("test_user", "chat1")
        assert retrieved_chat.plan_manager is not None
        assert retrieved_chat.plan_manager.plans == {}
        assert retrieved_chat.plan_manager.tool_call_id_to_plan_step_id == {}

    def test_plan_manager_with_multiple_plans(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test plan_manager with multiple plans and tool calls."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        dao.create_user("test_user")

        # Create multiple plans
        step1 = AgentStep(id="step1", title="Step 1", status="pending")
        step2 = AgentStep(id="step2", title="Step 2", status="in_progress")

        plan1 = AgentPlan(
            content_type="plan", id="plan1", title="Plan 1", goal="Goal 1", steps=[step1]
        )
        plan2 = AgentPlan(
            content_type="plan", id="plan2", title="Plan 2", goal="Goal 2", steps=[step2]
        )

        plan_manager = PlanManager()
        plan_manager.add_plan(plan1)
        plan_manager.add_plan(plan2)
        plan_manager.register_tool_call("tool1", "plan1", "step1")
        plan_manager.register_tool_call("tool2", "plan2", "step2")

        # Create and save chat
        chat = InternalChat(id="chat1", title="Multiple Plans Chat", plan_manager=plan_manager)
        dao.create_chat("test_user", chat)

        # Retrieve and verify
        retrieved_chat = dao.get_chat("test_user", "chat1")
        assert len(retrieved_chat.plan_manager.plans) == 2
        assert len(retrieved_chat.plan_manager.tool_call_id_to_plan_step_id) == 2

        # Verify individual plans
        assert retrieved_chat.plan_manager.get_plan("plan1").title == "Plan 1"
        assert retrieved_chat.plan_manager.get_plan("plan2").title == "Plan 2"

        # Verify tool call mappings
        plan_id, step_id = retrieved_chat.plan_manager.tool_call_id_to_plan_step_ids("tool1")
        assert plan_id == "plan1"
        assert step_id == "step1"


class TestSQLiteUserDatabaseEdgeCases:
    """Test edge cases and error conditions."""

    def test_special_characters_in_ids(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test handling of special characters in user and chat IDs."""
        dao = UsersDAO(sqlalchemy_db)

        # Use various special characters
        special_user_id = "user-with_special@chars.123"
        special_chat_id = "chat-with_special@chars.123"

        dao.create_user(special_user_id)
        dao.create_chat(special_user_id, InternalChat(id=special_chat_id, title="Special Chars"))

        # Verify we can retrieve them
        assert dao.user_exists(special_user_id)
        retrieved_chat = dao.get_chat(special_user_id, special_chat_id)
        assert retrieved_chat.id == special_chat_id

    def test_very_long_ids(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test handling of very long user and chat IDs."""
        dao = UsersDAO(sqlalchemy_db)

        # Create very long IDs (but not so long the are longer than the max column size)
        long_user_id = "u" + "x" * 200
        long_chat_id = "c" + "x" * 200

        dao.create_user(long_user_id)
        dao.create_chat(long_user_id, InternalChat(id=long_chat_id, title="Long ID Chat"))

        # Verify we can retrieve them
        assert dao.user_exists(long_user_id)
        retrieved_chat = dao.get_chat(long_user_id, long_chat_id)
        assert retrieved_chat.id == long_chat_id

    def test_empty_string_ids(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test handling of empty string IDs."""
        dao = UsersDAO(sqlalchemy_db)

        # Empty user ID
        dao.create_user("")
        assert dao.user_exists("")

        # Empty chat ID
        dao.create_chat("", InternalChat(id="", title="Empty ID Chat"))
        retrieved_chat = dao.get_chat("", "")
        assert retrieved_chat.id == ""

    def test_null_like_ids(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test handling of null-like string IDs."""
        dao = UsersDAO(sqlalchemy_db)

        null_like_ids = ["null", "NULL", "None", "undefined"]

        for user_id in null_like_ids:
            dao.create_user(user_id)
            assert dao.user_exists(user_id)

            chat_id = f"chat_{user_id}"
            dao.create_chat(user_id, InternalChat(id=chat_id, title=f"Chat {user_id}"))
            retrieved_chat = dao.get_chat(user_id, chat_id)
            assert retrieved_chat.id == chat_id

    def test_create_multiple_chats_same_user(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test creating multiple chats for the same user."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        dao.create_user("test_user")

        # Create many chats
        chat_ids = [f"chat_{i}" for i in range(100)]
        for chat_id in chat_ids:
            dao.create_chat("test_user", InternalChat(id=chat_id, title=f"Chat {chat_id}"))

        # Verify only 15 returned
        user_chats = dao.get_user_chats("test_user")

        # NOTE: In principle, we could also verify the ids are right, but we
        # sort on the timestamp, which only has millisecond granularity. A
        # better test would use a fixture to set distinct times.
        assert len(user_chats) == 25

    def test_update_chat_preserves_id(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test that updating a chat preserves its ID."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user and chat
        dao.create_user("test_user")
        original_chat = InternalChat(id="original_id", title="Original Chat")
        dao.create_chat("test_user", original_chat)

        # Update chat but try to change ID
        updated_chat = InternalChat(
            id="original_id", title="Updated Chat", messages=[ChatRequest(content="New message")]
        )
        result = dao.update_chat("test_user", updated_chat)

        # ID should remain the same
        assert result.id == "original_id"

        # Verify in database
        retrieved_chat = dao.get_chat("test_user", "original_id")
        assert retrieved_chat.id == "original_id"
        assert len(retrieved_chat.messages) == 1

    def test_migration_adds_title_column(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test that migration correctly adds the title column."""
        # This test verifies the migration logic works by creating a chat and retrieving it
        dao = UsersDAO(sqlalchemy_db)

        # Create user and chat - this will trigger migrations if needed
        dao.create_user("test_user")
        chat = InternalChat(id="chat1", title="Migration Test")
        dao.create_chat("test_user", chat)

        # Verify we can retrieve the chat with title
        retrieved_chat = dao.get_chat("test_user", "chat1")
        assert retrieved_chat.title == "Migration Test"

    def test_large_message_content(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test handling of large message content."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        dao.create_user("test_user")

        # Create chat with large content
        large_content = "x" * 10000  # 10KB of text
        request = ChatRequest(content=large_content)
        response = ChatResponse(content=Text(text=large_content))
        chat = InternalChat(id="chat1", title="Large Content Chat", messages=[request, response])

        dao.create_chat("test_user", chat)

        # Verify we can retrieve the large content
        retrieved_chat = dao.get_chat("test_user", "chat1")
        assert len(retrieved_chat.messages) == 2
        assert retrieved_chat.messages[0].content == large_content
        assert isinstance(retrieved_chat.messages[1].content, Text)
        assert retrieved_chat.messages[1].content.text == large_content

    def test_unicode_content(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test handling of Unicode content."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user
        dao.create_user("test_user")

        # Create chat with Unicode content
        unicode_content = "Hello 世界! 🌍 Café naïve résumé"
        request = ChatRequest(content=unicode_content)
        response = ChatResponse(content=Text(text=unicode_content))
        chat = InternalChat(id="chat1", title="Unicode Chat", messages=[request, response])

        dao.create_chat("test_user", chat)

        # Verify Unicode content is preserved
        retrieved_chat = dao.get_chat("test_user", "chat1")
        assert retrieved_chat.messages[0].content == unicode_content
        assert isinstance(retrieved_chat.messages[1].content, Text)
        assert retrieved_chat.messages[1].content.text == unicode_content

    def test_empty_chat_messages(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test chat with empty messages list."""
        dao = UsersDAO(sqlalchemy_db)

        # Create user and empty chat
        dao.create_user("test_user")
        chat = InternalChat(id="chat1", title="Empty Chat", messages=[])

        dao.create_chat("test_user", chat)

        # Verify empty chat is handled correctly
        retrieved_chat = dao.get_chat("test_user", "chat1")
        assert retrieved_chat.messages == []

    def test_database_persistence_file(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test that file database persists data across instances."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create first instance and add data
            db1 = UsersDAO(sqlalchemy_db)
            db1.create_user("persistent_user")
            db1.create_chat(
                "persistent_user", InternalChat(id="persistent_chat", title="Persistent Chat")
            )

            # Create second instance and verify data exists
            db2 = UsersDAO(sqlalchemy_db)
            assert db2.user_exists("persistent_user")
            assert "persistent_chat" in [chat.id for chat in db2.get_user_chats("persistent_user")]

            retrieved_chat = db2.get_chat("persistent_user", "persistent_chat")
            assert retrieved_chat.id == "persistent_chat"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_database_init_multiple_times(self, sqlalchemy_db: SQLAlchemyDB) -> None:
        """Test that database can be initialized multiple times safely."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Initialize database multiple times
            db1 = UsersDAO(sqlalchemy_db)
            db2 = UsersDAO(sqlalchemy_db)
            db3 = UsersDAO(sqlalchemy_db)

            # All should work without errors
            db1.create_user("user1")
            assert db2.user_exists("user1")
            assert db3.user_exists("user1")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
