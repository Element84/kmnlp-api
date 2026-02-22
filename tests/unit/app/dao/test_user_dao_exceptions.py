from app.dao.users_dao import ChatNotFoundError, UserAlreadyExistsError, UserNotFoundError


class TestUserDAOExceptions:
    """Test custom exception classes."""

    def test_user_already_exists_error(self) -> None:
        """Test UserAlreadyExistsError message."""
        error = UserAlreadyExistsError("test_user")
        assert str(error) == "User already exists: test_user"

    def test_user_not_found_error(self) -> None:
        """Test UserNotFoundError message."""
        error = UserNotFoundError("test_user")
        assert str(error) == "User not found: test_user"

    def test_chat_not_found_error(self) -> None:
        """Test ChatNotFoundError message."""
        error = ChatNotFoundError("test_chat")
        assert str(error) == "Chat not found: test_chat"
