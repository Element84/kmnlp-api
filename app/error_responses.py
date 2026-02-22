from fastapi import HTTPException


class UserAlreadyExistsError(HTTPException):
    """Error response indicating user already exists."""

    def __init__(self) -> None:
        """Initializer."""
        super().__init__(status_code=409, detail="User already exists")


class UserNotFoundError(HTTPException):
    """Error response indicating user not found."""

    def __init__(self, user_id: str) -> None:
        """Initializer."""
        self.user_id = user_id
        super().__init__(status_code=401, detail=f"User not found: {self.user_id}")


class MissingUserIdError(HTTPException):
    """Error response for missing user id."""

    def __init__(self) -> None:
        """Initializer."""
        super().__init__(status_code=401, detail="Missing user id")


class ChatNotFoundError(HTTPException):
    """Error response indicating chat not found."""

    def __init__(self, chat_id: str) -> None:
        """Initializer."""
        self.chat_id = chat_id
        super().__init__(status_code=401, detail=f"Chat not found: {self.chat_id}")
