from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.dao.users_dao import UserAlreadyExistsError as DAOUserAlreadyExistsError
from app.dao.users_dao import UserNotFoundError as DAOUserNotFoundError
from app.dao.users_dao import UsersDAO
from app.providers import users_dao_provider
from app.types.input import UserCreateRequest
from app.types.output import (
    UserResponse,
)

router = APIRouter(
    prefix="/users",
    tags=["users"],
)


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


@router.post("")
async def create_user(
    request: UserCreateRequest,
    users_dao: Annotated[UsersDAO, Depends(users_dao_provider)],
) -> str:
    """Create new user."""
    try:
        users_dao.create_user(request.user_id)
    except DAOUserAlreadyExistsError:
        raise UserAlreadyExistsError from None
    else:
        return request.user_id


@router.get("/{user_id}")
async def get_user(
    user_id: str,
    users_dao: Annotated[UsersDAO, Depends(users_dao_provider)],
) -> UserResponse:
    """Get a user with their chat IDs."""
    try:
        chat_ids = [chat.id for chat in users_dao.get_user_chats(user_id)]
        return UserResponse(user_id=user_id, chat_ids=chat_ids)
    except DAOUserNotFoundError:
        raise UserNotFoundError(user_id) from None
