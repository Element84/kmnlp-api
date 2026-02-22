import pytest
from e84_geoai_common.llm.core import LLMAssistantMessage, LLMUserMessage

from app.dao.users_dao import deserialize_llm_message


class TestDeserializeLLMMessage:
    """Test the _deserialize_llm_message function."""

    def test_deserialize_user_message(self) -> None:
        """Test deserializing a user message."""
        data = {"role": "user", "content": [{"text": "Hello"}]}

        result = deserialize_llm_message(data)

        assert isinstance(result, LLMUserMessage)
        assert result.role == "user"

    def test_deserialize_assistant_message(self) -> None:
        """Test deserializing an assistant message."""
        data = {"role": "assistant", "content": [{"text": "Hi there"}], "metadata": None}

        result = deserialize_llm_message(data)

        assert isinstance(result, LLMAssistantMessage)
        assert result.role == "assistant"

    def test_deserialize_invalid_role(self) -> None:
        """Test deserializing message with invalid role raises error."""
        data = {"role": "invalid", "content": [{"text": "Hello"}]}

        with pytest.raises(ValueError, match=r".*"):
            deserialize_llm_message(data)

    def test_deserialize_missing_role(self) -> None:
        """Test deserializing message without role raises error."""
        data = {"content": [{"text": "Hello"}]}

        with pytest.raises(ValueError, match=r".*"):
            deserialize_llm_message(data)
