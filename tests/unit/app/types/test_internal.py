from datetime import UTC, datetime

from e84_geoai_common.llm.core import LLMUserMessage, TextContent

from app.types.geojson import GeoJSONGeometry
from app.types.input import ChatRequest, UIState
from app.types.internal import InternalChat


class TestInternalChat:
    def test_initialization(self):
        chat = InternalChat(id="chat", title="Some chat")
        assert chat.id == "chat"
        assert chat.title == "Some chat"
        assert chat.messages == []
        assert chat.agent_conversation == []

    def test_insert_chat_request(self):
        chat = InternalChat(id="chat", title="Some other chat")
        chat_request = ChatRequest(
            content="What do you know about mangrove forests?",
            timestamp=datetime.now(UTC),
            ui_state=UIState(),
        )

        chat.insert_chat_request(chat_request, aoi=None)

        assert chat.messages == [chat_request]

        assert chat.agent_conversation == [
            LLMUserMessage(content=[TextContent(text="What do you know about mangrove forests?")])
        ]

    def test_insert_chat_request_with_aoi(self):
        chat = InternalChat(id="chat", title="Chat with AOI")
        chat_request = ChatRequest(
            content="What is the foliage like in this area?",
            timestamp=datetime.now(UTC),
            ui_state=UIState(aoi="00000000-0000-0000-0000-000000000001"),
        )

        chat.insert_chat_request(chat_request, aoi=GeoJSONGeometry(type="Polygon", coordinates=[]))

        assert chat.messages == [chat_request]
        assert chat.agent_conversation == [
            LLMUserMessage(
                content=[
                    TextContent(text="What is the foliage like in this area?"),
                    TextContent(
                        text="""Provided area: {"type":"Polygon","coordinates":[],"bbox":null}"""
                    ),
                ]
            )
        ]
