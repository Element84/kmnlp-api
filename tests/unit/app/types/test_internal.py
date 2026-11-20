import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.types.geojson import GeoJSONGeometry
from app.types.input import ChatRequest, UIState
from app.types.internal import AOI, InternalChat, serialize_agent_conversation


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
        assert len(chat.agent_conversation) == 1

        msg_parts_dict: list[dict[str, Any]] = json.loads(
            serialize_agent_conversation(chat.agent_conversation)
        )[0]["parts"]
        for part_dict in msg_parts_dict:
            part_dict.pop("timestamp", None)  # Remove timestamp for comparison

        assert msg_parts_dict == [
            {
                "part_kind": "user-prompt",
                "content": json.dumps(
                    {
                        "user_message": "What do you know about mangrove forests?",
                        "attached_data": [],
                    },
                    separators=(",", ":"),
                ),
            }
        ]

    def test_insert_chat_request_with_aoi(self):
        chat = InternalChat(id="chat", title="Chat with AOI")
        aoi_uuid = uuid4()
        chat_request = ChatRequest(
            content="What is the foliage like in this area?",
            timestamp=datetime.now(UTC),
            ui_state=UIState(aoi=aoi_uuid),
        )

        chat.insert_chat_request(
            chat_request,
            aoi=AOI(
                uuid=aoi_uuid,
                name="Test AOI",
                user_id="test-user",
                geojson=GeoJSONGeometry(type="Polygon", coordinates=[]),
            ),
        )

        assert chat.messages == [chat_request]
        assert len(chat.agent_conversation) == 1

        msg_parts_dict: list[dict[str, Any]] = json.loads(
            serialize_agent_conversation(chat.agent_conversation)
        )[0]["parts"]
        for part_dict in msg_parts_dict:
            part_dict.pop("timestamp", None)  # Remove timestamp for comparison

        assert msg_parts_dict == [
            {
                "part_kind": "user-prompt",
                "content": json.dumps(
                    {
                        "user_message": "What is the foliage like in this area?",
                        "attached_data": [
                            {
                                "data_ref_type": "geometry",
                                "key": str(aoi_uuid),
                                "name": "Test AOI",
                                "description": "Area of interest",
                            }
                        ],
                    },
                    separators=(",", ":"),
                ),
            }
        ]
