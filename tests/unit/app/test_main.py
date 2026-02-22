from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dao.aois_dao import AOIsDAO
from app.dependencies import get_app_state
from app.main import app
from app.providers import db_provider
from app.types.geojson import GeoJSONGeometry
from app.types.input import AOI as IAOI
from app.types.input import UserCreateRequest
from app.types.output import AOISummary, Chat, ChatRequest
from tests.utils.dependencies import override_app_state
from tests.utils.fixtures.aoi_fixture import AOIFixture
from tests.utils.providers import feature_test_db_provider


class TestAPI:
    @classmethod
    def setup_class(cls) -> None:
        app.dependency_overrides[get_app_state] = override_app_state
        app.dependency_overrides[db_provider] = feature_test_db_provider
        cls.client = TestClient(app)

    @pytest.fixture(autouse=True)
    def clear_caches(self):
        """Clear provider caches after each test."""
        # Run test
        yield

        # Clear caches
        feature_test_db_provider.cache_clear()

    def test_create_user(self):
        user_data = UserCreateRequest(user_id="xyz")
        create_user_response = self.client.post("/users", json=user_data.model_dump())
        assert create_user_response.status_code == 200

        create_user_response2 = self.client.post("/users", json=user_data.model_dump())
        assert create_user_response2.status_code == 409
        assert create_user_response2.json() == {"detail": "User already exists"}

    def test_create_chat(self):
        headers = {"x-user-id": "test"}
        user_data = UserCreateRequest(user_id="test")
        create_user_response = self.client.post("/users", json=user_data.model_dump())
        assert create_user_response.status_code in (200, 409)

        create_chat_response = self.client.post("/chats", headers=headers)
        assert create_chat_response.status_code == 200
        chat = Chat.model_validate(create_chat_response.json())
        assert len(chat.messages) == 0

    def test_create_chat_with_bad_user_id(self):
        headers = {"x-user-id": "test123"}
        response = self.client.post("/chats", headers=headers)
        assert response.status_code == 401
        assert response.json() == {"detail": "User not found: test123"}

    def test_chat_with_bad_chat_id(self):
        headers = {"x-user-id": "test"}
        user_data = UserCreateRequest(user_id="test")
        _ = self.client.post("/users", json=user_data.model_dump())
        user_msg = ChatRequest(content="test")
        response = self.client.post(
            "/chats/123", json=user_msg.model_dump(mode="json"), headers=headers
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "Chat not found: 123"}

    def test_create_aoi(self):
        # Create User
        headers = {"x-user-id": "test"}
        user_data = UserCreateRequest(user_id="test")
        _ = self.client.post("/users", json=user_data.model_dump())

        # Create AOI
        create_aoi_request = IAOI(
            name="Test AOI", geometry=GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None)
        )
        response = self.client.post(
            "/aois", json=create_aoi_request.model_dump(mode="json"), headers=headers
        )

        assert response.status_code == 201
        assert isinstance(response.json(), dict)
        assert response.json().keys() == {"uuid"}

    def test_get_aoi(self):
        # Create User
        headers = {"x-user-id": "test"}
        user_data = UserCreateRequest(user_id="test")
        _ = self.client.post("/users", json=user_data.model_dump())

        # Create AOI through DAO
        aois_dao = AOIsDAO(feature_test_db_provider())
        created_aoi = aois_dao.create_aoi(
            "test", "AOI", GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None)
        )

        # Get AOI request
        response = self.client.get(f"/aois/{created_aoi.uuid}", headers=headers)

        # Validate AOI is as expected
        assert response.status_code == 200

        response_dict: dict[str, Any] = response.json()
        assert response_dict["uuid"] == str(created_aoi.uuid)
        assert response_dict["name"] == "AOI"
        assert response_dict["geojson"] == {
            "bbox": None,
            "coordinates": [
                0.0,
                0.0,
            ],
            "type": "Point",
        }

        # NOTE: Ideally, we'd compare the strings here but the ISO formatters
        # differ: One uses 'Z' while the other uses +00:00.
        assert datetime.fromisoformat(response_dict["created_at"]) == created_aoi.created_at

    def test_get_aoi_with_wrong_user(self):
        # Create Users
        user_data = UserCreateRequest(user_id="test")
        _ = self.client.post("/users", json=user_data.model_dump())

        bad_headers = {"x-user-id": "bad"}
        user_data = UserCreateRequest(user_id="bad")
        _ = self.client.post("/users", json=user_data.model_dump())

        # Create AOI through DAO
        aois_dao = AOIsDAO(feature_test_db_provider())
        created_aoi = aois_dao.create_aoi(
            "test", "AOI", GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None)
        )

        # Get AOI request
        response = self.client.get(f"/aois/{created_aoi.uuid}", headers=bad_headers)

        assert response.status_code == 404
        assert response.json() == {"detail": "AOI not found"}

    def test_list_aois_with_pagination(self):
        # Create User
        headers = {"x-user-id": "test"}
        user_data = UserCreateRequest(user_id="test")
        _ = self.client.post("/users", json=user_data.model_dump())

        # Create 16 AOI with differing `created_at`s
        fixture = AOIFixture(feature_test_db_provider())
        for i in range(16):
            created_at = datetime(2026, 1, i + 1, tzinfo=UTC)
            fixture.create_aoi(user_id="test", created_at=created_at)

        # Fetch AOIs without cursor
        first_page_response = self.client.get("/aois", headers=headers)

        # Assert expected values
        assert first_page_response.status_code == 200
        first_page_response_json = first_page_response.json()
        assert len(first_page_response_json["aois"]) == 15
        assert first_page_response_json["pagination_info"]["total_count"] == 16

        next_cursor = first_page_response_json["pagination_info"]["next_cursor"]

        # Fetch AOIs with cursor from first page
        second_page_response = self.client.get(
            "/aois", params={"after_cursor": next_cursor}, headers=headers
        )

        # Assert expected values
        assert second_page_response.status_code == 200
        second_page_response_json = second_page_response.json()
        assert len(second_page_response_json["aois"]) == 1
        assert second_page_response_json["pagination_info"]["total_count"] == 16
        assert second_page_response_json["pagination_info"]["next_cursor"] is None

        # Assert can parse returned AOI
        parsed = AOISummary.model_validate(second_page_response_json["aois"][0])
        assert parsed.created_at == datetime(2026, 1, 16, tzinfo=UTC)
