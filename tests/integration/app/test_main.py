import logging
import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from rich import print as rprint
from sqlalchemy import create_engine
from testcontainers.postgres import PostgresContainer  # type: ignore[reportMissingTypeStubs]

from app.main import S3_USER_DATA_BUCKET, app
from app.providers import sql_alchemy_engine_provider
from app.types.geojson import GeoJSONGeometry
from app.types.input import AOI, UIState, UserCreateRequest
from app.types.output import (
    Chat,
    ChatRequest,
    ChatResponse,
    DataAnalysisDetails,
    DataDownloadURL,
    DatasetSearchDetails,
    EndOfTurn,
    FilteredByAOI,
    GeoTIFF,
    Image,
    Plan,
    Text,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@pytest.mark.integration
class TestAPI:
    @pytest.fixture
    def client(self, dask_scheduler_address: str) -> Iterator[TestClient]:
        with (
            patch.dict(os.environ, {"DASK_SCHEDULER_ADDRESS": dask_scheduler_address}),
            TestClient(app, base_url="http://testserver") as client,
            PostgresContainer("postgres:17") as postgres,
        ):
            psql_url = postgres.get_connection_url()
            engine = create_engine(psql_url)
            app.dependency_overrides[sql_alchemy_engine_provider] = lambda: engine

            log.info("Using dask scheduler address: %s", dask_scheduler_address)

            yield client

    @classmethod
    def setup_class(cls) -> None:
        cls.user_id = "integration-test-user"
        cls.headers = {"x-user-id": cls.user_id}
        cls.created_chat_ids: list[str] = []

    def _init_user_and_chat(self, client: TestClient) -> Chat:
        user_data = UserCreateRequest(user_id=self.user_id)
        response = client.post("/users", json=user_data.model_dump())
        assert response.status_code == 200, response.text

        response = client.post("/chats", headers=self.headers)
        assert response.status_code == 200, response.text
        chat = Chat.model_validate(response.json())
        self.created_chat_ids.append(chat.id)
        return chat

    def _validate_data_url_structure(self, url: str, chat: Chat) -> None:
        assert url.startswith(f"http://testserver/chats/{chat.id}/data/")

    def _validate_data_url_resolution(self, url: str, client: TestClient) -> None:
        # Check if it resolves to a pre-signed S3 URL
        response = client.get(url, headers=self.headers)
        assert response.status_code == 200, response.text
        url_response = DataDownloadURL.model_validate(response.json())
        assert url_response.url.startswith(f"https://{S3_USER_DATA_BUCKET}.s3")

    def test_chat_basic(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        msg = ChatRequest(content="List the names of your tools.")
        parsed_responses: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200, "Non-200 status code"
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses.append(parsed)
                rprint("==================================")
                rprint(parsed)

        assert len(parsed_responses) >= 2
        assert isinstance(parsed_responses[-2].content, Text)
        assert isinstance(parsed_responses[-1].content, EndOfTurn)

    def test_chat_sst_question(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        msg = ChatRequest(
            content=(
                "Compute the mean sea surface temperature in the "
                "Black Sea on June 01, 2024 to 2 decimal places. (Do not use coordinates "
                "from memory. Do not create any extra outputs. Do not retry on tool failure.)"
            )
        )
        parsed_responses: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200, "Non-200 status code"
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses.append(parsed)
                rprint("==================================")
                rprint(parsed)

        assert isinstance(parsed_responses[-1].content, EndOfTurn)

        parsed_plans = [r.content for r in parsed_responses if isinstance(r.content, Plan)]
        assert len(parsed_plans) > 0, "No Plan response found"

        assert len({plan.id for plan in parsed_plans}) == 1

        # check last plan's steps are in terminal states
        last_plan = parsed_plans[-1]
        assert all(step.status in ("success", "failure", "cancelled") for step in last_plan.steps)

        step_details = [step.details for step in last_plan.steps if step.details is not None]
        assert {type(d.content) for d in step_details} == {
            DatasetSearchDetails,
            DataAnalysisDetails,
        }
        for d in step_details:
            match d.content:
                case DatasetSearchDetails():
                    assert d.content.results is not None
                    assert len(d.content.results) > 0
                case DataAnalysisDetails():
                    assert d.content.solution is not None
                    assert d.content.outputs is not None
                    assert len(d.content.outputs) > 0

        for response in parsed_responses[::-1]:
            if isinstance(response.content, Text):
                assert "20.04" in response.content.text
                break
        else:
            pytest.fail("No Text response found with expected SST value")

    def test_chat_multi_turn(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        msg = ChatRequest(content=("Look up an SST dataset"))
        parsed_responses: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200, "Non-200 status code"
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses.append(parsed)
                rprint("==================================")
                rprint(parsed)

        msg = ChatRequest(content="Remind me what the dataset's temporal and spatial coverage is.")
        parsed_responses: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200, "Non-200 status code"
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses.append(parsed)
                rprint("==================================")
                rprint(parsed)

    def test_create_raster(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        msg = ChatRequest(content="Show me the SST in the Black Sea on June 01, 2024 on the map.")
        parsed_responses: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200, "Non-200 status code"
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses.append(parsed)
                rprint(parsed)

        assert isinstance(parsed_responses[-1].content, EndOfTurn)
        log.debug("parsed responses: %s", parsed_responses)
        # check last plan's steps are in terminal states
        for r in parsed_responses[::-1]:
            if isinstance(r, Plan):
                assert all(step.status in ("success", "failure", "cancelled") for step in r.steps)

        geotiff_responses = [
            response.content
            for response in parsed_responses
            if isinstance(response.content, GeoTIFF)
        ]
        assert len(geotiff_responses) == 1

        geotiff_response = geotiff_responses[0]
        self._validate_data_url_structure(geotiff_response.url, chat)
        self._validate_data_url_resolution(geotiff_response.url, client)
        assert geotiff_response.legend.units == "degrees_Celsius"
        assert geotiff_response.legend.min_value is not None
        assert geotiff_response.legend.max_value is not None
        # Not checking exact label as it is equivalent to the variable name in data_outputs
        # which is generated by the LLM and can vary for each run.

    def test_create_plot_image(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        msg = ChatRequest(
            content=(
                "Plot the daily avg SST in the bounding box [34.5, 41.5, 36.0, 43.0] "
                "for June 1-5, 2024 as a time series. Do not create any extra outputs and "
                "do not retry on tool failure."
            )
        )
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200, "Non-200 status code"
            parsed_responses = [
                ChatResponse.model_validate_json(line) for line in response_stream.iter_lines()
            ]

        for r in parsed_responses:
            if isinstance(r.content, Image):
                image_content = r.content
                assert image_content.media_type == "image/png"
                self._validate_data_url_structure(image_content.url, chat)
                self._validate_data_url_resolution(image_content.url, client)
                break
        else:
            pytest.fail("No Image response found")

    def test_aoi_flow(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        # Create AOI
        geometry = GeoJSONGeometry(
            type="Polygon",
            coordinates=[
                [
                    (-90.02485, 24.97748),
                    (-89.97515, 24.97748),
                    (-89.97515, 25.02252),
                    (-90.02485, 25.02252),
                    (-90.02485, 24.97748),
                ]
            ],
            bbox=None,
        )
        create_aoi_request = AOI(name="AOI 1", geometry=geometry)
        create_resp = client.post(
            "/aois", json=create_aoi_request.model_dump(mode="json"), headers=self.headers
        )

        assert create_resp.status_code == 201

        # List AOIs
        first_page_response = client.get("/aois", headers=self.headers)

        assert first_page_response.status_code == 200
        aois = first_page_response.json()["aois"]
        assert len(aois) == 1
        assert aois[0]["name"] == "AOI 1"

        # Get single AOI
        aoi_id = aois[0]["uuid"]
        aoi_response = client.get(f"/aois/{aoi_id}", headers=self.headers)

        assert aoi_response.status_code == 200
        aoi_dict = aoi_response.json()

        expected_aoi_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-90.02485, 24.97748],
                                [-89.97515, 24.97748],
                                [-89.97515, 25.02252],
                                [-90.02485, 25.02252],
                                [-90.02485, 24.97748],
                            ]
                        ],
                        "bbox": None,
                    },
                    "properties": None,
                    "bbox": None,
                }
            ],
            "bbox": None,
        }

        assert aoi_dict["uuid"] == aoi_id
        assert aoi_dict["name"] == "AOI 1"
        assert aoi_dict["geojson"] == expected_aoi_geojson

        # Prompt with AOI
        msg = ChatRequest(
            content=(
                "What was the mean sea surface temperature in the provided area June 01, "
                "2024? (Do not create any extra outputs. Do not retry on tool failure.)"
            ),
            ui_state=UIState(aoi=aoi_id),
        )
        parsed_responses: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200, "Non-200 status code"
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses.append(parsed)
                rprint(parsed)

        filtered_by_aoi_responses = [
            response.content
            for response in parsed_responses
            if isinstance(response.content, FilteredByAOI)
        ]
        assert len(filtered_by_aoi_responses) == 1

        filtered_by_aoi_response = filtered_by_aoi_responses[0]
        assert filtered_by_aoi_response.geojson.model_dump(mode="json") == expected_aoi_geojson
