import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from rich import print as rprint
from sqlalchemy import create_engine
from testcontainers.postgres import PostgresContainer  # type: ignore[reportMissingTypeStubs]

from app.main import app
from app.providers import sql_alchemy_engine_provider
from app.types.input import UserCreateRequest
from app.types.output import (
    Chat,
    ChatRequest,
    ChatResponse,
    DataAnalysisDetails,
    DatasetSearchDetails,
    EndOfTurn,
    GeoTIFF,
    Image,
    Plan,
    Text,
)
from app.utils.env import DASK_SCHEDULER_ADDRESS

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@pytest.mark.integration
class TestAPI:
    @pytest.fixture
    def client(self) -> Iterator[TestClient]:
        with TestClient(app) as client, PostgresContainer("postgres:17") as postgres:
            psql_url = postgres.get_connection_url()
            engine = create_engine(psql_url)
            app.dependency_overrides[sql_alchemy_engine_provider] = lambda: engine

            yield client

    @classmethod
    def setup_class(cls) -> None:
        cls.headers = {"x-user-id": "test"}
        cls.created_chat_ids: list[str] = []

        log.info("Using dask scheduler address: %s", DASK_SCHEDULER_ADDRESS)

    def _init_user_and_chat(self, client: TestClient) -> Chat:
        user_data = UserCreateRequest(user_id="test")
        response = client.post("/users", json=user_data.model_dump())

        response = client.post("/chats", headers=self.headers)
        assert response.status_code == 200
        chat = Chat.model_validate(response.json())
        self.created_chat_ids.append(chat.id)
        return chat

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
            assert response_stream.status_code == 200
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses.append(parsed)
                rprint("==================================")
                rprint(parsed)

        assert len(parsed_responses) == 2
        assert isinstance(parsed_responses[0].content, Text)
        assert isinstance(parsed_responses[1].content, EndOfTurn)

    def test_chat_sst_question(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        msg = ChatRequest(
            content=(
                "What was the mean sea surface temperature in the "
                "Black Sea on June 01, 2024? (Do not use coordinates "
                "from memory; let the data analysis tool lookup the "
                "exact polygon. Do not create any extra outputs. Do "
                "not retry on tool failure.)"
            )
        )
        parsed_responses: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200
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
            assert response_stream.status_code == 200
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
            assert response_stream.status_code == 200
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses.append(parsed)
                rprint("==================================")
                rprint(parsed)

    def test_create_raster(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        msg = ChatRequest(
            content=(
                "Please provide a raster of the sea surface temperature in the Black Sea on "
                "June 01, 2024."
            )
        )
        parsed_responses: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200
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
        assert geotiff_response.url.startswith(
            "https://test-geotiff-bucket.s3.us-east-1.amazonaws.com/transformed_data_array"
        )
        assert geotiff_response.url.endswith(".tif")
        assert geotiff_response.legend.units == "degrees_Celsius"
        assert geotiff_response.legend.min_value is not None
        assert geotiff_response.legend.max_value is not None
        # Not checking exact label as it is equivalent to the variable name in data_outputs
        # which is generated by the LLM and can vary for each run.

    def test_chat_real_image_response(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        msg = ChatRequest(
            content=(
                "Plot the daily avg SST in the bounding box [34.5, 41.5, 36.0, 43.0] "
                "for June 01-10, 2024 as a time series. Do not create any extra outputs and "
                "do not retry on tool failure."
            )
        )
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            params={"stub_image": True},
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200
            parsed_responses = [
                ChatResponse.model_validate_json(line) for line in response_stream.iter_lines()
            ]

        for r in parsed_responses:
            if isinstance(r.content, Image):
                image_content = r.content
                assert image_content.media_type == "image/png"
                assert image_content.url.startswith(
                    "https://test-geotiff-bucket.s3.us-east-1.amazonaws.com/"
                )
                break
        else:
            pytest.fail("No Image response found")

    def test_chat_stub_image_response(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        msg = ChatRequest(content="List the names of your tools.")
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg.model_dump(mode="json"),
            params={"stub_image": True},
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200
            parsed_responses = [
                ChatResponse.model_validate_json(line) for line in response_stream.iter_lines()
            ]

        assert len(parsed_responses) == 3
        assert isinstance(parsed_responses[-3].content, Text)
        assert isinstance(parsed_responses[-2].content, Image)
        assert isinstance(parsed_responses[-1].content, EndOfTurn)

        image_content = parsed_responses[-2].content
        assert image_content.media_type == "image/png"
        assert image_content.url  # Should be a URL
        # Valid URL format
        assert (
            image_content.url
            == "https://test-geotiff-bucket.s3.us-east-1.amazonaws.com/devel_image.png"
        )

    def test_multi_turn_plan_persistence(self, client: TestClient):
        chat = self._init_user_and_chat(client)

        # First message to create a plan
        msg1 = ChatRequest(content="Planning tool debugging check: create a plan with two steps.")
        parsed_responses1: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg1.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses1.append(parsed)
                rprint("==================================")
                rprint(parsed)

        # Second message to check plan persistence
        msg2 = ChatRequest(content="Planning tool debugging check: update the previous plan.")
        parsed_responses2: list[ChatResponse] = []
        with client.stream(
            "post",
            f"/chats/{chat.id}",
            json=msg2.model_dump(mode="json"),
            headers=self.headers,
        ) as response_stream:
            assert response_stream.status_code == 200
            for line in response_stream.iter_lines():
                parsed = ChatResponse.model_validate_json(line)
                parsed_responses2.append(parsed)
                rprint("==================================")
                rprint(parsed)

        # Check that both responses contain a Plan and the plan IDs match
        plan1: Plan | None = None
        for r in parsed_responses1[::-1]:
            if isinstance(r.content, Plan):
                plan1 = r.content
                break
        else:
            pytest.fail("No Plan response found")

        plan2: Plan | None = None
        for r in parsed_responses2[::-1]:
            if isinstance(r.content, Plan):
                plan2 = r.content
                break
        else:
            pytest.fail("No Plan response found")

        assert plan1.id == plan2.id
