from datetime import datetime
from uuid import UUID, uuid4

import pytest

from app.dao.aois_dao import AOIsDAO
from app.dao.users_dao import UserNotFoundError, UsersDAO
from app.db.sqlalchemy_db import SQLAlchemyDB
from app.types.geojson import GeoJSONGeometry
from app.types.internal import AOIListResponse, AOISummary
from tests.utils.fixtures.aoi_fixture import AOIFixture


class TestAOIsDAO:
    def test_create_aoi_returns_expected_value(self, sqlalchemy_db: SQLAlchemyDB):
        users_dao = UsersDAO(sqlalchemy_db)
        aois_dao = AOIsDAO(sqlalchemy_db)

        _ = users_dao.create_user("test")

        result = aois_dao.create_aoi(
            "test",
            "Gulf of Mexico",
            GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None),
        )

        assert isinstance(result.uuid, UUID)
        assert result.uuid.version == 4

        assert str(result.user_id) == "test"
        assert result.name == "Gulf of Mexico"
        assert result.geojson == GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None)

    def test_create_aoi_as_feature_collection(self, sqlalchemy_db: SQLAlchemyDB):
        users_dao = UsersDAO(sqlalchemy_db)
        aois_dao = AOIsDAO(sqlalchemy_db)

        _ = users_dao.create_user("test")

        result = aois_dao.create_aoi(
            "test",
            "Gulf of Mexico",
            GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None).as_feature_collection(),
        )

        assert isinstance(result.uuid, UUID)
        assert result.uuid.version == 4

        assert str(result.user_id) == "test"
        assert result.name == "Gulf of Mexico"
        assert result.geojson.model_dump() == {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": (0.0, 0.0),
                        "bbox": None,
                    },
                    "properties": None,
                    "bbox": None,
                }
            ],
            "bbox": None,
        }

    def test_create_aoi_fails_if_user_not_exists(self, sqlalchemy_db: SQLAlchemyDB):
        aois_dao = AOIsDAO(sqlalchemy_db)

        with pytest.raises(UserNotFoundError):
            aois_dao.create_aoi(
                "test",
                "Gulf of Mexico",
                GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None),
            )

    def test_get_aoi(self, sqlalchemy_db: SQLAlchemyDB):
        users_dao = UsersDAO(sqlalchemy_db)
        aois_dao = AOIsDAO(sqlalchemy_db)

        _ = users_dao.create_user("test")

        created_aoi = aois_dao.create_aoi(
            "test",
            "Gulf of Mexico",
            GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None),
        )

        aoi = aois_dao.get_aoi(created_aoi.uuid, "test")

        assert aoi == created_aoi

    def test_get_aoi_returns_none_if_wrong_user_requesting(self, sqlalchemy_db: SQLAlchemyDB):
        users_dao = UsersDAO(sqlalchemy_db)
        aois_dao = AOIsDAO(sqlalchemy_db)

        _ = users_dao.create_user("test")

        created_aoi = aois_dao.create_aoi(
            "test",
            "Gulf of Mexico",
            GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None),
        )

        aoi = aois_dao.get_aoi(created_aoi.uuid, "testz")

        assert aoi is None

    def test_get_aoi_return_none_if_not_exists(self, sqlalchemy_db: SQLAlchemyDB):
        aois_dao = AOIsDAO(sqlalchemy_db)
        aoi = aois_dao.get_aoi(uuid4(), "test")

        assert aoi is None

    def test_list_aois_when_empty(self, sqlalchemy_db: SQLAlchemyDB):
        aois_dao = AOIsDAO(sqlalchemy_db)
        result = aois_dao.list_aois("test")

        assert result == AOIListResponse(aois=[], cursor=None, total_count=0)

    def test_list_aois_empty_for_other_user(self, sqlalchemy_db: SQLAlchemyDB):
        # Init DAOs
        users_dao = UsersDAO(sqlalchemy_db)
        aois_dao = AOIsDAO(sqlalchemy_db)

        # Create user "test" and create an AOI for it.
        _ = users_dao.create_user("test")
        _ = aois_dao.create_aoi(
            "test",
            "Gulf of Mexico",
            GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None),
        )

        # Fetch AOIs for a different user.
        result = aois_dao.list_aois("wrong-user")

        # List should be empty.
        assert result == AOIListResponse(aois=[], cursor=None, total_count=0)

    def test_list_aois_for_user(self, sqlalchemy_db: SQLAlchemyDB):
        # Init DAOs
        users_dao = UsersDAO(sqlalchemy_db)
        aois_dao = AOIsDAO(sqlalchemy_db)

        # Create user "test" and create 2 AOIs for it.
        _ = users_dao.create_user("test")
        gulf_aoi = aois_dao.create_aoi(
            "test",
            "Gulf of Mexico",
            GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None),
        )
        ohio_aoi = aois_dao.create_aoi(
            "test",
            "Ohio",
            GeoJSONGeometry(type="Point", coordinates=(1, 1), bbox=None),
        )

        # Fetch AOIs for user.
        result = aois_dao.list_aois("test")

        # List should have both items.
        assert result == AOIListResponse(
            aois=[
                AOISummary(
                    uuid=gulf_aoi.uuid,
                    user_id="test",
                    name="Gulf of Mexico",
                    created_at=gulf_aoi.created_at,
                ),
                AOISummary(
                    uuid=ohio_aoi.uuid, user_id="test", name="Ohio", created_at=ohio_aoi.created_at
                ),
            ],
            cursor=None,
            total_count=2,
        )

    def test_list_aois_first_page(self, sqlalchemy_db: SQLAlchemyDB):
        # Init DAOs
        users_dao = UsersDAO(sqlalchemy_db)
        aois_dao = AOIsDAO(sqlalchemy_db)

        # Create test user.
        _ = users_dao.create_user("test")

        # Create 11 AOIs
        for i in range(16):
            _ = aois_dao.create_aoi(
                "test", f"AOI {i}", GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None)
            )

        # Fetch AOIs for user.
        result = aois_dao.list_aois("test")

        # List should have both items.
        assert len(result.aois) == 15
        assert result.total_count == 16

        assert [aoi.name for aoi in result.aois] == [
            "AOI 0",
            "AOI 1",
            "AOI 2",
            "AOI 3",
            "AOI 4",
            "AOI 5",
            "AOI 6",
            "AOI 7",
            "AOI 8",
            "AOI 9",
            "AOI 10",
            "AOI 11",
            "AOI 12",
            "AOI 13",
            "AOI 14",
        ]

    def test_list_aois_second_page(self, sqlalchemy_db: SQLAlchemyDB):
        # Init DAOs and fixture
        users_dao = UsersDAO(sqlalchemy_db)
        aois_dao = AOIsDAO(sqlalchemy_db)
        aoi_fixture = AOIFixture(sqlalchemy_db)

        # Create test user.
        _ = users_dao.create_user("test")

        # Setup Plan:
        #   * Create 1 AOI created on January 2
        #   * Create 15 AOIs created on January 1
        jan_2 = datetime.fromisoformat("2026-01-02T00:00:00Z")
        jan_1 = datetime.fromisoformat("2026-01-01T00:00:00Z")

        aoi_fixture.create_aoi(user_id="test", created_at=jan_2)

        # Create 15 AOIs with a created-at from the day before
        for _ in range(15):
            aoi_fixture.create_aoi(user_id="test", created_at=jan_1)

        # Fetch AOIs for user.
        early_aois = aois_dao.list_aois("test")

        # List should have 15 items, all with a January 1 created at.
        assert len(early_aois.aois) == 15

        assert {aoi.created_at for aoi in early_aois.aois} == {
            datetime.fromisoformat("2026-01-01T00:00:00Z")
        }

        # Fetch AOIs created after January 1 timestamp.
        aois_after_first_date = aois_dao.list_aois(
            "test", created_at_after=datetime.fromisoformat("2026-01-01T00:00:00Z")
        )

        # Result should be a single AOI with January 2 timestamp.
        assert len(aois_after_first_date.aois) == 1
        assert aois_after_first_date.aois[0].created_at == datetime.fromisoformat(
            "2026-01-02T00:00:00Z"
        )
        assert aois_after_first_date.cursor is None
