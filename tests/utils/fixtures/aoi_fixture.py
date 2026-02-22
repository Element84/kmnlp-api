from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import Table, insert

from app.db.sqlalchemy_db import SQLAlchemyDB
from app.types.geojson import GeoJSONGeometry
from app.types.internal import AOI


class AOIFixture:
    """Fixture for creating AOIs in DB with some expected values."""

    db: SQLAlchemyDB
    _table: Table

    def __init__(self, db: SQLAlchemyDB) -> None:
        """Initializer."""
        self.db = db
        self._table = Table("aois", db.metadata, autoload_with=db.engine)

    def create_aoi(self, user_id: str | None = None, created_at: datetime | None = None) -> AOI:
        aoi_id = uuid4()
        user_id = user_id if user_id else "test-user-id"
        name = "test-name"
        geometry = GeoJSONGeometry(type="Point", coordinates=(0, 0), bbox=None)
        created_at = created_at if created_at else datetime.now(tz=UTC)

        stmt = insert(self._table).values(
            id=aoi_id,
            user_id=user_id,
            name=name,
            geojson=geometry.model_dump_json(),
            created_at=created_at,
        )

        self.db.insert_or_update_single_row(stmt)

        return AOI(
            uuid=aoi_id,
            user_id=user_id,
            name=name,
            geojson=geometry,
            created_at=created_at,
        )
