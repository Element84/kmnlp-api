from collections.abc import Sequence
from datetime import MINYEAR, UTC, datetime
from typing import Final
from uuid import UUID, uuid4

from pydantic import TypeAdapter
from sqlalchemy import Table, func, insert, select
from sqlalchemy.exc import IntegrityError

from app.dao.users_dao import UserNotFoundError
from app.db.sqlalchemy_db import SQLAlchemyDB
from app.types.geojson import GeoJSONFeatureCollection, GeoJSONGeometry
from app.types.internal import AOI, AOIListResponse, AOISummary, Cursor

PAGE_SIZE: Final[int] = 15


class AOIsDAO:
    """AOIs DAO."""

    db: SQLAlchemyDB
    _table: Table

    def __init__(self, db: SQLAlchemyDB) -> None:
        """Initializer."""
        self.db = db
        self._table = db.aois_table

    def create_aoi(
        self, user_id: str, name: str, geometry: GeoJSONGeometry | GeoJSONFeatureCollection
    ) -> AOI:
        """Create the requested AOI and return it."""
        aoi_id = uuid4()
        created_at = datetime.now(UTC)

        insert_statement = insert(self._table).values(
            id=aoi_id,
            user_id=user_id,
            name=name,
            geojson=geometry.model_dump_json(),
            created_at=created_at,
        )

        try:
            self.db.insert_or_update_single_row(insert_statement)

        except IntegrityError as e:
            raise UserNotFoundError(user_id) from e

        return AOI(uuid=aoi_id, user_id=user_id, name=name, geojson=geometry, created_at=created_at)

    def get_aoi(self, uuid: str | UUID, user_id: str) -> AOI | None:
        """Return AOI by id or None if not present."""
        uuid = UUID(uuid) if isinstance(uuid, str) else uuid
        columns = self._table.c
        stmt = (
            select(columns.id, columns.user_id, columns.name, columns.geojson, columns.created_at)
            .where(columns.id == uuid)
            .where(columns.user_id == user_id)
            .limit(1)
        )
        row = self.db.select_one(stmt)

        if row is None:
            return None

        geometry_or_feature_collection = TypeAdapter[GeoJSONGeometry | GeoJSONFeatureCollection](
            GeoJSONGeometry | GeoJSONFeatureCollection
        ).validate_json(row.geojson)
        return AOI(
            uuid=row.id,
            user_id=row.user_id,
            name=row.name,
            geojson=geometry_or_feature_collection,
            created_at=row.created_at.replace(tzinfo=UTC),
        )

    def list_aois(
        self, user_id: str, *, created_at_after: datetime | None = None
    ) -> AOIListResponse:
        """Return list of AOIs belonging to user, excluding geometry."""
        effective_created_at_after = (
            created_at_after if created_at_after else datetime(MINYEAR, 1, 1, tzinfo=UTC)
        )

        columns = self._table.c
        stmt = (
            select(columns.id, columns.user_id, columns.name, columns.created_at)
            .where(columns.user_id == user_id)
            .where(columns.created_at > effective_created_at_after)
            .order_by(columns.created_at)
            .limit(PAGE_SIZE)
        )

        rows = self.db.select_many(stmt)

        aois = [
            AOISummary(
                uuid=row.id,
                user_id=row.user_id,
                name=row.name,
                created_at=row.created_at.replace(tzinfo=UTC),
            )
            for row in rows
        ]

        return AOIListResponse(
            aois=aois,
            cursor=_cursor_for_aois_list(aois, expected_page_size=PAGE_SIZE),
            total_count=self._count_aois(user_id),
        )

    def _count_aois(self, user_id: str) -> int:
        stmt = (
            select(func.count("*")).select_from(self._table).where(self._table.c.user_id == user_id)
        )

        with self.db.get_connection() as conn:
            return conn.execute(stmt).scalar_one()


def _cursor_for_aois_list(aois: Sequence[AOISummary], expected_page_size: int) -> Cursor | None:
    if len(aois) < expected_page_size:
        return None

    return Cursor("created_at", aois[-1].created_at)
