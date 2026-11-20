import sqlite3
from collections.abc import Sequence
from threading import Lock
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    Column,
    Connection,
    Engine,
    ForeignKey,
    Index,
    Insert,
    MetaData,
    Row,
    Select,
    String,
    Table,
    Text,
    Update,
    Uuid,
    event,
    text,
)

mutex = Lock()


class MultipleRowsUpdatedError(ValueError):
    """A query to update a single row updated multiple and issues a rollback."""

    def __init__(self, stmt: Insert | Update, rows: int) -> None:
        """Initializer."""
        super().__init__(
            "SQL Statement to update a single row updated multiple. Statement: %s, rows: %d",
            stmt,
            rows,
        )


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, _: Any) -> None:  # noqa: ANN401
    """Enable foreign keys on SQlite3 database.

    SQLite only enforces foreign keys if the PRAGMA statement turns them on.
    Turn them on whenever we get a new SQLite connection.
    """
    # the sqlite3 driver will not set PRAGMA foreign_keys
    # if autocommit=False; set to True temporarily
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
        dbapi_connection.commit()


class SQLAlchemyDB:
    """Represents a SQLAlchemy engine/db/configuration."""

    engine: Engine
    metadata: MetaData
    users_table: Table
    chats_table: Table
    aois_table: Table

    def __init__(self, engine: Engine, metadata: MetaData) -> None:
        """Initializer."""
        self.engine = engine
        self.metadata = metadata

        self._enable_fks_sqlite()

        # Use mutex to avoid race conditions creating tables.
        # NOTE: While this works because we'll be creating the DB from scratch,
        # if the schema ever changes in any way other than strictly adding a
        # table, migrations are the key. SQLAlchemy uses alembic migrations,
        # which seems likely to be the right path forward for us.
        with mutex:
            # Create schema
            self.users_table = Table(
                "users",
                metadata,
                Column(
                    "id", String(255), primary_key=True
                ),  # NOTE: it would be better if this were a UUID
                Column("date_created", TIMESTAMP(timezone=True), nullable=False),
                keep_existing=True,
            )
            self.chats_table = Table(
                "chats",
                metadata,
                Column(
                    "id", String(255), primary_key=True
                ),  # NOTE: it would be better if this were a UUID
                Column(
                    "user_id",
                    String(255),
                    ForeignKey("users.id", ondelete="CASCADE"),
                    nullable=False,
                ),
                Column("title", String(255), nullable=False),
                Column("messages_json", Text, nullable=False, default="[]"),
                Column("agent_conversation_json", Text, nullable=False, default="[]"),
                Column("plan_manager_json", Text, nullable=False, default="[]"),
                Column("data_refs_json", Text, nullable=False, default="[]"),
                Column("created_at", TIMESTAMP(timezone=True), nullable=False),
                Index("idx_chats_user_id", "user_id"),
                keep_existing=True,
            )
            self.aois_table = Table(
                "aois",
                metadata,
                Column("id", Uuid(native_uuid=False), primary_key=True),
                Column(
                    "user_id",
                    String(255),
                    ForeignKey("users.id", ondelete="CASCADE"),
                    nullable=False,
                ),
                Column("name", String(255), nullable=False),
                Column("geojson", Text, nullable=False),
                Column("created_at", TIMESTAMP(timezone=True), nullable=False),
                Index("idx_aois_user_id", "user_id"),
                keep_existing=True,
            )

            metadata.create_all(engine, checkfirst=True)

            self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """Add columns that may not exist in an already-created table.

        Uses ALTER TABLE ADD COLUMN for each column that is not yet present.
        SQLite (and most backends) will raise an OperationalError if the column
        already exists, which we silently ignore.
        """
        new_columns: list[tuple[str, str]] = [
            ("chats", "data_refs_json TEXT NOT NULL DEFAULT '[]'"),
        ]

        with self.get_connection() as conn:
            for table_name, col_def in new_columns:
                try:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_def}"))
                    conn.commit()
                except Exception:  # noqa: BLE001
                    conn.rollback()

    def _enable_fks_sqlite(self) -> None:
        with self.get_connection() as conn:
            if isinstance(conn, sqlite3.Connection):
                conn.execute(text("PRAGMA foreign_keys = ON;"))

            conn.commit()

    def get_connection(self) -> Connection:
        """Execute a SQL Query."""
        return self.engine.connect()

    def insert_or_update_single_row(self, stmt: Insert | Update) -> bool:
        """INSERT or UPDATE a single row.

        Given an INSERT or UPDATE SQL statement, get a connection to the db and execute it.

        Returns:
            bool: A value indicating whether any rows were changed. True
                indicates yes, False indicates no.

        Raises:
            MultipleRowsUpdatedError: If more than one row would have been
                updated. In the case this happens, a rollback is issues in the
                connection.
        """
        with self.get_connection() as conn:
            result = conn.execute(stmt)

            if result.rowcount == 1:
                conn.commit()
                return True

            if result.rowcount == 0:
                conn.commit()
                return False

            conn.rollback()

        raise MultipleRowsUpdatedError(stmt, result.rowcount)

    def select_many[T: tuple[Any, ...]](self, stmt: Select[T]) -> Sequence[Row[T]]:
        """Get a connection, execute a select statement in it, and return result."""
        with self.get_connection() as conn:
            return conn.execute(stmt).all()

    def select_one[T: tuple[Any, ...]](self, stmt: Select[T]) -> Row[T] | None:
        """Get a connection, execute a select statement in it, and return result."""
        with self.get_connection() as conn:
            return conn.execute(stmt).first()
