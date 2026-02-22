from functools import cache

from sqlalchemy import MetaData, StaticPool, create_engine

from app.db.sqlalchemy_db import SQLAlchemyDB


@cache
def feature_test_db_provider() -> SQLAlchemyDB:
    return SQLAlchemyDB(
        # Use StaticPool to always get the same connection to the in-memory
        # database. This makes it so we always have the same database.
        # Disable check_same_thread because FastAPI spins up many threads.
        create_engine(
            "sqlite+pysqlite:///:memory:",
            echo=True,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        ),
        MetaData(),
    )
