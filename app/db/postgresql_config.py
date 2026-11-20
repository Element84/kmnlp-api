import logging
from dataclasses import dataclass

from app.utils.env import DB_ENDPOINT, DB_NAME, DB_PASSWORD, DB_PORT, DB_USERNAME

log = logging.getLogger(__name__)


@dataclass()
class PostgresConfig:
    """Configuration needed to connect to Postgres."""

    host: str
    port: str
    name: str
    user: str
    password: str

    @property
    def connection_string(self) -> str:
        """PostgreSQL connection string built from connection parameters."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


def build_postgres_config() -> PostgresConfig:
    """Return PostgresConfig information.

    Result should not be cached because the token acquired and used as the PSQL
    password is short-lived. Calling this method should return valid
    credentials.
    """
    if DB_ENDPOINT is None:
        raise ValueError("DB_ENDPOINT cannot be None")

    if DB_PORT is None:
        raise ValueError("DB_PORT cannot be None")

    if DB_NAME is None:
        raise ValueError("DB_NAME cannot be None")

    if DB_USERNAME is None:
        raise ValueError("DB_USERNAME cannot be None")

    if DB_PASSWORD is None:
        raise ValueError("DB_PASSWORD cannot be None")

    endpoint = DB_ENDPOINT
    port = DB_PORT
    name = DB_NAME
    username = DB_USERNAME
    password = DB_PASSWORD

    return PostgresConfig(
        host=endpoint,
        port=port,
        name=name,
        user=username,
        password=password,
    )
