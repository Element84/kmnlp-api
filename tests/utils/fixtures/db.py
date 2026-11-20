from collections.abc import Iterator

import pytest
from sqlalchemy import MetaData, create_engine
from testcontainers.postgres import PostgresContainer  # type: ignore[reportMissingTypeStubs]

from app.dao.aois_dao import AOIsDAO
from app.dao.users_dao import UsersDAO
from app.db.sqlalchemy_db import SQLAlchemyDB


# The following 2 fixtures are used for database management. They are intended
# to work as follows:
#   * The `sqlalchemy_db` fixture should be used by any unit
#     test that needs the DB.
#       * n.b. Avoid using it for feature tests if possible. Feature tests
#         should prefer to override the FastAPI dependency rather than having a
#         reference to their own database. It is possible that the provider
#         override would want to use this fixture, though.)
#   * The first call to the sqlalchemy_db fixture will trigger the
#     postgres_container fixture, which has session scope.
#
# The intended behavior is that:
#   * Running a single test that does not require
#     a Postgres database does not wait for one to be created.
#   * Running the test suite creates only a single instance of the database.
#     Individual tests see a clean database because the sqlalchemy_db fixture
#     cleans up after itself.
@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Create a PostgresContainer with pytest session scope."""
    with PostgresContainer("postgres:17") as postgres:
        yield postgres


@pytest.fixture
def sqlalchemy_db(postgres_container: PostgresContainer) -> Iterator[SQLAlchemyDB]:
    """Provide a sqlalchemy_db for an individual test.

    NOTE: This fixture uses the shared postgres_container scope. If you see
    concurrency issues, that might be why.
    """
    # Set echo to True to get statement logging.
    engine = create_engine(postgres_container.get_connection_url(), echo=False)
    metadata = MetaData()

    yield SQLAlchemyDB(engine, metadata)

    metadata.drop_all(engine)


@pytest.fixture
def aois_dao(sqlalchemy_db: SQLAlchemyDB) -> AOIsDAO:
    """Fixture for a default AOIsDAO."""
    return AOIsDAO(sqlalchemy_db)


@pytest.fixture
def users_dao(sqlalchemy_db: SQLAlchemyDB) -> UsersDAO:
    """Fixture for a default UsersDAO."""
    return UsersDAO(sqlalchemy_db)


@pytest.fixture
def users_dao_with_user(sqlalchemy_db: SQLAlchemyDB) -> UsersDAO:
    """Fixture for a default UsersDAO."""
    users_dao = UsersDAO(sqlalchemy_db)
    users_dao.create_user("test-user")
    return users_dao
