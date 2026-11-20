import logging
import os
from typing import cast

import pytest
from dask.distributed import Client, LocalCluster

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def dask_scheduler_address() -> str:  # pyright: ignore[reportInvalidTypeForm]
    dask_scheduler_address = os.getenv("DASK_SCHEDULER_ADDRESS")
    if dask_scheduler_address is None:
        with LocalCluster(threads_per_worker=1) as cluster:  # pyright: ignore[reportUnknownVariableType]
            cluster = cast("LocalCluster", cluster)
            log.info(
                "Started local Dask cluster with scheduler address: %s", cluster.scheduler_address
            )
            yield cluster.scheduler_address  # pyright: ignore[reportReturnType]
    else:
        yield dask_scheduler_address  # pyright: ignore[reportReturnType]


@pytest.fixture(scope="session")
def default_dask_client(dask_scheduler_address: str) -> Client:  # pyright: ignore[reportInvalidTypeForm]
    # In pytest fixtures, yielding instead of returning allows adding teardown code after the yield.
    with Client(dask_scheduler_address, asynchronous=False) as dask_client:
        yield dask_client  # pyright: ignore[reportReturnType]
