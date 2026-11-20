import math
import time
from datetime import datetime

import pytest

from piecewise_dataset_pipeline.add_netcdf_file_to_icechunk_store import (
    Request,
    Response,
    add_file_to_icechunk_store,
)
from piecewise_dataset_pipeline.create_store_for_dataset_description import (
    CreateStoreFromS3WithIndividualFilesRequest,
    create_icechunk_store,
)
from tests.utils import delete_bucket_files_with_prefix, read_dataset_from_icechunk_store

TEST_BUCKET = "kendall-test-datasource"


@pytest.fixture(autouse=True)
def icechunk_store_prefix():
    icechunk_store_prefix = f"icechunk-{time.time()}/"
    yield icechunk_store_prefix
    delete_bucket_files_with_prefix(TEST_BUCKET, icechunk_store_prefix)


@pytest.mark.integration
def test_add_file_to_icechunk_store(icechunk_store_prefix: str):
    # Setup

    # Icechunk store with data for single day
    create_icechunk_store(
        CreateStoreFromS3WithIndividualFilesRequest(
            source_bucket_name="noaa-cdr-sea-surface-temp-whoi-pds",
            paths=[
                "data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191030_C20200225.nc",
            ],
            destination_bucket_name="kendall-test-datasource",
            destination_prefix=icechunk_store_prefix,
        ),
    )

    # Request
    request = Request(
        icechunk_bucket="kendall-test-datasource",
        icechunk_prefix=icechunk_store_prefix,
        file_bucket="noaa-cdr-sea-surface-temp-whoi-pds",
        file_path="data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191031_C20200225.nc",
    )

    # Execute
    result = add_file_to_icechunk_store(request)

    # Validate
    assert result == Response(
        icechunk_bucket="kendall-test-datasource",
        icechunk_prefix=icechunk_store_prefix,
    )

    z = read_dataset_from_icechunk_store(request.icechunk_prefix)
    october_30 = (
        z["sea_surface_temperature"]  # type: ignore[reportUnknownMemberType]
        .sel(
            time=datetime.fromisoformat("2019-10-30"),
            lat=25.0,
            lon=270.0,
            method="nearest",
        )
        .compute()
        .item()
    )
    october_31 = (
        z["sea_surface_temperature"]  # type: ignore[reportUnknownMemberType]
        .sel(
            time=datetime.fromisoformat("2019-10-31"),
            lat=25.0,
            lon=270.0,
            method="nearest",
        )
        .compute()
        .item()
    )
    assert math.isclose(october_30, 28.2, abs_tol=0.05)
    assert math.isclose(october_31, 28.1, abs_tol=0.05)
