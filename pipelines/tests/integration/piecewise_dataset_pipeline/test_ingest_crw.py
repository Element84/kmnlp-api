import time

import pytest

from piecewise_dataset_pipeline.create_store_for_dataset_description import (
    CreateStoreFromS3WithIndividualFilesRequest,
    CreateStoreResponse,
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
def test_create_store_with_specific_files(icechunk_store_prefix: str):
    full_prefix = f"{icechunk_store_prefix}crw_month_mean/"

    result = create_icechunk_store(
        CreateStoreFromS3WithIndividualFilesRequest(
            source_bucket_name="data-c6c22a2e42294c11b52ee7f0c792c071",
            paths=[
                "crw/5km/v3.1/nc/v1.0/monthly/1992/ct5km_sst-max_v3.1_199204.nc",
                "crw/5km/v3.1/nc/v1.0/monthly/1992/ct5km_sst-max_v3.1_199205.nc",
            ],
            destination_bucket_name="kendall-test-datasource",
            destination_prefix=full_prefix,
        ),
    )

    # Unfortunately, need the interpolation instead of literal because the time
    # is in the prefix.
    assert result == CreateStoreResponse(
        url=f"s3://kendall-test-datasource/{full_prefix}",
    )
    # Validate url ends with a /, and not 2
    assert result.url.endswith("/")
    assert not result.url.endswith("//")

    z = read_dataset_from_icechunk_store(full_prefix)
    assert len(z.isel(time=[0], lat=[0], lon=[0])) > 0
