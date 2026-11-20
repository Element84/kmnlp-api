import math
import time
from datetime import datetime

import pytest
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

from piecewise_dataset_pipeline.build_combined_dataset_descriptor import (
    SingleFileDatasetDescriptor,
)
from piecewise_dataset_pipeline.create_store_for_dataset_description import (
    CreateStoreFromS3WithGlobRequest,
    CreateStoreFromS3WithIndividualFilesRequest,
    CreateStoreResponse,
    add_dataset_to_icechunk_store,  # type: ignore[reportPrivateUsage]
    create_icechunk_store,
)
from piecewise_dataset_pipeline.dataset_chunk_summary import (
    VariableChunkSizesMap,
    dataset_chunk_sizes,
)
from piecewise_dataset_pipeline.icechunk_store import (
    session_for_icechunk_store,
)
from piecewise_dataset_pipeline.models import SingleFileDataset
from piecewise_dataset_pipeline.netcdf_dataset_loader import concretized_dataset
from piecewise_dataset_pipeline.utils import object_store_registry_for_bucket_names
from tests.utils import delete_bucket_files_with_prefix, read_dataset_from_icechunk_store

TEST_BUCKET = "kendall-test-datasource"


@pytest.fixture(autouse=True)
def icechunk_store_prefix():
    icechunk_store_prefix = f"icechunk-{time.time()}/"
    yield icechunk_store_prefix
    delete_bucket_files_with_prefix(TEST_BUCKET, icechunk_store_prefix)


@pytest.mark.integration
def test_create_store_happy_path(icechunk_store_prefix: str):
    result = create_icechunk_store(
        CreateStoreFromS3WithGlobRequest(
            source_bucket_name="noaa-cdr-sea-surface-temp-whoi-pds",
            prefix_glob="data/2019/*2019123*.nc",
            destination_bucket_name="kendall-test-datasource",
            destination_prefix=icechunk_store_prefix,
        ),
    )

    # Unfortunately, need the interpolation instead of literal because the time
    # is in the prefix.
    assert result == CreateStoreResponse(
        url=f"s3://kendall-test-datasource/{icechunk_store_prefix}",
    )
    # Validate url ends with a /, and not 2
    assert result.url.endswith("/")
    assert not result.url.endswith("//")

    z = read_dataset_from_icechunk_store(icechunk_store_prefix)
    assert len(z.isel(time=[0], lat=[0], lon=[0])) > 0


@pytest.mark.integration
def test_rechunk_and_store_single_file(icechunk_store_prefix: str):
    # Setup
    url = "s3://noaa-cdr-sea-surface-temp-whoi-pds/data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191201_C20200421.nc"
    registry = object_store_registry_for_bucket_names(
        ["noaa-cdr-sea-surface-temp-whoi-pds", "kendall-test-datasource"],
    )
    parser = HDFParser()
    dataset = SingleFileDataset(open_virtual_dataset(url, registry, parser))

    expected_source_chunk_sizes: VariableChunkSizesMap = {
        "sea_surface_temperature": {"time": 2, "lat": 360, "lon": 434},
        "fill_missing_qc": {"time": 4, "lat": 357, "lon": 714},
    }

    target_chunk_sizes: VariableChunkSizesMap = {
        "sea_surface_temperature": {"time": 3, "lat": 360, "lon": 720},
        "fill_missing_qc": {"time": 4, "lat": 360, "lon": 720},
    }

    # Validate chunk size is not the one we are chunking to
    assert len(dataset.data_vars) == 2
    assert dataset_chunk_sizes(dataset) == expected_source_chunk_sizes

    # Rechunk
    rechunked = concretized_dataset(
        SingleFileDatasetDescriptor(
            source_file_url=url,
            source_file_dataset=dataset,
            data_var_chunk_sizes=expected_source_chunk_sizes,
        ),
        target_chunk_sizes,
    )

    # Validate rechunked dataset looks as expected
    assert len(rechunked.data_vars) == 2

    # Create icechunk store
    request = CreateStoreFromS3WithGlobRequest(
        source_bucket_name="noaa-cdr-sea-surface-temp-whoi-pds",
        prefix_glob="data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191201_C20200421.nc",
        destination_bucket_name="kendall-test-datasource",
        destination_prefix=icechunk_store_prefix,
    )
    icechunk_session = session_for_icechunk_store(request.icechunk_repo_location())

    # Add rechunked dataset to icechunk store
    add_dataset_to_icechunk_store(
        rechunked,
        icechunk_session,
        target_chunk_sizes,
        is_append=False,
    )

    icechunk_session.commit("Initial commit")

    z = read_dataset_from_icechunk_store(icechunk_store_prefix)
    assert len(z.isel(time=[0], lat=[0], lon=[0])) > 0
    assert dataset_chunk_sizes(z) == target_chunk_sizes


@pytest.mark.integration
def test_create_store_with_specific_files(icechunk_store_prefix: str):
    result = create_icechunk_store(
        CreateStoreFromS3WithIndividualFilesRequest(
            source_bucket_name="noaa-cdr-sea-surface-temp-whoi-pds",
            paths=[
                "data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191030_C20200225.nc",
                "data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191031_C20200225.nc",
                "data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191201_C20200421.nc",
            ],
            destination_bucket_name="kendall-test-datasource",
            destination_prefix=icechunk_store_prefix,
        ),
    )

    # Unfortunately, need the interpolation instead of literal because the time
    # is in the prefix.
    assert result == CreateStoreResponse(
        url=f"s3://kendall-test-datasource/{icechunk_store_prefix}",
    )
    # Validate url ends with a /, and not 2
    assert result.url.endswith("/")
    assert not result.url.endswith("//")

    z = read_dataset_from_icechunk_store(icechunk_store_prefix)
    assert len(z.isel(time=[0], lat=[0], lon=[0])) > 0

    # November 2019 is missing. What do our data look like?
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
    december_1 = (
        z["sea_surface_temperature"]  # type: ignore[reportUnknownMemberType]
        .sel(
            time=datetime.fromisoformat("2019-12-01"),
            lat=25.0,
            lon=270.0,
            method="nearest",
        )
        .compute()
        .item()
    )
    november_1 = (
        z["sea_surface_temperature"]  # type: ignore[reportUnknownMemberType]
        .sel(
            time=datetime.fromisoformat("2019-11-01"),
            lat=25.0,
            lon=270.0,
            method="nearest",
        )
        .compute()
        .item()
    )

    assert math.isclose(october_30, 28.2, abs_tol=0.05)
    assert math.isclose(october_31, 28.1, abs_tol=0.05)
    assert math.isclose(december_1, 26.2, abs_tol=0.05)
    assert math.isnan(november_1)


@pytest.mark.integration
def test_create_store_with_mismatched_chunk_sizes(icechunk_store_prefix: str):
    result = create_icechunk_store(
        CreateStoreFromS3WithGlobRequest(
            source_bucket_name="noaa-cdr-sea-surface-temp-whoi-pds",
            prefix_glob="data/2019/*D20191*.nc",
            destination_bucket_name="kendall-test-datasource",
            destination_prefix=icechunk_store_prefix,
        ),
    )

    # Unfortunately, need the interpolation instead of literal because the time
    # is in the prefix.
    assert result == CreateStoreResponse(
        url=f"s3://kendall-test-datasource/{icechunk_store_prefix}",
    )
    # Validate url ends with a /, and not 2
    assert result.url.endswith("/")
    assert not result.url.endswith("//")

    z = read_dataset_from_icechunk_store(icechunk_store_prefix)
    assert len(z.isel(time=[0], lat=[0], lon=[0])) > 0
