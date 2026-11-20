import math
import time
from datetime import datetime

import pytest
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

from piecewise_dataset_pipeline.icechunk_store import (
    S3SourceIcechunkRepoLocation,
    session_for_icechunk_store,
)
from piecewise_dataset_pipeline.utils import object_store_registry_for_bucket_names
from tests.utils import read_dataset_from_icechunk_store


@pytest.mark.integration
def test_create_virtual_dataset_from_netcdf_file():
    # Define source NetCDF file
    source_bucket_name = "noaa-cdr-sea-surface-temp-whoi-pds"
    source_file_path = "data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191030_C20200225.nc"

    # Create registry and parser to parse NetCDF file.
    registry = object_store_registry_for_bucket_names([source_bucket_name])
    parser = HDFParser()

    # Build a virtual dataset from NetCDF file.
    virtual_dataset = open_virtual_dataset(
        f"s3://{source_bucket_name}/{source_file_path}",
        registry,
        parser,
    )

    # Assert dataset exists.
    assert virtual_dataset

    # Write dataset to Icechunk
    store_prefix = f"kmnlp-monthly-demo-{time.time()}/"
    session = session_for_icechunk_store(
        S3SourceIcechunkRepoLocation(
            bucket_name="kendall-test-datasource",
            bucket_prefix=store_prefix,
            source_dataset_bucket_name=source_bucket_name,
        ),
    )

    virtual_dataset.virtualize.to_icechunk(session.store)
    session.commit("Initial commit.")


@pytest.mark.integration
def test_read_icechunk_store():
    icechunk_prefix = "kmnlp-monthly-demo-1780676720.409843/"

    dataset_from_icechunk = read_dataset_from_icechunk_store(icechunk_prefix)

    sst_value = (
        dataset_from_icechunk["sea_surface_temperature"]  # type: ignore[reportUnknownMemberType]
        .sel(
            time=datetime.fromisoformat("2019-10-30"),
            lat=25.0,
            lon=270.0,
            method="nearest",
        )
        .compute()
        .item()
    )

    assert math.isclose(sst_value, 28.2, abs_tol=0.05)
