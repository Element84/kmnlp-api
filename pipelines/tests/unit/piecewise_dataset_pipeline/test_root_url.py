import pytest

from piecewise_dataset_pipeline.create_store_for_dataset_description import (
    CreateStoreFromThreddsSourceRequest,
    InvalidCatalogURLError,
)


def test_root_url_crw_dataset():
    request = CreateStoreFromThreddsSourceRequest(
        destination_bucket_name="",
        destination_prefix="",
        catalog_url="https://www.ncei.noaa.gov/thredds-ocean/catalog/crw/5km/v3.1/nc/v1.0/daily/sst/catalog.xml",
    )

    assert (
        request._root_url  # type: ignore[reportPrivateUsage]
        == "https://www.ncei.noaa.gov"
    )


def test_root_url_missing_scheme():
    request = CreateStoreFromThreddsSourceRequest(
        destination_bucket_name="",
        destination_prefix="",
        catalog_url="www.ncei.noaa.gov/thredds-ocean/catalog/crw/5km/v3.1/nc/v1.0/daily/sst/catalog.xml",
    )

    with pytest.raises(InvalidCatalogURLError):
        _ = request._root_url  # type: ignore[reportPrivateUsage]
