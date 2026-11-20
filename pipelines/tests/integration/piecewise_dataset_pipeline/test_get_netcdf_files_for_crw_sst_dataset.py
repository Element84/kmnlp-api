import pytest

from piecewise_dataset_pipeline.create_store_for_dataset_description import (
    CreateStoreFromThreddsSourceRequest,
)


@pytest.mark.integration
def test_inspect_dataset_files_for_rechunking_happy_path():
    sst_catalog = (
        "https://www.ncei.noaa.gov/thredds-ocean/catalog/crw/5km/v3.1/nc/v1.0/daily/sst/catalog.xml"
    )
    request = CreateStoreFromThreddsSourceRequest(
        destination_bucket_name="fake", destination_prefix="fake", catalog_url=sst_catalog
    )
    urls = request.get_source_file_urls()
    assert len(urls) > 10000
