import pytest

from piecewise_dataset_pipeline.build_combined_dataset_descriptor import (
    build_combined_dataset_descriptor,
)
from piecewise_dataset_pipeline.utils import object_store_registry_for_bucket_names


@pytest.mark.integration
def test_inspect_dataset_files_for_rechunking_happy_path():
    result = build_combined_dataset_descriptor(
        object_store_registry_for_bucket_names(["noaa-cdr-sea-surface-temp-whoi-pds"]),
        [
            "s3://noaa-cdr-sea-surface-temp-whoi-pds/data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191230_C20200421.nc",
            "s3://noaa-cdr-sea-surface-temp-whoi-pds/data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191231_C20200421.nc",
        ],
    )

    assert result.most_common_data_var_chunk_sizes == {
        "fill_missing_qc": {
            "time": 4,
            "lat": 357,
            "lon": 714,
        },
        "sea_surface_temperature": {
            "time": 2,
            "lat": 360,
            "lon": 434,
        },
    }

    assert [d.source_file_url for d in result.comprising_descriptors] == [
        "s3://noaa-cdr-sea-surface-temp-whoi-pds/data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191230_C20200421.nc",
        "s3://noaa-cdr-sea-surface-temp-whoi-pds/data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191231_C20200421.nc",
    ]
