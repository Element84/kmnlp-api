import pytest
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

from piecewise_dataset_pipeline.build_combined_dataset_descriptor import (
    SingleFileDatasetDescriptor,
)
from piecewise_dataset_pipeline.dataset_chunk_summary import (
    NoChunksError,
    VariableChunkSizesMap,
    dataset_chunk_sizes,
)
from piecewise_dataset_pipeline.models import SingleFileDataset
from piecewise_dataset_pipeline.netcdf_dataset_loader import (
    DatasetHasTargetChunkSizesError,
    concretized_dataset,
)
from piecewise_dataset_pipeline.utils import object_store_registry_for_bucket_names


@pytest.mark.integration
def test_concretize_dataset_raises_if_no_rechunking_needed():
    # Setup
    url = "s3://noaa-cdr-sea-surface-temp-whoi-pds/data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191001_C20200225.nc"
    registry = object_store_registry_for_bucket_names(
        ["noaa-cdr-sea-surface-temp-whoi-pds", "kendall-test-datasource"],
    )
    parser = HDFParser()
    dataset = SingleFileDataset(open_virtual_dataset(url, registry, parser))

    chunk_sizes: VariableChunkSizesMap = {
        "sea_surface_temperature": {"time": 3, "lat": 360, "lon": 720},
        "fill_missing_qc": {"time": 4, "lat": 360, "lon": 720},
    }

    # Execute and expect exception
    with pytest.raises(DatasetHasTargetChunkSizesError):
        concretized_dataset(
            SingleFileDatasetDescriptor(url, dataset, chunk_sizes),
            chunk_sizes,
        )


@pytest.mark.integration
def test_concretized_dataset_if_unexpected_chunk_sizes():
    # Setup
    url = "s3://noaa-cdr-sea-surface-temp-whoi-pds/data/2019/SEAFLUX-OSB-CDR_V02R00_SST_D20191201_C20200421.nc"
    registry = object_store_registry_for_bucket_names(
        ["noaa-cdr-sea-surface-temp-whoi-pds", "kendall-test-datasource"],
    )
    parser = HDFParser()
    dataset = SingleFileDataset(open_virtual_dataset(url, registry, parser))

    # Validate chunk size is not the one we are chunking to
    assert len(dataset.data_vars) == 2
    assert dataset_chunk_sizes(dataset) == {
        "sea_surface_temperature": {"time": 2, "lat": 360, "lon": 434},
        "fill_missing_qc": {"time": 4, "lat": 357, "lon": 714},
    }

    # Execute
    rechunked = concretized_dataset(
        SingleFileDatasetDescriptor(
            source_file_url=url,
            source_file_dataset=dataset,
            data_var_chunk_sizes={
                "sea_surface_temperature": {"time": 2, "lat": 360, "lon": 434},
                "fill_missing_qc": {"time": 4, "lat": 357, "lon": 714},
            },
        ),
        {
            "sea_surface_temperature": {"time": 3, "lat": 360, "lon": 720},
            "fill_missing_qc": {"time": 4, "lat": 360, "lon": 720},
        },
    )

    # Validate
    assert len(rechunked.data_vars) == 2

    # Data is in-memory. It should therefore have no chunks.
    with pytest.raises(NoChunksError):
        dataset_chunk_sizes(rechunked)

    # Data is in-memory. It therefore has no chunks. But the `shape` shows the
    # total dimensions of the array.
    assert rechunked["sea_surface_temperature"].shape == (8, 720, 1440)
    assert rechunked["fill_missing_qc"].shape == (8, 720, 1440)
