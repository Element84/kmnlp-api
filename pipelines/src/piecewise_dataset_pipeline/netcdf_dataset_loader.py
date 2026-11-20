"""Rechunk Dataset."""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from time import perf_counter

import fsspec  # type: ignore[reportMissingTypeStubs]
import xarray as xr
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.parsers.typing import Parser
from virtualizarr.registry import ObjectStoreRegistry

from piecewise_dataset_pipeline.dataset_chunk_summary import dataset_chunk_sizes
from piecewise_dataset_pipeline.models import (
    SingleFileDataset,
    SingleFileDatasetDescriptor,
    VariableChunkSizesMap,
)

log = logging.getLogger(__name__)


@dataclass
class DataFileURLAndVirtualDataset:
    """Data file URL, Dataset Pair."""

    url: str
    dataset: SingleFileDataset


class DatasetHasTargetChunkSizesError(ValueError):
    """Error indicating dataset already has target chunk size."""

    def __init__(self, dataset: xr.Dataset, target_chunk_sizes: VariableChunkSizesMap) -> None:
        """Initializer."""
        self.dataset = dataset
        self.target_chunk_sizes = target_chunk_sizes
        super().__init__("Dataset chunk sizes already matches target chunk sizes.")


class MismatchedDataVarsError(ValueError):
    """Error indicating dataset and target chunk sizes have mismatched dimensions."""

    def __init__(self, dataset: xr.Dataset, target_chunk_sizes: VariableChunkSizesMap) -> None:
        """Initializer."""
        self.dataset = dataset
        self.target_chunk_sizes = target_chunk_sizes
        super().__init__(
            "Dataset data vars do not match target chunk sizes. %s, %s",
            dataset.data_vars,
            target_chunk_sizes,
        )


def virtual_datasets_for_netcdf_urls(
    source_file_urls: list[str],
    registry: ObjectStoreRegistry,
) -> list[DataFileURLAndVirtualDataset]:
    """Return a virtual dataset for each NetCDF file provided."""
    parser = HDFParser()

    # Use a thread pool to build virtual datasets in parallel. The majority of
    # time is spent in I/O, and each item is completely independent.
    #
    # Start with 64 workers. Default in production would be 8 to limit
    # contention, but CPU contention is not a big concern for us because we are
    # heavily I/O-bound.
    with ThreadPoolExecutor(max_workers=64) as executor:
        # This line is functionally identical to the lambda version in this
        # comment, but Python is able to resolve the types.
        #
        # r = executor.map(lambda url: _virtual_dataset(url, registry, parser), source_file_urls) # noqa: ERA001 E501
        result = executor.map(
            partial(_virtual_dataset, registry=registry, parser=parser), source_file_urls
        )

    return list(result)


def concretized_dataset(
    descriptor: SingleFileDatasetDescriptor,
    target_chunk_sizes: VariableChunkSizesMap,
) -> SingleFileDataset:
    """Load the dataset for the descriptor into a fully-conretized xr.Dataset."""
    dataset = descriptor.source_file_dataset

    # Validate that we do not concretize datasets we don't need to.
    if dataset_chunk_sizes(dataset) == target_chunk_sizes:
        raise DatasetHasTargetChunkSizesError(
            dataset,
            target_chunk_sizes,
        )

    # Validate that the dataset has the dimensions we expect.
    non_scalar_data_vars = {k for k, v in dataset.data_vars.items() if len(v.dims) > 0}
    if non_scalar_data_vars != set(target_chunk_sizes.keys()):
        raise MismatchedDataVarsError(dataset, target_chunk_sizes)

    return _load_netcdf_file(descriptor.source_file_url)


def _virtual_dataset(
    url: str,
    registry: ObjectStoreRegistry,
    parser: Parser,
) -> DataFileURLAndVirtualDataset:
    t0 = perf_counter()
    with open_virtual_dataset(url, registry, parser) as dataset:
        t1 = perf_counter()
        log.debug("open_virtual_dataset for url %s took %.2f", url, t1 - t0)
        return DataFileURLAndVirtualDataset(
            url=url,
            dataset=SingleFileDataset(dataset),
        )


def _load_netcdf_file(
    data_file_url: str,
) -> SingleFileDataset:
    """Load the NetCDF file at a given URL into an in-memory xr.Dataset."""
    fs = fsspec.filesystem("s3", anon=True)  # type: ignore[reportUnknownMemberType]

    with fs.open(data_file_url) as f:
        return SingleFileDataset(xr.open_dataset(f, engine="h5netcdf", chunks="auto").load())  # type: ignore[reportUnknownMemberType]
