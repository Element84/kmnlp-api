"""Build combine dataset summary."""

import logging
from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from time import perf_counter

import fsspec  # type: ignore[reportMissingTypeStubs]
from virtualizarr.registry import ObjectStoreRegistry

from piecewise_dataset_pipeline.dataset_chunk_summary import (
    VariableChunkSize,
    dataset_chunk_sizes,
)
from piecewise_dataset_pipeline.models import (
    ChunkSizeInDimension,
    CombinedDatasetDescriptor,
    DimensionName,
    SingleFileDatasetDescriptor,
)
from piecewise_dataset_pipeline.netcdf_dataset_loader import (
    DataFileURLAndVirtualDataset,
    virtual_datasets_for_netcdf_urls,
)

log = logging.getLogger(__name__)


class InconsistentDimensionsError(ValueError):
    """Indicates comprising datasets have inconssitent dimension names."""

    def __init__(self, dimensions: set[frozenset[Hashable]]) -> None:
        """Initializer."""
        msg = (
            "Inconsistent dimensions for datasets from NetCDF files. Combined dimensions set: "
            f"{dimensions}"
        )
        super().__init__(msg)


class InconsistentDataVarNamesError(ValueError):
    """Indicates comprising datasets have inconsistent data varaibles names."""

    def __init__(self, data_variable_names: set[frozenset[Hashable]]) -> None:
        """Initializer."""
        msg = (
            "Inconsistent variable names for datasets from NetCDF files. "
            f"Combined variables names set: {data_variable_names}"
        )
        super().__init__(msg)


class MismatchedKeysError(ValueError):
    """Indicates a provided list of data arrays had chunk sizes with mismatched dimension names."""

    def __init__(self, keys: set[frozenset[DimensionName]]) -> None:
        """Initialzier."""
        self.keys = keys
        super().__init__(self, "Mismatched keys: %s", keys)


@dataclass(frozen=True)
class SourceDatasetFilePaths:
    """Source dataset bucket name and glob for data files."""

    source_bucket_name: str
    prefix_glob: str


def get_source_file_urls(source: SourceDatasetFilePaths) -> list[str]:
    """Use the bucket name and file path glob to build a list of file urls.

    The resulting URLs are returned in ascending alphabetical order. This
    method should only be used if sorting the files will result in an in-order
    list of the underlying data.
    """
    fs = fsspec.filesystem("s3")  # type: ignore[reportUnknownMemberType]

    # Get list of source files for the dataset based on the glob provided.
    #
    # The `sorted` call is important because we assume the files are in
    # increasing-time order when we concatenate the individual data arrays.
    return sorted(
        [
            "s3://" + f
            for f in fs.glob(
                f"s3://{source.source_bucket_name}/{source.prefix_glob}",
            )
        ],
    )


def build_combined_dataset_descriptor(
    registry: ObjectStoreRegistry,
    source_file_urls: list[str],
) -> CombinedDatasetDescriptor:
    """Find files in dataset."""
    # Get data files for dataset in order

    t0 = perf_counter()
    data_file_url_and_virtual_datasets = virtual_datasets_for_netcdf_urls(
        source_file_urls,
        registry,
    )
    t1 = perf_counter()
    log.debug("virtual_datasets_for_netcdf_urls took %.2f", t1 - t0)

    # Build descriptor of individual `xr.Dataset`s, which will be processed to
    # build the combined dataset.
    return _build_combined_datasets_descriptor(data_file_url_and_virtual_datasets)


def _build_combined_datasets_descriptor(
    data_file_url_and_virtual_datasets: list[DataFileURLAndVirtualDataset],
) -> CombinedDatasetDescriptor:
    """Build rechunking summary.

    Build descriptor of individual `xr.Dataset`s, in order, along with common
    information (notably the most common chunk size for each variable).
    """
    individual_descriptors = [
        SingleFileDatasetDescriptor(
            source_file_url=pair.url,
            source_file_dataset=pair.dataset,
            data_var_chunk_sizes=dataset_chunk_sizes(pair.dataset),
        )
        for pair in data_file_url_and_virtual_datasets
    ]

    _validate_individual_descriptors(individual_descriptors)

    return CombinedDatasetDescriptor(
        comprising_descriptors=individual_descriptors,
        most_common_data_var_chunk_sizes={
            variable_name: _most_common_chunk_size(
                [desc.data_var_chunk_sizes[variable_name] for desc in individual_descriptors],
            )
            for variable_name in individual_descriptors[0].data_var_chunk_sizes
        },
    )


def _most_common_chunk_size(chunk_sizes: Iterable[VariableChunkSize]) -> VariableChunkSize:
    """Return most common chunk size in an interable."""
    t0 = perf_counter()
    keys_set = {frozenset(chunk_size.keys()) for chunk_size in chunk_sizes}
    if len(keys_set) > 1:
        raise MismatchedKeysError(keys_set)

    # Hashable copy of items in dictionary. The 3 tuples are:
    # for each virtual dataset (represented by item in chunk_sizes):
    #   for each pair in dictionary (representing the chunks in each dimension):
    #     the pair (tuple) of key/value (representing the dimension and length)
    chunk_sizes_hashable: tuple[tuple[tuple[DimensionName, ChunkSizeInDimension], ...], ...] = (
        tuple(tuple(chunk_size.items()) for chunk_size in chunk_sizes)
    )

    size_counts = Counter(chunk_sizes_hashable)

    t1 = perf_counter()
    log.debug("_most_common_chunk_size calculated in %.2f", t1 - t0)

    # Counter.most_common returns a list of pairs, mapping the item in the
    # Counter to its count.
    #
    # Get the first item (for the most common item), and get the first item of
    # that pair (to get the value without its count).
    #
    # That value is tuple of dimension name: length pairs. Turn that tuple into
    # a dictionary.
    return dict(size_counts.most_common(1)[0][0])


def _validate_individual_descriptors(descriptors: list[SingleFileDatasetDescriptor]) -> None:
    """Validate that all individual datasets are compatible.

    Validate that:
        * All datasets have the same dimensions.
        * All datasets have the same data variables.
        * All data variables are data arrays.
    """
    dimensions = {frozenset(d.source_file_dataset.dims.keys()) for d in descriptors}
    data_variable_names = {frozenset(d.source_file_dataset.data_vars.keys()) for d in descriptors}

    if len(dimensions) != 1:
        raise InconsistentDimensionsError(dimensions)

    if len(data_variable_names) != 1:
        raise InconsistentDataVarNamesError(data_variable_names)
