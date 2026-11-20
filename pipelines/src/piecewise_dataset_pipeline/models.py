"""Models."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NewType

import xarray as xr

DimensionName = str
ChunkSizeInDimension = int
VariableChunkSize = Mapping[DimensionName, ChunkSizeInDimension]

VariableName = str
VariableChunkSizesMap = dict[VariableName, VariableChunkSize]

SingleFileDataset = NewType("SingleFileDataset", xr.Dataset)


@dataclass(frozen=True)
class ChunkSizeForVariable:
    """Chunk size for a variable."""

    variable_name: str
    chunk_size: VariableChunkSize


@dataclass(frozen=True)
class SingleFileDatasetDescriptor:
    """Descriptor for dataset backed by a single NetCDF file."""

    source_file_url: str
    source_file_dataset: SingleFileDataset
    data_var_chunk_sizes: VariableChunkSizesMap


@dataclass(frozen=True)
class CombinedDatasetDescriptor:
    """Descriptor for a dataset comprising multiple single-file datasets."""

    comprising_descriptors: list[SingleFileDatasetDescriptor]
    most_common_data_var_chunk_sizes: VariableChunkSizesMap
