"""Dataset chunk summary."""

from typing import Any

import xarray as xr

from piecewise_dataset_pipeline.models import VariableChunkSize, VariableChunkSizesMap


class NoChunksError(ValueError):
    """Error indicating there were no chunks for a data variable in a DataSet.

    Normally, this means that the data variable was not an array.
    """

    data_array: xr.DataArray

    def __init__(self, data_array: xr.DataArray) -> None:
        """Initializer."""
        self.data_array = data_array
        super().__init__(f"No chunks for data variable, source: {data_array}")


class UnexpectedChunkSizesError(ValueError):
    """Error indicating that chunks were not a tuple of ints.

    Note that this error is possible because of the mismatch of types between
    `xr.DataArray`'s `chunks` attribute at the same attribute on our virtual
    data array.
    """

    def __init__(self, chunk_sizes: Any) -> None:  # noqa: ANN401
        """Initializer."""
        super().__init__("Unexepected chunk sizes: %s", chunk_sizes)


def dataset_chunk_sizes(dataset: xr.Dataset) -> VariableChunkSizesMap:
    """Get dataset chunk sizes by data variable name.

    The methods that xarray provides on the dataset chunks raise errors if the
    variables have inconsistent chunks between them.
    """
    return {
        str(k): _get_chunks_or_raise(v)
        for k, v in dataset.data_vars.items()
        if len(v.dims) > 0  # Skip scalar values
    }


def _get_chunks_or_raise(data_array: xr.DataArray) -> VariableChunkSize:
    """Fix the chunks type, and raise if none are present.

    The `xr.DataArray` `chunksizes` attribute is documented as a
    `Mapping[Any, tuple[int, ...]]`.

    However, the `virtualizarr.manifests.array.ManifestArray` that represents a
    virtual dataset does not adhere to this interface. Instead, it returns a
    `Mapping[Any, int]`.

    The xarray spec tells you you'll get a tuple representing the chunk size in
    the given dimension for each chunk in the array. The value returned from
    VirtualiZarr is instead the general chunk size for chunks in the dataset.

    This function returns the general chunk size for chunks in the dataset.

    Args:
        data_array (xr.DataArray): The data_array to get chunk informaion for.

    Returns:
        (VariableChunkSize): A mapping representing the general chunk sizes in
            the given dimensions for the data array.

    """
    chunk_sizes = data_array.chunksizes

    if len(chunk_sizes) == 0:
        raise NoChunksError(data_array)

    return {
        str(dimension_name): _get_general_chunk_size_in_dimension_or_raise(chunk_size)
        for dimension_name, chunk_size in chunk_sizes.items()
    }


def _get_general_chunk_size_in_dimension_or_raise(chunk_sizes: tuple[int, ...] | int) -> int:
    if isinstance(chunk_sizes, tuple) and len(chunk_sizes) > 0:
        return chunk_sizes[0]

    if isinstance(chunk_sizes, int):
        return chunk_sizes

    raise UnexpectedChunkSizesError(chunk_sizes)
