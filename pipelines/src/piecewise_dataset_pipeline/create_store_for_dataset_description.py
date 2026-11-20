"""Create icechunk store dataset."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlparse

import xarray as xr
from icechunk import Session
from icechunk.xarray import to_icechunk
from obstore.store import HTTPStore
from siphon.catalog import TDSCatalog
from virtualizarr.manifests import ManifestArray
from virtualizarr.registry import ObjectStoreRegistry

from piecewise_dataset_pipeline.build_combined_dataset_descriptor import (
    CombinedDatasetDescriptor,
    SingleFileDatasetDescriptor,
    SourceDatasetFilePaths,
    build_combined_dataset_descriptor,
    get_source_file_urls,
)
from piecewise_dataset_pipeline.icechunk_store import (
    HTTPSourceIcechunkRepoLocation,
    IcechunkRepoLocation,
    S3SourceIcechunkRepoLocation,
    icechunk_store_exists,
    session_for_icechunk_store,
)
from piecewise_dataset_pipeline.models import SingleFileDataset, VariableChunkSizesMap
from piecewise_dataset_pipeline.netcdf_dataset_loader import concretized_dataset
from piecewise_dataset_pipeline.utils import object_store_registry_for_bucket_names

log = logging.getLogger(__name__)


class InvalidCatalogURLError(ValueError):
    """Could not parse Thredds catalog URL."""

    def __init__(self, url: str) -> None:
        """Initializer."""
        super().__init__(f"Unable to extract scheme and hostname from provided catalog URL: {url}")
        self.url = url


class CreateStoreRequest(Protocol):
    """Generic protocol for requests to build an Icechunk store."""

    destination_bucket_name: str
    destination_prefix: str

    def get_source_file_urls(self) -> list[str]:
        """Get source file URLs for request."""
        ...

    def icechunk_repo_location(self) -> IcechunkRepoLocation:
        """IcechunkRepoLocation object describing how to read original dataset files."""
        ...

    def source_object_store_registry(self) -> ObjectStoreRegistry:
        """Icechunk Object store registry for source data."""
        ...


@dataclass(frozen=True)
class CreateStoreFromThreddsSourceRequest(CreateStoreRequest):
    """Description of store to create from Thredds source."""

    destination_bucket_name: str
    destination_prefix: str

    catalog_url: str

    def get_source_file_urls(self) -> list[str]:
        """Get source file URLs for Thredds dataset."""
        catalog = TDSCatalog(self.catalog_url)
        return self._collect_urls_for_catalog(catalog)

    def icechunk_repo_location(self) -> IcechunkRepoLocation:
        """Icechunk repo location for HTTP source files."""
        return HTTPSourceIcechunkRepoLocation(
            bucket_name=self.destination_bucket_name,
            bucket_prefix=self.destination_prefix,
            source_url_root=self._root_url,
        )

    def source_object_store_registry(self) -> ObjectStoreRegistry:
        """Object store registry object for source files served over HTTP."""
        return ObjectStoreRegistry({self._root_url: HTTPStore.from_url(self._root_url)})

    @cached_property
    def _root_url(self) -> str:
        parsed_catalog_url = urlparse(self.catalog_url)

        scheme = parsed_catalog_url.scheme
        netloc = parsed_catalog_url.netloc

        if scheme and netloc:
            return f"{scheme}://{netloc}"

        raise InvalidCatalogURLError(self.catalog_url)

    def _collect_urls_for_catalog(self, catalog: TDSCatalog) -> list[str]:
        dataset_urls: list[str] = [
            dataset.access_urls["HTTPServer"]
            for dataset in catalog.datasets.values()
            if "HTTPServer" in dataset.access_urls
        ]
        catalog_urls = [
            self._collect_urls_for_catalog(TDSCatalog(ref.href))
            for ref in catalog.catalog_refs.values()
        ]

        return dataset_urls + list(chain.from_iterable(catalog_urls))


class CreateStoreFromS3SourceRequest(CreateStoreRequest, Protocol):
    """Generic protocol for requests to build an Icechunk store from an S3 source."""

    source_bucket_name: str

    def icechunk_repo_location(self) -> IcechunkRepoLocation:
        """IcechunkRepoLocation for source bucket."""
        return S3SourceIcechunkRepoLocation(
            bucket_name=self.destination_bucket_name,
            bucket_prefix=self.destination_prefix,
            source_dataset_bucket_name=self.source_bucket_name,
        )

    def source_object_store_registry(self) -> ObjectStoreRegistry:
        """Object store registry object for source bucket."""
        return object_store_registry_for_bucket_names([self.source_bucket_name])


@dataclass(frozen=True)
class CreateStoreFromS3WithGlobRequest(CreateStoreFromS3SourceRequest):
    """Description of store to create."""

    source_bucket_name: str
    prefix_glob: str

    destination_bucket_name: str
    destination_prefix: str

    def get_source_file_urls(self) -> list[str]:
        """Get source file URLs by resolving globs."""
        return get_source_file_urls(
            SourceDatasetFilePaths(
                source_bucket_name=self.source_bucket_name,
                prefix_glob=self.prefix_glob,
            ),
        )


@dataclass(frozen=True)
class CreateStoreFromS3WithIndividualFilesRequest(CreateStoreFromS3SourceRequest):
    """Description of store to create."""

    source_bucket_name: str
    paths: list[str]

    destination_bucket_name: str
    destination_prefix: str

    def get_source_file_urls(self) -> list[str]:
        """Get source file URLs by string concatenation.

        NOTE: This path is intended for use with ad hoc jobs and testing. It
        does not validate that the provided paths actually exist in S3.
        """
        return [f"s3://{self.source_bucket_name}/{path}" for path in self.paths]


@dataclass(frozen=True)
class CreateStoreResponse:
    """Responses indicating store was created successfully."""

    url: str


def create_icechunk_store(
    request: CreateStoreRequest,
    on_file_processed: Callable[[Any], None] | None = None,
) -> CreateStoreResponse:
    """Create an icechunk store from an object store with ordered NetCDF files."""
    log.info("create_icechunk_store with request: %s", request)
    source_file_urls = request.get_source_file_urls()
    log.info("source file length: %s", len(source_file_urls))
    log.info("source file URLs: %s", source_file_urls)

    t0 = perf_counter()
    datasets_descriptor = build_combined_dataset_descriptor(
        request.source_object_store_registry(),
        source_file_urls,
    )
    t1 = perf_counter()
    log.debug("build_combined_dataset_descriptor time: %.2f", t1 - t0)

    _create_icechunk_store(request, datasets_descriptor, on_file_processed=on_file_processed)

    return CreateStoreResponse(
        url=f"s3://{request.destination_bucket_name}/{request.destination_prefix}",
    )


def _create_icechunk_store(
    request: CreateStoreRequest,
    combined_descriptor: CombinedDatasetDescriptor,
    on_file_processed: Callable[[Any], None] | None = None,
) -> None:
    location = request.icechunk_repo_location()
    should_append = icechunk_store_exists(location)
    session = session_for_icechunk_store(location)

    target_chunk_sizes = combined_descriptor.most_common_data_var_chunk_sizes

    for descriptor in combined_descriptor.comprising_descriptors:
        add_dataset_to_icechunk_store(
            storable_dataset(
                descriptor,
                target_chunk_sizes,
            ),
            session,
            target_chunk_sizes,
            is_append=should_append,
        )

        should_append = True

        if on_file_processed is not None:
            on_file_processed(descriptor.source_file_url)

    session.commit("Initial commit")


def storable_dataset(
    descriptor: SingleFileDatasetDescriptor,
    target_chunk_sizes: VariableChunkSizesMap,
) -> SingleFileDataset:
    """Return a dataset that can be written to an Icechunk store.

    In order for a dataset to be able to be added to the icechunk store it:
        * can be virtual if its chunk sizes are already of the size in which we
          want them stored
        * must be concretized if it needs to be rechunked

    This function will return the dataset from the provided descriptor if it is
    usable with the target chunk size. Otherwise, it will concretize it with
    the needed chunk size and return that dataset.

    Args:
        descriptor (SingleFileDatasetDescriptor): Descriptor for a dataset
            built from a single underlying NetCDF file.

        target_chunk_sizes (VariableChunkSizesMap): The chunk sizes the dataset
            should have when stored.

    Returns:
        (xr.Dataset): The dataset that can be used for storage.
    """
    if descriptor.data_var_chunk_sizes == target_chunk_sizes:
        return descriptor.source_file_dataset

    return concretized_dataset(descriptor, target_chunk_sizes)


def add_dataset_to_icechunk_store(
    dataset: SingleFileDataset,
    session: Session,
    target_chunk_sizes: VariableChunkSizesMap,
    *,
    is_append: bool,
) -> None:
    """Add a single-file dataset to an icechunk store.

    Args:
        dataset (xr.Dataset): Dataset to add to an existing Icechunk store.

        session (Session): A writeable session to an existing Icechunk store.
            (The store might be empty.)

        target_chunk_sizes (VariableChunkSizesMap): The chunk sizes of the
            dataset we expect to write.

        is_append (bool): Whether this is an append operation or the first
            dataset added to the Icechunk store.
    """
    if _is_virtual(dataset):
        dataset.virtualize.to_icechunk(
            session.store,
            append_dim="time" if is_append else None,
        )
    else:
        # 1. Zarr format 3 for icechunk compatibility
        # 2. Not consolidated because Icechunk will "consolidate" metadata in
        #    its own format anyway.
        # 3. Encoding with chunk sizes so that chunks are written with the expected chunk size.
        to_icechunk(
            dataset,
            session,
            append_dim="time" if is_append else None,
            encoding={
                data_var_name: {
                    "chunks": tuple(
                        [chunk_sizes[str(dimension_name)] for dimension_name in dataset.dims],
                    ),
                }
                for data_var_name, chunk_sizes in target_chunk_sizes.items()
            }
            if not is_append
            else None,
        )


def _is_virtual(dataset: xr.Dataset) -> bool:
    for variable in chain(dataset.data_vars, dataset.coords):
        if isinstance(dataset[variable].data, ManifestArray):
            return True

    return False
