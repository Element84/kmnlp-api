"""Add NetCDF file to Icechunk store."""

from dataclasses import dataclass

import xarray as xr
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

from piecewise_dataset_pipeline.create_store_for_dataset_description import (
    add_dataset_to_icechunk_store,
    storable_dataset,
)
from piecewise_dataset_pipeline.dataset_chunk_summary import dataset_chunk_sizes
from piecewise_dataset_pipeline.icechunk_store import (
    IcechunkRepoLocation,
    S3SourceIcechunkRepoLocation,
    read_write_session_for_icechunk_store,
)
from piecewise_dataset_pipeline.models import SingleFileDataset, SingleFileDatasetDescriptor
from piecewise_dataset_pipeline.utils import object_store_registry_for_bucket_names


@dataclass(frozen=True)
class Request:
    """Request type."""

    icechunk_bucket: str
    icechunk_prefix: str

    file_bucket: str
    file_path: str

    def icechunk_repo_location(self) -> IcechunkRepoLocation:
        """Icechunk repo location for request."""
        return S3SourceIcechunkRepoLocation(
            bucket_name=self.icechunk_bucket,
            bucket_prefix=self.icechunk_prefix,
            source_dataset_bucket_name=self.file_bucket,
        )


@dataclass(frozen=True)
class Response:
    """Response type."""

    icechunk_bucket: str
    icechunk_prefix: str


# Keeping this code around for now, as we might want it later. It is not used
# in the first pass where we create the full store.
def add_file_to_icechunk_store(request: Request) -> Response:
    """Append a single file to an Icechunk Store."""
    # Create rw session
    session = read_write_session_for_icechunk_store(request.icechunk_repo_location())

    icechunk_dataset = xr.open_zarr(session.store)  # type: ignore[reportUnknownMemberType]
    target_chunk_sizes = dataset_chunk_sizes(icechunk_dataset)

    source_file_url = f"s3://{request.file_bucket}/{request.file_path}"

    registry = object_store_registry_for_bucket_names([request.file_bucket])
    parser = HDFParser()

    with open_virtual_dataset(source_file_url, registry, parser) as dataset:
        descriptor = SingleFileDatasetDescriptor(
            source_file_url=source_file_url,
            source_file_dataset=SingleFileDataset(dataset),
            data_var_chunk_sizes=dataset_chunk_sizes(dataset),
        )

        add_dataset_to_icechunk_store(
            storable_dataset(
                descriptor,
                target_chunk_sizes,
            ),
            session,
            target_chunk_sizes,
            is_append=True,
        )

    session.commit(f"Add file: {request.file_path}")

    return Response(
        icechunk_bucket=request.icechunk_bucket,
        icechunk_prefix=request.icechunk_prefix,
    )
