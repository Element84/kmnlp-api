from collections.abc import Generator

import boto3
import icechunk
import xarray as xr
from mypy_boto3_s3.client import S3Client
from mypy_boto3_s3.type_defs import DeleteTypeDef, ObjectIdentifierTypeDef

_s3_client: S3Client = boto3.client("s3")  # type: ignore[reportUnknownMemberType]


def list_objects_with_prefix(bucket_name: str, prefix: str) -> Generator[list[str], None, None]:
    objects_name_paginator = _s3_client.get_paginator("list_objects_v2")
    bucket_page_iterator = objects_name_paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix,
    )

    for page in bucket_page_iterator:
        yield [contents["Key"] for contents in page.get("Contents", {})]  # type: ignore[reportTypedDictNotRequiredAccess]


def delete_bucket_files_with_prefix(bucket_name: str, prefix: str) -> None:
    for keys in list_objects_with_prefix(bucket_name, prefix):
        delete_dict = DeleteTypeDef(Objects=[ObjectIdentifierTypeDef(Key=key) for key in keys])
        _s3_client.delete_objects(Bucket=bucket_name, Delete=delete_dict)


def read_dataset_from_icechunk_store(prefix: str) -> xr.Dataset:
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "s3://noaa-cdr-sea-surface-temp-whoi-pds/",
            icechunk.s3_store(region="us-east-1"),
        ),
    )
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            f"s3://kendall-test-datasource/{prefix}",
            icechunk.s3_store(region="us-east-1"),
        ),
    )

    # Credentials
    credentials = icechunk.containers_credentials(
        {
            "s3://noaa-cdr-sea-surface-temp-whoi-pds/": icechunk.s3_credentials(anonymous=True),
        },
    )

    # Open Icechunk store.
    storage = icechunk.s3_storage(
        bucket="kendall-test-datasource",
        prefix=prefix,
        from_env=True,
    )
    repo = icechunk.Repository.open(storage, config, credentials)

    # Create ro session
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)  # type: ignore[reportUnknownMemberType]
