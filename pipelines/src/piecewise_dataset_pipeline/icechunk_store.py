"""Create and read Icechunk Store."""

from dataclasses import dataclass
from typing import Protocol

import icechunk
from icechunk.credentials import AnyS3Credential


class IcechunkRepoLocation(Protocol):
    """Description of Icechunk Repository."""

    bucket_name: str
    bucket_prefix: str

    @property
    def icechunk_bucket_url(self) -> str:
        """S3 URL for bucket + prefix."""
        return f"s3://{self.bucket_name}/{self.bucket_prefix}"

    @property
    def source_dataset_url_prefix(self) -> str:
        """URL prefix for the files in the source dataset."""
        ...

    @property
    def source_dataset_credentials(
        self,
    ) -> AnyS3Credential | None:
        """Credentials needed to access the source dataset.

        In practice, there are not any credentials needed, but the way to
        provide no credentials differs between object store and HTTP datasets.
        """
        ...


@dataclass(frozen=True)
class HTTPSourceIcechunkRepoLocation(IcechunkRepoLocation):
    """Description of Icechunk Repository referencing files accessed over HTTP."""

    bucket_name: str
    bucket_prefix: str

    source_url_root: str

    @property
    def source_dataset_url_prefix(self) -> str:
        """URL prefix for the source dataset."""
        return self.source_url_root

    @property
    def source_dataset_credentials(self) -> AnyS3Credential | None:
        """No credentials for HTTP sources."""
        return None


@dataclass(frozen=True)
class S3SourceIcechunkRepoLocation(IcechunkRepoLocation):
    """Description of Icehunk Repository referencing files in S3."""

    bucket_name: str
    bucket_prefix: str

    source_dataset_bucket_name: str

    @property
    def source_dataset_url_prefix(self) -> str:
        """S3 URL for bucket of source dataset."""
        return f"s3://{self.source_dataset_bucket_name}/"

    @property
    def source_dataset_credentials(self) -> AnyS3Credential | None:
        """Anonymous S3 credential to access source dataset."""
        return icechunk.s3_credentials(anonymous=True)


def icechunk_store_exists(location: IcechunkRepoLocation) -> bool:
    """Return whether an Icechunk store already exists at the given location."""
    (storage, _, _) = _icechunk_storage_config_credentials(location)
    return icechunk.Repository.exists(storage)


def session_for_icechunk_store(location: IcechunkRepoLocation) -> icechunk.Session:
    """Open or create an Icechunk store and return a writeable session to it."""
    (storage, config, credentials) = _icechunk_storage_config_credentials(location)
    repo = icechunk.Repository.open_or_create(
        storage,
        config=config,
        authorize_virtual_chunk_access=credentials,
    )
    return repo.writable_session("main")


def read_write_session_for_icechunk_store(location: IcechunkRepoLocation) -> icechunk.Session:
    """Return readwrite session for Icechunk Repository."""
    (storage, config, credentials) = _icechunk_storage_config_credentials(location)
    repo = icechunk.Repository.open(storage, config, credentials)
    return repo.writable_session("main")


def readonly_session_for_icechunk_store(location: IcechunkRepoLocation) -> icechunk.Session:
    """Return readonly session for Icechunk Repository."""
    (storage, config, credentials) = _icechunk_storage_config_credentials(location)
    repo = icechunk.Repository.open(storage, config, credentials)
    return repo.readonly_session("main")


def _icechunk_storage_config_credentials(
    location: IcechunkRepoLocation,
) -> tuple[
    icechunk.Storage,
    icechunk.RepositoryConfig,
    dict[str, icechunk.AnyCredential | None],
]:
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            location.source_dataset_url_prefix,
            icechunk.s3_store(region="us-east-1"),
        ),
    )
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            location.icechunk_bucket_url,
            icechunk.s3_store(region="us-east-1"),
        ),
    )

    credentials = icechunk.containers_credentials(
        {
            location.source_dataset_url_prefix: icechunk.s3_credentials(anonymous=True),
        },
    )

    storage = icechunk.s3_storage(
        bucket=location.bucket_name,
        prefix=location.bucket_prefix,
        from_env=True,
    )

    return (storage, config, credentials)
