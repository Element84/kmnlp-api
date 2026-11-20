"""Extract metadata from Icechunk stores for knowledge graph building.

This script discovers and extracts metadata from Icechunk stores in S3, generating
a metadata dictionary compatible with the existing knowledge graph pipeline.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import s3fs  # pyright: ignore[reportMissingTypeStubs]
import xarray as xr
import zarr  # pyright: ignore[reportMissingTypeStubs]
from icechunk import (
    Repository,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
    s3_storage,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# S3 configuration
S3_BUCKET = "data-c6c22a2e42294c11b52ee7f0c792c071"
ICECHUNK_PREFIX = "icechunk/"
OUTPUT_FILE = "icechunk_metadata.json"

_MONTHLY_MID_DAY_MIN = 14
_MONTHLY_MID_DAY_MAX = 16
_DECEMBER = 12
_YEARLY_MID_MONTH_MIN = 6
_YEARLY_MID_MONTH_MAX = 8


def discover_icechunk_stores(s3_client: s3fs.S3FileSystem) -> dict[str, dict[str, str]]:
    """Discover all Icechunk stores under the icechunk/ prefix.

    Returns a nested dictionary: {dataset_name: {group: s3_path}}
    where group represents temporal resolution or other grouping.

    Args:
        s3_client: Configured S3 filesystem client

    Returns:
        Dictionary mapping dataset names to groups to S3 paths
    """
    log.info("Discovering Icechunk stores under %s/%s", S3_BUCKET, ICECHUNK_PREFIX)

    stores: dict[str, dict[str, str]] = {}
    base_path = f"{S3_BUCKET}/{ICECHUNK_PREFIX}"

    # List all top-level directories under icechunk/
    top_level_items = s3_client.ls(base_path, detail=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
    top_level_dirs = [  # pyright: ignore[reportUnknownVariableType]
        d["name"]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportTypedDictNotRequiredAccess]
        for d in top_level_items  # pyright: ignore[reportUnknownVariableType]
        if d.get("type") == "directory"  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    ]

    for top_dir in top_level_dirs:  # pyright: ignore[reportUnknownVariableType]
        dataset_name = Path(top_dir).name  # pyright: ignore[reportUnknownArgumentType]

        # Check if this directory itself is an icechunk store (flat structure)
        if _is_icechunk_store(s3_client, top_dir):  # pyright: ignore[reportUnknownArgumentType]
            log.info("Found flat icechunk store: %s", dataset_name)
            stores[dataset_name] = {"daily": f"s3://{top_dir}"}
            continue

        # Otherwise, check subdirectories (nested structure)
        try:
            subdir_items = s3_client.ls(top_dir, detail=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
            subdirs = [  # pyright: ignore[reportUnknownVariableType]
                d["name"]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportTypedDictNotRequiredAccess]
                for d in subdir_items  # pyright: ignore[reportUnknownVariableType]
                if d.get("type") == "directory"  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            ]

            for subdir in subdirs:  # pyright: ignore[reportUnknownVariableType]
                if _is_icechunk_store(s3_client, subdir):  # pyright: ignore[reportUnknownArgumentType]
                    group_name = Path(subdir).name  # pyright: ignore[reportUnknownArgumentType]
                    log.info("Found nested icechunk store: %s/%s", dataset_name, group_name)

                    if dataset_name not in stores:
                        stores[dataset_name] = {}
                    stores[dataset_name][group_name] = f"s3://{subdir}"
        except Exception as e:  # noqa: BLE001
            log.warning("Error exploring %s: %s", top_dir, e)  # pyright: ignore[reportUnknownArgumentType]
            continue

    total_stores = sum(len(g) for g in stores.values())
    log.info("Discovered %d datasets with %d total stores", len(stores), total_stores)
    return stores


def _is_icechunk_store(s3_client: s3fs.S3FileSystem, path: str) -> bool:
    """Check if a directory contains an Icechunk store.

    Args:
        s3_client: S3 filesystem client
        path: S3 path to check

    Returns:
        True if the path contains an Icechunk store
    """
    # Check for icechunk-specific files/directories
    config_exists = s3_client.exists(f"{path}/config.yaml")  # pyright: ignore[reportUnknownMemberType]
    snapshots_exists = s3_client.exists(f"{path}/snapshots")  # pyright: ignore[reportUnknownMemberType]

    return config_exists or snapshots_exists


def _detect_temporal_resolution(group_name: str) -> str | None:
    """Detect temporal resolution from group name.

    Args:
        group_name: Group name (e.g., 'daily', 'monthly', 'yearly', 'annual_mean')

    Returns:
        Temporal resolution: 'yearly', 'monthly', 'daily', 'hourly', or None
    """
    group_lower = group_name.lower()

    if "yearly" in group_lower or "annual" in group_lower:
        return "yearly"
    if "monthly" in group_lower:
        return "monthly"
    if "daily" in group_lower:
        return "daily"
    if "hourly" in group_lower:
        return "hourly"

    return None


def _adjust_centered_timestamp(
    timestamp: datetime,
    temporal_resolution: str | None,
    *,
    is_end_date: bool = False,
) -> datetime:
    """Adjust centered timestamp to period boundary.

    Climate datasets often use centered timestamps (mid-month for monthly data,
    mid-year for yearly data). This function adjusts them to period boundaries
    for time_coverage_start and time_coverage_end attributes.

    Args:
        timestamp: The timestamp to adjust
        temporal_resolution: 'yearly', 'monthly', 'daily', or None
        is_end_date: If True, adjust to end of period; if False, adjust to start

    Returns:
        Adjusted timestamp at period boundary
    """
    if temporal_resolution == "monthly":
        # Monthly data: centered timestamps use day 15
        # Adjust to first day of month for start, last day for end
        # Detect centered timestamp (days 14-16)
        if _MONTHLY_MID_DAY_MIN <= timestamp.day <= _MONTHLY_MID_DAY_MAX:
            if is_end_date:
                # Move to last day of month
                if timestamp.month == _DECEMBER:
                    next_month = timestamp.replace(year=timestamp.year + 1, month=1, day=1)
                else:
                    next_month = timestamp.replace(month=timestamp.month + 1, day=1)
                return next_month - timedelta(days=1)
            # Move to first day of month
            return timestamp.replace(day=1)
    elif (
        temporal_resolution == "yearly"
        # Detect centered timestamp: months 6-8 (July ± 1), day 1
        and _YEARLY_MID_MONTH_MIN <= timestamp.month <= _YEARLY_MID_MONTH_MAX
        and timestamp.day == 1
    ):
        # Yearly data: centered timestamps use July 1 (mid-year)
        # Adjust to January 1 for start, December 31 for end
        if is_end_date:
            # Move to December 31 of same year
            return timestamp.replace(month=_DECEMBER, day=31)
        # Move to January 1 of same year
        return timestamp.replace(month=1, day=1)

    # No adjustment needed for daily, hourly, or unrecognized resolutions
    return timestamp


def extract_metadata_from_store(  # noqa: C901, PLR0912
    store_uri: str,
    dataset_name: str,
    group_name: str,
) -> dict[str, Any] | None:
    """Extract metadata from a single Icechunk store.

    Args:
        store_uri: S3 URI to the Icechunk store
        dataset_name: Name of the dataset
        group_name: Group/temporal resolution name

    Returns:
        Dictionary containing uri and zarr_metadata, or None if extraction fails
    """
    log.info("Extracting metadata from %s/%s at %s", dataset_name, group_name, store_uri)

    try:
        # Open Icechunk repository
        storage = s3_storage(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            bucket=S3_BUCKET,
            prefix=store_uri.replace(f"s3://{S3_BUCKET}/", ""),
            from_env=True,
        )

        # Authorize virtual chunk access to external S3 buckets
        # Virtual chunks may reference data in external buckets (e.g., public NOAA buckets)
        # Use None to fetch credentials from environment or use anonymous access
        virtual_auth: dict[str, Any] = {
            f"s3://{dataset_name}/": None,  # Dataset's own bucket
        }

        repo = Repository.open(  # pyright: ignore[reportUnknownArgumentType]
            storage, authorize_virtual_chunk_access=virtual_auth
        )

        # Get latest snapshot
        session = repo.readonly_session(branch="main")

        # Open zarr group directly to preserve all attributes (including 'coordinates')
        # Reading through xarray strips the 'coordinates' attribute since xarray uses it internally
        group = zarr.open_group(  # pyright: ignore[reportUnknownMemberType]
            store=session.store,
            mode="r",
            zarr_format=3,
        )

        # Extract metadata in Zarr format (matching ds_metadata.json structure)
        zarr_metadata: dict[str, Any] = {}

        # Global attributes - read directly from zarr to preserve all attributes
        zarr_metadata[".zattrs"] = group.attrs.asdict()  # pyright: ignore[reportUnknownMemberType]

        # We still need xarray to get dimension information
        # Open with decode_coords=False to minimize attribute processing
        ds = xr.open_zarr(  # pyright: ignore[reportUnknownMemberType]
            session.store,
            zarr_format=3,
            consolidated=False,
            decode_coords=False,
        )

        # Variable attributes - read directly from zarr arrays to preserve 'coordinates'
        for var_name, array in group.arrays():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            # Get raw attributes from zarr (preserves 'coordinates' attribute)
            var_attrs = array.attrs.asdict()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

            # Add _ARRAY_DIMENSIONS from xarray (not stored in raw zarr)
            if var_name in ds.data_vars:
                var_attrs["_ARRAY_DIMENSIONS"] = [str(d) for d in ds[var_name].dims]  # pyright: ignore[reportUnknownMemberType]

            zarr_metadata[f"{var_name}/.zattrs"] = var_attrs

        # Dimension/coordinate attributes
        for dim_name in ds.dims:
            if dim_name in ds.coords:
                # Read from zarr group if it's an array, otherwise from xarray
                if str(dim_name) in group:  # pyright: ignore[reportUnknownMemberType]
                    dim_array = group[str(dim_name)]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    zarr_metadata[f"{dim_name}/.zattrs"] = dim_array.attrs.asdict()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                else:
                    coord = ds.coords[dim_name]
                    zarr_metadata[f"{dim_name}/.zattrs"] = dict(coord.attrs)

        # Override temporal coverage with actual time coordinate values
        # The global attributes may have incorrect dates from the Icechunk store creation
        if "time" in ds.coords:
            time_coord = ds.coords["time"]
            if len(time_coord) > 0:  # pyright: ignore[reportUnknownArgumentType]
                # Get first and last time values
                # Handle both pandas timestamps and cftime objects (for dates outside pandas range)
                time_values = time_coord.to_numpy()  # pyright: ignore[reportUnknownMemberType]
                first_val = time_values[0]
                last_val = time_values[-1]

                # Convert to datetime objects
                # cftime objects and pandas Timestamp both support to_pydatetime()
                if hasattr(first_val, "to_pydatetime"):
                    first_dt = first_val.to_pydatetime()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    last_dt = last_val.to_pydatetime()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                elif hasattr(first_val, "isoformat"):
                    # Fallback: parse from isoformat string
                    first_dt = datetime.fromisoformat(first_val.isoformat())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    last_dt = datetime.fromisoformat(last_val.isoformat())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                else:
                    # Last resort: convert via pandas
                    first_dt = cast("datetime", pd.Timestamp(first_val).to_pydatetime())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                    last_dt = cast("datetime", pd.Timestamp(last_val).to_pydatetime())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

                # Detect temporal resolution and adjust centered timestamps
                temporal_resolution = _detect_temporal_resolution(group_name)
                first_dt = _adjust_centered_timestamp(
                    first_dt, temporal_resolution, is_end_date=False
                )
                last_dt = _adjust_centered_timestamp(last_dt, temporal_resolution, is_end_date=True)

                # Convert to ISO format string
                first_time_str = first_dt.isoformat()
                last_time_str = last_dt.isoformat()

                # Override in global attributes
                zarr_metadata[".zattrs"]["time_coverage_start"] = first_time_str
                zarr_metadata[".zattrs"]["time_coverage_end"] = last_time_str

        # Build result
        result = {
            "uri": {
                "icechunk": store_uri,
                "parquet": None,  # Icechunk doesn't use parquet
                "json": None,
            },
            "zarr_metadata": zarr_metadata,
        }

    except Exception:
        log.exception("Failed to extract metadata from %s/%s", dataset_name, group_name)
        return None
    else:
        log.info("Successfully extracted metadata from %s/%s", dataset_name, group_name)
        return result


def generate_metadata_json(
    stores: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Generate metadata dictionary from discovered Icechunk stores.

    Args:
        stores: Dictionary of {dataset_name: {group: s3_path}}

    Returns:
        Complete metadata dictionary
    """
    log.info("Generating metadata JSON for %d datasets", len(stores))

    metadata: dict[str, Any] = {}

    for dataset_name, groups in stores.items():
        metadata[dataset_name] = {}

        for group_name, store_uri in groups.items():
            store_metadata = extract_metadata_from_store(store_uri, dataset_name, group_name)

            if store_metadata is not None:
                metadata[dataset_name][group_name] = store_metadata
            else:
                log.warning("Skipping %s/%s due to extraction failure", dataset_name, group_name)

    total_processed = sum(len(groups) for groups in metadata.values())
    log.info("Metadata generation complete. Processed %d stores.", total_processed)
    return metadata


def main() -> None:
    """Main execution function."""
    log.info("Starting Icechunk metadata extraction")

    # Initialize S3 client
    s3_client = s3fs.S3FileSystem()

    # Discover all icechunk stores
    stores = discover_icechunk_stores(s3_client)

    if not stores:
        log.error("No Icechunk stores found!")
        return

    # Extract metadata
    metadata = generate_metadata_json(stores)

    # Write to file for standalone use
    Path(OUTPUT_FILE).write_text(json.dumps(metadata, indent=2))
    log.info("Metadata saved to %s", OUTPUT_FILE)

    log.info("Extraction complete!")


if __name__ == "__main__":
    main()
