"""Build knowledge graph for NOAA Climate Data Records (CDR) datasets.

This module extracts metadata from NOAA CDR Icechunk stores and builds
an RDF knowledge graph using DCAT, Schema.org, and other standard vocabularies.
"""

import re
from typing import Any

from build_knowledge_graph_pipeline.graph_construction import build_graph, save_graph
from build_knowledge_graph_pipeline.metadata_parsing import Dataset, parse

# Custom variable mappings for datasets where cdr_variable doesn't match actual variable names
CUSTOM_CDR_VAR_MAPPINGS = {
    "noaa-cdr-aerosol-optical-thickness-pds": {
        "daily": "aot1",
        "monthly": "aot1",
    },
    "noaa-cdr-precip-gpcp-daily-pds": {
        "daily": "precip",
    },
    "noaa-cdr-precip-gpcp-monthly-pds": {
        "monthly": "precip",
    },
    "noaa-cdr-precip-cmorph-pds": {
        "daily": "cmorph",
        "hourly": "cmorph",
    },
}

# Dimension alternate names for canonical dimensions
DIM_ALT_NAMES = {
    "latitude": "lat",
    "longitude": "lon",
    "radiation_wavelength": "wavelength",
}

# Fixed landing page links for datasets with incorrect metadata_link values
METADATA_LINK_OVERRIDES = {
    "noaa-cdr-precip-gpcp-daily-pds": {"daily": "gov.noaa.ncdc:C00999"},
}


def _get_variable_field_names(
    dataset_name: str, dataset_group: str, collection_metadata: dict[str, Any]
) -> list[str]:
    """Get the actual variable names in the dataset.

    Some datasets have a cdr_variable attribute that doesn't match the actual
    variable names in the data. This function handles those edge cases.

    Args:
        dataset_name: Name of the dataset (e.g., 'noaa-cdr-aerosol-optical-thickness-pds')
        dataset_group: Dataset group/frequency (e.g., 'daily', 'monthly')
        collection_metadata: Zarr .zattrs metadata dictionary

    Returns:
        List of variable field names to extract from the dataset
    """
    try:
        cdr_variable = CUSTOM_CDR_VAR_MAPPINGS[dataset_name][dataset_group]
    except KeyError:
        cdr_variable = collection_metadata["cdr_variable"]
    return [var_name.strip() for var_name in cdr_variable.split(",")]


def _get_metadata_link(
    dataset_name: str, dataset_group: str, collection_metadata: dict[str, Any]
) -> str | None:
    """Get the landing page URL for the dataset.

    Handles overrides for datasets with incorrect metadata_link values and
    converts NOAA metadata IDs to full landing page URLs.

    Args:
        dataset_name: Name of the dataset
        dataset_group: Dataset group/frequency
        collection_metadata: Zarr .zattrs metadata dictionary

    Returns:
        Full landing page URL or None if not available
    """
    # Check for manual overrides first
    if METADATA_LINK_OVERRIDES:
        try:
            metadata_link = METADATA_LINK_OVERRIDES[dataset_name][dataset_group]
        except KeyError:
            metadata_link = collection_metadata.get("metadata_link")
    else:
        metadata_link = collection_metadata.get("metadata_link")

    # Convert NOAA metadata IDs to full URLs
    if metadata_link and re.match(r"gov\.noaa.+:.+", metadata_link):
        metadata_link = (
            f"https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id={metadata_link}"
        )

    return metadata_link


def build_noaa_cdr_graph(output_path: str, metadata: dict[str, Any]) -> None:
    """Build knowledge graph from NOAA CDR dataset metadata.

    Parses dataset information from provided metadata, builds an RDF graph,
    and saves it to the specified output path.

    Args:
        output_path: File path where the turtle (.ttl) file will be saved
        metadata: Icechunk metadata dict containing dataset information
    """
    ds_metadata = metadata

    # Parse all datasets
    datasets: list[Dataset] = []
    for ds_name, groups in ds_metadata.items():
        for group_name, group_data in groups.items():
            # Skip datasets without Icechunk URI
            if group_data["uri"]["icechunk"] is None:
                continue

            try:
                dataset = parse(
                    ds_name,
                    group_name,
                    group_data,
                    get_variable_field_names_func=_get_variable_field_names,
                    get_metadata_link_func=_get_metadata_link,
                )
                datasets.append(dataset)
            except KeyError:
                # Skip datasets that fail parsing
                continue

    # Build RDF graph
    graph = build_graph(
        datasets,
        canonical_dims=True,
        dim_alt_names=DIM_ALT_NAMES,
        media_type="application/icechunk-zarr",
    )

    # Save to file
    save_graph(graph, output_path)
