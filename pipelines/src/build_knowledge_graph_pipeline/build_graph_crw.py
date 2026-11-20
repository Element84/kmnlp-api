"""Build knowledge graph for NOAA Coral Reef Watch (CRW) Sea Surface Temperature datasets.

This module extracts metadata from CRW SST Icechunk stores and builds
an RDF knowledge graph using DCAT, Schema.org, and other standard vocabularies.
"""

from typing import Any

from build_knowledge_graph_pipeline.graph_construction import build_graph, save_graph
from build_knowledge_graph_pipeline.metadata_parsing import Dataset, parse

# Variable field names for each CRW dataset group
VARIABLE_FIELD_NAMES = {
    "sst_daily": ["analysed_sst", "sea_ice_fraction"],
    "sst_monthly_mean": ["sea_surface_temperature", "mask"],
    "sst_monthly_min": ["sea_surface_temperature", "mask"],
    "sst_monthly_max": ["sea_surface_temperature", "mask"],
    "sst_annual_mean": ["sea_surface_temperature", "mask"],
    "sst_annual_min": ["sea_surface_temperature", "mask"],
    "sst_annual_max": ["sea_surface_temperature", "mask"],
}


def _get_variable_field_names(
    dataset_name: str,  # noqa: ARG001
    dataset_group: str,
    collection_metadata: dict[str, Any],  # noqa: ARG001
) -> list[str]:
    """Get the variable names for the CRW dataset group.

    Args:
        dataset_name: Name of the dataset (unused, kept for compatibility)
        dataset_group: Dataset group (e.g., 'sst_daily', 'sst_monthly_mean')
        collection_metadata: Zarr .zattrs metadata dictionary (unused, kept for compatibility)

    Returns:
        List of variable field names for the dataset group
    """
    return VARIABLE_FIELD_NAMES[dataset_group]


def _get_metadata_link(
    dataset_name: str,  # noqa: ARG001
    dataset_group: str,
    collection_metadata: dict[str, Any],
) -> str | None:
    """Get the landing page URL for the CRW dataset.

    Args:
        dataset_name: Name of the dataset (unused, kept for compatibility)
        dataset_group: Dataset group
        collection_metadata: Zarr .zattrs metadata dictionary

    Returns:
        Landing page URL or None if not available
    """
    # Daily SST has a specific landing page
    if dataset_group == "sst_daily":
        return "https://coralreefwatch.noaa.gov/product/5km/index.php"

    # For other groups, use metadata_link from collection metadata
    return collection_metadata.get("metadata_link")


def build_crw_graph(output_path: str, metadata: dict[str, Any]) -> None:
    """Build knowledge graph from CRW SST dataset metadata.

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
        media_type="application/icechunk-zarr",
    )

    # Save to file
    save_graph(graph, output_path)
