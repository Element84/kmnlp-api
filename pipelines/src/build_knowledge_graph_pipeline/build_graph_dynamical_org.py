"""Build knowledge graph for dynamical.org weather forecast and analysis datasets.

This module extracts metadata from dynamical.org Zarr datasets accessed via HTTPS
and builds an RDF knowledge graph. It uses LLM-powered temporal metadata extraction
to handle inconsistent metadata formats.

Data sources:
- Forecast: GFS, GEFS, HRRR, ECMWF IFS-ENS
- Historical: GFS analysis, GEFS analysis, HRRR analysis

More info: https://dynamical.org/catalog/
"""

import json
from datetime import UTC, datetime, timedelta
from textwrap import dedent
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

import boto3
import xarray as xr
from pydantic import AliasChoices, BaseModel, Field, RootModel
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from build_knowledge_graph_pipeline.graph_construction import build_graph, save_graph
from build_knowledge_graph_pipeline.metadata_parsing import Dataset, Dimension, Variable

# Dataset URIs from dynamical.org
FORECAST_URIS = [
    "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
    "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr",
    "https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr",
    "https://data.dynamical.org/ecmwf/ifs-ens/forecast-15-day-0-25-degree/latest.zarr",
]

HISTORICAL_URIS = [
    "https://data.dynamical.org/noaa/gfs/analysis-hourly/latest.zarr",
    "https://data.dynamical.org/noaa/gefs/analysis/latest.zarr",
    "https://data.dynamical.org/noaa/hrrr/analysis/latest.zarr",
]

# Standard names for forecast dimensions
DIM_STD_NAMES = {
    "init_time": "forecast_reference_time",
    "lead_time": "forecast_period",
    "ensemble_member": "realization",
}

# Alternate names for canonical dimensions
DIM_ALT_NAMES = {
    "latitude": "lat",
    "longitude": "lon",
    "init_time": "time",
}

# Landing page URLs for each dataset
LANDING_PAGE_OVERRIDES = {
    "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr": "https://dynamical.org/catalog/noaa-gfs-forecast/",
    "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr": "https://dynamical.org/catalog/noaa-gefs-forecast-35-day",
    "https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr": "https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour",
    "https://data.dynamical.org/ecmwf/ifs-ens/forecast-15-day-0-25-degree/latest.zarr": "https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree",
    "https://data.dynamical.org/noaa/gfs/analysis-hourly/latest.zarr": "https://dynamical.org/catalog/noaa-gfs-analysis-hourly/",
    "https://data.dynamical.org/noaa/gefs/analysis/latest.zarr": "https://dynamical.org/catalog/noaa-gefs-analysis",
    "https://data.dynamical.org/noaa/hrrr/analysis/latest.zarr": "https://dynamical.org/catalog/noaa-hrrr-analysis",
}


# Pydantic models for dynamical.org metadata


class DatasetAttrs(BaseModel):
    """Metadata attributes from dynamical.org dataset."""

    dataset_id: str = Field(validation_alias=AliasChoices("dataset_id", "id"))
    dataset_version: str | None = None
    name: str
    description: str
    attribution: str
    spatial_domain: str
    spatial_resolution: str
    time_domain: str
    time_resolution: str
    forecast_domain: str | None = None
    forecast_resolution: str | None = None


class DataVarAttrs(BaseModel):
    """Variable metadata from dynamical.org dataset."""

    long_name: str
    short_name: str | None = None
    standard_name: str | None = None
    units: str
    comment: str | None = None
    step_type: Literal["instant", "accum", "avg", "min", "max"] | None = None
    ensemble_statistic: str | None = None


class HistoricalDataset(Dataset):
    """Historical/analysis dataset (no forecast lead times)."""


class ForecastDataset(Dataset):
    """Forecast dataset with prediction lead times."""

    lead_times: list[timedelta]


class HistoricalDatasetTemporalInfo(BaseModel):
    """Temporal information for historical datasets."""

    start_date: datetime
    end_date: datetime | None
    temporal_resolution: timedelta | None


class ForecastDatasetTemporalInfo(BaseModel):
    """Temporal information for forecast datasets."""

    start_date: datetime
    end_date: datetime | None
    temporal_resolution: timedelta | None
    lead_times: list[timedelta]


class DatasetTemporalInfo(RootModel[HistoricalDatasetTemporalInfo | ForecastDatasetTemporalInfo]):
    """Union type for temporal information."""


def _ds_to_prompt(ds: xr.Dataset) -> str:
    """Convert an xarray Dataset to a string prompt for LLM input.

    Args:
        ds: xarray Dataset with metadata

    Returns:
        Formatted prompt string for LLM
    """
    out = dedent(f"""
        Dataset Metadata:
        ```json
        {json.dumps(ds.attrs)}
        ```

        Dataset Dimensions:
        ```json
        {json.dumps(list(ds.dims))}
        ```
    """)
    if "lead_time" in ds:
        out += dedent(f"""
            `lead_time` data:
            ```
            {ds["lead_time"].values.tolist()}
            ```
        """)
    return out


def _extract_temporal_info_from_ds(ds: xr.Dataset) -> DatasetTemporalInfo:
    """Extract temporal information using LLM.

    Uses Claude to parse temporal metadata from the dataset, which can be
    inconsistently formatted across different dynamical.org datasets.

    Args:
        ds: xarray Dataset with metadata

    Returns:
        Structured temporal information
    """
    bedrock_client: BedrockRuntimeClient = boto3.client("bedrock-runtime")  # pyright: ignore[reportUnknownMemberType]
    model = BedrockConverseModel(
        "us.anthropic.claude-sonnet-4-6",
        provider=BedrockProvider(bedrock_client=bedrock_client),
    )
    agent = Agent(
        model,
        output_type=str,
        instructions=dedent(
            f"""
            Extract the temporal information from the given dataset metadata into
            the output schema below. Respond with only the JSON, no markdown fencing.

            Output Schema:
            {DatasetTemporalInfo.model_json_schema()}
            """
        ),
    )
    result = agent.run_sync(_ds_to_prompt(ds))
    output = result.output.strip()
    if output.startswith("```"):
        output = output.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return DatasetTemporalInfo.model_validate_json(output)


def _extract_dataset_metadata(uri: str) -> Dataset:
    """Extract metadata from a dynamical.org Zarr dataset.

    Opens the dataset via HTTPS, parses metadata, and uses LLM to extract
    temporal information.

    Args:
        uri: HTTPS URL to the Zarr dataset

    Returns:
        Parsed Dataset object (HistoricalDataset or ForecastDataset)
    """
    ds = xr.open_zarr(uri, chunks="auto")  # pyright: ignore[reportUnknownMemberType]
    ds_metadata = DatasetAttrs.model_validate(ds.attrs)

    # Parse variable metadata
    vars_metadata: dict[str, DataVarAttrs] = {
        str(var_name): DataVarAttrs.model_validate(ds[var_name].attrs) for var_name in ds.data_vars
    }

    # Parse dimensions
    dim_names: list[str] = [str(dim) for dim in ds.dims]
    dims_out = {
        dim_name: Dimension(
            id=dim_name,
            standard_name=DIM_STD_NAMES.get(dim_name, dim_name),
            field_name=dim_name,
            alternate_name=DIM_ALT_NAMES.get(dim_name),
            long_name=ds[dim_name].attrs.get("long_name"),
            units=ds[dim_name].attrs.get("units"),
        )
        for dim_name in dim_names
    }

    # Parse variables
    vars_out = [
        Variable(
            id=f"{var_name}_{ds_metadata.dataset_id}",
            standard_name=var_metadata.standard_name,
            field_name=var_name,
            long_name=var_metadata.long_name,
            units=var_metadata.units,
            dims=[dims_out[str(dim_name)] for dim_name in ds[var_name].dims],  # pyright: ignore[reportArgumentType]
        )
        for var_name, var_metadata in vars_metadata.items()
    ]

    # Extract temporal info using LLM
    temporal_info = _extract_temporal_info_from_ds(ds)
    if temporal_info.root.end_date is None:
        temporal_info.root.end_date = datetime.now(tz=UTC)

    # Create appropriate Dataset subclass
    if isinstance(temporal_info.root, ForecastDatasetTemporalInfo):
        description_extra_items = ds_metadata.model_dump(
            include={
                "spatial_domain",
                "spatial_resolution",
                "forecast_domain",
                "forecast_resolution",
                "attribution",
            }
        )
        description_extra = "\n".join(f"{k}: {v}" for k, v in description_extra_items.items())
        expanded_description = f"{ds_metadata.description}\n\n{description_extra}"

        return ForecastDataset(
            id=ds_metadata.dataset_id,
            title=ds_metadata.name,
            summary=expanded_description,
            uri=uri,
            temporal_resolution=temporal_info.root.temporal_resolution,
            start_date=temporal_info.root.start_date.replace(tzinfo=UTC),
            end_date=temporal_info.root.end_date.replace(tzinfo=UTC),
            lead_times=temporal_info.root.lead_times,
            variables=vars_out,
            keywords=set(),
            dims=list(dims_out.values()),
            metadata_link=LANDING_PAGE_OVERRIDES.get(uri),
        )

    description_extra_items = ds_metadata.model_dump(
        include={
            "spatial_domain",
            "spatial_resolution",
            "attribution",
        }
    )
    description_extra = "\n".join(f"{k}: {v}" for k, v in description_extra_items.items())
    expanded_description = f"{ds_metadata.description}\n\n{description_extra}"

    return HistoricalDataset(
        id=ds_metadata.dataset_id,
        title=ds_metadata.name,
        summary=expanded_description,
        uri=uri,
        temporal_resolution=temporal_info.root.temporal_resolution,
        start_date=temporal_info.root.start_date.replace(tzinfo=UTC),
        end_date=temporal_info.root.end_date.replace(tzinfo=UTC),
        variables=vars_out,
        keywords=set(),
        dims=list(dims_out.values()),
        metadata_link=LANDING_PAGE_OVERRIDES.get(uri),
    )


def build_dynamical_org_graph(output_path: str) -> None:
    """Build knowledge graph from dynamical.org datasets.

    Fetches and parses metadata from forecast and historical datasets,
    builds an RDF graph, and saves it to the specified output path.

    Note: GEFS forecast dataset is excluded due to chunking issues that
    cause preprocessing to stall during lat/lon coordinate sorting.

    Args:
        output_path: File path where the turtle (.ttl) file will be saved
    """
    # Extract metadata from all datasets
    datasets: list[Dataset] = []

    # Process historical datasets
    for uri in HISTORICAL_URIS:
        dataset = _extract_dataset_metadata(uri)
        datasets.append(dataset)

    # Process forecast datasets
    for uri in FORECAST_URIS:
        dataset = _extract_dataset_metadata(uri)
        datasets.append(dataset)

    # Exclude GEFS forecast - it has chunking issues
    datasets_filtered = [ds for ds in datasets if "gefs" not in ds.id or "forecast" not in ds.id]

    # Build RDF graph
    graph = build_graph(
        datasets_filtered,
        canonical_dims=True,
        media_type="application/icechunk-zarr",
    )

    # Save to file
    save_graph(graph, output_path)
