"""Metadata parsing utilities for Zarr datasets.

Provides Pydantic models and parsing functions to extract structured metadata
from Zarr dataset attributes stored in S3.
"""

# ruff: noqa: D101, D102, D103, S101
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel


class Dimension(BaseModel):
    id: str
    standard_name: str
    field_name: str
    alternate_name: str | None = None
    long_name: str | None = None
    units: str | None = None


class CanonicalDimension(BaseModel):
    standard_name: str
    alternate_names: set[str]

    @classmethod
    def from_dimensions(cls, dims: list[Dimension]) -> "CanonicalDimension":
        assert len({dim.standard_name for dim in dims}) == 1
        alternate_names = {dim.alternate_name for dim in dims if dim.alternate_name is not None}
        field_names = {dim.field_name for dim in dims}
        standard_name = dims[0].standard_name
        return cls(
            standard_name=standard_name,
            alternate_names=alternate_names | field_names - {standard_name},
        )


class Variable(BaseModel):
    id: str
    standard_name: str | None
    field_name: str
    long_name: str
    units: str | None = None
    dims: list[Dimension]
    description: str | None = None


class Dataset(BaseModel):
    id: str
    title: str | None = None
    summary: str | None = None
    uri: str
    temporal_resolution: str | timedelta | None
    start_date: datetime
    end_date: datetime
    variables: list[Variable]
    keywords: set[str]
    dims: list[Dimension]
    metadata_link: str | None


def build_variable_description(variable_attrs: dict[str, Any]) -> str:
    existing_fields = {"_ARRAY_DIMENSIONS", "units", "standard_name"}
    # Xarray automatically applies scale_factor and add_offset when decoding data.
    # We remove these from the description to avoid confusing the LLM.
    # _FillValue is also a zarr/xarray internal field that gets automatically handled.
    possibly_confusing_fields = {
        "scale_factor",
        "add_offset",
        "valid_min",
        "valid_max",
        "_FillValue",
    }
    v_attrs = {
        k: v
        for k, v in variable_attrs.items()
        if k not in existing_fields | possibly_confusing_fields
    }
    return f"Other metadata:\n{'\n'.join(f'{k}: {v}' for k, v in sorted(v_attrs.items()))}"


def parse(
    dataset_name: str,
    dataset_group: str,
    info: dict[str, Any],
    get_variable_field_names_func: Callable[[str, str, dict[str, Any]], list[str]],
    get_metadata_link_func: Callable[[str, str, dict[str, Any]], str | None],
) -> Dataset:
    collection_metadata: dict[str, Any] = info["zarr_metadata"][".zattrs"]
    var_names_in_data = get_variable_field_names_func(
        dataset_name, dataset_group, collection_metadata
    )
    variable_attrs = {
        var_name: info["zarr_metadata"][f"{var_name}/.zattrs"] for var_name in var_names_in_data
    }

    dims = parse_dims(dataset_name, dataset_group, info["zarr_metadata"], variable_attrs)
    variables = parse_variables(dataset_name, dataset_group, variable_attrs, dims)

    metadata_link = get_metadata_link_func(dataset_name, dataset_group, collection_metadata)

    return Dataset(
        id=f"{dataset_name}_{dataset_group}",
        title=collection_metadata.get("title"),
        summary=collection_metadata.get("summary"),
        uri=info["uri"]["icechunk"],
        temporal_resolution=collection_metadata.get("time_coverage_resolution"),
        start_date=datetime.fromisoformat(collection_metadata["time_coverage_start"]).replace(
            tzinfo=UTC
        ),
        end_date=datetime.fromisoformat(collection_metadata["time_coverage_end"]).replace(
            tzinfo=UTC
        ),
        variables=variables,
        keywords=set(re.split(r"\s*,\s*", collection_metadata.get("keywords", ""))),
        dims=list(dims.values()),
        metadata_link=metadata_link,
    )


def parse_variables(
    dataset_name: str,
    dataset_group: str,
    variable_attrs: dict[str, Any],
    dims: dict[str, Dimension],
) -> list[Variable]:
    variables: list[Variable] = []
    for var_name_in_data, v_metadata in variable_attrs.items():
        std_name = (
            v_metadata["standard_name"].replace(" ", "_") if "standard_name" in v_metadata else None
        )
        v = Variable(
            id=f"{std_name or var_name_in_data}_{dataset_name}_{dataset_group}",
            standard_name=std_name,
            field_name=var_name_in_data,
            long_name=v_metadata["long_name"],
            units=v_metadata["units"],
            dims=[dims[dim_name] for dim_name in v_metadata["_ARRAY_DIMENSIONS"]],
            description=build_variable_description(v_metadata),
        )
        variables.append(v)
    return variables


def parse_dims(
    dataset_name: str,
    dataset_group: str,
    zarr_metadata: dict[str, Any],
    variable_attrs: dict[str, Any],
) -> dict[str, Dimension]:
    dim_names = list(
        {d for v_attrs in variable_attrs.values() for d in v_attrs["_ARRAY_DIMENSIONS"]}
    )
    dims_metadata = {dim_name: zarr_metadata[f"{dim_name}/.zattrs"] for dim_name in dim_names}
    return {
        dim_name: Dimension(
            id=f"{dim_name}_{dataset_name}_{dataset_group}",
            **dim_metadata,
            field_name=dim_name,
        )
        for dim_name, dim_metadata in dims_metadata.items()
    }
