"""Build knowledge graph from dataset metadata (legacy module).

Note: This module contains legacy code with broken imports. Consider using the
individual builders (build_graph_noaa_cdr, build_graph_crw, build_graph_dynamical_org) instead.
"""

import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from rdflib import (
    DCAT,
    DCTERMS,
    OWL,
    QB,
    RDF,
    RDFS,
    SDO,
    TIME,
    XSD,
    BNode,
    Graph,
    Literal,
    URIRef,
)

from build_knowledge_graph_pipeline.common import (
    BASE,
    CF,
    DIM_ALT_NAMES,
    QUDT,
    UNIT,
)


class Dimension(BaseModel):
    """A dimension in the metadata store."""

    id: str
    standard_name: str
    long_name: str | None = None
    units: str | None = None


class Variable(BaseModel):
    """A variable in the metadata store."""

    id: str
    standard_name: str
    field_name: str
    long_name: str
    units: str | None
    dims: list[Dimension]


class Dataset(BaseModel):
    """A dataset in the metadata store."""

    id: str
    title: str | None
    summary: str | None
    uri: str
    temporal_resolution: str | None
    start_date: datetime
    end_date: datetime | None
    variables: list[Variable]
    keywords: list[str]
    dims: list[Dimension]


def datetime_from_metadata_str(string: str) -> datetime:
    """Build datetime from metadata datetime str.

    If no timezone is provided (which seems common), assume UTC.
    """
    d = datetime.fromisoformat(string)

    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)

    return d


def parse(
    dataset_name: str,
    dataset_group: str,
    info: dict[str, Any],
    custom_cdr_var_mappings: dict[str, dict[str, str]],
) -> Dataset:
    """Create Dataset from known metadata retrieved from S3."""
    collection_metadata: dict[str, Any] = info["zarr_metadata"][".zattrs"]

    # CDR: Climate data record
    # CDR variable: A variable measured in a CDR dataset
    if (
        dataset_name in custom_cdr_var_mappings
        and dataset_group in custom_cdr_var_mappings[dataset_name]
    ):
        cdr_variable = custom_cdr_var_mappings[dataset_name][dataset_group]
    else:
        cdr_variable = collection_metadata["cdr_variable"]

    # Variable names in the underlying data files
    var_names_in_data = [var_name.strip() for var_name in cdr_variable.split(",")]
    vars_metadata = {
        var_name: info["zarr_metadata"][f"{var_name}/.zattrs"] for var_name in var_names_in_data
    }

    # Dimension name in underlying data files
    dim_names = list({d for vm in vars_metadata.values() for d in vm["_ARRAY_DIMENSIONS"]})

    # Dimension metdata from Zarr file
    dims_metadata = {
        dim_name: info["zarr_metadata"][f"{dim_name}/.zattrs"] for dim_name in dim_names
    }

    # Full dimensions: {dim_name: Dimension}
    dims = {
        dim_name: Dimension(id=f"{dim_name}_{dataset_name}_{dataset_group}", **dim_metadata)
        for dim_name, dim_metadata in dims_metadata.items()
    }

    # Build variables
    variables: list[Variable] = []
    for var_name_in_data, variable_metadata in vars_metadata.items():
        std_name = variable_metadata["standard_name"].replace(" ", "_")
        v = Variable(
            id=f"{std_name}_{dataset_name}_{dataset_group}",
            standard_name=std_name,
            field_name=var_name_in_data,
            long_name=variable_metadata["long_name"],
            units=variable_metadata["units"],
            dims=[dims[dim_name] for dim_name in variable_metadata["_ARRAY_DIMENSIONS"]],
        )
        variables.append(v)

    end_date_str = collection_metadata.get("time_coverage_end")
    end_date = datetime_from_metadata_str(end_date_str) if isinstance(end_date_str, str) else None

    keywords_str = collection_metadata.get("keywords", "")
    keywords = re.split(r"\s*,\s", keywords_str) if isinstance(keywords_str, str) else []

    return Dataset(
        id=f"{dataset_name}_{dataset_group}",
        title=collection_metadata.get("title"),
        summary=collection_metadata.get("summary"),
        uri=info["uri"]["parquet"],
        temporal_resolution=collection_metadata.get("time_coverage_resolution"),
        start_date=datetime_from_metadata_str(collection_metadata["time_coverage_start"]),
        end_date=end_date,
        variables=variables,
        keywords=keywords,
        dims=list(dims.values()),
    )


def add_variable(g: Graph, variable: Variable) -> None:
    """Add a variable to the graph.

    NOTE: Some variable stuff logically groups better under dataset, and is
    therefore in that function.
    """
    var_uri = BASE[variable.id]

    # RDF
    g.add((var_uri, RDF.type, SDO.PropertyValue))
    g.add((var_uri, RDF.type, QB.MeasureProperty))

    # schema.org
    g.add((var_uri, SDO.name, Literal(variable.standard_name)))
    g.add((var_uri, SDO.description, Literal(variable.long_name)))
    g.add((var_uri, SDO.alternateName, Literal(variable.field_name)))
    if variable.units:
        g.add((var_uri, SDO.unitText, Literal(variable.units)))
    g.add((var_uri, SDO.propertyID, QB.measure))

    # RDFS
    g.add((var_uri, RDFS.label, Literal(variable.long_name)))

    # CF
    g.add((var_uri, BASE.cf_standard_name, CF[variable.standard_name]))


def add_dataset(g: Graph, dataset: Dataset) -> None:  # noqa: C901, PLR0912
    """Add a dataset to the graph."""
    for variable in dataset.variables:
        add_variable(g, variable)

    ds_uri = BASE[dataset.id]

    # Add RDF values
    g.add((ds_uri, RDF.type, DCAT.Dataset))
    g.add((ds_uri, RDF.type, SDO.Dataset))
    g.add((ds_uri, RDF.type, QB.DataSet))

    # Add schema.org values
    g.add((ds_uri, SDO.version, Literal("2")))
    if dataset.title:
        g.add((ds_uri, SDO.name, Literal(dataset.title)))
    if dataset.summary:
        g.add((ds_uri, SDO.description, Literal(dataset.summary)))
    g.add((ds_uri, SDO.contentUrl, URIRef(dataset.uri)))
    g.add((ds_uri, SDO.isAccessibleForFree, Literal(True)))  # noqa: FBT003

    for keyword in dataset.keywords:
        g.add((ds_uri, SDO.keywords, Literal(keyword)))
    for dimension in dataset.dims:
        dim_iri = BASE[dimension.standard_name]
        g.add((ds_uri, SDO.variableMeasured, dim_iri))
    for variable in dataset.variables:
        g.add((ds_uri, SDO.variableMeasured, BASE[variable.id]))

    # DC Terms
    if dataset.title:
        g.add((ds_uri, DCTERMS.title, Literal(dataset.title)))
    if dataset.summary:
        g.add((ds_uri, DCTERMS.description, Literal(dataset.summary)))

    # QB (Data Cube)
    for variable in dataset.variables:
        variable_structure_blank_node = BNode()
        g.add((variable_structure_blank_node, RDF.type, QB.DataStructureDefinition))

        for dimension in variable.dims:
            dimension_blank_node = BNode()
            dim_iri = BASE[dimension.standard_name]
            g.add((dimension_blank_node, QB.dimension, dim_iri))
            g.add((variable_structure_blank_node, QB.component, dimension_blank_node))

        variable_blank_node = BNode()
        g.add((variable_blank_node, QB.measure, BASE[variable.id]))
        g.add((variable_structure_blank_node, QB.component, variable_blank_node))
        g.add((ds_uri, QB.structure, variable_structure_blank_node))

    # RDFS
    if dataset.title:
        g.add((ds_uri, RDFS.label, Literal(dataset.title)))
    if dataset.summary:
        g.add((ds_uri, RDFS.comment, Literal(dataset.summary)))

    # DCAT
    dcat_distribution_blank_node = BNode()
    g.add((dcat_distribution_blank_node, RDF.type, DCAT.Distribution))
    g.add((dcat_distribution_blank_node, DCAT.accessURL, URIRef(dataset.uri)))
    g.add((dcat_distribution_blank_node, DCAT.mediaType, Literal("application/zarr")))
    g.add((ds_uri, DCAT.distribution, dcat_distribution_blank_node))

    if dataset.temporal_resolution:
        g.add(
            (
                ds_uri,
                DCAT.temporalResolution,
                Literal(dataset.temporal_resolution, datatype=XSD.duration),
            )
        )

    g.add(
        (
            ds_uri,
            DCAT.startDate,
            Literal(dataset.start_date, datatype=XSD.dateTime),
        )
    )
    g.add((ds_uri, DCAT.endDate, Literal(dataset.end_date, datatype=XSD.dateTime)))


def add_dimension(g: Graph, dim: Dimension) -> None:
    """Add a canonical dimension to the graph."""
    std_name = dim.standard_name
    dim_uri = BASE[std_name]

    # rdf
    g.add((dim_uri, RDF.type, SDO.PropertyValue))
    g.add((dim_uri, RDF.type, QB.DimensionProperty))

    # schema.org
    g.add((dim_uri, SDO.name, Literal(std_name)))
    g.add((dim_uri, SDO.alternateName, Literal(DIM_ALT_NAMES.get(std_name, std_name))))
    g.add((dim_uri, SDO.propertyID, QB.measureDimension))

    # rdfs
    g.add((dim_uri, RDFS.label, Literal(std_name)))

    # cf
    g.add((dim_uri, BASE.cf_standard_name, CF[std_name]))


def build_graph(datasets: list[Dataset]) -> Graph:
    """Build a graph from a list of datasets."""
    g = Graph()

    # bind prefixes
    g.bind("", BASE)
    g.bind("cf", CF)
    g.bind("dcat", DCAT)
    g.bind("dcterms", DCTERMS)
    g.bind("owl", OWL)
    g.bind("qb", QB)
    g.bind("qudt", QUDT)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("schema", SDO)
    g.bind("time", TIME)
    g.bind("unit", UNIT)
    g.bind("xsd", XSD)

    # Custom CF information
    cf_standard_name = BASE["cf_standard_name"]
    g.add((cf_standard_name, RDF.type, RDF.Property))
    g.add(
        (
            cf_standard_name,
            RDFS.label,
            Literal("Matching entry in the CF Standard Names table."),
        )
    )
    g.add((cf_standard_name, RDFS.range, CF.Standard_Name))

    added_dimensions_names: set[str] = set()
    for dataset in datasets:
        for dimension in dataset.dims:
            if dimension.standard_name not in added_dimensions_names:
                add_dimension(g, dimension)

        add_dataset(g, dataset)

    return g
