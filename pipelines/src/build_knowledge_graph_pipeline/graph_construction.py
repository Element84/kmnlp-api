"""RDF knowledge graph construction utilities.

Provides functions to build RDF graphs from dataset metadata using standard
vocabularies (DCAT, Schema.org, QB, DCTERMS, etc.) and CF standard names.
"""

# ruff: noqa: D103, S101, ERA001

from pathlib import Path
from typing import Literal

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import requests
from defusedxml.ElementTree import parse
from rdflib import DCTERMS, RDF, RDFS, BNode, Graph, Namespace, URIRef
from rdflib import Literal as RdfLiteral

# Import from private modules - these are stable but not officially exported
from rdflib.namespace import (  # pyright: ignore[reportPrivateImportUsage]
    DCAT,  # pyright: ignore[reportPrivateImportUsage]
    OWL,  # pyright: ignore[reportPrivateImportUsage]
    QB,  # pyright: ignore[reportPrivateImportUsage]
    SDO,  # pyright: ignore[reportPrivateImportUsage]
    TIME,  # pyright: ignore[reportPrivateImportUsage]
    XSD,  # pyright: ignore[reportPrivateImportUsage]
)

from build_knowledge_graph_pipeline.metadata_parsing import (
    CanonicalDimension,
    Dataset,
    Dimension,
    Variable,
)

# Define Namespaces
BASE = Namespace("http://kmnlp-demo.noaa.gov/")
QUDT = Namespace("http://qudt.org/3.1.1/schema/qudt/")
UNIT = Namespace("http://qudt.org/3.1.1/vocab/unit/")
CF = Namespace("http://mmisw.org/ont/cf/parameter/")


def get_cf_standard_names(path: str) -> list[str]:
    path_obj = Path(path)
    if not path_obj.exists():
        # Download
        url = "https://cfconventions.org/Data/cf-standard-names/76/src/cf-standard-name-table.xml"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_bytes(response.content)

    cf_names_tree = parse(str(path_obj))
    cf_names_root = cf_names_tree.getroot()
    assert cf_names_root is not None
    # collect `id` attribute from all entries
    return [entry.get("id") for entry in cf_names_root.findall("entry")]  # pyright: ignore[reportReturnType]


CF_STANDARD_NAMES = set(get_cf_standard_names("./data/cf_standard_names.xml"))


def add_dataset(  # noqa: C901, PLR0912, PLR0915
    g: Graph,
    dataset: Dataset,
    *,
    canonical_dims: bool = False,
    dim_alt_names: dict[str, str],
    media_type: Literal["application/kerchunk-parquet", "application/icechunk-zarr"],
) -> str:
    """Add a dataset to the graph."""
    # add dims
    if not canonical_dims:
        for d in dataset.dims:
            add_dim(g, d, dim_alt_names=dim_alt_names)

    # add vars
    for v in dataset.variables:
        add_variable(g, v)

    ds_uri = BASE[dataset.id]

    # rdf
    g.add((ds_uri, RDF.type, DCAT.Dataset))
    g.add((ds_uri, RDF.type, SDO.Dataset))
    g.add((ds_uri, RDF.type, QB.DataSet))

    # schema.org properties
    g.add((ds_uri, SDO.version, RdfLiteral("2")))
    if dataset.title:
        g.add((ds_uri, SDO.name, RdfLiteral(dataset.title)))
    if dataset.summary:
        g.add((ds_uri, SDO.description, RdfLiteral(dataset.summary)))

    if dataset.metadata_link:
        g.add((ds_uri, SDO.url, URIRef(dataset.metadata_link)))

    g.add((ds_uri, SDO.contentUrl, URIRef(dataset.uri)))
    g.add((ds_uri, SDO.isAccessibleForFree, RdfLiteral(True)))  # noqa: FBT003

    for k in dataset.keywords:
        g.add((ds_uri, SDO.keywords, RdfLiteral(k)))
    for dim in dataset.dims:
        dim_iri = BASE[dim.standard_name if canonical_dims else dim.id]
        g.add((ds_uri, SDO.variableMeasured, dim_iri))
    for v in dataset.variables:
        g.add((ds_uri, SDO.variableMeasured, BASE[v.id]))

    # dcterms
    if dataset.title:
        g.add((ds_uri, DCTERMS.title, RdfLiteral(dataset.title)))
    if dataset.summary:
        g.add((ds_uri, DCTERMS.description, RdfLiteral(dataset.summary)))

    # qb
    for v in dataset.variables:
        v_structure_def_bnode = BNode()
        g.add((v_structure_def_bnode, RDF.type, QB.DataStructureDefinition))
        for dim in v.dims:
            qb_dim_bnode = BNode()
            g.add((qb_dim_bnode, RDF.type, QB.ComponentSpecification))
            dim_iri = BASE[dim.standard_name if canonical_dims else dim.id]
            g.add((qb_dim_bnode, QB.dimension, dim_iri))
            g.add((qb_dim_bnode, BASE.field_name, RdfLiteral(dim.field_name)))
            g.add((v_structure_def_bnode, QB.component, qb_dim_bnode))
        qb_var_bnode = BNode()
        g.add((qb_var_bnode, QB.measure, BASE[v.id]))
        g.add((v_structure_def_bnode, QB.component, qb_var_bnode))
        g.add((ds_uri, QB.structure, v_structure_def_bnode))

    # rdfs
    if dataset.title:
        g.add((ds_uri, RDFS.label, RdfLiteral(dataset.title)))
    if dataset.summary:
        g.add((ds_uri, RDFS.comment, RdfLiteral(dataset.summary)))

    # dcat
    dcat_dist_bnode = BNode()
    g.add((dcat_dist_bnode, RDF.type, DCAT.Distribution))
    g.add((dcat_dist_bnode, DCAT.accessURL, URIRef(dataset.uri)))
    g.add((dcat_dist_bnode, DCAT.mediaType, RdfLiteral(media_type)))
    g.add((ds_uri, DCAT.distribution, dcat_dist_bnode))
    if dataset.temporal_resolution:
        if isinstance(dataset.temporal_resolution, str):
            duration_str = dataset.temporal_resolution
        else:
            duration_str = pd.Timedelta(dataset.temporal_resolution).isoformat()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAttributeAccessIssue]
        g.add((ds_uri, DCAT.temporalResolution, RdfLiteral(duration_str, datatype=XSD.duration)))
    g.add((ds_uri, DCAT.startDate, RdfLiteral(dataset.start_date, datatype=XSD.dateTime)))
    g.add((ds_uri, DCAT.endDate, RdfLiteral(dataset.end_date, datatype=XSD.dateTime)))

    return ds_uri


def add_variable(g: Graph, variable: Variable) -> str:
    """Add a variable to the graph."""
    var_uri = BASE[variable.id]

    # rdf
    g.add((var_uri, RDF.type, SDO.PropertyValue))
    g.add((var_uri, RDF.type, QB.MeasureProperty))

    # schema.org properties
    if variable.standard_name:
        g.add((var_uri, SDO.name, RdfLiteral(variable.standard_name)))
    if variable.description:
        g.add((var_uri, SDO.description, RdfLiteral(variable.description)))
    g.add((var_uri, SDO.alternateName, RdfLiteral(variable.field_name)))
    if variable.units:
        g.add((var_uri, SDO.unitText, RdfLiteral(variable.units)))
    g.add((var_uri, SDO.propertyID, QB.measure))

    # rdfs
    g.add((var_uri, RDFS.label, RdfLiteral(variable.long_name)))

    # qudt
    # g.add((var_uri, QUDT.hasUnit, UNIT[""]))

    # cf
    if variable.standard_name and variable.standard_name in CF_STANDARD_NAMES:
        g.add((var_uri, BASE.cf_standard_name, CF[variable.standard_name]))

    g.add((var_uri, BASE.field_name, RdfLiteral(variable.field_name)))

    return var_uri


def add_dim(g: Graph, dim: Dimension, dim_alt_names: dict[str, str]) -> str:
    """Add a dimension to the graph."""
    std_name = dim.standard_name
    dim_uri = BASE[dim.id]

    # rdf
    g.add((dim_uri, RDF.type, SDO.PropertyValue))
    g.add((dim_uri, RDF.type, QB.DimensionProperty))

    # schema.org
    if std_name in dim_alt_names:
        g.add((dim_uri, SDO.alternateName, RdfLiteral(dim_alt_names[std_name])))
    if dim.units:
        g.add((dim_uri, SDO.unitText, RdfLiteral(dim.units)))
    g.add((dim_uri, SDO.propertyID, QB.measureDimension))

    # rdfs
    g.add((dim_uri, RDFS.label, RdfLiteral(dim.long_name)))

    # cf
    if std_name in CF_STANDARD_NAMES:
        g.add((dim_uri, BASE.cf_standard_name, CF[std_name]))

    return dim_uri


def add_dim_canonical(g: Graph, dim: CanonicalDimension) -> str:
    """Add a canonical dimension to the graph."""
    std_name = dim.standard_name
    dim_uri = BASE[std_name]

    # rdf
    g.add((dim_uri, RDF.type, SDO.PropertyValue))
    g.add((dim_uri, RDF.type, QB.DimensionProperty))

    # schema.org
    g.add((dim_uri, SDO.name, RdfLiteral(std_name)))
    for alt_name in dim.alternate_names:
        g.add((dim_uri, SDO.alternateName, RdfLiteral(alt_name)))
    g.add((dim_uri, SDO.propertyID, QB.measureDimension))

    # rdfs
    g.add((dim_uri, RDFS.label, RdfLiteral(std_name)))

    # cf
    if std_name in CF_STANDARD_NAMES:
        g.add((dim_uri, BASE.cf_standard_name, CF[std_name]))

    return dim_uri


def build_graph(
    datasets: list[Dataset],
    *,
    canonical_dims: bool = False,
    dim_alt_names: dict[str, str] | None = None,
    media_type: Literal["application/kerchunk-parquet", "application/icechunk-zarr"],
) -> Graph:
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

    # custom properties
    cf_standard_name = BASE["cf_standard_name"]
    g.add((cf_standard_name, RDF.type, RDF.Property))
    g.add(
        (cf_standard_name, RDFS.label, RdfLiteral("Matching entry in the CF Standard Names table."))
    )
    g.add((cf_standard_name, RDFS.range, CF.Standard_Name))

    field_name_prop = BASE["field_name"]
    g.add((field_name_prop, RDF.type, RDF.Property))
    g.add((field_name_prop, RDFS.label, RdfLiteral("Field name in the dataset.")))
    g.add((field_name_prop, RDFS.range, XSD.string))

    if canonical_dims:
        std_name_to_dims: dict[str, list[Dimension]] = {}
        for ds in datasets:
            for dim in ds.dims:
                if dim.standard_name not in std_name_to_dims:
                    std_name_to_dims[dim.standard_name] = []
                std_name_to_dims[dim.standard_name].append(dim)

        for dims in std_name_to_dims.values():
            add_dim_canonical(g, CanonicalDimension.from_dimensions(dims))

    for ds in datasets:
        add_dataset(
            g,
            ds,
            canonical_dims=canonical_dims,
            dim_alt_names=dim_alt_names or {},
            media_type=media_type,
        )

    return g


def save_graph(g: Graph, path: str) -> None:
    ttl_str = g.serialize(format="turtle")
    Path(path).write_text(ttl_str.rstrip() + "\n")
