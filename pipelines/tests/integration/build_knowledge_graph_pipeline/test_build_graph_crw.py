"""Integration tests for CRW knowledge graph builder.

Tests that the CRW knowledge graph builder produces identical RDF output
to the reference graph generated from the original notebook.
"""

import logging
from pathlib import Path

import pytest
import s3fs  # pyright: ignore[reportMissingTypeStubs]
from rdflib import Graph
from rdflib.compare import graph_diff

from build_knowledge_graph_pipeline.build_graph_crw import build_crw_graph
from build_knowledge_graph_pipeline.extract_icechunk_metadata import (
    discover_icechunk_stores,
    generate_metadata_json,
)

log = logging.getLogger(__name__)

# Path to expected reference file
EXPECTED_FILE = Path(__file__).parent.parent.parent / "data" / "dataset-kg-crw-expected.ttl"


def dump_nt_sorted(g: Graph):
    for line in sorted(g.serialize(format="nt").splitlines()):
        if line:
            log.info("%s", line)


@pytest.mark.integration
def test_build_crw_graph_matches_expected(tmp_path: Path) -> None:
    """Test that CRW graph builder produces identical RDF to reference file.

    This integration test:
    1. Extracts real metadata from Icechunk stores in S3
    2. Builds the knowledge graph using build_crw_graph()
    3. Compares the generated RDF graph to the reference file
    4. Asserts graph isomorphism (same triples, ignoring blank node IDs)

    Args:
        tmp_path: Pytest fixture providing temporary directory for output
    """
    # Extract metadata from Icechunk stores
    s3 = s3fs.S3FileSystem()
    icechunk_stores = discover_icechunk_stores(s3)
    metadata = generate_metadata_json(icechunk_stores)

    # Generate graph to temp file
    output_file = tmp_path / "crw_generated.ttl"
    build_crw_graph(str(output_file), metadata)

    # Verify output file was created
    assert output_file.exists(), "Output TTL file was not created"

    # Load generated and expected graphs
    generated_graph = Graph()
    generated_graph.parse(output_file, format="turtle")

    expected_graph = Graph()
    expected_graph.parse(EXPECTED_FILE, format="turtle")

    in_both, in_generated, in_expected = graph_diff(generated_graph, expected_graph)

    log.info("in both:")
    dump_nt_sorted(in_both)
    log.info("in generated: %s", in_generated)
    dump_nt_sorted(in_generated)
    log.info("in expected: %s", in_expected)
    dump_nt_sorted(in_expected)

    # Assert graphs are isomorphic (same triples, ignoring blank node IDs and ordering)
    assert generated_graph.isomorphic(expected_graph), (
        f"Generated graph differs from expected. "
        f"Generated has {len(generated_graph)} triples, "
        f"expected has {len(expected_graph)} triples."
    )
