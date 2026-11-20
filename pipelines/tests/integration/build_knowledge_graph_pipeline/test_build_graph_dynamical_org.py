"""Integration tests for dynamical.org knowledge graph builder.

Tests that the dynamical.org knowledge graph builder produces identical RDF output
to the reference graph generated from the original notebook. Uses real HTTPS Zarr
metadata fetching but mocks LLM calls for determinism and speed.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr
from rdflib import Graph

from build_knowledge_graph_pipeline.build_graph_dynamical_org import (
    DatasetTemporalInfo,
    ForecastDatasetTemporalInfo,
    HistoricalDatasetTemporalInfo,
    build_dynamical_org_graph,
)

# Path to expected reference file
EXPECTED_FILE = (
    Path(__file__).parent.parent.parent / "data" / "dataset-kg-dynamical-org-expected.ttl"
)


@pytest.fixture
def mock_temporal_responses() -> dict[str, DatasetTemporalInfo]:
    """Provide fixed temporal metadata extracted from reference TTL.

    Returns mapping of dataset_id -> temporal information for mocking LLM responses.
    Values extracted from dataset-kg-dynamical-org-expected.ttl.
    """
    return {
        # Forecast datasets (with lead_times)
        "ecmwf-ifs-ens-forecast-15-day-0-25-degree": DatasetTemporalInfo(
            root=ForecastDatasetTemporalInfo(
                start_date=datetime(2024, 4, 1, tzinfo=UTC),
                end_date=datetime(2026, 9, 6, 17, 31, 15, 134352, tzinfo=UTC),
                temporal_resolution=timedelta(days=1),
                lead_times=[timedelta(hours=h) for h in range(0, 361, 6)],
            )
        ),
        "noaa-gefs-forecast-35-day": DatasetTemporalInfo(
            root=ForecastDatasetTemporalInfo(
                start_date=datetime(2021, 5, 1, tzinfo=UTC),
                end_date=datetime(2026, 9, 6, 17, 30, 50, tzinfo=UTC),
                temporal_resolution=timedelta(hours=6),
                lead_times=[timedelta(hours=h) for h in range(0, 841, 6)],
            )
        ),
        "noaa-gfs-forecast": DatasetTemporalInfo(
            root=ForecastDatasetTemporalInfo(
                start_date=datetime(2021, 5, 1, tzinfo=UTC),
                end_date=datetime(2026, 9, 6, 17, 30, 50, 830624, tzinfo=UTC),
                temporal_resolution=timedelta(hours=6),
                lead_times=[timedelta(hours=h) for h in range(0, 385, 6)],
            )
        ),
        "noaa-hrrr-forecast-48-hour": DatasetTemporalInfo(
            root=ForecastDatasetTemporalInfo(
                start_date=datetime(2018, 7, 13, 12, tzinfo=UTC),
                end_date=datetime(2026, 9, 6, 17, 31, 8, 908381, tzinfo=UTC),
                temporal_resolution=timedelta(hours=6),
                lead_times=[timedelta(hours=h) for h in range(0, 49, 1)],
            )
        ),
        # Historical datasets (no lead_times)
        "noaa-gefs-analysis": DatasetTemporalInfo(
            root=HistoricalDatasetTemporalInfo(
                start_date=datetime(2000, 1, 1, tzinfo=UTC),
                end_date=datetime(2026, 9, 6, 17, 30, 36, 207152, tzinfo=UTC),
                temporal_resolution=timedelta(hours=3),
            )
        ),
        "noaa-gfs-analysis-hourly": DatasetTemporalInfo(
            root=HistoricalDatasetTemporalInfo(
                start_date=datetime(2015, 1, 15, tzinfo=UTC),
                end_date=datetime(2024, 7, 1, tzinfo=UTC),
                temporal_resolution=timedelta(hours=1),
            )
        ),
        "noaa-hrrr-analysis": DatasetTemporalInfo(
            root=HistoricalDatasetTemporalInfo(
                start_date=datetime(2014, 10, 1, tzinfo=UTC),
                end_date=datetime(2026, 9, 6, 17, 30, 39, 383465, tzinfo=UTC),
                temporal_resolution=timedelta(hours=1),
            )
        ),
    }


@pytest.mark.integration
@patch("build_knowledge_graph_pipeline.build_graph_dynamical_org._extract_temporal_info_from_ds")
def test_build_dynamical_org_graph_matches_expected(
    mock_llm_extract: MagicMock,
    mock_temporal_responses: dict[str, DatasetTemporalInfo],
    tmp_path: Path,
) -> None:
    """Test that dynamical.org builder produces identical RDF to reference file.

    This integration test:
    1. Fetches real Zarr metadata from dynamical.org via HTTPS
    2. Mocks LLM temporal extraction with fixed responses from reference TTL
    3. Builds the knowledge graph using build_dynamical_org_graph()
    4. Compares the generated RDF graph to the reference file
    5. Asserts graph isomorphism (same triples, ignoring blank node IDs)

    Args:
        mock_llm_extract: Mocked _extract_temporal_info_from_ds function
        mock_temporal_responses: Fixture providing temporal metadata
        tmp_path: Pytest fixture providing temporary directory for output
    """

    # Configure mock to return appropriate temporal info based on dataset attrs
    def mock_extract(ds: xr.Dataset) -> DatasetTemporalInfo:
        # Get dataset_id from attrs (try both 'dataset_id' and 'id' fields)
        dataset_id = ds.attrs.get("dataset_id") or ds.attrs.get("id")

        # Return corresponding mock temporal info
        if dataset_id in mock_temporal_responses:
            return mock_temporal_responses[dataset_id]

        # Fallback for unexpected dataset_id
        msg = f"Unexpected dataset_id: {dataset_id}"
        raise ValueError(msg)

    mock_llm_extract.side_effect = mock_extract

    # Generate graph to temp file (fetches real data, uses mocked LLM)
    output_file = tmp_path / "dynamical_org_generated.ttl"
    build_dynamical_org_graph(str(output_file))

    # Verify output file was created
    assert output_file.exists(), "Output TTL file was not created"

    # Load generated and expected graphs
    generated_graph = Graph()
    generated_graph.parse(output_file, format="turtle")

    expected_graph = Graph()
    expected_graph.parse(EXPECTED_FILE, format="turtle")

    # Assert graphs are isomorphic (same triples, ignoring blank node IDs and ordering)
    assert generated_graph.isomorphic(expected_graph), (
        f"Generated graph differs from expected. "
        f"Generated has {len(generated_graph)} triples, "
        f"expected has {len(expected_graph)} triples."
    )
