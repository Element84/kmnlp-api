"""Build and upload knowledge graphs to S3.

Builds CRW, NOAA CDR, and dynamical.org knowledge graphs and uploads
them to S3 with static filenames.

Environment Variables (Required):
- OUTPUT_BUCKET: S3 bucket for knowledge graph outputs
- OUTPUT_PREFIX: S3 prefix/folder (must end with /)

Environment Variables (Optional):
- LOG_LEVEL: Logging level (default: INFO)

ECS Deployment Requirements:
- IAM Task Role Permissions:
  * S3: s3:GetObject, s3:PutObject, s3:ListBucket
    - Read: s3://data-c6c22a2e42294c11b52ee7f0c792c071/* (CRW, NOAA CDR metadata)
    - Write: OUTPUT_BUCKET/*
  * Bedrock: bedrock:InvokeModel for anthropic.claude-opus-4-*
    (dynamical.org builder uses Claude for temporal metadata extraction)
- Network: Requires internet access for HTTPS calls to:
  * data.dynamical.org (6+ Zarr datasets)
  * cfconventions.org (CF standard names table)
- Resources: Recommended 2 vCPU, 4 GB memory, 15-20 minute timeout
- Container Command: ["/app/.venv/bin/python", "entrypoint_kg.py"]
"""

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import s3fs  # pyright: ignore[reportMissingTypeStubs]

from build_knowledge_graph_pipeline.build_graph_crw import build_crw_graph
from build_knowledge_graph_pipeline.build_graph_dynamical_org import build_dynamical_org_graph
from build_knowledge_graph_pipeline.build_graph_noaa_cdr import build_noaa_cdr_graph
from build_knowledge_graph_pipeline.extract_icechunk_metadata import (
    discover_icechunk_stores,
    generate_metadata_json,
)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def extract_metadata() -> dict[str, Any]:
    """Extract Icechunk store metadata and return as dict."""
    log.info("Discovering and extracting Icechunk store metadata...")
    s3 = s3fs.S3FileSystem()
    icechunk_stores = discover_icechunk_stores(s3)
    log.info("Discovered %d Icechunk stores", len(icechunk_stores))

    metadata = generate_metadata_json(icechunk_stores)
    log.info("Icechunk metadata extraction complete")
    return metadata


def main() -> None:
    """Main execution function."""
    # Load environment variables
    output_bucket = os.getenv("OUTPUT_BUCKET")
    output_prefix = os.getenv("OUTPUT_PREFIX")

    # Validate required environment variables
    if not output_bucket:
        log.error("ERROR: OUTPUT_BUCKET environment variable must be set")
        sys.exit(1)

    if not output_prefix:
        log.error("ERROR: OUTPUT_PREFIX environment variable must be set")
        sys.exit(1)

    if not output_prefix.endswith("/"):
        log.error("ERROR: OUTPUT_PREFIX must end with a trailing slash (/)")
        sys.exit(1)

    log.info("Starting knowledge graph building process")
    log.info("Output bucket: %s", output_bucket)
    log.info("Output prefix: %s", output_prefix)

    try:
        # Create temporary directory for local graph files
        temp_dir = Path("/tmp/knowledge-graphs")  # noqa: S108
        temp_dir.mkdir(parents=True, exist_ok=True)
        log.info("Created temporary directory: %s", temp_dir)

        # Extract Icechunk metadata first
        metadata = extract_metadata()

        # Define graph configurations
        graphs: list[tuple[str, str, Callable[[str], None]]] = [
            ("CRW", "crw_knowledge_graph.ttl", lambda path: build_crw_graph(path, metadata)),
            (
                "NOAA CDR",
                "noaa_cdr_knowledge_graph.ttl",
                lambda path: build_noaa_cdr_graph(path, metadata),
            ),
            ("dynamical.org", "dynamical_org_knowledge_graph.ttl", build_dynamical_org_graph),
        ]

        # Build each graph locally
        local_files: list[tuple[str, str, Path]] = []
        for graph_name, filename, builder_func in graphs:
            log.info("Building %s knowledge graph...", graph_name)
            local_path = temp_dir / filename

            try:
                builder_func(str(local_path))
                local_files.append((graph_name, filename, local_path))
                log.info(
                    "Successfully built %s graph (%d bytes)", graph_name, local_path.stat().st_size
                )
            except Exception:
                log.exception("Failed to build %s knowledge graph", graph_name)
                raise

        # Upload all graphs to S3
        log.info("Uploading knowledge graphs to S3...")
        s3 = s3fs.S3FileSystem()

        for graph_name, filename, local_path in local_files:
            s3_path = f"s3://{output_bucket}/{output_prefix}{filename}"
            log.info("Uploading %s to %s", graph_name, s3_path)

            try:
                with s3.open(s3_path, "w") as f:  # pyright: ignore[reportUnknownMemberType]
                    f.write(local_path.read_text())  # pyright: ignore[reportUnknownMemberType]
                log.info("Successfully uploaded %s", s3_path)
            except Exception:
                log.exception("Failed to upload %s to S3", graph_name)
                raise

        log.info("All knowledge graphs built and uploaded successfully")

    except Exception:
        log.exception("Knowledge graph building process failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
