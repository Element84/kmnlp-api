#!/usr/bin/env python3
"""Generate OpenAPI specification for the KMNLP API.

This script generates the OpenAPI specification for the FastAPI application
and writes it to openapi.json in the project root. It does not require a
running server and works by importing the FastAPI app directly.
"""

import json
import sys
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

load_dotenv()


def generate_openapi_spec(output_path: Path) -> dict[str, Any]:
    """Generate OpenAPI specification from the FastAPI app.

    Args:
        output_path: Path where to write the OpenAPI spec

    Returns:
        The generated OpenAPI specification dictionary

    Raises:
        ImportError: If the FastAPI app cannot be imported
        OSError: If file writing fails
        RuntimeError: If OpenAPI generation fails
    """
    try:
        from app.main import app  # noqa: PLC0415
    except ImportError as e:
        click.echo(
            click.style("Error:", fg="red", bold=True)
            + f" Failed to import FastAPI app. Make sure dependencies are installed: {e}",
            err=True,
        )
        raise

    # Generate OpenAPI spec
    try:
        openapi_spec = app.openapi()
    except Exception as e:
        click.echo(
            click.style("Error:", fg="red", bold=True) + f" Failed to generate OpenAPI spec: {e}",
            err=True,
        )
        raise RuntimeError(f"OpenAPI generation failed: {e}") from e

    # Write to output file
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(openapi_spec, f, indent=2, ensure_ascii=False)
    except OSError as e:
        click.echo(
            click.style("Error:", fg="red", bold=True) + f" Failed to write output file: {e}",
            err=True,
        )
        raise

    return openapi_spec


@click.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="openapi.json",
    help="Output file path for the OpenAPI specification",
    show_default=True,
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress output except for errors",
)
def main(output: Path, *, quiet: bool) -> None:
    """Generate OpenAPI specification for the KMNLP API.

    This script generates the OpenAPI specification for the FastAPI application
    and writes it to a JSON file. It does not require a running server and works
    by importing the FastAPI app directly.

    Requirements:
    - All dependencies must be installed (run: uv sync)
    - Must be run from the project root directory

    Examples:
        # Generate to default openapi.json
        python scripts/generate_openapi.py

        # Generate to custom location
        python scripts/generate_openapi.py --output docs/api-spec.json

        # Quiet mode (only show errors)
        python scripts/generate_openapi.py --quiet
    """
    if not quiet:
        click.echo("Generating OpenAPI specification...")

    try:
        # Ensure we're working with absolute paths
        output_path = output.resolve()

        # Generate the OpenAPI spec
        openapi_spec = generate_openapi_spec(output_path)

        if not quiet:
            # Success output with checkmarks and stats
            click.echo(
                click.style("✓", fg="green", bold=True)
                + f" OpenAPI specification written to {output_path}"
            )

            endpoints_count = len(openapi_spec.get("paths", {}))
            click.echo(
                click.style("✓", fg="green", bold=True) + f" Found {endpoints_count} API endpoints"
            )

            schemas_count = len(openapi_spec.get("components", {}).get("schemas", {}))
            click.echo(
                click.style("✓", fg="green", bold=True)
                + f" Found {schemas_count} schema components"
            )

    except (ImportError, OSError, RuntimeError):
        sys.exit(1)


if __name__ == "__main__":
    main()
