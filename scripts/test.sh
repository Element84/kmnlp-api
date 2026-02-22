#!/bin/bash

set -e -o pipefail

EXTRA_ARGS=("${@:1}")

export FRONTEND_ENDPOINT=http://irrelevant_during_tests.noaa.gov
export AWS_DEFAULT_REGION=us-east-1

# -vv for verbose output
# -rA to capture stdout output
# --log-cli-level=INFO to capture INFO level logs
pytest -vv -rA --log-cli-level=INFO "${EXTRA_ARGS[@]}" tests/
