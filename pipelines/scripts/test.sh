#!/bin/bash

set -e -o pipefail

# -vv for verbose output
# -rA to capture stdout output
# --log-cli-level=INFO to capture INFO level logs
pytest -vv -rA --log-cli-level=INFO tests/

