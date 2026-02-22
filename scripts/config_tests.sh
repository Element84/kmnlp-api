#!/bin/bash

set -e -o pipefail

EXTRA_ARGS=("${@:1}")

# -vv for verbose output
# -rA to capture stdout output
# --log-cli-level=INFO to capture INFO level logs
pytest -m config \
  --log-cli-format="%(asctime)s %(levelname)s %(name)s:%(lineno)s %(message)s" \
  -vv \
  -rA \
  --log-cli-level=INFO \
  "${EXTRA_ARGS[@]}"
