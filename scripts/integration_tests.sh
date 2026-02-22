#!/bin/bash

set -e -o pipefail

EXTRA_ARGS=("${@:1}")

export FRONTEND_ENDPOINT=http://irrelevant_during_tests.noaa.gov

unset DASK_SCHEDULER_ADDRESS
export ALLOW_LOCAL_DASK_CLUSTER=True

# -vv for verbose output
# -rA to capture stdout output
# --log-cli-level=INFO to capture INFO level logs
pytest -m integration \
  --log-cli-format="%(asctime)s %(levelname)s %(name)s:%(lineno)s %(message)s" \
  -vv \
  -rA \
  --log-cli-level=INFO "${EXTRA_ARGS[@]}" \
  tests/integration/
