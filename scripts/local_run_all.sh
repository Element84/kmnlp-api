#!/bin/bash

set -e -o pipefail
set -x

scripts/opensearch_tunnel.sh &
dask scheduler --host="127.0.0.1" --port="8786" &
dask worker tcp://127.0.0.1:8786 --nworkers="auto" &
docker compose up db -d &
scripts/local_run_api.sh &

wait

