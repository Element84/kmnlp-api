#!/bin/bash

set -e -o pipefail
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.sh"

run_in_docker=false

usage() {
  echo "Usage: $0 [--docker]"
  echo "  --docker    Run the application in Docker container"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --docker)
      run_in_docker=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option $1"
      usage
      ;;
  esac
done


if [[ $run_in_docker == true ]]; then
  docker run \
    --publish 8000:8000 \
    --env-file ./.env \
    --volume ~/.aws/:/root/.aws/ \
    "${REPO_NAMESPACE:?}/api"

  # Use this with docker run to tunnel
  # --env RDS_DB_HOST=host.docker.internal \
else
  export PYTHONPATH=.
  uvicorn app.main:app --host localhost --port 8000
fi
