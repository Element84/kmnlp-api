#!/bin/bash

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

function build_container() {
  repo_name="$1"
  image_name="${REPO_NAMESPACE:?}/${repo_name}"

  if is_ci_environment; then
    docker buildx build \
        --secret id=CI_JOB_TOKEN \
        --build-arg BUILD_ENV_IS_CI=1 \
        -t "${image_name}" \
        --target "$repo_name" \
        --platform linux/amd64 \
        .
  else
    docker buildx build \
        --ssh default \
        --build-arg BUILD_ENV_IS_CI=0 \
        -t "${image_name}" \
        --target "$repo_name" \
        --platform linux/amd64 \
        .
  fi
}


echo "Building docker images ..."
build_container api
build_container dask
