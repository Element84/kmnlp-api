#!/bin/bash

##################################################
# Assume AWS role and export related env variables
##################################################

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../common.sh"

if [[ -n "${CI_COMMIT_SHORT_SHA:-}" ]]; then
  image_tag="ci-${CI_COMMIT_SHORT_SHA}"
else
  echo "This script must be called from Gitlab"
  exit 1
fi

# We always pull API for
repo_name=api

ecr_image_ref="${REGISTRY_URL:?}/${REPO_NAMESPACE:?}/$repo_name:$image_tag"

# Parse arguments - everything before the last argument (which should be the command)
# goes to docker run, the last argument is the command to run in container
docker_args=()
container_command=""

# If there are arguments, assume the last one is the container command
if [[ $# -gt 0 ]]; then
  # All arguments except the last one are docker run arguments
  docker_args=("${@:1:$#-1}")
  # Last argument is the container command
  container_command="${!#}"
fi

docker run \
  --rm \
  "${docker_args[@]}" \
  "$ecr_image_ref" \
  "$container_command"
