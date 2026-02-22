#!/bin/bash

##################################################
# Assume AWS role and export related env variables
##################################################

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../common.sh"

docker-ecr-login

if [[ -n "${CI_COMMIT_SHORT_SHA:-}" ]]; then
  image_tag="ci-${CI_COMMIT_SHORT_SHA}"
else
  echo "This script must be called from Gitlab"
  exit 1
fi

# We always pull API for
repo_name=api

ecr_image_ref="${REGISTRY_URL:?}/${REPO_NAMESPACE:?}/$repo_name:$image_tag"
docker pull "$ecr_image_ref"
