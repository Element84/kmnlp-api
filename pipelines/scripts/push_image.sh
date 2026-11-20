#!/bin/bash

################################
# Push the docker image to ECR #
################################

set -e -o pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
source "$REPO_ROOT/scripts/common.sh"

docker-ecr-login

function tag-and-push() {
  repo_name=$1
  local_image_ref="${REPO_NAMESPACE:?}/$repo_name"
  ecr_image_ref="${REGISTRY_URL:?}/$local_image_ref:${IMAGE_TAG:?}"

  echo "Pushing $local_image_ref with tag $IMAGE_TAG"

  docker tag "$local_image_ref" "$ecr_image_ref"
  docker push "$ecr_image_ref"
}

tag-and-push ingest-ubl
