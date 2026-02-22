#!/bin/bash

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/common.sh"

docker-ecr-login

function tag-and-push() {
  repo_name=$1
  local_image_ref="${REPO_NAMESPACE:?}/$repo_name"
  ecr_image_ref="${REGISTRY_URL:?}/$local_image_ref:${IMAGE_TAG:?}"

  # Create the ECR repo if it doesn't exist
  aws ecr describe-repositories --repository-names "$local_image_ref" &>/dev/null || \
    aws ecr create-repository --repository-name "$local_image_ref" \
      --image-scanning-configuration "scanOnPush=true"

  echo "Pushing $local_image_ref with tag $IMAGE_TAG"

  docker tag "$local_image_ref" "$ecr_image_ref"
  docker push "$ecr_image_ref"
}

tag-and-push api
tag-and-push dask
