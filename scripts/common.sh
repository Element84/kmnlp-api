#!/bin/bash

set -e -o pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export AWS_REGION=us-east-1
# Set default region for boto3
export AWS_DEFAULT_REGION=us-east-1

# Docker container constants for image repos
# Format is <REGISTRY_URL>/<REPO_NAMESPACE>/<REPO_NAME>:<IMAGE_TAG>

export REPO_NAMESPACE=kmnlp
export REGISTRY_URL="${AWS_ACCOUNT_ID:?}.dkr.ecr.${AWS_REGION}.amazonaws.com"

export IMAGE_TAG
if [[ -n "${CI_COMMIT_SHORT_SHA:-}" ]]; then
  IMAGE_TAG="ci-${CI_COMMIT_SHORT_SHA}"
else
  IMAGE_TAG="local-$(date +%Y%m%d-%H%M%S)"
fi

function docker-ecr-login() {
  echo "Logging in docker to ECR"
  aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY_URL"
}

# Evaluates to true if this is running in a CI environment
function is_ci_environment() {
  [[ "${GITLAB_CI:-}" == "true" ]]
}
