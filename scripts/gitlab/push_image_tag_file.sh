#!/bin/bash

####################################################################################################
# Pushes the image tags to the kmnlp-infrastructure rep.
####################################################################################################

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

# Path to the image version file in the kmnlp-infrastructure project
image_version_file=configs/image_versions/api.yaml

mkdir -p temp
cd temp


git clone "https://gitlab-ci-token:${INFRA_GITLAB_TOKEN:?}@${CI_SERVER_HOST:?}/noaa_km_nlp/kmnlp-infrastructure.git"

cd kmnlp-infrastructure
target_branch=main

git checkout "$target_branch"

cat > "$image_version_file" <<EOF
# Generated automatically by the kmnlp-api build.
# From commit: ${CI_COMMIT_SHA:?}
namespace: ${REPO_NAMESPACE:?}
repository: api
tag: ${IMAGE_TAG:?}
EOF

# Check whether any updates were made, and exit with success if not
if [[ -n "$(git status --porcelain)" ]]; then
  # There are changes
  echo "Changes made to repository. Continuing.";
else
  # This can happen if the same pipeline runs multiple times for the same
  # commit. In this project, that happens when the llm-agent triggers the
  # deploy of this (API project). The deploy job has (usually) already updated
  # the image version file in the infra repo, but the job may have failed.
  # (Otherwise it would be a good idea to not run the job calling this script
  # in the event that the job's source is another pipeline.)
  echo "No changes made to repository. Exiting early with success.";
  exit 0;
fi

git config user.email "kmnlp-eng@element84.com"
git config user.name "GitLab Auto-Update CD Job"

git add "$image_version_file"
git commit -m "Automated update of ${REPO_NAMESPACE:?}/api image"
git push origin "$target_branch"
