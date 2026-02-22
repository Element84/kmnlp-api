#!/bin/bash

set -euo pipefail
set -x

# Update apt and install necessary dependencies.
apt-get update && \
  apt-get install -y \
  curl \
  git \
  jq

# Update config config to fetch 2 private repos over https using the CI job token
git config --global url."https://gitlab-ci-token:${CI_JOB_TOKEN:-}@repo.element84.com/".insteadOf ssh://git@repo.element84.com/

# Create branch
git switch -c "${BRANCH_NAME:?}"

# Upgrade llm-agent
uv lock --upgrade-package llm-agent && uv export --no-hashes --all-extras --format requirements-txt > requirements.txt

# Check whether any updates were made, and exit with success if not
if [[ -n "$(git status --porcelain)" ]]; then
  # There are changes
  echo "Changes made to repository. Continuing.";
else
  echo "No changes made to repository. Exiting early with success.";
  exit 0;
fi


# Set Git author information
git config user.email "kmnlp-eng@element84.com"
git config user.name "GitLab Auto-Update CD Job"

# Commit and push
git add uv.lock requirements.txt
git commit -m "CD: Upgrade llm-agent to latest"
git remote set-url origin "https://gitlab-ci-token:${PROJECT_ACCESS_TOKEN:?}@${CI_SERVER_HOST:?}/${CI_PROJECT_PATH:?}.git"
git remote -v
git push origin "${BRANCH_NAME:?}"

# Extract llm-agent SHA
LLM_AGENT_VERSION=$(grep 'ssh://git@repo.element84.com/noaa_km_nlp/llm-agent.git' uv.lock | sed -n -e 's/.*#\(.*\)".*/\1/p')

echo "Creating merge request..."
mr_url=$(curl --fail-with-body --request POST \
--header "PRIVATE-TOKEN: ${PROJECT_ACCESS_TOKEN:?}" \
--header "Content-Type: application/json" \
--data "{
  \"source_branch\": \"${BRANCH_NAME:?}\",
  \"target_branch\": \"${CI_DEFAULT_BRANCH:?}\",
  \"title\": \"CD: Upgrade llm-agent to ${LLM_AGENT_VERSION}\",
  \"description\": \"Automated llm-agent dependency upgrade\n\n- **To**: ${LLM_AGENT_VERSION}\n- **Pipeline**: ${CI_PIPELINE_URL:?}\n\nThis MR was automatically created by the CI pipeline.\",
  \"remove_source_branch\": true,
  \"labels\": [\"dependency-upgrade\", \"automated\"]
}" \
  "${CI_API_V4_URL:?}/projects/${CI_PROJECT_ID:?}/merge_requests" | jq -r '.web_url' )

# Notify Slack
slack_text="CI created an MR: $mr_url"
curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"${slack_text}\"}" "${SLACK_WEBHOOK_URL:?}"

