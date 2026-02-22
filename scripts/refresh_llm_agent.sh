#!/bin/bash

#######################################################################
# Pulls downs the latest commit for llm-agent.
#######################################################################

set -e -o pipefail

uv lock --upgrade-package "llm-agent[data-processing]"
uv export --no-hashes --all-extras --format requirements-txt >requirements.txt
uv sync --all-extras
