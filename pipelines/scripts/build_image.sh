#!/usr/bin/env bash

###############################################
# Build the docker image to use in production #
###############################################

docker build -t kmnlp/ingest-ubl --platform linux/amd64 .
