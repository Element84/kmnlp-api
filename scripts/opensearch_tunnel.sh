#!/bin/bash

#######################################################################
# Tunnel throguh an SSM-managed EC2 instance to connect to OpenSearch #
#######################################################################

set -eo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

if [[ -z "$EC2_TUNNEL_INSTANCE_ID" ]] || [[ -z "$OPENSEARCH_ENDPOINT_HOST" ]]; then
  echo "TARGET_INSTANCE_ID and OPENSEARCH_ENDPOINT_HOST must both be set." >&2
  echo "Values are $EC2_TUNNEL_INSTANCE_ID and $OPENSEARCH_ENDPOINT_HOST respectively." >&2
  exit 1
fi

aws ssm start-session --target "$EC2_TUNNEL_INSTANCE_ID" --document-name AWS-StartPortForwardingSessionToRemoteHost --parameters "{\"host\":[\"$OPENSEARCH_ENDPOINT_HOST\"],\"portNumber\":[\"443\"], \"localPortNumber\":[\"9200\"]}"
