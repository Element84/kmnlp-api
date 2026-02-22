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

if [[ -z "$EC2_TUNNEL_INSTANCE_ID" ]] || [[ -z "$LANGFUSE_DB_HOST" ]]; then
  echo "TARGET_INSTANCE_ID and DASK_SCHEDULER_SERVICE_HOST must both be set." >&2
  echo "Values are $EC2_TUNNEL_INSTANCE_ID and $LANGFUSE_DB_HOST respectively." >&2
  exit 1
fi

aws ssm start-session --target "$EC2_TUNNEL_INSTANCE_ID" --document-name AWS-StartPortForwardingSessionToRemoteHost --parameters "{\"host\":[\"$LANGFUSE_DB_HOST\"],\"portNumber\":[\"5432\"], \"localPortNumber\":[\"5432\"]}"
