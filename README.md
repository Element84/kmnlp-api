# kmnlp-api
API for KMNLP demo backend


# Run locally the tried-and-true way

_tl;dr_ Set environment variables and run 2 provided scripts.

Set the following variables in your `.env`:
```sh
ALLOW_LOCAL_DASK_CLUSTER=true
AWS_BEARER_TOKEN_BEDROCK=<your-llm-lite-virtual-api-key>
AWS_PROFILE=<your-profile>
DB_NAME=<db-name, e.g. kmnlp_api_dev>
DB_PASSWORD=<db-pass, e.g. kmnlp_api_dev_pass>
DB_PORT=5432
DB_USERNAME=<db-username, e.g. kmnlp_api_dev>
FRONTEND_ENDPOINT="http://localhost:5173"
GEOCODE_INDEX_PORT="9200"
GEOCODE_INDEX_REGION="us-east-1"
LANGFUSE_DB_HOST=<your-langfuse-db-host>
LANGFUSE_HOST=<your-langfuse-host>
LANGFUSE_PUBLIC_KEY=<your-langfuse-public-key>
LANGFUSE_SECRET_KEY=<your-langfuse-secret-key>
LITE_LLM_PROXY_URL=<your-litellm-proxy-url>
```

Then run, however you like to run multiple commands in parallel:

```sh
docker compose up db -d
scripts/opensearch_tunnel.sh
scripts/local_run_api.sh
```

# Run locally the maybe more convenient way?

If you have:
* This project.
* The `llm-agent` project at `../llm-agent`.
* The `Front-End` project at `../Front-End`.

You should be able to start it all with:
```bash
scripts/opensearch_tunnel.sh
docker compose build
docker compose up -d
```

This will start:
* The frontend in a container;
* A container with the API running, live-reloading changes to both this project
  and llm-agent;
* A db;

all on the same network.

Stop all containers with
```bash
docker compose down
```

## Details

### Database options

In production, we use a PSQL database in AWS RDS/Aurora. Locally, you can
choose between Postgres or SQLite.

To use SQLITE, set the environment variable `USE_SQLITE` to true.

To use PSQL, you can run a local Postgres version 17, configure local SSL on
it, and create a user configuration and a database matching the variables:
```
DB_USERNAME
DB_PASSWORD
DB_NAME
```

Or, alternatively, you can use a local Postgres container. The
`postgres.Dockerfile` uses the Postgres docker image and installs a self-signed
SSL cert in it. The `docker-compose.yml`, makes it easy to start that DB server
with the right configuration. To use it, run:
```bash
docker compose up db -d
```


### Geocoding

Geocoding happens in one of 2 ways: Nominatim Place (deprecated) or a Geocode
Index in OpenSearch (preferred).

#### Using Geocode Index localy:

The OpenSearch index is behind the VPC in AWS. Developers should not try to reproduce it locally. Instead, they should connect to it. Because it is in the VPC, they will need to tunnel to connect to it.

##### Getting required AWS components in place

First, setup an EC2 instance that:
* can be managed by SSM -- and --
* can access the OpenSearch service.

Do this in the AWS console -> EC2, Launch Instance

Instance configuration:
* Make sure to select the `aws-controllertower-VPC`.
* Make selections to keep you in the free tier: AWS Linux 2 and a micro instance.
* Create your own security group that allows no ingress and all egress.
* Create your own role that has the following managed policies: `AmazonOpenSearchServiceFullAccess` and `AmazonSSMManagedInstanceCore`.

Navigate to Systems Manasger -> Fleet Manager to confirm that your EC2 instance is available.

###### Reference examples

If the steps provided do not work, compare your EC2 instance's configuration
with that for the instances named `jgilman tunneler` and `kndl tunneler`. Make
any required changes and update this documentation.

##### Install the SSM Client 

If it isn't already installed, [install the Session Manager plugin for your
OS](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).

##### Set instance variables

Upate your `.env` file to include the following values:
```sh
GEOCODE_INDEX_HOST=localhost
GEOCODE_INDEX_PORT="9200"
GEOCODE_INDEX_REGION=<you-region, e.g. us-east-1>

OPENSEARCH_ENDPOINT_HOST=<your-vpc-endpoint-host, e.g. your-vpc.your-region.es.amazonaws.com
>
EC2_TUNNEL_INSTANCE_ID=<your-instance-id, e.g. i-00000000000000000>
```

##### Run the script to setup a tunnel

Map your local port 9200 to the remote port 443:
```sh
scripts/opensearch_tunnel.sh
```

##### Run the API locally

```sh
scripts/local_run_api.sh
```

#### Using Nominatim Place

If you don't want to setup the Geocode index (even though it is recommended and
mirrors how production works), you can do so by setting the following in your
.env:

```sh
E84_NL_GEOCODING="e84_nl_geocoding"
```

Additionally, make sure neither of the following are set in your environment:
`GEOCODE_INDEX_HOST`, `GEOCODE_INDEX_REGION`.

Start the server with:
```sh
scripts/local_run_api.sh
```

# Seeing the production Dask dashboard

In order to connect to the production Dask dashboard, setup your ec2 instance
and SSM as above. Then:

1. `scripts/dask_scheduler_dashboard_tunnel.sh`
1. Visit http://localhost:8787
