# kmnlp-api
API for KMNLP demo backend


# Run locally

_tl;dr_ Set environment variables and run 2 provided scripts.

Set the following variables in your `.env`:
```sh
AWS_PROFILE=your-profile
AWS_ACCOUNT_ID=your-account-id
# For local testing
FRONTEND_ENDPOINT=http://localhost:5173

GEOCODE_INDEX_HOST=localhost
GEOCODE_INDEX_REGION=your-region

OPENSEARCH_ENDPOINT_HOST=your-opensearch-endpoint
EC2_TUNNEL_ISNTANCE_ID=your-ec2-tunnel-instance-id
```

Then run, however you like to run multiple commands in parallel:

```sh
scripts/opensearch_tunnel.sh
scripts/local_run_api.sh
```

## Details

Geocoding happens in one of 2 ways: Nominatim Place (deprecated) or a Geocode Index in OpenSearch (preferred).

### Using Geocode Index localy:

The OpenSearch index is behind the VPC in AWS. Developers should not try to reproduce it locally. Instead, they should connect to it. Because it is in the VPC, they will need to tunnel to connect to it.

#### Getting required AWS components in place

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

##### Reference examples

If the steps provided do not work, compare your EC2 instance's configuration with that for the instances named `jgilman tunneler` and `kndl tunneler`. Make any required changes and update this documentation.

#### Install the SSM Client 

If it isn't already installed, [install the Session Manager plugin for your OS](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).

#### Set instance variables

Upate your `.env` file to include the following values:
```sh
GEOCODE_INDEX_HOST=localhost
GEOCODE_INDEX_PORT="9200"
GEOCODE_INDEX_REGION=<you-region, e.g. us-east-1>

OPENSEARCH_ENDPOINT_HOST=<your-vpc-endpoint-host, e.g. your-vpc.your-region.es.amazonaws.com
>
EC2_TUNNEL_ISNTANCE_ID=<your-instance-id, e.g. i-00000000000000000>
```

#### Run the script to setup a tunnel

Map your local port 9200 to the remote port 443:
```sh
scripts/opensearch_tunnel.sh
```

#### Run the API locally

```sh
scripts/local_run_api.sh
```

### Using Nominatim Place

If you don't want to setup the Geocode index (even though it is recommended and mirrors how production works), you can do so by setting the following in your .env:

```sh
E84_NL_GEOCODING="e84_nl_geocoding"
```

Additionally, make sure neither of the following are set in your environment: `GEOCODE_INDEX_HOST`, `GEOCODE_INDEX_REGION`.

Start the server with:
```sh
scripts/local_run_api.sh
```

# Seeing the production Dask dashboard

In order to connect to the production Dask dashboard, setup your ec2 instance and SSM as above. Then:

1. `scripts/dask_scheduler_dashboard_tunnel.sh`
1. Visit http://localhost:8787
