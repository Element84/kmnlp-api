"""Container entrypoint for create_icechunk_store."""

import json
import logging
import os
import sys

from piecewise_dataset_pipeline.create_store_for_dataset_description import (
    CreateStoreFromS3WithGlobRequest,
    CreateStoreFromS3WithIndividualFilesRequest,
    CreateStoreFromThreddsSourceRequest,
    CreateStoreRequest,
    create_icechunk_store,
)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
log = logging.getLogger(__name__)

source_bucket = os.getenv("SOURCE_BUCKET")
destination_bucket = os.getenv("DESTINATION_BUCKET")
destination_prefix = os.getenv("DESTINATION_PREFIX")
prefix_glob = os.getenv("PREFIX_GLOB")
source_paths_raw = os.getenv("SOURCE_PATHS")
thredds_catalog_xml_url = os.getenv("THREDDS_CATALOG_XML_URL")

# There really should be an input validation function. Reading this? Go build
# an input validation function! kndl 2026-06-30
if destination_bucket is None or destination_prefix is None:
    log.error("ERROR: DESTINATION_BUCKET and DESTINATION_PREFIX must both be set.")
    sys.exit(1)

if source_bucket is None and thredds_catalog_xml_url is None:
    log.error("ERROR: One of SOURCE_BUCKET or THREDDS_CATALOG_XML_URL must be set.")
    sys.exit(1)

if source_bucket and thredds_catalog_xml_url:
    log.error("error: Either SOURCE_BUCKET or THREDDS_CATALOG_XML_URL must be set, not both.")

if prefix_glob and source_paths_raw:
    log.error("ERROR: Either PREFIX_GLOB or SOURCE_PATHS must be set, not both")
    sys.exit(1)

if not destination_prefix.endswith("/"):
    log.error("ERROR: DESTINATION_PREFIX must have a trailing /")
    sys.exit(1)

if source_bucket and (source_paths_raw is None and prefix_glob is None):
    log.error("ERROR: One of SOURCE_PATHS or PREFIX_GLOB must be set if SOURCE_BUCKET is set.")
    sys.exit(1)


try:
    if source_bucket and prefix_glob:
        request: CreateStoreRequest = CreateStoreFromS3WithGlobRequest(
            source_bucket_name=source_bucket,
            prefix_glob=prefix_glob,
            destination_bucket_name=destination_bucket,
            destination_prefix=destination_prefix,
        )
    elif source_bucket and source_paths_raw:
        request = CreateStoreFromS3WithIndividualFilesRequest(
            source_bucket_name=source_bucket,
            paths=json.loads(source_paths_raw),
            destination_bucket_name=destination_bucket,
            destination_prefix=destination_prefix,
        )
    elif thredds_catalog_xml_url:
        request = CreateStoreFromThreddsSourceRequest(
            destination_bucket_name=destination_bucket,
            destination_prefix=destination_prefix,
            catalog_url=thredds_catalog_xml_url,
        )
    else:
        log.error(
            "Programmer error: none of prefix glob, source paths, or catalog xml url set after "
            "validation."
        )
        sys.exit(2)

    create_icechunk_store(request, on_file_processed=lambda f: log.info("Processed file %s", f))

except Exception:
    log.exception("Ingest failed")
    sys.exit(1)
