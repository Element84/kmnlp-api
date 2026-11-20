import logging

# Suppress debug logs from talkative packages
logging.getLogger("botocore").setLevel(logging.INFO)
logging.getLogger("aiobotocore").setLevel(logging.INFO)
logging.getLogger("rastervision").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("fsspec").setLevel(logging.INFO)
logging.getLogger("rasterio").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("s3fs").setLevel(logging.INFO)
logging.getLogger("numcodecs").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.INFO)

# Set debug level of llm-agent to INFO
logging.getLogger("llm_agent").setLevel(logging.INFO)
logging.getLogger("app").setLevel(logging.INFO)
