import os


class EnvVariableIsUnsetError(Exception):
    """Error indicating required environment variable is unset."""

    def __init__(self, variable_name: str) -> None:
        """Initializer."""
        super().__init__("Env variable %s is unset.", variable_name)


NOMINATIM_USER_AGENT: str | None = os.getenv("NOMINATIM_USER_AGENT")
GEOCODE_INDEX_HOST: str | None = os.getenv("GEOCODE_INDEX_HOST")
ALLOW_LOCAL_DASK_CLUSTER: bool = bool(os.getenv("ALLOW_LOCAL_DASK_CLUSTER"))

USE_SQLITE: bool = bool(os.getenv("USE_SQLITE"))

DB_ENDPOINT: str | None = os.getenv("DB_ENDPOINT")
DB_PORT: str | None = os.getenv("DB_PORT")
DB_NAME: str | None = os.getenv("DB_NAME")
DB_USERNAME: str | None = os.getenv("DB_USERNAME")
DB_PASSWORD: str | None = os.getenv("DB_PASSWORD")

LITE_LLM_PROXY_URL: str | None = os.getenv("LITE_LLM_PROXY_URL")

DASK_COMPUTE_SIZE_LIMIT_GB: int = int(os.getenv("DASK_COMPUTE_SIZE_LIMIT_GB", "5"))  # 5 GB
DASK_PROGRESS_POLL_INTERVAL_SEC: float = float(os.getenv("DASK_PROGRESS_POLL_INTERVAL_SEC", "5.0"))

ORCHESTRATION_AGENT_MAX_CONSECUTIVE_RESPONSES: int = int(
    os.getenv("ORCHESTRATION_AGENT_MAX_CONSECUTIVE_RESPONSES", "100")
)
ALGORITHM_AGENT_MAX_TOKENS: int = int(os.getenv("ALGORITHM_AGENT_MAX_TOKENS", "40000"))
ORCHESTRATION_AGENT_MAX_TOKENS: int = int(os.getenv("ORCHESTRATION_AGENT_MAX_TOKENS", "10000"))
