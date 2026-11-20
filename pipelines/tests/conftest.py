import os

import pytest


# function executed right after test items collected but before test run
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mechanism to only run non-integration (or integration) tests."""
    if not config.getoption("-m"):
        skip_me = pytest.mark.skip(reason="use `-m integration` to run this test")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_me)

    # Enable socket connections for integration tests
    for item in items:
        # If test is marked as an integraion test, allow network connections
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.enable_socket)
        # Otherwise, test is not an integration test. Disable tracing.
        else:
            os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
