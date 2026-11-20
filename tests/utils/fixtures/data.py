from importlib.resources import files

import pytest

import tests.utils.data

DATA_DIR = files(tests.utils.data)


@pytest.fixture(scope="session")
def sample_image_path() -> str:
    return str(DATA_DIR / "sample_image.png")


@pytest.fixture(scope="session")
def sample_geotiff_path() -> str:
    return str(DATA_DIR / "sample_geotiff.tif")


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image_path: str) -> bytes:
    with open(sample_image_path, "rb") as f:
        return f.read()
