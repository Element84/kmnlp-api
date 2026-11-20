import json
from importlib.resources import files
from typing import Any

import pytest
from e84_geoai_common.geometry import geometry_from_geojson
from shapely.geometry.base import BaseGeometry

import tests.utils.data.geojson
from app.types.geojson import GeoJSONGeometry

GEOJSON_DIR = files(tests.utils.data.geojson)


@pytest.fixture(scope="session")
def black_sea_geojson_path() -> str:
    return str(GEOJSON_DIR / "black_sea.json")


@pytest.fixture(scope="session")
def gulf_of_mexico_geojson_path() -> str:
    return str(GEOJSON_DIR / "gulf_of_mexico.json")


@pytest.fixture(scope="session")
def gulf_of_mexico_geometry(gulf_of_mexico_geojson_path: str) -> BaseGeometry:
    with open(gulf_of_mexico_geojson_path) as f:
        return geometry_from_geojson(f.read())


@pytest.fixture(scope="session")
def gulf_of_mexico_geojson_model(gulf_of_mexico_geojson_path: str) -> GeoJSONGeometry:
    with open(gulf_of_mexico_geojson_path) as f:
        return GeoJSONGeometry.model_validate_json(f.read())


@pytest.fixture(scope="session")
def gulf_of_mexico_geojson(gulf_of_mexico_geojson_path: str) -> str:
    with open(gulf_of_mexico_geojson_path) as f:
        return f.read()


@pytest.fixture(scope="session")
def gulf_of_mexico_geojson_dict(gulf_of_mexico_geojson_path: str) -> dict[str, Any]:
    with open(gulf_of_mexico_geojson_path) as f:
        return json.loads(f.read())


@pytest.fixture(scope="session")
def black_sea_geometry(black_sea_geojson_path: str) -> BaseGeometry:
    with open(black_sea_geojson_path) as f:
        return geometry_from_geojson(f.read())
