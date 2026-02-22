import json
from pathlib import Path

from app.main import app


def test_openapi_spec_up_to_date():
    app_spec = app.openapi()

    spec_file_path = Path("openapi.json")
    with spec_file_path.open("r", encoding="utf-8") as f:
        file_spec = json.load(f)

    assert app_spec == file_spec
