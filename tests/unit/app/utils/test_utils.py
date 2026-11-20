import re
from importlib.resources import files
from unittest.mock import Mock

from llm_agent.agents.algorithm_agent.algorithm.exec import (
    FilterByPolygonUpdate,
    GeocodingSuccessUpdate,
)
from llm_agent.context import ToolProgressUpdate
from llm_agent.tools.data_analysis_tool.data_analysis_tool import DataAnalysisToolOutput
from llm_agent.tools.data_analysis_tool.types import GeoTIFFRef

import tests.utils.data
from app.context import KmnlpChatContext
from app.types.output import GeoTIFF
from app.utils.orchestration import data_analysis
from app.utils.plan_manager import PlanManager


class TestUtils:
    def test_geotiff_response_type_from_geotiff_ref(self, chat_context: KmnlpChatContext):
        plan_manager = PlanManager()

        key = chat_context.persist_data(
            str(files(tests.utils.data) / "sample_geotiff.tif"), data_type="geotiff"
        )

        data_analysis_tool_output = DataAnalysisToolOutput(
            outputs={
                "some_var": GeoTIFFRef(
                    key=key,
                    name="some_var",
                    units="degrees_C",
                    min_value=0,
                    max_value=5,
                )
            },
            solution_pseudocode=None,
        )

        result = data_analysis.tool_output_to_response_content(
            plan_manager=plan_manager,
            tool_call_id="abc123",
            data=data_analysis_tool_output,
            chat_context=chat_context,
        )
        result_as_list = list(result)

        assert len(result_as_list) == 1

        geotiff = result_as_list[0]
        assert isinstance(geotiff, GeoTIFF)

        assert re.match(r".+/test-chat-id/data/.+", geotiff.url) is not None
        assert geotiff.legend.label == "some_var"
        assert geotiff.legend.units == "degrees_C"
        assert geotiff.legend.min_value == 0.0
        assert geotiff.legend.max_value == 5.0

    def test_build_successfully_geocoded_message(self):
        plan_manager = Mock()
        plan_manager.tool_call_id_to_plan_step_ids.return_value = ("plan", "step")

        coordinates = [[[-105.0, 40.0], [-104.0, 40.0], [-104.0, 41.0], [-105.0, 40.0]]]
        result = next(
            data_analysis.tool_progress_update_to_response_content(
                plan_manager,
                ToolProgressUpdate(
                    tool_call_id="yo",
                    update=GeocodingSuccessUpdate(
                        query="somewhere",
                        geojson={
                            "type": "Polygon",
                            "coordinates": coordinates,
                        },
                    ),
                ),
                chat_context=Mock(),
            )
        )

        assert result.model_dump(mode="json") == {
            "content_type": "GeocodedAOI",
            "query": "somewhere",
            "label": "somewhere (geocoded)",
            "geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": coordinates,
                            "bbox": None,
                        },
                        "properties": None,
                        "bbox": None,
                    }
                ],
                "bbox": None,
            },
        }

    def test_build_aoi_message(self):
        plan_manager = Mock()
        plan_manager.tool_call_id_to_plan_step_ids.return_value = ("plan", "step")

        coordinates = [[[-105.0, 40.0], [-104.0, 40.0], [-104.0, 41.0], [-105.0, 40.0]]]
        result = next(
            data_analysis.tool_progress_update_to_response_content(
                plan_manager,
                ToolProgressUpdate(
                    tool_call_id="yo",
                    update=FilterByPolygonUpdate(
                        geojson={
                            "type": "Polygon",
                            "coordinates": coordinates,
                        }
                    ),
                ),
                chat_context=Mock(),
            )
        )

        assert result.model_dump(mode="json") == {
            "content_type": "FilteredByAOI",
            "geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": coordinates,
                            "bbox": None,
                        },
                        "properties": None,
                        "bbox": None,
                    }
                ],
                "bbox": None,
            },
        }
