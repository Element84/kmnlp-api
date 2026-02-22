from unittest.mock import Mock

import pytest
from e84_geoai_common.llm.core.llm import LLMToolResultContent
from llm_agent.tools.data_analysis_tool.data_analysis_tool import DataAnalysisToolOutput
from llm_agent.tools.data_analysis_tool.types import GeoTIFFRef

from app.types.output import GeoTIFF
from app.utils.plan_manager import PlanManager
from app.utils.utils import (
    ToolResultWithoutDataError,
    _parse_data_analysis_tool_output,  # pyright: ignore[reportPrivateUsage]
    _tool_result_to_response_content,  # pyright: ignore[reportPrivateUsage]
)
from tests.utils.stubs.app_state_stub import app_state_stub


class TestUtils:
    def test_tool_result_to_response_content_no_data(self):
        plan_manager = PlanManager()
        app_state = app_state_stub()
        with pytest.raises(ToolResultWithoutDataError):
            _ = _tool_result_to_response_content(
                plan_manager, LLMToolResultContent(id="", content=[]), app_state
            )

    def test_geotiff_response_type_from_geotiff_ref(self):
        plan_manager = PlanManager()
        data_analysis_tool_output = DataAnalysisToolOutput(
            data_outputs={
                "some_var": GeoTIFFRef(
                    url="url",
                    label="label",
                    units="degrees_C",
                    min_value=0,
                    max_value=5,
                )
            },
            image_outputs=None,
            solution_pseudocode=None,
        )

        result = _parse_data_analysis_tool_output(
            plan_manager=plan_manager,
            tool_call_id="abc123",
            data=data_analysis_tool_output,
            app_state=Mock(),
        )
        result_as_list = list(result)

        assert len(result_as_list) == 1

        geotiff = result_as_list[0]
        assert isinstance(geotiff, GeoTIFF)

        assert geotiff.url == "url"
        assert geotiff.legend.label == "some_var"
        assert geotiff.legend.units == "degrees_C"
        assert geotiff.legend.min_value == 0.0
        assert geotiff.legend.max_value == 5.0
