import base64
import logging
import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any, cast

from e84_geoai_common.llm.core.llm import (
    Base64ImageContent,
    LLMToolResultContent,
    LLMToolUseContent,
)
from e84_geoai_common.llm.models import BedrockClaudeLLM
from llm_agent import (
    DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID,
    ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID,
)
from llm_agent.agents.algorithm_agent.algorithm.base import BaseFunctionCall
from llm_agent.agents.algorithm_agent.algorithm.exec import GeocodingSuccessUpdate
from llm_agent.agents.algorithm_agent.interface import AlgorithmSolutionSpec
from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import DatasetPickingAgent
from llm_agent.agents.orchestration_agent import OrchestrationAgent
from llm_agent.agents.tool_only_agent import MAX_ALLOWED_TOOL_CALLS_PER_TURN
from llm_agent.agents.tools_config import ToolsConfig
from llm_agent.context import LLMAgentContext, ToolProgressUpdate
from llm_agent.model.dataset import Dataset
from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    DataAnalysisTool,
    DataAnalysisToolOutput,
    DataAnalysisToolProblemFormulatedUpdate,
    DataAnalysisToolSolutionAttemptUpdate,
    DataAnalysisToolSolutionGeneratedUpdate,
)
from llm_agent.tools.data_analysis_tool.types import GeoTIFFRef
from llm_agent.tools.data_search_tool import (
    DataSearchTool,
    DataSearchToolResult,
    SearchInitiatedUpdate,
)
from llm_agent.tools.llm_tool_typed import ErrorResponse
from llm_agent.tools.llm_tool_with_tracking import LLMToolWithTracking
from llm_agent.tools.response_shaping_tools.agent_plan_tool import (
    AgentPlan,
    AgentPlanOrUpdate,
    AgentPlanUpdate,
    agent_plan_tool,
)
from llm_agent.tools.response_shaping_tools.end_turn_tool import EmptyInput, end_turn_tool
from llm_agent.tools.response_shaping_tools.text_response_tool import (
    TextResponse,
    text_response_tool,
)
from llm_agent.uploaders.s3_file_uploader import upload_file
from llm_agent.utils.prompt_as_service import (
    PromptService,
    PromptServiceAdapter,
    SimplePromptServiceTransformer,
)
from llm_agent.utils.utils import EnrichedLLMToolResultContent
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel

from app.dao.users_dao import UsersDAO
from app.types.geojson import GeoJSONFeatureCollection
from app.types.internal import InternalChat
from app.types.output import (
    BackendError,
    ChatResponse,
    DataAnalysisDetails,
    DataAnalysisOutput,
    DataAnalysisProblem,
    DataAnalysisSolution,
    DatasetMetadata,
    DatasetSearchDetails,
    DatasetSearchQuery,
    EndOfTurn,
    GeocodedAOI,
    GeoTIFF,
    Image,
    Legend,
    Plan,
    ProgressUpdate,
    StepDetails,
    Text,
    Variable,
    chat_response_content_types,
)
from app.utils.plan_manager import PlanManager, UnknownToolCallIDError

tool_result_types = (
    EmptyInput
    | AgentPlanOrUpdate
    | TextResponse
    | DataSearchToolResult
    | DataAnalysisToolOutput
    | ErrorResponse
)


@dataclass()
class AppState:
    """Application state container."""

    context: LLMAgentContext
    agent: OrchestrationAgent


log = logging.getLogger(__name__)

MAX_CONSECUTIVE_AGENT_RESPONSES = 16

RESPONSE_LIMIT_EXCEEDED_ERROR = BackendError(
    message=(
        "Error: Agent exceeded allowed limit of "
        f"{MAX_CONSECUTIVE_AGENT_RESPONSES} consecutive "
        "responses."
    )
)
END_OF_TURN = EndOfTurn()


class ToolResultWithoutDataError(Exception):
    """Tool result is not an EnrichedLLMToolResultContent."""

    def __init__(self, tool_result: LLMToolResultContent) -> None:
        """Constructor."""
        self.tool_result = tool_result
        super().__init__(f"{self.tool_result.model_dump_json(indent=2)}")


class UnknownPlanIDError(Exception):
    """Plan ID in update does not match any current plan."""

    def __init__(self, plan_ids_curr: list[str], plan_id_recvd: str) -> None:  # noqa: D107
        super().__init__(
            f"Plan ID in update ({plan_id_recvd}) does not "
            f"match any current plan ({plan_ids_curr})."
        )


def make_agent(
    *,
    bedrock_runtime_client: BedrockRuntimeClient,
    orchestration_prompt_simple_transformer: SimplePromptServiceTransformer | None,
) -> OrchestrationAgent:
    """Instantiate and return orchestration agent."""
    dataset_picking_agent = DatasetPickingAgent(
        prompt_service=PromptServiceAdapter(
            BedrockClaudeLLM(
                model_id=DATASET_PICKING_AGENT_DEFAULT_LLM_MODEL_ID, client=bedrock_runtime_client
            ).prompt
        )
    )
    data_search_tool_with_tracking = LLMToolWithTracking(
        DataSearchTool(dataset_picking_agent=dataset_picking_agent)
    )

    data_analysis_tool_with_tracking = LLMToolWithTracking(DataAnalysisTool())

    orchestration_prompt_service: PromptService = PromptServiceAdapter(
        BedrockClaudeLLM(
            model_id=ORCHESTRATION_AGENT_DEFAULT_LLM_MODEL_ID, client=bedrock_runtime_client
        ).prompt
    )

    composed_service = (
        orchestration_prompt_service
        if orchestration_prompt_simple_transformer is None
        else orchestration_prompt_simple_transformer.and_then_service(orchestration_prompt_service)
    )

    return OrchestrationAgent(
        prompt_service=composed_service,
        tools_config=ToolsConfig(
            tools=[
                data_search_tool_with_tracking,
                data_analysis_tool_with_tracking,
                text_response_tool,
                agent_plan_tool,
            ],
            end_turn_tool=end_turn_tool,
            max_allowed_tool_calls_per_turn=MAX_ALLOWED_TOOL_CALLS_PER_TURN,
        ),
    )


async def get_agent_response_content_stream(  # noqa: C901, PLR0913
    agent: OrchestrationAgent,
    *,
    chat: InternalChat,
    app_state: AppState,
    users_dao: UsersDAO,
    user_id: str,
    emit_response: Callable[[chat_response_content_types], None],
) -> None:
    """Get response stream for a single user message."""
    plan_manager = chat.plan_manager

    def progress_callback(update: ToolProgressUpdate[Any]) -> None:
        try:
            for content in _tool_progress_update_to_response_content(
                plan_manager, update, app_state
            ):
                emit_response(content)
        except UnknownToolCallIDError:
            log.exception("Error while processing tool progress update.")

    # this modifies chat.agent_conversation in place
    responses = agent.run(
        app_state.context, chat.agent_conversation, progress_callback=progress_callback
    )

    i = 0
    turn_ended = False
    async for response in responses:
        if i >= MAX_CONSECUTIVE_AGENT_RESPONSES:
            emit_response(RESPONSE_LIMIT_EXCEEDED_ERROR)
            break
        try:
            match response:
                case LLMToolUseContent():
                    if "tracking" in response.input:
                        tracking_info = response.input["tracking"]
                        plan_manager.register_tool_call(
                            response.id, tracking_info["plan_id"], tracking_info["step_id"]
                        )
                case LLMToolResultContent():
                    for content in _tool_result_to_response_content(
                        plan_manager, response, app_state
                    ):
                        emit_response(content)
                        if isinstance(content, EndOfTurn):
                            turn_ended = True
        except Exception as e:
            log.exception("Error encountered while processing LLM response.")
            emit_response(BackendError(message=f"{e!r}"))
            break
        i += 1

    if not turn_ended:
        emit_response(END_OF_TURN)

    # ensure last msg in conversation gets saved
    users_dao.update_chat(user_id, chat)


def _tool_result_to_response_content(
    plan_manager: PlanManager, result: LLMToolResultContent, app_state: AppState
) -> Generator[chat_response_content_types]:
    if not isinstance(result, EnrichedLLMToolResultContent):
        raise ToolResultWithoutDataError(result)
    result = cast("EnrichedLLMToolResultContent[tool_result_types]", result)
    return _tool_result_data_to_response_content(
        plan_manager=plan_manager,
        tool_call_id=result.id,
        data=result.data,
        app_state=app_state,
    )


def _tool_progress_update_to_response_content(
    plan_manager: PlanManager,
    tool_update: ToolProgressUpdate[
        SearchInitiatedUpdate
        | GeocodingSuccessUpdate
        | DataAnalysisToolProblemFormulatedUpdate
        | DataAnalysisToolSolutionAttemptUpdate
        | DataAnalysisToolSolutionGeneratedUpdate[AlgorithmSolutionSpec[BaseFunctionCall]]
    ],
    app_state: AppState,  # noqa: ARG001
) -> Generator[Plan | GeocodedAOI]:
    plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_update.tool_call_id)
    match tool_update.update:
        case SearchInitiatedUpdate():
            content = DatasetSearchQuery(query=tool_update.update.query)
            update = ProgressUpdate(title=f'Searching for "{content.query}"')

            details = StepDetails(content=DatasetSearchDetails(query=content.query))
            plan_manager.set_step_details(plan_id, step_id, details=details)

        case GeocodingSuccessUpdate():
            # Besides being added to the progress updates, the geocoded AOI
            # is also sent as a separate ChatResponse so that the frontend
            # can handle it specially (e.g. display on map).
            content = GeocodedAOI(
                query=tool_update.update.query,
                geojson=GeoJSONFeatureCollection.from_serialized_geometry(
                    tool_update.update.geojson
                ),
            )
            update = ProgressUpdate(title=f'Geocoded "{content.query}"')
            yield content

        case DataAnalysisToolProblemFormulatedUpdate():
            problem = DataAnalysisProblem(
                statement=tool_update.update.problem.statement,
                inputs_provided=[
                    Variable(
                        name=v.name,
                        type=f"Array w/ dims: {', '.join([d.field_name for d in v.dimensions])}",
                        description=v.description,
                    )
                    for v in tool_update.update.problem.inputs_provided
                ],
                outputs_requested=[
                    Variable(name=v.name, type=v.variable_type, description=v.description)
                    for v in tool_update.update.problem.outputs_requested
                ],
            )
            update = ProgressUpdate(title="Formulated data analysis problem")
            _update_data_analysis_step_details(plan_manager, plan_id, step_id, data=problem)

        case DataAnalysisToolSolutionAttemptUpdate():
            attempt_num = tool_update.update.attempt_num
            if attempt_num == 1:
                title = "Constructing solution..."
            else:
                title = f"Solution did not validate. Retrying... (attempt #{attempt_num})"
            update = ProgressUpdate(title=title)

        case DataAnalysisToolSolutionGeneratedUpdate():
            solution = tool_update.update.solution
            solution_out = DataAnalysisSolution(
                thoughts=solution.thoughts,
                pseudocode=solution.pseudocode,
                explanation=solution.explanation,
            )
            update = ProgressUpdate(title="Successfully constructed solution. Executing...")
            _update_data_analysis_step_details(plan_manager, plan_id, step_id, data=solution_out)

    plan_manager.update_step_progress(plan_id, step_id, update=update)
    yield plan_manager.get_plan(plan_id)


def _tool_result_data_to_response_content(
    plan_manager: PlanManager, tool_call_id: str, data: tool_result_types, app_state: AppState
) -> Generator[chat_response_content_types]:
    match data:
        case EmptyInput():
            yield END_OF_TURN
        case AgentPlanOrUpdate():
            yield _parse_agent_plan_or_update(plan_manager, data)
        case TextResponse():
            yield Text.model_validate(data.model_dump())
        case DataAnalysisToolOutput():
            # NOTE: At this stage, the API only ever extracts GeoTIFFs and images from the output.
            # It assumes any other relevant data would be summarized by the Orchestration Agent.
            yield from _parse_data_analysis_tool_output(plan_manager, tool_call_id, data, app_state)
        case DataSearchToolResult():
            yield _parse_data_search_tool_output(plan_manager, tool_call_id, data)
        case ErrorResponse():
            yield BackendError(message=data.message)


def _parse_agent_plan_or_update(plan_manager: PlanManager, data: AgentPlanOrUpdate) -> Plan:
    match data.plan_or_update:
        case AgentPlan():
            agent_plan = data.plan_or_update
            plan_manager.add_plan(data.plan_or_update)
            return plan_manager.get_plan(agent_plan.id)
        case AgentPlanUpdate():
            agent_plan_update = data.plan_or_update
            plan_manager.update_plan(agent_plan_update)
            return plan_manager.get_plan(agent_plan_update.id)


def _parse_data_search_tool_output(
    plan_manager: PlanManager, tool_call_id: str, data: DataSearchToolResult
) -> Plan:
    plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_call_id)
    _update_data_search_step_details(plan_manager, plan_id, step_id, data)
    return plan_manager.get_plan(plan_id)


def _update_data_search_step_details(
    plan_manager: PlanManager,
    plan_id: str,
    step_id: str,
    data: DatasetSearchQuery | DataSearchToolResult,
) -> None:
    def _get_old_details(
        plan_manager: PlanManager, plan_id: str, step_id: str
    ) -> DatasetSearchDetails:
        details = plan_manager.get_step_details(plan_id, step_id)
        if details is None:
            raise ValueError("Step details not found")
        if not isinstance(details.content, DatasetSearchDetails):
            raise TypeError("Step details content is not of type DatasetSearchDetails")
        return details.content

    match data:
        case DatasetSearchQuery():
            new_details = DatasetSearchDetails(query=data.query)
        case DataSearchToolResult():
            details = _get_old_details(plan_manager, plan_id, step_id)

            def _to_out_dataset(ds: Dataset) -> DatasetMetadata:
                start_date, end_date = ds.temporal_coverage
                return DatasetMetadata(
                    title=ds.label,
                    description=ds.description,
                    temporal_coverage=f"{start_date} to {end_date}",
                    spatial_coverage=ds.spatial_coverage,
                    variables=[v.name for v in ds.variables],
                    landing_page=ds.landing_page,
                )

            details = _get_old_details(plan_manager, plan_id, step_id)
            results = [_to_out_dataset(ds) for ds in data.datasets]
            new_details = details.model_copy(update={"results": results})

    plan_manager.set_step_details(plan_id, step_id, details=StepDetails(content=new_details))


def _parse_data_analysis_tool_output(
    plan_manager: PlanManager, tool_call_id: str, data: DataAnalysisToolOutput, app_state: AppState
) -> Generator[Plan | GeoTIFF | Image]:
    if data.data_outputs is None:
        return

    for k, v in data.data_outputs.items():
        if isinstance(v, GeoTIFFRef):
            yield GeoTIFF(
                url=v.url,
                legend=Legend(
                    label=k,
                    units=v.units,
                    min_value=v.min_value,
                    max_value=v.max_value,
                ),
            )

    for img in data.image_data:
        url = upload_base64_image_to_s3(img, app_state)
        yield Image(url=url)

    # update step details
    try:
        plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_call_id)
        _update_data_analysis_step_details(plan_manager, plan_id, step_id, data)
        yield plan_manager.get_plan(plan_id)
    except UnknownToolCallIDError:
        pass


def _update_data_analysis_step_details(  # noqa: C901
    plan_manager: PlanManager,
    plan_id: str,
    step_id: str,
    data: DataAnalysisProblem | DataAnalysisSolution | DataAnalysisToolOutput,
) -> None:
    def _get_old_details(
        plan_manager: PlanManager, plan_id: str, step_id: str
    ) -> DataAnalysisDetails:
        details = plan_manager.get_step_details(plan_id, step_id)
        if details is None:
            raise ValueError("Step details not found")
        if not isinstance(details.content, DataAnalysisDetails):
            raise TypeError("Step details content is not of type DataAnalysisDetails")
        return details.content

    match data:
        case DataAnalysisProblem():
            new_details = DataAnalysisDetails(problem=data)
        case DataAnalysisSolution():
            details = _get_old_details(plan_manager, plan_id, step_id)
            new_details = details.model_copy(update={"solution": data})
        case DataAnalysisToolOutput():
            if data.data_outputs is None:
                return

            def _output_val_to_str(val: str | int | float | GeoTIFFRef) -> str:  # noqa: PYI041
                match val:
                    case str():
                        return val
                    case float():
                        return f"{val:.3f}"
                    case int():
                        return str(val)
                    case GeoTIFFRef():
                        return "<GeoTIFF generated>"

            details = _get_old_details(plan_manager, plan_id, step_id)
            outputs = [
                DataAnalysisOutput(name=k, value=_output_val_to_str(v))
                for k, v in data.data_outputs.items()
            ]
            new_details = details.model_copy(update={"outputs": outputs})

    plan_manager.set_step_details(plan_id, step_id, details=StepDetails(content=new_details))


async def serialize_obj_stream(stream: AsyncGenerator[BaseModel]) -> AsyncGenerator[str]:
    """Serialize a pydantic object stream to ND-JSON."""
    async for obj in stream:
        yield f"{obj.model_dump_json()}\n"


async def save_responses_to_db(
    response_stream: AsyncGenerator[ChatResponse],
    chat: InternalChat,
    users_dao: UsersDAO,
    user_id: str,
) -> AsyncGenerator[ChatResponse]:
    """Intercept responses to save them to database."""
    async for r in response_stream:
        chat.messages.append(r)
        # Update the chat in the database after each response
        users_dao.update_chat(user_id, chat)
        yield r


def upload_base64_image_to_s3(base64_img: Base64ImageContent, app_state: AppState) -> str:
    """Upload base64-encoded image data to S3 and return the URL."""
    with TemporaryDirectory() as tmpdir:
        temp_path = f"{tmpdir}/image_{uuid.uuid4().hex}.png"
        with open(temp_path, "wb") as f:
            f.write(base64.b64decode(base64_img.data))
        url = upload_file(app_state.context.s3_client, temp_path, "test-geotiff-bucket")
    return url
