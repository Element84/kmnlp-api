import asyncio
import json
import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from e84_geoai_common.llm.models.claude import CLAUDE_4_6_OPUS
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from llm_agent.agents.algorithm_agent.algorithm.base import BaseFunctionCall
from llm_agent.agents.algorithm_agent.algorithm.exec.animation import AnimationProgressUpdate
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    ComputedVariableUpdate,
    ComputingVariableUpdate,
    ExecutionConfig,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import (
    DaskExecutionProgressUpdate,
    FilterByPolygonUpdate,
    GeocodingSuccessUpdate,
    SpatialSlicingUpdate,
)
from llm_agent.agents.algorithm_agent.interface import AlgorithmSolutionSpec
from llm_agent.agents.dataset_picking_agent.dataset_picking_agent import DatasetPickingAgent
from llm_agent.agents.orchestration_agent.interface import UserMessage
from llm_agent.agents.orchestration_agent.orchestration_agent import OrchestrationAgent
from llm_agent.context import ToolProgressUpdate
from llm_agent.tools.agent_plan_tool import (
    AgentPlan,
    AgentPlanOrUpdate,
    AgentPlanUpdate,
    agent_plan_tool,
)
from llm_agent.tools.agent_tool import ErrorResponse
from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    AlgorithmAgentThinkingCompleteUpdate,
    AlgorithmAgentThinkingDeltaUpdate,
    DataAnalysisTool,
    DataAnalysisToolOutput,
    DataAnalysisToolProblemFormulatedUpdate,
    DataAnalysisToolSolutionAttemptUpdate,
    DataAnalysisToolSolutionGeneratedUpdate,
)
from llm_agent.tools.data_analysis_tool.types import ImageRef
from llm_agent.tools.data_search_tool import (
    DataSearchTool,
    DataSearchToolResult,
    SearchCompletedUpdate,
    SearchInitiatedUpdate,
)
from llm_agent.tools.fetch_datasets_tool import KnowledgeGraphQueryUpdate, QueryResultsUpdate
from llm_agent.tools.nl_geocoding_tool import (
    GeocodingToolAttemptUpdate,
    GeocodingToolSuccessUpdate,
    NaturalLanguageGeocodingTool,
)
from llm_agent.tools.resolve_data_ref_tool import ResolveDataRefTool, ResolveDataRefToolInput
from llm_agent.tools.response_shaping_tool import ResponseShapingToolAck
from llm_agent.tools.tracking import TrackingInfo, TrackingTool
from pydantic import BaseModel
from pydantic_ai import AgentRunResultEvent, AgentStreamEvent, ModelSettings, capture_run_messages
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.providers.bedrock import BedrockProvider
from rich.pretty import pretty_repr

from app.context import KmnlpChatContext
from app.dao.users_dao import UsersDAO
from app.types.geojson import GeoJSONFeatureCollection
from app.types.internal import InternalChat
from app.types.output import (
    BackendError,
    EndOfTurn,
    FilteredByAOI,
    GeocodedAOI,
    Plan,
    ProgressUpdate,
    Text,
    TextDelta,
    Thinking,
    ThinkingDelta,
    chat_response_content_types,
)
from app.utils.env import (
    ALGORITHM_AGENT_MAX_TOKENS,
    DASK_COMPUTE_SIZE_LIMIT_GB,
    DASK_PROGRESS_POLL_INTERVAL_SEC,
    ORCHESTRATION_AGENT_MAX_CONSECUTIVE_RESPONSES,
    ORCHESTRATION_AGENT_MAX_TOKENS,
)
from app.utils.orchestration import data_analysis, data_search
from app.utils.orchestration.system_prompt import build_system_prompt
from app.utils.plan_manager import PlanManager, UnknownToolCallIDError

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

log = logging.getLogger(__name__)


RESPONSE_LIMIT_EXCEEDED_ERROR = BackendError(
    message=(
        "Error: Agent exceeded allowed limit of "
        f"{ORCHESTRATION_AGENT_MAX_CONSECUTIVE_RESPONSES} consecutive "
        "responses."
    )
)

END_OF_TURN = EndOfTurn()


class UnknownPlanIDError(Exception):
    """Plan ID in update does not match any current plan."""

    def __init__(self, plan_ids_curr: list[str], plan_id_recvd: str) -> None:  # noqa: D107
        super().__init__(
            f"Plan ID in update ({plan_id_recvd}) does not "
            f"match any current plan ({plan_ids_curr})."
        )


class UnexpectedAgentEventTypeError(Exception):
    """Agent event type is unexpected."""

    def __init__(self, event: Any) -> None:  # noqa: ANN401, D107
        super().__init__(f"Unexpected agent event type: {event}")


class _ProgressDispatcher:
    """Maps tool progress updates to response content, buffering ones that race registration.

    A tool can emit a progress update before we've processed the `FunctionToolCallEvent` that
    registers its tool-call ID (the two arrive on separate, concurrent channels). Such an update
    can't be mapped to a plan/step yet, so it is buffered by tool-call ID and replayed via `flush`
    once the corresponding tool call is registered.
    """

    def __init__(
        self,
        plan_manager: PlanManager,
        chat_context: KmnlpChatContext,
        emit_response: Callable[[chat_response_content_types], None],
    ) -> None:
        """Constructor."""
        self._plan_manager = plan_manager
        self._chat_context = chat_context
        self._emit_response = emit_response
        self._pending: dict[str, list[ToolProgressUpdate[Any]]] = {}

    def _try_emit(self, update: ToolProgressUpdate[Any]) -> bool:
        """Emit the update as response content, or return False if its tool call is unregistered."""
        try:
            contents = list(
                _tool_progress_update_to_response_content(
                    self._plan_manager, update, chat_context=self._chat_context
                )
            )
        except UnknownToolCallIDError:
            return False
        for content in contents:
            self._emit_response(content)
        return True

    def callback(self, update: ToolProgressUpdate[Any]) -> None:
        """Progress callback: emit the update, or buffer it if its tool call isn't registered."""
        if not self._try_emit(update):
            self._pending.setdefault(update.tool_call_id, []).append(update)

    def flush(self, tool_call_id: str) -> None:
        """Replay buffered progress updates for a tool call now that it is registered."""
        for update in self._pending.pop(tool_call_id, []):
            if not self._try_emit(update):
                log.warning(
                    "Progress update for tool call %s still unresolved after registration.",
                    tool_call_id,
                )


def make_agent(*, bedrock_runtime_client: "BedrockRuntimeClient") -> OrchestrationAgent:
    """Instantiate and return orchestration agent."""
    dataset_picking_agent = DatasetPickingAgent(
        InstrumentedModel(
            BedrockConverseModel(
                CLAUDE_4_6_OPUS, provider=BedrockProvider(bedrock_client=bedrock_runtime_client)
            )
        )
    )
    data_search_tool_with_tracking = TrackingTool(
        DataSearchTool(dataset_picking_agent=dataset_picking_agent)
    )

    data_analysis_tool_with_tracking = TrackingTool(
        DataAnalysisTool(
            InstrumentedModel(
                BedrockConverseModel(
                    CLAUDE_4_6_OPUS,
                    provider=BedrockProvider(bedrock_client=bedrock_runtime_client),
                    settings=ModelSettings(
                        thinking="high",
                        temperature=1.0,  # required to be 1 when using thinking
                        max_tokens=ALGORITHM_AGENT_MAX_TOKENS,
                    ),
                )
            ),
            exec_config=ExecutionConfig(
                data_array_compute_size_limit=DASK_COMPUTE_SIZE_LIMIT_GB * 10**9,
                dask_progress_poll_interval_sec=DASK_PROGRESS_POLL_INTERVAL_SEC,
            ),
        )
    )

    return OrchestrationAgent(
        InstrumentedModel(
            BedrockConverseModel(
                CLAUDE_4_6_OPUS,
                provider=BedrockProvider(bedrock_client=bedrock_runtime_client),
                settings=ModelSettings(
                    thinking=True,  # default effort level
                    temperature=1.0,  # required to be 1 when using thinking
                    max_tokens=ORCHESTRATION_AGENT_MAX_TOKENS,  # per turn
                ),
            ),
        ),
        system_prompt=build_system_prompt(),
        tools=[
            data_search_tool_with_tracking,
            data_analysis_tool_with_tracking,
            TrackingTool(NaturalLanguageGeocodingTool()),
            ResolveDataRefTool(),
        ],
        response_shaping_tools=[agent_plan_tool],
    )


@observe(
    as_type="chain",
    name="get_agent_response_content_stream",
    capture_input=False,
    capture_output=False,
)
async def get_agent_response_content_stream(  # noqa: PLR0913
    agent: OrchestrationAgent,
    *,
    new_user_msg: UserMessage,
    chat: InternalChat,
    chat_context: KmnlpChatContext,
    users_dao: UsersDAO,
    user_id: str,
    emit_response: Callable[[chat_response_content_types], None],
) -> None:
    """Get response stream for a single user message."""
    with capture_run_messages() as captured_messages:
        try:
            await _get_agent_response_content_stream(
                agent,
                new_user_msg=new_user_msg,
                chat=chat,
                chat_context=chat_context,
                emit_response=emit_response,
            )
        except asyncio.CancelledError:
            # Client disconnected - cleanup and propagate cancellation
            log.warning("Agent execution cancelled due to client disconnection")

            log.info("Sanitizing and saving conversation")
            agent_messages = captured_messages[1:]  # Exclude the initial user message
            sanitized_messages = _close_orphaned_tool_calls(agent_messages)
            chat.insert_conversation_messages(sanitized_messages)

            # pydantic_ai captures that partial text/thinking in the interrupted
            # `ModelResponse`, so recover it from there and persist it.
            if (
                partial_text_or_thinking := _extract_partial_response(captured_messages)
            ) is not None:
                log.info(
                    "Saving partial %s response: %s",
                    type(partial_text_or_thinking).__name__,
                    partial_text_or_thinking,
                )
                emit_response(partial_text_or_thinking)

            cancelled_plans = chat.plan_manager.cancel_all()
            log.info("Cancelling %d incomplete plans", len(cancelled_plans))
            for cancelled_plan in cancelled_plans:
                # This does not reach the client because it is disconnected, but `emit_response`
                # persists it to the chat history, so it is emitted when the chat is reloaded.
                emit_response(cancelled_plan)

            emit_response(BackendError(message="Request cancelled by client"))
            raise
        finally:
            # Ensure chat state is saved
            log.info("Saving chat state to database")
            users_dao.update_chat(user_id, chat)


@dataclass
class TrackedTimes:
    """Class to track the start times of events for computing durations."""

    thinking_start_time: float | None = None


async def _get_agent_response_content_stream(
    agent: OrchestrationAgent,
    *,
    new_user_msg: UserMessage,
    chat: InternalChat,
    chat_context: KmnlpChatContext,
    emit_response: Callable[[chat_response_content_types], None],
) -> None:
    """Get response stream for a single user message."""
    plan_manager = chat.plan_manager
    progress_dispatcher = _ProgressDispatcher(plan_manager, chat_context, emit_response)

    event_stream = agent.run_stream(
        chat_context,
        new_user_msg,
        message_history=chat.agent_conversation[:-1],
        progress_callback=progress_dispatcher.callback,
    )

    i = 0
    tracked_times = TrackedTimes()
    async for event in event_stream:
        if i >= ORCHESTRATION_AGENT_MAX_CONSECUTIVE_RESPONSES:
            emit_response(RESPONSE_LIMIT_EXCEEDED_ERROR)
            break

        # Only count completed events; ignore streaming deltas.
        if isinstance(event, PartEndEvent):
            i += 1

        try:
            _handle_agent_event(
                event,
                plan_manager=plan_manager,
                chat=chat,
                chat_context=chat_context,
                emit_response=emit_response,
                progress_dispatcher=progress_dispatcher,
                tracked_times=tracked_times,
            )
        except Exception as e:
            log.exception("Error encountered while processing LLM response.")
            emit_response(BackendError(message=f"{e!r}"))
            break

    emit_response(END_OF_TURN)


def _handle_agent_event(  # noqa: C901, PLR0912, PLR0913
    event: AgentStreamEvent | AgentRunResultEvent[str],
    *,
    plan_manager: PlanManager,
    chat: InternalChat,
    chat_context: KmnlpChatContext,
    emit_response: Callable[[chat_response_content_types], None],
    progress_dispatcher: _ProgressDispatcher,
    tracked_times: TrackedTimes,
) -> None:
    match event:
        # -----------------------------------------
        # 1. Text Streaming & Deltas
        # -----------------------------------------
        case PartDeltaEvent(delta=delta):
            log.debug("\n[Part Delta] %s", delta)
            match delta:
                case TextPartDelta(content_delta=content_delta):
                    emit_response(TextDelta(content_delta=content_delta))
                case ThinkingPartDelta(content_delta=str(content_delta)):
                    emit_response(ThinkingDelta(content_delta=content_delta))
                case _:
                    pass

        # -----------------------------------------
        # 2. Tool / Function Calling
        # -----------------------------------------
        case FunctionToolCallEvent(part=part):
            _handle_tool_call_event(
                event,
                plan_manager=plan_manager,
                chat_context=chat_context,
                emit_response=emit_response,
                progress_dispatcher=progress_dispatcher,
            )

        case FunctionToolResultEvent(part=part):
            _handle_tool_result_event(
                event,
                plan_manager=plan_manager,
                chat_context=chat_context,
                emit_response=emit_response,
            )

        # -----------------------------------------
        # 3. Specialized Tools (Output & Built-in)
        # -----------------------------------------
        case OutputToolCallEvent() | OutputToolResultEvent():
            raise UnexpectedAgentEventTypeError(event)

        # -----------------------------------------
        # 4. Message/Part Lifecycle Events
        # -----------------------------------------
        case PartStartEvent(part=part):
            # Triggered when a new part (like a text response or tool call) begins
            log.info("\n[Part Start] %s", event)
            match part:
                case TextPart(content=content):
                    emit_response(TextDelta(content_delta=content))
                case ThinkingPart(content=content):
                    tracked_times.thinking_start_time = perf_counter()
                    emit_response(ThinkingDelta(content_delta=content))
                case _:
                    pass

        case PartEndEvent(part=part):
            # Triggered when the current part finishes streaming
            log.info("\n[Part End] %s", event)
            match part:
                case TextPart(content=content):
                    emit_response(Text(text=content))
                case ThinkingPart(content=content):
                    thinking_duration = (
                        (perf_counter() - tracked_times.thinking_start_time)
                        if tracked_times.thinking_start_time is not None
                        else None
                    )
                    emit_response(
                        Thinking(
                            thinking=content,
                            duration=thinking_duration,
                        )
                    )
                    tracked_times.thinking_start_time = None
                case _:
                    pass

        # -----------------------------------------
        # 5. Final Results
        # -----------------------------------------
        case FinalResultEvent():
            log.info("\n\n[Final Result Captured]")

        case AgentRunResultEvent(result=final_result):
            log.info("\n[Run Finished] The agent run has completely finished.")
            agent_messages = final_result.new_messages()[1:]  # Exclude the initial user message
            chat.insert_conversation_messages(agent_messages)

        # -----------------------------------------
        # Fallback
        # -----------------------------------------
        case _:
            log.info("\n[Unknown Event] Unhandled event type: {%s}", event)
            raise UnexpectedAgentEventTypeError(event)


def _handle_tool_call_event(
    event: FunctionToolCallEvent,
    *,
    plan_manager: PlanManager,
    chat_context: KmnlpChatContext,
    emit_response: Callable[[chat_response_content_types], None],
    progress_dispatcher: _ProgressDispatcher,
) -> None:
    part = event.part
    args = part.args

    log.info("\n[Tool Call] Agent requested '%s' with args: %s", part.tool_name, part.args)

    if args is None:
        return

    if isinstance(args, str):
        args = json.loads(args, strict=False)

    if "tracking" in args:
        tracking_info = TrackingInfo.model_validate(args["tracking"])
        plan_manager.register_tool_call(
            part.tool_call_id, tracking_info.plan_id, tracking_info.step_id
        )
        # A tool call was just registered; replay any progress updates that raced ahead of it.
        progress_dispatcher.flush(event.part.tool_call_id)

    for out_content in _tool_call_to_response_content(
        plan_manager,
        tool_call_id=part.tool_call_id,
        tool_name=part.tool_name,
        args=args,
        chat_context=chat_context,
    ):
        emit_response(out_content)


def _handle_tool_result_event(
    event: FunctionToolResultEvent,
    *,
    plan_manager: PlanManager,
    chat_context: KmnlpChatContext,
    emit_response: Callable[[chat_response_content_types], None],
) -> None:
    part = event.part
    content = part.content
    tool_name = part.tool_name

    if tool_name is None:
        raise ValueError("Tool name is None in FunctionToolResultEvent")

    if not isinstance(content, BaseModel):
        # Likely binary image content
        return

    log.info("\n[Tool Result] '%s' returned: %s", tool_name, content)

    for out_content in _tool_result_to_response_content(
        plan_manager=plan_manager,
        data=content,
        tool_call_id=event.tool_call_id,
        tool_name=tool_name,
        chat_context=chat_context,
    ):
        emit_response(out_content)


def _close_orphaned_tool_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Close out any orphaned tool calls in the captured messages.

    This is necessary to be able to resume the conversation since LLM APIs don't allow orphaned
    tool calls.
    """
    all_calls: dict[str, ToolCallPart] = {
        part.tool_call_id: part
        for msg in messages
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    }
    calls_with_responses: set[str] = {
        part.tool_call_id
        for msg in messages
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    }

    orphaned = [call_id for call_id in all_calls if call_id not in calls_with_responses]

    if orphaned:
        # Inject an explicit cancellation block so the API remains completely valid
        cancellation_parts = [
            ToolReturnPart(
                tool_name=all_calls[call_id].tool_name,
                tool_call_id=call_id,
                content=ErrorResponse(
                    message="Tool execution aborted due to client disconnection."
                ),
            )
            for call_id in orphaned
        ]
        messages.append(ModelRequest(parts=cancellation_parts))

    return messages


def _extract_partial_response(messages: list[ModelMessage]) -> Text | Thinking | None:
    """Recover partially-streamed text/thinking from an interrupted response, if any.

    On cancellation, pydantic_ai appends the in-flight `ModelResponse` (with `state='interrupted'`)
    to the captured messages. So a partial text or thinking exists only when that trailing part is
    an unfinished `TextPart` or `ThinkingPart`, in which case its accumulated content is returned.
    """
    for message in reversed(messages):
        if isinstance(message, ModelResponse) and message.state == "interrupted":
            log.info("Found interrupted ModelResponse: %s.", pretty_repr(message))
            last_part = message.parts[-1] if message.parts else None
            match last_part:
                case TextPart(content=content) if content:
                    return Text(text=content)
                case ThinkingPart(content=content) if content:
                    return Thinking(thinking=content)
                case _:
                    log.info("Last part is neither TextPart nor ThinkingPart or has no content.")
                    return None
    return None


def _tool_call_to_response_content(
    plan_manager: PlanManager,
    tool_call_id: str,  # noqa: ARG001
    tool_name: str,
    args: dict[str, Any],
    chat_context: KmnlpChatContext,  # noqa: ARG001
) -> Generator[chat_response_content_types]:
    match tool_name:
        case "AgentPlanTool":
            data = AgentPlanOrUpdate.model_validate(args)
            yield _agent_plan_or_update_to_response_content(plan_manager, data)
        case ResolveDataRefTool.name:
            data = ResolveDataRefToolInput.model_validate(args)
            match data.data_ref:
                case ImageRef():
                    yield ProgressUpdate(title=f"🖼️ Viewed image: {data.data_ref.name}")
        case _:
            pass


def _tool_result_to_response_content(
    *,
    plan_manager: PlanManager,
    data: BaseModel,
    tool_call_id: str,
    tool_name: str,
    chat_context: KmnlpChatContext,
) -> Generator[chat_response_content_types]:
    match data:
        case ResponseShapingToolAck():
            log.info(
                "Received ResponseShapingToolAck for tool: %s; tool call ID: %s",
                tool_name,
                tool_call_id,
            )
            return
        case DataAnalysisToolOutput():
            # NOTE: At this stage, the API only ever extracts GeoTIFFs and images from the output.
            # It assumes any other relevant data would be summarized by the Orchestration Agent.
            yield from data_analysis.tool_output_to_response_content(
                plan_manager, tool_call_id, data, chat_context
            )
            return
        case DataSearchToolResult():
            log.info("Parsed DataSearchToolResult: %s", pretty_repr(data)[:1000])
            yield data_search.tool_output_to_response_content(plan_manager, tool_call_id, data)
            return
        case ErrorResponse():
            # Add a progress update to the corresponding step
            plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_call_id)
            update = ProgressUpdate(title=f'❌ Tool error: "{data.message}"')
            plan_manager.update_step_progress(plan_id, step_id, update=update)
            yield plan_manager.get_plan(plan_id)
            return
        case _:
            log.warning(
                "Received tool result for unrecognized tool '%s'. Serialized result: %s",
                tool_name,
                pretty_repr(data),
            )


def _agent_plan_or_update_to_response_content(
    plan_manager: PlanManager, data: AgentPlanOrUpdate
) -> Plan:
    match data.plan_or_update:
        case AgentPlan():
            agent_plan = data.plan_or_update
            plan_manager.add_plan(data.plan_or_update)
            return plan_manager.get_plan(agent_plan.id)
        case AgentPlanUpdate():
            agent_plan_update = data.plan_or_update
            plan_manager.update_plan(agent_plan_update)
            return plan_manager.get_plan(agent_plan_update.id)


data_search_tool_update_types = (
    SearchInitiatedUpdate | KnowledgeGraphQueryUpdate | QueryResultsUpdate | SearchCompletedUpdate
)

data_analysis_tool_update_types = (
    GeocodingSuccessUpdate
    | FilterByPolygonUpdate
    | SpatialSlicingUpdate
    | DaskExecutionProgressUpdate
    | DataAnalysisToolProblemFormulatedUpdate
    | DataAnalysisToolSolutionAttemptUpdate
    | DataAnalysisToolSolutionGeneratedUpdate[AlgorithmSolutionSpec[BaseFunctionCall]]
    | AnimationProgressUpdate
    | ComputingVariableUpdate
    | ComputedVariableUpdate
    | AlgorithmAgentThinkingDeltaUpdate
    | AlgorithmAgentThinkingCompleteUpdate
)

geocoding_tool_update_types = GeocodingToolAttemptUpdate | GeocodingToolSuccessUpdate

type PossiblyNestedToolProgressUpdate[T: BaseModel] = (
    ToolProgressUpdate[T] | ToolProgressUpdate[PossiblyNestedToolProgressUpdate[T]]
)


def _tool_progress_update_to_response_content(
    plan_manager: PlanManager,
    tool_update: PossiblyNestedToolProgressUpdate[
        data_search_tool_update_types
        | data_analysis_tool_update_types
        | geocoding_tool_update_types
    ],
    chat_context: KmnlpChatContext,
) -> Generator[Plan | GeocodedAOI | FilteredByAOI]:
    """Map a tool progress update to user-facing content."""
    log.debug("Processing tool progress update: %s", pretty_repr(tool_update)[:1000])
    match tool_update.update:
        case (
            SearchInitiatedUpdate()
            | KnowledgeGraphQueryUpdate()
            | QueryResultsUpdate()
            | SearchCompletedUpdate()
        ):
            ds_tool_update = cast("ToolProgressUpdate[data_search_tool_update_types]", tool_update)
            yield from data_search.tool_progress_update_to_response_content(
                plan_manager, ds_tool_update
            )
        case (
            GeocodingSuccessUpdate()
            | FilterByPolygonUpdate()
            | SpatialSlicingUpdate()
            | DaskExecutionProgressUpdate()
            | DataAnalysisToolProblemFormulatedUpdate()
            | DataAnalysisToolSolutionAttemptUpdate()
            | DataAnalysisToolSolutionGeneratedUpdate()
            | AnimationProgressUpdate()
            | ComputingVariableUpdate()
            | ComputedVariableUpdate()
            | AlgorithmAgentThinkingDeltaUpdate()
            | AlgorithmAgentThinkingCompleteUpdate()
        ):
            da_tool_update = cast(
                "ToolProgressUpdate[data_analysis_tool_update_types]", tool_update
            )
            yield from data_analysis.tool_progress_update_to_response_content(
                plan_manager, da_tool_update, chat_context=chat_context
            )
        case GeocodingToolAttemptUpdate() | GeocodingToolSuccessUpdate():
            geo_tool_update = cast("ToolProgressUpdate[geocoding_tool_update_types]", tool_update)
            plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_update.tool_call_id)
            match geo_tool_update.update:
                case GeocodingToolAttemptUpdate(query=query):
                    update = ProgressUpdate(title=f'🌍 Geocoding "{query}"')
                case GeocodingToolSuccessUpdate(query=query, geojson=geojson):
                    content = GeocodedAOI(
                        query=query,
                        label=f"{query} (geocoded)",
                        geojson=GeoJSONFeatureCollection.from_serialized_geometry(geojson),
                    )
                    update = ProgressUpdate(title=f'📍 Geocoded "{query}"')
                    yield content
            plan_manager.update_step_progress(plan_id, step_id, update=update)
            yield plan_manager.get_plan(plan_id)
        case ToolProgressUpdate():
            plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_update.tool_call_id)
            plan_manager.register_tool_call(tool_update.update.tool_call_id, plan_id, step_id)
            yield from _tool_progress_update_to_response_content(
                plan_manager, tool_update.update, chat_context=chat_context
            )
        case _:
            raise TypeError(
                f"Unsupported tool update type: {type(tool_update.update)}\n{tool_update.update}"
            )
