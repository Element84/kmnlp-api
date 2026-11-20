import logging
from collections.abc import Generator
from typing import cast

from llm_agent.agents.algorithm_agent.algorithm.base import BaseFunctionCall
from llm_agent.agents.algorithm_agent.algorithm.exec.animation import AnimationProgressUpdate
from llm_agent.agents.algorithm_agent.algorithm.exec.base import (
    ComputedVariableUpdate,
    ComputingVariableUpdate,
)
from llm_agent.agents.algorithm_agent.algorithm.exec.data_array import (
    DaskExecutionProgressUpdate,
    FilterByPolygonUpdate,
    GeocodingSuccessUpdate,
    SpatialSlicingUpdate,
)
from llm_agent.agents.algorithm_agent.interface import (
    AlgorithmSolutionSpec,
    GeneralDataArrayVariable,
    InputVariable,
)
from llm_agent.context import ToolProgressUpdate
from llm_agent.tools.data_analysis_tool.data_analysis_tool import (
    AlgorithmAgentThinkingCompleteUpdate,
    AlgorithmAgentThinkingDeltaUpdate,
    DataAnalysisToolOutput,
    DataAnalysisToolProblemFormulatedUpdate,
    DataAnalysisToolSolutionAttemptUpdate,
    DataAnalysisToolSolutionGeneratedUpdate,
)
from llm_agent.tools.data_analysis_tool.types import AnimationRef, GeoTIFFRef, ImageRef
from shapely.geometry import shape

from app.context import KmnlpChatContext
from app.types.geojson import GeoJSONFeatureCollection
from app.types.output import (
    DataAnalysisDetails,
    DataAnalysisOutput,
    DataAnalysisProblem,
    DataAnalysisSolution,
    DownloadLink,
    EphemeralPlan,
    FilteredByAOI,
    GeocodedAOI,
    GeoTIFF,
    Image,
    Legend,
    Plan,
    ProgressUpdate,
    StepDetails,
    Variable,
    Video,
)
from app.utils.plan_manager import PlanManager, UnknownToolCallIDError

log = logging.getLogger(__name__)


def tool_output_to_response_content(
    plan_manager: PlanManager,
    tool_call_id: str,
    data: DataAnalysisToolOutput,
    chat_context: KmnlpChatContext,
) -> Generator[Plan | GeoTIFF | Image | Video]:
    """Map data analysis tool output to user-facing content."""
    if data.outputs is None:
        return

    for k, v in data.outputs.items():
        match v:
            case GeoTIFFRef():
                url = chat_context.retrieve_url(v.key)
                yield GeoTIFF(
                    url=url,
                    legend=Legend(
                        label=k,
                        units=v.units,
                        min_value=v.min_value,
                        max_value=v.max_value,
                    ),
                )
            case ImageRef():
                url = chat_context.retrieve_url(v.key)
                yield Image(url=url)
            case AnimationRef(format=format):
                url = chat_context.retrieve_url(v.key)
                match format:
                    case "gif":
                        yield Image(url=url, media_type="image/gif")
                    case "mp4":
                        yield Video(url=url, media_type="video/mp4")
            case _:
                pass

    # update step details
    try:
        plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_call_id)
        update_step_details(plan_manager, plan_id, step_id, chat_context=chat_context, data=data)
        yield plan_manager.get_plan(plan_id)
    except UnknownToolCallIDError:
        pass


def update_step_details(  # noqa: C901
    plan_manager: PlanManager,
    plan_id: str,
    step_id: str,
    data: DataAnalysisProblem | DataAnalysisSolution | DataAnalysisToolOutput,
    chat_context: KmnlpChatContext,
) -> None:
    """Update plan step details based on data analysis tool outputs."""

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
            if data.outputs is None:
                return

            def _output_val_to_output_content(
                val: str | int | float | ImageRef | AnimationRef | GeoTIFFRef,  # noqa: PYI041
            ) -> str | DownloadLink:
                match val:
                    case str():
                        return val
                    case float():
                        return f"{val:.3f}"
                    case int():
                        return str(val)
                    case ImageRef() as ref:
                        return DownloadLink(
                            file_type="image", url=chat_context.retrieve_url(ref.key)
                        )
                    case AnimationRef() as ref:
                        return DownloadLink(
                            file_type="animation", url=chat_context.retrieve_url(ref.key)
                        )
                    case GeoTIFFRef() as ref:
                        return DownloadLink(
                            file_type="geotiff", url=chat_context.retrieve_url(ref.key)
                        )

            outputs: list[DataAnalysisOutput] = []
            if data.outputs:
                outputs.extend(
                    DataAnalysisOutput(name=k, value=_output_val_to_output_content(v))
                    for k, v in data.outputs.items()
                )

            details = _get_old_details(plan_manager, plan_id, step_id)
            new_details = details.model_copy(update={"outputs": outputs})

    plan_manager.set_step_details(plan_id, step_id, details=StepDetails(content=new_details))


_algorithm_update_types = (
    DataAnalysisToolProblemFormulatedUpdate
    | DataAnalysisToolSolutionAttemptUpdate
    | DataAnalysisToolSolutionGeneratedUpdate[AlgorithmSolutionSpec[BaseFunctionCall]]
    | AlgorithmAgentThinkingDeltaUpdate
    | AlgorithmAgentThinkingCompleteUpdate
)

_execution_update_types = (
    GeocodingSuccessUpdate
    | FilterByPolygonUpdate
    | SpatialSlicingUpdate
    | DaskExecutionProgressUpdate
    | AnimationProgressUpdate
    | ComputingVariableUpdate
    | ComputedVariableUpdate
)


def tool_progress_update_to_response_content(
    plan_manager: PlanManager,
    tool_update: ToolProgressUpdate[_algorithm_update_types | _execution_update_types],
    chat_context: KmnlpChatContext,
) -> Generator[Plan | GeocodedAOI | FilteredByAOI]:
    """Parse tool progress update into user-facing content."""
    match tool_update.update:
        case (
            DataAnalysisToolProblemFormulatedUpdate()
            | DataAnalysisToolSolutionAttemptUpdate()
            | DataAnalysisToolSolutionGeneratedUpdate()
            | AlgorithmAgentThinkingDeltaUpdate()
            | AlgorithmAgentThinkingCompleteUpdate()
        ):
            yield from _handle_algorithm_updates(
                plan_manager,
                cast("ToolProgressUpdate[_algorithm_update_types]", tool_update),
                chat_context,
            )
        case (
            GeocodingSuccessUpdate()
            | FilterByPolygonUpdate()
            | SpatialSlicingUpdate()
            | DaskExecutionProgressUpdate()
            | AnimationProgressUpdate()
            | ComputingVariableUpdate()
            | ComputedVariableUpdate()
        ):
            yield from _handle_execution_updates(
                plan_manager,
                cast("ToolProgressUpdate[_execution_update_types]", tool_update),
                chat_context,
            )


def _handle_algorithm_updates(
    plan_manager: PlanManager,
    tool_update: ToolProgressUpdate[_algorithm_update_types],
    chat_context: KmnlpChatContext,
) -> Generator[Plan | GeocodedAOI | FilteredByAOI]:
    """Parse tool progress update into user-facing content."""
    plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_update.tool_call_id)
    update: ProgressUpdate | None = None
    match tool_update.update:
        case DataAnalysisToolProblemFormulatedUpdate():
            update, problem = _make_problem_formulated_update(tool_update.update)
            update_step_details(
                plan_manager, plan_id, step_id, chat_context=chat_context, data=problem
            )

        case DataAnalysisToolSolutionAttemptUpdate(attempt_num=attempt_num):
            if attempt_num == 1:
                title = "🔧 Subagent: Constructing solution..."
            else:
                title = (
                    f"⚠️ Subagent: Solution did not validate. Retrying... (attempt #{attempt_num})"
                )
            update = ProgressUpdate(title=title)

        case DataAnalysisToolSolutionGeneratedUpdate(solution=solution, pseudocode=pseudocode):
            old_details = plan_manager.get_step_details(plan_id, step_id)
            match old_details:
                case StepDetails(
                    content=DataAnalysisDetails(solution=DataAnalysisSolution(reasoning=reasoning))
                ):
                    solution_out = DataAnalysisSolution(
                        reasoning=reasoning,
                        analysis=solution.analysis,
                        pseudocode=pseudocode,
                        explanation=solution.explanation,
                    )
                    update_step_details(
                        plan_manager, plan_id, step_id, chat_context=chat_context, data=solution_out
                    )
                case _:
                    pass

            num_steps = len(solution.algorithm.statements)
            update = ProgressUpdate(title=f"🚀 Constructed {num_steps}-step solution. Executing...")

        case AlgorithmAgentThinkingDeltaUpdate(content_delta=content_delta):
            yield from _handle_algorithm_agent_thinking_delta_update(
                plan_manager=plan_manager,
                plan_id=plan_id,
                step_id=step_id,
                chat_context=chat_context,
                content_delta=content_delta,
            )

        case AlgorithmAgentThinkingCompleteUpdate(content=content):
            update = _handle_algorithm_agent_thinking_update(
                plan_manager=plan_manager,
                plan_id=plan_id,
                step_id=step_id,
                content=content,
                chat_context=chat_context,
            )

    if update is not None:
        plan_manager.update_step_progress(plan_id, step_id, update=update)
        yield plan_manager.get_plan(plan_id)


def _handle_execution_updates(
    plan_manager: PlanManager,
    tool_update: ToolProgressUpdate[_execution_update_types],
    chat_context: KmnlpChatContext,  # noqa: ARG001
) -> Generator[Plan | GeocodedAOI | FilteredByAOI]:
    """Parse tool progress update into user-facing content."""
    plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_update.tool_call_id)
    match tool_update.update:
        case DaskExecutionProgressUpdate(
            completed_tasks=completed_tasks, total_tasks=total_tasks, label=label
        ):
            completed_pct = (completed_tasks / total_tasks) * 100
            update = ProgressUpdate(title=f"⚙️ Dask: computing {label} | {completed_pct:.1f}%")

        case ComputingVariableUpdate(step_num=step_num, name=name):
            update = ProgressUpdate(title=f"⏳ Step {step_num + 1}: Computing variable: {name}")

        case ComputedVariableUpdate(step_num=step_num, name=name, value_summary=value_summary):
            title = f"✅ Step {step_num + 1}: Computed variable: {name}"
            if value_summary is not None:
                title += f" | {value_summary}"
            update = ProgressUpdate(title=title)

        case GeocodingSuccessUpdate(query=query, geojson=geojson):
            # Besides being added to the progress updates, the geocoded AOI
            # is also sent as a separate ChatResponse so that the frontend
            # can handle it specially (e.g. display on map).
            content = GeocodedAOI(
                query=query,
                label=f"{query} (geocoded)",
                geojson=GeoJSONFeatureCollection.from_serialized_geometry(geojson),
            )
            update = ProgressUpdate(title=f'📍 Geocoded "{query}"')
            yield content

        case SpatialSlicingUpdate(label=label, geojson=geojson):
            # FUTURE: maybe introduce a new specialized response type for this.
            # GeocodedAOI is good enough for now because it produces the intended effect in the UI.
            content = GeocodedAOI(
                query="",
                label=label,
                geojson=GeoJSONFeatureCollection.from_serialized_geometry(geojson),
            )
            update = ProgressUpdate(title=f"🗺️ Filtering data to {shape(geojson).wkt}")
            yield content

        case FilterByPolygonUpdate(geojson=geojson):
            # Besides being added to the progress updates, the geocoded AOI
            # is also sent as a separate ChatResponse so that the frontend
            # can handle it specially (e.g. display on map).
            content = FilteredByAOI(
                geojson=GeoJSONFeatureCollection.from_serialized_geometry(geojson),
            )
            update = ProgressUpdate(title="🗺️ Filtering data by polygon")
            yield content

        case AnimationProgressUpdate(frame_num=frame_num, total_frames=total_frames):
            update = ProgressUpdate(
                title=f"🎬 Processing animation frame {frame_num}/{total_frames}"
            )

    plan_manager.update_step_progress(plan_id, step_id, update=update)
    yield plan_manager.get_plan(plan_id)


def _handle_algorithm_agent_thinking_delta_update(
    plan_manager: PlanManager,
    plan_id: str,
    step_id: str,
    content_delta: str,
    chat_context: KmnlpChatContext,
) -> Generator[EphemeralPlan, None, None]:
    old_details = plan_manager.get_step_details(plan_id, step_id)
    update_text = "🤔 Subagent: Constructing solution: thinking..."
    match old_details:
        case StepDetails(content=DataAnalysisDetails(solution=None)):
            solution_out = DataAnalysisSolution(
                reasoning=content_delta,
                analysis="",
                pseudocode="",
                explanation="",
            )
            update_step_details(
                plan_manager, plan_id, step_id, chat_context=chat_context, data=solution_out
            )
            update = ProgressUpdate(title=update_text)
            plan_manager.update_step_progress(plan_id, step_id, update=update)
            yield EphemeralPlan.model_validate(plan_manager.get_plan(plan_id).model_dump())
            return
        case StepDetails(
            content=DataAnalysisDetails(
                solution=DataAnalysisSolution(
                    reasoning=reasoning,
                    analysis=analysis,
                    pseudocode=pseudocode,
                    explanation=explanation,
                )
            )
        ):
            solution_out = DataAnalysisSolution(
                reasoning=(content_delta if reasoning is None else reasoning + content_delta),
                analysis=analysis,
                pseudocode=pseudocode,
                explanation=explanation,
            )
            update_step_details(
                plan_manager, plan_id, step_id, chat_context=chat_context, data=solution_out
            )
            if reasoning is None:
                update = ProgressUpdate(title=update_text)
                plan_manager.update_step_progress(plan_id, step_id, update=update)
            yield EphemeralPlan.model_validate(plan_manager.get_plan(plan_id).model_dump())
            return
        case _:
            raise TypeError(f"Found old_details in unexpected state: {old_details}")


def _handle_algorithm_agent_thinking_update(
    plan_manager: PlanManager,
    plan_id: str,
    step_id: str,
    content: str,
    chat_context: KmnlpChatContext,
) -> ProgressUpdate:
    old_details = plan_manager.get_step_details(plan_id, step_id)
    match old_details:
        case StepDetails(content=DataAnalysisDetails(solution=None)):
            solution_out = DataAnalysisSolution(
                reasoning=content,
                analysis="",
                pseudocode="",
                explanation="",
            )
            update_step_details(
                plan_manager, plan_id, step_id, chat_context=chat_context, data=solution_out
            )
        case StepDetails(
            content=DataAnalysisDetails(
                solution=DataAnalysisSolution(
                    analysis=analysis,
                    pseudocode=pseudocode,
                    explanation=explanation,
                )
            )
        ):
            solution_out = DataAnalysisSolution(
                reasoning=content,
                analysis=analysis,
                pseudocode=pseudocode,
                explanation=explanation,
            )
            update_step_details(
                plan_manager, plan_id, step_id, chat_context=chat_context, data=solution_out
            )
        case _:
            raise TypeError(f"Found old_details in unexpected state: {old_details}")

    return ProgressUpdate(title="✍️ Subagent: Constructing solution: writing...")


def _make_problem_formulated_update(
    update: DataAnalysisToolProblemFormulatedUpdate,
) -> tuple[ProgressUpdate, DataAnalysisProblem]:
    def _to_var(v: InputVariable) -> Variable:
        if isinstance(v, GeneralDataArrayVariable):
            return Variable(
                name=v.name,
                type=f"Array w/ dims: {', '.join([d.field_name for d in v.dimensions])}",
                description=v.description,
            )
        return Variable(name=v.name, type=v.variable_type, description=v.description)

    problem = DataAnalysisProblem(
        statement=update.problem.statement,
        inputs_provided=[_to_var(v) for v in update.problem.inputs_provided],
        outputs_requested=[
            Variable(name=v.name, type=v.variable_type, description=v.description)
            for v in update.problem.outputs_requested
        ],
    )
    return ProgressUpdate(title="🧩 Formulated data analysis problem"), problem
