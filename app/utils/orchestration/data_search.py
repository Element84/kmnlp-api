import logging
from collections.abc import Generator

from llm_agent.context import ToolProgressUpdate
from llm_agent.model.dataset import Dataset
from llm_agent.tools.data_search_tool import (
    DataSearchToolResult,
    SearchCompletedUpdate,
    SearchInitiatedUpdate,
)
from llm_agent.tools.fetch_datasets_tool import KnowledgeGraphQueryUpdate, QueryResultsUpdate

from app.types.output import (
    DatasetMetadata,
    DatasetSearchDetails,
    FilteredByAOI,
    GeocodedAOI,
    Plan,
    ProgressUpdate,
    StepDetails,
)
from app.utils.plan_manager import PlanManager

log = logging.getLogger(__name__)


def tool_output_to_response_content(
    plan_manager: PlanManager, tool_call_id: str, data: DataSearchToolResult
) -> Plan:
    """Map data search tool output to user-facing content."""
    plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_call_id)
    update_step_details(plan_manager, plan_id, step_id, data)
    return plan_manager.get_plan(plan_id)


def update_step_details(
    plan_manager: PlanManager, plan_id: str, step_id: str, data: DataSearchToolResult
) -> None:
    """Update plan step details based on data search tool outputs."""

    def _get_old_details(
        plan_manager: PlanManager, plan_id: str, step_id: str
    ) -> DatasetSearchDetails:
        details = plan_manager.get_step_details(plan_id, step_id)
        if details is None:
            return DatasetSearchDetails(query="N/A")
        if not isinstance(details.content, DatasetSearchDetails):
            raise TypeError("Step details content is not of type DatasetSearchDetails")
        return details.content

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
    log.info("New details for plan_id=%s, step_id=%s: %s", plan_id, step_id, new_details)

    plan_manager.set_step_details(plan_id, step_id, details=StepDetails(content=new_details))


def tool_progress_update_to_response_content(
    plan_manager: PlanManager,
    tool_update: ToolProgressUpdate[
        SearchInitiatedUpdate
        | KnowledgeGraphQueryUpdate
        | QueryResultsUpdate
        | SearchCompletedUpdate
    ],
) -> Generator[Plan | GeocodedAOI | FilteredByAOI]:
    """Parse tool progress update into user-facing content."""
    plan_id, step_id = plan_manager.tool_call_id_to_plan_step_ids(tool_update.tool_call_id)
    match tool_update.update:
        case SearchInitiatedUpdate(query=query):
            update = ProgressUpdate(title=f'🔍 Searching for "{query}"')
            details = StepDetails(content=DatasetSearchDetails(query=query))
            plan_manager.set_step_details(plan_id, step_id, details=details)

        case KnowledgeGraphQueryUpdate(
            id_filter=id_filter,
            temporal_filter=temporal_filter,
            regex_filter=regex_filter,
        ):
            parts: list[str] = []
            if id_filter:
                parts.append(f"with IDs {id_filter}")
            if temporal_filter:
                parts.append(f"intersecting time range {temporal_filter}")
            if regex_filter:
                parts.append(f"matching regex {regex_filter!r}")

            if parts:
                update = ProgressUpdate(
                    title=f"🕸️ Subagent: Querying knowledge graph {', '.join(parts)}"
                )
            else:
                update = ProgressUpdate(
                    title="🕸️ Subagent: Querying knowledge graph with no filters"
                )

        case QueryResultsUpdate(
            num_datasets=num_datasets,
            execution_time_sec=execution_time_sec,
            detailed_view=detailed_view,
        ):
            title = f"📊 Subagent: Found {num_datasets} results in {execution_time_sec:.2f} seconds"
            if num_datasets > 0:
                title += f". Inspecting {'expanded' if detailed_view else 'summarized'} metadata..."
            update = ProgressUpdate(title=title)

        case SearchCompletedUpdate(num_datasets=num_datasets):
            update = ProgressUpdate(title=f"✅ Search completed with {num_datasets} results")

    plan_manager.update_step_progress(plan_id, step_id, update=update)
    yield plan_manager.get_plan(plan_id)
