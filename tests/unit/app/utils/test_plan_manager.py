import pytest
from llm_agent.tools.response_shaping_tools.agent_plan_tool import (
    AgentPlan,
    AgentPlanUpdate,
    AgentStep,
    AgentStepUpdate,
)

from app.types.output import (
    DataAnalysisDetails,
    DataAnalysisProblem,
    DataAnalysisSolution,
    DatasetSearchDetails,
    ProgressUpdate,
    StepDetails,
    Variable,
)
from app.utils.plan_manager import (
    PlanIDMismatchError,
    PlanManager,
    SinglePlanManager,
    UnknownPlanError,
    UnknownStepIDError,
    UnknownToolCallIDError,
)


class TestSinglePlanManager:
    def test_updates(self):
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[
                AgentStep(id="step-1", title="The first step", status="in_progress"),
                AgentStep(id="step-2", title="The second step", status="pending"),
            ],
        )
        mgr = SinglePlanManager.from_agent_plan(agent_plan)
        agent_plan_update = AgentPlanUpdate(
            content_type="plan_update",
            id="plan-1",
            step_updates=[
                AgentStepUpdate(id="step-1", status="success", outcome_summary="Done."),
                AgentStepUpdate(id="step-2", status="in_progress"),
            ],
        )
        mgr.update(agent_plan_update)
        assert mgr.plan.model_dump(mode="json") == {
            "content_type": "Plan",
            "id": "plan-1",
            "title": "The Plan",
            "steps": [
                {
                    "id": "step-1",
                    "title": "The first step",
                    "status": "success",
                    "outcome_summary": "Done.",
                    "details": None,
                    "progress_updates": None,
                },
                {
                    "id": "step-2",
                    "title": "The second step",
                    "status": "in_progress",
                    "outcome_summary": None,
                    "details": None,
                    "progress_updates": None,
                },
            ],
        }

    def test_plan_id_error(self):
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[
                AgentStep(id="step-1", title="The first step", status="in_progress"),
            ],
        )
        mgr = SinglePlanManager.from_agent_plan(agent_plan)
        agent_plan_update = AgentPlanUpdate(
            content_type="plan_update",
            id="plan-2",
            step_updates=[
                AgentStepUpdate(id="step-1", status="success", outcome_summary="Done."),
            ],
        )
        with pytest.raises(PlanIDMismatchError):
            mgr.update(agent_plan_update)

    def test_step_id_error(self):
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[
                AgentStep(id="step-1", title="The first step", status="in_progress"),
            ],
        )
        mgr = SinglePlanManager.from_agent_plan(agent_plan)
        agent_plan_update = AgentPlanUpdate(
            content_type="plan_update",
            id="plan-1",
            step_updates=[
                AgentStepUpdate(id="step-10", status="success", outcome_summary="Done."),
            ],
        )
        with pytest.raises(UnknownStepIDError):
            mgr.update(agent_plan_update)


class TestPlanManager:
    """Test cases for PlanManager class."""

    def test_add_plan(self):
        """Test adding a plan to the manager."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[
                AgentStep(id="step-1", title="The first step", status="pending"),
                AgentStep(id="step-2", title="The second step", status="pending"),
            ],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        plan = mgr.get_plan("plan-1")
        assert plan.model_dump(mode="json") == {
            "content_type": "Plan",
            "id": "plan-1",
            "title": "The Plan",
            "steps": [
                {
                    "id": "step-1",
                    "title": "The first step",
                    "status": "pending",
                    "outcome_summary": None,
                    "details": None,
                    "progress_updates": None,
                },
                {
                    "id": "step-2",
                    "title": "The second step",
                    "status": "pending",
                    "outcome_summary": None,
                    "details": None,
                    "progress_updates": None,
                },
            ],
        }

    def test_add_multiple_plans(self):
        """Test adding multiple plans to the manager."""
        agent_plan_1 = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="First Plan",
            goal="Complete first plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        agent_plan_2 = AgentPlan(
            content_type="plan",
            id="plan-2",
            title="Second Plan",
            goal="Complete second plan.",
            steps=[AgentStep(id="step-2", title="Second step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan_1)
        mgr.add_plan(agent_plan_2)

        plan_1 = mgr.get_plan("plan-1")
        plan_2 = mgr.get_plan("plan-2")

        assert plan_1.id == "plan-1"
        assert plan_1.title == "First Plan"
        assert plan_2.id == "plan-2"
        assert plan_2.title == "Second Plan"

    def test_update_plan(self):
        """Test updating a plan in the manager."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[
                AgentStep(id="step-1", title="The first step", status="in_progress"),
                AgentStep(id="step-2", title="The second step", status="pending"),
            ],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        agent_plan_update = AgentPlanUpdate(
            content_type="plan_update",
            id="plan-1",
            step_updates=[
                AgentStepUpdate(id="step-1", status="success", outcome_summary="Completed."),
                AgentStepUpdate(id="step-2", status="in_progress"),
            ],
        )
        mgr.update_plan(agent_plan_update)

        plan = mgr.get_plan("plan-1")
        assert plan.steps[0].status == "success"
        assert plan.steps[0].outcome_summary == "Completed."
        assert plan.steps[1].status == "in_progress"

    def test_get_plan_unknown_plan_error(self):
        """Test getting a plan that doesn't exist raises UnknownPlanError."""
        mgr = PlanManager()
        with pytest.raises(UnknownPlanError) as exc_info:
            mgr.get_plan("nonexistent-plan")
        assert "No plan with ID nonexistent-plan exists." in str(exc_info.value)

    def test_update_plan_unknown_plan_error(self):
        """Test updating a plan that doesn't exist raises UnknownPlanError."""
        mgr = PlanManager()
        agent_plan_update = AgentPlanUpdate(
            content_type="plan_update",
            id="nonexistent-plan",
            step_updates=[
                AgentStepUpdate(id="step-1", status="success"),
            ],
        )
        with pytest.raises(UnknownPlanError) as exc_info:
            mgr.update_plan(agent_plan_update)
        assert "No plan with ID nonexistent-plan exists." in str(exc_info.value)

    def test_register_tool_call(self):
        """Test registering a tool call with a plan and step."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[
                AgentStep(id="step-1", title="The first step", status="pending"),
                AgentStep(id="step-2", title="The second step", status="pending"),
            ],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        mgr.register_tool_call("tool-call-1", "plan-1", "step-1")
        mgr.register_tool_call("tool-call-2", "plan-1", "step-2")

        plan_id, step_id = mgr.tool_call_id_to_plan_step_ids("tool-call-1")
        assert plan_id == "plan-1"
        assert step_id == "step-1"

        plan_id, step_id = mgr.tool_call_id_to_plan_step_ids("tool-call-2")
        assert plan_id == "plan-1"
        assert step_id == "step-2"

    def test_tool_call_id_to_plan_step_ids_unknown_tool_call_error(self):
        """Test getting plan/step IDs for unknown tool call raises UnknownToolCallIDError."""
        mgr = PlanManager()
        with pytest.raises(UnknownToolCallIDError) as exc_info:
            mgr.tool_call_id_to_plan_step_ids("nonexistent-tool-call")
        assert "No tool call with ID nonexistent-tool-call exists." in str(exc_info.value)

    def test_multiple_plans_with_tool_calls(self):
        """Test managing multiple plans with different tool calls."""
        agent_plan_1 = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="First Plan",
            goal="Complete first plan.",
            steps=[
                AgentStep(id="step-1", title="First step", status="pending"),
                AgentStep(id="step-2", title="Second step", status="pending"),
            ],
        )
        agent_plan_2 = AgentPlan(
            content_type="plan",
            id="plan-2",
            title="Second Plan",
            goal="Complete second plan.",
            steps=[
                AgentStep(id="step-a", title="Step A", status="pending"),
                AgentStep(id="step-b", title="Step B", status="pending"),
            ],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan_1)
        mgr.add_plan(agent_plan_2)

        # Register tool calls for different plans
        mgr.register_tool_call("tool-1", "plan-1", "step-1")
        mgr.register_tool_call("tool-2", "plan-2", "step-a")
        mgr.register_tool_call("tool-3", "plan-1", "step-2")

        # Verify tool call mappings
        plan_id, step_id = mgr.tool_call_id_to_plan_step_ids("tool-1")
        assert plan_id == "plan-1"
        assert step_id == "step-1"

        plan_id, step_id = mgr.tool_call_id_to_plan_step_ids("tool-2")
        assert plan_id == "plan-2"
        assert step_id == "step-a"

        plan_id, step_id = mgr.tool_call_id_to_plan_step_ids("tool-3")
        assert plan_id == "plan-1"
        assert step_id == "step-2"

    def test_update_multiple_plans(self):
        """Test updating multiple plans independently."""
        agent_plan_1 = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="First Plan",
            goal="Complete first plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        agent_plan_2 = AgentPlan(
            content_type="plan",
            id="plan-2",
            title="Second Plan",
            goal="Complete second plan.",
            steps=[AgentStep(id="step-2", title="Second step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan_1)
        mgr.add_plan(agent_plan_2)

        # Update first plan
        update_1 = AgentPlanUpdate(
            content_type="plan_update",
            id="plan-1",
            step_updates=[AgentStepUpdate(id="step-1", status="success", outcome_summary="Done 1")],
        )
        mgr.update_plan(update_1)

        # Update second plan
        update_2 = AgentPlanUpdate(
            content_type="plan_update",
            id="plan-2",
            step_updates=[AgentStepUpdate(id="step-2", status="in_progress")],
        )
        mgr.update_plan(update_2)

        # Verify updates
        plan_1 = mgr.get_plan("plan-1")
        plan_2 = mgr.get_plan("plan-2")

        assert plan_1.steps[0].status == "success"
        assert plan_1.steps[0].outcome_summary == "Done 1"
        assert plan_2.steps[0].status == "in_progress"
        assert plan_2.steps[0].outcome_summary is None

    def test_update_step_progress(self):
        """Test updating step progress with a progress update."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[
                AgentStep(id="step-1", title="The first step", status="pending"),
                AgentStep(id="step-2", title="The second step", status="pending"),
            ],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        # Create a progress update
        progress_update = ProgressUpdate(title="Geocoding completed")

        # Update step progress
        mgr.update_step_progress("plan-1", "step-1", progress_update)

        # Verify the progress update was added
        plan = mgr.get_plan("plan-1")
        assert plan.model_dump(mode="json") == {
            "content_type": "Plan",
            "id": "plan-1",
            "title": "The Plan",
            "steps": [
                {
                    "id": "step-1",
                    "title": "The first step",
                    "status": "in_progress",
                    "outcome_summary": None,
                    "details": None,
                    "progress_updates": [
                        {
                            "content_type": "ProgressUpdate",
                            "title": "Geocoding completed",
                        }
                    ],
                },
                {
                    "id": "step-2",
                    "title": "The second step",
                    "status": "pending",
                    "outcome_summary": None,
                    "details": None,
                    "progress_updates": None,
                },
            ],
        }

    def test_update_step_progress_multiple_updates(self):
        """Test adding multiple progress updates to the same step."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="The first step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        # Add first progress update
        progress_update_1 = ProgressUpdate(title="First update")
        mgr.update_step_progress("plan-1", "step-1", progress_update_1)

        # Add second progress update
        progress_update_2 = ProgressUpdate(title="Second update")
        mgr.update_step_progress("plan-1", "step-1", progress_update_2)

        # Verify both updates are present
        plan = mgr.get_plan("plan-1")
        assert plan.model_dump(mode="json") == {
            "content_type": "Plan",
            "id": "plan-1",
            "title": "The Plan",
            "steps": [
                {
                    "id": "step-1",
                    "title": "The first step",
                    "status": "in_progress",
                    "outcome_summary": None,
                    "details": None,
                    "progress_updates": [
                        {
                            "content_type": "ProgressUpdate",
                            "title": "First update",
                        },
                        {
                            "content_type": "ProgressUpdate",
                            "title": "Second update",
                        },
                    ],
                }
            ],
        }

    def test_update_step_progress_unknown_plan_error(self):
        """Test updating progress for a plan that doesn't exist raises UnknownPlanError."""
        mgr = PlanManager()
        progress_update = ProgressUpdate(title="Test update")

        with pytest.raises(UnknownPlanError) as exc_info:
            mgr.update_step_progress("nonexistent-plan", "step-1", progress_update)
        assert "No plan with ID nonexistent-plan exists." in str(exc_info.value)

    def test_update_step_progress_unknown_step_error(self):
        """Test updating progress for a step that doesn't exist raises UnknownStepIDError."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="The first step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        progress_update = ProgressUpdate(title="Test update")

        with pytest.raises(UnknownStepIDError) as exc_info:
            mgr.update_step_progress("plan-1", "nonexistent-step", progress_update)
        expected_msg = (
            "Step ID in update (nonexistent-step) does not match "
            "any step in current plan (['step-1'])."
        )
        assert expected_msg in str(exc_info.value)

    def test_update_step_progress_multiple_plans(self):
        """Test updating progress for steps in different plans."""
        agent_plan_1 = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="First Plan",
            goal="Complete first plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        agent_plan_2 = AgentPlan(
            content_type="plan",
            id="plan-2",
            title="Second Plan",
            goal="Complete second plan.",
            steps=[AgentStep(id="step-2", title="Second step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan_1)
        mgr.add_plan(agent_plan_2)

        # Update progress for step in first plan
        progress_update_1 = ProgressUpdate(title="Plan 1 update")
        mgr.update_step_progress("plan-1", "step-1", progress_update_1)

        # Update progress for step in second plan
        progress_update_2 = ProgressUpdate(title="Plan 2 update")
        mgr.update_step_progress("plan-2", "step-2", progress_update_2)

        # Verify updates are correctly associated with their respective plans
        plan_1 = mgr.get_plan("plan-1")
        plan_2 = mgr.get_plan("plan-2")

        assert plan_1.model_dump(mode="json") == {
            "content_type": "Plan",
            "id": "plan-1",
            "title": "First Plan",
            "steps": [
                {
                    "id": "step-1",
                    "title": "First step",
                    "status": "in_progress",
                    "outcome_summary": None,
                    "details": None,
                    "progress_updates": [
                        {
                            "content_type": "ProgressUpdate",
                            "title": "Plan 1 update",
                        }
                    ],
                }
            ],
        }
        assert plan_2.model_dump(mode="json") == {
            "content_type": "Plan",
            "id": "plan-2",
            "title": "Second Plan",
            "steps": [
                {
                    "id": "step-2",
                    "title": "Second step",
                    "status": "in_progress",
                    "outcome_summary": None,
                    "details": None,
                    "progress_updates": [
                        {
                            "content_type": "ProgressUpdate",
                            "title": "Plan 2 update",
                        }
                    ],
                }
            ],
        }

    def test_update_step_progress_changes_pending_to_in_progress(self):
        """Test that adding a progress update changes step status from pending to in_progress."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[
                AgentStep(id="step-1", title="The first step", status="pending"),
                AgentStep(id="step-2", title="The second step", status="in_progress"),
            ],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        # Verify initial status
        plan = mgr.get_plan("plan-1")
        assert plan.steps[0].status == "pending"
        assert plan.steps[1].status == "in_progress"

        # Create a progress update
        progress_update = ProgressUpdate(title="Starting data processing")

        # Update step progress for the pending step
        mgr.update_step_progress("plan-1", "step-1", progress_update)

        # Verify that the pending step is now in_progress
        updated_plan = mgr.get_plan("plan-1")
        assert updated_plan.steps[0].status == "in_progress"
        assert updated_plan.steps[1].status == "in_progress"  # Should remain unchanged

        # Verify progress update was added
        assert updated_plan.steps[0].progress_updates is not None
        assert len(updated_plan.steps[0].progress_updates) == 1
        assert updated_plan.steps[0].progress_updates[0].title == "Starting data processing"

    def test_update_step_progress_does_not_change_non_pending_status(self):
        """Test that adding a progress update doesn't change non-pending step statuses."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[
                AgentStep(id="step-1", title="In progress step", status="in_progress"),
                AgentStep(id="step-2", title="Success step", status="success"),
                AgentStep(id="step-3", title="Failure step", status="failure"),
            ],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        # Create a progress update
        progress_update = ProgressUpdate(title="Additional update")

        # Update step progress for each step
        mgr.update_step_progress("plan-1", "step-1", progress_update)
        mgr.update_step_progress("plan-1", "step-2", progress_update)
        mgr.update_step_progress("plan-1", "step-3", progress_update)

        # Verify that non-pending statuses remain unchanged
        updated_plan = mgr.get_plan("plan-1")
        assert updated_plan.steps[0].status == "in_progress"  # Should remain in_progress
        assert updated_plan.steps[1].status == "success"  # Should remain success
        assert updated_plan.steps[2].status == "failure"  # Should remain failure

        # Verify progress updates were added to all steps
        for step in updated_plan.steps:
            assert step.progress_updates is not None
            assert len(step.progress_updates) == 1
            assert step.progress_updates[0].title == "Additional update"

    def test_get_step_details_none(self):
        """Test getting step details when no details are set."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        details = mgr.get_step_details("plan-1", "step-1")
        assert details is None

    def test_get_step_details_with_dataset_search_details(self):
        """Test getting step details with DatasetSearchDetails."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        # Set details
        details = StepDetails(
            content=DatasetSearchDetails(
                query="temperature data",
                results=None,
            )
        )
        mgr.set_step_details("plan-1", "step-1", details)

        # Get details
        retrieved_details = mgr.get_step_details("plan-1", "step-1")
        assert retrieved_details is not None
        assert retrieved_details.model_dump(mode="json") == {
            "content_type": "StepDetails",
            "content": {
                "content_type": "DatasetSearchDetails",
                "query": "temperature data",
                "results": None,
            },
        }

    def test_get_step_details_with_data_analysis_details(self):
        """Test getting step details with DataAnalysisDetails."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        # Set details
        problem = DataAnalysisProblem(
            statement="Calculate average temperature",
            inputs_provided=[Variable(name="analysed_sst", type="DataArray", description="")],
            outputs_requested=[Variable(name="mean_sst", type="scalar_float", description="")],
        )
        solution = DataAnalysisSolution(
            thoughts="Sum all values and divide by count",
            pseudocode="avg = sum(temps) / len(temps)",
            explanation="Simple arithmetic mean calculation",
        )
        details = StepDetails(
            content=DataAnalysisDetails(
                problem=problem,
                solution=solution,
            )
        )
        mgr.set_step_details("plan-1", "step-1", details)

        # Get details
        retrieved_details = mgr.get_step_details("plan-1", "step-1")
        assert retrieved_details is not None
        assert retrieved_details.model_dump(mode="json") == {
            "content_type": "StepDetails",
            "content": {
                "content_type": "DataAnalysisDetails",
                "problem": {
                    "content_type": "DataAnalysisProblem",
                    "statement": "Calculate average temperature",
                    "inputs_provided": [
                        {
                            "name": "analysed_sst",
                            "type": "DataArray",
                            "description": "",
                        }
                    ],
                    "outputs_requested": [
                        {
                            "name": "mean_sst",
                            "type": "scalar_float",
                            "description": "",
                        }
                    ],
                },
                "solution": {
                    "content_type": "DataAnalysisSolution",
                    "thoughts": "Sum all values and divide by count",
                    "pseudocode": "avg = sum(temps) / len(temps)",
                    "explanation": "Simple arithmetic mean calculation",
                },
                "outputs": None,
            },
        }

    def test_set_step_details(self):
        """Test setting step details."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        # Set details
        details = StepDetails(
            content=DatasetSearchDetails(
                query="ocean temperature",
                results=None,
            )
        )
        mgr.set_step_details("plan-1", "step-1", details)

        # Verify details were set
        plan = mgr.get_plan("plan-1")
        assert plan.steps[0].details is not None
        assert plan.steps[0].details.model_dump(mode="json") == {
            "content_type": "StepDetails",
            "content": {
                "content_type": "DatasetSearchDetails",
                "query": "ocean temperature",
                "results": None,
            },
        }

    def test_set_step_details_overwrites_existing(self):
        """Test that setting step details overwrites existing details."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        # Set initial details
        details1 = StepDetails(
            content=DatasetSearchDetails(
                query="first query",
                results=None,
            )
        )
        mgr.set_step_details("plan-1", "step-1", details1)

        # Overwrite with new details
        details2 = StepDetails(
            content=DatasetSearchDetails(
                query="second query",
                results=None,
            )
        )
        mgr.set_step_details("plan-1", "step-1", details2)

        # Verify new details replaced old ones
        retrieved_details = mgr.get_step_details("plan-1", "step-1")
        assert retrieved_details is not None
        assert isinstance(retrieved_details.content, DatasetSearchDetails)
        assert retrieved_details.content.query == "second query"

    def test_get_step_details_unknown_plan_error(self):
        """Test getting step details for a plan that doesn't exist raises UnknownPlanError."""
        mgr = PlanManager()
        with pytest.raises(UnknownPlanError) as exc_info:
            mgr.get_step_details("nonexistent-plan", "step-1")
        assert "No plan with ID nonexistent-plan exists." in str(exc_info.value)

    def test_get_step_details_unknown_step_error(self):
        """Test getting step details for a step that doesn't exist raises UnknownStepIDError."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        with pytest.raises(UnknownStepIDError) as exc_info:
            mgr.get_step_details("plan-1", "nonexistent-step")
        expected_msg = (
            "Step ID in update (nonexistent-step) does not match "
            "any step in current plan (['step-1'])."
        )
        assert expected_msg in str(exc_info.value)

    def test_set_step_details_unknown_plan_error(self):
        """Test setting step details for a plan that doesn't exist raises UnknownPlanError."""
        mgr = PlanManager()
        details = StepDetails(
            content=DatasetSearchDetails(
                query="test query",
                results=None,
            )
        )
        with pytest.raises(UnknownPlanError) as exc_info:
            mgr.set_step_details("nonexistent-plan", "step-1", details)
        assert "No plan with ID nonexistent-plan exists." in str(exc_info.value)

    def test_set_step_details_unknown_step_error(self):
        """Test setting step details for a step that doesn't exist raises UnknownStepIDError."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        details = StepDetails(
            content=DatasetSearchDetails(
                query="test query",
                results=None,
            )
        )
        with pytest.raises(UnknownStepIDError) as exc_info:
            mgr.set_step_details("plan-1", "nonexistent-step", details)
        expected_msg = (
            "Step ID in update (nonexistent-step) does not match "
            "any step in current plan (['step-1'])."
        )
        assert expected_msg in str(exc_info.value)

    def test_get_step_details_returns_deep_copy(self):
        """Test that get_step_details returns a deep copy of the details."""
        agent_plan = AgentPlan(
            content_type="plan",
            id="plan-1",
            title="The Plan",
            goal="Carry out the plan.",
            steps=[AgentStep(id="step-1", title="First step", status="pending")],
        )
        mgr = PlanManager()
        mgr.add_plan(agent_plan)

        details = StepDetails(
            content=DatasetSearchDetails(
                query="test query",
                results=None,
            )
        )
        mgr.set_step_details("plan-1", "step-1", details)

        retrieved_details = mgr.get_step_details("plan-1", "step-1")
        assert retrieved_details is not None
        assert isinstance(retrieved_details.content, DatasetSearchDetails)
        assert retrieved_details.content.query == "test query"

        # Modify the retrieved details and ensure it does not affect the stored details
        retrieved_details.content.query = "modified query"
        original_details = mgr.get_step_details("plan-1", "step-1")
        assert original_details is not None
        assert isinstance(original_details.content, DatasetSearchDetails)
        assert original_details.content.query == "test query"
