from typing import Self

import langfuse
from langfuse import observe  # pyright: ignore[reportUnknownVariableType]
from llm_agent.tools.agent_plan_tool import AgentPlan, AgentPlanUpdate, AgentStepUpdate
from pydantic import BaseModel, Field

from app.types.output import Plan, ProgressUpdate, Step, StepDetails


class UnknownPlanError(Exception):
    """No plan with the given ID exists."""

    def __init__(self, plan_id: str, known_plans_ids: list[str]) -> None:  # noqa: D107
        super().__init__(f"No plan with ID {plan_id} exists. Known plan IDs: {known_plans_ids}")


class PlanIDMismatchError(Exception):
    """Plan ID in update does not match current plan."""

    def __init__(self, plan_id_curr: str, plan_id_recvd: str) -> None:  # noqa: D107
        super().__init__(
            f"Plan ID in update ({plan_id_recvd}) does not match current plan ({plan_id_curr})."
        )


class UnknownStepIDError(Exception):
    """Step ID in update does not match any step in current plan."""

    def __init__(self, plan: Plan, step_id_recvd: str) -> None:  # noqa: D107
        super().__init__(
            f"Step ID in update ({step_id_recvd}) does not "
            f"match any step in current plan ({[step.id for step in plan.steps]})."
        )


class UnknownToolCallIDError(Exception):
    """No tool call with the given ID exists."""

    def __init__(self, tool_call_id: str) -> None:  # noqa: D107
        super().__init__(f"No tool call with ID {tool_call_id} exists.")


class SinglePlanManager(BaseModel):
    """Class for managing a plan's state."""

    plan: Plan

    @classmethod
    def from_agent_plan(cls, agent_plan: AgentPlan) -> Self:
        """Create SinglePlanManager from an initial AgentPlan.

        Args:
            agent_plan (AgentPlan): Initial plan.

        Returns:
            SinglePlanManager: SinglePlanManager instance.
        """
        steps = [
            Step(
                id=step.id,
                title=step.title,
                status=step.status,
                outcome_summary=step.outcome_summary,
            )
            for step in agent_plan.steps
        ]
        plan = Plan(
            id=agent_plan.id,
            title=agent_plan.title,
            steps=steps,
        )
        return cls(plan=plan)

    @observe(as_type="chain", name="SinglePlanManager.update")
    def update(self, agent_plan_update: AgentPlanUpdate) -> None:
        """Update plan state.

        Args:
            agent_plan_update (AgentPlanUpdate): Plan update.

        Raises:
            PlanIDMismatchError: If the plan ID in the update does not match the current plan.
            UnknownStepIDError: If the step ID in the update does not match any step in the
                current plan.
        """
        if agent_plan_update.id != self.id:
            raise PlanIDMismatchError(self.plan.id, agent_plan_update.id)

        for step_update in agent_plan_update.step_updates:
            self._update_step(step_update)

    def get_step(self, step_id: str) -> Step:
        """Get a step by ID.

        Args:
            step_id (str): Step ID.

        Returns:
            Step: Step with the given ID.

        Raises:
            UnknownStepIDError: If the step ID does not match any step in the current plan.
        """
        for step in self.plan.steps:
            if step.id == step_id:
                return step
        raise UnknownStepIDError(self.plan, step_id)

    @observe(as_type="chain", name="SinglePlanManager.update_step_progress")
    def update_step_progress(self, step_id: str, update: ProgressUpdate) -> None:
        """Update a step's progress updates.

        Args:
            step_id (str): Step ID.
            update (ProgressUpdate): Progress update to add.

        Raises:
            UnknownStepIDError: If the step ID is not known.
        """
        step = self.get_step(step_id)

        if step.progress_updates is None:
            step.progress_updates = []

        if step.status == "pending":
            step.status = "in_progress"

        step.progress_updates.append(update)

    def _update_step(self, step_update: AgentStepUpdate) -> None:
        """Update a single step's state.

        Args:
            step_update (AgentStepUpdate): Step update.

        Raises:
            UnknownStepIDError: If the step ID in the update does not match any step in the
                current plan.
        """
        step = self.get_step(step_update.id)

        if step_update.status is not None:
            step.status = step_update.status

        if step_update.outcome_summary is not None:
            step.outcome_summary = step_update.outcome_summary

    @observe(as_type="chain", name="SinglePlanManager.cancel")
    def cancel(self) -> bool:
        """Cancel all incomplete steps in the current plan.

        Returns:
            bool: True if any steps were cancelled, False otherwise.
        """
        cancelled_any = False
        for step in self.plan.steps:
            if step.status in {"pending", "in_progress"}:
                step.status = "cancelled"
                cancelled_any = True
        return cancelled_any

    @property
    def id(self) -> str:
        """Plan ID."""
        return self.plan.id


class PlanManager(BaseModel):
    """Class for managing multiple plans."""

    plans: dict[str, SinglePlanManager] = Field(default_factory=dict)
    tool_call_id_to_plan_step_id: dict[str, tuple[str, str]] = Field(default_factory=dict)

    @observe(as_type="chain", name="PlanManager.add_plan")
    def add_plan(self, agent_plan: AgentPlan) -> None:
        """Add a new plan.

        Args:
            agent_plan (AgentPlan): Initial plan.
        """
        if agent_plan.id not in self.plans:
            self.plans[agent_plan.id] = SinglePlanManager.from_agent_plan(agent_plan)
        else:
            # Replace the placeholder plan with the actual plan, merging any step updates
            existing_plan_manager = self.plans[agent_plan.id]
            new_plan_manager = SinglePlanManager.from_agent_plan(agent_plan)
            for existing_step in existing_plan_manager.plan.steps:
                new_step = new_plan_manager.get_step(existing_step.id)
                if new_step.status == "pending":
                    new_step.status = existing_step.status
                if existing_step.progress_updates:
                    for update in existing_step.progress_updates:
                        new_plan_manager.update_step_progress(new_step.id, update)
            self.plans[agent_plan.id] = new_plan_manager

    def update_plan(self, agent_plan_update: AgentPlanUpdate) -> None:
        """Update a plan's state.

        Args:
            agent_plan_update (AgentPlanUpdate): Plan update.

        Raises:
            UnknownPlanError: If the plan ID is not known.
        """
        plan_manager = self.get_plan_manager(agent_plan_update.id)
        plan_manager.update(agent_plan_update)

    @observe(as_type="chain", name="PlanManager.register_tool_call")
    def register_tool_call(self, tool_call_id: str, plan_id: str, step_id: str) -> None:
        """Register a tool call ID with a step ID in a plan.

        Args:
            tool_call_id (str): Tool call ID.
            plan_id (str): Plan ID.
            step_id (str): Step ID.
        """
        self.tool_call_id_to_plan_step_id[tool_call_id] = (plan_id, step_id)
        if plan_id not in self.plans:
            # Tool call has preceded the creation of the plan, so we create a placeholder plan
            self.plans[plan_id] = SinglePlanManager(
                plan=Plan(
                    id=plan_id,
                    title="",
                    steps=[Step(id=step_id, title="", status="in_progress")],
                )
            )

    def get_plan(self, plan_id: str) -> Plan:
        """Get the current state of a plan.

        Args:
            plan_id (str): Plan ID.

        Returns:
            Plan: Current state of the plan.

        Raises:
            UnknownPlanError: If the plan ID is not known.
        """
        plan_manager = self.get_plan_manager(plan_id)
        return plan_manager.plan.model_copy(deep=True)

    def tool_call_id_to_plan_step_ids(self, tool_call_id: str) -> tuple[str, str]:
        """Get the plan ID and step ID associated with a tool call ID.

        Args:
            tool_call_id (str): Tool call ID.

        Returns:
            tuple[str, str]: Tuple of plan ID and step ID.

        Raises:
            UnknownToolCallIDError: If the tool call ID is not known.
        """
        try:
            return self.tool_call_id_to_plan_step_id[tool_call_id]
        except KeyError as e:
            raise UnknownToolCallIDError(tool_call_id) from e

    def update_step_progress(self, plan_id: str, step_id: str, update: ProgressUpdate) -> None:
        """Update a step's progress updates.

        Args:
            plan_id (str): Plan ID.
            step_id (str): Step ID.
            update (ProgressUpdate): Progress update to add.

        Raises:
            UnknownPlanError: If the plan ID is not known.
            UnknownStepIDError: If the step ID is not known.
        """
        plan_manager = self.get_plan_manager(plan_id)
        plan_manager.update_step_progress(step_id, update)

    def get_step_details(self, plan_id: str, step_id: str) -> StepDetails | None:
        """Get a step's details.

        Args:
            plan_id (str): Plan ID.
            step_id (str): Step ID.

        Returns:
            StepDetails | None: The details of the step.

        Raises:
            UnknownPlanError: If the plan ID is not known.
            UnknownStepIDError: If the step ID is not known.
        """
        plan_manager = self.get_plan_manager(plan_id)
        step = plan_manager.get_step(step_id)
        return step.details.model_copy(deep=True) if step.details else None

    def set_step_details(self, plan_id: str, step_id: str, details: StepDetails) -> None:
        """Set a step's details.

        Args:
            plan_id (str): Plan ID.
            step_id (str): Step ID.
            details (StepDetails): Details to set.

        Raises:
            UnknownPlanError: If the plan ID is not known.
            UnknownStepIDError: If the step ID is not known.
        """
        langfuse.get_client().update_current_span(
            input={
                "plan_id": plan_id,
                "step_id": step_id,
                "details": {"content": type(details.content)},
            }
        )
        plan_manager = self.get_plan_manager(plan_id)
        step = plan_manager.get_step(step_id)
        step.details = details

    def get_plan_manager(self, plan_id: str) -> SinglePlanManager:
        """Get the SinglePlanManager for a given plan ID.

        Args:
            plan_id (str): Plan ID.

        Returns:
            SinglePlanManager: The plan manager for the given plan ID.
        """
        try:
            return self.plans[plan_id]
        except KeyError as e:
            raise UnknownPlanError(plan_id, list(self.plans.keys())) from e

    def cancel_all(self) -> list[Plan]:
        """Cancel all incomplete steps in all plans.

        Returns:
            list[Plan]: A list of plans that had any steps cancelled.
        """
        return [
            plan_manager.plan.model_copy(deep=True)
            for plan_manager in self.plans.values()
            if plan_manager.cancel()
        ]
