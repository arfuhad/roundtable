"""The agent roles: Planner, Main Orchestrator, Phase Orchestrator, Task Agent.

Each agent is a thin wrapper binding a *choosable* model to a role. Control flow
lives in the engine; these classes only turn inputs into the LLM call for their
role and return the produced text (Markdown, or JSON for the planner).
"""

from __future__ import annotations

from typing import Callable

from . import prompts
from .llm import LLMProvider, extract_json
from .models import AgentRef, Phase, Plan, Task
from .prompts import render_prompt


def _truncate(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."


def _models_meta(defaults: dict[str, AgentRef]) -> dict[str, str]:
    """Stringify role -> AgentRef so the (scripted) backend gets plain strings."""
    return {k: str(v) for k, v in defaults.items()}


class Planner:
    def __init__(self, provider: LLMProvider, ref: AgentRef, temperature: float = 0.2):
        self.provider, self.ref, self.temperature = provider, ref, temperature

    async def create_plan(
        self, goal: str, allowed_models: list[str], defaults: dict[str, AgentRef]
    ) -> Plan:
        user = (
            f"GOAL:\n{goal}\n\n"
            f"ALLOWED (agent:model): {', '.join(allowed_models)}\n"
            f"ROLE DEFAULTS: phase orchestrator -> {defaults.get('phase')}, "
            f"task agent -> {defaults.get('task')}\n\n"
            "Produce the plan JSON now."
        )
        raw = await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=prompts.PLANNER_SYSTEM, user=user,
            json_mode=True, temperature=self.temperature, role="planner",
            meta={"models": _models_meta(defaults)},
        )
        return Plan.model_validate(extract_json(raw))

    async def structure_plan(
        self, existing: str, allowed_models: list[str], defaults: dict[str, AgentRef]
    ) -> Plan:
        """Convert an existing free-form plan / PRD into the harness schema."""
        user = (
            f"EXISTING PLAN / PRD:\n{existing}\n\n"
            f"ALLOWED (agent:model): {', '.join(allowed_models)}\n"
            f"ROLE DEFAULTS: phase orchestrator -> {defaults.get('phase')}, "
            f"task agent -> {defaults.get('task')}\n\n"
            "Convert it into the plan JSON now."
        )
        raw = await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=prompts.PLAN_IMPORT_SYSTEM, user=user,
            json_mode=True, temperature=self.temperature, role="planner",
            meta={"models": _models_meta(defaults)},
        )
        return Plan.model_validate(extract_json(raw))


class Analyst:
    """Maps an existing codebase: an architecture overview, then a PRD.

    Both calls take the (already size-bounded) codebase digest from ``scan``.
    The PRD call also gets the architecture overview for grounding.
    """

    def __init__(self, provider: LLMProvider, ref: AgentRef, temperature: float = 0.2):
        self.provider, self.ref, self.temperature = provider, ref, temperature

    async def architecture(self, digest: str) -> str:
        user = f"CODEBASE DIGEST:\n{digest}\n\nWrite the architecture overview."
        return await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=prompts.MAP_ARCH_SYSTEM, user=user,
            temperature=self.temperature, role="map_arch", meta={"digest_bytes": len(digest)},
        )

    async def prd(self, digest: str, architecture: str) -> str:
        user = (
            f"CODEBASE DIGEST:\n{digest}\n\n"
            f"ARCHITECTURE OVERVIEW:\n{_truncate(architecture, 8000)}\n\n"
            "Write the reverse-engineered PRD."
        )
        return await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=prompts.MAP_PRD_SYSTEM, user=user,
            temperature=self.temperature, role="map_prd", meta={"digest_bytes": len(digest)},
        )


class MainOrchestrator:
    def __init__(self, provider: LLMProvider, ref: AgentRef, temperature: float = 0.2):
        self.provider, self.ref, self.temperature = provider, ref, temperature

    async def kickoff(self, plan: Plan) -> str:
        roadmap = "\n".join(f"- {p.index}. {p.title}: {p.objective}" for p in plan.phases)
        user = f"GOAL:\n{plan.goal}\n\nPHASE ROADMAP:\n{roadmap}\n\nWrite the overview."
        return await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=prompts.MAIN_KICKOFF_SYSTEM, user=user,
            temperature=self.temperature, role="main_kickoff", meta={"goal": plan.goal},
        )

    async def integrate_phase(self, goal: str, phase: Phase, summary: str) -> str:
        """Main sees ONLY the phase summary here — never raw task transcripts."""
        user = (
            f"GOAL:\n{goal}\n\nCOMPLETED PHASE: {phase.index}. {phase.title}\n\n"
            f"PHASE SUMMARY:\n{_truncate(summary)}\n\nWrite the progress entry."
        )
        return await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=prompts.MAIN_INTEGRATE_SYSTEM, user=user,
            temperature=self.temperature, role="main_integrate",
            meta={"goal": goal, "phase_title": phase.title},
        )

    async def finalize(self, plan: Plan) -> str:
        roadmap = "\n".join(f"- {p.index}. {p.title}" for p in plan.phases)
        user = f"GOAL:\n{plan.goal}\n\nPHASES:\n{roadmap}\n\nWrite the final report."
        return await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=prompts.MAIN_FINALIZE_SYSTEM, user=user,
            temperature=self.temperature, role="main_finalize", meta={"goal": plan.goal},
        )


class PhaseOrchestrator:
    """Instantiated fresh per phase (its context is discarded after summarize)."""

    def __init__(self, provider: LLMProvider, ref: AgentRef, temperature: float = 0.2):
        self.provider, self.ref, self.temperature = provider, ref, temperature

    async def define_task(self, goal: str, phase: Phase, task: Task, *, project_context: str = "") -> str:
        user = (
            f"PROJECT GOAL:\n{goal}\n\nPHASE: {phase.title} — {phase.objective}\n\n"
            f"TASK: {task.title}\nDESCRIPTION: {task.description}\n\n"
            "Write the work definition."
        )
        system = render_prompt(prompts.PHASE_DEFINE_SYSTEM, project_context=project_context)
        return await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=system, user=user,
            temperature=self.temperature, role="phase_define",
            meta={"task_title": task.title, "phase_title": phase.title},
        )

    async def replan(
        self,
        phase: Phase,
        failed_task: Task,
        failure_output: str,
        remaining: list[Task],
    ) -> dict[str, str]:
        """After a task failure, return updated descriptions for remaining tasks."""
        remaining_lines = "\n".join(f"- [{t.id}] {t.title}: {t.description}" for t in remaining)
        user = (
            f"PHASE: {phase.title} — {phase.objective}\n\n"
            f"FAILED TASK: [{failed_task.id}] {failed_task.title}\n"
            f"FAILURE OUTPUT:\n{_truncate(failure_output, 1000)}\n\n"
            f"REMAINING TASKS:\n{remaining_lines}\n\n"
            "Return a JSON object mapping task_id to updated description for tasks "
            "that need adjustment. Return {{}} if no changes are needed."
        )
        raw = await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent,
            system=prompts.PHASE_REPLAN_SYSTEM, user=user,
            json_mode=True, temperature=self.temperature, role="phase_replan",
            meta={"phase_title": phase.title, "failed_task_id": failed_task.id},
        )
        try:
            result = extract_json(raw)
        except ValueError:
            return {}
        if not isinstance(result, dict):  # a model may return a list; ignore it
            return {}
        return {k: v for k, v in result.items() if isinstance(k, str) and isinstance(v, str)}

    async def summarize(self, phase: Phase, results: list[tuple[Task, str]]) -> str:
        body = "\n\n".join(
            f"### {t.title} [{t.id}]\n{_truncate(r, 1500)}" for t, r in results
        )
        user = (
            f"PHASE: {phase.title} — {phase.objective}\n\n"
            f"TASK RESULTS:\n{body}\n\nWrite the phase summary for the main orchestrator."
        )
        return await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=prompts.PHASE_SUMMARY_SYSTEM, user=user,
            temperature=self.temperature, role="phase_summary",
            meta={"phase_title": phase.title, "task_count": len(results)},
        )


class TaskAgent:
    def __init__(self, provider: LLMProvider, ref: AgentRef, temperature: float = 0.2):
        self.provider, self.ref, self.temperature = provider, ref, temperature

    async def execute(
        self, goal: str, phase: Phase, task: Task, task_def: str, deps: list[tuple[Task, str]],
        *, project_context: str = "", on_output: Callable[[str], None] | None = None,
    ) -> str:
        deps_ctx = (
            "\n\n".join(f"### Dependency {t.title} [{t.id}]\n{_truncate(r, 1200)}" for t, r in deps)
            if deps
            else "(none)"
        )
        user = (
            f"PROJECT GOAL:\n{goal}\n\nPHASE: {phase.title}\n\n"
            f"WORK DEFINITION:\n{task_def}\n\nUPSTREAM RESULTS:\n{deps_ctx}\n\n"
            "Execute the task and produce the deliverable."
        )
        system = render_prompt(prompts.TASK_EXEC_SYSTEM, project_context=project_context)
        return await self.provider.complete(
            model=self.ref.model, agent=self.ref.agent, system=system, user=user,
            temperature=self.temperature, role="task_exec",
            meta={"task_title": task.title, "phase_title": phase.title},
            on_output=on_output,
        )
