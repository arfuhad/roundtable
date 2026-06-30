"""Orchestration engine: the deterministic control flow tying agents together.

Run shape (after the plan is approved)::

    Main.kickoff -> docs/OVERVIEW.md
    for each phase (in order, resumable):
        Phase Orchestrator (FRESH context):
            define each task     -> tasks/.../TASK.md
            schedule tasks by dependency, run Task Agents in concurrent waves
                                 -> tasks/.../result.md (+ output/)
            summarize phase      -> phase-summary.md
        Main.integrate_phase(summary only) -> docs/PROGRESS.md     # context clean
    Main.finalize -> docs/FINAL.md

Context-cleaning invariant: a Phase Orchestrator and its Task Agents are created
per phase and dropped after the summary; the Main Orchestrator only ever receives
the phase *summary*, never task transcripts.
"""

from __future__ import annotations

import asyncio

from .agents import MainOrchestrator, PhaseOrchestrator, TaskAgent
from .config import Config
from .errors import HarnessError
from .llm import LLMProvider
from .models import Phase, Plan, Status, Task
from .store import Store


class Engine:
    def __init__(self, store: Store, config: Config, provider: LLMProvider):
        self.store = store
        self.config = config
        self.provider = provider
        self.temp = config.defaults.temperature
        self._save_lock = asyncio.Lock()

    async def run(self) -> Plan:
        plan = self.store.load_plan()
        if not plan.approved:
            raise HarnessError("plan is not approved; run `harness approve` first")

        self.store.scaffold_plan_tree(plan)
        plan.status = Status.in_progress
        self.store.save_plan(plan)
        self.store.record_event(
            "run_started", message="run started", goal=plan.goal,
            phases=len(plan.phases), tasks=sum(len(p.tasks) for p in plan.phases),
        )

        main = MainOrchestrator(self.provider, plan.main_runner, self.temp)

        if not self.store.read_doc("OVERVIEW.md"):
            overview = await main.kickoff(plan)
            self.store.write_doc("OVERVIEW.md", overview)
            self.store.record_event("kickoff", message="main kickoff -> docs/OVERVIEW.md")

        for phase in plan.phases:
            if phase.status == Status.done:
                self.store.record_event(
                    "phase_skipped", message=f"phase {phase.id} already done; skipping",
                    phase_id=phase.id,
                )
                continue
            await self._run_phase(plan, phase, main)

        plan.status = Status.done
        self.store.save_plan(plan)
        final = await main.finalize(plan)
        self.store.write_doc("FINAL.md", final)
        self.store.record_event(
            "run_done", message="run complete -> docs/FINAL.md",
            phases=len(plan.phases), tasks=sum(len(p.tasks) for p in plan.phases),
        )
        return plan

    # ------------------------------------------------------------------ #
    async def _run_phase(self, plan: Plan, phase: Phase, main: MainOrchestrator) -> None:
        phase.status = Status.in_progress
        await self._save(plan)
        self.store.write_phase_md(phase, _phase_md(phase))
        self.store.record_event(
            "phase_started", message=f"phase {phase.id} started",
            phase_id=phase.id, index=phase.index, title=phase.title, runner=str(phase.runner),
        )

        # Fresh Phase Orchestrator — its context lives only for this phase.
        po = PhaseOrchestrator(self.provider, phase.runner, self.temp)

        results = await self._schedule(plan, phase, po)

        ordered = [(t, results.get(t.id, "")) for t in phase.tasks]
        summary = await po.summarize(phase, ordered)
        self.store.write_phase_summary(phase, summary)
        phase.summary_path = str(self.store.phase_dir(phase) / "phase-summary.md")
        self.store.record_event(
            "phase_summarized", message=f"phase {phase.id} summarized", phase_id=phase.id,
        )

        # Context clean: Main receives ONLY the summary string.
        entry = await main.integrate_phase(plan.goal, phase, summary)
        self._append_progress(phase, entry)

        phase.status = Status.done
        await self._save(plan)
        self.store.record_event(
            "phase_done", message=f"phase {phase.id} done; docs updated",
            phase_id=phase.id, index=phase.index, title=phase.title,
        )
        del po  # explicit: phase orchestrator + its task agents are discarded here

    async def _schedule(self, plan: Plan, phase: Phase, po: PhaseOrchestrator) -> dict[str, str]:
        """Run tasks in dependency order, concurrently within each ready wave."""
        phase.topological_order()  # validate graph up front (raises on cycle/unknown dep)
        sem = asyncio.Semaphore(self.config.defaults.max_concurrency)
        by_id = {t.id: t for t in phase.tasks}
        results: dict[str, str] = {}
        remaining: set[str] = set()

        for t in phase.tasks:
            if t.status == Status.done:
                results[t.id] = _read_result(self.store, phase, t)
            else:
                remaining.add(t.id)

        while remaining:
            ready = [tid for tid in remaining if all(d in results for d in by_id[tid].depends_on)]
            if not ready:
                raise HarnessError(
                    f"phase {phase.id}: unmet dependencies among {sorted(remaining)}"
                )
            ready.sort()  # deterministic ordering within a wave

            async def run_one(tid: str) -> tuple[str, str]:
                task = by_id[tid]
                deps = [(by_id[d], results[d]) for d in task.depends_on]
                res = await self._run_task(plan, phase, task, po, deps, sem)
                return tid, res

            for tid, res in await asyncio.gather(*(run_one(t) for t in ready)):
                results[tid] = res
                remaining.discard(tid)

        return results

    async def _run_task(
        self,
        plan: Plan,
        phase: Phase,
        task: Task,
        po: PhaseOrchestrator,
        deps: list[tuple[Task, str]],
        sem: asyncio.Semaphore,
    ) -> str:
        async with sem:
            task.status = Status.in_progress
            await self._save(plan)
            self.store.record_event(
                "task_started", message=f"task {task.id} started",
                phase_id=phase.id, task_id=task.id, title=task.title,
                agent=task.runner.agent, model=task.runner.model,
            )

            # Phase Orchestrator defines the task's work -> TASK.md
            task_def = await po.define_task(plan.goal, phase, task)
            self.store.write_task_def(phase, task, task_def)

            # Task Agent (its own choosable agent + model) executes -> result.md
            agent = TaskAgent(self.provider, task.runner, self.temp)
            
            result = ""
            for attempt in range(3):
                result = await agent.execute(plan.goal, phase, task, task_def, deps)
                s = result.strip()
                if s and not s.lower().startswith("error:") and not s.lower().startswith("exception:"):
                    break
                
                self.store.record_event(
                    "task_retry", message=f"task {task.id} returned suspicious output, retrying",
                    phase_id=phase.id, task_id=task.id,
                )
                if attempt < 2:
                    await asyncio.sleep(1.0)
                    
            self.store.write_result(phase, task, result)

            task.status = Status.done
            task.result_path = str(self.store.task_dir(phase, task) / "result.md")
            await self._save(plan)
            self.store.record_event(
                "task_done", message=f"task {task.id} done",
                phase_id=phase.id, task_id=task.id, title=task.title,
                agent=task.runner.agent, model=task.runner.model,
            )
            return result

    # ------------------------------------------------------------------ #
    async def _save(self, plan: Plan) -> None:
        async with self._save_lock:
            self.store.save_plan(plan)

    def _append_progress(self, phase: Phase, entry: str) -> None:
        prior = self.store.read_doc("PROGRESS.md") or "# Progress\n\n"
        block = f"## Phase {phase.index}: {phase.title}\n\n{entry.strip()}\n\n"
        self.store.write_doc("PROGRESS.md", prior + block)


def _phase_md(phase: Phase) -> str:
    lines = [f"# Phase {phase.index}: {phase.title}", "", phase.objective, "",
             f"- Orchestrator: `{phase.runner}`", "", "## Tasks", ""]
    for i, t in enumerate(phase.tasks, start=1):
        dep = f" (depends on {', '.join(t.depends_on)})" if t.depends_on else ""
        lines.append(f"{i}. **{t.title}** `[{t.id}]` — `{t.runner}`{dep}")
    return "\n".join(lines) + "\n"


def _read_result(store: Store, phase: Phase, task: Task) -> str:
    p = store.task_dir(phase, task) / "result.md"
    return p.read_text() if p.exists() else ""
