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
import json
import logging

from .agents import MainOrchestrator, PhaseOrchestrator, TaskAgent
from .config import Config
from .errors import HarnessError, TaskFailed
from .llm import LLMProvider
from .models import Phase, Plan, Status, Task
from .store import Store

logger = logging.getLogger("harness")


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

        has_failure = False
        for phase in plan.phases:
            if phase.status == Status.done:
                self.store.record_event(
                    "phase_skipped", message=f"phase {phase.id} already done; skipping",
                    phase_id=phase.id,
                )
                continue
            await self._run_phase(plan, phase, main)
            if phase.status == Status.failed:
                has_failure = True

        if has_failure:
            plan.status = Status.failed
            self.store.save_plan(plan)
            self.store.record_event(
                "run_failed", message="run failed — one or more phases failed",
                phases=len(plan.phases), tasks=sum(len(p.tasks) for p in plan.phases),
            )
        else:
            plan.status = Status.done
            self.store.save_plan(plan)
            final = await main.finalize(plan)
            self.store.write_doc("FINAL.md", final)
            self.store.record_event(
                "run_done", message="run complete -> docs/FINAL.md",
                phases=len(plan.phases), tasks=sum(len(p.tasks) for p in plan.phases),
            )

        # Record accumulated provider usage stats (token counts, call counts).
        if hasattr(self.provider, "stats"):
            self.store.record_event("usage", message="provider usage stats", **self.provider.stats.snapshot())
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

        results, failed_ids, skipped_ids = await self._schedule(plan, phase, po)

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

        # A phase is done only if every task completed; failed OR skipped tasks
        # (the latter possibly blocked by a cross-phase failure) mean it did not.
        if failed_ids or skipped_ids:
            phase.status = Status.failed
            self.store.record_event(
                "phase_failed",
                message=f"phase {phase.id} failed; failed={sorted(failed_ids)}, skipped={sorted(skipped_ids)}",
                phase_id=phase.id, index=phase.index, title=phase.title,
                failed=sorted(failed_ids), skipped=sorted(skipped_ids),
            )
        else:
            phase.status = Status.done
            self.store.record_event(
                "phase_done", message=f"phase {phase.id} done; docs updated",
                phase_id=phase.id, index=phase.index, title=phase.title,
            )
        await self._save(plan)
        del po  # explicit: phase orchestrator + its task agents are discarded here

    async def _schedule(self, plan: Plan, phase: Phase, po: PhaseOrchestrator) -> tuple[dict[str, str], set[str], set[str]]:
        """Run tasks in dependency order, concurrently within each ready wave.

        Returns (results, failed_ids, skipped_ids).
        """
        all_task_ids = {t.id for ph in plan.phases for t in ph.tasks}
        phase.topological_order(external_ids=all_task_ids)  # validate graph (cross-phase aware)
        sem = asyncio.Semaphore(self.config.defaults.max_concurrency)

        if self.config.defaults.max_concurrency > 1:
            logger.warning(
                "max_concurrency > 1 with provider: cli — concurrent agents edit "
                "the same directory; ensure tasks touch disjoint files."
            )

        by_id = {t.id: t for t in phase.tasks}
        results: dict[str, str] = {}
        remaining: set[str] = set()
        failed_ids: set[str] = set()
        skipped_ids: set[str] = set()
        failure_detail: dict[str, str] = {}  # task_id -> error string, for replanning

        # A dependency id may name a task in THIS phase (intra) or in an already
        # completed EARLIER phase (cross-phase). Since phases run in order, a
        # cross-phase dep is always in a terminal state by the time we get here.
        def _dep_blocked(dep_id: str) -> bool:
            if dep_id in by_id:
                return dep_id in failed_ids or dep_id in skipped_ids
            found = plan.task_by_id(dep_id)  # cross-phase: an earlier phase's task
            return found is not None and found[1].status in (Status.failed, Status.skipped)

        def _dep_ready(dep_id: str) -> bool:
            if dep_id in by_id:
                return dep_id in results
            found = plan.task_by_id(dep_id)
            return found is not None and found[1].status == Status.done

        def _dep_context(task: Task) -> list[tuple[Task, str]]:
            """Upstream (task, result) pairs, pulled from this phase or an earlier one."""
            ctx: list[tuple[Task, str]] = []
            for d in task.depends_on:
                if d in by_id:
                    ctx.append((by_id[d], results.get(d, "")))
                else:
                    found = plan.task_by_id(d)
                    if found is not None:
                        dep_phase, dep_task = found
                        ctx.append((dep_task, _read_result(self.store, dep_phase, dep_task)))
            return ctx

        for t in phase.tasks:
            if t.status == Status.done:
                results[t.id] = _read_result(self.store, phase, t)
            elif t.status in (Status.failed, Status.skipped):
                # Already terminated from a previous run — respect the state.
                if t.status == Status.failed:
                    failed_ids.add(t.id)
                else:
                    skipped_ids.add(t.id)
            else:
                remaining.add(t.id)

        while remaining:
            # Block dependents of failed/skipped tasks — mark them skipped.
            newly_skipped = set()
            for tid in list(remaining):
                task = by_id[tid]
                if any(_dep_blocked(d) for d in task.depends_on):
                    task.status = Status.skipped
                    skipped_ids.add(tid)
                    newly_skipped.add(tid)
                    self.store.record_event(
                        "task_skipped",
                        message=f"task {tid} skipped (depends on failed/skipped task)",
                        phase_id=phase.id, task_id=tid, title=task.title,
                    )
            remaining -= newly_skipped

            if not remaining:
                break

            ready = [tid for tid in remaining if all(_dep_ready(d) for d in by_id[tid].depends_on)]
            if not ready:
                # All remaining tasks have unmet dependencies (all blocked by failures).
                for tid in remaining:
                    by_id[tid].status = Status.skipped
                    skipped_ids.add(tid)
                    self.store.record_event(
                        "task_skipped",
                        message=f"task {tid} skipped (unmet dependencies among failed tasks)",
                        phase_id=phase.id, task_id=tid, title=by_id[tid].title,
                    )
                remaining.clear()
                break
            ready.sort()  # deterministic ordering within a wave

            async def run_one(tid: str) -> tuple[str, str | TaskFailed]:
                task = by_id[tid]
                deps = _dep_context(task)
                try:
                    res = await self._run_task(plan, phase, task, po, deps, sem)
                    return tid, res
                except TaskFailed as e:
                    return tid, e

            wave_results = await asyncio.gather(*(run_one(t) for t in ready))

            wave_failed: list[str] = []
            for tid, res in wave_results:
                if isinstance(res, TaskFailed):
                    failed_ids.add(tid)
                    wave_failed.append(tid)
                    failure_detail[tid] = res.detail
                    # Defensive: ensure the task is marked failed even for paths
                    # that raised without setting status (persisted at phase end).
                    if by_id[tid].status != Status.failed:
                        by_id[tid].status = Status.failed
                    remaining.discard(tid)
                else:
                    results[tid] = res
                    remaining.discard(tid)

            # After each wave: if any task failed and non-failed/non-skipped tasks
            # remain, ask the PhaseOrchestrator to adapt remaining task descriptions.
            if wave_failed and remaining:
                # Filter out tasks that are going to be skipped.
                replannable = [
                    by_id[t] for t in remaining
                    if not any(_dep_blocked(d) for d in by_id[t].depends_on)
                ]
                if replannable and wave_failed:
                    first_failed = by_id[wave_failed[0]]
                    updates = await po.replan(
                        phase,
                        first_failed,
                        failure_detail.get(wave_failed[0], first_failed.status.value),
                        replannable,
                    )
                    for task_id, new_desc in updates.items():
                        if task_id in by_id:
                            by_id[task_id].description = new_desc
                            self.store.record_event(
                                "task_replanned",
                                message=f"task {task_id} description updated after failure of {wave_failed[0]}",
                                phase_id=phase.id, task_id=task_id,
                            )

        return results, failed_ids, skipped_ids

    async def _run_task(
        self,
        plan: Plan,
        phase: Phase,
        task: Task,
        po: PhaseOrchestrator,
        deps: list[tuple[Task, str]],
        sem: asyncio.Semaphore,
    ) -> str:
        if task.requires_approval:
            await self._wait_for_approval(plan, task)

        async with sem:
            task.status = Status.in_progress
            await self._save(plan)
            self.store.record_event(
                "task_started", message=f"task {task.id} started",
                phase_id=phase.id, task_id=task.id, title=task.title,
                agent=task.runner.agent, model=task.runner.model,
            )

            # Phase Orchestrator defines the task's work -> TASK.md
            project_ctx = self.config.project_context
            task_def = await po.define_task(plan.goal, phase, task, project_context=project_ctx)
            self.store.write_task_def(phase, task, task_def)

            # Task Agent (its own choosable agent + model) executes -> result.md
            agent = TaskAgent(self.provider, task.runner, self.temp)

            def _on_output(chunk: str) -> None:
                # Streamed agent output — recorded (truncated) for the live dashboard/watch.
                # May be called from the PTY drain thread; record_event is lock-guarded.
                text = chunk.strip()
                if text:
                    self.store.record_event(
                        "task_output", message=text[-500:],
                        phase_id=phase.id, task_id=task.id,
                    )

            max_attempts = self.config.defaults.max_retries + 1
            last_error: str = ""
            for attempt in range(max_attempts):
                try:
                    result = await agent.execute(
                        plan.goal, phase, task, task_def, deps,
                        project_context=project_ctx, on_output=_on_output,
                    )
                    break  # success — exit retry loop
                except (RuntimeError, TimeoutError) as e:
                    last_error = str(e)
                    self.store.record_event(
                        "task_retry",
                        message=f"task {task.id} attempt {attempt + 1}/{max_attempts} failed: {last_error[:200]}",
                        phase_id=phase.id, task_id=task.id, attempt=attempt + 1,
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
            else:
                # All retries exhausted — mark failed and raise.
                task.status = Status.failed
                self.store.write_result(phase, task, f"error: {last_error}")
                await self._save(plan)
                self.store.record_event(
                    "task_failed",
                    message=f"task {task.id} failed after {max_attempts} attempt(s): {last_error[:200]}",
                    phase_id=phase.id, task_id=task.id, title=task.title,
                )
                raise TaskFailed(task.id, last_error)

            # Post-success validation (optional) — a non-zero exit fails the task.
            if task.validate_command:
                try:
                    result = await self._validate_task(task, result)
                except TaskFailed as e:
                    task.status = Status.failed
                    self.store.write_result(phase, task, f"error: {e.detail}\n\n{result}")
                    await self._save(plan)
                    raise

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

    async def _wait_for_approval(self, plan: Plan, task: Task) -> None:
        """Write a HITL checkpoint and block until `harness resume` approves it."""
        checkpoint = self.store.hitl_path(task.id)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps({"status": "waiting", "task_id": task.id, "title": task.title})
        )

        task.status = Status.waiting
        await self._save(plan)

        self.store.record_event(
            "task_waiting",
            message=f"task {task.id} awaiting human approval",
            task_id=task.id, title=task.title,
        )
        logger.info(
            "Task [%s] '%s' requires approval. "
            "Run: harness resume --task %s --project %s",
            task.id, task.title, task.id, self.store.root,
        )
        print(
            f"\n[harness] Task [{task.id}] '{task.title}' requires approval.\n"
            f"  Run: harness resume --task {task.id} --project {self.store.root}\n"
            f"  Waiting…\n",
            flush=True,
        )

        timeout = self.config.defaults.hitl_timeout
        elapsed = 0.0
        while True:
            try:
                data = json.loads(checkpoint.read_text())
                if data.get("status") == "approved":
                    break
            except (json.JSONDecodeError, OSError):
                pass

            if timeout > 0 and elapsed >= timeout:
                task.status = Status.failed
                await self._save(plan)
                self.store.record_event(
                    "task_timeout",
                    message=f"task {task.id} timed out waiting for approval after {timeout}s",
                    task_id=task.id,
                )
                try:
                    checkpoint.unlink()
                except OSError:
                    pass
                raise TaskFailed(task.id, f"HITL approval timed out after {timeout}s")

            await asyncio.sleep(2.0)
            elapsed += 2.0

        try:
            checkpoint.unlink()
        except OSError:
            pass
        self.store.record_event(
            "task_approved", message=f"task {task.id} approved", task_id=task.id
        )

    async def _validate_task(self, task: Task, result: str) -> str:
        """Run task.validate_command; raise TaskFailed on non-zero exit."""
        assert task.validate_command  # caller guarantees this
        try:
            proc = await asyncio.create_subprocess_exec(
                *task.validate_command,
                cwd=str(self.store.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            raise TaskFailed(task.id, "validate_command timed out after 120s")
        if proc.returncode != 0:
            detail = (
                err.decode("utf-8", "replace") + out.decode("utf-8", "replace")
            ).strip()[-400:]
            self.store.record_event(
                "task_validation_failed",
                message=f"task {task.id} validate_command exited {proc.returncode}",
                task_id=task.id,
            )
            raise TaskFailed(task.id, f"validate_command exited {proc.returncode}: {detail}")
        self.store.record_event(
            "task_validated", message=f"task {task.id} validate_command passed", task_id=task.id
        )
        return result

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
