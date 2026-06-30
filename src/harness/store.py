"""Filesystem persistence and project layout.

Layout::

    <root>/
      harness.config.yaml
      plan/{BRIEF.md, PLAN.md, plan.json}
      phases/phase-NN-<slug>/
        PHASE.md
        phase-summary.md            # Phase Orchestrator -> Main report
        tasks/task-NN-<slug>/
          TASK.md                   # the task's work definition
          result.md                 # the agent's completed work
          output/                   # artifacts produced by the agent
      docs/                         # maintained by the Main Orchestrator
      runs/run.log                  # append-only event log

The ``plan.json`` manifest is the source of truth; the ``*.md`` files are
human-readable renderings/definitions.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .models import Phase, Plan, Task

PLAN_DIR = "plan"
PHASES_DIR = "phases"
DOCS_DIR = "docs"
RUNS_DIR = "runs"
MANIFEST = "plan.json"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class Store:
    """All disk reads/writes for one harness project.

    ``root`` is the target project directory (where agents do their work).
    All harness artifacts live under ``root/workdir`` (default ``.harness``) so
    they never clutter an existing repo. Only ``harness.config.yaml`` sits at the
    project root.
    """

    def __init__(self, root: Path | str, workdir: str = ".harness"):
        self.root = Path(root).resolve()
        self.base = self.root / workdir

    # ---- paths -----------------------------------------------------------
    @property
    def plan_dir(self) -> Path:
        return self.base / PLAN_DIR

    @property
    def phases_dir(self) -> Path:
        return self.base / PHASES_DIR

    @property
    def docs_dir(self) -> Path:
        return self.base / DOCS_DIR

    @property
    def runs_dir(self) -> Path:
        return self.base / RUNS_DIR

    @property
    def manifest_path(self) -> Path:
        return self.plan_dir / MANIFEST

    def phase_dir(self, phase: Phase) -> Path:
        return self.phases_dir / phase.dir_name

    def _task_index(self, phase: Phase, task: Task) -> int:
        for i, t in enumerate(phase.tasks, start=1):
            if t.id == task.id:
                return i
        raise KeyError(f"task {task.id!r} not in phase {phase.id!r}")

    def task_dir(self, phase: Phase, task: Task) -> Path:
        idx = self._task_index(phase, task)
        return self.phase_dir(phase) / "tasks" / phase.task_dir_name(task, idx)

    def task_output_dir(self, phase: Phase, task: Task) -> Path:
        return self.task_dir(phase, task) / "output"

    # ---- scaffolding -----------------------------------------------------
    def scaffold(self) -> None:
        for d in (self.plan_dir, self.phases_dir, self.docs_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def scaffold_plan_tree(self, plan: Plan) -> None:
        """Create every phase/task directory for an approved plan."""
        self.scaffold()
        for phase in plan.phases:
            self.phase_dir(phase).mkdir(parents=True, exist_ok=True)
            for task in phase.tasks:
                self.task_output_dir(phase, task).mkdir(parents=True, exist_ok=True)

    # ---- manifest --------------------------------------------------------
    def save_plan(self, plan: Plan) -> None:
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(plan.model_dump_json(indent=2))

    def load_plan(self) -> Plan:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"no plan manifest at {self.manifest_path}; run `harness plan` first"
            )
        return Plan.model_validate_json(self.manifest_path.read_text())

    def has_plan(self) -> bool:
        return self.manifest_path.exists()

    # ---- documents -------------------------------------------------------
    def write_brief(self, goal: str) -> Path:
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        p = self.plan_dir / "BRIEF.md"
        p.write_text(f"# Brief\n\n{goal.strip()}\n")
        return p

    def write_plan_md(self, plan: Plan) -> Path:
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        p = self.plan_dir / "PLAN.md"
        p.write_text(render_plan_md(plan))
        return p

    def write_phase_md(self, phase: Phase, content: str) -> Path:
        d = self.phase_dir(phase)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "PHASE.md"
        p.write_text(content)
        return p

    def write_task_def(self, phase: Phase, task: Task, content: str) -> Path:
        d = self.task_dir(phase, task)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "TASK.md"
        p.write_text(content)
        return p

    def write_result(self, phase: Phase, task: Task, content: str) -> Path:
        d = self.task_dir(phase, task)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "result.md"
        p.write_text(content)
        return p

    def write_artifact(self, phase: Phase, task: Task, name: str, content: str) -> Path:
        d = self.task_output_dir(phase, task)
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def write_phase_summary(self, phase: Phase, content: str) -> Path:
        d = self.phase_dir(phase)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "phase-summary.md"
        p.write_text(content)
        return p

    def write_doc(self, name: str, content: str) -> Path:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        p = self.docs_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def read_doc(self, name: str) -> str:
        p = self.docs_dir / name
        return p.read_text() if p.exists() else ""

    # ---- event log -------------------------------------------------------
    def record_event(self, event_type: str, message: str = "", **fields: object) -> None:
        """Append one structured JSONL event to ``runs/run.log``.

        ``type`` + ``fields`` (e.g. ``task_id``, ``agent``, ``model``) let the
        dashboard reconstruct live activity and timings; ``msg`` stays
        human-readable for tailing the log.
        """
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        rec: dict[str, object] = {"ts": _utcnow(), "type": event_type}
        if message:
            rec["msg"] = message
        rec.update(fields)
        with (self.runs_dir / "run.log").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_event(self, message: str) -> None:
        """Back-compat: a plain human log line (records as a ``log`` event)."""
        self.record_event("log", message=message)

    def read_events(self) -> list[dict]:
        """All recorded events in order; malformed/partial lines are skipped."""
        p = self.runs_dir / "run.log"
        if not p.exists():
            return []
        events: list[dict] = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a concurrent writer's partial last line
        return events


def render_plan_md(plan: Plan) -> str:
    lines = [
        "# Plan",
        "",
        f"**Goal:** {plan.goal}",
        "",
        f"- Created: {plan.created_at or 'n/a'}",
        f"- Main orchestrator: `{plan.main_runner or 'n/a'}`",
        f"- Approved: {plan.approved}",
        "",
        "## Phases",
        "",
    ]
    for phase in plan.phases:
        lines.append(f"### Phase {phase.index}: {phase.title}  `[{phase.id}]`")
        if phase.objective:
            lines.append(f"\n{phase.objective}\n")
        lines.append(f"- Phase orchestrator: `{phase.runner or 'n/a'}`")
        lines.append("")
        for i, task in enumerate(phase.tasks, start=1):
            dep = f" (depends on: {', '.join(task.depends_on)})" if task.depends_on else ""
            lines.append(f"{i}. **{task.title}** `[{task.id}]` — `{task.runner or 'n/a'}`{dep}")
            if task.description:
                lines.append(f"   - {task.description}")
            for st in task.subtasks:
                lines.append(f"   - [ ] {st.description}")
        lines.append("")
    return "\n".join(lines)
