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
      archive/<YYYY-MM-DDTHH-MM-SS>/
        plan/  phases/  runs/  docs/  # snapshot of the previous run

The ``plan.json`` manifest is the source of truth; the ``*.md`` files are
human-readable renderings/definitions.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import threading
from pathlib import Path

from .models import Phase, Plan, Task

PLAN_DIR = "plan"
PHASES_DIR = "phases"
DOCS_DIR = "docs"
RUNS_DIR = "runs"
HITL_DIR = "hitl"
MANIFEST = "plan.json"


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically: write a sibling .tmp then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


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
        self._log_lock = threading.Lock()  # serialize run.log appends across threads

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
    def hitl_dir(self) -> Path:
        return self.base / HITL_DIR

    def hitl_path(self, task_id: str) -> Path:
        return self.hitl_dir / f"{task_id}.json"

    @property
    def run_pid_path(self) -> Path:
        return self.runs_dir / "run.pid"

    def write_run_pid(self, pid: int) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.run_pid_path, str(pid))

    def read_run_pid(self) -> int | None:
        try:
            return int(self.run_pid_path.read_text().strip())
        except (OSError, ValueError):
            return None

    def clear_run_pid(self) -> None:
        try:
            self.run_pid_path.unlink()
        except OSError:
            pass

    def list_waiting_checkpoints(self) -> list[dict]:
        """Return all HITL checkpoints with status 'waiting'."""
        results: list[dict] = []
        if not self.hitl_dir.exists():
            return results
        for p in self.hitl_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                if data.get("status") == "waiting":
                    results.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return results

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

    def archive_current_run(self) -> Path | None:
        """Move the current run's artifacts to .harness/archive/<created_at>/.

        Moves plan/, phases/, runs/ entirely and the run-specific docs
        (OVERVIEW.md, PROGRESS.md, FINAL.md).  Project-level docs
        (ARCHITECTURE.md, PRD.md) stay in place.

        Returns the archive directory path, or None if there was no plan to archive.
        """
        if not self.has_plan():
            return None

        try:
            ts = Plan.model_validate_json(self.manifest_path.read_text()).created_at
            # Normalise to a filesystem-safe name: 2025-06-30T14-22-00
            ts = ts[:19].replace(":", "-")
        except Exception:
            ts = _utcnow()[:19].replace(":", "-")

        archive_dir = self.base / "archive" / ts
        # Avoid clobbering an existing archive slot (e.g. two plans in one second)
        suffix, attempt = "", 1
        while (archive_dir.parent / (ts + suffix)).exists():
            attempt += 1
            suffix = f"-{attempt}"
        archive_dir = archive_dir.parent / (ts + suffix)
        archive_dir.mkdir(parents=True, exist_ok=True)

        for name in ("plan", "phases", "runs"):
            src = self.base / name
            if src.exists():
                shutil.move(str(src), str(archive_dir / name))

        run_docs = ("OVERVIEW.md", "PROGRESS.md", "FINAL.md")
        for doc_name in run_docs:
            src = self.docs_dir / doc_name
            if src.exists():
                (archive_dir / "docs").mkdir(exist_ok=True)
                shutil.move(str(src), str(archive_dir / "docs" / doc_name))

        return archive_dir

    # ---- manifest --------------------------------------------------------
    def save_plan(self, plan: Plan) -> None:
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.manifest_path, plan.model_dump_json(indent=2))

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
        _atomic_write(p, f"# Brief\n\n{goal.strip()}\n")
        return p

    def write_plan_md(self, plan: Plan) -> Path:
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        p = self.plan_dir / "PLAN.md"
        _atomic_write(p, render_plan_md(plan))
        return p

    def write_phase_md(self, phase: Phase, content: str) -> Path:
        d = self.phase_dir(phase)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "PHASE.md"
        _atomic_write(p, content)
        return p

    def write_task_def(self, phase: Phase, task: Task, content: str) -> Path:
        d = self.task_dir(phase, task)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "TASK.md"
        _atomic_write(p, content)
        return p

    def write_result(self, phase: Phase, task: Task, content: str) -> Path:
        d = self.task_dir(phase, task)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "result.md"
        _atomic_write(p, content)
        return p

    def write_artifact(self, phase: Phase, task: Task, name: str, content: str) -> Path:
        d = self.task_output_dir(phase, task)
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(p, content)
        return p

    def write_phase_summary(self, phase: Phase, content: str) -> Path:
        d = self.phase_dir(phase)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "phase-summary.md"
        _atomic_write(p, content)
        return p

    def write_doc(self, name: str, content: str) -> Path:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        p = self.docs_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(p, content)
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

        Thread safety: guarded by ``self._log_lock``. Most callers run in the
        asyncio event loop (single-threaded, cooperative), but streamed
        ``task_output`` events arrive from the PTY drain worker thread, so the
        lock is needed to keep appends from interleaving.
        """
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        rec: dict[str, object] = {"ts": _utcnow(), "type": event_type}
        if message:
            rec["msg"] = message
        rec.update(fields)
        line = json.dumps(rec) + "\n"
        with self._log_lock, (self.runs_dir / "run.log").open("a") as f:
            f.write(line)

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
        lines.append("")
    return "\n".join(lines)
