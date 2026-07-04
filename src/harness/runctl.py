"""Run control shared by the CLI, the MCP server, and the REST API.

One place owns the ``run.pid`` protocol:

* a live run writes its pid to ``.harness/runs/run.pid`` and clears it on exit;
* starting a run (attached or detached) first checks for a live pid and refuses
  to double-launch;
* ``stop_run`` SIGTERMs the recorded pid.

Detached launches (MCP / REST) spawn ``python -m harness.cli run`` in a new
session so the run outlives the server that started it; its output is appended
to ``.harness/runs/run.out`` for later inspection. Plan generation can likewise
be launched detached (``start_plan``) since planning may take minutes.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .errors import HarnessError
from .store import Store


def pid_alive(pid: int) -> bool:
    """True if ``pid`` is a live process (or one we may not signal)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by someone else — treat as alive


def current_run_pid(store: Store) -> int | None:
    """Pid of a live in-progress run, or None. Stale pid files are cleaned up.

    The calling process itself never counts as "another run" — the run process
    checks the pid file that its (MCP/REST) launcher may have already written.
    """
    pid = store.read_run_pid()
    if pid is None or pid == os.getpid():
        return None
    if pid_alive(pid):
        return pid
    store.clear_run_pid()
    return None


def _detached(cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        return subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach: survive the launching server
        )


def start_run(store: Store, *, approve: bool = False) -> tuple[int | None, str]:
    """Spawn a detached ``harness run``; returns ``(pid, message)``.

    ``pid`` is None when the launch was refused (a run is already live).
    """
    existing = current_run_pid(store)
    if existing is not None:
        return None, (
            f"run already in progress (pid={existing}); "
            "stop it first or wait for it to finish"
        )
    cmd = [
        sys.executable, "-m", "harness.cli", "run",
        "--project", str(store.root), "--no-watch", "--no-dashboard",
    ]
    if approve:
        cmd.append("--approve")
    proc = _detached(cmd, store.runs_dir / "run.out")
    store.write_run_pid(proc.pid)
    return proc.pid, f"run started (pid={proc.pid})"


def stop_run(store: Store) -> tuple[bool, str]:
    """SIGTERM the recorded run pid; returns ``(stopped, message)``."""
    pid = store.read_run_pid()
    if pid is None:
        return False, "no run.pid found — no run appears to be in progress"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        store.clear_run_pid()
        return False, f"process {pid} is not running (stale pid file cleaned up)"
    except PermissionError:
        return False, f"cannot signal process {pid} (permission denied)"
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        return False, f"could not stop process {pid}: {e}"
    store.clear_run_pid()
    return True, f"sent SIGTERM to process {pid}"


def approve_hitl(store: Store, task_id: str) -> str:
    """Approve a waiting HITL checkpoint so the paused run continues."""
    checkpoint = store.hitl_path(task_id)
    if not checkpoint.exists():
        raise HarnessError(
            f"no pending approval checkpoint for task {task_id!r}; "
            f"is the run paused and waiting at that task?"
        )
    try:
        data = json.loads(checkpoint.read_text())
    except (ValueError, OSError) as e:
        raise HarnessError(f"could not read checkpoint for task {task_id!r}: {e}") from e
    if data.get("status") != "waiting":
        raise HarnessError(
            f"task {task_id!r} checkpoint has status {data.get('status')!r}, expected 'waiting'"
        )
    data["status"] = "approved"
    checkpoint.write_text(json.dumps(data))
    return f"task {task_id!r} approved — run will continue"


# --------------------------------------------------------------------------- #
# Detached plan generation (planning can take minutes; the API must not block)
# --------------------------------------------------------------------------- #
def _plan_pid_path(store: Store) -> Path:
    return store.runs_dir / "plan.pid"


def _plan_log_path(store: Store) -> Path:
    return store.runs_dir / "plan.log"


def start_plan(
    store: Store,
    *,
    goal: str | None = None,
    prd: str | None = None,
    plan_file: str | None = None,
    model: str | None = None,
) -> tuple[int | None, str]:
    """Spawn a detached ``harness plan``; returns ``(pid, message)``."""
    pid = plan_pid(store)
    if pid is not None:
        return None, f"plan generation already in progress (pid={pid})"
    if not (goal or prd or plan_file):
        raise HarnessError("nothing to plan from: pass goal, prd, or plan_file")
    cmd = [sys.executable, "-m", "harness.cli", "plan", "--project", str(store.root)]
    if goal:
        cmd += ["--goal", goal]
    if prd:
        cmd += ["--prd", prd]
    if plan_file:
        cmd += ["--plan", plan_file]
    if model:
        cmd += ["--model", model]
    log = _plan_log_path(store)
    try:
        log.unlink()  # fresh log per generation
    except OSError:
        pass
    proc = _detached(cmd, log)
    store.runs_dir.mkdir(parents=True, exist_ok=True)
    _plan_pid_path(store).write_text(str(proc.pid))
    return proc.pid, f"plan generation started (pid={proc.pid})"


def plan_pid(store: Store) -> int | None:
    """Pid of a live detached plan generation, or None (stale files cleaned)."""
    p = _plan_pid_path(store)
    try:
        pid = int(p.read_text().strip())
    except (OSError, ValueError):
        return None
    if pid_alive(pid):
        return pid
    try:
        p.unlink()
    except OSError:
        pass
    return None


def plan_status(store: Store) -> dict:
    """Status of a detached plan generation: running flag, log tail, plan presence."""
    log = _plan_log_path(store)
    tail = ""
    if log.exists():
        try:
            tail = log.read_text()[-2000:]
        except OSError:
            pass
    return {
        "running": plan_pid(store) is not None,
        "has_plan": store.has_plan(),
        "log_tail": tail,
    }
