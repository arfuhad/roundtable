"""MCP server that exposes roundtable operations as callable tools.

Typical workflow an LLM should follow:
  1. roundtable_init        — set up .roundtable/ if this is a fresh project
  2. roundtable_plan        — generate a plan from a goal string
  3. roundtable_approve     — mark the plan approved (required before run)
  4. roundtable_run         — start the run (non-blocking; returns immediately)
  5. roundtable_status      — poll until status == "done" or "failed"

Optionally call roundtable_map first to generate ARCHITECTURE.md + PRD.md from
an existing codebase before planning.

Entry point (stdio, for Claude Code / Claude Desktop):
    roundtable mcp
    roundtable-mcp

Register in Claude Code .claude/settings.json:
    {
        "mcpServers": {
            "roundtable": { "command": "roundtable-mcp" }
        }
    }

Requires: pip install 'roundtable-cli[mcp]'
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import]
    except ImportError:
        print(
            "error: 'mcp' package not installed — run: pip install 'roundtable-cli[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp = FastMCP(
        "roundtable",
        instructions=(
            "Roundtable orchestrates multi-agent LLM tasks. "
            "Workflow: roundtable_init (once) → roundtable_plan → roundtable_approve → roundtable_run. "
            "roundtable_run is non-blocking; poll roundtable_status until status is 'done' or 'failed'. "
            "All tools accept a `project` argument (path to the project directory; defaults to '.')."
        ),
    )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _cli(*args: str) -> str:
        r = subprocess.run(
            ["roundtable", *args],
            capture_output=True, text=True,
        )
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if r.returncode not in (0,):
            return f"error (exit {r.returncode}): {err or out or '(no output)'}"
        return out or err or "ok"

    def _read_state(project: str) -> dict[str, Any]:
        from .store import Store
        from .insights import build_state
        return build_state(Store(Path(project).resolve()))

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    @mcp.tool()
    def roundtable_init(directory: str = ".") -> str:
        """Initialize a .roundtable/ directory and default config in the given project directory."""
        return _cli("init", directory, "--no-models")

    @mcp.tool()
    def roundtable_map(
        target: str = ".",
        project: str = ".",
        model: str | None = None,
    ) -> str:
        """Scan a codebase and generate ARCHITECTURE.md and PRD.md under .roundtable/docs/.

        Use this before roundtable_plan when you want to base the plan on an
        existing codebase rather than a bare goal string.

        Args:
            target:  Path to the codebase to scan (default: project root).
            project: Roundtable project root where .roundtable/ lives (default: '.').
            model:   Override the analyst agent/model (e.g. 'claude:opus').
        """
        args = ["map", "--project", project, "--target", target]
        if model:
            args += ["--model", model]
        return _cli(*args)

    @mcp.tool()
    def roundtable_plan(
        goal: str,
        project: str = ".",
        model: str | None = None,
    ) -> str:
        """Generate a multi-agent execution plan from a goal string.

        Writes plan.json and PLAN.md under .roundtable/plan/. The plan must be
        approved with roundtable_approve before it can be run.

        Args:
            goal:    Natural-language description of what the agents should build or do.
            project: Roundtable project root (default: '.').
            model:   Override the planner agent/model (e.g. 'claude:opus').
        """
        args = ["plan", "--goal", goal, "--project", project]
        if model:
            args += ["--model", model]
        return _cli(*args)

    @mcp.tool()
    def roundtable_approve(project: str = ".") -> str:
        """Approve the current plan so it can be executed by roundtable_run.

        Args:
            project: Roundtable project root (default: '.').
        """
        return _cli("approve", "--project", project)

    @mcp.tool()
    def roundtable_run(project: str = ".", approve: bool = False) -> str:
        """Start executing the approved plan in the background (non-blocking).

        Returns immediately after starting the run process. Use roundtable_status
        to monitor progress. The run continues even if this MCP server exits.

        Args:
            project: Roundtable project root (default: '.').
            approve: If True, auto-approve the plan before running (combines
                     roundtable_approve + roundtable_run in one call).
        """
        from . import runctl
        from .store import Store

        store = Store(Path(project).resolve())
        pid, msg = runctl.start_run(store, approve=approve)
        if pid is None:
            return f"error: {msg}. Call roundtable_status to monitor, or roundtable_stop to cancel."
        return (
            f"Run started (pid={pid}). "
            "Agents are now executing tasks in the background. "
            "Call roundtable_status() to monitor progress."
        )

    @mcp.tool()
    def roundtable_stop(project: str = ".") -> str:
        """Stop a running roundtable execution.

        Reads the PID from the run.pid file and sends SIGTERM to gracefully
        stop the process.

        Args:
            project: Roundtable project root (default: '.').
        """
        from . import runctl
        from .store import Store

        store = Store(Path(project).resolve())
        _, msg = runctl.stop_run(store)
        return msg

    @mcp.tool()
    def roundtable_status(project: str = ".") -> str:
        """Return the current run state as JSON.

        Includes: overall status, per-phase/task progress, active tasks,
        per-agent stats, timing, and recent events.

        Args:
            project: Roundtable project root (default: '.').
        """
        state = _read_state(project)
        return json.dumps(state, indent=2)

    # ------------------------------------------------------------------ #
    # Resources (read-only context; always operates on cwd / project='.')
    @mcp.tool()
    def roundtable_usage(project: str = ".") -> str:
        """Return aggregated provider usage stats (calls, tokens, duration).

        Reads 'usage' events from the run log and returns a summary. Only
        available after a run completes or partially completes.

        Args:
            project: Roundtable project root (default: '.').
        """
        from .store import Store

        store = Store(Path(project).resolve())
        events = store.read_events()
        usage_events = [e for e in events if e.get("type") == "usage"]
        if not usage_events:
            return json.dumps({"message": "no usage data recorded yet"})
        # Return the most recent usage snapshot.
        latest = usage_events[-1]
        return json.dumps({
            k: v for k, v in latest.items()
            if k not in ("type", "ts", "msg")
        }, indent=2)

    # ------------------------------------------------------------------ #

    @mcp.resource("roundtable://plan")
    def resource_plan() -> str:
        """The current plan.json — phases, tasks, runners, and approval state."""
        from .store import Store
        store = Store(Path(".").resolve())
        p = store.manifest_path
        if not p.exists():
            return json.dumps({"error": "no plan found — call roundtable_plan first"})
        return p.read_text()

    @mcp.resource("roundtable://state")
    def resource_state() -> str:
        """Live run state snapshot (same data as roundtable_status tool, project='.')."""
        return json.dumps(_read_state("."), indent=2)

    @mcp.resource("roundtable://logs")
    def resource_logs() -> str:
        """Last 50 run events from run.log as a JSON array (newest first)."""
        from .store import Store
        store = Store(Path(".").resolve())
        events = store.read_events()
        return json.dumps(list(reversed(events[-50:])), indent=2)

    return mcp


def run_server() -> None:
    """Entry point: start the roundtable MCP server over stdio."""
    server = _build_server()
    server.run()
