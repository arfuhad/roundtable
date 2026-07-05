"""Failure-handling hardening: runner validation, crash events, interrupt resume,
and the shared run.pid protocol in runctl."""

import asyncio
import json
import os
import sys

import pytest

from roundtable import runctl
from roundtable.config import Config
from roundtable.engine import Engine, validate_runners
from roundtable.errors import RoundtableError
from roundtable.llm import CLIProvider, ScriptedProvider
from roundtable.models import Phase, Plan, Status, Task
from roundtable.store import Store


def _plan(runner="claude", approved=True):
    p = Plan(goal="g", main_runner=runner, phases=[
        Phase(id="p1", index=1, title="P", runner=runner,
              tasks=[Task(id="p1-t1", title="T", runner=runner)]),
    ])
    p.approved = approved
    return p


# --------------------------------------------------------------------------- #
# validate_runners
# --------------------------------------------------------------------------- #
def test_validate_runners_rejects_unknown_and_missing():
    config = Config()  # provider cli; agents: claude/codex/gemini
    validate_runners(_plan("claude"), config)  # known agent -> fine

    with pytest.raises(RoundtableError) as ei:
        validate_runners(_plan("m/task"), config)
    msg = str(ei.value)
    assert "m/task" in msg and "available agents" in msg

    plan = _plan("claude")
    plan.phases[0].tasks[0].runner.agent = ""
    plan.phases[0].tasks[0].runner.model = ""
    with pytest.raises(RoundtableError) as ei:
        validate_runners(plan, config)
    assert "no runner" in str(ei.value)


def test_validate_runners_skipped_for_non_cli_provider():
    config = Config(provider="scripted")
    validate_runners(_plan("anything-goes"), config)  # no raise


async def test_engine_validates_before_running_with_cli_provider(tmp_path):
    store = Store(tmp_path)
    store.save_plan(_plan("nope"))
    config = Config()
    provider = CLIProvider(config.agents, cwd=tmp_path)
    engine = Engine(store, config, provider)
    with pytest.raises(RoundtableError, match="nope"):
        await engine.run()
    # nothing started: no events, plan untouched
    assert store.load_plan().status == Status.pending
    assert not any(e["type"] == "run_started" for e in store.read_events())


# --------------------------------------------------------------------------- #
# crash + interrupt visibility
# --------------------------------------------------------------------------- #
async def test_engine_crash_records_run_error_and_fails_plan(tmp_path):
    def exploding(role, model, system, user, meta):
        raise ValueError("boom in kickoff")

    store = Store(tmp_path)
    store.save_plan(_plan("m/task"))
    engine = Engine(store, Config(provider="scripted"), ScriptedProvider(exploding))
    with pytest.raises(ValueError, match="boom"):
        await engine.run()

    assert store.load_plan().status == Status.failed
    events = store.read_events()
    assert any(e["type"] == "run_error" and "boom" in e["msg"] for e in events)


async def test_interrupt_resets_in_progress_to_pending(tmp_path):
    class SlowProvider(ScriptedProvider):
        async def complete(self, **kw):
            if kw.get("role") == "task_exec":
                await asyncio.sleep(30)
            return await super().complete(**kw)

    store = Store(tmp_path)
    store.save_plan(_plan("m/task"))
    engine = Engine(store, Config(provider="scripted"), SlowProvider())

    run = asyncio.create_task(engine.run())
    # wait until the task is actually in_progress on disk
    for _ in range(100):
        await asyncio.sleep(0.02)
        plan = store.load_plan()
        if plan.phases[0].tasks[0].status == Status.in_progress:
            break
    else:
        pytest.fail("task never reached in_progress")

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    plan = store.load_plan()
    assert plan.status == Status.pending                      # resumable
    assert plan.phases[0].tasks[0].status == Status.pending   # nothing left "running"
    assert any(e["type"] == "run_interrupted" for e in store.read_events())


async def test_roundtable_error_in_task_fails_task_without_retry(tmp_path):
    calls = {"n": 0}

    class ConfigErrorProvider(ScriptedProvider):
        async def complete(self, **kw):
            if kw.get("role") == "task_exec":
                calls["n"] += 1
                raise RoundtableError("agent 'x' is not defined")
            return await super().complete(**kw)

    store = Store(tmp_path)
    store.save_plan(_plan("m/task"))
    config = Config(provider="scripted")
    config.defaults.max_retries = 3  # would retry transient errors 3 times
    engine = Engine(store, config, ConfigErrorProvider())
    plan = await engine.run()

    assert calls["n"] == 1  # no retries for config errors
    assert plan.status == Status.failed
    assert plan.phases[0].tasks[0].status == Status.failed


# --------------------------------------------------------------------------- #
# phase-level validate_command (completion gate)
# --------------------------------------------------------------------------- #
async def test_phase_validation_failure_fails_phase_despite_done_tasks(tmp_path):
    store = Store(tmp_path)
    plan = _plan("m/task")
    plan.phases[0].validate_command = [sys.executable, "-c", "import sys; sys.exit(1)"]
    store.save_plan(plan)

    engine = Engine(store, Config(provider="scripted"), ScriptedProvider())
    result = await engine.run()

    assert result.status == Status.failed
    assert result.phases[0].status == Status.failed
    # every task succeeded — only the phase gate failed
    assert all(t.status == Status.done for t in result.phases[0].tasks)
    events = store.read_events()
    assert any(e["type"] == "phase_validation_failed" for e in events)
    failed = next(e for e in events if e["type"] == "phase_failed")
    assert "validation failed" in failed["msg"]
    assert store.read_doc("FINAL.md") == ""  # failed runs never finalize


async def test_phase_validation_passes_and_rerun_heals(tmp_path):
    store = Store(tmp_path)
    plan = _plan("m/task")
    plan.phases[0].validate_command = [
        sys.executable, "-c",
        "import pathlib, sys; sys.exit(0 if pathlib.Path('ok.marker').exists() else 1)",
    ]
    store.save_plan(plan)

    engine = Engine(store, Config(provider="scripted"), ScriptedProvider())
    result = await engine.run()
    assert result.status == Status.failed  # marker missing -> gate fails

    # "Fix the project" (validate_command runs in the project root), then re-run:
    # tasks are already done, so only the gate re-runs — and now passes.
    (tmp_path / "ok.marker").write_text("")
    rerun = Engine(store, Config(provider="scripted"), ScriptedProvider())
    result = await rerun.run()
    assert result.status == Status.done
    assert result.phases[0].status == Status.done
    events = store.read_events()
    assert any(e["type"] == "phase_validated" for e in events)
    assert store.read_doc("FINAL.md") != ""


async def test_phase_validation_unrunnable_command_fails_phase(tmp_path):
    store = Store(tmp_path)
    plan = _plan("m/task")
    plan.phases[0].validate_command = ["/nonexistent/validator-binary"]
    store.save_plan(plan)

    engine = Engine(store, Config(provider="scripted"), ScriptedProvider())
    result = await engine.run()
    assert result.status == Status.failed
    assert any(
        e["type"] == "phase_validation_failed" and "could not run" in e["msg"]
        for e in store.read_events()
    )


# --------------------------------------------------------------------------- #
# runctl pid protocol
# --------------------------------------------------------------------------- #
def test_pid_alive():
    assert runctl.pid_alive(os.getpid())
    assert runctl.pid_alive(1)  # launchd/init: alive but not signalable
    assert not runctl.pid_alive(2 ** 22 + 1)  # beyond pid space on macOS/Linux


def test_current_run_pid_cleans_stale_and_ignores_self(tmp_path):
    store = Store(tmp_path)
    assert runctl.current_run_pid(store) is None

    store.write_run_pid(os.getpid())
    assert runctl.current_run_pid(store) is None  # our own pid: we ARE the run

    store.write_run_pid(2 ** 22 + 1)  # dead pid -> stale file cleaned
    assert runctl.current_run_pid(store) is None
    assert store.read_run_pid() is None

    store.write_run_pid(1)  # alive, not ours
    assert runctl.current_run_pid(store) == 1


def test_start_run_refuses_when_live(tmp_path):
    store = Store(tmp_path)
    store.write_run_pid(1)
    pid, msg = runctl.start_run(store)
    assert pid is None and "already in progress" in msg


def test_approve_hitl_transitions(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(RoundtableError, match="no pending approval"):
        runctl.approve_hitl(store, "p1-t1")

    cp = store.hitl_path("p1-t1")
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"status": "waiting", "task_id": "p1-t1"}))
    msg = runctl.approve_hitl(store, "p1-t1")
    assert "approved" in msg
    assert json.loads(cp.read_text())["status"] == "approved"

    with pytest.raises(RoundtableError, match="expected 'waiting'"):
        runctl.approve_hitl(store, "p1-t1")  # already approved
