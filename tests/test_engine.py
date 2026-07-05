"""End-to-end engine tests against a scripted (offline) backend.

The engine runs for real; only the LLM transport is deterministic. Sentinels in
the scripted outputs let us assert the orchestration contract precisely.
"""

import asyncio
import json

import pytest

from roundtable.config import Config, Defaults
from roundtable.engine import Engine
from roundtable.errors import RoundtableError
from roundtable.llm import ScriptedProvider
from roundtable.models import AgentRef, Phase, Plan, Status, Task
from roundtable.store import Store

ROLES = {"planner": "m/planner", "main": "m/main", "phase": "m/phase", "task": "m/task"}


def sentinel_responder(role, model, system, user, meta):
    """Each role emits a unique, traceable string."""
    if role == "planner":
        plan = {
            "goal": "g",
            "phases": [
                {
                    "id": "p1", "title": "Phase One", "objective": "obj1", "runner": ROLES["phase"],
                    "tasks": [
                        {"id": "p1-t1", "title": "Task A", "description": "a",
                         "runner": ROLES["task"], "depends_on": [], "subtasks": []},
                        {"id": "p1-t2", "title": "Task B", "description": "b",
                         "runner": ROLES["task"], "depends_on": ["p1-t1"], "subtasks": []},
                    ],
                },
                {
                    "id": "p2", "title": "Phase Two", "objective": "obj2", "runner": ROLES["phase"],
                    "tasks": [
                        {"id": "p2-t1", "title": "Task C", "description": "c",
                         "runner": ROLES["task"], "depends_on": [], "subtasks": []},
                    ],
                },
            ],
        }
        return json.dumps(plan)
    if role == "phase_define":
        return f"DEF::{meta['task_title']}"
    if role == "task_exec":
        return f"SENTINEL::{meta['task_title']}"
    if role == "phase_summary":
        return f"SUMMARY::{meta['phase_title']}"
    if role.startswith("main_"):
        return f"DOC::{role}"
    return "?"


async def _make_approved_plan(store: Store, provider: ScriptedProvider) -> Plan:
    from roundtable.agents import Planner

    plan = await Planner(provider, AgentRef(agent=ROLES["planner"])).create_plan(
        "goal", list(ROLES.values()), ROLES
    )
    plan.main_runner = AgentRef(agent=ROLES["main"])
    for i, ph in enumerate(plan.phases, start=1):
        ph.index = i
    plan = Plan.model_validate(plan.model_dump())
    plan.approved = True
    store.save_plan(plan)
    return plan


async def test_full_run_contract(tmp_path):
    provider = ScriptedProvider(sentinel_responder)
    store = Store(tmp_path)
    await _make_approved_plan(store, provider)

    engine = Engine(store, Config(), provider)
    plan = await engine.run()

    # 1. everything completed
    assert plan.status == Status.done
    assert all(t.status == Status.done for ph in plan.phases for t in ph.tasks)

    # 2. each task's work-definition + result land in its phase folder
    p1 = plan.phases[0]
    tA, tB = p1.tasks
    assert store.write_task_def  # sanity
    assert (store.task_dir(p1, tB) / "TASK.md").read_text() == "DEF::Task B"
    assert (store.task_dir(p1, tB) / "result.md").read_text() == "SENTINEL::Task B"
    assert (store.phase_dir(p1) / "phase-summary.md").read_text() == "SUMMARY::Phase One"

    # 3. docs were produced by the main orchestrator
    assert store.read_doc("OVERVIEW.md") == "DOC::main_kickoff"
    assert "SUMMARY" not in store.read_doc("PROGRESS.md")  # main wrote its own entry
    assert store.read_doc("FINAL.md") == "DOC::main_finalize"

    calls = provider.calls

    # 4. dependency flow: Task B's agent received Task A's result as upstream context
    b_exec = next(c for c in calls if c["role"] == "task_exec" and c["meta"]["task_title"] == "Task B")
    assert "SENTINEL::Task A" in b_exec["user"]

    # 5. wave ordering: A executed before B (B depends on A)
    order = [c["meta"]["task_title"] for c in calls if c["role"] == "task_exec"]
    assert order.index("Task A") < order.index("Task B")

    # 6. CONTEXT-CLEAN invariant: the main orchestrator never sees task transcripts,
    #    only phase summaries.
    main_calls = [c for c in calls if c["role"].startswith("main_")]
    assert main_calls  # main was actually invoked
    for c in main_calls:
        assert "SENTINEL::" not in c["user"], f"task transcript leaked into {c['role']}"
    integrate_one = next(
        c for c in calls if c["role"] == "main_integrate" and c["meta"]["phase_title"] == "Phase One"
    )
    assert "SUMMARY::Phase One" in integrate_one["user"]

    # 7. cross-phase isolation: phase-two work never sees phase-one transcripts
    p2_calls = [c for c in calls if c["meta"].get("phase_title") == "Phase Two"]
    assert p2_calls
    for c in p2_calls:
        assert "SENTINEL::Task A" not in c["user"]
        assert "SENTINEL::Task B" not in c["user"]


async def test_run_requires_approval(tmp_path):
    store = Store(tmp_path)
    plan = Plan(goal="g", main_runner="m/main", phases=[
        Phase(id="p1", index=1, title="P", runner="m/phase",
              tasks=[Task(id="p1-t1", title="T", runner="m/task")]),
    ])
    plan.approved = False
    store.save_plan(plan)
    engine = Engine(store, Config(), ScriptedProvider(sentinel_responder))
    with pytest.raises(RoundtableError, match="not approved"):
        await engine.run()


async def test_resume_skips_completed_tasks(tmp_path):
    store = Store(tmp_path)
    plan = Plan(goal="g", main_runner="m/main", phases=[
        Phase(id="p1", index=1, title="Phase One", runner="m/phase", tasks=[
            Task(id="p1-t1", title="Alpha", runner="m/task"),
            Task(id="p1-t2", title="Beta", runner="m/task", depends_on=["p1-t1"]),
        ]),
    ])
    plan.approved = True
    store.scaffold_plan_tree(plan)
    # Pre-complete Alpha with a result on disk.
    alpha = plan.phases[0].tasks[0]
    store.write_result(plan.phases[0], alpha, "PRE-DONE-ALPHA")
    alpha.status = Status.done
    store.save_plan(plan)

    provider = ScriptedProvider(sentinel_responder)
    await Engine(store, Config(), provider).run()

    execs = [c["meta"]["task_title"] for c in provider.calls if c["role"] == "task_exec"]
    assert "Alpha" not in execs           # skipped, not re-run
    assert "Beta" in execs                # still executed
    # Beta saw Alpha's preserved on-disk result as its dependency context.
    beta_exec = next(c for c in provider.calls
                     if c["role"] == "task_exec" and c["meta"]["task_title"] == "Beta")
    assert "PRE-DONE-ALPHA" in beta_exec["user"]
    # Alpha's result file untouched.
    assert (store.task_dir(plan.phases[0], alpha) / "result.md").read_text() == "PRE-DONE-ALPHA"


def _failing_responder(fail_title: str):
    """Responder that raises (like a non-zero CLI exit) for one task's execution."""
    def responder(role, model, system, user, meta):
        if role == "task_exec" and meta.get("task_title") == fail_title:
            raise RuntimeError(f"agent {model!r} exited 1: simulated failure")
        return sentinel_responder(role, model, system, user, meta)
    return responder


async def test_task_failure_skips_dependents_and_fails_run(tmp_path):
    # Task A fails; B depends on A (must be skipped, never executed); phase 1 fails.
    # Phase 2 is independent, so it still runs — but the overall run is failed.
    provider = ScriptedProvider(_failing_responder("Task A"))
    store = Store(tmp_path)
    await _make_approved_plan(store, provider)
    cfg = Config(defaults=Defaults(max_retries=0))  # 1 attempt, no retry backoff sleeps

    plan = await Engine(store, cfg, provider).run()

    p1, p2 = plan.phases
    tA, tB = p1.tasks
    (tC,) = p2.tasks

    # Failure propagation
    assert tA.status == Status.failed
    assert tB.status == Status.skipped     # dependent of a failed task
    assert p1.status == Status.failed
    # Independent phase still completes
    assert tC.status == Status.done
    assert p2.status == Status.done
    # Overall run failed; finalize skipped (no FINAL.md)
    assert plan.status == Status.failed
    assert store.read_doc("FINAL.md") == ""
    # Failed task records an error result; dependent never executed
    assert (store.task_dir(p1, tA) / "result.md").read_text().startswith("error:")
    execs = [c["meta"]["task_title"] for c in provider.calls if c["role"] == "task_exec"]
    assert "Task B" not in execs
    assert "Task C" in execs
    # Structured events emitted for the failure lifecycle
    types = {e["type"] for e in store.read_events()}
    assert {"task_failed", "task_skipped", "phase_failed", "run_failed"} <= types


async def test_validate_command_failure_marks_task_failed(tmp_path):
    store = Store(tmp_path)
    plan = Plan(goal="g", main_runner="m/main", phases=[
        Phase(id="p1", index=1, title="P", runner="m/phase", tasks=[
            Task(id="p1-t1", title="V", runner="m/task",
                 validate_command=["sh", "-c", "exit 3"]),
        ]),
    ])
    plan.approved = True
    store.save_plan(plan)
    cfg = Config(defaults=Defaults(max_retries=0))

    plan = await Engine(store, cfg, ScriptedProvider(sentinel_responder)).run()

    t = plan.phases[0].tasks[0]
    assert t.status == Status.failed          # validation gate downgrades a passing exec
    assert plan.status == Status.failed
    types = {e["type"] for e in store.read_events()}
    assert "task_validation_failed" in types
    assert (store.task_dir(plan.phases[0], t) / "result.md").read_text().startswith("error:")


async def test_hitl_approval_sets_waiting_then_resumes(tmp_path):
    store = Store(tmp_path)
    plan = Plan(goal="g", main_runner="m/main", phases=[
        Phase(id="p1", index=1, title="P", runner="m/phase", tasks=[
            Task(id="p1-t1", title="Gated", runner="m/task", requires_approval=True),
        ]),
    ])
    plan.approved = True
    store.save_plan(plan)

    run_task = asyncio.create_task(
        Engine(store, Config(), ScriptedProvider(sentinel_responder)).run()
    )
    cp = store.hitl_path("p1-t1")

    # Wait for the engine to park the task at the approval gate.
    for _ in range(200):
        if cp.exists() and json.loads(cp.read_text()).get("status") == "waiting":
            break
        await asyncio.sleep(0.02)
    else:  # pragma: no cover - only trips if the gate never engages
        run_task.cancel()
        pytest.fail("task never reached the waiting checkpoint")

    # The task is visibly waiting, and the run has not completed.
    assert store.load_plan().phases[0].tasks[0].status == Status.waiting
    assert not run_task.done()
    assert store.list_waiting_checkpoints()  # surfaced for `roundtable status`

    # Approve it (as `roundtable resume` would) and let the run finish.
    data = json.loads(cp.read_text())
    data["status"] = "approved"
    cp.write_text(json.dumps(data))

    plan = await asyncio.wait_for(run_task, timeout=10)
    assert plan.status == Status.done
    assert plan.phases[0].tasks[0].status == Status.done
    assert not cp.exists()  # checkpoint consumed


async def test_replan_ignores_non_dict_model_output(tmp_path):
    # A replanning model that returns a JSON array (not an object) must not crash;
    # replan should degrade to "no changes".
    from roundtable.agents import PhaseOrchestrator

    def responder(role, model, system, user, meta):
        if role == "phase_replan":
            return "[1, 2, 3]"
        return "?"

    po = PhaseOrchestrator(ScriptedProvider(responder), AgentRef(agent="m/phase"))
    phase = Phase(id="p1", index=1, title="P", runner="m/phase",
                  tasks=[Task(id="p1-t1", title="A", runner="m/task")])
    updates = await po.replan(phase, phase.tasks[0], "boom", [phase.tasks[0]])
    assert updates == {}


async def test_cross_phase_dependency_flows_result(tmp_path):
    # A phase-2 task depends on a phase-1 task; it must receive that task's
    # result as upstream context and the run must complete cleanly.
    store = Store(tmp_path)
    plan = Plan(goal="g", main_runner="m/main", phases=[
        Phase(id="p1", index=1, title="Phase One", runner="m/phase",
              tasks=[Task(id="p1-t1", title="Alpha", runner="m/task")]),
        Phase(id="p2", index=2, title="Phase Two", runner="m/phase",
              tasks=[Task(id="p2-t1", title="Beta", runner="m/task", depends_on=["p1-t1"])]),
    ])
    plan.approved = True
    store.save_plan(plan)

    provider = ScriptedProvider(sentinel_responder)
    result = await Engine(store, Config(), provider).run()

    assert result.status == Status.done
    assert all(t.status == Status.done for ph in result.phases for t in ph.tasks)
    # Beta (phase 2) saw Alpha's (phase 1) result flow across the phase boundary.
    beta = next(c for c in provider.calls
                if c["role"] == "task_exec" and c["meta"]["task_title"] == "Beta")
    assert "SENTINEL::Alpha" in beta["user"]


async def test_cross_phase_failed_dependency_skips_dependent(tmp_path):
    # Phase-1 task fails; the phase-2 task that depends on it must be skipped
    # (never executed), and the whole run must be marked failed.
    store = Store(tmp_path)
    plan = Plan(goal="g", main_runner="m/main", phases=[
        Phase(id="p1", index=1, title="Phase One", runner="m/phase",
              tasks=[Task(id="p1-t1", title="Alpha", runner="m/task")]),
        Phase(id="p2", index=2, title="Phase Two", runner="m/phase",
              tasks=[Task(id="p2-t1", title="Beta", runner="m/task", depends_on=["p1-t1"])]),
    ])
    plan.approved = True
    store.save_plan(plan)
    cfg = Config(defaults=Defaults(max_retries=0))

    provider = ScriptedProvider(_failing_responder("Alpha"))
    result = await Engine(store, cfg, provider).run()

    assert result.phases[0].tasks[0].status == Status.failed
    assert result.phases[1].tasks[0].status == Status.skipped   # cross-phase cascade
    assert result.phases[1].status == Status.failed             # phase didn't complete
    assert result.status == Status.failed
    assert store.read_doc("FINAL.md") == ""
    execs = [c["meta"]["task_title"] for c in provider.calls if c["role"] == "task_exec"]
    assert "Beta" not in execs
