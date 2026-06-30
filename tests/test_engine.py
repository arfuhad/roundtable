"""End-to-end engine tests against a scripted (offline) backend.

The engine runs for real; only the LLM transport is deterministic. Sentinels in
the scripted outputs let us assert the orchestration contract precisely.
"""

import json

import pytest

from harness.config import Config
from harness.engine import Engine
from harness.errors import HarnessError
from harness.llm import ScriptedProvider
from harness.models import AgentRef, Phase, Plan, Status, Task
from harness.store import Store

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
    from harness.agents import Planner

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
    with pytest.raises(HarnessError, match="not approved"):
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
