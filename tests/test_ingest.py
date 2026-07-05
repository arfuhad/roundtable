"""Plan ingestion (existing JSON plan) + project-friendly .roundtable/ layout."""

from roundtable.cli import make_plan
from roundtable.config import Config
from roundtable.models import Phase, Plan, Task
from roundtable.store import Store


async def test_import_existing_json_plan_no_llm(tmp_path):
    # An existing plan already in our schema is loaded directly (no model call).
    existing = Plan(goal="ship feature", phases=[
        Phase(id="p1", index=1, title="Build", tasks=[
            Task(id="p1-t1", title="Write code"),
            Task(id="p1-t2", title="Test", depends_on=["p1-t1"]),
        ]),
    ])
    f = tmp_path / "existing-plan.json"
    f.write_text(existing.model_dump_json())

    store = Store(tmp_path)
    plan = await make_plan(store, Config(), plan_path=str(f))

    assert plan.goal == "ship feature"            # preserved
    assert [p.id for p in plan.phases] == ["p1"]
    assert plan.phases[0].tasks[1].depends_on == ["p1-t1"]
    # role runners (agent, model) backfilled from config defaults
    assert plan.phases[0].tasks[0].runner == Config().models.task
    assert plan.main_runner == Config().models.main


async def test_artifacts_stay_under_roundtable_workdir(tmp_path):
    # Simulate an existing project with its own files.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n")

    existing = Plan(goal="g", phases=[Phase(id="p1", index=1, title="P",
                    tasks=[Task(id="p1-t1", title="T")])])
    f = tmp_path / "plan.json"
    f.write_text(existing.model_dump_json())

    store = Store(tmp_path)
    await make_plan(store, Config(), plan_path=str(f))

    # Everything roundtable wrote is under .roundtable/, project files untouched.
    assert (tmp_path / ".roundtable" / "plan" / "plan.json").exists()
    assert (tmp_path / "src" / "app.py").read_text() == "print('hi')\n"
    top = {p.name for p in tmp_path.iterdir()}
    assert "phases" not in top and "plan" not in top  # not polluting the repo root
    assert ".roundtable" in top
