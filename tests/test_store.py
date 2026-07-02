import json

from harness.models import Phase, Plan, Task
from harness.store import Store, render_plan_md


def _plan():
    return Plan(goal="demo", main_runner="m/main", phases=[
        Phase(id="p1", index=1, title="Set Up", runner="m/phase", tasks=[
            Task(id="p1-t1", title="Init Repo", runner="m/task"),
            Task(id="p1-t2", title="Add CI", runner="m/task", depends_on=["p1-t1"]),
        ]),
    ])


def test_scaffold_tree_and_dir_naming(tmp_path):
    plan = _plan()
    s = Store(tmp_path)
    s.scaffold_plan_tree(plan)
    phase = plan.phases[0]
    assert s.phase_dir(phase).name == "phase-01-set-up"
    assert s.task_dir(phase, phase.tasks[0]).name == "task-01-init-repo"
    assert s.task_dir(phase, phase.tasks[1]).name == "task-02-add-ci"
    assert s.task_output_dir(phase, phase.tasks[1]).is_dir()


def test_manifest_roundtrip(tmp_path):
    plan = _plan()
    s = Store(tmp_path)
    s.save_plan(plan)
    assert s.has_plan()
    loaded = s.load_plan()
    assert loaded == plan
    # JSON is human-inspectable
    raw = json.loads(s.manifest_path.read_text())
    assert raw["phases"][0]["tasks"][1]["depends_on"] == ["p1-t1"]


def test_writers(tmp_path):
    plan = _plan()
    s = Store(tmp_path)
    s.scaffold_plan_tree(plan)
    phase, task = plan.phases[0], plan.phases[0].tasks[0]
    s.write_task_def(phase, task, "WORKDEF")
    s.write_result(phase, task, "RESULT")
    s.write_artifact(phase, task, "out.txt", "ARTIFACT")
    s.write_phase_summary(phase, "SUMMARY")
    s.write_doc("OVERVIEW.md", "OVR")

    td = s.task_dir(phase, task)
    assert (td / "TASK.md").read_text() == "WORKDEF"
    assert (td / "result.md").read_text() == "RESULT"
    assert (td / "output" / "out.txt").read_text() == "ARTIFACT"
    assert (s.phase_dir(phase) / "phase-summary.md").read_text() == "SUMMARY"
    assert s.read_doc("OVERVIEW.md") == "OVR"
    assert s.read_doc("missing.md") == ""


def test_record_event_appends_jsonl(tmp_path):
    s = Store(tmp_path)
    s.record_event("log", message="one")
    s.record_event("log", message="two")
    lines = (s.runs_dir / "run.log").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["msg"] == "one"
    assert "ts" in json.loads(lines[1])


def test_record_event_structured_and_read(tmp_path):
    s = Store(tmp_path)
    s.record_event("task_started", message="task p1-t1 started",
                   task_id="p1-t1", agent="opencode", model="mimo-v2.5-pro")
    s.record_event("log", message="plain")  # a plain human log line
    # a partial/garbage line must not break the reader
    with (s.runs_dir / "run.log").open("a") as f:
        f.write("{not valid json\n")

    events = s.read_events()
    assert [e["type"] for e in events] == ["task_started", "log"]
    assert events[0]["task_id"] == "p1-t1" and events[0]["agent"] == "opencode"
    assert events[1]["msg"] == "plain"


def test_render_plan_md_contains_models_and_deps():
    md = render_plan_md(_plan())
    assert "Set Up" in md and "`m/phase`" in md and "`m/task`" in md
    assert "depends on: p1-t1" in md
