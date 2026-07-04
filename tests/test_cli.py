"""CLI integration: drive the real entrypoint end-to-end on the scripted backend."""

import json

from harness.cli import main
from harness.config import write_default_config
from harness.models import Status
from harness.store import Store


def _use_scripted(root):
    write_default_config(root)
    cfg = root / "harness.config.yaml"
    cfg.write_text(cfg.read_text().replace("provider: cli", "provider: scripted"))


def test_cli_full_lifecycle(tmp_path, capsys):
    root = tmp_path / "proj"
    proj = str(root)

    # --no-models keeps init offline (no shelling out to real CLIs for model lists).
    assert main(["init", proj, "--no-models"]) == 0
    _use_scripted(root)

    # run before a plan exists -> friendly error, exit 2
    assert main(["run", "--project", proj]) == 2

    assert main(["plan", "--goal", "Build a CLI todo app", "--project", proj]) == 0
    store = Store(root)
    assert store.has_plan()
    assert store.load_plan().approved is False

    # run before approval -> gate blocks, exit 2
    assert main(["run", "--project", proj]) == 2

    assert main(["approve", "--project", proj]) == 0
    assert store.load_plan().approved is True

    # full autonomous run
    assert main(["run", "--project", proj]) == 0
    assert main(["status", "--project", proj]) == 0

    plan = store.load_plan()
    assert plan.status == Status.done
    assert all(t.status == Status.done for ph in plan.phases for t in ph.tasks)

    # artifacts on disk
    for ph in plan.phases:
        assert (store.phase_dir(ph) / "phase-summary.md").exists()
        for t in ph.tasks:
            assert (store.task_dir(ph, t) / "TASK.md").exists()
            assert (store.task_dir(ph, t) / "result.md").exists()
    assert store.read_doc("OVERVIEW.md")
    assert store.read_doc("FINAL.md")

    # status output mentions the goal; run surfaced a live dashboard link
    out = capsys.readouterr().out
    assert "Build a CLI todo app" in out
    assert "dashboard: http://127.0.0.1:" in out


def test_status_json(tmp_path, capsys):
    root = tmp_path / "proj"
    proj = str(root)

    assert main(["init", proj, "--no-models"]) == 0
    _use_scripted(root)

    assert main(["plan", "--goal", "Build a CLI todo app", "--project", proj]) == 0
    assert main(["approve", "--project", proj]) == 0
    assert main(["run", "--project", proj]) == 0
    capsys.readouterr()  # drain prior output

    assert main(["status", "--project", proj, "--json"]) == 0

    out = capsys.readouterr().out
    state = json.loads(out)

    assert isinstance(state, dict)
    assert state["exists"] is True
    assert state["goal"] == "Build a CLI todo app"
    assert state["approved"] is True
    assert state["status"] == "done"
    assert "totals" in state
    assert "phases" in state
    assert "by_agent" in state
    assert "timings" in state
    assert "events" in state
    assert "generated_at" in state

    assert state["totals"]["phases"] > 0
    assert state["totals"]["done"] == state["totals"]["tasks"]
