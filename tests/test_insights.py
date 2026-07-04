"""Analytics/insights: build_state derives live status, durations, per-agent stats."""

import datetime as _dt
import json

from harness.insights import build_state, fmt_dur, render_text
from harness.models import Phase, Plan, Status, Task
from harness.store import Store


def _seed(tmp_path):
    """A plan mid-run: p1-t1 done (30s), p1-t2 running, with crafted event times."""
    plan = Plan(goal="ship it", main_runner="claude:opus-4.8", status=Status.in_progress, phases=[
        Phase(id="p1", index=1, title="Build", runner="agy:gemini-3.5-flash", status=Status.in_progress, tasks=[
            Task(id="p1-t1", title="Alpha", runner="opencode:mimo-v2.5-pro", status=Status.done),
            Task(id="p1-t2", title="Beta", runner="opencode:mimo-v2.5-pro",
                 status=Status.in_progress, depends_on=["p1-t1"]),
        ]),
    ])
    store = Store(tmp_path)
    store.save_plan(plan)
    # Anchor events a few minutes in the past so "elapsed since start" is positive,
    # while keeping the 30s started->done gap so durations are deterministic.
    base = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=5)
    t0 = base.isoformat(timespec="seconds")
    t30 = (base + _dt.timedelta(seconds=30)).isoformat(timespec="seconds")
    lines = [
        {"ts": t0, "type": "run_started", "goal": "ship it"},
        {"ts": t0, "type": "task_started", "task_id": "p1-t1", "agent": "opencode", "model": "mimo-v2.5-pro"},
        {"ts": t30, "type": "task_done", "task_id": "p1-t1", "agent": "opencode", "model": "mimo-v2.5-pro"},
        {"ts": t30, "type": "usage", "calls": 3, "prompt_tokens": 900, "completion_tokens": 300,
         "total_tokens": 1200, "total_duration_s": 42.0, "estimated": True},
        {"ts": t30, "type": "task_started", "task_id": "p1-t2", "agent": "opencode", "model": "mimo-v2.5-pro"},
    ]
    store.runs_dir.mkdir(parents=True, exist_ok=True)
    (store.runs_dir / "run.log").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return store


def test_build_state_progress_and_durations(tmp_path):
    state = build_state(_seed(tmp_path))
    assert state["exists"] and state["status"] == "running"
    t = state["totals"]
    assert (t["tasks"], t["done"], t["in_progress"], t["percent"]) == (2, 1, 1, 50)
    # duration paired from started/done events
    alpha = state["phases"][0]["tasks"][0]
    assert alpha["id"] == "p1-t1" and alpha["duration_s"] == 30.0
    # timings insight
    assert state["timings"]["avg_task_s"] == 30.0
    assert state["timings"]["slowest"]["task_id"] == "p1-t1"


def test_build_state_now_running_and_by_agent(tmp_path):
    state = build_state(_seed(tmp_path))
    # exactly the in_progress task is "now", with elapsed since its start
    assert [n["id"] for n in state["now"]] == ["p1-t2"]
    assert state["now"][0]["elapsed_s"] is not None and state["now"][0]["elapsed_s"] > 0
    # per-agent: opencode completed 1 task for 30s
    assert state["by_agent"]["opencode"] == {"tasks": 1, "total_s": 30.0}
    # most-recent-first event feed
    assert state["events"][0]["type"] == "task_started"
    assert state["events"][0]["task_id"] == "p1-t2"


def test_build_state_no_plan(tmp_path):
    assert build_state(Store(tmp_path)) == {"exists": False, "events": []}


def test_render_text_contains_key_facts(tmp_path):
    out = render_text(build_state(_seed(tmp_path)))
    assert "ship it" in out
    assert "1/2 tasks" in out and "50%" in out
    assert "Beta" in out                      # the currently-running task
    assert "opencode" in out                  # per-agent line


def test_usage_surfaced_and_filtered_from_feed(tmp_path):
    state = build_state(_seed(tmp_path))
    u = state["usage"]
    assert (u["total_tokens"], u["prompt_tokens"], u["completion_tokens"]) == (1200, 900, 300)
    assert u["calls"] == 3 and u["estimated"] is True
    # usage events feed the tile, not the raw event stream
    assert all(e["type"] != "usage" for e in state["events"])
    # terminal renderer shows the usage line, flagged as an estimate
    out = render_text(state)
    assert "usage:" in out and "1,200 tokens" in out and "(est)" in out


def test_fmt_dur():
    assert fmt_dur(None) == "—"
    assert fmt_dur(0) == "0:00"
    assert fmt_dur(95) == "1:35"
    assert fmt_dur(3725) == "1:02:05"
