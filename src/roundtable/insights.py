"""Live state + insights derived from ``plan.json`` and the run event log.

``build_state`` is a pure read over the on-disk source of truth, so any number of
viewers (the web dashboard, the terminal watch, a script hitting ``/api/state``)
can poll it without touching the running engine. ``render_text`` is the terminal
rendering. Insights: progress, what each agent is doing *now*, per-agent task
counts/time, and task durations paired from ``task_started``/``task_done`` events.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from .models import Status
from .store import Store

_MARK = {"done": "x", "in_progress": "~", "pending": " ", "failed": "!", "skipped": "-", "waiting": "?"}


def _parse(ts: str | None) -> _dt.datetime | None:
    if not ts:
        return None
    try:
        return _dt.datetime.fromisoformat(ts)
    except ValueError:
        return None


def fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_state(store: Store, *, event_limit: int = 40) -> dict[str, Any]:
    """Snapshot of the run: totals, current activity, per-agent + timing insights."""
    if not store.has_plan():
        return {"exists": False, "events": []}

    plan = store.load_plan()
    events = store.read_events()
    now = _dt.datetime.now(_dt.timezone.utc)

    # Pair task_started/task_done by task_id for start times + durations.
    started_at: dict[str, str] = {}
    duration: dict[str, float] = {}
    for e in events:
        tid = e.get("task_id")
        if not tid:
            continue
        if e.get("type") == "task_started":
            started_at[tid] = e.get("ts", "")
        elif e.get("type") == "task_done" and tid in started_at:
            a, b = _parse(started_at[tid]), _parse(e.get("ts"))
            if a and b:
                duration[tid] = (b - a).total_seconds()

    totals = {"phases": len(plan.phases), "tasks": 0,
              "done": 0, "in_progress": 0, "pending": 0, "failed": 0, "skipped": 0, "waiting": 0}
    by_agent: dict[str, dict[str, float]] = {}
    durations: list[tuple[str, float]] = []
    now_running: list[dict[str, Any]] = []
    phases_out: list[dict[str, Any]] = []

    for ph in plan.phases:
        tasks_out = []
        for t in ph.tasks:
            totals["tasks"] += 1
            totals[t.status.value] = totals.get(t.status.value, 0) + 1
            dur = duration.get(t.id)
            if dur is not None:
                durations.append((t.id, dur))

            key = t.runner.agent or t.runner.model or "?"
            agg = by_agent.setdefault(key, {"tasks": 0, "total_s": 0.0})
            if t.status == Status.done:
                agg["tasks"] += 1
                if dur is not None:
                    agg["total_s"] += dur

            entry = {
                "id": t.id, "title": t.title, "runner": str(t.runner),
                "agent": t.runner.agent, "model": t.runner.model,
                "status": t.status.value, "duration_s": dur,
                "depends_on": list(t.depends_on),
            }
            if t.status == Status.in_progress:
                start = _parse(started_at.get(t.id))
                entry["elapsed_s"] = (now - start).total_seconds() if start else None
                now_running.append({"phase_id": ph.id, **entry})
            tasks_out.append(entry)

        done_n = sum(1 for t in ph.tasks if t.status == Status.done)
        phases_out.append({
            "id": ph.id, "index": ph.index, "title": ph.title, "objective": ph.objective,
            "runner": str(ph.runner), "status": ph.status.value,
            "tasks": tasks_out, "done": done_n, "total": len(ph.tasks),
        })

    pct = round(100 * totals["done"] / totals["tasks"]) if totals["tasks"] else 0
    avg = round(sum(d for _, d in durations) / len(durations), 1) if durations else None
    slowest = max(durations, key=lambda x: x[1]) if durations else None

    run_state = (
        "done" if plan.status == Status.done
        else "running" if plan.status == Status.in_progress
        else "failed" if plan.status == Status.failed
        else "pending"
    )
    for key, agg in by_agent.items():
        agg["total_s"] = round(agg["total_s"], 1)

    # Latest provider usage snapshot (emitted per task + at run end by the engine).
    usage_evt = next((e for e in reversed(events) if e.get("type") == "usage"), None)
    usage = None
    if usage_evt:
        usage = {
            "calls": usage_evt.get("calls", 0),
            "total_tokens": usage_evt.get("total_tokens", 0),
            "prompt_tokens": usage_evt.get("prompt_tokens", 0),
            "completion_tokens": usage_evt.get("completion_tokens", 0),
            "total_duration_s": usage_evt.get("total_duration_s", 0.0),
            "estimated": usage_evt.get("estimated", False),
            "cost_usd": usage_evt.get("cost_usd", 0.0),
        }
    # Usage events feed the tile above; keep them out of the raw event stream.
    feed = [e for e in events if e.get("type") != "usage"]

    return {
        "exists": True,
        "goal": plan.goal,
        "approved": plan.approved,
        "status": run_state,
        "started_at": next((e.get("ts") for e in events if e.get("type") == "run_started"), None),
        "updated_at": events[-1].get("ts") if events else None,
        "now": now_running,
        "totals": {**totals, "percent": pct},
        "phases": phases_out,
        "by_agent": by_agent,
        "timings": {
            "avg_task_s": avg,
            "slowest": ({"task_id": slowest[0], "duration_s": round(slowest[1], 1)}
                        if slowest else None),
        },
        "usage": usage,
        "events": list(reversed(feed[-event_limit:])),
        "generated_at": now.isoformat(timespec="seconds"),
    }


def render_text(state: dict[str, Any], *, width: int = 64) -> str:
    """Compact terminal rendering of a state snapshot."""
    if not state.get("exists"):
        return "no plan yet — run `roundtable plan` first"

    t = state["totals"]
    badge = {"running": "● running", "done": "✓ done",
             "failed": "✗ failed", "pending": "· pending"}.get(state["status"], state["status"])
    filled = round((width - 2) * t["percent"] / 100)
    bar = "█" * filled + "░" * (width - 2 - filled)

    lines = [
        f"Roundtable   {badge}    {t['done']}/{t['tasks']} tasks · {t['percent']}%",
        f"Goal: {state['goal']}",
        f"[{bar}]",
        "",
    ]

    if state["now"]:
        for n in state["now"]:
            lines.append(f"▶ NOW  {n['id']}  {n['title']!r}")
            lines.append(f"        {n['runner']} · running {fmt_dur(n.get('elapsed_s'))}")
    else:
        idle = "done" if state["status"] == "done" else "idle"
        lines.append(f"▶ NOW  ({idle})")
    lines.append("")

    lines.append("Phases")
    for ph in state["phases"]:
        mark = _MARK.get(ph["status"], " ")
        lines.append(f" [{mark}] {ph['index']} {ph['title']}   {ph['done']}/{ph['total']}  ({ph['runner']})")
    lines.append("")

    if state["by_agent"]:
        ba = " · ".join(
            f"{k} {v['tasks']}" + (f"/{fmt_dur(v['total_s'])}" if v["total_s"] else "")
            for k, v in sorted(state["by_agent"].items())
        )
        lines.append(f"by agent: {ba}")
    tim = state["timings"]
    if tim["avg_task_s"] is not None:
        slow = tim["slowest"]
        extra = f" · slowest {slow['task_id']} {fmt_dur(slow['duration_s'])}" if slow else ""
        lines.append(f"avg task {fmt_dur(tim['avg_task_s'])}{extra}")

    u = state.get("usage")
    if u:
        note = " (est)" if u.get("estimated") else ""
        cost = f" · ${u['cost_usd']:.4f}" if u.get("cost_usd") else ""
        lines.append(
            f"usage: {u['total_tokens']:,} tokens"
            f" ({u['prompt_tokens']:,} in / {u['completion_tokens']:,} out)"
            f" · {u['calls']} calls{cost}{note}"
        )

    return "\n".join(lines)
