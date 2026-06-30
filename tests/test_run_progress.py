"""`harness run` live progress: web dashboard link + inline terminal watch view."""

import argparse
import asyncio
import contextlib
import io

from harness.cli import _run_with_progress, make_plan
from harness.config import Config
from harness.engine import Engine
from harness.llm import ScriptedProvider
from harness.models import Status
from harness.store import Store


def _approved_scripted_plan(tmp_path):
    store = Store(tmp_path)
    config = Config(provider="scripted")  # scripted planner + engine, no network
    asyncio.run(make_plan(store, config, goal="live demo"))
    plan = store.load_plan()
    plan.approved = True
    store.save_plan(plan)
    return store, config


def _args(tmp_path, **over):
    base = dict(project=str(tmp_path), no_dashboard=True, no_watch=False,
               interval=0.01, host="127.0.0.1", port=0, open=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_run_renders_live_terminal_frames(tmp_path):
    store, config = _approved_scripted_plan(tmp_path)
    engine = Engine(store, config, ScriptedProvider())

    buf = io.StringIO()
    buf.isatty = lambda: True  # force the live (TTY) path
    with contextlib.redirect_stdout(buf):
        rc = asyncio.run(_run_with_progress(engine, store, _args(tmp_path)))
    out = buf.getvalue()

    assert rc == 0
    assert "\x1b[2J" in out          # cleared/redrawn at least once
    assert "llm-harness" in out      # the watch header was rendered
    assert "run complete" in out
    assert store.load_plan().status == Status.done


def test_run_prints_dashboard_link(tmp_path):
    store, config = _approved_scripted_plan(tmp_path)
    engine = Engine(store, config, ScriptedProvider())

    buf = io.StringIO()  # non-tty -> no live frames, but the link still prints
    with contextlib.redirect_stdout(buf):
        rc = asyncio.run(_run_with_progress(engine, store, _args(tmp_path, no_dashboard=False)))
    out = buf.getvalue()

    assert rc == 0
    assert "dashboard: http://127.0.0.1:" in out
    assert "\x1b[2J" not in out  # non-tty: no screen clearing
