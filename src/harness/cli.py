"""Command-line interface.

    harness init    [dir]                 scaffold .harness/ + default config;
                                          lists installed CLIs and their models
    harness agents                        list configured agents + their models
    harness map                           scan an existing project into docs + a PRD
    harness plan    --goal "..."          plan from a goal
    harness plan    --prd FILE            plan from a PRD / requirements file
    harness plan    --plan FILE           ingest an existing plan (JSON or doc)
    harness approve                       approve the plan (the human gate)
    harness run                           execute all phases autonomously
                                          (live web dashboard + inline progress)
    harness status                        show phase/task progress
    harness dashboard                     live web dashboard (stdlib http.server)
    harness watch                         live terminal dashboard
    harness mcp                           start the MCP server (stdio)

Designed to run *inside an existing project*: all harness artifacts live under
``.harness/`` and agents run with the project directory as their working dir, so
they read and edit the real files. After ``plan``, review ``.harness/plan/PLAN.md``;
``run`` refuses until ``approve``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import sys
import threading
import time
from pathlib import Path

from .agents import Analyst, Planner
from .config import Config, load_config, write_default_config
from .dashboard import make_server
from .discovery import AgentStatus, discover
from .engine import Engine
from .errors import HarnessError
from .insights import build_state, render_text
from .llm import CLIProvider, LiteLLMProvider, LLMProvider, ScriptedProvider
from .models import AgentRef, Plan, Status
from .scan import build_digest
from .store import Store


def build_provider(config: Config, cwd: Path | str | None = None) -> LLMProvider:
    if config.provider == "cli":
        return CLIProvider(
            config.agents, cwd=cwd,
            timeout=config.defaults.timeout, max_retries=config.defaults.max_retries,
        )
    if config.provider == "scripted":
        return ScriptedProvider()
    if config.provider == "litellm":
        return LiteLLMProvider(max_retries=config.defaults.max_retries)
    raise HarnessError(f"unknown provider {config.provider!r} (use 'cli', 'litellm', or 'scripted')")


def _role_models(config: Config) -> dict[str, AgentRef]:
    return {
        "planner": config.models.planner,
        "main": config.models.main,
        "phase": config.models.phase,
        "task": config.models.task,
    }


def _first_line(text: str, default: str = "") -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("# ").strip()
        if s:
            return s[:120]
    return default


async def make_plan(
    store: Store,
    config: Config,
    *,
    goal: str | None = None,
    prd_path: str | None = None,
    plan_path: str | None = None,
    planner_model: str | None = None,
) -> Plan:
    roles = _role_models(config)
    allowed = sorted({str(r) for r in roles.values()})
    provider = build_provider(config, store.root)
    planner_ref = AgentRef.model_validate(planner_model) if planner_model else roles["planner"]
    planner = Planner(provider, planner_ref, config.defaults.temperature)

    source_text = ""
    if plan_path:
        source_text = Path(plan_path).read_text()
        try:
            plan = Plan.model_validate_json(source_text)  # already in our schema -> no LLM
        except Exception:
            plan = await planner.structure_plan(source_text, allowed, roles)  # convert via LLM
    elif prd_path or goal:
        prd_text = Path(prd_path).read_text() if prd_path else ""
        source_text = prd_text
        combined = "\n\n".join(x for x in [goal or "", prd_text] if x).strip()
        plan = await planner.create_plan(combined, allowed, roles)
    else:
        raise HarnessError("nothing to plan from: pass --goal, --prd, or --plan")

    # Backfill metadata + role defaults; normalize indices; re-validate the graph.
    if goal:
        plan.goal = goal.strip()
    if not plan.goal:
        plan.goal = _first_line(source_text, "(imported plan)")
    plan.created_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    plan.main_runner = plan.main_runner or roles["main"]
    plan.runners = roles
    for i, phase in enumerate(plan.phases, start=1):
        phase.index = i
        phase.runner = phase.runner or roles["phase"]
        for task in phase.tasks:
            task.runner = task.runner or roles["task"]
    plan = Plan.model_validate(plan.model_dump())  # re-run validators after edits

    archive_dir = store.archive_current_run()
    if archive_dir:
        print(f"archived previous run → {archive_dir.relative_to(store.root)}")

    store.scaffold()
    store.write_brief(plan.goal if not source_text else source_text)
    store.save_plan(plan)
    store.write_plan_md(plan)
    return plan


async def map_project(
    store: Store,
    config: Config,
    *,
    target: Path,
    analyst_model: str | None = None,
    max_files: int = 400,
    max_bytes: int = 60000,
) -> tuple[str, str]:
    """Scan ``target`` and write ARCHITECTURE.md + PRD.md to ``.harness/docs/``."""
    provider = build_provider(config, target)  # cwd = scanned project
    ref = AgentRef.model_validate(analyst_model) if analyst_model else config.models.main
    store.scaffold()
    digest = build_digest(target, max_files=max_files, max_total_bytes=max_bytes)
    store.record_event(
        "map_started", message=f"mapping {target}", target=str(target), digest_bytes=len(digest)
    )
    analyst = Analyst(provider, ref, config.defaults.temperature)
    arch = await analyst.architecture(digest)
    store.write_doc("ARCHITECTURE.md", arch)
    prd = await analyst.prd(digest, arch)
    store.write_doc("PRD.md", prd)
    store.record_event("map_done", message="map complete", docs=["ARCHITECTURE.md", "PRD.md"])
    return arch, prd


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def _print_agents(statuses: list[AgentStatus], *, limit: int | None = None) -> None:
    if not statuses:
        print("  (no agents configured)")
        return
    for st in sorted(statuses, key=lambda s: (not s.installed, s.name)):
        if not st.installed:
            print(f"  [ ] {st.name}  ({st.binary}) — not on PATH")
        elif st.models:
            shown = st.models if limit is None else st.models[:limit]
            print(f"  [x] {st.name}  ({st.binary}) — {len(st.models)} model(s):")
            for m in shown:
                print(f"          {m}")
            if limit is not None and len(st.models) > limit:
                print(f"          … (+{len(st.models) - limit} more — run `harness agents`)")
        else:
            print(f"  [x] {st.name}  ({st.binary}) — models: n/a ({st.note})")


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root)
    store.scaffold()
    cfg_path = root / "harness.config.yaml"
    if cfg_path.exists() and not args.force:
        print(f"config already exists: {cfg_path} (use --force to overwrite)")
    else:
        write_default_config(root)
        print(f"wrote {cfg_path}")
    print(f"initialized harness in {root} (artifacts under {store.base})")

    config = load_config(root)
    if config.provider == "cli" and not args.no_models:
        print("\navailable agents & models (probing installed CLIs — Ctrl-C or --no-models to skip):")
        statuses = asyncio.run(discover(config.agents, timeout=args.models_timeout))
        _print_agents(statuses, limit=8)
        print("\nassign an agent:model to each role in harness.config.yaml under `models:`.")

    print("next: edit harness.config.yaml, then `harness plan --goal \"...\"` "
          "(or --prd FILE / --plan FILE)")
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    root = Path(args.project).resolve()
    config = load_config(root)
    if config.provider != "cli":
        print(f"provider is {config.provider!r}; agent discovery applies to `provider: cli`.")
        return 0
    statuses = asyncio.run(discover(config.agents, timeout=args.timeout))
    if args.json:
        from dataclasses import asdict
        import json
        print(json.dumps([asdict(s) for s in statuses], indent=2))
    else:
        _print_agents(statuses, limit=None)
    return 0


def cmd_map(args: argparse.Namespace) -> int:
    root = Path(args.project).resolve()
    target = Path(args.target).resolve() if args.target else root
    config = load_config(root)
    store = Store(root)
    asyncio.run(map_project(
        store, config, target=target, analyst_model=args.model,
        max_files=args.max_files, max_bytes=args.max_bytes,
    ))
    prd_path = store.docs_dir / "PRD.md"
    print(f"mapped {target}")
    print(f"wrote {store.docs_dir / 'ARCHITECTURE.md'} and {prd_path}")
    print(f"review/edit {prd_path}, then: "
          f"harness plan --prd {prd_path} --project {args.project}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    root = Path(args.project).resolve()
    config = load_config(root)
    store = Store(root)
    plan = asyncio.run(make_plan(
        store, config, goal=args.goal, prd_path=args.prd, plan_path=args.plan,
        planner_model=args.model,
    ))
    print(f"planned {len(plan.phases)} phase(s) for: {plan.goal}")
    for p in plan.phases:
        print(f"  [{p.id}] {p.title}  ({len(p.tasks)} task(s), runner={p.runner})")
    print(f"\nwrote {store.manifest_path}")
    print(f"review {store.plan_dir / 'PLAN.md'}, then: harness approve --project {args.project}")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    root = Path(args.project).resolve()
    store = Store(root)
    plan = store.load_plan()
    plan.approved = True
    store.save_plan(plan)
    print("plan approved. run it with: harness run --project", args.project)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.project).resolve()
    config = load_config(root)
    store = Store(root)

    if args.approve:
        # Best-effort auto-approve; a missing/invalid plan is surfaced by the engine.
        try:
            plan = store.load_plan()
            if not plan.approved:
                plan.approved = True
                store.save_plan(plan)
                print("plan auto-approved via --approve flag.")
        except (FileNotFoundError, HarnessError):
            pass

    provider = build_provider(config, root)
    engine = Engine(store, config, provider)
    return asyncio.run(_run_with_progress(engine, store, args))


async def _run_with_progress(engine: Engine, store: Store, args: argparse.Namespace) -> int:
    """Drive the engine while showing live progress: a web dashboard link up front
    and the terminal ``watch`` view rendered inline (TTY only)."""
    live = sys.stdout.isatty() and not args.no_watch

    # Serve the web dashboard in a background thread so the link is live as the run
    # goes. Port 0 picks a free port, side-stepping any port collision.
    httpd = url = None
    if not args.no_dashboard:
        try:
            httpd, url = make_server(store, host=args.host, port=args.port)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            if args.open:
                import webbrowser
                webbrowser.open(url)
        except HarnessError as e:
            print(f"(web dashboard unavailable: {e})", file=sys.stderr)

    def banner() -> str:
        web = f"dashboard: {url}" if url else "dashboard: off (--no-dashboard)"
        return f"{web}    ·    Ctrl-C to stop (resumable)"

    print(banner())  # show the link immediately, before the first render

    task = asyncio.create_task(engine.run())
    try:
        if live:
            while not task.done():
                try:
                    frame = render_text(build_state(store))
                except Exception:  # transient read during an engine write
                    frame = "loading…"
                sys.stdout.write("\x1b[2J\x1b[H" + banner() + "\n\n" + frame + "\n")
                sys.stdout.flush()
                await asyncio.sleep(max(0.1, args.interval))
        plan = await task  # awaits completion; re-raises any engine error
    except KeyboardInterrupt:
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        print("\ninterrupted — re-run `harness run` to resume", file=sys.stderr)
        return 130
    finally:
        if httpd:
            httpd.shutdown()
            httpd.server_close()

    if live:
        try:
            sys.stdout.write("\x1b[2J\x1b[H" + render_text(build_state(store)) + "\n")
        except Exception:
            pass
    print(f"\nrun complete: {len(plan.phases)} phase(s), status={plan.status.value}")
    print(f"docs: {store.docs_dir}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.project).resolve()
    store = Store(root)
    plan = store.load_plan()
    mark = {
        Status.done: "[x]", Status.in_progress: "[~]", Status.pending: "[ ]",
        Status.failed: "[!]", Status.skipped: "[-]", Status.waiting: "[?]",
    }
    print(f"goal: {plan.goal}")
    print(f"role runners: {({k: str(v) for k, v in plan.runners.items()})}")
    print(f"approved: {plan.approved} | status: {plan.status.value}\n")
    for p in plan.phases:
        print(f"{mark[p.status]} Phase {p.index}: {p.title}  [{p.id}]")
        for t in p.tasks:
            dep = f" <- {','.join(t.depends_on)}" if t.depends_on else ""
            print(f"    {mark[t.status]} {t.title}  [{t.id}] ({t.runner}){dep}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    store = Store(Path(args.project).resolve())
    if not store.has_plan():
        raise HarnessError("no plan to show; run `harness plan` first")
    httpd, url = make_server(store, host=args.host, port=args.port)
    print(f"dashboard live at {url}  (Ctrl-C to stop)")
    print("watching .harness/ — run `harness run` in another terminal to see it update")
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping dashboard")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    import json as _json
    root = Path(args.project).resolve()
    store = Store(root)
    checkpoint = store.hitl_path(args.task)
    if not checkpoint.exists():
        raise HarnessError(
            f"no pending approval checkpoint for task {args.task!r}; "
            f"is the run paused and waiting at that task?"
        )
    try:
        data = _json.loads(checkpoint.read_text())
    except (ValueError, OSError) as e:
        raise HarnessError(f"could not read checkpoint for task {args.task!r}: {e}") from e
    if data.get("status") != "waiting":
        raise HarnessError(
            f"task {args.task!r} checkpoint has status {data.get('status')!r}, expected 'waiting'"
        )
    data["status"] = "approved"
    checkpoint.write_text(_json.dumps(data))
    print(f"task {args.task!r} approved — run will continue")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from .mcp import run_server
    except ImportError:
        print("error: 'mcp' package not installed — run: pip install 'llm-harness[mcp]'",
              file=sys.stderr)
        return 2
    run_server()
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    store = Store(Path(args.project).resolve())
    if not store.has_plan():
        raise HarnessError("no plan to watch; run `harness plan` first")
    try:
        while True:
            state = build_state(store)
            sys.stdout.write("\x1b[2J\x1b[H")  # clear screen + home
            sys.stdout.write(render_text(state) + "\n")
            sys.stdout.flush()
            if state.get("status") == "done" and not args.follow:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description="Multi-LLM planning & orchestration harness")
    p.add_argument("-v", "--verbose", action="store_true", help="enable verbose/debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="scaffold .harness/ + default config")
    sp.add_argument("dir", nargs="?", default=".")
    sp.add_argument("--force", action="store_true", help="overwrite an existing config")
    sp.add_argument("--no-models", action="store_true",
                    help="skip probing installed CLIs for their model lists")
    sp.add_argument("--models-timeout", type=float, default=15.0,
                    help="seconds to wait per CLI when listing models (default 15)")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("agents", help="list configured agents, which are installed, and their models")
    sp.add_argument("--project", default=".")
    sp.add_argument("--timeout", type=float, default=20.0,
                    help="seconds to wait per CLI when listing models (default 20)")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_agents)

    sp = sub.add_parser("map", help="scan an existing project into outline docs + a PRD to confirm")
    sp.add_argument("--project", default=".", help="harness root (where .harness/ lives)")
    sp.add_argument("--target", default=None,
                    help="codebase to scan (default: the project root)")
    sp.add_argument("--model", default=None,
                    help="override the analyst agent/model (default: the 'main' role)")
    sp.add_argument("--max-files", type=int, default=400,
                    help="max files to include in the scan digest (default 400)")
    sp.add_argument("--max-bytes", type=int, default=60000,
                    help="max total bytes of file contents in the digest (default 60000)")
    sp.set_defaults(func=cmd_map)

    sp = sub.add_parser("plan", help="generate or import a plan")
    sp.add_argument("--goal", default=None, help="goal text")
    sp.add_argument("--prd", default=None, help="path to a PRD / requirements file")
    sp.add_argument("--plan", default=None, help="path to an existing plan (JSON schema or free-form)")
    sp.add_argument("--project", default=".")
    sp.add_argument("--model", default=None, help="override the planner model/agent")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("approve", help="approve the plan (human gate)")
    sp.add_argument("--project", default=".")
    sp.set_defaults(func=cmd_approve)

    sp = sub.add_parser("run", help="execute the approved plan (live progress + web dashboard)")
    sp.add_argument("--project", default=".")
    sp.add_argument("--approve", action="store_true", help="auto-approve the plan before running")
    sp.add_argument("--no-dashboard", action="store_true",
                    help="don't serve the live web dashboard during the run")
    sp.add_argument("--no-watch", action="store_true",
                    help="don't render the live terminal view (just run)")
    sp.add_argument("--interval", type=float, default=1.0,
                    help="live terminal refresh seconds (default 1)")
    sp.add_argument("--host", default="127.0.0.1", help="dashboard bind address")
    sp.add_argument("--port", type=int, default=0,
                    help="dashboard port (0 = pick a free port)")
    sp.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status", help="show phase/task progress")
    sp.add_argument("--project", default=".")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("dashboard", help="serve a live web dashboard of the run")
    sp.add_argument("--project", default=".")
    sp.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 for LAN access)")
    sp.add_argument("--port", type=int, default=8787)
    sp.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser("watch", help="live terminal dashboard of the run")
    sp.add_argument("--project", default=".")
    sp.add_argument("--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    sp.add_argument("--follow", action="store_true", help="keep watching after the run completes")
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser(
        "resume",
        help="approve a paused HITL task and let the run continue",
    )
    sp.add_argument("--task", required=True, help="task ID to approve (e.g. p1-t2)")
    sp.add_argument("--project", default=".")
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("mcp", help="start the harness MCP server over stdio (requires mcp extra)")
    sp.set_defaults(func=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    import logging
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if getattr(args, 'verbose', False) else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except HarnessError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
