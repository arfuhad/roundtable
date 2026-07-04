"""Zero-dependency web dashboard + local REST control API for a harness project.

A stdlib ``http.server`` serves one self-contained page (no build step, no JS
framework) that polls ``/api/state`` ~1s and renders live progress, plus a JSON
control API used by the page's buttons and by the desktop app:

    GET  /api/state           live run state snapshot (insights.build_state)
    GET  /api/project         project root + what exists (plan/config)
    GET  /api/plan            plan.json parsed
    PUT  /api/plan            save an edited plan (validates; resets approval)
    POST /api/plan/generate   spawn a detached `harness plan` (goal/prd/plan_file)
    GET  /api/plan/generate   poll a detached plan generation (running/log tail)
    POST /api/approve         validate runners + set approved
    POST /api/run             spawn a detached `harness run` (guarded by run.pid)
    POST /api/stop            SIGTERM the recorded run pid
    POST /api/resume          approve a waiting HITL task  {"task": "p1-t2"}
    POST /api/init            scaffold .harness/ + default config
    GET  /api/config          harness.config.yaml text
    PUT  /api/config          save config text (validated as YAML + schema)
    GET  /api/agents          probe configured CLIs + their models (?timeout=s)
    GET  /api/usage           provider usage snapshots from the event log

State-changing requests from browsers are restricted to localhost / Tauri
origins (non-browser clients send no Origin header and pass). Reads come
fresh from disk per request, so the server tracks runs started elsewhere.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml
from pydantic import ValidationError

from . import runctl
from .config import CONFIG_FILENAME, Config, load_config, write_default_config
from .discovery import discover
from .engine import validate_runners
from .errors import HarnessError
from .insights import build_state
from .models import Plan
from .store import Store


def _origin_allowed(origin: str) -> bool:
    """Browser origins that may drive the control API.

    Local pages (the dashboard itself, a dev server) and the Tauri desktop app
    are allowed; arbitrary web sites are not. Requests without an Origin header
    (curl, the desktop app's Rust side, same-origin GETs) never reach this.
    """
    if origin in ("tauri://localhost", "http://tauri.localhost"):
        return True
    try:
        host = urlparse(origin).hostname or ""
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1")


def make_server(store: Store, *, host: str = "127.0.0.1", port: int = 8787) -> tuple[ThreadingHTTPServer, str]:
    """Build (but do not start) the dashboard/API server; returns (server, url)."""
    page = PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # keep the console quiet
            pass

        # ---- plumbing ---------------------------------------------------- #
        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            origin = self.headers.get("Origin", "")
            if origin and _origin_allowed(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, "application/json; charset=utf-8", json.dumps(obj).encode("utf-8"))

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise HarnessError(f"request body is not valid JSON: {e}") from e
            if not isinstance(data, dict):
                raise HarnessError("request body must be a JSON object")
            return data

        def _guard_origin(self) -> bool:
            origin = self.headers.get("Origin", "")
            if origin and not _origin_allowed(origin):
                self._json(403, {"ok": False, "error": f"origin {origin!r} not allowed"})
                return False
            return True

        def do_OPTIONS(self) -> None:  # CORS preflight
            origin = self.headers.get("Origin", "")
            self.send_response(204)
            if origin and _origin_allowed(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Vary", "Origin")
            self.end_headers()

        def _dispatch(self, method: str) -> None:
            path = urlparse(self.path).path
            query = {k: v[-1] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                handler = _ROUTES.get((method, path))
                if handler is None:
                    if method == "GET" and path in ("/", "/index.html"):
                        self._send(200, "text/html; charset=utf-8", page)
                    else:
                        self._json(404, {"ok": False, "error": f"no route {method} {path}"})
                    return
                body = self._read_body() if method in ("POST", "PUT") else {}
                code, obj = handler(store, body, query)
                self._json(code, obj)
            except (HarnessError, ValidationError, yaml.YAMLError) as e:
                self._json(400, {"ok": False, "error": str(e)})
            except FileNotFoundError as e:
                self._json(404, {"ok": False, "error": str(e)})
            except Exception as e:  # keep the server alive; report the failure
                self._json(500, {"ok": False, "error": f"internal error: {e}"})

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_HEAD(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            if self._guard_origin():
                self._dispatch("POST")

        def do_PUT(self) -> None:
            if self._guard_origin():
                self._dispatch("PUT")

    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        raise HarnessError(
            f"could not start the dashboard on {host}:{port} ({e.strerror or e}); "
            f"the port is likely in use by another app — retry with "
            f"`harness dashboard --port <other>` (e.g. 8899)"
        ) from e
    bound_host, bound_port = httpd.server_address[0], httpd.server_address[1]
    # Show the explicit IPv4 loopback rather than "localhost": on dual-stack hosts
    # "localhost" can resolve to ::1 first and miss this IPv4-bound server.
    shown = "127.0.0.1" if bound_host in ("0.0.0.0", "127.0.0.1") else bound_host
    return httpd, f"http://{shown}:{bound_port}"


# --------------------------------------------------------------------------- #
# API handlers: (store, body, query) -> (status_code, json_payload)
# --------------------------------------------------------------------------- #
Payload = tuple[int, dict[str, Any]]


def _api_state(store: Store, body: dict, query: dict) -> Payload:
    return 200, build_state(store)


def _api_project(store: Store, body: dict, query: dict) -> Payload:
    return 200, {
        "root": str(store.root),
        "name": store.root.name,
        "has_plan": store.has_plan(),
        "has_config": (store.root / CONFIG_FILENAME).exists(),
        "run_pid": runctl.current_run_pid(store),
        "waiting": store.list_waiting_checkpoints(),
    }


def _api_plan_get(store: Store, body: dict, query: dict) -> Payload:
    if not store.has_plan():
        return 200, {"exists": False, "plan": None}
    return 200, {"exists": True, "plan": json.loads(store.manifest_path.read_text())}


def _api_plan_put(store: Store, body: dict, query: dict) -> Payload:
    plan_data = body.get("plan", body)
    plan = Plan.model_validate(plan_data)
    plan.approved = False  # editing a plan always resets the human gate
    store.save_plan(plan)
    store.write_plan_md(plan)
    store.record_event("plan_edited", message="plan edited via API (approval reset)")
    return 200, {"ok": True, "message": "plan saved (approval reset — re-approve to run)"}


def _api_plan_generate_post(store: Store, body: dict, query: dict) -> Payload:
    pid, msg = runctl.start_plan(
        store,
        goal=body.get("goal"),
        prd=body.get("prd"),
        plan_file=body.get("plan_file"),
        model=body.get("model"),
    )
    if pid is None:
        return 409, {"ok": False, "error": msg}
    return 200, {"ok": True, "pid": pid, "message": msg}


def _api_plan_generate_get(store: Store, body: dict, query: dict) -> Payload:
    return 200, runctl.plan_status(store)


def _api_approve(store: Store, body: dict, query: dict) -> Payload:
    if not store.has_plan():
        return 404, {"ok": False, "error": "no plan to approve; generate one first"}
    plan = store.load_plan()
    config = load_config(store.root)
    validate_runners(plan, config)  # HarnessError -> 400 with the problem list
    plan.approved = True
    store.save_plan(plan)
    store.record_event("plan_approved", message="plan approved")
    return 200, {"ok": True, "message": "plan approved"}


def _api_run(store: Store, body: dict, query: dict) -> Payload:
    if not store.has_plan():
        return 404, {"ok": False, "error": "no plan to run; generate and approve one first"}
    pid, msg = runctl.start_run(store, approve=bool(body.get("approve")))
    if pid is None:
        return 409, {"ok": False, "error": msg}
    return 200, {"ok": True, "pid": pid, "message": msg}


def _api_stop(store: Store, body: dict, query: dict) -> Payload:
    stopped, msg = runctl.stop_run(store)
    return 200, {"ok": stopped, "message": msg}


def _api_resume(store: Store, body: dict, query: dict) -> Payload:
    task_id = body.get("task")
    if not task_id:
        raise HarnessError("missing 'task' in request body")
    msg = runctl.approve_hitl(store, str(task_id))
    return 200, {"ok": True, "message": msg}


def _api_init(store: Store, body: dict, query: dict) -> Payload:
    store.scaffold()
    cfg = store.root / CONFIG_FILENAME
    created = False
    if not cfg.exists():
        write_default_config(store.root)
        created = True
    return 200, {
        "ok": True,
        "created_config": created,
        "message": f"initialized harness in {store.root}"
        + ("" if created else " (config already existed)"),
    }


def _api_config_get(store: Store, body: dict, query: dict) -> Payload:
    path = store.root / CONFIG_FILENAME
    return 200, {
        "path": str(path),
        "exists": path.exists(),
        "text": path.read_text() if path.exists() else "",
    }


def _api_config_put(store: Store, body: dict, query: dict) -> Payload:
    text = body.get("text")
    if not isinstance(text, str):
        raise HarnessError("missing 'text' (the YAML config content) in request body")
    Config.model_validate(yaml.safe_load(text) or {})  # reject invalid configs
    (store.root / CONFIG_FILENAME).write_text(text)
    return 200, {"ok": True, "message": f"wrote {CONFIG_FILENAME}"}


def _api_agents(store: Store, body: dict, query: dict) -> Payload:
    config = load_config(store.root)
    try:
        timeout = float(query.get("timeout", 10.0))
    except ValueError:
        timeout = 10.0
    statuses = asyncio.run(discover(config.agents, timeout=timeout))
    return 200, {
        "provider": config.provider,
        "models": {k: str(v) for k, v in {
            "planner": config.models.planner, "main": config.models.main,
            "phase": config.models.phase, "task": config.models.task,
        }.items()},
        "agents": [asdict(s) for s in statuses],
    }


def _api_usage(store: Store, body: dict, query: dict) -> Payload:
    snapshots = [
        {k: v for k, v in e.items() if k not in ("type", "msg")}
        for e in store.read_events()
        if e.get("type") == "usage"
    ]
    return 200, {"latest": snapshots[-1] if snapshots else None, "history": snapshots}


_ROUTES: dict[tuple[str, str], Any] = {
    ("GET", "/api/state"): _api_state,
    ("GET", "/api/project"): _api_project,
    ("GET", "/api/plan"): _api_plan_get,
    ("PUT", "/api/plan"): _api_plan_put,
    ("POST", "/api/plan/generate"): _api_plan_generate_post,
    ("GET", "/api/plan/generate"): _api_plan_generate_get,
    ("POST", "/api/approve"): _api_approve,
    ("POST", "/api/run"): _api_run,
    ("POST", "/api/stop"): _api_stop,
    ("POST", "/api/resume"): _api_resume,
    ("POST", "/api/init"): _api_init,
    ("GET", "/api/config"): _api_config_get,
    ("PUT", "/api/config"): _api_config_put,
    ("GET", "/api/agents"): _api_agents,
    ("GET", "/api/usage"): _api_usage,
}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llm-harness dashboard</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         background: #0d1117; color: #c9d1d9; }
  header { padding: 18px 22px; border-bottom: 1px solid #21262d; position: sticky; top: 0;
           background: #0d1117; }
  h1 { font-size: 15px; margin: 0 0 6px; letter-spacing: .04em; color: #8b949e; font-weight: 600; }
  .goal { color: #e6edf3; font-size: 16px; margin: 2px 0 12px; }
  .badge { padding: 2px 9px; border-radius: 999px; font-size: 12px; margin-left: 8px; }
  .running { background: #1f6feb33; color: #58a6ff; }
  .done    { background: #23863633; color: #3fb950; }
  .failed  { background: #f8514933; color: #f85149; }
  .pending { background: #6e768133; color: #8b949e; }
  .bar { height: 8px; border-radius: 5px; background: #21262d; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: linear-gradient(90deg,#1f6feb,#3fb950); transition: width .4s; }
  .meta { color: #8b949e; font-size: 12px; margin-top: 6px; }
  main { display: grid; grid-template-columns: 1.5fr 1fr; gap: 16px; padding: 16px 22px; align-items: start; }
  @media (max-width: 860px){ main { grid-template-columns: 1fr; } }
  .card { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 14px 16px; }
  .card h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: #8b949e;
             margin: 0 0 10px; }
  .now { border-color: #1f6feb55; }
  .now .row { display: flex; justify-content: space-between; gap: 10px; padding: 6px 0; }
  .now .t { color: #e6edf3; } .now .r { color: #58a6ff; } .now .e { color: #8b949e; }
  .idle { color: #6e7681; }
  .phase { margin-bottom: 12px; }
  .phase .ph { display: flex; justify-content: space-between; color: #e6edf3; margin-bottom: 4px; }
  .ph .runner { color: #8b949e; font-size: 12px; }
  .tasks { list-style: none; margin: 0; padding: 0 0 0 2px; }
  .tasks li { display: flex; align-items: center; gap: 8px; padding: 2px 0; color: #adbac7; }
  .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; background: #30363d; }
  .s-done .dot { background: #3fb950; } .s-in_progress .dot { background: #58a6ff; animation: pulse 1s infinite; }
  .s-failed .dot { background: #f85149; } .s-skipped .dot { background: #6e7681; opacity: .5; }
  .s-waiting .dot { background: #d29922; animation: pulse 1.5s infinite; }
  @keyframes pulse { 50% { opacity: .35; } }
  .tasks .id { color: #6e7681; } .tasks .meta2 { margin-left: auto; color: #6e7681; font-size: 12px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: #21262d; border-radius: 6px; padding: 3px 8px; font-size: 12px; }
  .chip b { color: #e6edf3; }
  .ev { list-style: none; margin: 0; padding: 0; max-height: 320px; overflow: auto; }
  .ev li { display: flex; gap: 8px; padding: 3px 0; border-bottom: 1px solid #1b1f24; font-size: 12.5px; }
  .ev .ts { color: #6e7681; flex: none; } .ev .ty { color: #58a6ff; flex: none; width: 110px; }
  .ev .ms { color: #adbac7; }
  #conn { font-size: 12px; color: #6e7681; }
  #conn.off { color: #f85149; }
  #controls { margin-top: 10px; display: flex; align-items: center; gap: 8px; }
  button { font: inherit; font-size: 12px; padding: 4px 12px; border-radius: 6px; cursor: pointer;
           background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
  button:hover { background: #30363d; }
  button.primary { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  button.primary:hover { background: #388bfd; }
  button.danger { color: #f85149; border-color: #f8514966; }
  button.mini { padding: 0 8px; font-size: 11px; margin-left: 6px; color: #d29922; border-color: #d2992266; }
  #actmsg { font-size: 12px; color: #8b949e; }
</style>
</head>
<body>
<header>
  <h1>LLM-HARNESS <span id="conn">connecting…</span></h1>
  <div class="goal" id="goal">—</div>
  <div>status <span id="badge" class="badge pending">—</span>
       <span id="counts" class="meta"></span></div>
  <div class="bar" style="margin-top:8px"><i id="fill" style="width:0%"></i></div>
  <div id="controls">
    <button id="btn-approve" class="primary" style="display:none">approve plan</button>
    <button id="btn-run" class="primary" style="display:none">▶ run</button>
    <button id="btn-stop" class="danger" style="display:none">■ stop</button>
    <span id="actmsg"></span>
  </div>
</header>
<main>
  <div>
    <div class="card now" id="nowcard"><h2>Now running</h2><div id="now"></div></div>
    <div class="card" style="margin-top:16px"><h2>Phases &amp; tasks</h2><div id="phases"></div></div>
  </div>
  <div>
    <div class="card"><h2>Insights</h2><div id="insights"></div></div>
    <div class="card" style="margin-top:16px"><h2>Events</h2><ul class="ev" id="events"></ul></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const dur = s => { if (s==null) return "—"; s=Math.max(0,Math.floor(s)); const h=(s/3600|0), m=((s%3600)/60|0), x=s%60;
  const p=n=>String(n).padStart(2,"0"); return h?`${h}:${p(m)}:${p(x)}`:`${m}:${p(x)}`; };

async function post(path, body) {
  try {
    const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                                 body: JSON.stringify(body || {})});
    const j = await r.json();
    $("actmsg").textContent = j.message || j.error || "";
  } catch (e) { $("actmsg").textContent = String(e); }
  clearTimeout(post._t); post._t = setTimeout(() => { $("actmsg").textContent = ""; }, 8000);
  tick();
}
$("btn-approve").onclick = () => post("/api/approve");
$("btn-run").onclick = () => post("/api/run");
$("btn-stop").onclick = () => { if (confirm("Stop the in-progress run?")) post("/api/stop"); };
$("phases").onclick = e => {
  const t = e.target.closest("button[data-task]");
  if (t) post("/api/resume", {task: t.dataset.task});
};

async function tick() {
  let st;
  try { st = await (await fetch("/api/state", {cache:"no-store"})).json(); }
  catch (e) { $("conn").textContent = "● disconnected"; $("conn").className = "off"; return; }
  $("conn").textContent = "● live"; $("conn").className = "";
  if (!st.exists) { $("goal").textContent = "no plan yet — run `harness plan`"; return; }

  $("goal").textContent = st.goal;
  const t = st.totals;
  $("badge").textContent = st.status; $("badge").className = "badge " + st.status;
  $("counts").textContent = `${t.done}/${t.tasks} tasks · ${t.percent}% · ${t.phases} phases`
    + (t.in_progress?` · ${t.in_progress} running`:"") + (t.failed?` · ${t.failed} failed`:"");
  $("fill").style.width = t.percent + "%";

  // controls
  $("btn-approve").style.display = !st.approved ? "" : "none";
  $("btn-run").style.display = st.approved && st.status !== "running" ? "" : "none";
  $("btn-stop").style.display = st.status === "running" ? "" : "none";

  // now
  $("nowcard").style.display = "";
  if (st.now.length) {
    $("now").innerHTML = st.now.map(n => `<div class="row">
      <span><span class="t">${esc(n.title)}</span> <span class="id">${esc(n.id)}</span></span>
      <span class="r">${esc(n.runner)}</span><span class="e">${dur(n.elapsed_s)}</span></div>`).join("");
  } else {
    $("now").innerHTML = `<div class="idle">${st.status==="done"?"run complete":"idle — nothing running"}</div>`;
  }

  // phases
  $("phases").innerHTML = st.phases.map(p => `<div class="phase">
    <div class="ph"><span>[${p.index}] ${esc(p.title)} <span class="runner">${esc(p.runner)}</span></span>
      <span class="meta">${p.done}/${p.total}</span></div>
    <ul class="tasks">${p.tasks.map(tk => `<li class="s-${tk.status}">
      <span class="dot"></span><span class="id">${esc(tk.id)}</span> ${esc(tk.title)}
      ${tk.status==="waiting"?`<button class="mini" data-task="${esc(tk.id)}">approve task</button>`:""}
      <span class="meta2">${esc(tk.runner)}${tk.duration_s!=null?` · ${dur(tk.duration_s)}`:""}</span></li>`).join("")}</ul>
    </div>`).join("");

  // insights
  const ba = Object.entries(st.by_agent).sort((a,b)=>b[1].tasks-a[1].tasks)
    .map(([k,v]) => `<span class="chip"><b>${esc(k)}</b> ${v.tasks} task${v.tasks===1?"":"s"}${v.total_s?` · ${dur(v.total_s)}`:""}</span>`).join("");
  const tm = st.timings;
  $("insights").innerHTML =
    `<div class="chips">${ba || '<span class="idle">no agent activity yet</span>'}</div>`
    + (tm.avg_task_s!=null ? `<div class="meta" style="margin-top:10px">avg task ${dur(tm.avg_task_s)}`
        + (tm.slowest?` · slowest ${esc(tm.slowest.task_id)} ${dur(tm.slowest.duration_s)}`:"") + `</div>` : "")
    + (st.usage ? `<div class="meta" style="margin-top:10px">usage: <b>${st.usage.total_tokens.toLocaleString()}</b> tokens `
        + `(${st.usage.prompt_tokens.toLocaleString()} in / ${st.usage.completion_tokens.toLocaleString()} out) · ${st.usage.calls} calls`
        + (st.usage.estimated ? ` <span class="idle">(est)</span>` : "") + `</div>` : "");

  // events
  $("events").innerHTML = st.events.map(e => `<li>
    <span class="ts">${esc((e.ts||"").slice(11,19))}</span>
    <span class="ty">${esc(e.type||"")}</span>
    <span class="ms">${esc(e.msg || e.task_id || "")}</span></li>`).join("");
}
tick(); setInterval(tick, 1000);
</script>
</body>
</html>
"""
