"""Zero-dependency web dashboard for a harness run.

A stdlib ``http.server`` serves one self-contained page (no build step, no JS
framework) that polls ``/api/state`` ~1s and renders live progress, what each
agent is doing now, per-agent + timing insights, and an event timeline. Reads the
on-disk state fresh per request, so it tracks a ``harness run`` happening in
another terminal — or just shows the plan before/after a run.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .insights import build_state
from .store import Store


def make_server(store: Store, *, host: str = "127.0.0.1", port: int = 8787) -> tuple[ThreadingHTTPServer, str]:
    """Build (but do not start) the dashboard server; returns (server, url)."""
    page = PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # keep the console quiet
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                body = json.dumps(build_state(store)).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", page)
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

        do_HEAD = do_GET

    httpd = ThreadingHTTPServer((host, port), Handler)
    bound_host, bound_port = httpd.server_address[0], httpd.server_address[1]
    shown = "localhost" if bound_host in ("0.0.0.0", "127.0.0.1") else bound_host
    return httpd, f"http://{shown}:{bound_port}"


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
  .s-failed .dot { background: #f85149; }
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
</style>
</head>
<body>
<header>
  <h1>LLM-HARNESS <span id="conn">connecting…</span></h1>
  <div class="goal" id="goal">—</div>
  <div>status <span id="badge" class="badge pending">—</span>
       <span id="counts" class="meta"></span></div>
  <div class="bar" style="margin-top:8px"><i id="fill" style="width:0%"></i></div>
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
      <span class="meta2">${esc(tk.runner)}${tk.duration_s!=null?` · ${dur(tk.duration_s)}`:""}</span></li>`).join("")}</ul>
    </div>`).join("");

  // insights
  const ba = Object.entries(st.by_agent).sort((a,b)=>b[1].tasks-a[1].tasks)
    .map(([k,v]) => `<span class="chip"><b>${esc(k)}</b> ${v.tasks} task${v.tasks===1?"":"s"}${v.total_s?` · ${dur(v.total_s)}`:""}</span>`).join("");
  const tm = st.timings;
  $("insights").innerHTML =
    `<div class="chips">${ba || '<span class="idle">no agent activity yet</span>'}</div>`
    + (tm.avg_task_s!=null ? `<div class="meta" style="margin-top:10px">avg task ${dur(tm.avg_task_s)}`
        + (tm.slowest?` · slowest ${esc(tm.slowest.task_id)} ${dur(tm.slowest.duration_s)}`:"") + `</div>` : "");

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
