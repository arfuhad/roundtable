"""REST control API tests: real socket, JSON round-trips, origin guard."""

import json
import threading
import urllib.error
import urllib.request

from roundtable.dashboard import make_server
from roundtable.errors import RoundtableError
from roundtable.models import Phase, Plan, Task
from roundtable.store import Store


def _serve(store):
    httpd, url = make_server(store, host="127.0.0.1", port=0)  # ephemeral port
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, url


def _req(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _plan(runner="claude"):
    return Plan(goal="api goal", main_runner=runner, phases=[
        Phase(id="p1", index=1, title="P", runner=runner,
              tasks=[Task(id="p1-t1", title="T", runner=runner)]),
    ])


def test_project_plan_roundtrip_and_approval_reset(tmp_path):
    store = Store(tmp_path)
    httpd, url = _serve(store)
    try:
        code, proj = _req(url + "/api/project")
        assert code == 200 and proj["root"] == str(store.root) and not proj["has_plan"]

        code, got = _req(url + "/api/plan")
        assert code == 200 and got["exists"] is False

        plan = _plan()
        plan.approved = True  # PUT must reset this
        code, res = _req(url + "/api/plan", "PUT", {"plan": json.loads(plan.model_dump_json())})
        assert code == 200 and res["ok"]

        code, got = _req(url + "/api/plan")
        assert got["exists"] and got["plan"]["goal"] == "api goal"
        assert got["plan"]["approved"] is False  # editing resets the human gate
        assert (store.plan_dir / "PLAN.md").exists()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_plan_put_rejects_invalid_graph(tmp_path):
    store = Store(tmp_path)
    httpd, url = _serve(store)
    try:
        bad = {
            "goal": "g",
            "phases": [
                {"id": "p1", "title": "P", "tasks": [
                    {"id": "t", "title": "A"}, {"id": "t", "title": "B"},  # duplicate id
                ]},
            ],
        }
        code, res = _req(url + "/api/plan", "PUT", {"plan": bad})
        assert code == 400 and "duplicate" in res["error"]
        assert not store.has_plan()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_approve_validates_runners_against_config(tmp_path):
    store = Store(tmp_path)
    httpd, url = _serve(store)
    try:
        # default config: provider cli, agents claude/codex/gemini
        store.save_plan(_plan(runner="m/task"))  # unknown agent
        code, res = _req(url + "/api/approve", "POST", {})
        assert code == 400 and "m/task" in res["error"]
        assert store.load_plan().approved is False

        store.save_plan(_plan(runner="claude"))
        code, res = _req(url + "/api/approve", "POST", {})
        assert code == 200 and res["ok"]
        assert store.load_plan().approved is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_run_refusals(tmp_path):
    store = Store(tmp_path)
    httpd, url = _serve(store)
    try:
        code, res = _req(url + "/api/run", "POST", {})
        assert code == 404  # no plan

        store.save_plan(_plan())
        store.write_run_pid(1)  # pid 1 is alive and not ours -> guard trips
        code, res = _req(url + "/api/run", "POST", {})
        assert code == 409 and "already in progress" in res["error"]
        store.clear_run_pid()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_stop_without_run(tmp_path):
    store = Store(tmp_path)
    httpd, url = _serve(store)
    try:
        code, res = _req(url + "/api/stop", "POST", {})
        assert code == 200 and res["ok"] is False and "no run.pid" in res["message"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dashboard_rejects_non_loopback_bind(tmp_path):
    store = Store(tmp_path)
    try:
        make_server(store, host="0.0.0.0", port=0)
        assert False, "expected non-loopback bind to be rejected"
    except RoundtableError as e:
        assert "localhost" in str(e)


def test_mutating_api_rejects_while_run_live(tmp_path):
    store = Store(tmp_path)
    store.write_run_pid(1)  # pid 1 is alive and not ours -> run considered live
    httpd, url = _serve(store)
    try:
        plan = _plan()
        plan_body = {"plan": json.loads(plan.model_dump_json())}
        for path, method, body in [
            ("/api/plan", "PUT", plan_body),
            ("/api/config", "PUT", {"text": "provider: scripted\n"}),
            ("/api/init", "POST", {}),
            ("/api/plan/generate", "POST", {"goal": "g"}),
        ]:
            code, res = _req(url + path, method, body)
            assert code == 400, path
            assert "run is in progress" in res["error"]
    finally:
        store.clear_run_pid()
        httpd.shutdown()
        httpd.server_close()


def test_resume_flow(tmp_path):
    store = Store(tmp_path)
    httpd, url = _serve(store)
    try:
        code, res = _req(url + "/api/resume", "POST", {"task": "p1-t9"})
        assert code == 400  # no checkpoint

        cp = store.hitl_path("p1-t2")
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"status": "waiting", "task_id": "p1-t2", "title": "T"}))
        code, res = _req(url + "/api/resume", "POST", {"task": "p1-t2"})
        assert code == 200 and res["ok"]
        assert json.loads(cp.read_text())["status"] == "approved"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_init_and_config_roundtrip(tmp_path):
    store = Store(tmp_path)
    httpd, url = _serve(store)
    try:
        code, res = _req(url + "/api/init", "POST", {})
        assert code == 200 and res["created_config"]
        assert (tmp_path / "roundtable.config.yaml").exists()
        assert store.plan_dir.exists()

        code, cfg = _req(url + "/api/config")
        assert code == 200 and cfg["exists"] and "provider: cli" in cfg["text"]

        code, res = _req(url + "/api/config", "PUT", {"text": "provider: scripted\n"})
        assert code == 200 and res["ok"]
        assert (tmp_path / "roundtable.config.yaml").read_text() == "provider: scripted\n"

        # invalid YAML and invalid schema are both rejected without writing
        code, res = _req(url + "/api/config", "PUT", {"text": "provider: [unclosed"})
        assert code == 400
        code, res = _req(url + "/api/config", "PUT",
                         {"text": "defaults:\n  max_concurrency: 0\n"})
        assert code == 400
        assert (tmp_path / "roundtable.config.yaml").read_text() == "provider: scripted\n"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_usage_endpoint(tmp_path):
    store = Store(tmp_path)
    store.record_event("usage", message="provider usage stats", calls=3, total_tokens=1200)
    httpd, url = _serve(store)
    try:
        code, res = _req(url + "/api/usage")
        assert code == 200
        assert res["latest"]["calls"] == 3 and res["latest"]["total_tokens"] == 1200
        assert len(res["history"]) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_plan_generate_status_idle(tmp_path):
    store = Store(tmp_path)
    httpd, url = _serve(store)
    try:
        code, res = _req(url + "/api/plan/generate")
        assert code == 200 and res["running"] is False and res["has_plan"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_origin_guard_blocks_foreign_sites(tmp_path):
    store = Store(tmp_path)
    store.save_plan(_plan())
    httpd, url = _serve(store)
    try:
        code, res = _req(url + "/api/approve", "POST", {},
                         headers={"Origin": "https://evil.example"})
        assert code == 403
        assert store.load_plan().approved is False

        # localhost + tauri origins pass
        code, _ = _req(url + "/api/approve", "POST", {},
                       headers={"Origin": "http://localhost:1420"})
        assert code == 200
        code, _ = _req(url + "/api/state", headers={"Origin": "tauri://localhost"})
        assert code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
