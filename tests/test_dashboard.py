"""Dashboard server smoke test: real socket, /api/state JSON + the HTML page."""

import json
import threading
import urllib.request

from roundtable.dashboard import make_server
from roundtable.models import Phase, Plan, Task
from roundtable.store import Store


def _serve(store):
    httpd, url = make_server(store, host="127.0.0.1", port=0)  # ephemeral port
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, url, t


def test_dashboard_serves_state_and_page(tmp_path):
    store = Store(tmp_path)
    store.save_plan(Plan(goal="dash goal", main_runner="claude:opus-4.8", phases=[
        Phase(id="p1", index=1, title="P", runner="agy:gemini-3.5-flash",
              tasks=[Task(id="p1-t1", title="T", runner="opencode:mimo-v2.5-pro")]),
    ]))
    httpd, url, _ = _serve(store)
    try:
        with urllib.request.urlopen(url + "/api/state", timeout=5) as r:
            assert r.status == 200
            state = json.loads(r.read())
        assert state["exists"] and state["goal"] == "dash goal"
        assert state["phases"][0]["tasks"][0]["runner"] == "opencode:mimo-v2.5-pro"

        with urllib.request.urlopen(url + "/", timeout=5) as r:
            body = r.read().decode()
        assert r.status == 200 and "roundtable" in body.lower()
        assert "/api/state" in body  # the page polls the API
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dashboard_404(tmp_path):
    store = Store(tmp_path)
    store.save_plan(Plan(goal="g", phases=[]))
    httpd, url, _ = _serve(store)
    try:
        try:
            urllib.request.urlopen(url + "/nope", timeout=5)
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
