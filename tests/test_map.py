"""`roundtable map`: scan -> ARCHITECTURE.md + PRD.md, then plan from the PRD."""

from roundtable.cli import main
from roundtable.config import write_default_config
from roundtable.store import Store


def _use_scripted(root):
    write_default_config(root)
    cfg = root / "roundtable.config.yaml"
    cfg.write_text(cfg.read_text().replace("provider: cli", "provider: scripted"))


def test_map_writes_docs_then_plan_consumes_prd(tmp_path):
    root = tmp_path / "proj"
    proj = str(root)

    # a small project to scan
    root.mkdir(parents=True)
    (root / "README.md").write_text("# Proj\n\nA tiny tool.\n")
    (root / "pyproject.toml").write_text("[project]\nname = \"proj\"\n")

    assert main(["init", proj, "--no-models"]) == 0
    _use_scripted(root)

    # map: produces the two docs
    assert main(["map", "--project", proj]) == 0
    store = Store(root)
    assert store.read_doc("ARCHITECTURE.md").strip()
    assert store.read_doc("PRD.md").strip()

    # map records its lifecycle events
    types = [e["type"] for e in store.read_events()]
    assert "map_started" in types and "map_done" in types

    # the confirm -> plan handoff: feed the PRD straight into planning
    prd_path = str(store.docs_dir / "PRD.md")
    assert main(["plan", "--prd", prd_path, "--project", proj]) == 0
    assert store.has_plan()
