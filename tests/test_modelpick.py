"""Model listing + interactive per-role picker (`roundtable models` / init)."""

from roundtable.config import AgentSpec, Config
from roundtable.discovery import AgentStatus
from roundtable.models import AgentRef
from roundtable.modelpick import (
    ModelChoice,
    ModelGroup,
    _ref_yaml,
    fmt_ref,
    groups_from_statuses,
    list_model_groups,
    omp_groups_from_json,
    pi_groups_from_text,
    pick_models,
    render_models_block,
    update_config_models,
)


# --------------------------------------------------------------------------- #
# listing / parsing
# --------------------------------------------------------------------------- #
def test_omp_groups_from_json():
    data = {"models": [
        {"provider": "anthropic", "id": "cx", "selector": "anthropic/cx", "name": "Claude X", "contextWindow": 200000},
        {"provider": "opencode-go", "id": "glm", "selector": "opencode-go/glm", "name": "GLM"},  # no ctx
        {"provider": "z", "id": "y"},  # no selector -> derived
        {"provider": "z", "not_a_dict": True},
        "junk",
    ]}
    groups = omp_groups_from_json(data)
    assert [g.name for g in groups] == ["anthropic", "opencode-go", "z"]  # sorted
    assert groups[0].choices[0].ref.model == "anthropic/cx"
    assert "200K" in groups[0].choices[0].label
    assert groups[2].choices[0].ref.model == "z/y"  # provider/id fallback


def test_pi_groups_from_text():
    text = "anthropic/claude-x\nopencode/mimo\n# a comment\nUsage: pi ...\n"
    groups = pi_groups_from_text(text)
    names = {g.name for g in groups}
    assert names == {"anthropic", "opencode"}


def test_pi_groups_from_text_empty_is_flagged():
    groups = pi_groups_from_text("no models here\n")
    assert len(groups) == 1 and not groups[0].installed and groups[0].note


def test_groups_from_statuses_cli():
    statuses = [
        AgentStatus(name="claude", binary="claude", installed=True, models=["opus", "haiku"]),
        AgentStatus(name="codex", binary="codex", installed=False, note="not on PATH"),
    ]
    groups = groups_from_statuses(statuses)
    claude = next(g for g in groups if g.name == "claude")
    assert [c.ref.model for c in claude.choices] == ["opus", "haiku"]
    assert all(c.ref.agent == "claude" for c in claude.choices)
    codex = next(g for g in groups if g.name == "codex")
    assert not codex.installed and codex.choices == []


def test_list_model_groups_noop_backends():
    assert list_model_groups(Config(provider="litellm")) == []
    assert list_model_groups(Config(provider="scripted")) == []


def test_list_model_groups_cli(tmp_path):
    cfg = Config(provider="cli", agents={
        "echo": AgentSpec(command=["true"], models_command=["printf", "a\nb\n"]),
    })
    groups = list_model_groups(cfg, timeout=10)
    g = next(x for x in groups if x.name == "echo")
    assert [c.ref.model for c in g.choices] == ["a", "b"]
    assert all(c.ref.agent == "echo" for c in g.choices)


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def test_fmt_and_ref_yaml():
    assert fmt_ref(AgentRef(model="m")) == "m"
    assert fmt_ref(AgentRef(agent="a", model="m")) == "a:m"
    assert fmt_ref(AgentRef()) == "(unset)"
    assert fmt_ref(None) == "(unset)"
    assert _ref_yaml(AgentRef(model="m")) == "{ model: m }"
    assert _ref_yaml(AgentRef(agent="a", model="m")) == "{ agent: a, model: m }"
    assert _ref_yaml(AgentRef(agent="a")) == "{ agent: a }"


# --------------------------------------------------------------------------- #
# interactive picker (scripted I/O)
# --------------------------------------------------------------------------- #
def _driver(inputs):
    it = iter(inputs)
    return lambda _prompt: next(it)


def _groups():
    return [
        ModelGroup(name="pA", choices=[ModelChoice("m1", AgentRef(model="pA/m1")),
                                       ModelChoice("m2", AgentRef(model="pA/m2"))]),
        ModelGroup(name="pB", choices=[ModelChoice("n1", AgentRef(model="pB/n1"))]),
    ]


def test_pick_models_provider_then_model():
    # planner: pA->m2 ; main: keep(Enter) ; phase: pB->n1 ; task: quit
    inputs = ["1", "2", "", "2", "1", "q"]
    picks = pick_models(_groups(), {"planner": AgentRef(model="old")},
                        input_fn=_driver(inputs), out=lambda s: None)
    assert picks["planner"].model == "pA/m2"
    assert "main" not in picks
    assert picks["phase"].model == "pB/n1"
    assert "task" not in picks


def test_pick_models_filter_on_long_list():
    big = ModelGroup(name="big", choices=[ModelChoice(f"m{i}", AgentRef(model=f"p/m{i}")) for i in range(30)])
    # planner: provider 1 (big), filter "m1" -> matches m1,m10..m19; pick 1 -> m1 ; then quit
    inputs = ["1", "m1", "1", "q"]
    picks = pick_models([big], {}, input_fn=_driver(inputs), out=lambda s: None)
    assert picks["planner"].model == "p/m1"


def test_pick_models_no_usable_groups():
    groups = [ModelGroup(name="omp", installed=False, note="not on PATH")]
    msgs = []
    picks = pick_models(groups, {}, input_fn=_driver([]), out=msgs.append)
    assert picks == {}
    assert any("no connected models" in m for m in msgs)


def test_pick_models_invalid_keeps():
    # planner: provider "9" (invalid) -> keeps; then quit
    picks = pick_models(_groups(), {}, input_fn=_driver(["9", "q"]), out=lambda s: None)
    assert picks == {}


# --------------------------------------------------------------------------- #
# config writing
# --------------------------------------------------------------------------- #
def test_render_models_block():
    block = render_models_block({
        "planner": AgentRef(model="m1"), "main": AgentRef(agent="claude", model="opus"),
        "phase": AgentRef(model="m3"), "task": AgentRef(),
    })
    assert block.splitlines()[0] == "models:"
    assert "planner: { model: m1 }" in block
    assert "main:    { agent: claude, model: opus }" in block


def test_update_config_models_preserves_rest(tmp_path):
    p = tmp_path / "roundtable.config.yaml"
    p.write_text(
        "provider: pi\n\n"
        "# a comment above models\n"
        "models:\n"
        "  planner: { model: old1 }\n"
        "  main:    { model: old2 }\n"
        "  phase:   { model: old3 }\n"
        "  task:    { model: old4 }\n\n"
        "pi:\n  flavor: omp\n  command: [\"omp\"]\n\n"
        "project_context: keep-me\n"
    )
    refs = {"planner": AgentRef(model="new1"), "main": AgentRef(agent="claude", model="opus"),
            "phase": AgentRef(model="old3"), "task": AgentRef(model="old4")}
    update_config_models(p, refs)

    from roundtable.config import load_config
    c = load_config(tmp_path)
    assert c.models.planner.model == "new1"
    assert c.models.main.agent == "claude" and c.models.main.model == "opus"
    txt = p.read_text()
    for keep in ("provider: pi", "flavor: omp", "command: [\"omp\"]", "project_context: keep-me",
                 "# a comment above models"):
        assert keep in txt


def test_update_config_models_appends_when_missing(tmp_path):
    p = tmp_path / "roundtable.config.yaml"
    p.write_text("provider: pi\npi:\n  flavor: omp\n")
    update_config_models(p, {"planner": AgentRef(model="x")})
    from roundtable.config import load_config
    assert load_config(tmp_path).models.planner.model == "x"


# --------------------------------------------------------------------------- #
# init interactive hook (the picker actually runs + writes during `roundtable init`)
# --------------------------------------------------------------------------- #
def test_init_runs_interactive_picker(tmp_path, monkeypatch):
    import argparse
    import builtins

    from roundtable import cli
    from roundtable import modelpick as mp
    from roundtable.config import load_config

    fake = [ModelGroup(name="prov", choices=[
        ModelChoice("Model One", AgentRef(model="prov/one")),
        ModelChoice("Model Two", AgentRef(model="prov/two")),
    ])]
    monkeypatch.setattr(mp, "list_model_groups", lambda *a, **k: fake)   # no real backend needed
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)               # force the interactive path
    # "y" to the init prompt, then: planner prov/two ; main prov/one ; phase keep ; task quit
    seq = iter(["y", "1", "2", "1", "1", "", "q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(seq))

    args = argparse.Namespace(dir=str(tmp_path), force=False, no_models=False, models_timeout=5.0)
    assert cli.cmd_init(args) == 0

    c = load_config(tmp_path)
    assert c.models.planner.model == "prov/two"
    assert c.models.main.model == "prov/one"
