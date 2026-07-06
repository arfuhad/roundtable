"""Interactive per-role model selection for `roundtable models` and `roundtable init`.

Lists the models actually connected to the configured backend and lets you pick one
per role (planner / main / phase / task) with a two-step *provider -> model* prompt,
then writes the choices into ``roundtable.config.yaml`` (preserving the rest of the
file). Works for two backends:

* ``pi``  — the pi-family tool reports its own catalog: ``omp models --json`` (rich)
            or ``pi --list-models`` (best-effort text). "provider" is the model's
            provider (anthropic, opencode-go, ...); the chosen ref is ``{model: sel}``.
* ``cli`` — each configured agent's ``models_command`` (via :mod:`.discovery`).
            "provider" is the agent (claude, opencode, ...); the ref is
            ``{agent, model}``.

Listing/parsing are pure functions; the picker takes injectable ``input_fn``/``out``
so it can be driven from tests without a real terminal.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import CONFIG_FILENAME, Config
from .discovery import AgentStatus, discover
from .models import AgentRef

ROLES = ["planner", "main", "phase", "task"]


@dataclass
class ModelChoice:
    label: str          # human-readable line shown in the picker
    ref: AgentRef       # what gets written to the config for this choice


@dataclass
class ModelGroup:
    name: str           # provider (pi backend) or agent (cli backend)
    installed: bool = True
    note: str = ""      # why there are no choices, if empty
    choices: list[ModelChoice] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Listing / grouping (pure where possible)
# --------------------------------------------------------------------------- #
def list_model_groups(config: Config, *, timeout: float = 20.0) -> list[ModelGroup]:
    """Model groups for the configured backend (empty for litellm/scripted)."""
    if config.provider == "pi":
        return _pi_groups(config, timeout)
    if config.provider == "cli":
        return _cli_groups(config, timeout)
    return []


def _pi_groups(config: Config, timeout: float) -> list[ModelGroup]:
    flavor = (config.pi.flavor or "pi").lower()
    cmd = list(config.pi.command) or (["omp"] if flavor == "omp" else ["pi"])
    if shutil.which(cmd[0]) is None:
        return [ModelGroup(name=cmd[0], installed=False, note="not on PATH")]
    argv = cmd + (["models", "--json"] if flavor == "omp" else ["--list-models"])
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:  # pragma: no cover - env-specific
        return [ModelGroup(name=cmd[0], installed=False, note=f"model query failed: {e}")]
    if res.returncode != 0:
        tail = (res.stderr or res.stdout).strip().splitlines()
        return [ModelGroup(name=cmd[0], installed=False,
                           note=f"exited {res.returncode}" + (f": {tail[-1][:80]}" if tail else ""))]
    if flavor == "omp":
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            return [ModelGroup(name="omp", installed=False, note="`omp models --json` returned non-JSON")]
        return omp_groups_from_json(data)
    return pi_groups_from_text(res.stdout)


def omp_groups_from_json(data: object) -> list[ModelGroup]:
    """Parse ``omp models --json`` ({"models":[{provider,selector,name,contextWindow,...}]})."""
    recs = data.get("models") if isinstance(data, dict) else data
    groups: dict[str, ModelGroup] = {}
    for r in recs or []:
        if not isinstance(r, dict):
            continue
        prov = r.get("provider") or "?"
        sel = r.get("selector") or (f"{prov}/{r['id']}" if r.get("id") else None)
        if not sel:
            continue
        name = r.get("name") or r.get("id") or sel
        ctx = r.get("contextWindow")
        ctx_s = f"  · {ctx // 1000}K" if isinstance(ctx, int) and ctx >= 1000 else ""
        groups.setdefault(prov, ModelGroup(name=prov)).choices.append(
            ModelChoice(label=f"{name}{ctx_s}   [{sel}]", ref=AgentRef(model=sel))
        )
    return [groups[k] for k in sorted(groups)]


def pi_groups_from_text(text: str) -> list[ModelGroup]:
    """Best-effort parse of ``pi --list-models`` text output (one model id per line)."""
    groups: dict[str, ModelGroup] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "-", "Usage", "Available", "Models")):
            continue
        tok = s.split()[0]
        if "/" not in tok and " " in s:  # a header-ish line, skip
            continue
        prov = tok.split("/")[0] if "/" in tok else "pi"
        groups.setdefault(prov, ModelGroup(name=prov)).choices.append(
            ModelChoice(label=tok, ref=AgentRef(model=tok))
        )
    if not any(g.choices for g in groups.values()):
        return [ModelGroup(name="pi", installed=False,
                           note="`pi --list-models` returned no parseable models; edit config manually")]
    return [groups[k] for k in sorted(groups)]


def _cli_groups(config: Config, timeout: float) -> list[ModelGroup]:
    statuses = asyncio.run(discover(config.agents, timeout=timeout))
    return groups_from_statuses(statuses)


def groups_from_statuses(statuses: list[AgentStatus]) -> list[ModelGroup]:
    """One group per cli agent; each model becomes a {agent, model} choice."""
    groups: list[ModelGroup] = []
    for st in sorted(statuses, key=lambda s: (not s.installed, s.name)):
        g = ModelGroup(name=st.name, installed=st.installed, note=st.note)
        for m in st.models:
            g.choices.append(ModelChoice(label=m, ref=AgentRef(agent=st.name, model=m)))
        groups.append(g)
    return groups


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_ref(ref: AgentRef | None) -> str:
    if ref is None:
        return "(unset)"
    if ref.agent and ref.model:
        return f"{ref.agent}:{ref.model}"
    return ref.model or ref.agent or "(unset)"


def _ref_yaml(ref: AgentRef | None) -> str:
    if ref is None or (not ref.agent and not ref.model):
        return "{ }"
    if ref.agent and ref.model:
        return f"{{ agent: {ref.agent}, model: {ref.model} }}"
    if ref.model:
        return f"{{ model: {ref.model} }}"
    return f"{{ agent: {ref.agent} }}"


def current_refs(config: Config) -> dict[str, AgentRef]:
    m = config.models
    return {"planner": m.planner, "main": m.main, "phase": m.phase, "task": m.task}


# --------------------------------------------------------------------------- #
# Interactive picker (injectable I/O for tests)
# --------------------------------------------------------------------------- #
_MODEL_PAGE = 25  # long provider lists prompt for a filter above this many


def pick_models(
    groups: list[ModelGroup],
    current: dict[str, AgentRef],
    *,
    roles: list[str] = ROLES,
    input_fn: Callable[[str], str] | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, AgentRef]:
    """Two-step provider->model prompt per role. Returns only the roles the user
    changed (empty dict if nothing changed / aborted)."""
    if input_fn is None:  # resolved at call time so a patched builtins.input is honored
        input_fn = input
    usable = [g for g in groups if g.installed and g.choices]
    if not usable:
        out("no connected models found (check the tool is installed and authenticated).")
        for g in groups:
            if g.note:
                out(f"  - {g.name}: {g.note}")
        return {}

    picks: dict[str, AgentRef] = {}
    out("select a model per role — Enter keeps the current value, 'q' stops.\n")
    try:
        for role in roles:
            out(f"── {role}  (current: {fmt_ref(current.get(role))})")
            g = _choose_group(usable, input_fn, out)
            if g is _QUIT:
                break
            if g is None:
                out("   kept.\n")
                continue
            ref = _choose_model(g, input_fn, out)
            if ref is _QUIT:
                break
            if ref is None:
                out("   kept.\n")
                continue
            picks[role] = ref
            out(f"   → {fmt_ref(ref)}\n")
    except (EOFError, KeyboardInterrupt):
        out("\naborted.")
        return {}
    return picks


_QUIT = object()  # sentinel: user asked to stop


def _choose_group(groups: list[ModelGroup], input_fn, out):
    for i, g in enumerate(groups, 1):
        out(f"   {i:>2}. {g.name}  ({len(g.choices)})")
    raw = input_fn("   provider # (Enter=keep, q=quit): ").strip().lower()
    if raw in ("q", "quit"):
        return _QUIT
    if raw == "":
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(groups):
        return groups[int(raw) - 1]
    out("   ? invalid choice; keeping current.")
    return None


def _choose_model(group: ModelGroup, input_fn, out):
    choices = group.choices
    if len(choices) > _MODEL_PAGE:
        flt = input_fn(f"   {len(choices)} models in {group.name}; type a filter (Enter=show all): ").strip().lower()
        if flt:
            choices = [c for c in choices if flt in c.label.lower()] or group.choices
    for i, c in enumerate(choices, 1):
        out(f"   {i:>2}. {c.label}")
    raw = input_fn("   model # (Enter=keep, q=quit): ").strip().lower()
    if raw in ("q", "quit"):
        return _QUIT
    if raw == "":
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return choices[int(raw) - 1].ref
    out("   ? invalid choice; keeping current.")
    return None


# --------------------------------------------------------------------------- #
# Config writing (surgical: replace only the models: block)
# --------------------------------------------------------------------------- #
def render_models_block(refs: dict[str, AgentRef], roles: list[str] = ROLES) -> str:
    lines = ["models:"]
    for r in roles:
        lines.append(f"  {(r + ':'):<9}{_ref_yaml(refs.get(r))}")
    return "\n".join(lines)


def update_config_models(path: Path, refs: dict[str, AgentRef], roles: list[str] = ROLES) -> None:
    """Replace the top-level ``models:`` block in the config, preserving the rest."""
    text = path.read_text()
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"^models:\s*(#.*)?$", l)), None)
    block = render_models_block(refs, roles).splitlines()
    if start is None:
        new = lines + ([""] if lines and lines[-1].strip() else []) + block
    else:
        end = start + 1
        while end < len(lines) and (lines[end].startswith((" ", "\t"))):
            end += 1
        new = lines[:start] + block + lines[end:]
    path.write_text("\n".join(new) + ("\n" if text.endswith("\n") else ""))


# --------------------------------------------------------------------------- #
# Optional verification: ping each picked model once (no tools) to catch 401/500
# --------------------------------------------------------------------------- #
def verify_refs(config: Config, refs: dict[str, AgentRef], *, out: Callable[[str], None] = print) -> bool:
    """Send a tiny no-tools prompt to each ref so runtime errors surface now, not
    mid-run. Returns True if all succeeded."""
    from .llm import CLIProvider, PiProvider

    ok_all = True
    out("verifying selections (one tiny call each) …")
    for role, ref in refs.items():
        try:
            if config.provider == "pi":
                prov = PiProvider(config.pi, timeout=60, max_retries=0)
                asyncio.run(prov.complete(model=ref.model, system="", user="Reply with: ok", role="planner"))
            else:
                prov = CLIProvider(config.agents, timeout=60, max_retries=0)
                asyncio.run(prov.complete(agent=ref.agent or None, model=ref.model,
                                          system="", user="Reply with: ok", role="planner"))
            out(f"   ✓ {role}  {fmt_ref(ref)}")
        except Exception as e:  # noqa: BLE001 - report any backend failure
            ok_all = False
            out(f"   ✗ {role}  {fmt_ref(ref)} — {str(e).splitlines()[0][:100]}")
    return ok_all
