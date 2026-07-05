"""Discover which configured CLI agents are installed and which models they offer.

CLI detection is reliable (a PATH lookup). Model listing is best-effort: only
agents with a ``models_command`` can be enumerated, and those commands often hit
the network/auth and can be slow — so each runs with a bounded timeout and any
failure degrades to a note instead of raising. Used by ``roundtable init`` and
``roundtable agents`` to show what you can assign to roles.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field

from .config import AgentSpec


@dataclass
class AgentStatus:
    name: str
    binary: str
    installed: bool
    models: list[str] = field(default_factory=list)
    note: str = ""  # why models is empty (no command / timed out / error), if so


async def _list_models(spec: AgentSpec, timeout: float) -> tuple[list[str], str]:
    argv = list(spec.models_command or [])
    exe = shutil.which(argv[0])
    if exe is None:
        return [], f"{argv[0]!r} not found"
    argv[0] = exe
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:  # pragma: no cover - exec failure is environment-specific
        return [], str(e)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return [], f"timed out after {timeout:g}s"
    if proc.returncode != 0:
        tail = err.decode("utf-8", "replace").strip().splitlines()
        return [], f"exited {proc.returncode}" + (f": {tail[-1]}" if tail else "")
    lines = [ln.strip() for ln in out.decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines, "" if lines else "no models reported"


async def discover(agents: dict[str, AgentSpec], *, timeout: float = 15.0) -> list[AgentStatus]:
    """Probe every agent concurrently; total time ~= the slowest single probe."""

    async def one(name: str, spec: AgentSpec) -> AgentStatus:
        binary = spec.command[0] if spec.command else ""
        installed = shutil.which(binary) is not None if binary else False
        st = AgentStatus(name=name, binary=binary, installed=installed)
        if not installed:
            st.note = "not on PATH"
            return st
        if spec.models_command:
            st.models, st.note = await _list_models(spec, timeout)
        else:
            st.note = "no models_command"
        return st

    return list(await asyncio.gather(*(one(n, s) for n, s in agents.items())))
