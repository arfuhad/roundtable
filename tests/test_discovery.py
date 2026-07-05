"""Agent discovery: detect installed CLIs + best-effort model listing.

Uses POSIX utilities (printf/true) so the probes are deterministic and offline.
"""

from roundtable.config import AgentSpec
from roundtable.discovery import discover


def _by_name(statuses):
    return {s.name: s for s in statuses}


async def test_discover_lists_models_marks_missing_and_no_command():
    agents = {
        # installed + a models_command that prints two models
        "lister": AgentSpec(command=["printf", "%s"], models_command=["printf", "m1\nm2\n"]),
        # installed but no models_command -> n/a
        "plain": AgentSpec(command=["true"]),
        # binary not on PATH -> not installed
        "missing": AgentSpec(command=["definitely-not-a-real-binary-xyz"]),
    }
    st = _by_name(await discover(agents, timeout=5))

    assert st["lister"].installed and st["lister"].models == ["m1", "m2"]
    assert st["plain"].installed and st["plain"].models == [] and st["plain"].note == "no models_command"
    assert not st["missing"].installed and st["missing"].note == "not on PATH"


async def test_discover_models_command_timeout_degrades_gracefully():
    # `sleep 5` will exceed the 0.3s timeout; discovery must degrade, not raise.
    agents = {"slow": AgentSpec(command=["true"], models_command=["sleep", "5"])}
    st = _by_name(await discover(agents, timeout=0.3))
    assert st["slow"].installed
    assert st["slow"].models == []
    assert "timed out" in st["slow"].note


async def test_discover_nonzero_models_command_reports_note():
    agents = {"bad": AgentSpec(command=["true"], models_command=["false"])}
    st = _by_name(await discover(agents, timeout=5))
    assert st["bad"].models == []
    assert "exited" in st["bad"].note
