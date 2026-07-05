"""CLIProvider: reach LLMs via terminal commands. Tested with POSIX utilities."""

import pytest

from roundtable.config import AgentSpec
from roundtable.errors import RoundtableError
from roundtable.llm import CLIProvider, _render_command


def test_render_command_combines_system_when_no_placeholder():
    spec = AgentSpec(command=["x", "{prompt}"])
    argv, stdin = _render_command(spec, "SYS", "USER")
    assert argv == ["x", "SYS\n\nUSER"]
    assert stdin is None


def test_render_command_keeps_system_separate_with_placeholder():
    spec = AgentSpec(command=["x", "-s", "{system}", "{prompt}"])
    argv, stdin = _render_command(spec, "SYS", "USER")
    assert argv == ["x", "-s", "SYS", "USER"]
    assert stdin is None


def test_render_command_stdin_mode():
    spec = AgentSpec(command=["x"], stdin=True)
    argv, stdin = _render_command(spec, "SYS", "USER")
    assert argv == ["x"]
    assert stdin == "SYS\n\nUSER"


def test_render_command_substitutes_model():
    spec = AgentSpec(command=["x", "--model", "{model}", "{prompt}"])
    argv, _ = _render_command(spec, "", "USER", "opus-4.8")
    assert argv == ["x", "--model", "opus-4.8", "USER"]
    # no model chosen -> placeholder collapses to empty string
    argv, _ = _render_command(spec, "", "USER", "")
    assert argv == ["x", "--model", "", "USER"]


async def test_cli_provider_passes_agent_and_model():
    # agent names the command; model fills {model}. printf echoes "<model>|<prompt>"
    provider = CLIProvider(
        {"opencode": AgentSpec(command=["printf", "%s|%s", "{model}", "{prompt}"])}
    )
    out = await provider.complete(
        agent="opencode", model="mimo-v2.5-pro", system="", user="hello"
    )
    assert out == "mimo-v2.5-pro|hello"


async def test_cli_provider_unknown_agent_key_reports_agent():
    provider = CLIProvider({"claude": AgentSpec(command=["true"])})
    with pytest.raises(RoundtableError, match="'opencode' is not defined"):
        await provider.complete(agent="opencode", model="x", system="", user="y")


async def test_cli_provider_arg_prompt():
    # printf "%s" <prompt> echoes the (combined) prompt to stdout
    provider = CLIProvider({"echo": AgentSpec(command=["printf", "%s", "{prompt}"])})
    out = await provider.complete(model="echo", system="SYS", user="USER")
    assert out == "SYS\n\nUSER"


async def test_cli_provider_stdin():
    provider = CLIProvider({"cat": AgentSpec(command=["cat"], stdin=True)})
    out = await provider.complete(model="cat", system="", user="piped text")
    assert out == "piped text"


async def test_cli_provider_runs_in_project_cwd(tmp_path):
    provider = CLIProvider({"pwd": AgentSpec(command=["pwd"])}, cwd=tmp_path)
    out = await provider.complete(model="pwd", system="", user="ignored")
    assert out.strip() == str(tmp_path.resolve())


async def test_cli_provider_unknown_agent():
    provider = CLIProvider({"a": AgentSpec(command=["true"])})
    with pytest.raises(RoundtableError, match="not defined"):
        await provider.complete(model="missing", system="", user="x")


async def test_cli_provider_command_not_found():
    provider = CLIProvider({"nope": AgentSpec(command=["definitely-not-a-real-binary-xyz"])})
    with pytest.raises(RoundtableError, match="not found on PATH"):
        await provider.complete(model="nope", system="", user="x")


async def test_cli_provider_nonzero_exit_raises():
    provider = CLIProvider({"fail": AgentSpec(command=["false"])}, max_retries=0)
    with pytest.raises(RuntimeError, match="exited"):
        await provider.complete(model="fail", system="", user="x")


async def test_cli_provider_surfaces_stdout_error_on_failure():
    # CLIs that print their error to stdout (not stderr) then exit non-zero
    # (e.g. claude's "model may not exist") must still show the reason.
    spec = AgentSpec(command=["sh", "-c", "echo bad-model-message; exit 1"])
    provider = CLIProvider({"agent": spec}, max_retries=0)
    with pytest.raises(RuntimeError, match="bad-model-message"):
        await provider.complete(agent="agent", model="x", system="", user="y")
