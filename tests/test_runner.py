"""Runner tests with a fake SDK client emitting real SDK message dataclasses."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import types
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from software_factory import runner
from software_factory.config import Profile
from software_factory.policy import ToolPolicy
from software_factory.runner import (
    RunSummary,
    WorkerRun,
    build_options,
    json_schema_for,
    load_prompt,
    run_worker,
)
from software_factory.workers.schemas import OUTPUT_MODELS, CiOutput, RefineOutput, RepoPick

WORKER_ENV = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-faketoken", "GH_PROMPT_DISABLED": "1"}
VALID_CI = {
    "status": "done",
    "summary": "fixed the lint step",
    "push_ok": True,
    "head_verified": True,
    "verification": [{"command": "npm test", "exit_code": 0, "passed": True}],
    "failures": [{"check": "lint", "classification": "caused_by_pr", "cause": "unused import", "fixed": True}],
}


def _result(**overrides: Any) -> ResultMessage:
    base: dict[str, Any] = dict(
        subtype="success", duration_ms=1234, duration_api_ms=1000, is_error=False, num_turns=3,
        session_id="sess-1", total_cost_usd=0.042, usage={"input_tokens": 10, "output_tokens": 5},
        structured_output=VALID_CI, model_usage={"claude-opus-5-5": {"inputTokens": 10}},
    )
    base.update(overrides)
    return ResultMessage(**base)


class FakeClient:
    """Minimal stand-in for ClaudeSDKClient (connect/query/receive_response/interrupt/disconnect)."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        messages: list[Any],
        block_until_interrupt: bool = False,
        ignore_interrupt: bool = False,
        connect_error: Exception | None = None,
        pid: int | None = None,
        stderr_lines: list[str] | None = None,
    ):
        self.options = options
        self.messages = messages
        self.block_until_interrupt = block_until_interrupt
        self.ignore_interrupt = ignore_interrupt
        self.connect_error = connect_error
        self.stderr_lines = stderr_lines or []
        self.calls: list[Any] = []
        self._interrupted = asyncio.Event()
        if pid is not None:  # the shape the real SDK client exposes: client._transport._process.pid
            self._transport = types.SimpleNamespace(_process=types.SimpleNamespace(pid=pid))

    async def connect(self, prompt: Any = None) -> None:
        self.calls.append("connect")
        for line in self.stderr_lines:  # the SDK invokes options.stderr from its stderr reader
            if self.options.stderr is not None:
                self.options.stderr(line)
        if self.connect_error:
            raise self.connect_error

    async def query(self, prompt: Any, session_id: str = "default") -> None:
        self.calls.append(("query", prompt))

    async def receive_response(self) -> AsyncIterator[Any]:
        for msg in self.messages:
            yield msg
        if self.block_until_interrupt:
            if self.ignore_interrupt:
                await asyncio.sleep(3600)
            await self._interrupted.wait()
            yield _result(subtype="error_during_execution", is_error=True, structured_output=None,
                          errors=["Request was aborted."])

    async def interrupt(self) -> None:
        self.calls.append("interrupt")
        self._interrupted.set()

    async def disconnect(self) -> None:
        self.calls.append("disconnect")


class Factory:
    def __init__(self, **kw: Any):
        self.kw = kw
        self.client: FakeClient | None = None
        self.options: ClaudeAgentOptions | None = None

    def __call__(self, options: ClaudeAgentOptions) -> FakeClient:
        self.options = options
        self.client = FakeClient(options, **self.kw)
        return self.client


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake = types.ModuleType("software_factory.providers")
    fake.worker_env = lambda profile: dict(WORKER_ENV)  # type: ignore[attr-defined]
    fake.model_id = lambda profile, logical=None: "claude-opus-5-5-resolved"  # type: ignore[attr-defined]
    fake.fallback_model_id = lambda profile: "claude-sonnet-5-resolved"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "software_factory.providers", fake)
    return fake


@pytest.fixture
def profile() -> Profile:
    return Profile(name="personal")


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    wt = tmp_path / "wt"
    wt.mkdir()
    return wt


@pytest.fixture
def policy(worktree: Path, profile: Profile) -> ToolPolicy:
    return ToolPolicy(worktree, profile)


def _run(**kw: Any) -> WorkerRun:
    defaults: dict[str, Any] = dict(
        kind="ci", prompt="fix CI", schema=CiOutput, system_append="You are the ci worker.",
        timeout_s=5.0, tools_enabled=True, interrupt_grace_s=1.0,
    )
    defaults.update(kw)
    return asyncio.run(run_worker(**defaults))


def _served(run: WorkerRun) -> bool:
    """The session's structured output came back (validated, defaults filled in)."""
    return run.output is not None and run.output["summary"] == VALID_CI["summary"] and run.output["status"] == "done"


# ----------------------------------------------------------------------------------- options
def test_build_options_has_required_fields(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_should_not_leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    opts = build_options(
        cwd=worktree, schema=CiOutput, profile=profile, policy=policy, system_append="APPEND", kind="ci"
    )
    assert isinstance(opts, ClaudeAgentOptions)
    assert Path(opts.cwd) == worktree
    assert opts.allowed_tools == []
    assert opts.permission_mode == "default"
    assert opts.mcp_servers == {}
    assert opts.strict_mcp_config is True
    # SDK isolation mode: no filesystem settings at all (a checked-out .claude/settings.json can
    # define hooks / apiKeyHelper / env that run outside the tool policy); nothing extra either.
    assert opts.setting_sources == []
    assert opts.settings is None
    assert opts.system_prompt == {"type": "preset", "preset": "claude_code", "append": "APPEND"}
    assert opts.output_format["type"] == "json_schema"
    assert "$ref" not in json.dumps(opts.output_format["schema"])
    assert opts.can_use_tool == policy.can_use_tool
    assert opts.hooks is not None and list(opts.hooks) == ["PreToolUse"]
    matcher = opts.hooks["PreToolUse"][0]
    assert isinstance(matcher, HookMatcher) and matcher.matcher is None
    assert matcher.hooks == [policy.pre_tool_use_hook]
    assert matcher.timeout == 30.0  # seconds (SDK default 60): an explicit bound reaches the CLI
    assert opts.env == WORKER_ENV  # exactly worker_env: nothing inherited from os.environ
    assert "LINEAR_API_KEY" not in opts.env and "ANTHROPIC_API_KEY" not in opts.env
    assert opts.model == "claude-opus-5-5-resolved"
    assert opts.fallback_model == "claude-sonnet-5-resolved"
    assert opts.effort == profile.provider.effort
    assert opts.max_turns == profile.provider.max_turns
    assert opts.tools is None
    assert opts.include_partial_messages is False


def test_build_options_feeds_repo_instructions_without_loading_project_settings(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path
) -> None:
    """Regression: setting_sources=["project"] loaded the worktree's .claude/settings.json, whose
    hooks (e.g. one that starts a dev server in the user's main checkout) run
    outside can_use_tool / PreToolUse. CLAUDE.md / AGENTS.md now reach the model via the prompt."""
    (worktree / "CLAUDE.md").write_text("# repo rules\n\nRun `npm test`.\n", encoding="utf-8")
    (worktree / "AGENTS.md").write_text("Agents: be nice.\n", encoding="utf-8")
    hooks_dir = worktree / ".claude"
    hooks_dir.mkdir()
    (hooks_dir / "settings.json").write_text(
        json.dumps({"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "evil.sh"}]}]},
                    "permissions": {"allow": ["Bash(git push:*)"]}}),
        encoding="utf-8",
    )
    opts = build_options(cwd=worktree, schema=CiOutput, profile=profile, policy=policy, system_append="APPEND", kind="ci")
    assert opts.setting_sources == [] and opts.settings is None  # the hooks file is never read by the CLI
    append = opts.system_prompt["append"]
    assert append.startswith("APPEND\n\n## Repository instructions (CLAUDE.md)\n\n# repo rules")
    assert "Run `npm test`." in append
    assert "## Repository instructions (AGENTS.md)\n\nAgents: be nice." in append
    assert "evil.sh" not in append and "git push" not in append
    # the tool-less picker (scratch cwd) gets no repo instructions
    picker = build_options(cwd=worktree, schema=RepoPick, profile=profile, policy=None, system_append="P", tools_enabled=False)
    assert picker.system_prompt["append"] == "P"


def test_project_instructions_truncates_and_tolerates_missing_files(worktree: Path) -> None:
    assert runner.project_instructions(worktree) == ""
    (worktree / "CLAUDE.md").write_text("x" * 100, encoding="utf-8")
    text = runner.project_instructions(worktree, limit=40)
    assert text.startswith("## Repository instructions (CLAUDE.md)\n\n" + "x" * 40)
    assert "truncated after 40 characters" in text


def test_build_options_tools_disabled_for_picker(providers: types.ModuleType, profile: Profile, worktree: Path) -> None:
    opts = build_options(
        cwd=worktree, schema=RepoPick, profile=profile, policy=None, system_append="", tools_enabled=False
    )
    assert opts.tools == []
    assert opts.can_use_tool is None and opts.hooks is None
    assert opts.allowed_tools == [] and opts.permission_mode == "default"


def test_build_options_refuses_tools_without_policy(providers: types.ModuleType, profile: Profile, worktree: Path) -> None:
    with pytest.raises(ValueError, match="ToolPolicy"):
        build_options(cwd=worktree, schema=CiOutput, profile=profile, policy=None, system_append="")


def test_build_options_falls_back_to_profile_model_when_provider_lacks_helpers(
    monkeypatch: pytest.MonkeyPatch, profile: Profile, policy: ToolPolicy, worktree: Path
) -> None:
    fake = types.ModuleType("software_factory.providers")
    fake.worker_env = lambda p: {"ANTHROPIC_API_KEY": "k"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "software_factory.providers", fake)
    profile.provider.fallback_model = "claude-sonnet-5"
    opts = build_options(cwd=worktree, schema=CiOutput, profile=profile, policy=policy, system_append="")
    assert opts.model == profile.provider.model == "claude-opus-5-5"
    assert opts.fallback_model == "claude-sonnet-5"
    assert opts.env == {"ANTHROPIC_API_KEY": "k"}


# ------------------------------------------------------------------------------------ schema
@pytest.mark.parametrize("model", list(OUTPUT_MODELS.values()))
def test_json_schema_for_inlines_refs_for_every_worker_model(model: type) -> None:
    schema = json_schema_for(model)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in schema
    assert schema["type"] == "object"
    assert "title" not in schema  # keyword noise dropped


def test_json_schema_for_keeps_property_named_title_and_nested_structure() -> None:
    schema = json_schema_for(RefineOutput)
    assert "title" in schema["properties"]  # RefineOutput.title is a real field
    assert schema["properties"]["title"]["type"] == "string"
    ci = json_schema_for(CiOutput)
    failure = ci["properties"]["failures"]["items"]
    assert failure["type"] == "object" and "classification" in failure["properties"]
    assert "title" not in failure
    assert set(failure["properties"]["classification"]["enum"]) >= {"caused_by_pr", "flaky"}
    evidence = ci["properties"]["evidence"]
    assert evidence["type"] == "object" and "ac_proofs" in evidence["properties"]


# ------------------------------------------------------------------------------------ prompts
def test_load_prompt_combines_kind_and_common(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "PROMPTS_DIR", tmp_path)
    (tmp_path / "ci.md").write_text("# ci\nFix CI.\n", encoding="utf-8")
    (tmp_path / "_common.md").write_text("Common rules.\n", encoding="utf-8")
    assert load_prompt("ci") == "# ci\nFix CI.\n\nCommon rules."
    (tmp_path / "_common.md").unlink()
    assert load_prompt("ci") == "# ci\nFix CI."
    with pytest.raises(FileNotFoundError):
        load_prompt("nope")


# ------------------------------------------------------------------------------------ running
def test_run_worker_success_validates_structured_output(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, tmp_path: Path
) -> None:
    messages = [
        AssistantMessage(
            content=[TextBlock("Looking at the failure. token sk-ant-oat01-abcdefghijklmnop ok"),
                     ToolUseBlock(id="toolu_1", name="Bash", input={"command": "npm test"})],
            model="claude-opus-5-5",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="toolu_1", content="SECRET-FILE-CONTENT ok")]),
        _result(),
    ]
    factory = Factory(messages=messages)
    transcript = tmp_path / "logs" / "transcripts" / "ci-1.jsonl"
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, transcript_path=transcript)

    assert run.error is None and run.auth_failed is False and run.timed_out is False
    assert run.output == CiOutput(**VALID_CI).model_dump(mode="json")
    assert isinstance(run.summary, RunSummary)
    assert run.summary.cost_usd == 0.042 and run.summary.num_turns == 3 and run.summary.session_id == "sess-1"
    assert run.summary.usage == {"input_tokens": 10, "output_tokens": 5}
    assert run.summary.model == "claude-opus-5-5" and run.summary.subtype == "success"
    assert run.summary.permission_denials == 0 and run.summary.duration_ms == 1234
    assert factory.client is not None
    assert factory.client.calls == ["connect", ("query", "fix CI"), "disconnect"]
    assert factory.options is not None and factory.options.allowed_tools == []

    lines = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    types_ = [line["type"] for line in lines]
    assert types_ == ["start", "assistant", "user", "result", "end"]
    assistant = lines[1]
    assert assistant["tools"] == [{"name": "Bash", "id": "toolu_1", "command": "npm test"}]
    assert "[redacted]" in assistant["text"] and "abcdefghijklmnop" not in assistant["text"]
    text = transcript.read_text(encoding="utf-8")
    assert "SECRET-FILE-CONTENT" not in text  # tool results are summarised, never copied
    assert lines[2]["tool_results"] == [{"tool_use_id": "toolu_1", "is_error": False, "chars": 22}]
    assert lines[3]["has_structured_output"] is True
    assert lines[4]["timed_out"] is False


def test_run_worker_rejects_invalid_structured_output(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path
) -> None:
    bad = {"status": "done", "push_ok": True}  # summary + head_verified missing
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=[_result(structured_output=bad)]))
    assert run.output is None
    assert run.error is not None and "validation" in run.error and "CiOutput" in run.error
    assert run.summary is not None and run.summary.is_error is False
    assert run.auth_failed is False and run.timed_out is False


def test_run_worker_missing_structured_output_is_an_error(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path
) -> None:
    run = _run(cwd=worktree, profile=profile, policy=policy,
               client_factory=Factory(messages=[_result(structured_output=None)]))
    assert run.output is None and run.error and "structured output" in run.error


def test_run_worker_error_result(providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path) -> None:
    msg = _result(subtype="error_max_turns", is_error=True, structured_output=None, errors=["max turns"],
                  permission_denials=[{"tool_name": "Bash"}, {"tool_name": "Bash"}])
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=[msg]))
    assert run.output is None
    assert run.error == "max turns"
    assert run.summary is not None and run.summary.is_error and run.summary.subtype == "error_max_turns"
    assert run.summary.permission_denials == 2
    assert run.auth_failed is False


@pytest.mark.parametrize(
    "messages",
    [
        [AssistantMessage(content=[TextBlock("")], model="m", error="authentication_failed"),
         _result(subtype="error_during_execution", is_error=True, structured_output=None)],
        [_result(subtype="error_during_execution", is_error=True, structured_output=None, api_error_status=401)],
        [_result(subtype="error_during_execution", is_error=True, structured_output=None, api_error_status=403)],
        [_result(subtype="error_during_execution", is_error=True, structured_output=None,
                 result="Invalid API key · Please run /login")],
    ],
    ids=["assistant-error", "status-401", "status-403", "result-text"],
)
def test_run_worker_detects_auth_failure(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, messages: list[Any]
) -> None:
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=messages))
    assert run.auth_failed is True
    assert run.rate_limited is False
    assert run.output is None and run.error


def test_run_worker_rate_limit_is_not_auth_failure(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path
) -> None:
    messages = [AssistantMessage(content=[], model="m", error="rate_limit"),
                _result(subtype="error_during_execution", is_error=True, structured_output=None, api_error_status=429)]
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=messages))
    assert run.auth_failed is False and run.error
    assert run.rate_limited is True


@pytest.mark.parametrize(
    "messages",
    [
        [_result(subtype="success", is_error=True, structured_output=None, api_error_status=429,
                 result="API Error: 429 rate limit")],
        [_result(subtype="success", is_error=True, structured_output=None, api_error_status=529)],
        [AssistantMessage(content=[], model="m", error="server_error"),
         _result(subtype="error_during_execution", is_error=True, structured_output=None)],
        [AssistantMessage(content=[], model="m", error="billing_error"),
         _result(subtype="error_during_execution", is_error=True, structured_output=None)],
        [_result(subtype="error_during_execution", is_error=True, structured_output=None,
                 result="You've hit your usage limit. Try again at 3pm.")],
        [_result(subtype="error_during_execution", is_error=True, structured_output=None,
                 errors=["API Error: 529 Overloaded"])],
    ],
    ids=["status-429", "status-529", "assistant-server-error", "assistant-billing-error", "usage-limit-text", "overloaded-text"],
)
def test_run_worker_marks_provider_unavailability_as_rate_limited(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, messages: list[Any]
) -> None:
    """Regression: a usage-limit / 429 / 529 session looked like any failed worker, so three ticks
    inside one usage window burned max_attempts for every open target. The dispatcher can now treat
    ``rate_limited`` like an auth stop (no attempt, stop launching)."""
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=messages))
    assert run.rate_limited is True
    assert run.auth_failed is False
    assert run.output is None and run.error


def test_run_worker_ordinary_failures_are_not_rate_limited(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path
) -> None:
    msg = _result(subtype="error_max_turns", is_error=True, structured_output=None, errors=["max turns"])
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=[msg]))
    assert run.rate_limited is False
    ok = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=[_result()]))
    assert ok.rate_limited is False and ok.output is not None


def test_run_worker_timeout_interrupts_then_disconnects(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, tmp_path: Path
) -> None:
    factory = Factory(messages=[AssistantMessage(content=[TextBlock("working...")], model="m")],
                      block_until_interrupt=True)
    transcript = tmp_path / "t.jsonl"
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, timeout_s=0.05,
               interrupt_grace_s=2.0, transcript_path=transcript)
    assert run.timed_out is True
    assert run.output is None and run.error
    assert factory.client is not None
    calls = factory.client.calls
    assert calls.index("interrupt") < calls.index("disconnect")
    assert calls[-1] == "disconnect"
    assert run.summary is not None and run.summary.subtype == "error_during_execution"
    types_ = [json.loads(line)["type"] for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert "timeout" in types_ and types_[-1] == "end"


def test_run_worker_hard_timeout_when_interrupt_is_ignored(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path
) -> None:
    factory = Factory(messages=[], block_until_interrupt=True, ignore_interrupt=True)
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, timeout_s=0.05,
               interrupt_grace_s=0.1)
    assert run.timed_out is True
    assert run.error and "timed out" in run.error
    assert run.summary is None and run.output is None
    assert factory.client is not None and factory.client.calls[-1] == "disconnect"
    assert "interrupt" in factory.client.calls


def test_run_worker_connect_failure(providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path) -> None:
    factory = Factory(messages=[], connect_error=RuntimeError("Claude Code not found at /nowhere"))
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory)
    assert run.error == "RuntimeError: Claude Code not found at /nowhere"
    assert run.timed_out is False and run.auth_failed is False and run.summary is None
    assert factory.client is not None and "disconnect" not in factory.client.calls


def test_run_worker_records_cli_stderr_and_appends_it_to_startup_errors(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, tmp_path: Path
) -> None:
    """Regression: stderr was never piped, so a CLI that died during initialize (old CLI, bad flag,
    rejected token) was recorded as an opaque 'Command failed with exit code 1'."""
    factory = Factory(
        messages=[], connect_error=RuntimeError("ProcessError: Command failed with exit code 1"),
        stderr_lines=["error: unknown option '--json-schema'\n", "token sk-ant-oat01-abcdefghijklmnop rejected\n", "   \n"],
    )
    transcript = tmp_path / "t.jsonl"
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, transcript_path=transcript)
    assert factory.options is not None and callable(factory.options.stderr)
    assert run.error is not None and run.error.startswith("RuntimeError: ProcessError: Command failed with exit code 1")
    assert "; stderr: error: unknown option '--json-schema' | token [redacted] rejected" in run.error
    assert "abcdefghijklmnop" not in run.error
    lines = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    stderr_records = [line for line in lines if line["type"] == "stderr"]
    assert [r["line"] for r in stderr_records] == ["error: unknown option '--json-schema'", "token [redacted] rejected"]
    # a run that produced a ResultMessage keeps its own error text (stderr stays in the transcript)
    ok = _run(cwd=worktree, profile=profile, policy=policy,
              client_factory=Factory(messages=[_result()], stderr_lines=["warning: something\n"]))
    assert ok.output is not None and ok.error is None


def test_run_worker_kills_the_cli_process_tree_on_timeout_before_disconnect(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path
) -> None:
    """Regression: disconnect only terminates claude.exe; a dev server the worker started kept the
    worktree and its port busy. The tree is killed from the still-alive CLI pid, then disconnected."""
    factory = Factory(messages=[], block_until_interrupt=True, ignore_interrupt=True, pid=4242)
    killed: list[int] = []

    def kill_tree(pid: int) -> None:
        killed.append(pid)
        assert factory.client is not None
        factory.client.calls.append("kill_tree")

    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, timeout_s=0.05,
               interrupt_grace_s=0.1, kill_tree=kill_tree)
    assert run.timed_out is True and killed == [4242]
    calls = factory.client.calls  # type: ignore[union-attr]
    assert calls.index("kill_tree") < calls.index("disconnect")
    # the soft path (interrupt honoured, result delivered) also kills what the worker left behind
    soft = Factory(messages=[], block_until_interrupt=True, pid=77)
    killed.clear()
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=soft, timeout_s=0.05,
               interrupt_grace_s=2.0, kill_tree=kill_tree)
    assert run.timed_out is True and killed == [77]


def test_run_worker_kills_the_tree_at_the_end_of_a_normal_run_before_disconnect(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, tmp_path: Path
) -> None:
    """Regression: only timeout / error sessions had their process tree killed, so a background
    loop or dev server a worker started in a *successful* session outlived it (still running with
    the worker's env after the worktree was handed back). Every session ends with the kill, before
    disconnect (taskkill /T needs the CLI alive to walk its tree); the output is unaffected."""
    factory = Factory(messages=[_result()], pid=4242)
    killed: list[int] = []

    def kill_tree(pid: int) -> None:
        killed.append(pid)
        assert factory.client is not None
        factory.client.calls.append("kill_tree")

    transcript = tmp_path / "t.jsonl"
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, kill_tree=kill_tree,
               transcript_path=transcript)
    assert _served(run) and run.error is None and killed == [4242]
    calls = factory.client.calls  # type: ignore[union-attr]
    assert calls.index("kill_tree") < calls.index("disconnect") and calls[-1] == "disconnect"
    records = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert {"type": "kill_tree", "pid": 4242}.items() <= next(r for r in records if r["type"] == "kill_tree").items()

    # a failed (non-timeout) session and an invalid-output session are cleaned up the same way
    killed.clear()
    bad = _result(subtype="error_max_turns", is_error=True, structured_output=None, errors=["max turns"])
    factory = Factory(messages=[bad], pid=5)
    _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, kill_tree=kill_tree)
    factory = Factory(messages=[_result(structured_output={"status": "nonsense"})], pid=6)
    _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, kill_tree=kill_tree)
    assert killed == [5, 6]

    # fakes without a transport (no pid) never trigger a kill, even on timeout
    killed.clear()
    run = _run(cwd=worktree, profile=profile, policy=policy,
               client_factory=Factory(messages=[], block_until_interrupt=True, ignore_interrupt=True),
               timeout_s=0.05, interrupt_grace_s=0.1, kill_tree=killed.append)
    assert run.timed_out is True and killed == []
    # nor does a CLI that has already exited (its pid may have been reaped and reused)
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=_exited_factory(4244), kill_tree=killed.append)
    assert _served(run) and killed == []


def _exited_factory(pid: int) -> Callable[[ClaudeAgentOptions], FakeClient]:
    """A client whose CLI process has already exited (``returncode`` set) when the session ends."""
    def make(options: ClaudeAgentOptions) -> FakeClient:
        client = FakeClient(options, messages=[_result()], pid=pid)
        client._transport._process.returncode = 1  # type: ignore[attr-defined]
        return client

    return make


def test_run_worker_end_of_session_kill_is_best_effort_and_bounded(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill that raises or hangs neither loses the run's output nor blocks the disconnect."""
    def boom(pid: int) -> None:
        raise OSError("access denied")

    factory = Factory(messages=[_result()], pid=11)
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, kill_tree=boom)
    assert _served(run) and factory.client is not None and factory.client.calls[-1] == "disconnect"

    monkeypatch.setattr(runner, "KILL_TREE_TIMEOUT_S", 0.05)
    factory = Factory(messages=[_result()], pid=12)

    def stuck(pid: int) -> None:
        time.sleep(0.5)
        assert factory.client is not None
        factory.client.calls.append("kill_returned")

    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=factory, kill_tree=stuck)
    assert _served(run) and factory.client is not None
    calls = factory.client.calls
    assert calls.index("disconnect") < calls.index("kill_returned")  # the disconnect did not wait for it


def test_the_sdk_reader_error_for_the_cli_we_killed_is_not_logged(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Killing the CLI makes the SDK's stdout reader log "Fatal error in message reader: Command
    failed with exit code 1" -- after every session now. That record is the engine's own doing and is
    dropped while the kill / disconnect runs; the same record outside that window is kept."""
    sdk_logger = logging.getLogger(runner.SDK_QUERY_LOGGER)

    def kill_tree(pid: int) -> None:
        sdk_logger.error("Fatal error in message reader: Command failed with exit code 1")

    caplog.set_level(logging.ERROR, logger=runner.SDK_QUERY_LOGGER)
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=[_result()], pid=9),
               kill_tree=kill_tree)
    assert _served(run)
    assert not [r for r in caplog.records if "message reader" in r.getMessage()]
    sdk_logger.error("Fatal error in message reader: Command failed with exit code 1")
    assert [r for r in caplog.records if "message reader" in r.getMessage()]


def test_run_worker_ignores_a_subagent_rate_limit_when_the_session_succeeds(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path
) -> None:
    """Regression: a subagent's rate-limited message (``parent_tool_use_id`` set) marked a session
    that went on to finish with valid structured output as ``rate_limited``, so the handler threw the
    output away and the dispatcher stopped the wave as if the provider were down."""
    messages = [
        AssistantMessage(content=[], model="m", error="rate_limit", parent_tool_use_id="toolu_task_1"),
        AssistantMessage(content=[TextBlock("done")], model="m"),
        _result(),
    ]
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=messages))
    assert run.rate_limited is False and _served(run) and run.error is None

    # a top-level rate limit the CLI recovered from (successful result, valid output) is no outage either
    messages = [AssistantMessage(content=[], model="m", error="rate_limit"), _result()]
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=messages))
    assert run.rate_limited is False and _served(run)

    # a subagent's rate limit does not turn an ordinary failure into a provider outage
    messages = [
        AssistantMessage(content=[], model="m", error="rate_limit", parent_tool_use_id="toolu_task_1"),
        _result(subtype="error_max_turns", is_error=True, structured_output=None, errors=["max turns"]),
    ]
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=messages))
    assert run.rate_limited is False and run.output is None and run.error

    # the top-level signal still counts when the session did not produce output
    messages = [
        AssistantMessage(content=[], model="m", error="rate_limit"),
        _result(subtype="error_during_execution", is_error=True, structured_output=None),
    ]
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=messages))
    assert run.rate_limited is True and run.output is None


def test_run_worker_without_transcript_path_writes_nothing(
    providers: types.ModuleType, profile: Profile, policy: ToolPolicy, worktree: Path, tmp_path: Path
) -> None:
    before = set(os.listdir(tmp_path))
    run = _run(cwd=worktree, profile=profile, policy=policy, client_factory=Factory(messages=[_result()]))
    assert run.output is not None
    assert set(os.listdir(tmp_path)) == before


def test_run_worker_picker_mode_has_no_tools(providers: types.ModuleType, profile: Profile, worktree: Path) -> None:
    pick = {"repo": "o/r", "confidence": 0.9, "multi_repo": False, "reasoning": "name match"}
    factory = Factory(messages=[_result(structured_output=pick)])
    run = _run(kind="resolve", cwd=worktree, profile=profile, policy=None, schema=RepoPick,
               client_factory=factory, tools_enabled=False)
    assert run.output == pick
    assert factory.options is not None and factory.options.tools == [] and factory.options.can_use_tool is None
