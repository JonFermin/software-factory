"""Claude Agent SDK wrapper: one narrow worker session with structured output.

The runner builds ``ClaudeAgentOptions`` so that *every* non-auto-approved tool call reaches
``ToolPolicy.can_use_tool`` (``allowed_tools=[]``, ``permission_mode="default"``) and is additionally
gated by the ``PreToolUse`` hook. Filesystem settings are not loaded at all
(``setting_sources=[]``): a checked-out repo's ``.claude/settings.json`` can define ``hooks``,
``apiKeyHelper``, ``env`` and ``enabledPlugins``, all of which run commands outside the tool policy
(a PR head from a fork controls that file). The repo's ``CLAUDE.md`` / ``AGENTS.md``, which that
setting source would otherwise load, are read by the engine and appended to the system prompt
instead. Secrets reach the CLI subprocess only through ``env`` (from ``providers.worker_env``),
never through ``os.environ``.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import importlib
import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from pydantic import BaseModel, ValidationError

from .config import Profile
from .policy import ToolPolicy

PROMPTS_DIR = Path(__file__).resolve().parent / "workers"
COMMON_PROMPT = "_common.md"
# Statuses the CLI reports for a run that did not finish normally.
AUTH_STATUSES = (401, 403)
# Provider-side unavailability: the session could not run at all, through no fault of the target.
RATE_LIMIT_STATUSES = (429, 529)
RATE_LIMIT_ERRORS = ("rate_limit", "server_error", "billing_error")
_AUTH_TEXT_RE = re.compile(
    r"(invalid api key|not logged in|please run /login|authentication[_ ]failed|oauth token (has )?expired|"
    r"invalid authentication|401 unauthorized|403 forbidden)",
    re.IGNORECASE,
)
_RATE_LIMIT_TEXT_RE = re.compile(
    r"(rate.?limit|usage limit|hit your limit|limit reached|overloaded|too many requests|\b429\b|\b529\b)",
    re.IGNORECASE,
)
# Repo instruction files the engine feeds into the system prompt (setting_sources=[] loads none).
PROJECT_INSTRUCTION_FILES = ("CLAUDE.md", "AGENTS.md")
PROJECT_INSTRUCTIONS_LIMIT = 40_000
STDERR_TAIL_LINES = 20
# Upper bound for the end-of-session process-tree kill (taskkill has its own 30 s timeout).
KILL_TREE_TIMEOUT_S = 30.0
# The SDK logger that reports the CLI's non-zero exit once ``run_worker`` has killed it on purpose.
SDK_QUERY_LOGGER = "claude_agent_sdk._internal.query"
_SECRET_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{8,}|gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,}|"
    r"xox[abpr]-[A-Za-z0-9\-]{8,}|lin_api_[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._\-]{12,})"
)


class RunSummary(BaseModel):
    cost_usd: float | None = None
    usage: dict | None = None
    num_turns: int = 0
    duration_ms: int = 0
    session_id: str = ""
    is_error: bool = False
    subtype: str = ""
    permission_denials: int = 0
    model: str | None = None


class WorkerRun(BaseModel):
    output: dict | None = None
    summary: RunSummary | None = None
    error: str | None = None
    auth_failed: bool = False
    timed_out: bool = False
    # The provider refused or could not serve the session (429/529, rate_limit / server_error /
    # billing_error, usage-limit text): an infrastructure failure, not a verdict on the target, so
    # the dispatcher should stop launching and spend no attempt (invariant 5 semantics).
    rate_limited: bool = False


# ------------------------------------------------------------------------------- schema / prompts
def json_schema_for(model: type[BaseModel]) -> dict:
    """``model_json_schema()`` with every ``$ref`` inlined and ``title`` noise dropped."""
    raw = model.model_json_schema()
    defs: dict[str, Any] = raw.pop("$defs", {})

    def inline(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [inline(item, stack) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref[len("#/$defs/") :]
            if name in stack:
                raise ValueError(f"recursive schema reference {name!r} cannot be inlined")
            target = inline(copy.deepcopy(defs[name]), (*stack, name))
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return {**target, **inline(extra, stack)} if extra else target
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title" and isinstance(value, str):
                continue
            if key in ("properties", "$defs", "patternProperties"):
                out[key] = {k: inline(v, stack) for k, v in value.items()}
            else:
                out[key] = inline(value, stack)
        return out

    return inline(raw, ())


def load_prompt(kind: str) -> str:
    """``workers/<kind>.md`` followed by ``workers/_common.md`` (the latter optional)."""
    kind_path = PROMPTS_DIR / f"{kind}.md"
    if not kind_path.is_file():
        raise FileNotFoundError(f"no worker prompt for kind {kind!r}: {kind_path}")
    parts = [kind_path.read_text(encoding="utf-8").strip()]
    common = PROMPTS_DIR / COMMON_PROMPT
    if common.is_file():
        parts.append(common.read_text(encoding="utf-8").strip())
    return "\n\n".join(p for p in parts if p)


def project_instructions(cwd: Path, *, limit: int = PROJECT_INSTRUCTIONS_LIMIT) -> str:
    """The repo's ``CLAUDE.md`` / ``AGENTS.md`` (root of ``cwd``) as system-prompt text.

    This replaces what ``setting_sources=["project"]`` used to load, without the ``hooks`` /
    ``apiKeyHelper`` / ``env`` / ``enabledPlugins`` keys of ``.claude/settings.json`` that would run
    repo-controlled commands outside the tool policy. Empty when neither file exists.
    """
    blocks: list[str] = []
    for name in PROJECT_INSTRUCTION_FILES:
        path = Path(cwd) / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        if len(text) > limit:
            text = text[:limit].rstrip() + f"\n\n[... {name} truncated after {limit} characters]"
        blocks.append(f"## Repository instructions ({name})\n\n{text}")
    return "\n\n".join(blocks)


# ------------------------------------------------------------------------------------- options
def _providers() -> Any:
    """``providers.py`` is imported lazily (tests inject a fake via ``sys.modules``)."""
    return importlib.import_module("software_factory.providers")


def build_options(
    *,
    cwd: Path,
    schema: type[BaseModel],
    profile: Profile,
    policy: ToolPolicy | None,
    system_append: str,
    tools_enabled: bool = True,
    kind: str = "",
    stderr: Callable[[str], None] | None = None,
) -> ClaudeAgentOptions:
    if tools_enabled and policy is None:
        raise ValueError("a ToolPolicy is required when tools are enabled")
    providers = _providers()
    env: dict[str, str] = dict(providers.worker_env(profile))
    model_id = getattr(providers, "model_id", None)
    fallback_id = getattr(providers, "fallback_model_id", None)
    model = model_id(profile) if callable(model_id) else profile.provider.model
    fallback = fallback_id(profile) if callable(fallback_id) else profile.provider.fallback_model
    cli_path = profile.provider.cli_path
    instructions = project_instructions(Path(cwd)) if tools_enabled else ""
    append = "\n\n".join(part for part in (system_append.strip(), instructions) if part)
    return ClaudeAgentOptions(
        cwd=Path(cwd),
        cli_path=Path(cli_path).expanduser() if cli_path else None,
        model=model,
        fallback_model=fallback,
        effort=profile.provider.effort,
        max_turns=profile.provider.max_turns,
        # No filesystem settings at all (SDK isolation mode): the worktree's .claude/settings.json is
        # repo-controlled content and its hooks/apiKeyHelper/env/enabledPlugins would run outside the
        # policy. CLAUDE.md / AGENTS.md go in through ``append`` instead. SDK hooks (``hooks=``) are
        # sent in the initialize request and are unaffected.
        setting_sources=[],
        system_prompt={"type": "preset", "preset": "claude_code", "append": append},
        output_format={"type": "json_schema", "schema": json_schema_for(schema)},
        can_use_tool=policy.can_use_tool if policy else None,
        hooks=policy.hooks() if policy else None,
        env=env,
        permission_mode="default",
        mcp_servers={},
        strict_mcp_config=True,
        allowed_tools=[],
        tools=None if tools_enabled else [],
        include_partial_messages=False,
        # Pipe the CLI's stderr (otherwise it inherits ours and a startup crash is an opaque
        # "Command failed with exit code 1" in the transcript / flag).
        stderr=stderr,
        # Note: ClaudeAgentOptions.user is the OS account the CLI subprocess runs as (unsupported on
        # Windows), not a metadata label -- never set it.
    )


# ---------------------------------------------------------------------------------- transcript
def _redact(text: str) -> str:
    return _SECRET_RE.sub("[redacted]", text)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def _tool_summary(block: ToolUseBlock) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": block.name, "id": block.id}
    inp = block.input or {}
    for key in ("command", "file_path", "notebook_path", "path", "pattern", "url", "description"):
        value = inp.get(key)
        if isinstance(value, str):
            entry[key] = _clip(_redact(value), 500)
    return entry


class _Transcript:
    """Compact JSONL transcript: message types, text, tool names/paths/commands. No file contents."""

    def __init__(self, path: Path | None):
        self.path = path
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            return
        record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **record}
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def record(self, msg: Any) -> None:
        if self._fh is None:
            return
        if isinstance(msg, AssistantMessage):
            texts = [_clip(_redact(b.text), 2000) for b in msg.content if isinstance(b, TextBlock)]
            tools = [_tool_summary(b) for b in msg.content if isinstance(b, ToolUseBlock)]
            rec: dict[str, Any] = {"type": "assistant", "model": msg.model}
            if msg.error:
                rec["error"] = msg.error
            if texts:
                rec["text"] = "\n".join(texts)
            if tools:
                rec["tools"] = tools
            self.write(rec)
        elif isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                self.write({"type": "user", "chars": len(msg.content)})
                return
            results = [
                {"tool_use_id": b.tool_use_id, "is_error": bool(b.is_error), "chars": len(str(b.content or ""))}
                for b in msg.content
                if isinstance(b, ToolResultBlock)
            ]
            self.write({"type": "user", "tool_results": results})
        elif isinstance(msg, ResultMessage):
            self.write(
                {
                    "type": "result",
                    "subtype": msg.subtype,
                    "is_error": msg.is_error,
                    "num_turns": msg.num_turns,
                    "duration_ms": msg.duration_ms,
                    "cost_usd": msg.total_cost_usd,
                    "has_structured_output": msg.structured_output is not None,
                    "errors": msg.errors,
                    "api_error_status": msg.api_error_status,
                }
            )
        elif isinstance(msg, SystemMessage):
            self.write({"type": "system", "subtype": msg.subtype})
        else:
            self.write({"type": type(msg).__name__})

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# -------------------------------------------------------------------------------------- run
def _summary_from(result: ResultMessage, model: str | None) -> RunSummary:
    if model is None and result.model_usage:
        model = next(iter(result.model_usage), None)
    return RunSummary(
        cost_usd=result.total_cost_usd,
        usage=result.usage,
        num_turns=result.num_turns,
        duration_ms=result.duration_ms,
        session_id=result.session_id,
        is_error=result.is_error,
        subtype=result.subtype,
        permission_denials=len(result.permission_denials or []),
        model=model,
    )


def _result_error_text(result: ResultMessage) -> str:
    parts = [e for e in (result.errors or []) if e]
    if result.result:
        parts.append(result.result)
    text = "; ".join(parts) or f"worker ended with subtype {result.subtype!r}"
    return _clip(_redact(text), 2000)


def _cli_pid(client: Any) -> int | None:
    """PID of the live CLI subprocess behind a ``ClaudeSDKClient`` (``None`` for fakes, before
    connect, and once the CLI has exited: its tree can no longer be walked from it, and on POSIX a
    reaped pid may already belong to an unrelated process)."""
    transport = getattr(client, "_transport", None)
    process = getattr(transport, "_process", None)
    pid = getattr(process, "pid", None)
    if getattr(process, "returncode", None) is not None:
        return None
    return pid if isinstance(pid, int) and pid > 0 else None


def _default_kill_tree(pid: int) -> None:
    from .readiness import kill_tree  # lazy: readiness pulls in httpx/discover

    kill_tree(pid)


class _ExpectedCliExitFilter(logging.Filter):
    """Drops the SDK reader's "Fatal error in message reader: Command failed with exit code N" while
    at least one ``run_worker`` is between killing a CLI it is done with and disconnecting it: that
    exit is the engine's own doing, and the record would otherwise reach stderr after every session.
    Outside such a window nothing is filtered (a real crash still surfaces through the message
    stream either way)."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._active = 0

    @contextlib.contextmanager
    def expecting(self) -> Any:
        with self._lock:
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1

    def filter(self, record: logging.LogRecord) -> bool:
        if self._active <= 0:
            return True
        message = record.getMessage()
        return not (message.startswith("Fatal error in message reader") and "exit code" in message)


_EXPECTED_CLI_EXIT = _ExpectedCliExitFilter()
logging.getLogger(SDK_QUERY_LOGGER).addFilter(_EXPECTED_CLI_EXIT)


async def run_worker(
    *,
    kind: str,
    prompt: str,
    cwd: Path,
    schema: type[BaseModel],
    profile: Profile,
    policy: ToolPolicy | None,
    system_append: str,
    timeout_s: float,
    transcript_path: Path | None = None,
    tools_enabled: bool = True,
    client_factory: Callable[[ClaudeAgentOptions], Any] | None = None,
    interrupt_grace_s: float = 60.0,
    kill_tree: Callable[[int], None] | None = None,
) -> WorkerRun:
    """Run one worker session to completion (or timeout) and validate its structured output.

    Timeout: after ``timeout_s`` the session is interrupted (``client.interrupt()``) so the CLI ends
    the turn with a ``ResultMessage``; if nothing arrives within ``interrupt_grace_s`` the whole
    session is cancelled (``asyncio.timeout``). Either way the client is disconnected and
    ``timed_out=True``. At the end of *every* session (normal, error or timeout) the CLI's whole
    process tree is killed before the disconnect (``kill_tree``, default ``readiness.kill_tree``,
    best effort and bounded by ``KILL_TREE_TIMEOUT_S``): disconnect only terminates ``claude``
    itself, and a background loop, dev server or test watcher the worker started would otherwise
    outlive it -- keeping the worktree and its port busy, or running on with the worker's env.

    ``rate_limited`` counts only top-level assistant messages (a subagent's carry a
    ``parent_tool_use_id``) and is never set for a session that ended in a successful result with
    valid structured output (the provider did serve it).
    """
    stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
    transcript = _Transcript(transcript_path)

    def on_stderr(line: str) -> None:
        text = _clip(_redact(line.rstrip("\r\n")), 500)
        if not text.strip():
            return
        stderr_tail.append(text)
        transcript.write({"type": "stderr", "line": text})

    try:
        options = build_options(
            cwd=cwd, schema=schema, profile=profile, policy=policy, system_append=system_append,
            tools_enabled=tools_enabled, kind=kind, stderr=on_stderr,
        )
    except BaseException:
        transcript.close()
        raise
    client = (client_factory or ClaudeSDKClient)(options)
    transcript.write({"type": "start", "kind": kind, "cwd": str(cwd), "model": options.model,
                      "timeout_s": timeout_s})
    started = time.monotonic()
    result: ResultMessage | None = None
    last_model: str | None = None
    auth_failed = False
    rate_limited = False
    timed_out = False
    error: str | None = None
    connected = False
    kill = kill_tree or _default_kill_tree

    async def watchdog() -> None:
        nonlocal timed_out
        await asyncio.sleep(timeout_s)
        timed_out = True
        transcript.write({"type": "timeout", "after_s": round(time.monotonic() - started, 1)})
        with contextlib.suppress(Exception):
            await client.interrupt()

    dog: asyncio.Task[None] | None = None
    try:
        async with asyncio.timeout(timeout_s + max(interrupt_grace_s, 0.0)):
            await client.connect()
            connected = True
            dog = asyncio.create_task(watchdog())
            await client.query(prompt)
            async for msg in client.receive_response():
                transcript.record(msg)
                if isinstance(msg, AssistantMessage):
                    last_model = msg.model or last_model
                    top_level = getattr(msg, "parent_tool_use_id", None) is None
                    if msg.error == "authentication_failed":
                        auth_failed = True
                    elif msg.error in RATE_LIMIT_ERRORS and top_level:
                        # a subagent's rate limit is retried / absorbed by the main session; only the
                        # top-level conversation tells whether the provider could serve the session
                        rate_limited = True
                elif isinstance(msg, ResultMessage):
                    result = msg
                    break
    except TimeoutError:
        timed_out = True
        error = f"worker timed out after {timeout_s:.0f}s and did not stop when interrupted"
    except Exception as exc:  # SDK/process errors: CLINotFoundError, ProcessError, ...
        error = _clip(_redact(f"{type(exc).__name__}: {exc}"), 2000)
        if _AUTH_TEXT_RE.search(str(exc)):
            auth_failed = True
    finally:
        if dog is not None and not dog.done():
            dog.cancel()
            with contextlib.suppress(BaseException):
                await dog
        if connected:
            # Every session, before disconnect: taskkill /T walks the tree from the still-alive CLI
            # pid, so whatever the worker left running in the background dies with the session.
            pid = _cli_pid(client)
            with _EXPECTED_CLI_EXIT.expecting():
                if pid is not None:
                    transcript.write({"type": "kill_tree", "pid": pid})
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(asyncio.to_thread(kill, pid), timeout=KILL_TREE_TIMEOUT_S)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.disconnect(), timeout=30)
        transcript.write({"type": "end", "elapsed_s": round(time.monotonic() - started, 1),
                          "timed_out": timed_out, "auth_failed": auth_failed, "error": error})
        transcript.close()

    summary = _summary_from(result, last_model) if result is not None else None
    output: dict | None = None
    if result is not None:
        if result.api_error_status in AUTH_STATUSES:
            auth_failed = True
        if result.api_error_status in RATE_LIMIT_STATUSES:
            rate_limited = True
        if result.is_error or result.subtype != "success":
            text = _result_error_text(result)
            if _AUTH_TEXT_RE.search(text):
                auth_failed = True
            elif _RATE_LIMIT_TEXT_RE.search(text):
                rate_limited = True
            error = error or text
        elif result.structured_output is None:
            error = "worker finished without structured output"
        else:
            try:
                output = schema.model_validate(result.structured_output).model_dump(mode="json")
            except ValidationError as exc:
                error = _clip(f"structured output failed validation against {schema.__name__}: {exc}", 4000)
    elif error is None:
        error = "worker session ended without a result message"
    if result is None and error is not None and stderr_tail:
        error = _clip(f"{error}; stderr: " + " | ".join(stderr_tail), 4000)
    if timed_out and error is None:
        error = f"worker timed out after {timeout_s:.0f}s"
    if auth_failed and error is None:
        error = "authentication failed"
    if auth_failed:
        rate_limited = False  # an auth failure is the stronger, already-handled signal
    if output is not None:
        # A successful result with valid structured output means the provider served the session; a
        # rate limit seen on the way (a subagent's, or a request the CLI retried) is not an outage.
        rate_limited = False
    return WorkerRun(
        output=output, summary=summary, error=error, auth_failed=auth_failed, timed_out=timed_out,
        rate_limited=rate_limited,
    )
