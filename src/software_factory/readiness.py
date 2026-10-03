"""Per-repo readiness score (informational; never blocks dispatch).

Weighted checks, score 0-100:

- ``commands_discovered``   verification commands were discovered (and from where)
- ``commands_pass``         each verify command exits 0 on the default branch
- ``runtime_under_budget``  one pass of the verify commands takes < 120 s
- ``not_flaky``             every command gives the same exit code on a second run
- ``agent_doc_present``     CLAUDE.md or AGENTS.md exists
- ``verification_section``  that doc has a verification section
- ``dev_server``            (``visual`` repos only) the dev server responds on its port within 60 s

Commands run in a throwaway worktree at ``origin/<default>``, never in the user's checkout. Both the
command runner and the workdir provider are injectable so tests need neither git nor subprocesses.
With ``run_commands=False`` only the static checks run and the runtime checks are reported as
"not run" (they earn no credit, so the score stays honest).
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from .discover import Verification, agent_doc, discover_cached, git_head_sha, has_verification_section

if TYPE_CHECKING:
    from .repos import RepoInfo
    from .state import State
    from .worktree import WorktreeManager

# (exit_code, seconds, output)
Runner = Callable[[str, Path, float], tuple[int, float, str]]
# context manager yielding the directory to run commands in
WorkdirProvider = Callable[[Any], Any]
# (ok, detail) after starting the dev command, polling the port and killing the process tree
DevProbe = Callable[[str, Path, int, float], tuple[bool, str]]

WEIGHTS: dict[str, int] = {
    "commands_discovered": 20,
    "commands_pass": 25,
    "runtime_under_budget": 10,
    "not_flaky": 15,
    "agent_doc_present": 15,
    "verification_section": 15,
    "dev_server": 15,  # visual repos only
}
RUNTIME_CHECKS: tuple[str, ...] = ("commands_pass", "runtime_under_budget", "not_flaky")
NOT_RUN = "not run"
TIMEOUT_EXIT_CODE = 124
DEFAULT_COMMAND_TIMEOUT_S = 600.0
DEFAULT_RUNTIME_BUDGET_S = 120.0
DEFAULT_DEV_TIMEOUT_S = 60.0


class ReadinessCheck(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    weight: int

    @property
    def not_run(self) -> bool:
        """Skipped on purpose (static-only run): earns nothing, but it is not a failure to report."""
        return self.detail == NOT_RUN


class Readiness(BaseModel):
    repo: str
    score: int
    checks: list[ReadinessCheck] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)  # names of checks that are not ok
    flaky: list[str] = Field(default_factory=list)  # commands whose two runs disagreed
    runtime_s: float = 0.0  # one pass of the verify commands (setup excluded)
    computed_at: str = ""
    head_sha: str = ""

    def check(self, name: str) -> ReadinessCheck | None:
        return next((c for c in self.checks if c.name == name), None)


# ---------------------------------------------------------------------------------- scoring
def score_repo(
    repo: RepoInfo,
    verification: Verification,
    *,
    run_commands: bool = True,
    wt_manager: WorktreeManager | None = None,
    runner: Runner | None = None,
    workdir: WorkdirProvider | None = None,
    dev_probe: DevProbe | None = None,
    command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
    runtime_budget_s: float = DEFAULT_RUNTIME_BUDGET_S,
    dev_timeout_s: float = DEFAULT_DEV_TIMEOUT_S,
) -> Readiness:
    """Score one repo. ``runner``/``workdir``/``dev_probe`` default to real subprocesses and a
    throwaway worktree from ``wt_manager`` (or a temp-rooted ``WorktreeManager``)."""
    visual = bool(getattr(repo.override, "visual", False))
    commands = verification.verify_commands
    checks: list[ReadinessCheck] = []

    # -- static
    checks.append(
        _check(
            "commands_discovered",
            bool(commands),
            f"{len(commands)} command(s) from {verification.source}" if commands else "no verification commands discovered",
        )
    )
    doc = agent_doc(repo.path)
    checks.append(_check("agent_doc_present", doc is not None, doc.name if doc else "no CLAUDE.md or AGENTS.md"))
    section = doc is not None and has_verification_section(repo.path)
    checks.append(
        _check(
            "verification_section",
            section,
            "verification section found" if section else "no verification/commands section in the agent doc",
        )
    )

    # -- runtime
    report = _RunReport()
    if not run_commands:
        report.skipped = NOT_RUN
    elif not commands and not (visual and verification.dev_command):
        report.skipped = "no commands to run"
    else:
        runner = runner or default_runner
        workdir = workdir or worktree_workdir(wt_manager)
        try:
            with workdir(repo) as cwd:
                cwd = Path(cwd)
                if commands:
                    report = _run_suite(commands, verification.setup_commands, cwd, runner, command_timeout_s)
                else:
                    report.skipped = "no commands to run"
                if visual:
                    report.dev = _probe_dev_server(verification, cwd, dev_probe or default_dev_probe, dev_timeout_s)
        except Exception as exc:  # worktree creation failed, provider bug: report, never crash
            report = _RunReport(skipped=f"workdir unavailable: {exc}")

    checks.extend(_runtime_checks(report, commands, runtime_budget_s))
    if visual:
        if report.dev is not None:
            ok, detail = report.dev
        elif not verification.dev_command or not verification.dev_port:
            ok, detail = False, "visual repo without dev_command/dev_port (set repos.overrides.<repo>.dev_command/dev_port)"
        else:
            ok, detail = False, report.skipped or NOT_RUN
        checks.append(_check("dev_server", ok, detail))

    total = sum(c.weight for c in checks)
    score = round(100 * sum(c.weight for c in checks if c.ok) / total) if total else 0
    return Readiness(
        repo=repo.slug,
        score=score,
        checks=checks,
        missing=[c.name for c in checks if not c.ok],
        flaky=report.flaky,
        runtime_s=round(report.runtime_s, 2),
        computed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        head_sha=git_head_sha(repo.path),
    )


def readiness_cached(
    state: State,
    repo: RepoInfo,
    verification: Verification | None = None,
    *,
    run_commands: bool = True,
    force: bool = False,
    **kwargs: Any,
) -> Readiness:
    """``score_repo`` memoised in ``state`` keyed by HEAD sha. A cached full run satisfies a static
    request; a cached static run never satisfies a full one. ``kwargs`` go to ``score_repo``."""
    sha = git_head_sha(repo.path)
    key = f"readiness:{repo.slug}"
    cached = state.cache_get("readiness", key) if sha and not force else None
    if (
        isinstance(cached, dict)
        and cached.get("head_sha") == sha
        and (cached.get("run_commands") or not run_commands)
    ):
        try:
            return Readiness.model_validate(cached["readiness"])
        except (KeyError, ValidationError):
            pass
    if verification is None:
        verification = discover_cached(state, repo)
    readiness = score_repo(repo, verification, run_commands=run_commands, **kwargs)
    if sha:
        state.cache_set(
            "readiness",
            key,
            {"head_sha": sha, "run_commands": run_commands, "readiness": readiness.model_dump(mode="json")},
        )
    return readiness


# ------------------------------------------------------------------------------ run the suite
@dataclass
class _RunReport:
    skipped: str = ""  # reason runtime checks did not run (empty = they ran)
    setup_failed: str = ""  # "<cmd> (exit N)" when a setup command failed
    first: list[tuple[str, int, float, str]] = field(default_factory=list)  # (cmd, exit, secs, tail)
    second: list[tuple[str, int, float, str]] = field(default_factory=list)
    dev: tuple[bool, str] | None = None

    @property
    def runtime_s(self) -> float:
        return sum(secs for _c, _code, secs, _t in self.first)

    @property
    def flaky(self) -> list[str]:
        second = {cmd: code for cmd, code, _s, _t in self.second}
        return [cmd for cmd, code, _s, _t in self.first if cmd in second and second[cmd] != code]


def _run_suite(
    commands: list[str], setup: list[str], cwd: Path, runner: Runner, timeout_s: float
) -> _RunReport:
    report = _RunReport()
    for cmd in setup:
        code, _secs, out = runner(cmd, cwd, timeout_s)
        if code != 0:
            report.setup_failed = f"{cmd} (exit {code}): {_tail(out)}"
            return report
    for cmd in commands:
        code, secs, out = runner(cmd, cwd, timeout_s)
        report.first.append((cmd, code, secs, _tail(out)))
    for cmd, code, _secs, _text in report.first:
        if code == TIMEOUT_EXIT_CODE:
            continue  # do not wait for a second timeout; a timeout is a consistent failure
        code2, secs2, out2 = runner(cmd, cwd, timeout_s)
        report.second.append((cmd, code2, secs2, _tail(out2)))
    return report


def _runtime_checks(report: _RunReport, commands: list[str], budget_s: float) -> list[ReadinessCheck]:
    if report.skipped:
        return [_check(name, False, report.skipped) for name in RUNTIME_CHECKS]
    if report.setup_failed:
        detail = f"setup failed: {report.setup_failed}"
        return [_check("commands_pass", False, detail)] + [
            _check(name, False, f"{NOT_RUN} (setup failed)") for name in RUNTIME_CHECKS[1:]
        ]
    failed = [(cmd, code, tail) for cmd, code, _s, tail in report.first if code != 0]
    passed = not failed
    detail = (
        f"{len(commands)} command(s) passed"
        if passed
        else "; ".join(f"`{cmd}` exit {code}: {tail}" for cmd, code, tail in failed)
    )
    runtime = report.runtime_s
    flaky = report.flaky
    return [
        _check("commands_pass", passed, detail),
        _check(
            "runtime_under_budget",
            runtime < budget_s,
            f"{runtime:.1f}s for one pass (budget {budget_s:.0f}s)",
        ),
        _check(
            "not_flaky",
            not flaky,
            "both runs agreed" if not flaky else "exit code changed between runs: " + ", ".join(flaky),
        ),
    ]


def _probe_dev_server(
    verification: Verification, cwd: Path, probe: DevProbe, timeout_s: float
) -> tuple[bool, str] | None:
    if not verification.dev_command or not verification.dev_port:
        return None
    try:
        return probe(verification.dev_command, cwd, verification.dev_port, timeout_s)
    except Exception as exc:
        return False, f"dev server probe failed: {exc}"


def _check(name: str, ok: bool, detail: str) -> ReadinessCheck:
    return ReadinessCheck(name=name, ok=ok, detail=detail, weight=WEIGHTS[name])


def _tail(output: str, limit: int = 300) -> str:
    text = " ".join((output or "").split())
    return text if len(text) <= limit else "..." + text[-limit:]


# --------------------------------------------------------------------------- default providers
def worktree_workdir(wt_manager: WorktreeManager | None = None) -> WorkdirProvider:
    """Workdir provider backed by ``WorktreeManager.for_default_branch`` + ``remove``.

    ``worktree`` is imported lazily (it is another module; readiness must import without it).
    Without a manager, one is created under a private temp directory, which is removed again when
    the provider exits. Hooks need no directory of their own: the manager runs every clone-level
    git with the unplantable ``core.hooksPath`` (``providers.unplantable_hooks_path``)."""

    @contextmanager
    def provider(repo: RepoInfo) -> Iterator[Path]:
        manager = wt_manager
        scratch: Path | None = None
        try:
            if manager is None:
                from .worktree import WorktreeManager

                scratch = Path(tempfile.mkdtemp(prefix="sf-readiness-"))
                manager = WorktreeManager(scratch / "wt")
            wt = manager.for_default_branch(repo)
            try:
                yield Path(wt.path)
            finally:
                manager.remove(wt)
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)

    return provider


def default_runner(command: str, cwd: Path, timeout: float) -> tuple[int, float, str]:
    """Run one shell line in ``cwd``; ``(exit_code, seconds, combined output)``.

    Timeouts kill the whole process tree and return exit code 124. A command whose executable
    cannot be found returns 127 without spawning anything."""
    argv = shell_argv(command)
    if argv is None:
        return 127, 0.0, f"command not found: {command.split()[0]}"
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=command_env(),
            **_popen_group_kwargs(),
        )
    except OSError as exc:
        return 127, time.monotonic() - start, f"failed to start: {exc}"
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc.pid)
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out = ""
        return TIMEOUT_EXIT_CODE, time.monotonic() - start, f"{out or ''}\n[timed out after {timeout:.0f}s]"
    return proc.returncode, time.monotonic() - start, out or ""


# Dev servers bind either loopback family: Vite on Windows resolves ``localhost`` to ``::1`` only, so
# a probe of 127.0.0.1 alone reports "no response" for a server that is up.
LOOPBACK_HOSTS: tuple[str, ...] = ("127.0.0.1", "::1")


def port_in_use(port: int, host: str | None = None, timeout: float = 1.0) -> bool:
    """Something already accepts connections on ``host:port`` (both loopback families by default)."""
    for candidate in (host,) if host else LOOPBACK_HOSTS:
        family = socket.AF_INET6 if ":" in candidate else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                if sock.connect_ex((candidate, port)) == 0:
                    return True
        except OSError:  # family unavailable on this host
            continue
    return False


def _loopback_urls(port: int) -> list[str]:
    return [f"http://[{h}]:{port}/" if ":" in h else f"http://{h}:{port}/" for h in LOOPBACK_HOSTS]


def default_dev_probe(command: str, cwd: Path, port: int, timeout: float) -> tuple[bool, str]:
    """Start ``command`` in ``cwd``, poll ``http://127.0.0.1:<port>/`` and ``http://[::1]:<port>/``
    until one answers or ``timeout``
    elapses, then kill the process tree (Windows: ``taskkill /T /F``).

    The port must be free first: the probe only knows "something answered", so a server that was
    already listening (the operator's own ``npm run dev``, another repo on the same port) would make
    it pass without this repo's server ever starting."""
    argv = shell_argv(command)
    if argv is None:
        return False, f"command not found: {command.split()[0]}"
    if port_in_use(port):
        return (
            False,
            f"port {port} is already in use before the dev server was started; stop whatever listens on it "
            "(or change dev_port) so the probe cannot pass on another server",
        )
    urls = _loopback_urls(port)
    url = " / ".join(urls)
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=command_env(),
            **_popen_group_kwargs(),
        )
    except OSError as exc:
        return False, f"failed to start dev server: {exc}"
    try:
        while True:
            if proc.poll() is not None:
                return False, f"dev server exited with {proc.returncode} before responding on {url}"
            for candidate in urls:
                try:
                    response = httpx.get(candidate, timeout=2.0, follow_redirects=False)
                    return True, f"HTTP {response.status_code} from {candidate} after {time.monotonic() - start:.1f}s"
                except httpx.HTTPError:
                    pass
            if time.monotonic() - start >= timeout:
                return False, f"no response on {url} within {timeout:.0f}s"
            time.sleep(1.0)
    finally:
        kill_tree(proc.pid)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


_SHELL_META = re.compile(r"[&|;<>`$()^]")
_QUOTED = re.compile(r'"[^"]*"' + "|'[^']*'")
_WIN_TOKEN = re.compile(r'"([^"]*)"|(\S+)')


def shell_argv(command: str) -> list[str] | None:
    """argv for one shell line, as a list (never ``shell=True``).

    Lines without shell operators run the executable directly (``shutil.which`` resolves Windows
    ``.cmd`` shims such as ``npm.cmd``); ``None`` when the executable is missing. Lines with
    operators (``&&``, ``|``, redirects...) go through ``cmd.exe /d /s /c`` or ``sh -c``."""
    command = command.strip()
    if not command:
        return None
    if _SHELL_META.search(_QUOTED.sub("", command)):
        if os.name == "nt":
            return [os.environ.get("COMSPEC") or "cmd.exe", "/d", "/s", "/c", command]
        return ["/bin/sh", "-c", command]
    tokens = split_command(command)
    if not tokens:
        return None
    exe = shutil.which(tokens[0])
    if exe is None:
        return None
    return [exe, *tokens[1:]]


def split_command(command: str) -> list[str]:
    """Split a shell line into tokens: shlex on POSIX; double-quote aware on Windows (backslashes
    are path separators there, not escapes)."""
    if os.name == "nt":
        return [m.group(1) if m.group(1) is not None else m.group(2) for m in _WIN_TOKEN.finditer(command)]
    return shlex.split(command)


def command_env() -> dict[str, str]:
    """Inherited environment plus CI-style flags (no watch modes, no prompts, no colour)."""
    env = dict(os.environ)
    env.update({"CI": "1", "GIT_TERMINAL_PROMPT": "0", "NO_COLOR": "1", "FORCE_COLOR": "0", "PYTHONUNBUFFERED": "1"})
    return env


def kill_tree(pid: int) -> None:
    """Kill a process and all its descendants (Windows: ``taskkill /T /F``; POSIX: see
    ``_kill_tree_posix``). Best effort: errors (already gone, access denied) are ignored."""
    if os.name == "nt":
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        return
    _kill_tree_posix(pid)


def _kill_tree_posix(pid: int) -> None:
    """SIGKILL ``pid``, every descendant ``ps`` reports, and ``pid``'s process group when it leads
    one of its own (``start_new_session``). Never the caller's own group: the Claude CLI the SDK
    spawns shares the engine's process group, and ``killpg`` on it would kill the engine itself."""
    sigkill = getattr(signal, "SIGKILL", 9)
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return  # already gone (its orphans, if any, are no longer its descendants)
    descendants = _posix_descendants(pid)  # snapshot first: killing re-parents the children
    if pgid == pid and pgid != os.getpgrp():
        with suppress(OSError):
            os.killpg(pgid, sigkill)
    me = os.getpid()
    for target in (pid, *descendants):
        if target == me:
            continue
        with suppress(OSError):  # ProcessLookupError / PermissionError are OSError subclasses
            os.kill(target, sigkill)


def _posix_descendants(pid: int) -> list[int]:
    """Every descendant of ``pid`` (breadth first) from ``ps -A -o pid= -o ppid=``; ``[]`` on error."""
    try:
        proc = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "ppid="],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, check=False,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return []
    return descendants_from_table(pid, proc.stdout or "")


def descendants_from_table(pid: int, table: str) -> list[int]:
    """Descendants of ``pid`` (breadth first) in a ``<pid> <ppid>`` per line process table."""
    children: dict[int, list[int]] = {}
    for line in table.splitlines():
        parts = line.split()
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            continue
        children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found: list[int] = []
    seen = {pid}
    queue = [pid]
    while queue:
        for child in children.get(queue.pop(0), []):
            if child not in seen:
                seen.add(child)
                found.append(child)
                queue.append(child)
    return found


def _popen_group_kwargs() -> dict[str, Any]:
    # POSIX: own process group so kill_tree can take out the whole tree. Windows: taskkill /T
    # walks the tree by parent pid, so no flags are needed.
    return {} if os.name == "nt" else {"start_new_session": True}
