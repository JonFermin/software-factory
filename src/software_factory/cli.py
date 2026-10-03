"""``factory`` command line (typer).

One process, one profile (``--profile``, default ``personal``; ``--profiles-dir`` overrides where
profiles live). Every command loads that profile, resolves its runtime paths under the per-profile
data dir and talks to the engine modules; nothing here performs an outward write itself -- ``tick``,
``run``, ``ready`` and ``revert-refine`` go through the dispatcher / kind handlers, whose writes go
through ``effects.py`` (observe mode and ``--dry-run`` therefore record instead of writing).

Output is plain ASCII through ``typer.echo`` (stdout is switched to UTF-8 with replacement so a
cp1252 Windows console never raises). Exit codes: ``0`` ok, ``1`` something failed (a doctor check,
an action, a missing record), ``2`` usage error or -- for ``tick`` / ``run`` / ``ready`` -- an auth
failure or a model provider rate / usage limit (``provider unavailable``).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import httpx
import typer
from filelock import FileLock, Timeout

from . import dispatch, metrics, providers, scheduler
from .actions import ISSUE_KINDS, PR_KINDS, Action, Kind
from .config import DEFAULT_PROFILE, ConfigError, Profile, load_profile
from .discover import discover
from .effects import EffectError
from .kinds.base import Deps, Outcome
from .kinds.ready import ready_action
from .kinds.refine import revert_refine
from .notify import Audit, Notifier
from .paths import ProfilePaths
from .readiness import Readiness, readiness_cached, score_repo
from .repos import RepoIndex
from .scan import pr_actions
from .scheduler import SchedulerError
from .sources.github import GhAuthError, GhError, GitHub
from .state import State
from .trackers.base import Issue, Tracker, TrackerAuthError, TrackerError
from .worktree import WorktreeError, WorktreeManager

if TYPE_CHECKING:
    from .config import LinearConfig
    from .repos import RepoInfo

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
SLACK_AUTH_TEST = "https://slack.com/api/auth.test"
SLACK_TIMEOUT_S = 15.0
CLI_VERSION_TIMEOUT_S = 20.0
LAST_RUNS = 10
NOTE_WIDTH = 100

# Errors a command turns into one line + exit 1 (bugs still surface as tracebacks).
KNOWN_ERRORS: tuple[type[BaseException], ...] = (
    ConfigError, GhError, TrackerError, SchedulerError, WorktreeError, EffectError,
    providers.ProviderError, OSError, ValueError,
)  # fmt: skip

_PR_TARGET = re.compile(r"^(?:(?P<owner>[A-Za-z0-9_.-]+)/)?(?P<name>[A-Za-z0-9_.-]+)#(?P<number>\d+)$")
_PR_URL = re.compile(r"^https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+)/pull/(?P<number>\d+)")
_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)")

app = typer.Typer(
    name="factory",
    help="Local, scheduled software factory: Linear is the source of truth, workers propose, the engine acts.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
schedule_app = typer.Typer(
    help="Install, remove or show the OS scheduler jobs (tick every N minutes + weekly digest).",
    no_args_is_help=True,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)
app.add_typer(schedule_app, name="schedule")


# ------------------------------------------------------------------------------------- settings
@dataclass
class Settings:
    profile: str = DEFAULT_PROFILE
    profiles_dir: Path | None = None


def _settings(ctx: typer.Context) -> Settings:
    obj = ctx.find_object(Settings)
    return obj if obj is not None else Settings()


def _utf8_console() -> None:
    """Windows consoles default to cp1252; switch to UTF-8 with replacement so output never raises."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError, AttributeError):
            reconfigure(encoding="utf-8", errors="replace")


@app.callback()
def main(
    ctx: typer.Context,
    profile: Annotated[str, typer.Option("--profile", "-p", help="Profile name (<profiles-dir>/<name>/factory.toml).")] = DEFAULT_PROFILE,
    profiles_dir: Annotated[
        Path | None, typer.Option("--profiles-dir", help="Where profiles live (default: $SOFTWARE_FACTORY_PROFILES_DIR or <repo>/profiles).")
    ] = None,
) -> None:
    """Software factory: scan Linear + GitHub, dispatch narrow Claude workers, apply policy-gated effects."""
    _utf8_console()
    ctx.obj = Settings(profile=profile, profiles_dir=profiles_dir)


# -------------------------------------------------------------------------------------- helpers
def fail(message: str, code: int = EXIT_FAIL) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code)


def _profile(ctx: typer.Context) -> Profile:
    settings = _settings(ctx)
    try:
        return load_profile(settings.profile, settings.profiles_dir)
    except ConfigError as exc:
        fail(f"profile error: {exc}")


def _paths(profile: Profile) -> ProfilePaths:
    return ProfilePaths.for_profile(profile.name)


@contextlib.contextmanager
def _known_errors(what: str) -> Iterator[None]:
    """Turn a known engine error into one line + exit 1."""
    try:
        yield
    except typer.Exit:
        raise
    except KNOWN_ERRORS as exc:
        fail(f"{what} failed: {type(exc).__name__}: {exc}")


@contextlib.contextmanager
def _tick_idle(paths: ProfilePaths) -> Iterator[None]:
    """Hold the tick lock while touching worktrees; refuse when a tick is running."""
    paths.ensure()
    lock = FileLock(str(paths.lock_file), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        fail("a tick is running for this profile; try again when it has finished")
    try:
        yield
    finally:
        lock.release()


def _clip(text: str, limit: int = NOTE_WIDTH) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    fmt = lambda cells: "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()  # noqa: E731
    return [fmt(headers), *(fmt(r) for r in rows)]


def _echo_lines(lines: list[str]) -> None:
    for line in lines:
        typer.echo(line)


# ``run_one`` notes of the outcomes that stopped on infrastructure, not on the target: an auth
# failure or a provider that could not serve the session. Both exit 2, like a tick (invariant 5).
WAVE_STOP_NOTES: tuple[str, ...] = ("auth failure", "provider unavailable")


def _outcome_exit_code(outcome: Outcome) -> int:
    if outcome.status != "failed":
        return EXIT_OK
    return dispatch.EXIT_AUTH if outcome.note.startswith(WAVE_STOP_NOTES) else EXIT_FAIL


# ------------------------------------------------------------------------- injectable seams
# Module-level so tests (and a future work profile) can swap the real clients for fakes.
def build_deps(profile: Profile, paths: ProfilePaths, *, dry_run: bool) -> Deps:
    return Deps.build(profile, paths, dry_run=dry_run)


def gh_client(profile: Profile) -> GitHub:
    return GitHub.from_profile(profile)


def tracker_client(profile: Profile) -> Tracker | None:
    """The Linear tracker when it is enabled and the profile has a key; ``None`` otherwise."""
    if not profile.linear.enabled or not profile.secret("LINEAR_API_KEY"):
        return None
    from .trackers.linear import LinearTracker

    return LinearTracker(profile)


def discover_repos(profile: Profile) -> RepoIndex:
    return RepoIndex.discover(profile)


def scheduler_status(profile: Profile, paths: ProfilePaths) -> str:
    return scheduler.show(profile, paths)


def slack_auth_test(token: str, http: httpx.Client | None = None) -> tuple[bool, str]:
    """Slack ``auth.test`` with the bot token: ``(ok, detail)``; never raises."""
    try:
        client = http or httpx.Client(timeout=SLACK_TIMEOUT_S)
        response = client.post(SLACK_AUTH_TEST, headers={"Authorization": f"Bearer {token}"}, timeout=SLACK_TIMEOUT_S)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - a doctor check reports, never raises
        return False, f"{type(exc).__name__}: {exc}"
    if not isinstance(data, dict) or not data.get("ok"):
        return False, str(data.get("error", "unknown_error")) if isinstance(data, dict) else "bad json"
    return True, f"bot {data.get('user') or '?'} in workspace {data.get('team') or '?'}"


def gh_detected_login() -> str:
    """The login ``gh`` is really authenticated as (``gh api user --jq .login``). Deliberately
    ignores ``github.login``: ``GitHub.login()`` echoes that setting without ever checking it."""
    return GitHub().login()


def claude_cli_status(profile: Profile | None = None) -> tuple[str, str]:
    """Which Claude Code CLI the Agent SDK will launch, and whether it is new enough.

    Mirrors the SDK's lookup: ``provider.cli_path`` when set (the SDK uses it verbatim and never
    searches; on Windows it must be a native ``.exe``/``.com`` because the SDK refuses ``.cmd``/``.bat``
    shims and nothing else spawns), else the SDK's bundled binary, else ``claude`` on PATH (Windows:
    a native ``.exe`` only), else the SDK's extra locations. A CLI older than the version the SDK
    was built against (``claude_agent_sdk._cli_version``) is a ``fail``: it rejects the models the
    SDK (and ``provider.model``) expect, so every worker dies at startup.
    """
    try:
        import claude_agent_sdk
        from claude_agent_sdk._cli_version import __cli_version__ as expected
    except Exception as exc:  # noqa: BLE001
        return "fail", f"claude-agent-sdk is not importable: {type(exc).__name__}: {exc}"
    windows = sys.platform == "win32"
    configured = profile.provider.cli_path if profile is not None else None
    if configured:
        path = Path(os.path.expandvars(configured)).expanduser()
        if not path.is_file():
            return "fail", f"provider.cli_path {configured!r} is not a file; fix it or unset it to use the SDK's own lookup"
        if windows and not _is_native_exe(str(path)):
            return (
                "fail",
                f"provider.cli_path {path} is not a native .exe/.com; on Windows the SDK refuses batch shims "
                "(claude.cmd) and cannot spawn anything else -- point it at claude.exe",
            )
        return _cli_version_status(str(path), expected, f"provider.cli_path {path}", windows)
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / ("claude.exe" if windows else "claude")
    if bundled.is_file():
        return "ok", f"bundled claude CLI {expected} at {bundled}"
    found, shim = _find_cli_like_sdk(windows)
    if found is None:
        if shim is not None:
            return (
                "fail",
                f"no bundled CLI in the SDK and `claude` on PATH is a script shim ({shim}) the SDK refuses to "
                "run; install the native claude.exe (claude.ai/install.ps1) or set provider.cli_path",
            )
        return (
            "fail",
            f"no claude CLI: the SDK has no bundled binary (expected {expected}), `claude` is not on PATH and "
            "provider.cli_path is unset",
        )
    return _cli_version_status(found, expected, f"no bundled CLI in the SDK; using {found}", windows)


def _is_native_exe(path: str) -> bool:
    """What the SDK accepts on Windows: the final component names an ``.exe`` / ``.com`` image."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return name.rstrip(". ").lower().endswith((".exe", ".com"))


def _cli_extra_locations(windows: bool) -> list[Path]:
    """The fixed locations the SDK probes after PATH (tests patch this out)."""
    home = Path.home()
    if windows:
        return [home / ".local/bin/claude.exe"]
    return [
        home / ".npm-global/bin/claude",
        Path("/usr/local/bin/claude"),
        home / ".local/bin/claude",
        home / "node_modules/.bin/claude",
        home / ".yarn/bin/claude",
        home / ".claude/local/claude",
    ]


def _find_cli_like_sdk(windows: bool) -> tuple[str | None, str | None]:
    """``(cli, shim)``: the binary the SDK's ``_find_cli`` would pick, or ``(None, shim)`` when only
    a script shim was found (which the SDK then refuses) / ``(None, None)`` when nothing was."""
    shim: str | None = None
    found = shutil.which("claude")
    if found is not None:
        if not windows or _is_native_exe(found):
            return found, None
        exe = shutil.which("claude.exe")
        if exe and _is_native_exe(exe):
            return exe, None
        shim = found
    for candidate in _cli_extra_locations(windows):
        if candidate.is_file():
            return str(candidate), None
    return None, shim


def _cli_version_status(path: str, expected: str, label: str, windows: bool) -> tuple[str, str]:
    """``ok`` when ``path --version`` >= the SDK's expected CLI version, ``fail`` when older, ``warn``
    when the version cannot be read."""
    text = _cli_version(path)
    found, want = _parse_semver(text), _parse_semver(expected)
    if found is None or want is None:
        return "warn", f"{label} ({text}); could not compare it with the CLI the SDK expects ({expected})"
    if found < want:
        exe = "claude.exe" if windows else "claude"
        return (
            "fail",
            f"{label} is {text}, older than the CLI the SDK expects ({expected}) and too old for the configured "
            f"model; run `claude update` or set provider.cli_path to a newer {exe}",
        )
    return "ok", f"{label} ({text}; SDK expects {expected})"


def _parse_semver(text: str) -> tuple[int, int, int] | None:
    match = _SEMVER.search(text or "")
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def _cli_version(path: str) -> str:
    try:
        proc = subprocess.run(
            [path, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=CLI_VERSION_TIMEOUT_S, check=False,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"version unknown: {type(exc).__name__}"
    text = (proc.stdout or proc.stderr or "").strip().splitlines()
    return text[0] if text and proc.returncode == 0 else f"--version exited {proc.returncode}"


def toast_backend_status() -> tuple[str, str]:
    try:
        import desktop_notifier  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return "warn", f"desktop-notifier not importable ({type(exc).__name__}: {exc}); toasts will not show"
    return "ok", "desktop-notifier importable"


# ------------------------------------------------------------------------------------------ tick
@app.command()
def tick(
    ctx: typer.Context,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Scan, filter and print; start no worker, write no state.")] = False,
) -> None:
    """Run one tick: lock -> precheck -> scan -> filter -> dispatch -> record. Exit 1 if an action failed, 2 on an auth failure or a provider rate / usage limit."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("tick"):
        report = asyncio.run(dispatch.run_tick(profile, paths, dry_run=dry_run, print_fn=typer.echo))
    if dry_run and not report.locked and report.auth_error is None:
        typer.echo(
            f"dry-run: {len(report.outcomes)} action(s) would run, {len(report.skipped)} skipped, "
            f"{len(report.errors)} scan error(s); nothing was started"
        )
    raise typer.Exit(report.exit_code)


# ------------------------------------------------------------------------------------------- run
@app.command()
def run(
    ctx: typer.Context,
    kind: Annotated[str, typer.Argument(help="ci | conflict | rc | rv | refine | plan | build | ready")],
    target: Annotated[str, typer.Argument(help="owner/name#123 or name#123 (PR kinds), issue key (issue kinds), repo (ready)")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Build the action and print it; start no worker.")] = False,
) -> None:
    """Run one action by hand. Bypasses the dedupe/attempt/enabled-kind filters; still respects the profile's mode. Exit 1 if it failed, 2 on an auth failure or a provider rate / usage limit."""
    kind_value = _parse_kind(kind)
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("run"):
        deps = build_deps(profile, paths, dry_run=dry_run)
        action = build_action(deps, kind_value, target)
        typer.echo(f"action: {action.describe()} (version {action.version})")
        outcome = asyncio.run(
            dispatch.run_one(profile, paths, action, dry_run=dry_run, deps=deps, print_fn=typer.echo, force=True)
        )
    raise typer.Exit(_outcome_exit_code(outcome))


def _parse_kind(text: str) -> Kind:
    try:
        return Kind(text.strip().lower())
    except ValueError:
        raise typer.BadParameter(f"unknown kind {text!r}; expected one of {', '.join(k.value for k in Kind)}") from None


def parse_pr_target(repos: RepoIndex, target: str) -> tuple[RepoInfo, int]:
    """``owner/name#123`` | ``name#123`` | a GitHub PR URL -> ``(managed repo, number)``."""
    match = _PR_TARGET.match(target.strip()) or _PR_URL.match(target.strip())
    if match is None:
        raise typer.BadParameter(f"{target!r} is not a pull request target; use owner/name#123 or name#123")
    owner, name, number = match.group("owner"), match.group("name"), int(match.group("number"))
    key = f"{owner}/{name}" if owner else name
    repo = repos.get(key)
    if repo is None:
        managed = ", ".join(repos.slugs()) or "none"
        raise typer.BadParameter(f"repo {key!r} is not managed by this profile; managed repos: {managed}")
    return repo, number


def issue_action(kind: Kind, issue: Issue) -> Action:
    """The action a scan would build for ``issue`` (version: ``updated_at``, or the id for ``build``)."""
    context: dict[str, Any] = {"issue_id": issue.id, "labels": list(issue.labels), "state_type": issue.state_type}
    if kind == Kind.BUILD:
        context.update(
            branch_name=issue.branch_name,
            team_id=issue.team_id,
            unfinished_blockers=[b.key for b in issue.unfinished_blockers],
        )
    return Action(
        kind=kind,
        version=issue.id if kind == Kind.BUILD else issue.updated_at,
        title=issue.title,
        issue_id=issue.id,
        issue_key=issue.key,
        url=issue.url,
        context=context,
    )


def build_action(deps: Deps, kind: Kind, target: str) -> Action:
    """The ``Action`` for ``factory run <kind> <target>`` from the *current* external state."""
    if kind in PR_KINDS:
        repo, number = parse_pr_target(deps.repos, target)
        pr = deps.gh.pr_details(repo.slug, number)
        if pr.state != "OPEN":
            fail(f"{repo.slug}#{number} is {pr.state.lower()}; only open pull requests can be worked on")
        base = {"title": pr.title, "repo": repo.slug, "pr": pr.number, "url": pr.url}
        if kind == Kind.RV:
            context = {
                "head_ref_name": pr.head_ref_name, "head_ref_oid": pr.head_ref_oid,
                "base_ref_name": pr.base_ref_name, "base_ref_oid": pr.base_ref_oid,
                "head_repo": pr.head_repo, "is_fork": pr.is_fork, "is_draft": pr.is_draft, "author": pr.author,
            }  # fmt: skip
            return Action(kind=Kind.RV, version=pr.head_ref_oid, context=context, **base)
        detected = [a for a in pr_actions(pr, gh=deps.gh, state=deps.state) if a.kind == kind]
        if not detected:
            fail(f"nothing to run: {repo.slug}#{number} has no {kind.value} trigger right now ({_pr_hint(kind)})")
        return detected[0]
    if kind in ISSUE_KINDS:
        if deps.tracker is None:
            fail("no tracker configured for this profile (linear.enabled = true + LINEAR_API_KEY in the profile .env)")
        return issue_action(kind, deps.tracker.get_issue(target.strip()))
    try:
        return ready_action(deps, target.strip())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _pr_hint(kind: Kind) -> str:
    return {
        Kind.CI: "no failing check on the current head",
        Kind.CONFLICT: "mergeable is not CONFLICTING",
        Kind.RC: "no reviewer feedback newer than your last commit and no unresolved thread",
    }.get(kind, "nothing detected")


# ---------------------------------------------------------------------------------------- status
@app.command()
def status(ctx: typer.Context) -> None:
    """Open flags, kept worktrees, factory PRs and the last runs for this profile."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("status"), State(paths.state_db) as state:
        lock = FileLock(str(paths.lock_file), timeout=0)
        try:
            lock.acquire()
            running = "idle"
            lock.release()
        except Timeout:
            running = "tick running"
        typer.echo(f"profile {profile.name}: mode {profile.mode}, provider {profile.provider.kind}, {running}")
        typer.echo(f"enabled kinds: {', '.join(k.value for k in profile.enabled_kinds) or 'none'}")
        typer.echo(f"data dir: {paths.root}")

        flags = state.flags()
        typer.echo("")
        typer.echo(f"open flags ({len(flags)}):")
        _echo_lines(_flag_rows(flags))

        kept = state.kept_worktrees()
        typer.echo("")
        typer.echo(f"kept worktrees ({len(kept)}):")
        if kept:
            rows = [[k["target"], k["kind"], k["path"], _clip(k.get("reason") or "", 60), k["kept_at"]] for k in kept]
            _echo_lines(_table(["target", "kind", "path", "reason", "kept at"], rows))

        prs = state.factory_prs()
        typer.echo("")
        typer.echo(f"factory-opened PRs ({len(prs)}):")
        if prs:
            rows = [[f"{p['repo']}#{p['number']}", p["branch"], p["kind"], p.get("issue_key") or "-", p.get("url") or ""] for p in prs]
            _echo_lines(_table(["pr", "branch", "kind", "issue", "url"], rows))

        runs = state.runs()[-LAST_RUNS:]
        typer.echo("")
        typer.echo(f"last {len(runs)} run(s):")
        if runs:
            rows = [
                [str(r.get("ts") or ""), str(r.get("kind") or ""), str(r.get("target") or ""), str(r.get("outcome") or ""),
                 "pushed" if r.get("pushed") else "-", _clip(str(r.get("note") or ""), 80)]
                for r in runs
            ]  # fmt: skip
            _echo_lines(_table(["ts", "kind", "target", "outcome", "push", "note"], rows))


def _flag_rows(flags: list[dict]) -> list[str]:
    if not flags:
        return []
    rows = [[str(f["id"]), f["created_at"], f["status"], f["kind"], f["target"], _clip(f["reason"], 50), _clip(f.get("detail") or "", 70)] for f in flags]
    return _table(["id", "created", "status", "kind", "target", "reason", "detail"], rows)


@app.command()
def flags(
    ctx: typer.Context,
    ack: Annotated[int | None, typer.Option("--ack", help="Acknowledge (close) the flag with this id.")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Include acknowledged flags.")] = False,
) -> None:
    """List open flags (things awaiting a human), or acknowledge one with --ack ID."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("flags"), State(paths.state_db) as state:
        if ack is not None:
            if ack not in {f["id"] for f in state.flags()}:
                fail(f"flag {ack} is not an open flag (see `factory flags --all`)")
            state.ack_flag(ack)
            typer.echo(f"acked flag {ack}")
            return
        rows = state.flags(open_only=not all_)
        typer.echo(f"{'all' if all_ else 'open'} flags ({len(rows)}):")
        _echo_lines(_flag_rows(rows))


# ---------------------------------------------------------------------------------------- doctor
@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | n/a (a check that was not run; never counted)
    detail: str = ""

    @property
    def line(self) -> str:
        return f"{self.status:<4} {self.name}: {self.detail}".rstrip()


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> Check:
        check = Check(name, status, detail)
        self.checks.append(check)
        return check

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)


@app.command()
def doctor(
    ctx: typer.Context,
    repo: Annotated[str | None, typer.Option("--repo", help="Also show discovery + static readiness for this managed repo.")] = None,
) -> None:
    """Check the profile, credentials, gh, Linear, Slack, toasts, data dir, repos and scheduler. Exit 1 on any fail."""
    settings = _settings(ctx)
    report = DoctorReport()
    try:
        profile = load_profile(settings.profile, settings.profiles_dir)
    except ConfigError as exc:
        report.add("profile", "fail", str(exc))
        _echo_lines([c.line for c in report.checks])
        raise typer.Exit(EXIT_FAIL) from None
    run_doctor(profile, report, repo=repo)
    _echo_lines([c.line for c in report.checks])
    fails = sum(1 for c in report.checks if c.status == "fail")
    warns = sum(1 for c in report.checks if c.status == "warn")
    typer.echo("")
    typer.echo(f"{len(report.checks)} checks: {fails} fail, {warns} warn")
    raise typer.Exit(EXIT_FAIL if report.failed else EXIT_OK)


def run_doctor(profile: Profile, report: DoctorReport, *, repo: str | None = None) -> DoctorReport:
    """Every doctor check for a loaded profile (pure apart from the injectable seams above)."""
    _doctor_profile(profile, report)
    _doctor_provider(profile, report)
    _doctor_gh(profile, report)
    _doctor_worker_gh(profile, report)
    repos = _doctor_repos(profile, report)
    _doctor_dev_ports(profile, report)
    _doctor_tracker(profile, report, repos)
    _doctor_notify(profile, report)
    paths = _doctor_data_dir(profile, report)
    _doctor_scheduler(profile, paths, report)
    if repo is not None:
        _doctor_repo(repo, repos, report)
    return report


def _doctor_profile(profile: Profile, report: DoctorReport) -> None:
    kinds = ", ".join(k.value for k in profile.enabled_kinds) or "none"
    report.add("profile", "ok", f"{profile.name} from {profile.profile_dir} (mode {profile.mode}, kinds {kinds})")
    env_path = (profile.profile_dir / ".env") if profile.profile_dir else None
    if env_path is not None and not env_path.is_file():
        report.add("secrets", "warn", f"{env_path} not found; provider/Linear/Slack secrets are read from it")
    if not profile.employer:
        report.add("employer rules", "ok", "personal profile (employer = false)")
        return
    # The model validator already rejected oauth / an empty footer; report what holds.
    report.add("employer rules", "ok", f"employer profile: provider {profile.provider.kind}, disclosure footer set, mode {profile.mode}")
    checkout = _git_checkout_of(profile.profile_dir)
    if checkout is not None:
        report.add(
            "employer profile location", "warn",
            f"profile dir sits inside the git checkout {checkout}; move it outside (SOFTWARE_FACTORY_PROFILES_DIR)",
        )  # fmt: skip
    else:
        report.add("employer profile location", "ok", "outside any git checkout")


def _git_checkout_of(path: Path | None) -> Path | None:
    if path is None:
        return None
    for candidate in (path, *path.resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _doctor_provider(profile: Profile, report: DoctorReport) -> None:
    problems = providers.check_credentials(profile)
    if problems:
        report.add("provider credentials", "fail", "; ".join(problems))
    else:
        report.add("provider credentials", "ok", f"{profile.provider.kind} configured (model {profile.provider.model})")
    report.add("claude cli", *claude_cli_status(profile))


def _doctor_gh(profile: Profile, report: DoctorReport) -> None:
    try:
        gh = gh_client(profile)
        gh.auth_check()
        actual = gh_detected_login()
    except GhAuthError as exc:
        report.add("gh auth", "fail", str(exc))
        return
    except GhError as exc:
        report.add("gh auth", "fail", f"gh is not usable: {exc}")
        return
    configured = profile.github.login.strip().lstrip("@")
    if not configured:
        report.add("gh auth", "ok", f"logged in as {actual} (github.login is empty: detected)")
    elif configured.lower() != actual.lstrip("@").lower():
        report.add(
            "gh auth", "fail",
            f"gh is logged in as {actual} but github.login = {configured} (fix github.login or leave it empty; "
            "the scan filters your own PRs and comments by that login)",
        )  # fmt: skip
    else:
        report.add("gh auth", "ok", f"logged in as {actual}")


def _doctor_worker_gh(profile: Profile, report: DoctorReport) -> None:
    """Workers never inherit the engine's ``gh`` login (``providers.gh_isolation_env`` points their
    ``GH_CONFIG_DIR`` at a file and sets ``GH_TOKEN`` to the optional read-only ``GH_WORKER_TOKEN``
    or nothing). Without the token they run ``gh`` unauthenticated by design; the prompt already
    carries the PR, CI and review context the engine fetched, so this is a warning, not a failure."""
    key = providers.GH_WORKER_TOKEN_KEY
    if profile.secret(key):
        report.add(
            "worker gh token", "ok",
            f"{key} set in the profile .env (workers run gh with it; keep it a fine-grained read-only PAT: "
            "contents, pull requests, actions: read)",
        )  # fmt: skip
        return
    report.add(
        "worker gh token", "warn",
        f"{key} not in the profile .env: workers run gh unauthenticated; the engine supplies PR/CI/review "
        "context (optional: a fine-grained read-only PAT with contents, pull requests and actions: read)",
    )  # fmt: skip


def _doctor_dev_ports(profile: Profile, report: DoctorReport) -> None:
    """Visual repos must not share a dev port: two builds in one tick would start both servers on it
    (Vite auto-increments the second) and the worker would screenshot the wrong repo."""
    by_port: dict[int, list[str]] = {}
    for name, override in profile.repos.overrides.items():
        if override.dev_port:
            by_port.setdefault(override.dev_port, []).append(name)
    if not by_port:
        return
    shared = {port: names for port, names in by_port.items() if len(names) > 1}
    if shared:
        detail = "; ".join(f"{port}: {', '.join(names)}" for port, names in sorted(shared.items()))
        report.add(
            "dev ports", "warn",
            f"dev_port shared across repos.overrides ({detail}); concurrent builds would capture the wrong "
            'server -- give each repo its own port (dev_command = "npm run dev -- --port N --strictPort")',
        )  # fmt: skip
    else:
        report.add("dev ports", "ok", f"{len(by_port)} repo(s) with a dev_port, all unique")


def _doctor_repos(profile: Profile, report: DoctorReport) -> RepoIndex:
    try:
        repos = discover_repos(profile)
    except OSError as exc:
        report.add("repos", "fail", f"discovery under {profile.repos.root} failed: {exc}")
        return RepoIndex([])
    root = profile.repos.root_path
    if not root.is_dir():
        report.add("repos", "fail", f"repos.root {root} is not a directory")
    elif len(repos) == 0:
        report.add("repos", "warn", f"no managed repos under {root} (github.owners = {profile.github.owners}, repos.include = {profile.repos.include})")
    else:
        names = ", ".join(r.name for r in repos.all())
        report.add("repos", "ok", f"{len(repos)} managed under {root}: {_clip(names, 160)}")
    return repos


def _doctor_tracker(profile: Profile, report: DoctorReport, repos: RepoIndex) -> None:
    cfg = profile.linear
    issue_kinds = any(k in ISSUE_KINDS for k in profile.enabled_kinds)
    if not cfg.enabled:
        report.add("linear", "ok", "disabled (linear.enabled = false)")
        return
    if not profile.secret("LINEAR_API_KEY"):
        status = "fail" if issue_kinds else "warn"
        why = "an issue kind is enabled" if issue_kinds else "only needed once an issue kind (refine/plan/build) is enabled"
        report.add("linear", status, f"LINEAR_API_KEY missing from the profile .env ({why})")
        return
    tracker = tracker_client(profile)
    if tracker is None:
        report.add("linear", "fail", "tracker could not be constructed")
        return
    try:
        who = tracker.ping()
    except TrackerAuthError as exc:
        report.add("linear", "fail", f"authentication failed: {exc}")
        return
    except TrackerError as exc:
        report.add("linear", "fail", f"unreachable: {exc}")
        return
    report.add("linear", "ok", f"reachable as {who or 'unknown user'}")
    try:
        existing = tracker.labels_exist(cfg.factory_labels)
        missing = [name for name in cfg.factory_labels if not existing.get(name)]
        if missing:
            report.add(
                "linear labels", "fail" if issue_kinds else "warn",
                f"missing {', '.join(missing)} (create them in Linear; label creation is not an allowed write)",
            )  # fmt: skip
        else:
            report.add("linear labels", "ok", ", ".join(cfg.factory_labels))
        _doctor_repo_labels(tracker, cfg, repos, report)
    except TrackerError as exc:
        report.add("linear labels", "fail", f"label lookup failed: {exc}")


def _doctor_repo_labels(tracker: Tracker, cfg: LinearConfig, repos: RepoIndex, report: DoctorReport) -> None:
    """``repo:<name>`` labels (optional: the planner falls back to a description marker). Warn only.

    With ``linear.team_keys`` the check runs per team through ``tracker.ensure_label``, which is the
    lookup sub-issue creation uses: ``labels_exist`` is workspace-wide, so a label owned by another
    team counts as present there while Linear silently drops it from the child issue. A team whose
    id cannot be learnt yet is reported as ``n/a`` (not run), never as a problem.
    """
    repo_labels = [f"{cfg.repo_label_prefix}{r.name}" for r in repos.all()]
    if not repo_labels:
        return
    hint = "optional; the planner falls back to a description marker"
    if cfg.team_keys and callable(getattr(tracker, "ensure_label", None)):
        team_ids = _linear_team_ids(tracker, cfg)
        for key in cfg.team_keys:
            name = f"linear repo labels ({key})"
            team_id = team_ids.get(key.lower())
            if team_id is None:
                report.add(
                    name, "n/a",
                    f"team id unknown (no open issue in team {key} carries a factory label yet); per-team label "
                    "check skipped",
                )  # fmt: skip
                continue
            missing = [lbl for lbl in repo_labels if not _label_usable(tracker, team_id, lbl)]
            if missing:
                report.add(
                    name, "warn",
                    f"{len(missing)} of {len(repo_labels)} repo:<name> labels not usable by team {key} ({hint}): "
                    f"{_clip(', '.join(missing), 200)}",
                )  # fmt: skip
            else:
                report.add(name, "ok", f"all {len(repo_labels)} repo:<name> labels usable by team {key}")
        return
    existing = tracker.labels_exist(repo_labels)
    missing = [name for name in repo_labels if not existing.get(name)]
    if missing:
        report.add(
            "linear repo labels", "warn",
            f"{len(missing)} of {len(repo_labels)} repo:<name> labels missing ({hint}): {_clip(', '.join(missing), 200)}",
        )  # fmt: skip
    else:
        report.add("linear repo labels", "ok", f"all {len(repo_labels)} repo:<name> labels exist")


def _linear_team_ids(tracker: Tracker, cfg: LinearConfig) -> dict[str, str]:
    """Lower-cased team key -> team id for ``linear.team_keys``, learnt from the open issues that
    carry a factory label (``list_issues_with_label`` already honours ``team_keys``; the ``Tracker``
    protocol has no team listing and ``ensure_label`` wants the id). Keys without such an issue are
    absent from the result."""
    wanted = {key.lower() for key in cfg.team_keys}
    found: dict[str, str] = {}
    for label in cfg.factory_labels:
        if wanted <= set(found):
            break
        for issue in tracker.list_issues_with_label(label):
            key = issue.team_key.lower()
            if key in wanted and key not in found and issue.team_id:
                found[key] = issue.team_id
    return found


def _label_usable(tracker: Tracker, team_id: str, label: str) -> bool:
    """``ensure_label`` is a lookup (label creation is not an allowed write): a miss is a
    ``TrackerError``; auth errors propagate to the caller's fail line."""
    try:
        return bool(tracker.ensure_label(team_id, label))
    except TrackerAuthError:
        raise
    except TrackerError:
        return False


def _doctor_notify(profile: Profile, report: DoctorReport) -> None:
    if profile.notify.slack:
        token, user = profile.secret("SLACK_BOT_TOKEN"), profile.secret("SLACK_USER_ID")
        if not token or not user:
            missing = [k for k, v in (("SLACK_BOT_TOKEN", token), ("SLACK_USER_ID", user)) if not v]
            report.add("slack", "fail", f"{', '.join(missing)} missing from the profile .env (notify.slack = true)")
        else:
            ok, detail = slack_auth_test(token)
            report.add("slack", "ok" if ok else "fail", f"auth.test: {detail}" + (f"; DMs go to {user}" if ok else ""))
    else:
        report.add("slack", "ok", "disabled (notify.slack = false)")
    if profile.notify.desktop:
        report.add("desktop toast", *toast_backend_status())
    else:
        report.add("desktop toast", "ok", "disabled (notify.desktop = false)")


def _doctor_data_dir(profile: Profile, report: DoctorReport) -> ProfilePaths:
    paths = _paths(profile)
    probe = paths.root / ".doctor-write-probe"
    try:
        paths.ensure()
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        report.add("data dir", "ok", f"{paths.root} writable")
    except OSError as exc:
        report.add("data dir", "fail", f"{paths.root} not writable: {exc}")
    _doctor_hooks_path(report)
    return paths


def _doctor_hooks_path(report: DoctorReport) -> None:
    """Every engine and worker git runs with ``core.hooksPath`` under a regular file; without it
    no git runs at all (fail closed), so a host where it cannot be resolved can do nothing."""
    try:
        hooks = providers.unplantable_hooks_path()
    except providers.ProviderError as exc:
        report.add("git hooks", "fail", f"cannot disable git hooks safely: {exc}")
        return
    report.add("git hooks", "ok", f"disabled for engine and worker git (core.hooksPath={hooks})")


def _doctor_scheduler(profile: Profile, paths: ProfilePaths, report: DoctorReport) -> None:
    """One ``scheduler`` line, never a traceback: a scheduler the host cannot query is a warning;
    a spec that cannot be built or compared (``ValueError`` -- pydantic validation included -- or a
    ``SchedulerError`` past the query) is a failing line, so the other checks still print."""
    try:
        text = scheduler_status(profile, paths)
    except SchedulerError as exc:
        report.add("scheduler", "warn", f"could not query the OS scheduler: {exc}")
        return
    except ValueError as exc:
        report.add("scheduler", "fail", f"could not query the OS scheduler: {type(exc).__name__}: {exc}")
        return
    try:
        _doctor_scheduler_jobs(profile, paths, text, report)
    except (ValueError, SchedulerError) as exc:
        report.add("scheduler", "fail", f"installed jobs could not be evaluated: {type(exc).__name__}: {exc}")


def _doctor_scheduler_jobs(profile: Profile, paths: ProfilePaths, text: str, report: DoctorReport) -> None:
    lowered = text.lower()
    if "not installed" in lowered or not text.strip():
        report.add("scheduler", "warn", "jobs not installed (run `factory schedule install`)")
        return
    if "error" in lowered:
        report.add("scheduler", "warn", _clip(text, 200))
        return
    spec = scheduler.ScheduleSpec.from_profile(profile, paths, require_uv=False)
    drift = scheduler.install_drift(spec, text)
    if drift:
        report.add(
            "scheduler", "warn",
            "jobs installed but " + "; ".join(drift) + " -- re-run `factory schedule install` from this shell "
            f"(the jobs embed {scheduler.DATA_DIR_ENV} as set at install time)",
        )  # fmt: skip
    else:
        report.add("scheduler", "ok", "jobs installed (see `factory schedule show`)")
    if scheduler.POWER_RESTRICTED_MARKER in lowered:
        report.add(
            "scheduler power", "warn",
            "the tasks keep Task Scheduler's defaults (no start on battery, stop when going on battery); "
            "re-run `factory schedule install` to allow running on battery",
        )  # fmt: skip


def _doctor_repo(key: str, repos: RepoIndex, report: DoctorReport) -> None:
    repo = repos.get(key)
    if repo is None:
        report.add("repo", "fail", f"{key!r} is not a managed repo; managed: {', '.join(repos.slugs()) or 'none'}")
        return
    report.add("repo", "ok", f"{repo.slug} at {repo.path} (default branch {repo.default_branch}, visual {repo.override.visual})")
    verification = discover(repo.path, repo.override)
    if verification.verify:
        commands = "; ".join(f"{c.name}: {c.command}" for c in verification.verify)
        report.add("repo verification", "ok", f"{len(verification.verify)} command(s) from {verification.source}: {_clip(commands, 200)}")
    else:
        report.add("repo verification", "warn", "no verification commands discovered (add a verification section to CLAUDE.md or an override)")
    if verification.setup:
        report.add("repo setup", "ok", "; ".join(c.command for c in verification.setup))
    if verification.dev_command:
        report.add("repo dev server", "ok", f"{verification.dev_command} on port {verification.dev_port or '?'}")
    readiness = score_repo(repo, verification, run_commands=False)
    # Static runs can only earn the static weights (~50 of 100, 43 of 100 for visual repos), so the
    # verdict compares against those, not against the full score.
    static = [c for c in readiness.checks if not c.not_run]
    earned, possible = sum(c.weight for c in static if c.ok), sum(c.weight for c in static)
    report.add(
        "repo readiness (static)", "ok" if earned == possible else "warn",
        f"{earned}/{possible} static points, {sum(1 for c in static if c.ok)}/{len(static)} static checks ok "
        f"(score {readiness.score}/100 with runtime checks not run; `factory readiness --repo {repo.name}` runs them)",
    )  # fmt: skip
    for check in readiness.checks:
        report.add(f"  {check.name}", "ok" if check.ok else ("n/a" if check.not_run else "warn"), check.detail)


# ------------------------------------------------------------------------------------- readiness
@app.command()
def readiness(
    ctx: typer.Context,
    repo: Annotated[str | None, typer.Option("--repo", help="Only this managed repo (slug, name or dir name).")] = None,
    no_run: Annotated[bool, typer.Option("--no-run", help="Static checks only; do not run verification commands.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Ignore the cached score for the current HEAD.")] = False,
) -> None:
    """Readiness scores per managed repo (full runs execute the verification commands in a throwaway worktree)."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("readiness"):
        repos = discover_repos(profile)
        if repo is not None:
            found = repos.get(repo)
            if found is None:
                raise typer.BadParameter(f"{repo!r} is not a managed repo; managed: {', '.join(repos.slugs()) or 'none'}")
            targets = [found]
        else:
            targets = repos.all()
        if not targets:
            fail("no managed repos found; check repos.root / github.owners in the profile")
        paths.ensure()
        wt_manager = WorktreeManager(paths.worktrees)
        with State(paths.state_db) as state:
            results = _readiness_rows(state, targets, wt_manager, run_commands=not no_run, force=force)
        if no_run:
            typer.echo("static checks only (runtime checks not run; scores cap around 50). Drop --no-run for a full run.")
        if len(targets) == 1 and results:
            typer.echo("")
            for check in results[0].checks:
                mark = "ok  " if check.ok else ("n/a " if check.not_run else "FAIL")
                typer.echo(f"  {mark} {check.name}: {check.detail}")


def _readiness_rows(
    state: State, targets: list[RepoInfo], wt_manager: WorktreeManager, *, run_commands: bool, force: bool
) -> list[Readiness]:
    width = max(len(r.slug) for r in targets)
    typer.echo(f"{'repo'.ljust(width)}  score  missing / flaky / runtime")
    results: list[Readiness] = []
    for repo in targets:
        try:
            result = readiness_cached(state, repo, run_commands=run_commands, force=force, wt_manager=wt_manager)
        except Exception as exc:  # noqa: BLE001 - one repo must not stop the table
            typer.echo(f"{repo.slug.ljust(width)}    n/a  error: {_clip(f'{type(exc).__name__}: {exc}', 120)}")
            continue
        results.append(result)
        extras = []
        if result.missing:
            extras.append("missing " + ", ".join(result.missing))
        if result.flaky:
            extras.append("flaky " + ", ".join(result.flaky))
        if result.runtime_s:
            extras.append(f"{result.runtime_s:.0f}s")
        typer.echo(f"{repo.slug.ljust(width)}  {result.score:>5}  {'; '.join(extras) or 'all checks ok'}")
    return results


# ----------------------------------------------------------------------------------------- ready
@app.command()
def ready(
    ctx: typer.Context,
    repo: Annotated[str, typer.Argument(help="Managed repo (slug, name or dir name).")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Build the action and print it; start no worker.")] = False,
) -> None:
    """Improve a repo's agent readiness: worker commits on factory/ready-<date>, engine pushes + opens a draft PR. Exit codes as `run`."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("ready"):
        deps = build_deps(profile, paths, dry_run=dry_run)
        try:
            action = ready_action(deps, repo.strip())
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from None
        typer.echo(f"action: {action.describe()} (version {action.version})")
        outcome = asyncio.run(
            dispatch.run_one(profile, paths, action, dry_run=dry_run, deps=deps, print_fn=typer.echo, force=True)
        )
    raise typer.Exit(_outcome_exit_code(outcome))


# ---------------------------------------------------------------------------------------- report
@app.command()
def report(
    ctx: typer.Context,
    since: Annotated[str, typer.Option("--since", help="Window: 7d, 36h, 2w or an ISO timestamp.")] = "7d",
    digest: Annotated[bool, typer.Option("--digest", help="Also send the compact digest as a Slack DM to yourself.")] = False,
) -> None:
    """Outcome report: runs per kind, merged/closed/open PRs, rework, reverts, cost, top flag reasons."""
    try:
        since_iso = metrics.since_iso_from_spec(since)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--since") from None
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("report"), State(paths.state_db) as state:
        try:
            refreshed = metrics.refresh_outcomes(state, gh_client(profile))
            typer.echo(f"refreshed {refreshed} factory-touched PR(s)")
        except Exception as exc:  # noqa: BLE001 - the report still renders from stored outcomes
            typer.echo(f"warning: PR outcome refresh skipped ({type(exc).__name__}: {exc})", err=True)
        result = metrics.build_report(state, since_iso)
        typer.echo(metrics.render_report(result), nl=False)
        if digest:
            paths.ensure()
            notifier = Notifier(profile, Audit(paths.audit_log, profile.name))
            if notifier.slack_dm(metrics.digest_text(result)):
                typer.echo("digest sent as a Slack DM")
            else:
                fail("digest not sent: the Slack DM failed (check notify.slack, SLACK_BOT_TOKEN and SLACK_USER_ID)")


# --------------------------------------------------------------------------------- revert-refine
@app.command("revert-refine")
def revert_refine_cmd(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Issue key the factory refined, e.g. ABC-123.")],
) -> None:
    """Restore the original title/description of a refined issue (the refine label is not re-added)."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("revert-refine"):
        deps = build_deps(profile, paths, dry_run=False)
        outcome = asyncio.run(revert_refine(deps, key.strip()))
    typer.echo(f"revert-refine {key} {outcome.status} {outcome.note}".rstrip())
    raise typer.Exit(EXIT_FAIL if outcome.status == "failed" else EXIT_OK)


# -------------------------------------------------------------------------------------- schedule
@schedule_app.command("install")
def schedule_install(
    ctx: typer.Context,
    interval: Annotated[int | None, typer.Option("--interval", help="Tick interval in minutes (default: schedule.interval_minutes).")] = None,
    digest: Annotated[bool, typer.Option("--digest/--no-digest", help="Also install the weekly digest job.")] = True,
) -> None:
    """Install the tick job (and the weekly digest) in the OS scheduler for this profile."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("schedule install"):
        paths.ensure()
        report = scheduler.install(profile, paths, interval_minutes=interval, digest=digest)
    typer.echo(report.summary())


@schedule_app.command("uninstall")
def schedule_uninstall(ctx: typer.Context) -> None:
    """Remove this profile's scheduler jobs and wrapper files."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("schedule uninstall"):
        report = scheduler.uninstall(profile, paths)
    typer.echo(report.summary())


@schedule_app.command("show")
def schedule_show(ctx: typer.Context) -> None:
    """Show this profile's scheduler jobs as the OS reports them."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("schedule show"):
        typer.echo(scheduler_status(profile, paths))


# --------------------------------------------------------------------------- cleanup / release
@app.command()
def cleanup(
    ctx: typer.Context,
    all_: Annotated[bool, typer.Option("--all", help="Also release and remove the kept worktrees (asks first).")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before removing kept worktrees.")] = False,
) -> None:
    """Remove leftover worktrees that are not kept for you; --all also releases and removes the kept ones."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("cleanup"), _tick_idle(paths), State(paths.state_db) as state:
        manager = WorktreeManager(paths.worktrees)
        kept = state.kept_worktrees()
        removed = manager.cleanup({k["path"] for k in kept})
        for path in removed:
            typer.echo(f"removed {path}")
        typer.echo(f"removed {len(removed)} unkept worktree(s); {len(kept)} kept")
        if not all_ or not kept:
            return
        for record in kept:
            typer.echo(f"kept: {record['target']} ({record['kind']}) {record['path']} -- {_clip(record.get('reason') or '', 80)}")
        if not yes:
            typer.confirm(f"Release and remove these {len(kept)} kept worktree(s)?", abort=True)
        for record in kept:
            state.release_worktree(record["target"])
        removed = manager.cleanup(set())
        for path in removed:
            typer.echo(f"removed {path}")
        typer.echo(f"released {len(kept)} kept worktree(s); removed {len(removed)}")


@app.command()
def release(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(help="Kept-worktree target, e.g. owner/name#123 or ABC-123.")],
    remove: Annotated[bool, typer.Option("--remove", help="Also delete the worktree directory.")] = False,
) -> None:
    """Release a kept worktree so the target can be dispatched again; --remove also deletes the directory."""
    profile = _profile(ctx)
    paths = _paths(profile)
    with _known_errors("release"), State(paths.state_db) as state:
        record = state.kept_worktree(target.strip())
        if record is None:
            fail(f"no kept worktree for {target!r} (see `factory status`)")
        state.release_worktree(record["target"])
        typer.echo(f"released {record['target']} ({record['kind']}): {record['path']}")
        if not remove:
            return
        with _tick_idle(paths):
            path = Path(record["path"])
            if not path.exists():
                typer.echo("directory already gone")
                return
            root = paths.worktrees
            if not root.is_dir() or path.resolve().parent != root.resolve():
                fail(f"{path} is not under the worktree root {root}; remove it by hand")
            keep = {str(child) for child in root.iterdir() if child.resolve() != path.resolve()}
            removed = WorktreeManager(root).cleanup(keep)
            typer.echo(f"removed {removed[0]}" if removed else f"nothing removed at {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
