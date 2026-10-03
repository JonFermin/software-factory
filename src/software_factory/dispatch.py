"""Dispatcher: one tick = lock -> precheck -> scan -> filter -> dispatch -> record.

Deterministic, no LLM. Groups (one per target) run concurrently under
``asyncio.Semaphore(max_parallel)``; inside a group actions run strictly sequentially in
``ci -> conflict -> rc -> rv`` order (they share a branch, and a push by ``ci`` makes the later
ones stale -- the per-action PR re-check skips them, and a worktree kept by an earlier action
skips the rest of the group). State is recorded per invariant 4; auth failures (precheck or
mid-wave) flag once, stop launching, spend no attempt and exit 2, and a provider that cannot serve
sessions (rate limit, overloaded: ``ProviderUnavailable``) is treated the same way under its own
once-a-day flag. The flags the scan attaches to its skips (attempts exhausted) are raised here, off
the event loop. ``--dry-run`` scans, filters and prints and never starts a worker or writes state:
it needs no provider credentials (a missing token is printed as a warning), an auth failure is
printed, not flagged, and skip flags are not raised. In ``mode = "observe"`` only ``rv`` runs for
real; every other kind is recorded as ``dry_run`` ("would run").
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from pydantic import BaseModel, Field

from . import metrics, providers
from . import scan as scan_mod
from .actions import ISSUE_KINDS, OBSERVE_KINDS, Action, Kind, group_actions
from .config import Profile
from .kinds import get_handler
from .kinds.base import AuthFailure, Deps, Outcome, ProviderUnavailable, flag
from .paths import ProfilePaths
from .scan import ScanResult, Skip, skip_reason
from .sources.github import GhAuthError, GhError
from .trackers.base import TrackerAuthError

EXIT_OK, EXIT_FAILED, EXIT_AUTH = 0, 1, 2
ATTEMPT_OUTCOMES = ("failed", "kept", "flagged")
# Exceptions that stop the wave without an attempt: auth failures (invariant 5) and a provider
# that cannot serve sessions. Handlers let them propagate; ``_run_action`` never turns them into
# a ``failed`` outcome.
AUTH_ERRORS: tuple[type[BaseException], ...] = (AuthFailure, GhAuthError, TrackerAuthError)
WAVE_STOPPERS: tuple[type[BaseException], ...] = (*AUTH_ERRORS, ProviderUnavailable)
PrintFn = Callable[[str], Any]
_GH_TOKEN_VARS = ("GH_TOKEN", "GITHUB_TOKEN")
GH_TOKEN_HINT = (
    "gh was authenticated only through GH_TOKEN/GITHUB_TOKEN, which the engine scrubs from its "
    "environment before running (workers must not inherit it); log gh in with `gh auth login` "
    "(keyring) for the factory to use it"
)


class RunRecord(BaseModel):
    action: Action
    outcome: Outcome


class TickReport(BaseModel):
    """What a tick did. ``exit_code``: 0 ok, 1 some action failed, 2 auth failure or provider
    unavailable (``auth_error`` / ``provider_error`` say which)."""

    lines: list[str] = Field(default_factory=list)
    outcomes: list[RunRecord] = Field(default_factory=list)
    skipped: list[Skip] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    auth_error: str | None = None
    provider_error: str | None = None
    locked: bool = False
    dry_run: bool = False
    exit_code: int = EXIT_OK

    def say(self, print_fn: PrintFn, line: str) -> None:
        self.lines.append(line)
        print_fn(line)


# ------------------------------------------------------------------------------------- tick
async def run_tick(
    profile: Profile,
    paths: ProfilePaths,
    *,
    dry_run: bool = False,
    deps: Deps | None = None,
    print_fn: PrintFn = print,
) -> TickReport:
    report = TickReport(dry_run=dry_run)
    paths.ensure()
    lock = FileLock(str(paths.lock_file), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        report.locked = True
        report.say(print_fn, "tick already running")
        return report
    try:
        gh_token_scrubbed = _scrub_env()
        deps = deps or Deps.build(profile, paths, dry_run=dry_run)
        deps.audit.write("tick.start", run_id=deps.run_id, dry_run=dry_run, mode=profile.mode)
        result = await _precheck_and_scan(profile, deps, report, print_fn, dry_run=dry_run, gh_token_scrubbed=gh_token_scrubbed)
        if result is None:
            return _finish(deps, report, print_fn)
        report.skipped = list(result.skipped)
        report.errors = list(result.errors)
        for error in result.errors:
            report.say(print_fn, f"scan error: {error}")
        for skip in result.skipped:
            report.say(print_fn, f"skip {skip.action.kind.value} {skip.action.target} {skip.reason}")
        if dry_run:
            for action in result.actions:
                outcome = Outcome(status="dry_run", note="would run (--dry-run)")
                report.outcomes.append(RunRecord(action=action, outcome=outcome))
                report.say(print_fn, _line(action, outcome))
            return _finish(deps, report, print_fn)
        await asyncio.to_thread(_raise_skip_flags, result.skipped, deps)
        await _dispatch(result.actions, deps, report, print_fn)
        if report.auth_error is None:  # after an auth failure every gh read would fail too
            try:
                await asyncio.to_thread(metrics.refresh_outcomes, deps.state, deps.gh)
            except Exception as exc:  # noqa: BLE001 - metrics never fail a tick
                report.say(print_fn, f"metrics refresh failed: {type(exc).__name__}: {exc}")
                deps.audit.write("metrics.refresh_failed", error=f"{type(exc).__name__}: {exc}")
        return _finish(deps, report, print_fn)
    finally:
        lock.release()


async def run_one(
    profile: Profile,
    paths: ProfilePaths,
    action: Action,
    *,
    dry_run: bool = False,
    deps: Deps | None = None,
    print_fn: PrintFn = print,
    force: bool = False,
) -> Outcome:
    """``factory run <kind> <target>``: the same prechecks and recording as a tick, one action.

    The scan filters still apply (``force`` bypasses them); the dispatcher's PR re-check does too.
    """
    paths.ensure()
    lock = FileLock(str(paths.lock_file), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        print_fn("tick already running")
        return Outcome(status="skipped", note="tick already running")
    try:
        gh_token_scrubbed = _scrub_env()
        deps = deps or Deps.build(profile, paths, dry_run=dry_run)
        report = TickReport(dry_run=dry_run)
        if dry_run:
            _warn_provider(profile, report, print_fn)
        failure = await asyncio.to_thread(
            _precheck, profile, deps, needs_tracker=action.kind in ISSUE_KINDS, check_provider=not dry_run,
            gh_token_scrubbed=gh_token_scrubbed,
        )  # fmt: skip
        if failure is not None:
            await asyncio.to_thread(_auth_failure, deps, report, print_fn, *failure, dry_run=dry_run)
            return Outcome(status="failed", note=f"auth failure ({failure[0]}): {failure[1]}")
        if not force:
            reason = skip_reason(profile, action, deps.state, deps.tracker)
            if reason is not None:
                outcome = Outcome(status="skipped", note=reason)
                print_fn(_line(action, outcome))
                return outcome
        if dry_run:
            outcome = Outcome(status="dry_run", note="would run (--dry-run)")
            print_fn(_line(action, outcome))
            return outcome
        kept = _kept_reason(action, deps)
        if kept is not None:  # even with force: running would overwrite the human handoff record
            outcome = Outcome(status="skipped", note=f"{kept}; `factory release {action.target}` first")
            print_fn(_line(action, outcome))
            return outcome
        try:
            outcome = await _run_action(action, deps)
        except AUTH_ERRORS as exc:
            await asyncio.to_thread(_auth_failure, deps, report, print_fn, _auth_source(exc), str(exc))
            return Outcome(status="failed", note=f"auth failure: {exc}")
        except ProviderUnavailable as exc:
            await asyncio.to_thread(_provider_unavailable, deps, report, print_fn, str(exc))
            return Outcome(status="failed", note=f"provider unavailable: {exc}")
        _record(action, outcome, deps, report, print_fn)
        return outcome
    finally:
        lock.release()


# ------------------------------------------------------------------------------- precheck
def _scrub_env() -> bool:
    """``providers.scrub_process_env``; returns whether a gh token was present before and gone
    after it (the hint for a ``gh`` auth failure that follows: ``doctor`` does not scrub, so it
    passed). False when the scrub leaves the engine's gh token alone."""
    had_token = any(os.environ.get(name) for name in _GH_TOKEN_VARS)
    providers.scrub_process_env()
    return had_token and not any(os.environ.get(name) for name in _GH_TOKEN_VARS)


async def _precheck_and_scan(
    profile: Profile,
    deps: Deps,
    report: TickReport,
    print_fn: PrintFn,
    *,
    dry_run: bool = False,
    gh_token_scrubbed: bool = False,
) -> ScanResult | None:
    needs_tracker = any(k in ISSUE_KINDS for k in profile.enabled_kinds)
    if dry_run:
        _warn_provider(profile, report, print_fn)
    failure = await asyncio.to_thread(
        _precheck, profile, deps, needs_tracker=needs_tracker, check_provider=not dry_run,
        gh_token_scrubbed=gh_token_scrubbed,
    )  # fmt: skip
    if failure is not None:
        await asyncio.to_thread(_auth_failure, deps, report, print_fn, *failure, dry_run=dry_run)
        return None
    try:
        return await asyncio.to_thread(scan_sources, profile, deps)
    except (GhAuthError, TrackerAuthError, AuthFailure) as exc:
        await asyncio.to_thread(_auth_failure, deps, report, print_fn, _auth_source(exc), str(exc), dry_run=dry_run)
        return None


def _warn_provider(profile: Profile, report: TickReport, print_fn: PrintFn) -> None:
    """``--dry-run`` starts no worker, so provider credentials are reported, never required."""
    problems = providers.check_credentials(profile)
    if problems:
        report.say(print_fn, "warning: provider credentials: " + "; ".join(problems) + " (a real tick would stop here)")


def _precheck(
    profile: Profile,
    deps: Deps,
    *,
    needs_tracker: bool,
    check_provider: bool = True,
    gh_token_scrubbed: bool = False,
) -> tuple[str, str] | None:
    """``(source, message)`` of the first auth problem, else ``None``. Order: provider creds
    (skipped for ``--dry-run``), ``gh auth``, tracker ping (only when an issue kind is enabled and
    Linear is enabled)."""
    if check_provider:
        problems = providers.check_credentials(profile)
        if problems:
            return "provider", "; ".join(problems)
    try:
        deps.gh.auth_check()
    except GhAuthError as exc:
        return "gh", f"{exc} ({GH_TOKEN_HINT})" if gh_token_scrubbed else str(exc)
    except GhError as exc:
        return "gh", f"gh is not usable: {exc}"
    if needs_tracker and profile.linear.enabled:
        if deps.tracker is None:
            return "tracker", "LINEAR_API_KEY missing from the profile .env (linear.enabled with issue kinds)"
        try:
            deps.tracker.ping()
        except TrackerAuthError as exc:
            return "tracker", str(exc)
    return None


def scan_sources(profile: Profile, deps: Deps) -> ScanResult:
    """``scan.scan`` over the deps (a seam for tests)."""
    tracker = deps.tracker if profile.linear.enabled else None
    return scan_mod.scan(profile, gh=deps.gh, tracker=tracker, repos=deps.repos, state=deps.state)


def _auth_source(exc: BaseException) -> str:
    if isinstance(exc, GhAuthError):
        return "gh"
    if isinstance(exc, TrackerAuthError):
        return "tracker"
    return "worker"


def _auth_failure(
    deps: Deps, report: TickReport, print_fn: PrintFn, source: str, message: str, *, dry_run: bool = False
) -> None:
    """Flag once per source and day (dedupe ``auth:<source>:<date>``), exit code 2. A dry run
    only prints it (no flag, no notification: it writes no state). Blocking (notifier): call it
    through ``asyncio.to_thread`` from coroutine context."""
    report.auth_error = f"{source}: {message}"
    report.exit_code = EXIT_AUTH
    flag_id: int | None = None
    if not dry_run:
        date = deps.now().date().isoformat()
        target = f"profile:{deps.profile.name}"
        flag_id = deps.state.add_flag(target, "auth", f"auth failure: {source}", message, dedupe_key=f"auth:{source}:{date}")
        if flag_id is not None:
            deps.notifier.notify(f"auth failure: {source}", message, target=target, kind="auth", level="error")
    deps.audit.write("auth_failure", source=source, error=message, flagged=flag_id is not None, dry_run=dry_run)
    report.say(print_fn, f"auth failure ({source}): {message}")


def _provider_unavailable(deps: Deps, report: TickReport, print_fn: PrintFn, message: str) -> None:
    """The model provider could not serve a session (``ProviderUnavailable``): exit code 2, flag
    once per day (dedupe ``provider:<date>``), no attempt. Like ``_auth_failure`` it blocks on the
    notifier, so call it through ``asyncio.to_thread``. Unlike one, it does not mean gh is down."""
    report.provider_error = message
    report.exit_code = EXIT_AUTH
    date = deps.now().date().isoformat()
    target = f"profile:{deps.profile.name}"
    flag_id = deps.state.add_flag(target, "provider", "provider unavailable", message, dedupe_key=f"provider:{date}")
    if flag_id is not None:
        deps.notifier.notify("provider unavailable", message, target=target, kind="provider", level="error")
    deps.audit.write("provider_unavailable", error=message, flagged=flag_id is not None)
    report.say(print_fn, f"provider unavailable: {message}")


def _raise_skip_flags(skipped: list[Skip], deps: Deps) -> None:
    """Raise the flag the scan attached to a skip (attempts exhausted): ``state.add_flag`` +
    notify, flag-once by the spec's ``dedupe_key``. The scan itself writes nothing, so a dry run
    (which never gets here) stays pure. A flag that cannot be written is audited and never stops
    the tick. Blocking (notifier): call it through ``asyncio.to_thread``."""
    for skip in skipped:
        spec = skip.flag
        if spec is None:
            continue
        try:
            flag(deps, skip.action, spec.reason, spec.detail, level=spec.level, dedupe_key=spec.dedupe_key)
        except Exception as exc:  # noqa: BLE001 - a skip flag must never take the tick down
            deps.audit.write(
                "skip_flag.failed", kind=skip.action.kind.value, target=skip.action.target,
                error=f"{type(exc).__name__}: {exc}",
            )  # fmt: skip


def _kept_reason(action: Action, deps: Deps) -> str | None:
    """A kept worktree for the target (a human handoff) blocks every further action on it until
    ``factory release``: running one would overwrite the record and orphan the kept checkout."""
    kept = deps.state.kept_worktree(action.target)
    if kept is None:
        return None
    detail = f": {kept.get('reason')}" if kept.get("reason") else ""
    return f"kept worktree awaiting me ({kept.get('kind')}){detail}"


# ------------------------------------------------------------------------------- dispatch
async def _dispatch(actions: list[Action], deps: Deps, report: TickReport, print_fn: PrintFn) -> None:
    groups = group_actions(actions)
    if not groups:
        report.say(print_fn, "nothing to do")
        return
    semaphore = asyncio.Semaphore(max(1, deps.profile.max_parallel))
    stop = asyncio.Event()

    async def run_group(group: list[Action]) -> None:
        async with semaphore:
            for action in group:
                if stop.is_set():
                    return
                kept = await asyncio.to_thread(_kept_reason, action, deps)
                if kept is not None:  # an earlier action of this group (or tick) kept its worktree
                    _record(action, Outcome(status="skipped", note=kept), deps, report, print_fn)
                    continue
                try:
                    outcome = await _run_action(action, deps)
                except AUTH_ERRORS as exc:
                    stop.set()
                    await asyncio.to_thread(_auth_failure, deps, report, print_fn, _auth_source(exc), str(exc))
                    return
                except ProviderUnavailable as exc:
                    stop.set()  # in-flight groups finish; nothing new starts this tick
                    await asyncio.to_thread(_provider_unavailable, deps, report, print_fn, str(exc))
                    return
                _record(action, outcome, deps, report, print_fn)
                if stop.is_set():
                    return

    await asyncio.gather(*(run_group(group) for group in groups.values()))


async def _run_action(action: Action, deps: Deps) -> Outcome:
    """Observe-mode gate, PR re-check, handler (exceptions -> ``failed`` + flag)."""
    profile = deps.profile
    if profile.mode == "observe" and action.kind not in OBSERVE_KINDS:
        deps.audit.write(
            "would_run", kind=action.kind.value, target=action.target, version=action.version,
            title=action.title, reason="observe mode",
        )  # fmt: skip
        return Outcome(status="dry_run", note="observe mode: would run")
    if action.pr is not None and action.repo:
        try:
            reason = await asyncio.to_thread(_pr_recheck, action, deps)
        except GhAuthError:
            raise
        except GhError as exc:
            return Outcome(status="failed", note=f"pr re-check failed: {exc}")
        if reason is not None:
            return Outcome(status="skipped", note=reason)
    try:
        handler = get_handler(action.kind)
        outcome = await handler.run(action, deps)
    except WAVE_STOPPERS:
        raise
    except Exception as exc:  # noqa: BLE001 - a handler bug must never take the tick down
        summary = _traceback_summary(exc)
        deps.audit.write("handler.crashed", kind=action.kind.value, target=action.target, error=summary)
        try:
            await asyncio.to_thread(_crash_flag, action, deps, summary)
        except Exception as flag_exc:  # noqa: BLE001
            deps.audit.write("flag.failed", target=action.target, error=f"{type(flag_exc).__name__}: {flag_exc}")
        return Outcome(status="failed", note=f"{type(exc).__name__}: {exc}"[:500])
    if not isinstance(outcome, Outcome):
        return Outcome(status="failed", note=f"handler returned {type(outcome).__name__}, not Outcome")
    return outcome


def _crash_flag(action: Action, deps: Deps, summary: str) -> None:
    flag_id = deps.state.add_flag(
        action.target, action.kind.value, "handler crashed", summary,
        dedupe_key=f"crash:{action.target}:{action.kind.value}:{action.version}",
    )  # fmt: skip
    if flag_id is not None:
        deps.notifier.notify("handler crashed", summary, target=action.target, kind=action.kind.value, level="error")


def _pr_recheck(action: Action, deps: Deps) -> str | None:
    """Re-fetch the PR right before running: closed/merged or a changed version -> skip reason."""
    assert action.repo is not None and action.pr is not None
    pr = deps.gh.pr_state(action.repo, action.pr)
    if pr.state != "OPEN":
        return f"pr {pr.state.lower()}"
    if action.kind in (Kind.CI, Kind.RV) and pr.head_ref_oid and pr.head_ref_oid != action.version:
        return "stale: head moved"
    if action.kind == Kind.CONFLICT and pr.head_ref_oid and f"{pr.head_ref_oid}:{pr.base_ref_oid}" != action.version:
        return "stale: head or base moved"
    return None


def _traceback_summary(exc: BaseException, limit: int = 2000) -> str:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    return text if len(text) <= limit else "..." + text[-limit:]


# -------------------------------------------------------------------------------- recording
def _record(action: Action, outcome: Outcome, deps: Deps, report: TickReport, print_fn: PrintFn) -> None:
    target, kind, version = action.dedupe_key
    if outcome.record_handled:
        deps.state.mark_handled(target, kind, version, outcome.status, outcome.note)
    elif outcome.status in ATTEMPT_OUTCOMES:
        deps.state.record_attempt(target, kind, version, outcome.status, outcome.note)
    record = {
        "ts": deps.now().isoformat(timespec="seconds"),
        "profile": deps.profile.name,
        "run_id": deps.run_id,
        "kind": kind,
        "target": target,
        "version": version,
        "outcome": outcome.status,
        "pushed": outcome.pushed,
        "cost_usd": outcome.cost_usd,
        "usage": outcome.usage,
        "duration_ms": outcome.duration_ms,
        "note": outcome.note,
        "pr_url": outcome.pr_url,
        "url": action.url,
    }
    _append_jsonl(deps.paths.runs_log, record)
    deps.state.record_run(record)
    if outcome.status == "failed":
        report.exit_code = max(report.exit_code, EXIT_FAILED)
    report.outcomes.append(RunRecord(action=action, outcome=outcome))
    report.say(print_fn, _line(action, outcome))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def _line(action: Action, outcome: Outcome) -> str:
    """``kind target outcome pushed note`` -- one line per action."""
    note = " ".join(outcome.note.split())
    return f"{action.kind.value} {action.target} {outcome.status} {'pushed' if outcome.pushed else 'not-pushed'} {note}".rstrip()


def _finish(deps: Deps, report: TickReport, print_fn: PrintFn) -> TickReport:
    counts: dict[str, int] = {}
    for rec in report.outcomes:
        counts[rec.outcome.status] = counts.get(rec.outcome.status, 0) + 1
    deps.audit.write(
        "tick.end", run_id=deps.run_id, dry_run=report.dry_run, exit_code=report.exit_code, outcomes=counts,
        skipped=len(report.skipped), errors=len(report.errors), auth_error=report.auth_error,
        provider_error=report.provider_error,
    )  # fmt: skip
    return report
