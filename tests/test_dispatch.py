"""Dispatcher tests with fake handlers, a fake GitHub and a scripted scan.

Ordering/parallelism, the non-blocking lock, auth prechecks and mid-wave auth failures, observe
mode, ``--dry-run``, state recording per outcome, handler crashes and the run log.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock

from software_factory import dispatch, metrics
from software_factory.actions import Action, Kind
from software_factory.config import Profile
from software_factory.effects import RecordingEffects, make_effects
from software_factory.kinds.base import AuthFailure, Deps, Outcome, ProviderUnavailable
from software_factory.notify import Audit, Notifier
from software_factory.paths import ProfilePaths
from software_factory.repos import RepoIndex
from software_factory.scan import EXHAUSTED_REASON, ScanResult, Skip, exhausted_flag
from software_factory.sources.github import GhAuthError, GhError
from software_factory.sources.models import PullRequest
from software_factory.state import State
from software_factory.trackers.base import TrackerAuthError
from software_factory.worktree import WorktreeManager

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
HEAD, BASE = "h1", "b1"


# ------------------------------------------------------------------------------------- fakes
class FakeGh:
    def __init__(self, prs: dict[tuple[str, int], PullRequest] | None = None, *, auth_error: str | None = None) -> None:
        self.prs = prs or {}
        self.auth_error = auth_error
        self.calls: list[tuple[str, Any]] = []
        self.auth_checks = 0

    def auth_check(self) -> None:
        self.auth_checks += 1
        if self.auth_error:
            raise GhAuthError(self.auth_error)

    def login(self) -> str:
        return "me"

    def pr_state(self, repo: str, number: int) -> PullRequest:
        self.calls.append(("pr_state", repo, number))
        pr = self.prs.get((repo, number))
        if pr is None:
            raise GhError(f"{repo}#{number} not found")
        return pr

    def pr_commits(self, repo: str, number: int) -> list:
        self.calls.append(("pr_commits", repo, number))
        return []


class FakeTracker:
    name = "fake"

    def __init__(self, *, auth_error: str | None = None) -> None:
        self.auth_error = auth_error
        self.pings = 0

    def ping(self) -> str:
        self.pings += 1
        if self.auth_error:
            raise TrackerAuthError(self.auth_error)
        return "me"


class FakeHandler:
    def __init__(self, kind: Kind, fn: Callable[..., Any]) -> None:
        self.kind = kind
        self.fn = fn
        self.calls: list[Action] = []

    async def run(self, action: Action, deps: Deps) -> Outcome:
        self.calls.append(action)
        result = self.fn(action, deps)
        if asyncio.iscoroutine(result):
            result = await result
        return result


def open_pr(repo: str, number: int, *, head: str = HEAD, base: str = BASE, state: str = "OPEN") -> PullRequest:
    return PullRequest(repo=repo, number=number, state=state, head_ref_oid=head, base_ref_oid=base, head_ref_name="f", base_ref_name="main")


def pr_action(kind: Kind, number: int, *, repo: str = "o/r", version: str | None = None) -> Action:
    default = {Kind.CI: HEAD, Kind.CONFLICT: f"{HEAD}:{BASE}", Kind.RC: "2026-01-01T00:00:00Z", Kind.RV: HEAD}[kind]
    return Action(kind=kind, version=version if version is not None else default, repo=repo, pr=number, title=f"{kind.value} {number}")


# ---------------------------------------------------------------------------------- fixtures
@pytest.fixture
def paths(tmp_path: Path) -> ProfilePaths:
    return ProfilePaths.for_profile("personal", root=tmp_path / "data root").ensure()


@pytest.fixture
def build_profile(make_profile: Callable[..., Path], load: Callable[[str], Profile]) -> Callable[..., Profile]:
    def _build(*, mode: str = "full", max_parallel: int = 3, kinds: list[str] | None = None, token: bool = True,
               linear_key: bool = False, **profile_fields: Any) -> Profile:
        toml: dict[str, Any] = {
            "profile": {"mode": mode, "max_parallel": max_parallel, **profile_fields},
            "notify": {"desktop": False, "slack": False},
            "github": {"owners": ["o"], "login": "me"},
        }
        if kinds is not None:
            toml["profile"]["enabled_kinds"] = kinds
        env: dict[str, str] = {}
        if token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = "sk-ant-oat-test"
        if linear_key:
            env["LINEAR_API_KEY"] = "lin_api_test"
        make_profile("personal", toml, env)
        return load("personal")  # fmt: skip

    return _build


@pytest.fixture
def make_deps(paths: ProfilePaths) -> Callable[..., Deps]:
    def _make(profile: Profile, *, gh: FakeGh | None = None, tracker: Any = None, now: datetime = NOW) -> Deps:
        state = State(paths.state_db)
        audit = Audit(paths.audit_log, profile.name)
        gh = gh or FakeGh()
        effects = make_effects(profile, dry_run=False, paths=paths, state=state, gh=gh, tracker=tracker, audit=audit)  # type: ignore[arg-type]
        return Deps(
            profile=profile, paths=paths, state=state, gh=gh, tracker=tracker, repos=RepoIndex([]),  # type: ignore[arg-type]
            effects=effects, notifier=Notifier(profile, audit), audit=audit,
            wt_manager=WorktreeManager(paths.worktrees), resolver=None,
            run_worker=_no_worker, now=lambda: now, run_id="run-1",
        )  # fmt: skip

    return _make


async def _no_worker(**_: Any) -> Any:
    raise AssertionError("a real worker must never be started by these tests")


@pytest.fixture
def handlers(monkeypatch: pytest.MonkeyPatch) -> dict[Kind, FakeHandler]:
    registry: dict[Kind, FakeHandler] = {}

    def get_handler(kind: Kind | str) -> FakeHandler:
        key = kind if isinstance(kind, Kind) else Kind(kind)
        if key not in registry:
            raise AssertionError(f"handler for {key} should not have been requested")
        return registry[key]

    monkeypatch.setattr(dispatch, "get_handler", get_handler)
    return registry


@pytest.fixture
def scanned(monkeypatch: pytest.MonkeyPatch) -> Callable[..., ScanResult]:
    def _set(actions: list[Action], *, skipped: list[Skip] | None = None, errors: list[str] | None = None) -> ScanResult:
        result = ScanResult(actions=actions, skipped=skipped or [], errors=errors or [])
        calls: list[Any] = []

        def scan_sources(profile: Profile, deps: Deps) -> ScanResult:
            calls.append(profile)
            return result

        monkeypatch.setattr(dispatch, "scan_sources", scan_sources)
        _set.calls = calls  # type: ignore[attr-defined]
        return result

    return _set


def done(_action: Action, _deps: Deps) -> Outcome:
    return Outcome(status="done", note="ok")


def run(profile: Profile, paths: ProfilePaths, deps: Deps, *, dry_run: bool = False) -> dispatch.TickReport:
    lines: list[str] = []
    report = asyncio.run(dispatch.run_tick(profile, paths, dry_run=dry_run, deps=deps, print_fn=lines.append))
    assert report.lines == lines  # every printed line is also in the report
    return report


# ------------------------------------------------------------------- ordering / parallelism
def test_sequential_within_group_and_parallel_across_groups_bounded(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult]
) -> None:
    profile = build_profile(max_parallel=2)
    gh = FakeGh({("o/r", n): open_pr("o/r", n) for n in (1, 2, 3)})
    deps = make_deps(profile, gh=gh)
    events: list[tuple[str, str, str]] = []
    active = {"now": 0, "max": 0}

    async def slow(action: Action, _deps: Deps) -> Outcome:
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        events.append(("start", action.target, action.kind.value))
        await asyncio.sleep(0.02)
        events.append(("end", action.target, action.kind.value))
        active["now"] -= 1
        return Outcome(status="done", note="ok")

    for kind in (Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV):
        handlers[kind] = FakeHandler(kind, slow)
    # scrambled on purpose: the dispatcher must impose ci -> conflict -> rc -> rv per PR
    actions = [pr_action(k, n) for n in (3, 1, 2) for k in (Kind.RV, Kind.RC, Kind.CONFLICT, Kind.CI)]
    scanned(actions)

    report = run(profile, deps.paths, deps)

    assert report.exit_code == 0 and len(report.outcomes) == 12
    for target in ("o/r#1", "o/r#2", "o/r#3"):
        starts = [kind for phase, t, kind in events if phase == "start" and t == target]
        assert starts == ["ci", "conflict", "rc", "rv"], target
        # strictly sequential inside the group: every end precedes the next start
        seq = [(phase, kind) for phase, t, kind in events if t == target]
        assert seq == [p for kind in ("ci", "conflict", "rc", "rv") for p in (("start", kind), ("end", kind))]
    assert active["max"] == 2  # parallel across groups, bounded by max_parallel
    for action in actions:
        assert deps.state.is_handled(*action.dedupe_key)
    assert [line.split()[2] for line in report.lines if line.startswith(("ci", "conflict", "rc", "rv"))] == ["done"] * 12


# ------------------------------------------------------------------------------------- lock
def test_lock_held_prints_and_exits_zero(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], scanned: Callable[..., ScanResult], paths: ProfilePaths
) -> None:
    profile = build_profile()
    deps = make_deps(profile)
    result = scanned([pr_action(Kind.CI, 1)])
    holder = FileLock(str(paths.lock_file))
    holder.acquire()
    try:
        report = run(profile, paths, deps)
    finally:
        holder.release()
    assert report.locked and report.exit_code == 0 and report.lines == ["tick already running"]
    assert result is not None and getattr(scanned, "calls", []) == []  # scan never ran
    assert deps.state.runs() == []
    # the lock is released after a normal tick
    run(profile, paths, deps)
    FileLock(str(paths.lock_file), timeout=0).acquire().lock.release()


# --------------------------------------------------------------------------------- precheck
def test_precheck_provider_credentials_flag_once_exit_2_no_attempts(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult]
) -> None:
    profile = build_profile(token=False)
    deps = make_deps(profile)
    handlers[Kind.CI] = FakeHandler(Kind.CI, done)
    scanned([pr_action(Kind.CI, 1)])

    first = run(profile, deps.paths, deps)
    second = run(profile, deps.paths, deps)

    assert first.exit_code == 2 and second.exit_code == 2
    assert "CLAUDE_CODE_OAUTH_TOKEN" in (first.auth_error or "")
    assert first.lines[0].startswith("auth failure (provider)")
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["dedupe_key"] == "auth:provider:2026-10-01"
    assert deps.gh.auth_checks == 0  # creds are checked before gh
    assert handlers[Kind.CI].calls == [] and getattr(scanned, "calls", []) == []
    assert deps.state.attempts("o/r#1", "ci", HEAD) == 0 and deps.state.runs() == []
    notifies = [r for r in deps.audit.read() if r["event"] == "notify"]
    assert len(notifies) == 1 and notifies[0]["level"] == "error"


def test_precheck_gh_and_tracker_auth(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], scanned: Callable[..., ScanResult]
) -> None:
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh(auth_error="gh auth login required"))
    scanned([])
    report = run(profile, deps.paths, deps)
    assert report.exit_code == 2 and report.auth_error and report.auth_error.startswith("gh:")
    assert deps.state.flags()[0]["dedupe_key"] == "auth:gh:2026-10-01"

    # issue kind enabled + linear enabled: the tracker is pinged ...
    build_profile(kinds=["ci", "build"])
    profile = build_profile(kinds=["ci", "build"])
    tracker = FakeTracker(auth_error="Linear authentication failed")
    deps = make_deps(profile, tracker=tracker)
    report = run(profile, deps.paths, deps)
    assert tracker.pings == 1 and report.exit_code == 2 and report.auth_error.startswith("tracker:")
    # ... and a missing key counts as an auth failure too
    deps = make_deps(profile, tracker=None)
    report = run(profile, deps.paths, deps)
    assert report.exit_code == 2 and "LINEAR_API_KEY" in report.auth_error

    # upkeep-only kinds never touch the tracker
    profile = build_profile(kinds=["ci", "rv"])
    tracker = FakeTracker(auth_error="would fail")
    deps = make_deps(profile, tracker=tracker)
    report = run(profile, deps.paths, deps)
    assert tracker.pings == 0 and report.exit_code == 0


def test_missing_linear_key_without_issue_kinds_is_not_an_auth_failure(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult]
) -> None:
    profile = build_profile()  # default upkeep kinds; linear.enabled defaults to true; no LINEAR_API_KEY
    assert profile.linear.enabled and profile.secret("LINEAR_API_KEY") is None
    deps = make_deps(profile, gh=FakeGh({("o/r", 1): open_pr("o/r", 1)}), tracker=None)
    handlers[Kind.CI] = FakeHandler(Kind.CI, done)
    scanned([pr_action(Kind.CI, 1)])

    report = run(profile, deps.paths, deps)

    assert report.exit_code == 0 and report.auth_error is None and deps.state.flags() == []
    assert [rec.outcome.status for rec in report.outcomes] == ["done"]


def test_precheck_checks_gh_before_the_tracker(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], scanned: Callable[..., ScanResult]
) -> None:
    profile = build_profile(kinds=["ci", "build"])
    tracker = FakeTracker(auth_error="would fail too")
    deps = make_deps(profile, gh=FakeGh(auth_error="gh auth login required"), tracker=tracker)
    scanned([])

    report = run(profile, deps.paths, deps)

    assert report.exit_code == 2 and (report.auth_error or "").startswith("gh:")
    assert deps.gh.auth_checks == 1 and tracker.pings == 0  # the first failure stops the precheck
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["dedupe_key"] == "auth:gh:2026-10-01"
    assert getattr(scanned, "calls", []) == []


def test_scan_auth_error_is_a_precheck_failure(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = build_profile()
    deps = make_deps(profile)

    def failing_scan(profile: Profile, deps: Deps) -> ScanResult:
        raise GhAuthError("HTTP 401 bad credentials")

    monkeypatch.setattr(dispatch, "scan_sources", failing_scan)
    report = run(profile, deps.paths, deps)
    assert report.exit_code == 2 and deps.state.flags()[0]["reason"] == "auth failure: gh"


# ---------------------------------------------------------------------------- mid-wave auth
@pytest.mark.parametrize("exc", [AuthFailure("oauth token expired"), GhAuthError("gh: HTTP 401"), TrackerAuthError("linear 401")])
def test_mid_wave_auth_failure_stops_launching_flags_once_no_attempt(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult], exc: Exception
) -> None:
    profile = build_profile(max_parallel=1)
    gh = FakeGh({("o/r", n): open_pr("o/r", n) for n in (1, 2, 3)})
    deps = make_deps(profile, gh=gh)

    async def first_fails(action: Action, _deps: Deps) -> Outcome:
        await asyncio.sleep(0.01)
        raise exc

    handlers[Kind.CI] = FakeHandler(Kind.CI, first_fails)
    scanned([pr_action(Kind.CI, n) for n in (1, 2, 3)])

    report = run(profile, deps.paths, deps)

    assert report.exit_code == 2
    assert [a.target for a in handlers[Kind.CI].calls] == ["o/r#1"]  # nothing launched after the failure
    assert report.outcomes == []
    for n in (1, 2, 3):
        assert deps.state.attempts(f"o/r#{n}", "ci", HEAD) == 0 and not deps.state.is_handled(f"o/r#{n}", "ci", HEAD)
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["dedupe_key"].startswith("auth:") and flags[0]["dedupe_key"].endswith(":2026-10-01")
    assert deps.state.runs() == []


def test_mid_wave_auth_failure_lets_running_actions_finish(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult]
) -> None:
    profile = build_profile(max_parallel=2)
    gh = FakeGh({("o/r", n): open_pr("o/r", n) for n in (1, 2, 3)})
    deps = make_deps(profile, gh=gh)

    async def handler(action: Action, _deps: Deps) -> Outcome:
        if action.pr == 1:
            await asyncio.sleep(0.01)
            raise AuthFailure("expired")
        await asyncio.sleep(0.05)  # #2 is already running when #1 fails
        return Outcome(status="done", note="finished")

    handlers[Kind.CI] = FakeHandler(Kind.CI, handler)
    scanned([pr_action(Kind.CI, n) for n in (1, 2, 3)])
    report = run(profile, deps.paths, deps)

    assert report.exit_code == 2
    assert sorted(a.target for a in handlers[Kind.CI].calls) == ["o/r#1", "o/r#2"]
    assert deps.state.is_handled("o/r#2", "ci", HEAD)  # the in-flight action completed and was recorded
    assert not deps.state.is_handled("o/r#3", "ci", HEAD) and deps.state.attempts("o/r#3", "ci", HEAD) == 0


# ------------------------------------------------------------------- provider unavailable
def test_provider_unavailable_stops_launching_lets_in_flight_finish_flags_once_per_day(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate / usage limit is no verdict on the target: exit 2, no attempt, nothing new launched,
    the running action finishes, one flag + notification per day however often it recurs."""
    profile = build_profile(max_parallel=2)
    gh = FakeGh({("o/r", n): open_pr("o/r", n) for n in (1, 2, 3)})
    deps = make_deps(profile, gh=gh)
    refreshed: list[Any] = []
    monkeypatch.setattr(metrics, "refresh_outcomes", lambda state, gh, **kw: refreshed.append(gh) or 0)

    async def handler(action: Action, _deps: Deps) -> Outcome:
        if action.pr == 1:
            await asyncio.sleep(0.01)
            raise ProviderUnavailable("usage limit reached; resets at 15:00")
        await asyncio.sleep(0.05)  # #2 is already running when #1 hits the limit
        return Outcome(status="done", note="finished")

    handlers[Kind.CI] = FakeHandler(Kind.CI, handler)
    scanned([pr_action(Kind.CI, n) for n in (1, 2, 3)])

    report = run(profile, deps.paths, deps)

    assert report.exit_code == 2 and report.auth_error is None
    assert report.provider_error == "usage limit reached; resets at 15:00"
    assert "provider unavailable: usage limit reached; resets at 15:00" in report.lines
    assert sorted(a.target for a in handlers[Kind.CI].calls) == ["o/r#1", "o/r#2"]  # #3 never launched
    assert deps.state.is_handled("o/r#2", "ci", HEAD)  # the in-flight action completed and was recorded
    for n in (1, 3):
        assert deps.state.attempts(f"o/r#{n}", "ci", HEAD) == 0 and not deps.state.is_handled(f"o/r#{n}", "ci", HEAD)
    assert [r.action.target for r in report.outcomes] == ["o/r#2"]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["dedupe_key"] == "provider:2026-10-01" and flags[0]["kind"] == "provider"
    assert flags[0]["target"] == "profile:personal" and "usage limit" in flags[0]["detail"]
    assert len(refreshed) == 1  # gh still works: the metrics refresh is not skipped

    # the next tick the same day hits the limit again: still one flag, one notification, no attempt
    handlers[Kind.CI] = FakeHandler(Kind.CI, handler)
    scanned([pr_action(Kind.CI, 1)])
    again = run(profile, deps.paths, deps)
    assert again.exit_code == 2 and again.provider_error
    assert len(deps.state.flags()) == 1 and deps.state.attempts("o/r#1", "ci", HEAD) == 0
    notes = [r for r in deps.audit.read() if r["event"] == "notify"]
    assert len(notes) == 1 and notes[0]["title"] == "provider unavailable"
    assert [r["flagged"] for r in deps.audit.read() if r["event"] == "provider_unavailable"] == [True, False]

    # another day: a new flag
    later = make_deps(profile, gh=gh, now=datetime(2026, 10, 2, 9, 0, tzinfo=UTC))
    handlers[Kind.CI] = FakeHandler(Kind.CI, handler)
    run(profile, later.paths, later)
    assert [f["dedupe_key"] for f in later.state.flags()] == ["provider:2026-10-01", "provider:2026-10-02"]


def test_provider_unavailable_in_run_one_fails_without_an_attempt_and_flags_once(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, paths: ProfilePaths
) -> None:
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh({("o/r", 1): open_pr("o/r", 1)}))

    def limited(_action: Action, _deps: Deps) -> Outcome:
        raise ProviderUnavailable("overloaded")

    handlers[Kind.CI] = FakeHandler(Kind.CI, limited)
    outcome = asyncio.run(dispatch.run_one(profile, paths, pr_action(Kind.CI, 1), deps=deps, print_fn=lambda _l: None))

    assert outcome.status == "failed" and outcome.note == "provider unavailable: overloaded"
    assert deps.state.attempts("o/r#1", "ci", HEAD) == 0 and deps.state.runs() == []
    assert [f["dedupe_key"] for f in deps.state.flags()] == ["provider:2026-10-01"]


# ----------------------------------------------------------------------------- skip flags
def test_attempts_exhausted_skip_is_flagged_once_across_ticks_and_never_in_dry_run(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult],
    paths: ProfilePaths,
) -> None:
    """The scan attaches a flag spec to an attempts-exhausted skip but writes nothing itself; the
    dispatcher raises it (flag-once by its dedupe key) -- and a dry run does not."""
    profile = build_profile()
    deps = make_deps(profile)
    exhausted = pr_action(Kind.RC, 2)
    spec = exhausted_flag(profile, exhausted, 3, {"ts": "2026-10-01T11:00:00Z", "outcome": "failed", "note": "boom"})
    skips = [Skip(action=exhausted, reason="attempts exhausted (3/3)", flag=spec),
             Skip(action=pr_action(Kind.CI, 2), reason="handled at this version")]  # fmt: skip

    scanned([], skipped=skips)
    dry = run(profile, paths, deps, dry_run=True)
    assert dry.exit_code == 0 and deps.state.flags() == []
    assert [r for r in deps.audit.read() if r["event"] == "notify"] == []

    for _ in range(2):
        scanned([], skipped=skips)
        report = run(profile, paths, deps)
        assert report.exit_code == 0 and "skip rc o/r#2 attempts exhausted (3/3)" in report.lines

    flags = deps.state.flags()
    assert len(flags) == 1
    assert flags[0]["reason"] == EXHAUSTED_REASON and flags[0]["target"] == "o/r#2" and flags[0]["kind"] == "rc"
    assert flags[0]["dedupe_key"] == f"exhausted:o/r#2:rc:{exhausted.version}" and "last attempt" in flags[0]["detail"]
    notes = [r for r in deps.audit.read() if r["event"] == "notify"]
    assert len(notes) == 1 and notes[0]["title"] == EXHAUSTED_REASON and notes[0]["level"] == "warn"


def test_a_skip_flag_that_cannot_be_written_never_stops_the_tick(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh({("o/r", 1): open_pr("o/r", 1)}))
    exhausted = pr_action(Kind.RC, 2)
    handlers[Kind.CI] = FakeHandler(Kind.CI, done)
    scanned([pr_action(Kind.CI, 1)], skipped=[Skip(action=exhausted, reason="attempts exhausted", flag=exhausted_flag(profile, exhausted, 3))])

    def broken(*_a: Any, **_kw: Any) -> int:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(deps.state, "add_flag", broken)
    report = run(profile, deps.paths, deps)

    assert report.exit_code == 0 and [r.outcome.status for r in report.outcomes] == ["done"]
    failed = [r for r in deps.audit.read() if r["event"] == "skip_flag.failed"]
    assert len(failed) == 1 and failed[0]["target"] == "o/r#2" and "database is locked" in failed[0]["error"]


# ----------------------------------------------------------------------------- observe mode
def test_observe_mode_runs_only_rv_and_dry_runs_the_rest(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult]
) -> None:
    profile = build_profile(mode="observe")
    gh = FakeGh({("o/r", 1): open_pr("o/r", 1)})
    deps = make_deps(profile, gh=gh)
    assert isinstance(deps.effects, RecordingEffects)
    handlers[Kind.RV] = FakeHandler(Kind.RV, lambda a, d: Outcome(status="done", note="review drafted"))
    scanned([pr_action(Kind.CI, 1), pr_action(Kind.CONFLICT, 1), pr_action(Kind.RV, 1)])

    report = run(profile, deps.paths, deps)

    assert report.exit_code == 0
    by_kind = {rec.action.kind: rec.outcome for rec in report.outcomes}
    assert by_kind[Kind.CI].status == "dry_run" and by_kind[Kind.CONFLICT].status == "dry_run"
    assert by_kind[Kind.RV].status == "done"
    assert [a.kind for a in handlers[Kind.RV].calls] == [Kind.RV]
    would = [r for r in deps.audit.read() if r["event"] == "would_run"]
    assert sorted(r["kind"] for r in would) == ["ci", "conflict"]
    assert deps.state.is_handled("o/r#1", "rv", HEAD)
    assert not deps.state.is_handled("o/r#1", "ci", HEAD) and deps.state.attempts("o/r#1", "ci", HEAD) == 0
    assert not any(e.startswith("effect.") for e in (r["event"] for r in deps.audit.read()))
    assert gh.calls.count(("pr_state", "o/r", 1)) >= 1  # the rv re-check still happened


# --------------------------------------------------------------------------------- --dry-run
def test_dry_run_prints_actions_and_skips_and_writes_no_state(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult], paths: ProfilePaths
) -> None:
    profile = build_profile()
    deps = make_deps(profile)
    handlers[Kind.CI] = FakeHandler(Kind.CI, done)
    skip = Skip(action=pr_action(Kind.RC, 2), reason="handled at this version")
    scanned([pr_action(Kind.CI, 1)], skipped=[skip], errors=["github: o/x#9: boom"])

    report = run(profile, paths, deps, dry_run=True)

    assert report.exit_code == 0 and report.dry_run
    assert "scan error: github: o/x#9: boom" in report.lines
    assert "skip rc o/r#2 handled at this version" in report.lines
    assert "ci o/r#1 dry_run not-pushed would run (--dry-run)" in report.lines
    assert [rec.outcome.status for rec in report.outcomes] == ["dry_run"]
    assert handlers[Kind.CI].calls == []
    assert deps.state.runs() == [] and deps.state.flags() == []
    assert not deps.state.is_handled("o/r#1", "ci", HEAD) and deps.state.attempts("o/r#1", "ci", HEAD) == 0
    assert not paths.runs_log.exists()
    assert deps.gh.calls == []  # no PR re-check, no metrics refresh


def test_dry_run_needs_no_provider_credentials_and_flags_nothing(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult], paths: ProfilePaths
) -> None:
    """A dry run starts no worker, so a missing token is a printed warning, not an exit-2 auth
    failure that writes a flag and sends a notification (K1)."""
    profile = build_profile(token=False)
    deps = make_deps(profile)
    handlers[Kind.CI] = FakeHandler(Kind.CI, done)
    scanned([pr_action(Kind.CI, 1)])

    report = run(profile, paths, deps, dry_run=True)

    assert report.exit_code == 0 and report.auth_error is None
    assert any(line.startswith("warning: provider credentials: ") and "CLAUDE_CODE_OAUTH_TOKEN" in line for line in report.lines)
    assert "ci o/r#1 dry_run not-pushed would run (--dry-run)" in report.lines
    assert deps.gh.auth_checks == 1  # the read-only gh / tracker checks still run
    assert deps.state.flags() == [] and deps.state.runs() == [] and handlers[Kind.CI].calls == []
    assert [r for r in deps.audit.read() if r["event"] == "notify"] == []


def test_dry_run_auth_failure_is_printed_but_not_flagged(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], scanned: Callable[..., ScanResult], paths: ProfilePaths
) -> None:
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh(auth_error="gh auth login required"))
    scanned([])

    report = run(profile, paths, deps, dry_run=True)

    assert report.exit_code == 2 and "auth failure (gh): gh auth login required" in report.lines
    assert deps.state.flags() == [] and [r for r in deps.audit.read() if r["event"] == "notify"] == []
    assert next(r for r in deps.audit.read() if r["event"] == "auth_failure")["dry_run"] is True
    outcome = asyncio.run(dispatch.run_one(profile, paths, pr_action(Kind.CI, 1), dry_run=True, deps=deps, print_fn=lambda _l: None))
    assert outcome.status == "failed" and deps.state.flags() == []


def test_gh_auth_failure_names_the_gh_token_only_when_the_scrub_removed_it(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], scanned: Callable[..., ScanResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``doctor`` does not scrub; when the tick's scrub drops a GH_TOKEN that gh depended on, the
    auth failure must say so. When the scrub leaves the engine's token alone, no hint."""
    profile = build_profile()
    monkeypatch.setenv("GH_TOKEN", "ghp_test")
    real_scrub = dispatch.providers.scrub_process_env

    def scrubbing_gh_token() -> None:
        real_scrub()
        os.environ.pop("GH_TOKEN", None)

    monkeypatch.setattr(dispatch.providers, "scrub_process_env", scrubbing_gh_token)
    deps = make_deps(profile, gh=FakeGh(auth_error="gh is not authenticated: To log in, run: gh auth login"))
    scanned([])
    report = run(profile, deps.paths, deps)
    assert report.exit_code == 2 and "GH_TOKEN" in (report.auth_error or "") and "gh auth login" in report.auth_error
    assert "GH_TOKEN" not in os.environ

    monkeypatch.setenv("GH_TOKEN", "ghp_test")
    monkeypatch.setattr(dispatch.providers, "scrub_process_env", real_scrub)
    deps = make_deps(profile, gh=FakeGh(auth_error="gh is not authenticated"))
    report = run(profile, deps.paths, deps)
    hinted = "GH_TOKEN" in (report.auth_error or "")
    assert hinted is ("GH_TOKEN" not in os.environ)  # the hint tracks what the scrub actually did

    monkeypatch.delenv("GH_TOKEN", raising=False)
    deps = make_deps(profile, gh=FakeGh(auth_error="gh is not authenticated"))
    report = run(profile, deps.paths, deps)
    assert "GH_TOKEN" not in (report.auth_error or "")  # no token was there: no hint


# ------------------------------------------------------------------------- state recording
@pytest.mark.parametrize(
    ("status", "handled", "attempts"),
    [("done", True, 0), ("nothing", True, 0), ("failed", False, 1), ("kept", False, 1), ("flagged", False, 1),
     ("skipped", False, 0), ("dry_run", False, 0)],
)  # fmt: skip
def test_state_recording_per_outcome(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult],
    status: str, handled: bool, attempts: int,
) -> None:
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh({("o/r", 1): open_pr("o/r", 1)}))
    handlers[Kind.CI] = FakeHandler(Kind.CI, lambda a, d: Outcome(status=status, note=f"n-{status}"))  # type: ignore[arg-type]
    scanned([pr_action(Kind.CI, 1)])

    report = run(profile, deps.paths, deps)

    assert deps.state.is_handled("o/r#1", "ci", HEAD) is handled
    assert deps.state.attempts("o/r#1", "ci", HEAD) == attempts
    assert report.exit_code == (1 if status == "failed" else 0)
    runs = deps.state.runs()
    assert len(runs) == 1 and runs[0]["outcome"] == status and runs[0]["note"] == f"n-{status}"
    assert report.lines[-1] == f"ci o/r#1 {status} not-pushed n-{status}"


def test_handler_exception_is_failed_and_flagged_never_crashes_the_tick(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult]
) -> None:
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh({("o/r", n): open_pr("o/r", n) for n in (1, 2)}))

    def boom(action: Action, _deps: Deps) -> Outcome:
        if action.pr == 1:
            raise RuntimeError("boom")
        return Outcome(status="done")

    handlers[Kind.CI] = FakeHandler(Kind.CI, boom)
    scanned([pr_action(Kind.CI, 1), pr_action(Kind.CI, 2)])

    report = run(profile, deps.paths, deps)

    assert report.exit_code == 1
    outcomes = {rec.action.target: rec.outcome for rec in report.outcomes}
    assert outcomes["o/r#1"].status == "failed" and outcomes["o/r#1"].note == "RuntimeError: boom"
    assert outcomes["o/r#2"].status == "done"  # the other group still ran
    assert deps.state.attempts("o/r#1", "ci", HEAD) == 1
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "handler crashed"
    assert "RuntimeError: boom" in flags[0]["detail"] and "Traceback" in flags[0]["detail"]
    run(profile, deps.paths, deps)  # same crash again -> flag deduped
    assert len(deps.state.flags()) == 1


def test_run_record_in_runs_jsonl_and_state(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult], paths: ProfilePaths
) -> None:
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh({("o/r", 1): open_pr("o/r", 1)}))
    outcome = Outcome(status="done", pushed=True, note="fixed lint", cost_usd=0.5, usage={"input_tokens": 7}, duration_ms=123, pr_url="https://github.com/o/r/pull/1")
    handlers[Kind.CI] = FakeHandler(Kind.CI, lambda a, d: outcome)
    scanned([pr_action(Kind.CI, 1)])

    report = run(profile, paths, deps)

    lines = paths.runs_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {
        "ts": "2026-10-01T12:00:00+00:00", "profile": "personal", "run_id": "run-1", "kind": "ci",
        "target": "o/r#1", "version": HEAD, "outcome": "done", "pushed": True, "cost_usd": 0.5,
        "usage": {"input_tokens": 7}, "duration_ms": 123, "note": "fixed lint",
        "pr_url": "https://github.com/o/r/pull/1", "url": None,
    }  # fmt: skip
    stored = deps.state.runs()[0]
    assert stored["kind"] == "ci" and stored["cost_usd"] == 0.5 and stored["pushed"] is True
    assert "ci o/r#1 done pushed fixed lint" in report.lines


# ------------------------------------------------------------------------------ PR re-check
def test_pr_recheck_skips_closed_and_stale_actions(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult]
) -> None:
    profile = build_profile()
    prs = {
        ("o/r", 1): open_pr("o/r", 1, state="MERGED"),
        ("o/r", 2): open_pr("o/r", 2, head="h2"),
        ("o/r", 3): open_pr("o/r", 3, base="b9"),
        ("o/r", 4): open_pr("o/r", 4, head="moved"),
    }
    deps = make_deps(profile, gh=FakeGh(prs))
    for kind in (Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV):
        handlers[kind] = FakeHandler(kind, done)
    scanned([pr_action(Kind.CI, 1), pr_action(Kind.RV, 2), pr_action(Kind.CONFLICT, 3), pr_action(Kind.RC, 4), pr_action(Kind.CI, 5)])

    report = run(profile, deps.paths, deps)

    outcomes = {(rec.action.kind, rec.action.pr): rec.outcome for rec in report.outcomes}
    assert outcomes[(Kind.CI, 1)].status == "skipped" and outcomes[(Kind.CI, 1)].note == "pr merged"
    assert outcomes[(Kind.RV, 2)].status == "skipped" and "stale" in outcomes[(Kind.RV, 2)].note
    assert outcomes[(Kind.CONFLICT, 3)].status == "skipped" and "stale" in outcomes[(Kind.CONFLICT, 3)].note
    assert outcomes[(Kind.RC, 4)].status == "done"  # rc is keyed by feedback time, not the head
    assert outcomes[(Kind.CI, 5)].status == "failed" and "re-check failed" in outcomes[(Kind.CI, 5)].note
    called = {(a.kind, a.pr) for h in handlers.values() for a in h.calls}
    assert called == {(Kind.RC, 4)}
    for kind, n in ((Kind.CI, 1), (Kind.RV, 2), (Kind.CONFLICT, 3)):
        assert deps.state.attempts(f"o/r#{n}", kind.value, pr_action(kind, n).version) == 0


def test_kept_worktree_from_an_earlier_action_skips_the_rest_of_the_group(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult],
    paths: ProfilePaths, tmp_path: Path,
) -> None:
    """ci keeps its worktree (push refused); the conflict action on the same PR must not run and
    overwrite the kept record (the ci handoff would vanish from ``status`` and ``cleanup`` would
    delete it). The same goes for ``factory run`` with force."""
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh({("o/r", 7): open_pr("o/r", 7)}))
    path1, path2 = tmp_path / "wt" / "ci", tmp_path / "wt" / "conflict"

    def keep_ci(action: Action, d: Deps) -> Outcome:
        path1.mkdir(parents=True)
        d.state.keep_worktree(action.target, "ci", str(path1), "sensitive path")
        return Outcome(status="kept", note="push refused")

    def keep_conflict(action: Action, d: Deps) -> Outcome:
        path2.mkdir(parents=True)
        d.state.keep_worktree(action.target, "conflict", str(path2), "unsafe hunks")
        return Outcome(status="kept", note="conflict needs a human")

    handlers[Kind.CI] = FakeHandler(Kind.CI, keep_ci)
    handlers[Kind.CONFLICT] = FakeHandler(Kind.CONFLICT, keep_conflict)
    scanned([pr_action(Kind.CONFLICT, 7), pr_action(Kind.CI, 7)])

    report = run(profile, paths, deps)

    outcomes = {rec.action.kind: rec.outcome for rec in report.outcomes}
    assert outcomes[Kind.CI].status == "kept" and outcomes[Kind.CONFLICT].status == "skipped"
    assert outcomes[Kind.CONFLICT].note == "kept worktree awaiting me (ci): sensitive path"
    assert handlers[Kind.CONFLICT].calls == [] and not path2.exists()
    assert deps.state.kept_worktree("o/r#7")["path"] == str(path1)  # type: ignore[index]
    assert deps.state.attempts("o/r#7", "conflict", f"{HEAD}:{BASE}") == 0 and deps.state.attempts("o/r#7", "ci", HEAD) == 1
    assert "conflict o/r#7 skipped not-pushed kept worktree awaiting me (ci): sensitive path" in report.lines

    outcome = asyncio.run(dispatch.run_one(profile, paths, pr_action(Kind.CONFLICT, 7), deps=deps, print_fn=lambda _l: None, force=True))
    assert outcome.status == "skipped" and "`factory release o/r#7`" in outcome.note and handlers[Kind.CONFLICT].calls == []
    deps.state.release_worktree("o/r#7")
    outcome = asyncio.run(dispatch.run_one(profile, paths, pr_action(Kind.CONFLICT, 7), deps=deps, print_fn=lambda _l: None, force=True))
    assert outcome.status == "kept" and deps.state.kept_worktree("o/r#7")["path"] == str(path2)  # type: ignore[index]


# -------------------------------------------------------------------------------- metrics
def test_metrics_refresh_after_dispatch_never_fails_the_tick(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, scanned: Callable[..., ScanResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_profile()
    gh = FakeGh({("o/r", 8): PullRequest(repo="o/r", number=8, state="OPEN", title="t", created_at="2026-09-01T00:00:00Z")})
    deps = make_deps(profile, gh=gh)
    deps.state.register_pr("o/r", 8, "factory/x", "build")
    scanned([])
    report = run(profile, deps.paths, deps)
    assert report.exit_code == 0 and "nothing to do" in report.lines
    outcomes = deps.state.pr_outcomes()
    assert len(outcomes) == 1 and outcomes[0]["number"] == 8 and outcomes[0]["state"] == "OPEN"

    def explode(state: State, gh: Any, **kw: Any) -> int:
        raise RuntimeError("github down")

    monkeypatch.setattr(metrics, "refresh_outcomes", explode)
    report = run(profile, deps.paths, deps)
    assert report.exit_code == 0 and any(line.startswith("metrics refresh failed") for line in report.lines)


def test_metrics_refresh_is_skipped_after_an_auth_failure(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], scanned: Callable[..., ScanResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh(auth_error="gh auth login required"))
    scanned([])
    calls: list[Any] = []
    monkeypatch.setattr(metrics, "refresh_outcomes", lambda state, gh, **kw: calls.append(gh) or 0)
    report = run(profile, deps.paths, deps)
    assert report.exit_code == 2 and calls == []


# --------------------------------------------------------------------------------- run_one
def test_run_one_runs_a_single_action_with_prechecks_filters_and_recording(
    build_profile: Callable[..., Profile], make_deps: Callable[..., Deps], handlers: dict, paths: ProfilePaths
) -> None:
    profile = build_profile()
    deps = make_deps(profile, gh=FakeGh({("o/r", 1): open_pr("o/r", 1)}))
    handlers[Kind.CI] = FakeHandler(Kind.CI, done)
    action = pr_action(Kind.CI, 1)
    lines: list[str] = []

    dry = asyncio.run(dispatch.run_one(profile, paths, action, dry_run=True, deps=deps, print_fn=lines.append))
    assert dry.status == "dry_run" and handlers[Kind.CI].calls == [] and deps.state.runs() == []

    outcome = asyncio.run(dispatch.run_one(profile, paths, action, deps=deps, print_fn=lines.append))
    assert outcome.status == "done" and deps.state.is_handled("o/r#1", "ci", HEAD) and len(deps.state.runs()) == 1

    again = asyncio.run(dispatch.run_one(profile, paths, action, deps=deps, print_fn=lines.append))
    assert again.status == "skipped" and again.note == "handled at this version"
    forced = asyncio.run(dispatch.run_one(profile, paths, action, deps=deps, print_fn=lines.append, force=True))
    assert forced.status == "done" and len(handlers[Kind.CI].calls) == 2

    bad_profile = build_profile(token=False)
    bad = make_deps(bad_profile)
    failed = asyncio.run(dispatch.run_one(bad_profile, paths, action, deps=bad, print_fn=lines.append))
    assert failed.status == "failed" and "auth failure" in failed.note
    assert bad.state.flags()[0]["dedupe_key"] == "auth:provider:2026-10-01"
    assert len(bad.state.runs()) == 2  # same state.db as `deps`: the two done runs, nothing new
