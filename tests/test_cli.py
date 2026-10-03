"""``factory`` CLI through ``typer.testing.CliRunner`` with every dependency faked.

No network, no workers, no real scheduler: ``Deps.build`` / the CLI seams are monkeypatched with
fakes, the scan is scripted, the scheduler gets a fake runner, and state lives in the per-test data
dir the conftest isolates. Covers ``tick --dry-run``, observe-mode ticks, exit codes, ``doctor``
(all green and failing), ``status`` / ``flags`` / ``report``, ``revert-refine`` wiring, ``schedule``
with a fake runner, ``cleanup`` / ``release`` and the ``run`` target parsing.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock
from typer.testing import CliRunner

from software_factory import cli, dispatch, scheduler
from software_factory.actions import Action, Kind
from software_factory.cli import app
from software_factory.config import Profile, RepoOverride, load_profile
from software_factory.effects import RecordingEffects
from software_factory.kinds.base import AuthFailure, Deps, Outcome, ProviderUnavailable
from software_factory.notify import Audit, Notifier
from software_factory.paths import DATA_DIR_ENV, ProfilePaths
from software_factory.repos import RepoIndex, RepoInfo
from software_factory.runner import WorkerRun
from software_factory.scan import ScanResult, Skip
from software_factory.scheduler import RunResult
from software_factory.sources.github import GhAuthError, GhError
from software_factory.sources.models import Check, PullRequest
from software_factory.state import State
from software_factory.trackers.base import Issue, TrackerError

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
SLUG = "JaneDoe/repo"
HEAD = "a" * 40
NOT_FOUND = "ERROR: The system cannot find the file specified."

runner = CliRunner()


# ------------------------------------------------------------------------------------- fakes
class FakeGh:
    def __init__(self, prs: dict[tuple[str, int], PullRequest] | None = None, *, auth_error: str | None = None) -> None:
        self.prs = prs or {}
        self.auth_error = auth_error
        self.calls: list[tuple[str, ...]] = []

    def auth_check(self) -> None:
        if self.auth_error:
            raise GhAuthError(self.auth_error)

    def login(self) -> str:
        return "JaneDoe"

    def pr_details(self, repo: str, number: int) -> PullRequest:
        self.calls.append(("pr_details", repo, str(number)))
        pr = self.prs.get((repo, number))
        if pr is None:
            raise GhError(f"{repo}#{number} not found")
        return pr

    pr_state = pr_details

    def pr_commits(self, repo: str, number: int) -> list:
        return []

    def find_open_pr_for_branch(self, repo: str, branch: str) -> PullRequest | None:
        return None


class FakeTracker:
    name = "fake"

    def __init__(
        self,
        issues: list[Issue] | None = None,
        *,
        existing_labels: tuple[str, ...] = (),
        ping_error: Exception | None = None,
        team_labels: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.issues = {i.id: i for i in (issues or [])}
        self.existing = {lbl.lower() for lbl in existing_labels}
        self.ping_error = ping_error
        # team id -> labels that team can apply (``ensure_label``); other teams see the workspace set
        self.team_labels = {team: {lbl.lower() for lbl in labels} for team, labels in (team_labels or {}).items()}
        self.label_queries: list[list[str]] = []
        self.ensure_calls: list[tuple[str, str]] = []

    def ping(self) -> str:
        if self.ping_error is not None:
            raise self.ping_error
        return "Test User"

    def labels_exist(self, names: list[str]) -> dict[str, bool]:
        self.label_queries.append(list(names))
        return {n: n.lower() in self.existing for n in names}

    def ensure_label(self, team_id: str, name: str) -> str:
        self.ensure_calls.append((team_id, name))
        if name.lower() in self.team_labels.get(team_id, self.existing):
            return f"label-{team_id}-{name}"
        raise TrackerError(f"label {name!r} does not exist for team {team_id} (create it in Linear)")

    def list_issues_with_label(self, label: str) -> list[Issue]:
        return [i for i in self.issues.values() if i.has_label(label)]

    def get_issue(self, ref: str) -> Issue:
        for issue in self.issues.values():
            if ref in (issue.id, issue.key):
                return issue
        raise TrackerError(f"issue {ref} not found")


class FakeEffects:
    """Records the tracker-side effects ``revert-refine`` uses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None:
        self.calls.append(("update_issue_text", issue_id, title, description))

    def remove_label(self, issue_id: str, label: str) -> None:
        self.calls.append(("remove_label", issue_id, label))


class FakeWtManager:
    """Only what ``ready_action`` touches: ``git(["fetch", ...])`` and ``git(["rev-parse", ...])``."""

    def __init__(self, sha: str = HEAD) -> None:
        self.sha = sha
        self.calls: list[list[str]] = []

    def git(self, args: list[str], *, cwd: Path | None = None) -> str:
        self.calls.append(list(args))
        return f"{self.sha}\n" if args and args[0] == "rev-parse" else ""


class FakeHandler:
    def __init__(self, outcome: Outcome) -> None:
        self.outcome = outcome
        self.calls: list[Action] = []

    async def run(self, action: Action, deps: Deps) -> Outcome:
        self.calls.append(action)
        return self.outcome


class FakeRunner:
    def __init__(self, handler: Callable[[list[str]], tuple[int, str, str]] | None = None) -> None:
        self.argvs: list[list[str]] = []
        self.handler = handler or (lambda argv: (0, "", ""))

    def __call__(self, argv: Any, *, input: str | None = None) -> RunResult:
        args = [str(a) for a in argv]
        self.argvs.append(args)
        rc, out, err = self.handler(args)
        return RunResult(argv=args, returncode=rc, stdout=out, stderr=err)


async def no_worker(**_: Any) -> WorkerRun:  # pragma: no cover - a worker must never be started here
    raise AssertionError("run_worker must not be called from CLI tests")


def repo_info(path: Path, override: RepoOverride | None = None) -> RepoInfo:
    return RepoInfo(slug=SLUG, name="repo", dirname="repo", path=path, default_branch="main", override=override or RepoOverride())


def installed_show(profile: Profile, paths: ProfilePaths, *, extra: str = "") -> str:
    """What ``schedule show`` prints (Windows shape) for jobs installed from *this* shell: the task
    runs the wrapper under the current data dir and the wrapper carries the env install embeds."""
    spec = scheduler.ScheduleSpec.from_profile(profile, paths, require_uv=False)
    env = ", ".join(f"{k}={v}" for k, v in spec.env().items()) or "(none)"
    wrapper = paths.scripts / "software-factory-personal-tick.cmd"
    return (
        f"TaskName:                             \\software-factory-personal-tick\n"
        f'Task To Run:                          "{wrapper}"\n'
        f"Status:                               Ready\n"
        f"Wrapper:                              {wrapper}\n"
        f"Wrapper Env:                          {env}{extra}"
    )


def open_pr(number: int = 1, *, failing: bool = False, state: str = "OPEN", repo: str = SLUG) -> PullRequest:
    checks = [Check(name="test", state="failure", details_url="https://github.com/x/actions/runs/7/job/8")] if failing else []
    return PullRequest(
        repo=repo, number=number, title=f"PR {number}", url=f"https://github.com/{repo}/pull/{number}", author="someone",
        state=state, head_ref_name="feature", head_ref_oid=HEAD, base_ref_name="main", base_ref_oid="b" * 40,
        mergeable="MERGEABLE", checks=checks,
    )  # fmt: skip


def pr_action(kind: Kind, number: int, *, repo: str = "o/r") -> Action:
    return Action(kind=kind, version=HEAD, repo=repo, pr=number, title=f"{kind.value} {number}")


# ---------------------------------------------------------------------------------- fixtures
@pytest.fixture
def profile_toml(repos_root: Path) -> dict[str, Any]:
    return {
        "profile": {"mode": "full"},
        "notify": {"desktop": False, "slack": False},
        "github": {"owners": ["JaneDoe"], "login": "JaneDoe"},
        "repos": {"root": str(repos_root)},
    }


@pytest.fixture
def secrets() -> dict[str, str]:
    return {
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test", "LINEAR_API_KEY": "lin_api_test",
        "SLACK_BOT_TOKEN": "xoxb-test", "SLACK_USER_ID": "U1", "GH_WORKER_TOKEN": "github_pat_worker_test",
    }  # fmt: skip


@pytest.fixture
def invoke(profiles_dir: Path) -> Callable[..., Any]:
    def _invoke(*args: str, input: str | None = None) -> Any:
        return runner.invoke(app, ["--profiles-dir", str(profiles_dir), *args], input=input)

    return _invoke


@pytest.fixture
def paths() -> ProfilePaths:
    return ProfilePaths.for_profile("personal").ensure()  # SOFTWARE_FACTORY_DATA_DIR is isolated by conftest


@pytest.fixture
def fake_deps(tmp_path: Path, paths: ProfilePaths) -> Callable[..., Deps]:
    """Deps with fakes + a real State; ``gh`` / ``tracker`` / ``effects`` / ``wt`` are overridable."""

    def _make(profile: Profile, *, gh: Any = None, tracker: Any = None, effects: Any = None, wt: Any = None) -> Deps:
        state = State(paths.state_db)
        audit = Audit(paths.audit_log, profile.name)
        gh = gh or FakeGh()
        effects = effects or RecordingEffects(profile, paths, state, gh, tracker, audit)
        return Deps(
            profile=profile, paths=paths, state=state, gh=gh, tracker=tracker, repos=RepoIndex([repo_info(tmp_path / "repo")]),
            effects=effects, notifier=Notifier(profile, audit), audit=audit, wt_manager=wt or FakeWtManager(),
            resolver=None, run_worker=no_worker, now=lambda: NOW, run_id="run-1",
        )  # fmt: skip

    return _make


@pytest.fixture
def patch_build(monkeypatch: pytest.MonkeyPatch, fake_deps: Callable[..., Deps]) -> Callable[..., dict[str, Any]]:
    """Route both ``cli.build_deps`` and ``Deps.build`` (used inside ``run_tick``) to fake deps."""

    def _patch(**overrides: Any) -> dict[str, Any]:
        holder: dict[str, Any] = {}

        def build(profile: Profile, paths: ProfilePaths, *, dry_run: bool, **_: Any) -> Deps:
            holder["dry_run"] = dry_run
            holder["deps"] = fake_deps(profile, **overrides)
            return holder["deps"]

        monkeypatch.setattr(cli, "build_deps", build)
        monkeypatch.setattr(Deps, "build", staticmethod(build))
        return holder

    return _patch


def scripted_scan(monkeypatch: pytest.MonkeyPatch, actions: list[Action], skipped: list[Skip] | None = None) -> None:
    monkeypatch.setattr(dispatch, "scan_sources", lambda profile, deps: ScanResult(actions=actions, skipped=skipped or []))


# ======================================================================================== help
def test_help_lists_every_command(invoke: Callable[..., Any]) -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("tick", "run", "status", "flags", "doctor", "readiness", "ready", "report", "revert-refine", "schedule", "cleanup", "release"):
        assert command in result.output
    assert "--profile" in result.output and "--profiles-dir" in result.output


def test_missing_profile_is_a_clean_error(invoke: Callable[..., Any]) -> None:
    result = invoke("status")
    assert result.exit_code == 1
    assert "profile error" in result.output and "not found" in result.output


# ======================================================================================== tick
def test_tick_dry_run_prints_actions_and_skips_and_starts_nothing(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict,
    patch_build: Callable[..., dict], monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    holder = patch_build()
    scripted_scan(monkeypatch, [pr_action(Kind.CI, 1), pr_action(Kind.RV, 2)], [Skip(action=pr_action(Kind.RC, 3), reason="handled at this version")])
    result = invoke("tick", "--dry-run")
    assert result.exit_code == 0, result.output
    assert holder["dry_run"] is True
    assert "ci o/r#1 dry_run not-pushed would run (--dry-run)" in result.output
    assert "rv o/r#2 dry_run not-pushed would run (--dry-run)" in result.output
    assert "skip rc o/r#3 handled at this version" in result.output
    assert "dry-run: 2 action(s) would run, 1 skipped, 0 scan error(s); nothing was started" in result.output
    state: State = holder["deps"].state
    assert state.runs() == [] and state.flags() == []


def test_tick_in_observe_mode_runs_only_rv_and_records_would_run(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict,
    patch_build: Callable[..., dict], monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    profile_toml["profile"]["mode"] = "observe"
    make_profile("personal", profile_toml, secrets)
    holder = patch_build(gh=FakeGh({("o/r", 1): open_pr(1, repo="o/r")}))
    scripted_scan(monkeypatch, [pr_action(Kind.CI, 1), pr_action(Kind.RV, 1)])
    handlers = {Kind.RV: FakeHandler(Outcome(status="done", note="review draft ready")), Kind.CI: FakeHandler(Outcome(status="done"))}
    monkeypatch.setattr(dispatch, "get_handler", lambda kind: handlers[kind])
    result = invoke("tick")
    assert result.exit_code == 0, result.output
    assert "ci o/r#1 dry_run not-pushed observe mode: would run" in result.output
    assert "rv o/r#1 done not-pushed review draft ready" in result.output
    assert handlers[Kind.CI].calls == [] and len(handlers[Kind.RV].calls) == 1
    deps: Deps = holder["deps"]
    assert deps.state.is_handled("o/r#1", "rv", HEAD)
    assert any(r["event"] == "would_run" and r["kind"] == "ci" for r in deps.audit.read())


def test_tick_exit_codes_for_failed_action_and_auth_failure(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict,
    patch_build: Callable[..., dict], monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    patch_build(gh=FakeGh({("o/r", 1): open_pr(1, repo="o/r")}))
    scripted_scan(monkeypatch, [pr_action(Kind.CI, 1)])
    monkeypatch.setattr(dispatch, "get_handler", lambda kind: FakeHandler(Outcome(status="failed", note="boom")))
    result = invoke("tick")
    assert result.exit_code == 1, result.output
    assert "ci o/r#1 failed not-pushed boom" in result.output

    patch_build(gh=FakeGh(auth_error="gh auth login needed"))
    result = invoke("tick")
    assert result.exit_code == 2
    assert "auth failure (gh): gh auth login needed" in result.output


def test_tick_without_linear_key_and_only_upkeep_kinds_is_not_an_auth_failure(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict,
    patch_build: Callable[..., dict], monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test"})
    patch_build()  # tracker=None
    scripted_scan(monkeypatch, [])
    result = invoke("tick", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "auth failure" not in result.output
    assert "dry-run: 0 action(s) would run, 0 skipped, 0 scan error(s); nothing was started" in result.output


# ====================================================================================== doctor
@pytest.fixture
def green_doctor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    tracker = FakeTracker(existing_labels=("factory:refine", "factory:plan", "factory:build", "repo:repo"))
    monkeypatch.setattr(cli, "gh_client", lambda profile: FakeGh())
    monkeypatch.setattr(cli, "gh_detected_login", lambda: "JaneDoe")
    monkeypatch.setattr(cli, "tracker_client", lambda profile: tracker)
    monkeypatch.setattr(cli, "slack_auth_test", lambda token, http=None: (True, "bot factory in workspace Acme"))
    monkeypatch.setattr(cli, "claude_cli_status", lambda profile=None: ("ok", "bundled claude CLI 2.1.286"))
    monkeypatch.setattr(cli, "toast_backend_status", lambda: ("ok", "desktop-notifier importable"))
    monkeypatch.setattr(cli, "discover_repos", lambda profile: RepoIndex([repo_info(tmp_path / "repo")]))
    monkeypatch.setattr(cli, "scheduler_status", installed_show)
    return {"tracker": tracker}


def test_doctor_all_green_exits_zero(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict
) -> None:
    profile_toml["notify"] = {"desktop": True, "slack": True}
    make_profile("personal", profile_toml, secrets)
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    checks = [ln for ln in result.output.splitlines() if ln[:4].strip() in ("ok", "warn", "fail")]
    assert len(checks) >= 12 and all(ln.startswith("ok") for ln in checks), result.output
    assert "gh auth: logged in as JaneDoe" in result.output
    assert "ok   worker gh token: GH_WORKER_TOKEN set in the profile .env" in result.output
    assert "linear: reachable as Test User" in result.output
    assert "linear labels: factory:refine, factory:plan, factory:build" in result.output
    assert "linear repo labels: all 1 repo:<name> labels exist" in result.output
    assert "slack: auth.test: bot factory in workspace Acme; DMs go to U1" in result.output
    assert "data dir:" in result.output and "writable" in result.output
    assert "ok   git hooks: disabled for engine and worker git" in result.output
    assert "scheduler: jobs installed" in result.output
    assert "0 fail, 0 warn" in result.output


def test_doctor_fails_when_git_hooks_cannot_be_disabled_safely(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    from software_factory import providers

    def broken() -> str:
        raise providers.HooksPathError("parent is a directory")

    monkeypatch.setattr(providers, "unplantable_hooks_path", broken)
    make_profile("personal", profile_toml, secrets)
    result = invoke("doctor")
    assert result.exit_code == 1, result.output
    assert "fail git hooks: cannot disable git hooks safely: parent is a directory" in result.output


def test_doctor_reports_failures_and_exits_one(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, green_doctor: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    profile_toml["notify"] = {"desktop": True, "slack": True}
    profile_toml["profile"]["enabled_kinds"] = ["ci", "build"]
    make_profile("personal", profile_toml)  # no .env at all
    monkeypatch.setattr(cli, "gh_client", lambda profile: FakeGh(auth_error="gh auth login"))
    monkeypatch.setattr(cli, "scheduler_status", lambda profile, paths: "software-factory-personal-tick: not installed")
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "warn secrets:" in result.output and ".env not found" in result.output
    assert "fail provider credentials: CLAUDE_CODE_OAUTH_TOKEN missing" in result.output
    assert "fail gh auth: gh auth login" in result.output
    assert "warn worker gh token: GH_WORKER_TOKEN not in the profile .env" in result.output
    assert "fail linear: LINEAR_API_KEY missing from the profile .env (an issue kind is enabled)" in result.output
    assert "fail slack: SLACK_BOT_TOKEN, SLACK_USER_ID missing" in result.output
    assert "warn scheduler: jobs not installed" in result.output


def test_doctor_missing_linear_key_is_only_a_warning_for_upkeep_profiles(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, green_doctor: dict
) -> None:
    make_profile("personal", profile_toml, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test"})
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "warn linear: LINEAR_API_KEY missing" in result.output and "only needed once an issue kind" in result.output


def test_doctor_missing_labels_and_unreachable_linear(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    profile_toml["profile"]["enabled_kinds"] = ["ci", "refine"]
    make_profile("personal", profile_toml, secrets)
    monkeypatch.setattr(cli, "tracker_client", lambda profile: FakeTracker(existing_labels=("factory:refine",)))
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "fail linear labels: missing factory:plan, factory:build" in result.output
    assert "warn linear repo labels: 1 of 1 repo:<name> labels missing" in result.output and "repo:repo" in result.output

    monkeypatch.setattr(cli, "tracker_client", lambda profile: FakeTracker(ping_error=TrackerError("connection refused")))
    result = invoke("doctor")
    assert result.exit_code == 1 and "fail linear: unreachable: connection refused" in result.output


def test_doctor_employer_profile_inside_a_git_checkout_warns(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profiles_dir: Path, profile_toml: dict, green_doctor: dict
) -> None:
    (profiles_dir.parent / ".git").mkdir()
    profile_toml["profile"] = {"employer": True, "enabled_kinds": ["rv"]}
    profile_toml["provider"] = {"kind": "api_key"}
    make_profile("personal", profile_toml, {"ANTHROPIC_API_KEY": "sk-ant-api-test", "LINEAR_API_KEY": "lin"})
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "ok   employer rules: employer profile: provider api_key" in result.output
    assert "warn employer profile location: profile dir sits inside the git checkout" in result.output
    assert "mode observe" in result.output  # employer profiles default to observe


def test_doctor_unloadable_profile_fails(invoke: Callable[..., Any], make_profile: Callable[..., Path]) -> None:
    make_profile("personal", "this is not = toml [[", None)
    result = invoke("doctor")
    assert result.exit_code == 1 and result.output.startswith("fail profile:")


def test_doctor_repo_shows_discovery_and_static_readiness(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict,
    git_repo_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    path = git_repo_factory("repo", files={"CLAUDE.md": "# repo\n\n## Verification\n\n```\nnpm test\n```\n", "README.md": "x\n"})
    monkeypatch.setattr(cli, "discover_repos", lambda profile: RepoIndex([repo_info(path)]))
    make_profile("personal", profile_toml, secrets)
    result = invoke("doctor", "--repo", "repo")
    assert result.exit_code == 0, result.output
    assert f"ok   repo: {SLUG} at {path}" in result.output
    assert "ok   repo verification: 1 command(s) from CLAUDE.md: test: npm test" in result.output
    assert "ok   repo readiness (static): 50/50 static points, 3/3 static checks ok (score 50/100" in result.output
    assert "commands_discovered: 1 command(s) from CLAUDE.md" in result.output
    assert "n/a    commands_pass: not run" in result.output

    result = invoke("doctor", "--repo", "nope")
    assert result.exit_code == 1 and "fail repo: 'nope' is not a managed repo" in result.output


def test_claude_cli_status_follows_the_sdk_lookup_and_fails_on_an_old_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import claude_agent_sdk
    from claude_agent_sdk._cli_version import __cli_version__ as expected

    bundled_dir = Path(claude_agent_sdk.__file__).parent / "_bundled"
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli, "_cli_extra_locations", lambda windows: [])
    status, detail = cli.claude_cli_status()
    if any(bundled_dir.glob("claude*")):
        assert status == "ok" and "bundled claude CLI" in detail
        return  # the PATH lookup below is never reached with a bundled CLI
    assert status == "fail" and "not on PATH" in detail and "provider.cli_path" in detail
    exe = tmp_path / "claude.exe"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(exe))
    monkeypatch.setattr(cli, "_cli_version", lambda path: f"{expected} (Claude Code)")
    status, detail = cli.claude_cli_status()
    assert status == "ok" and str(exe) in detail and expected in detail
    # K2: a CLI older than the one the SDK was built against cannot run the configured model.
    monkeypatch.setattr(cli, "_cli_version", lambda path: "2.1.272 (Claude Code)")
    status, detail = cli.claude_cli_status()
    assert status == "fail", detail
    assert "2.1.272" in detail and f"older than the CLI the SDK expects ({expected})" in detail
    assert "claude update" in detail and "provider.cli_path" in detail
    monkeypatch.setattr(cli, "_cli_version", lambda path: "version unknown: OSError")
    status, detail = cli.claude_cli_status()
    assert status == "warn" and "could not compare" in detail


def test_claude_cli_status_honours_provider_cli_path_before_any_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from claude_agent_sdk._cli_version import __cli_version__ as expected

    def never(name: str) -> str | None:  # pragma: no cover - the assertion is the point
        raise AssertionError("cli_path set: PATH must not be searched")

    monkeypatch.setattr(cli.shutil, "which", never)
    monkeypatch.setattr(cli, "_cli_extra_locations", lambda windows: [])
    monkeypatch.setattr(cli, "_cli_version", lambda path: f"{expected} (Claude Code)")
    status, detail = cli.claude_cli_status(Profile(name="p", provider={"cli_path": str(tmp_path / "nope" / "claude.exe")}))
    assert status == "fail" and "provider.cli_path" in detail and "is not a file" in detail

    monkeypatch.setattr(cli.sys, "platform", "win32")
    shim = tmp_path / "claude.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")
    status, detail = cli.claude_cli_status(Profile(name="p", provider={"cli_path": str(shim)}))
    assert status == "fail" and "not a native .exe/.com" in detail

    exe = tmp_path / "newer" / "claude.exe"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    status, detail = cli.claude_cli_status(Profile(name="p", provider={"cli_path": str(exe)}))
    assert status == "ok", detail
    assert f"provider.cli_path {exe}" in detail and expected in detail
    monkeypatch.setattr(cli, "_cli_version", lambda path: "1.0.0 (Claude Code)")
    status, detail = cli.claude_cli_status(Profile(name="p", provider={"cli_path": str(exe)}))
    assert status == "fail" and "older than the CLI the SDK expects" in detail


def test_doctor_checks_the_gh_account_against_github_login(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)  # github.login = "JaneDoe"
    monkeypatch.setattr(cli, "gh_detected_login", lambda: "someone-else")
    result = invoke("doctor")
    assert result.exit_code == 1, result.output
    assert "fail gh auth: gh is logged in as someone-else but github.login = JaneDoe (fix github.login or leave it empty" in result.output

    monkeypatch.setattr(cli, "gh_detected_login", lambda: "janedoe")  # GitHub logins are case-insensitive
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "ok   gh auth: logged in as janedoe" in result.output

    profile_toml["github"]["login"] = ""
    make_profile("personal", profile_toml, secrets)
    monkeypatch.setattr(cli, "gh_detected_login", lambda: "someone-else")
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "ok   gh auth: logged in as someone-else (github.login is empty: detected)" in result.output

    def broken() -> str:
        raise GhError("gh api user: HTTP 500")

    monkeypatch.setattr(cli, "gh_detected_login", broken)
    result = invoke("doctor")
    assert result.exit_code == 1 and "fail gh auth: gh is not usable: gh api user: HTTP 500" in result.output


def test_gh_detected_login_asks_gh_not_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from software_factory.sources import github as gh_module

    seen: list[list[str]] = []

    def fake_run(args: list[str], **_: Any) -> str:
        seen.append(list(args))
        return "real-login\n"

    monkeypatch.setattr(gh_module, "default_gh_runner", fake_run)
    assert cli.gh_detected_login() == "real-login"
    assert seen == [["api", "user", "--jq", ".login"]]


def test_doctor_warns_when_visual_repos_share_a_dev_port(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict
) -> None:
    profile_toml["repos"]["overrides"] = {
        "puzzle": {"visual": True, "dev_command": "npm run dev", "dev_port": 5173},
        "stacker": {"visual": True, "dev_command": "npm run dev", "dev_port": 5173},
        "arcade": {"visual": True, "dev_command": "npm run dev", "dev_port": 5174},
    }
    make_profile("personal", profile_toml, secrets)
    result = invoke("doctor")
    assert result.exit_code == 0, result.output  # a warning, not a failure
    assert "warn dev ports: dev_port shared across repos.overrides (5173: puzzle, stacker)" in result.output
    assert "--strictPort" in result.output

    profile_toml["repos"]["overrides"]["stacker"]["dev_port"] = 5175
    make_profile("personal", profile_toml, secrets)
    result = invoke("doctor")
    assert result.exit_code == 0 and "ok   dev ports: 3 repo(s) with a dev_port, all unique" in result.output

    del profile_toml["repos"]["overrides"]
    make_profile("personal", profile_toml, secrets)
    assert "dev ports" not in invoke("doctor").output


def test_doctor_warns_when_scheduled_jobs_were_installed_from_another_data_dir(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    assert "ok   scheduler: jobs installed" in invoke("doctor").output  # installed from this shell: fine

    other = tmp_path / "other data" / "software-factory" / "personal" / "scripts" / "software-factory-personal-tick.cmd"

    def elsewhere(profile: Profile, paths: ProfilePaths) -> str:
        return (
            f'TaskName: \\software-factory-personal-tick\nTask To Run: "{other}"\nStatus: Ready\n'
            "Wrapper: missing (no wrapper for this task under the current data dir; installed from another SOFTWARE_FACTORY_DATA_DIR?)"
        )

    monkeypatch.setattr(cli, "scheduler_status", elsewhere)
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "warn scheduler: jobs installed but they do not run from the current data dir" in result.output
    assert f"they do not carry {DATA_DIR_ENV}=" in result.output
    assert "re-run `factory schedule install` from this shell" in result.output


def test_doctor_warns_when_windows_tasks_keep_the_no_battery_default(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    power = "\nPower Management:                     Stop On Battery Mode, No Start On Batteries"
    monkeypatch.setattr(cli, "scheduler_status", lambda profile, paths: installed_show(profile, paths, extra=power))
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "ok   scheduler: jobs installed" in result.output
    assert "warn scheduler power: the tasks keep Task Scheduler's defaults (no start on battery" in result.output
    assert "re-run `factory schedule install`" in result.output


def test_doctor_warns_without_a_worker_gh_token_and_never_fails_on_it(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict
) -> None:
    """Workers get no gh login unless the profile carries the optional read-only GH_WORKER_TOKEN;
    its absence is the documented default (the engine supplies the context), so: warn, exit 0."""
    without = {k: v for k, v in secrets.items() if k != "GH_WORKER_TOKEN"}
    make_profile("personal", profile_toml, without)
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert (
        "warn worker gh token: GH_WORKER_TOKEN not in the profile .env: workers run gh unauthenticated; "
        "the engine supplies PR/CI/review context" in result.output
    )
    assert "read-only PAT" in result.output and "0 fail, 1 warn" in result.output

    make_profile("personal", profile_toml, secrets)
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "ok   worker gh token: GH_WORKER_TOKEN set in the profile .env" in result.output
    assert "0 fail, 0 warn" in result.output


def test_doctor_scheduler_exceptions_become_a_fail_line_not_a_crash(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    """A ScheduleSpec that cannot be built (ValueError, pydantic included) or a backend error after
    the query used to escape doctor as a traceback, hiding every other check."""
    make_profile("personal", profile_toml, secrets)

    def bad_spec(cls: Any, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("digest_time must be HH:MM (24h), got 'noon'")

    original = scheduler.ScheduleSpec.__dict__["from_profile"]
    monkeypatch.setattr(scheduler.ScheduleSpec, "from_profile", classmethod(bad_spec))
    result = invoke("doctor")
    assert result.exit_code == 1, result.output
    assert not isinstance(result.exception, ValueError) and "Traceback" not in result.output
    assert "fail scheduler: " in result.output and "ValueError: digest_time must be HH:MM" in result.output
    assert "data dir:" in result.output and "checks: 1 fail" in result.output  # the other checks still printed
    monkeypatch.setattr(scheduler.ScheduleSpec, "from_profile", original)

    def broken_drift(spec: Any, text: str) -> list[str]:
        raise scheduler.SchedulerError("schtasks output could not be parsed")

    monkeypatch.setattr(scheduler, "install_drift", broken_drift)
    result = invoke("doctor")
    assert result.exit_code == 1, result.output
    assert "fail scheduler: installed jobs could not be evaluated: SchedulerError: schtasks output" in result.output

    monkeypatch.setattr(cli, "scheduler_status", lambda profile, paths: (_ for _ in ()).throw(ValueError("bad profile dir")))
    result = invoke("doctor")
    assert result.exit_code == 1 and "fail scheduler: could not query the OS scheduler: ValueError: bad profile dir" in result.output


def test_doctor_checks_repo_labels_per_team_when_team_keys_are_configured(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    """``labels_exist`` is workspace-wide: a ``repo:<name>`` label owned by another team looks present
    but sub-issue creation (``ensure_label`` per team) drops it. With ``linear.team_keys`` doctor asks
    per team, learning the team ids from the issues that carry a factory label; a team without one is
    ``n/a``, a team that cannot use a label is a warning, never a failure."""
    profile_toml["linear"] = {"team_keys": ["ENG", "OPS"]}
    make_profile("personal", profile_toml, secrets)
    eng_issue = Issue(
        id="uuid-eng", key="ENG-1", title="x", labels=["factory:build"], state_type="unstarted",
        updated_at="2026-10-01T00:00:00Z", team_id="team-eng-id", team_key="ENG",
    )  # fmt: skip
    factory = ("factory:refine", "factory:plan", "factory:build")
    # repo:repo exists in the workspace (owned by a third team): the workspace check would pass.
    tracker = FakeTracker([eng_issue], existing_labels=(*factory, "repo:repo"), team_labels={"team-eng-id": factory})
    monkeypatch.setattr(cli, "tracker_client", lambda profile: tracker)
    result = invoke("doctor")
    assert result.exit_code == 0, result.output  # warn only
    assert "warn linear repo labels (ENG): 1 of 1 repo:<name> labels not usable by team ENG" in result.output
    assert "repo:repo" in result.output
    assert "n/a  linear repo labels (OPS): team id unknown (no open issue in team OPS carries a factory label yet)" in result.output
    assert ("team-eng-id", "repo:repo") in tracker.ensure_calls
    assert ["repo:repo"] not in tracker.label_queries  # the workspace-wide check is not what decides
    assert "0 fail, 1 warn" in result.output

    tracker = FakeTracker([eng_issue], existing_labels=(*factory, "repo:repo"), team_labels={"team-eng-id": (*factory, "repo:repo")})
    monkeypatch.setattr(cli, "tracker_client", lambda profile: tracker)
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "ok   linear repo labels (ENG): all 1 repo:<name> labels usable by team ENG" in result.output
    assert "0 fail, 0 warn" in result.output

    # Without team_keys the workspace-wide check stays (and ensure_label is never consulted).
    profile_toml["linear"] = {"team_keys": []}
    make_profile("personal", profile_toml, secrets)
    tracker = FakeTracker([eng_issue], existing_labels=(*factory, "repo:repo"))
    monkeypatch.setattr(cli, "tracker_client", lambda profile: tracker)
    result = invoke("doctor")
    assert result.exit_code == 0 and "ok   linear repo labels: all 1 repo:<name> labels exist" in result.output
    assert tracker.ensure_calls == [] and ["repo:repo"] in tracker.label_queries


def test_doctor_repo_static_verdict_counts_only_the_static_checks(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, green_doctor: dict,
    git_repo_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    # A visual repo with every static check passing can only reach 50/115 = 43 of the full score.
    path = git_repo_factory("repo", files={"CLAUDE.md": "# repo\n\n## Verification\n\n```\nnpm test\n```\n", "README.md": "x\n"})
    visual = RepoOverride(visual=True, dev_command="npm run dev", dev_port=5173)
    monkeypatch.setattr(cli, "discover_repos", lambda profile: RepoIndex([repo_info(path, visual)]))
    make_profile("personal", profile_toml, secrets)
    result = invoke("doctor", "--repo", "repo")
    assert result.exit_code == 0, result.output
    assert "ok   repo readiness (static): 50/50 static points, 3/3 static checks ok (score 43/100" in result.output
    assert "n/a    commands_pass: not run" in result.output and "n/a    dev_server: not run" in result.output
    assert "warn   commands_pass" not in result.output and "warn   dev_server" not in result.output
    assert "0 fail, 0 warn" in result.output

    (path / "CLAUDE.md").unlink()
    result = invoke("doctor", "--repo", "repo")
    assert "warn repo readiness (static): 0/50 static points, 0/3 static checks ok" in result.output
    assert "warn   agent_doc_present: no CLAUDE.md or AGENTS.md" in result.output


def test_slack_auth_test_uses_the_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True, "user": "factory", "team": "Acme"})

    ok, detail = cli.slack_auth_test("xoxb-1", http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert ok and detail == "bot factory in workspace Acme"
    assert seen == {"auth": "Bearer xoxb-1", "url": cli.SLACK_AUTH_TEST}
    bad = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": False, "error": "invalid_auth"})))
    assert cli.slack_auth_test("xoxb-1", http=bad) == (False, "invalid_auth")


# ============================================================================= status / flags
def test_status_shows_flags_worktrees_prs_and_runs(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, paths: ProfilePaths
) -> None:
    make_profile("personal", profile_toml, secrets)
    with State(paths.state_db) as state:
        state.add_flag("o/r#1", "ci", "push refused", "sensitive path\nworktree: C:/wt/x")
        state.keep_worktree("o/r#1", "ci", "C:/wt/x", "sensitive path")
        state.register_pr("o/r", 7, "factory/abc-1", "build", "ABC-1", "https://github.com/o/r/pull/7")
        for i in range(12):
            state.record_run({"ts": f"2026-10-01T10:{i:02d}:00+00:00", "kind": "ci", "target": f"o/r#{i}", "outcome": "done", "pushed": i % 2 == 0, "note": "fixed"})
    result = invoke("status")
    assert result.exit_code == 0, result.output
    assert "profile personal: mode full, provider oauth, idle" in result.output
    assert "enabled kinds: ci, conflict, rc, rv" in result.output
    assert "open flags (1):" in result.output and "push refused" in result.output and "sensitive path worktree: C:/wt/x" in result.output
    assert "kept worktrees (1):" in result.output and "C:/wt/x" in result.output
    assert "factory-opened PRs (1):" in result.output and "o/r#7" in result.output and "ABC-1" in result.output
    assert "last 10 run(s):" in result.output and "o/r#11" in result.output and "o/r#1 " not in result.output.split("last 10")[1]


def test_status_reports_a_running_tick(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, paths: ProfilePaths
) -> None:
    make_profile("personal", profile_toml, secrets)
    lock = FileLock(str(paths.lock_file))
    with lock:
        result = invoke("status")
    assert result.exit_code == 0 and "tick running" in result.output


def test_flags_list_ack_and_all(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, paths: ProfilePaths
) -> None:
    make_profile("personal", profile_toml, secrets)
    with State(paths.state_db) as state:
        first = state.add_flag("o/r#1", "ci", "worker failed", "timeout")
        state.add_flag("ABC-1", "build", "unresolved repo", "")
    result = invoke("flags")
    assert result.exit_code == 0 and "open flags (2):" in result.output and "worker failed" in result.output

    result = invoke("flags", "--ack", str(first))
    assert result.exit_code == 0 and f"acked flag {first}" in result.output
    with State(paths.state_db) as state:
        assert [f["id"] for f in state.flags()] == [first + 1]
    assert "open flags (1):" in invoke("flags").output
    result = invoke("flags", "--all")
    assert "all flags (2):" in result.output and "acked" in result.output

    result = invoke("flags", "--ack", "999")
    assert result.exit_code == 1 and "not an open flag" in result.output


# ====================================================================================== report
def test_report_renders_and_digest_goes_through_the_notifier(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, paths: ProfilePaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    with State(paths.state_db) as state:
        state.record_run({"ts": "2026-10-01T10:00:00+00:00", "kind": "ci", "target": "o/r#1", "outcome": "done", "pushed": True, "cost_usd": 0.5})
    monkeypatch.setattr(cli, "gh_client", lambda profile: FakeGh())
    sent: list[str] = []

    class FakeNotifier:
        def __init__(self, profile: Profile, audit: Audit, **_: Any) -> None:
            self.ok = True

        def slack_dm(self, text: str) -> bool:
            sent.append(text)
            return FakeNotifier.result

    FakeNotifier.result = True  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "Notifier", FakeNotifier)
    result = invoke("report", "--since", "2026-09-01T00:00:00Z", "--digest")
    assert result.exit_code == 0, result.output
    assert "refreshed 0 factory-touched PR(s)" in result.output
    assert "Factory report since 2026-09-01" in result.output and "ci" in result.output and "$0.50" in result.output
    assert "digest sent as a Slack DM" in result.output
    assert sent and sent[0].startswith("*Factory digest*")

    FakeNotifier.result = False  # type: ignore[attr-defined]
    result = invoke("report", "--digest")
    assert result.exit_code == 1 and "digest not sent" in result.output

    result = invoke("report", "--since", "soon")
    assert result.exit_code == 2 and "invalid --since" in result.output


def test_report_survives_a_failing_outcome_refresh(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_profile("personal", profile_toml, secrets)

    def broken(profile: Profile) -> Any:
        raise GhError("gh executable not found on PATH")

    monkeypatch.setattr(cli, "gh_client", broken)
    result = invoke("report")
    assert result.exit_code == 0, result.output
    assert "warning: PR outcome refresh skipped (GhError: gh executable not found" in result.output
    assert "Factory report since" in result.output


# =============================================================================== revert-refine
def test_revert_refine_restores_through_effects(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, paths: ProfilePaths,
    patch_build: Callable[..., dict],
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    effects = FakeEffects()
    patch_build(effects=effects)
    with State(paths.state_db) as state:
        state.save_refinement("uuid-1", "ABC-1", "Old title", "old body", "New title", "new body")
        state.set_refinement_status("ABC-1", "applied")
    result = invoke("revert-refine", "ABC-1")
    assert result.exit_code == 0, result.output
    assert "revert-refine ABC-1 done ABC-1: original title and description restored" in result.output
    assert effects.calls == [("update_issue_text", "uuid-1", "Old title", "old body")]
    with State(paths.state_db) as state:
        assert state.get_refinement("ABC-1")["status"] == "reverted"

    result = invoke("revert-refine", "ABC-404")
    assert result.exit_code == 1 and "no refinement saved for ABC-404" in result.output


# ==================================================================================== schedule
@pytest.fixture
def fake_scheduler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[..., FakeRunner]:
    monkeypatch.setattr(scheduler, "current_os", lambda: "windows")
    monkeypatch.setattr(scheduler, "find_uv", lambda: tmp_path / "uv.exe")

    def _install(handler: Callable[[list[str]], tuple[int, str, str]] | None = None) -> FakeRunner:
        fake = FakeRunner(handler)
        monkeypatch.setattr(scheduler, "default_runner", fake)
        return fake

    return _install


def test_schedule_show_uses_the_fake_runner(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, fake_scheduler: Callable[..., FakeRunner]
) -> None:
    make_profile("personal", profile_toml, secrets)
    fake = fake_scheduler(lambda argv: (1, "", NOT_FOUND))
    result = invoke("schedule", "show")
    assert result.exit_code == 0, result.output
    assert "software-factory-personal-tick: not installed" in result.output
    assert "software-factory-personal-digest: not installed" in result.output
    assert all(argv[:2] == ["schtasks", "/Query"] for argv in fake.argvs) and len(fake.argvs) == 2


def test_schedule_install_and_uninstall_through_the_fake_runner(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict,
    fake_scheduler: Callable[..., FakeRunner], paths: ProfilePaths,
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    fake = fake_scheduler()
    result = invoke("schedule", "install", "--interval", "15")
    assert result.exit_code == 0, result.output
    assert "install (windows) profile personal" in result.output
    assert "job   software-factory-personal-tick" in result.output and "job   software-factory-personal-digest" in result.output
    creates = [argv for argv in fake.argvs if argv[:2] == ["schtasks", "/Create"]]
    assert len(creates) == 2 and "15" in creates[0]
    assert (paths.scripts / "software-factory-personal-tick.cmd").is_file()

    result = invoke("schedule", "uninstall")
    assert result.exit_code == 0, result.output
    assert "uninstall (windows) profile personal" in result.output
    assert not (paths.scripts / "software-factory-personal-tick.cmd").exists()

    fake_scheduler(lambda argv: (1, "", "Access is denied."))
    result = invoke("schedule", "install", "--no-digest")
    assert result.exit_code == 1 and "schedule install failed: SchedulerError" in result.output


def test_schedule_install_embeds_the_data_dir_and_show_prints_it(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict,
    fake_scheduler: Callable[..., FakeRunner], paths: ProfilePaths,
) -> None:  # fmt: skip
    """The shell's SOFTWARE_FACTORY_DATA_DIR (set by conftest) must reach the scheduled jobs: without
    it a scheduled tick would use %LOCALAPPDATA% while status/flags/cleanup look at this dir."""
    make_profile("personal", profile_toml, secrets)
    fake = fake_scheduler()
    assert invoke("schedule", "install").exit_code == 0
    wrapper = (paths.scripts / "software-factory-personal-tick.cmd").read_text(encoding="utf-8")
    data_dir = scheduler.env_data_dir()
    assert data_dir is not None and f'set "{DATA_DIR_ENV}={data_dir}"' in wrapper
    assert any(argv[0] == "powershell" and "-AllowStartIfOnBatteries" in argv[-1] for argv in fake.argvs)

    fake_scheduler(lambda argv: (0, f"TaskName: {argv[3]}\nStatus: Ready\n", ""))
    result = invoke("schedule", "show")
    assert result.exit_code == 0, result.output
    env_line = next(line for line in result.output.splitlines() if line.startswith("Wrapper Env:"))
    assert f"{DATA_DIR_ENV}={data_dir}" in env_line


# ============================================================================ cleanup / release
@pytest.fixture
def worktree_dirs(paths: ProfilePaths) -> tuple[Path, Path]:
    unkept, kept = paths.worktrees / "pr1-aaaa", paths.worktrees / "pr2-bbbb"
    for d in (unkept, kept):
        d.mkdir(parents=True)
        (d / "file.txt").write_text("x", encoding="utf-8")
    with State(paths.state_db) as state:
        state.keep_worktree("o/r#2", "conflict", str(kept), "unresolved hunks")
    return unkept, kept


def test_cleanup_removes_unkept_only_and_all_releases_after_confirmation(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, paths: ProfilePaths,
    worktree_dirs: tuple[Path, Path],
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    unkept, kept = worktree_dirs
    result = invoke("cleanup")
    assert result.exit_code == 0, result.output
    assert f"removed {unkept}" in result.output and "removed 1 unkept worktree(s); 1 kept" in result.output
    assert not unkept.exists() and kept.exists()

    result = invoke("cleanup", "--all", input="n\n")
    assert result.exit_code == 1 and "Aborted" in result.output and kept.exists()

    result = invoke("cleanup", "--all", "--yes")
    assert result.exit_code == 0, result.output
    assert "kept: o/r#2 (conflict)" in result.output and "released 1 kept worktree(s); removed 1" in result.output
    assert not kept.exists()
    with State(paths.state_db) as state:
        assert state.kept_worktrees() == []


def test_cleanup_refuses_while_a_tick_runs(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, paths: ProfilePaths,
    worktree_dirs: tuple[Path, Path],
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    with FileLock(str(paths.lock_file)):
        result = invoke("cleanup")
    assert result.exit_code == 1 and "a tick is running" in result.output
    assert worktree_dirs[0].exists()


def test_release_and_release_remove(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, paths: ProfilePaths,
    worktree_dirs: tuple[Path, Path],
) -> None:  # fmt: skip
    make_profile("personal", profile_toml, secrets)
    unkept, kept = worktree_dirs
    result = invoke("release", "o/r#2")
    assert result.exit_code == 0 and f"released o/r#2 (conflict): {kept}" in result.output
    assert kept.exists()
    with State(paths.state_db) as state:
        assert state.kept_worktree("o/r#2") is None
        state.keep_worktree("o/r#2", "conflict", str(kept), "again")
    result = invoke("release", "o/r#2", "--remove")
    assert result.exit_code == 0, result.output
    assert f"removed {kept}" in result.output and not kept.exists() and unkept.exists()

    result = invoke("release", "o/r#9")
    assert result.exit_code == 1 and "no kept worktree for 'o/r#9'" in result.output


# ========================================================================================= run
def test_run_rejects_bad_kinds_and_targets(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, patch_build: Callable[..., dict]
) -> None:
    make_profile("personal", profile_toml, secrets)
    patch_build()
    result = invoke("run", "bogus", "x")
    assert result.exit_code == 2 and "unknown kind 'bogus'" in result.output

    result = invoke("run", "ci", "nothash")
    assert result.exit_code == 2 and "not a pull request target" in result.output

    result = invoke("run", "ci", "other/thing#1")
    assert result.exit_code == 2 and "repo 'other/thing' is not managed" in result.output and SLUG in result.output

    result = invoke("run", "ready", "nope")
    assert result.exit_code == 2 and "unknown repo 'nope'" in result.output


def test_run_pr_kinds_build_the_action_from_current_state(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, patch_build: Callable[..., dict]
) -> None:
    make_profile("personal", profile_toml, secrets)
    gh = FakeGh({(SLUG, 1): open_pr(1), (SLUG, 2): open_pr(2, failing=True), (SLUG, 3): open_pr(3, state="MERGED")})
    patch_build(gh=gh)
    result = invoke("run", "ci", "repo#1", "--dry-run")
    assert result.exit_code == 1 and "no ci trigger right now (no failing check" in result.output

    result = invoke("run", "ci", f"{SLUG}#2", "--dry-run")
    assert result.exit_code == 0, result.output
    assert f"action: ci       {SLUG}#2  PR 2 (version {HEAD})" in result.output
    assert f"ci {SLUG}#2 dry_run not-pushed would run (--dry-run)" in result.output

    result = invoke("run", "rv", f"https://github.com/{SLUG}/pull/1", "--dry-run")
    assert result.exit_code == 0, result.output
    assert f"rv {SLUG}#1 dry_run not-pushed would run (--dry-run)" in result.output

    result = invoke("run", "rc", "repo#3", "--dry-run")
    assert result.exit_code == 1 and "is merged" in result.output

    result = invoke("run", "conflict", "repo#9", "--dry-run")
    assert result.exit_code == 1 and "run failed: GhError" in result.output


def test_run_issue_kinds_need_a_tracker_and_use_the_issue(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, patch_build: Callable[..., dict]
) -> None:
    make_profile("personal", profile_toml, secrets)
    patch_build()  # no tracker
    result = invoke("run", "build", "ABC-1", "--dry-run")
    assert result.exit_code == 1 and "no tracker configured" in result.output

    issue = Issue(id="uuid-1", key="ABC-1", title="Add thing", labels=["factory:build"], state_type="unstarted", updated_at="2026-10-01T00:00:00Z", branch_name="abc-1-add-thing", team_id="t1")
    patch_build(tracker=FakeTracker([issue]))
    result = invoke("run", "build", "ABC-1", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "action: build    ABC-1  Add thing (version uuid-1)" in result.output
    assert "build ABC-1 dry_run not-pushed would run (--dry-run)" in result.output

    result = invoke("run", "refine", "ABC-1", "--dry-run")
    assert result.exit_code == 0 and "(version 2026-10-01T00:00:00Z)" in result.output

    result = invoke("run", "plan", "ABC-404", "--dry-run")
    assert result.exit_code == 1 and "run failed: TrackerError: issue ABC-404 not found" in result.output


def test_issue_action_mirrors_the_scan_shape() -> None:
    issue = Issue(id="u", key="ABC-2", title="T", labels=["factory:build"], state_type="unstarted", updated_at="2026-01-01T00:00:00Z", branch_name="b", team_id="t")
    build = cli.issue_action(Kind.BUILD, issue)
    assert build.version == "u" and build.target == "ABC-2"
    assert build.context == {"issue_id": "u", "labels": ["factory:build"], "state_type": "unstarted", "branch_name": "b", "team_id": "t", "unfinished_blockers": []}
    plan = cli.issue_action(Kind.PLAN, issue)
    assert plan.version == "2026-01-01T00:00:00Z" and "branch_name" not in plan.context


def test_run_ready_and_ready_command_dry_run(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, patch_build: Callable[..., dict]
) -> None:
    make_profile("personal", profile_toml, secrets)
    wt = FakeWtManager(sha="c" * 40)
    patch_build(wt=wt)
    result = invoke("run", "ready", "repo", "--dry-run")
    assert result.exit_code == 0, result.output
    assert f"ready {SLUG} dry_run not-pushed would run (--dry-run)" in result.output
    fetch = wt.calls[0]
    at = fetch.index("fetch")
    assert fetch[at : at + 3] == ["fetch", "--no-tags", "origin"]
    # the engine's fetch in the user's clone runs with hooks disabled (``-c core.hooksPath=...``)
    assert any(arg.startswith("core.hooksPath=") for arg in fetch[:at]), fetch

    result = invoke("ready", "repo", "--dry-run")
    assert result.exit_code == 0, result.output
    assert f"action: ready    {SLUG}  Improve factory readiness: {SLUG} (version {'c' * 40})" in result.output

    result = invoke("ready", "nope")
    assert result.exit_code == 2 and "unknown repo 'nope'" in result.output


class RaisingHandler:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def run(self, action: Action, deps: Deps) -> Outcome:
        raise self.exc


def test_run_and_ready_exit_2_on_a_provider_limit_like_an_auth_failure(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict,
    patch_build: Callable[..., dict], monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    """Regression: ``factory run`` exited 1 ("an action failed") when the provider refused the
    session, although nothing about the target failed and a tick exits 2 for the same stop."""
    make_profile("personal", profile_toml, secrets)
    patch_build(gh=FakeGh({(SLUG, 2): open_pr(2, failing=True)}), wt=FakeWtManager(sha="c" * 40))
    monkeypatch.setattr(dispatch, "get_handler", lambda kind: RaisingHandler(ProviderUnavailable("usage limit reached")))
    result = invoke("run", "ci", "repo#2")
    assert result.exit_code == 2, result.output
    assert "provider unavailable: usage limit reached" in result.output
    result = invoke("ready", "repo")
    assert result.exit_code == 2, result.output

    monkeypatch.setattr(dispatch, "get_handler", lambda kind: RaisingHandler(AuthFailure("gh token expired")))
    result = invoke("run", "ci", "repo#2")
    assert result.exit_code == 2 and "auth failure" in result.output

    monkeypatch.setattr(dispatch, "get_handler", lambda kind: FakeHandler(Outcome(status="failed", note="provider said no")))
    assert invoke("run", "ci", "repo#2").exit_code == 1  # an ordinary failure is still 1
    monkeypatch.setattr(dispatch, "get_handler", lambda kind: FakeHandler(Outcome(status="done", note="ok")))
    assert invoke("run", "ci", "repo#2").exit_code == 0


def test_tick_and_run_help_mention_provider_limits(invoke: Callable[..., Any]) -> None:
    for command in ("tick", "run"):
        result = invoke(command, "--help")
        assert result.exit_code == 0
        assert "provider rate / usage limit" in " ".join(result.output.split()), command


def test_run_respects_observe_mode(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, patch_build: Callable[..., dict]
) -> None:
    profile_toml["profile"]["mode"] = "observe"
    make_profile("personal", profile_toml, secrets)
    patch_build(gh=FakeGh({(SLUG, 2): open_pr(2, failing=True)}))
    result = invoke("run", "ci", "repo#2")
    assert result.exit_code == 0, result.output
    assert f"ci {SLUG}#2 dry_run not-pushed observe mode: would run" in result.output


# =================================================================================== readiness
def test_readiness_no_run_scores_managed_repos(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, git_repo_factory: Callable[..., Path]
) -> None:
    git_repo_factory("repo", files={"CLAUDE.md": "# repo\n\n## Testing\n\n- `npm test`\n", "README.md": "x\n"})
    git_repo_factory("other", origin="git@github.com:JaneDoe/other.git", files={"README.md": "x\n"})
    make_profile("personal", profile_toml, secrets)
    result = invoke("readiness", "--no-run")
    assert result.exit_code == 0, result.output
    assert "static checks only" in result.output
    lines = {ln.split()[0]: ln for ln in result.output.splitlines() if ln.startswith("JaneDoe/")}
    assert set(lines) == {"JaneDoe/repo", "JaneDoe/other"}
    assert "missing commands_pass" in lines["JaneDoe/repo"] and "commands_discovered" not in lines["JaneDoe/repo"]
    assert "missing commands_discovered" in lines["JaneDoe/other"]

    result = invoke("readiness", "--repo", "other", "--no-run")
    assert result.exit_code == 0 and "FAIL agent_doc_present: no CLAUDE.md or AGENTS.md" in result.output
    # checks that were deliberately not run are n/a, not failures
    assert "n/a  commands_pass: not run" in result.output and "FAIL commands_pass" not in result.output
    result = invoke("readiness", "--repo", "nope", "--no-run")
    assert result.exit_code == 2 and "'nope' is not a managed repo" in result.output


def test_profile_option_selects_the_profile(
    invoke: Callable[..., Any], make_profile: Callable[..., Path], profile_toml: dict, secrets: dict, profiles_dir: Path
) -> None:
    make_profile("work", {**profile_toml, "profile": {"mode": "observe"}}, secrets)
    result = invoke("-p", "work", "status")
    assert result.exit_code == 0, result.output
    assert "profile work: mode observe" in result.output
    assert load_profile("work", profiles_dir).name == "work"
