"""Issue-kind handlers (``build`` / ``ready`` / ``refine`` / ``plan``) and ``revert_refine``.

Fake tracker + fake gh + fake worker + a real tmp git world (local bare origin) wherever branching
and pushing matter. Writes go through the real ``LiveEffects`` (the fakes record them) or through
``RecordingEffects`` with fakes that raise on any write, which proves the dry path never writes.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from software_factory.actions import Action, Kind
from software_factory.config import DEFAULT_DISCLOSURE, Profile, RepoOverride
from software_factory.discover import git_head_sha
from software_factory.effects import REPO_MARKER, LiveEffects, RecordingEffects
from software_factory.evidence import END, START
from software_factory.kinds import base as kb
from software_factory.kinds import build as build_mod
from software_factory.kinds import get_handler
from software_factory.kinds import plan as plan_mod
from software_factory.kinds import ready as ready_mod
from software_factory.kinds import refine as refine_mod
from software_factory.kinds.base import Deps, Outcome, ProviderUnavailable, finish_code_worker
from software_factory.kinds.plan import find_cycle, validate_specs
from software_factory.kinds.ready import ready_action
from software_factory.kinds.refine import revert_refine
from software_factory.notify import Audit, Notifier
from software_factory.paths import ProfilePaths
from software_factory.readiness import Readiness, ReadinessCheck
from software_factory.repos import RepoIndex, RepoInfo
from software_factory.resolve import Resolution, Resolver
from software_factory.runner import RunSummary, WorkerRun
from software_factory.sources.github import GhAuthError
from software_factory.sources.models import PullRequest
from software_factory.state import State
from software_factory.trackers.base import Issue, IssueRef, TrackerAuthError, TrackerError
from software_factory.workers.schemas import (
    BuildOutput,
    CiOutput,
    PlanOutput,
    ReadyOutput,
    RefineOutput,
    SubIssueSpec,
    VerificationRun,
)
from software_factory.worktree import WorktreeManager

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

SLUG = "JaneDoe/repo"
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
FOOTER_LAST_LINE = DEFAULT_DISCLOSURE.splitlines()[-1]


# ------------------------------------------------------------------------------------- fakes
class FakeToaster:
    """Desktop toaster stand-in: records what was sent; ``probe`` is evaluated at send time so a
    test can see what state existed *before* the notification went out."""

    def __init__(self, *, fail: bool = False, probe: Callable[[], Any] | None = None) -> None:
        self.fail = fail
        self.probe = probe
        self.sent: list[tuple[str, str]] = []
        self.seen: list[Any] = []

    def send(self, title: str, message: str) -> None:
        if self.probe is not None:
            self.seen.append(self.probe())
        if self.fail:
            raise RuntimeError("no desktop session")
        self.sent.append((title, message))


class FakeGh:
    def __init__(self, *, open_prs: dict[tuple[str, str], PullRequest] | None = None, raise_on_write: bool = False) -> None:
        self.open_prs = open_prs or {}
        self.raise_on_write = raise_on_write
        self.created: list[dict[str, Any]] = []
        self.calls: list[tuple[str, ...]] = []
        self.edits: list[tuple[str, int, str]] = []

    def auth_check(self) -> None:
        pass

    def login(self) -> str:
        return "JaneDoe"

    def find_open_pr_for_branch(self, repo: str, branch: str) -> PullRequest | None:
        self.calls.append(("find_open_pr_for_branch", repo, branch))
        return self.open_prs.get((repo, branch))

    def create_draft_pr(self, repo: str, head: str, base: str, title: str, body: str) -> PullRequest:
        if self.raise_on_write:
            raise AssertionError("gh write attempted: create_draft_pr")
        number = 42 + len(self.created)
        self.created.append({"repo": repo, "head": head, "base": base, "title": title, "body": body, "number": number})
        return PullRequest(
            repo=repo, number=number, url=f"https://github.com/{repo}/pull/{number}", title=title, body=body,
            is_draft=True, head_ref_name=head, base_ref_name=base,
        )  # fmt: skip

    def pr_body(self, repo: str, number: int) -> str:
        return next((c["body"] for c in self.created if c["number"] == number), "")

    def edit_pr_body(self, repo: str, number: int, body: str) -> None:
        if self.raise_on_write:
            raise AssertionError("gh write attempted: edit_pr_body")
        self.edits.append((repo, number, body))


class FakeTracker:
    name = "fake"

    def __init__(
        self, issues: list[Issue] | None = None, *, existing_labels: tuple[str, ...] = ("repo:repo",),
        raise_on_write: bool = False, read_error: Exception | None = None,
    ) -> None:  # fmt: skip
        self.issues: dict[str, Issue] = {i.id: i for i in (issues or [])}
        self.existing = {lbl.lower() for lbl in existing_labels}
        self.raise_on_write = raise_on_write
        self.read_error = read_error
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._counter = 0

    def ping(self) -> str:
        return "me"

    def get_issue(self, ref: str) -> Issue:
        if self.read_error is not None:
            raise self.read_error
        for issue in self.issues.values():
            if ref in (issue.id, issue.key):
                return issue
        raise TrackerError(f"issue {ref} not found")

    def labels_exist(self, names: list[str]) -> dict[str, bool]:
        return {n: n.lower() in self.existing for n in names}

    def _write(self, name: str, *args: Any) -> None:
        if self.raise_on_write:
            raise AssertionError(f"tracker write attempted: {name}")
        self.calls.append((name, args))

    def create_sub_issue(self, parent: Issue, title: str, description: str, labels: list[str]) -> Issue:
        self._write("create_sub_issue", parent.key, title, description, list(labels))
        self._counter += 1
        child = Issue(
            id=f"child-{self._counter}", key=f"{parent.team_key or 'ABC'}-{100 + self._counter}", title=title,
            description=description, labels=list(labels), team_id=parent.team_id, team_key=parent.team_key,
            state_type="backlog", parent=IssueRef(id=parent.id, key=parent.key),
        )  # fmt: skip
        self.issues[child.id] = child
        return child

    def add_blocked_by(self, issue_id: str, blocker_id: str) -> None:
        self._write("add_blocked_by", issue_id, blocker_id)

    def move_to_state_type(self, issue_id: str, state_type: str) -> None:
        self._write("move_to_state_type", issue_id, state_type)

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None:
        self._write("update_issue_text", issue_id, title, description)

    def remove_label(self, issue_id: str, label: str) -> None:
        self._write("remove_label", issue_id, label)


class FakeResolver:
    def __init__(self, resolution: Resolution) -> None:
        self.resolution = resolution
        self.calls: list[str] = []

    async def resolve(self, issue: Issue) -> Resolution:
        self.calls.append(issue.key)
        return self.resolution


class FakeWorker:
    """``run_worker`` stand-in: records the call, lets ``on_run`` act in the worktree (commit files,
    write screenshots), returns the configured structured output."""

    def __init__(
        self, output: dict[str, Any] | None = None, *, error: str | None = None,
        on_run: Callable[[dict[str, Any]], None] | None = None, timed_out: bool = False,
    ) -> None:  # fmt: skip
        self.output = output
        self.error = error
        self.on_run = on_run
        self.timed_out = timed_out
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> WorkerRun:
        self.calls.append(kwargs)
        if self.on_run is not None:
            self.on_run(kwargs)
        return WorkerRun(
            output=self.output, error=self.error, timed_out=self.timed_out,
            summary=RunSummary(cost_usd=0.5, duration_ms=1234, usage={"input_tokens": 9}),
        )  # fmt: skip


async def no_worker(**_: Any) -> WorkerRun:
    raise AssertionError("the worker must not run in this scenario")


def raising_git(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
    raise AssertionError(f"git invoked by RecordingEffects: {args}")


# ---------------------------------------------------------------------------------- fixtures
@pytest.fixture
def paths(tmp_path: Path) -> ProfilePaths:
    return ProfilePaths.for_profile("personal", root=tmp_path / "data root").ensure()


@pytest.fixture
def quiet_profile(
    make_profile: Callable[..., Path], load: Callable[[str], Profile], repos_root: Path
) -> Callable[..., Profile]:
    """Toast on (the test toaster records it), Slack off, repos.root = the tmp repos root."""

    def _build(mode: str = "full", **fields: Any) -> Profile:
        make_profile(
            "personal",
            {
                "profile": {"mode": mode, "session_timeout_min": 7, **fields},
                "notify": {"desktop": True, "slack": False},
                "github": {"owners": ["JaneDoe"], "login": "JaneDoe"},
                "repos": {"root": str(repos_root)},
            },
            {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test"},
        )
        return load("personal")

    return _build


@pytest.fixture
def world(bare_repo_factory: Callable[..., Path], git_repo_factory: Callable[..., Path]) -> dict:
    bare = bare_repo_factory("origin")
    clone = git_repo_factory("repo", origin=bare)
    repo = RepoInfo(slug=SLUG, name="repo", dirname="repo", path=clone, default_branch="main", override=RepoOverride())
    return {"bare": bare, "clone": clone, "repo": repo}


@pytest.fixture
def make_deps(paths: ProfilePaths, quiet_profile: Callable[..., Profile], world: dict) -> Iterator[Callable[..., Deps]]:
    created: list[Deps] = []

    def _make(
        *, mode: str = "full", tracker: Any = None, gh: FakeGh | None = None, resolver: Any = None,
        worker: Any = None, toaster: FakeToaster | None = None, repos: RepoIndex | None = None,
        recording: bool = False,
    ) -> Deps:  # fmt: skip
        profile = quiet_profile(mode)
        state = State(paths.state_db)
        audit = Audit(paths.audit_log, profile.name)
        gh = gh or FakeGh()
        effects: Any
        if recording:
            effects = RecordingEffects(profile, paths, state, gh, tracker, audit, raising_git)  # type: ignore[arg-type]
        else:
            effects = LiveEffects(profile, paths, state, gh, tracker, audit)  # type: ignore[arg-type]
        deps = Deps(
            profile=profile, paths=paths, state=state, gh=gh, tracker=tracker, repos=repos or RepoIndex([world["repo"]]),  # type: ignore[arg-type]
            effects=effects, notifier=Notifier(profile, audit, toaster=toaster or FakeToaster()), audit=audit,
            wt_manager=WorktreeManager(paths.worktrees), resolver=resolver, run_worker=worker or FakeWorker(),
            now=lambda: NOW, run_id="run-1",
        )  # fmt: skip
        created.append(deps)
        return deps

    yield _make
    for deps in created:
        deps.state.close()


# ----------------------------------------------------------------------------------- helpers
def commit_file(git: Callable[..., str], cwd: Path, name: str, content: str, message: str = "change") -> str:
    target = cwd / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(["add", "-A"], cwd)
    git(["commit", "-q", "-m", message], cwd)
    return git(["rev-parse", "HEAD"], cwd)


def committing(git: Callable[..., str], name: str = "src/feature.py", content: str = "x = 1\n") -> Callable[[dict], None]:
    return lambda call: commit_file(git, Path(call["cwd"]), name, content)


def run_handler(kind: Kind, action: Action, deps: Deps) -> Outcome:
    return asyncio.run(get_handler(kind).run(action, deps))


def notifies(deps: Deps) -> list[dict]:
    return [r for r in deps.audit.read() if r["event"] == "notify"]


def effect_events(deps: Deps) -> list[str]:
    return [r["event"] for r in deps.audit.read() if r["event"].startswith(("effect.", "would_"))]


def build_issue(**over: Any) -> Issue:
    base: dict[str, Any] = dict(
        id="iss-1", key="ABC-1", title="Build it",
        description="Add the thing.\n\n## Acceptance criteria\n\n1. AC one\n2. AC two",
        url="https://linear.app/x/issue/ABC-1", labels=["factory:build"], state_type="unstarted", state_name="Todo",
        team_id="team-1", team_key="ABC", updated_at="2026-09-30T10:00:00Z", branch_name="tester/abc-1-build-it",
    )  # fmt: skip
    base.update(over)
    return Issue(**base)


def issue_action(kind: Kind, issue: Issue) -> Action:
    return Action(
        kind=kind, version=issue.id if kind == Kind.BUILD else issue.updated_at, title=issue.title,
        issue_id=issue.id, issue_key=issue.key, url=issue.url,
        context={"issue_id": issue.id, "labels": list(issue.labels), "state_type": issue.state_type},
    )  # fmt: skip


def build_output(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": "done",
        "summary": "implemented the thing",
        "push_ok": True,
        "push_reasons": [],
        "verification": [{"command": "uv run pytest -q", "exit_code": 0, "passed": True, "duration_s": 3.2}],
        "evidence": {
            "ac_proofs": [
                {"criterion": "AC one", "proof": "tests/test_x.py::test_one", "status": "proven"},
                {"criterion": "AC two", "proof": "", "status": "unproven"},
            ],
            "screenshots": [],
            "risk_notes": ["touches the config loader"],
        },
        "pr_title": "Add the thing",
        "pr_summary": "This adds the thing.\n\nReviewers: start with src/feature.py.",
        "tests_added": ["tests/test_x.py::test_one"],
    }
    base.update(over)
    return base


def resolved() -> FakeResolver:
    return FakeResolver(Resolution(repo=SLUG, confidence=1.0, source="label", candidates=[SLUG]))


def seed_full_readiness(deps: Deps, repo: RepoInfo, score: int) -> None:
    sha = git_head_sha(repo.path)
    readiness = Readiness(
        repo=repo.slug, score=score, head_sha=sha, computed_at=NOW.isoformat(),
        checks=[ReadinessCheck(name="commands_pass", ok=True, detail="2 command(s) passed", weight=25)],
    )  # fmt: skip
    deps.state.cache_set(
        "readiness", f"readiness:{repo.slug}",
        {"head_sha": sha, "run_commands": True, "readiness": readiness.model_dump(mode="json")},
    )  # fmt: skip


def heading_positions(prompt: str, *headings: str) -> list[int]:
    """Offsets of the heading *lines* (a backticked mention inside a section does not count)."""
    positions = [("\n" + prompt).find(f"\n{h}\n") for h in headings]
    assert all(p >= 0 for p in positions), f"missing heading(s) in prompt: {[h for h, p in zip(headings, positions, strict=True) if p < 0]}"
    return positions


# ------------------------------------------------------------------------------ registration
def test_handlers_are_registered_by_kind() -> None:
    assert get_handler(Kind.BUILD) is build_mod.HANDLER and build_mod.HANDLER.kind == Kind.BUILD
    assert get_handler(Kind.READY) is ready_mod.HANDLER and ready_mod.HANDLER.kind == Kind.READY
    assert get_handler(Kind.REFINE) is refine_mod.HANDLER and refine_mod.HANDLER.kind == Kind.REFINE
    assert get_handler(Kind.PLAN) is plan_mod.HANDLER and plan_mod.HANDLER.kind == Kind.PLAN


# ------------------------------------------------------------------------------------- build
def test_build_happy_path_opens_draft_pr_with_evidence_and_screenshots(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths, git: Callable[..., str]
) -> None:
    override = RepoOverride(visual=True, dev_command="npm run dev", dev_port=5173, dev_routes=["/", "/play"])
    repo = world["repo"].model_copy(update={"override": override})
    issue = build_issue()
    tracker = FakeTracker([issue])
    resolver = resolved()

    def work(call: dict[str, Any]) -> None:
        cwd = Path(call["cwd"])
        commit_file(git, cwd, "src/feature.py", "x = 1\ny = 2\n")
        shots = cwd / ".factory" / "evidence"
        shots.mkdir(parents=True)
        (shots / "home.png").write_bytes(b"\x89PNG before")
        (shots / "after-home.png").write_bytes(b"\x89PNG after")

    evidence = build_output()["evidence"]
    evidence["screenshots"] = [
        {"path": ".factory/evidence/home.png", "caption": "home", "phase": "before"},
        {"path": ".factory/evidence/after-home.png", "caption": "home", "phase": "after", "kind": "png"},
        {"path": ".factory/evidence/missing.png", "caption": "gone", "phase": "after"},
    ]
    worker = FakeWorker(build_output(evidence=evidence), on_run=work)
    deps = make_deps(tracker=tracker, resolver=resolver, worker=worker, repos=RepoIndex([repo]))
    seed_full_readiness(deps, repo, 90)
    action = issue_action(Kind.BUILD, issue)

    outcome = run_handler(Kind.BUILD, action, deps)

    assert outcome.status == "done" and outcome.pushed is True and outcome.record_handled
    assert outcome.pr_url == f"https://github.com/{SLUG}/pull/42" and outcome.cost_usd == 0.5 and outcome.duration_ms == 1234
    assert "screenshot not found: .factory/evidence/missing.png" in outcome.note

    # the branch reached origin, the worktree is gone, the screenshots sit on factory/evidence
    assert git(["ls-remote", "--heads", "origin", "tester/abc-1-build-it"], world["clone"]).endswith("refs/heads/tester/abc-1-build-it")
    assert list(paths.worktrees.iterdir()) == []
    tree = git(["ls-tree", "-r", "--name-only", "factory/evidence"], world["bare"]).splitlines()
    assert tree == ["repo/tester-abc-1-build-it/after-home.png", "repo/tester-abc-1-build-it/before-home.png"]
    assert deps.state.pushes(SLUG)[0]["branch"] == "tester/abc-1-build-it"

    # one DRAFT PR, registered as a factory PR, with the composed body
    gh: FakeGh = deps.gh  # type: ignore[assignment]
    assert len(gh.created) == 1
    created = gh.created[0]
    assert created["head"] == "tester/abc-1-build-it" and created["base"] == "main" and created["title"] == "Add the thing"
    body = created["body"]
    assert body.startswith("This adds the thing.")
    assert body.index("Reviewers: start with") < body.index("Fixes ABC-1") < body.index(START) < body.index(END)
    assert body.index(END) < body.index(FOOTER_LAST_LINE) and body.rstrip().endswith(FOOTER_LAST_LINE)
    assert "### Risk summary" in body and "`src/feature.py`" in body and "+2 / -0 lines" in body
    assert "touches the config loader" in body
    assert "| AC one | `tests/test_x.py::test_one` | proven |" in body and "| AC two | none | UNPROVEN |" in body
    assert "### Verification" in body and "`uv run pytest -q`" in body and "passed (exit 0)" in body
    assert "### Screenshots" in body and "**Before**" in body and "**After**" in body
    assert f"https://github.com/{SLUG}/blob/factory/evidence/repo/tester-abc-1-build-it/after-home.png?raw=true" in body
    assert f"https://github.com/{SLUG}/blob/factory/evidence/repo/tester-abc-1-build-it/before-home.png?raw=true" in body
    assert "Low repo readiness" not in body and "kind build, run run-1, head " in body
    assert deps.state.is_factory_pr(SLUG, 42)
    assert deps.state.factory_prs()[0]["issue_key"] == "ABC-1" and deps.state.factory_prs()[0]["kind"] == "build"

    # issue transitioned to started through effects; the human notified
    assert tracker.calls == [("move_to_state_type", ("iss-1", "started"))]
    assert effect_events(deps) == ["effect.push_evidence_assets", "effect.push", "effect.open_draft_pr", "effect.transition_issue"]
    note = notifies(deps)[-1]
    assert note["title"] == "draft PR opened" and note["level"] == "info" and note["target"] == "ABC-1"
    assert outcome.pr_url in note["body"] and "factory ready" not in note["body"]

    # the worker call: prompt headings (verbatim, in order), issue context, visual block, policy
    assert resolver.calls == ["ABC-1"]
    call = worker.calls[0]
    assert call["kind"] == "build" and call["schema"] is BuildOutput and call["policy"].read_only is False
    prompt = call["prompt"]
    positions = heading_positions(prompt, "## Action", "## Issue", "## Verification commands", "## Repo readiness")
    assert positions == sorted(positions)
    assert "# ABC-1: Build it" in prompt and "1. AC one" in prompt and "- Branch: `tester/abc-1-build-it`" in prompt
    assert "- branch: `tester/abc-1-build-it`" in prompt and "repo resolved via label" in prompt
    # the worktree is detached at origin/<default>; the engine pushes HEAD, the worker never branches
    assert "- worktree: a detached HEAD at `origin/main`" in prompt and "`HEAD` is what gets pushed" in prompt
    assert "Do not create, switch or rename branches" in prompt and "created from" not in prompt
    assert "This repo is visual" in prompt and "dev command: `npm run dev`" in prompt and "dev port: 5173" in prompt
    assert "routes to capture: `/`, `/play`" in prompt
    assert "Score: 90/100 - OK" in prompt and "No verification commands were discovered" in prompt


def test_build_low_readiness_warns_in_evidence_and_suggests_factory_ready(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]
) -> None:
    issue = build_issue(branch_name="")  # no Linear branch name -> factory/<key>
    tracker = FakeTracker([issue])
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps = make_deps(tracker=tracker, resolver=resolved(), worker=worker)  # no cached readiness -> static score

    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    assert outcome.status == "done" and outcome.pushed
    assert git(["ls-remote", "--heads", "origin", "factory/abc-1"], world["clone"]).endswith("refs/heads/factory/abc-1")
    body = deps.gh.created[0]["body"]  # type: ignore[attr-defined]
    assert "> **Low repo readiness: 0/100 (minimum 60).**" in body
    assert f"> Improve it with `factory ready {SLUG}`." in body and "`agent_doc_present`" in body
    note = notifies(deps)[-1]
    assert note["title"] == "draft PR opened" and note["level"] == "warn"
    assert f"factory ready {SLUG}" in note["body"] and "0/100 is below the minimum of 60" in note["body"]
    assert "LOW (minimum 60)" in worker.calls[0]["prompt"]


def test_build_unresolved_repo_flags_once_and_skips_without_running_a_worker(make_deps: Callable[..., Deps]) -> None:
    issue = build_issue()
    tracker = FakeTracker([issue])
    resolver = FakeResolver(Resolution(repo=None, confidence=0.4, source="picker", candidates=[SLUG, "o/other"]))
    deps = make_deps(tracker=tracker, resolver=resolver, worker=no_worker)
    action = issue_action(Kind.BUILD, issue)

    first = run_handler(Kind.BUILD, action, deps)
    second = run_handler(Kind.BUILD, action, deps)

    assert first.status == second.status == "skipped" and "repo unresolved" in first.note
    assert not first.record_handled and deps.state.attempts("ABC-1", "build", "iss-1") == 0
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "repo unresolved" and flags[0]["target"] == "ABC-1"
    assert "best guess JaneDoe/repo via picker (confidence 0.40)" in flags[0]["detail"]
    assert "repo:<name>" in flags[0]["detail"]
    assert len(notifies(deps)) == 1 and notifies(deps)[0]["title"] == "repo unresolved"
    assert deps.gh.created == [] and tracker.calls == [] and deps.gh.calls == []  # type: ignore[attr-defined]

    # a multi-repo issue is reported as such (the planner splits it)
    resolver.resolution = Resolution(repo=None, confidence=0.95, source="attachment", multi_repo=True, candidates=[SLUG, "o/other"])
    issue2 = build_issue(id="iss-9", key="ABC-9", updated_at="2026-09-30T11:00:00Z")
    tracker.issues[issue2.id] = issue2
    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue2), deps)
    assert outcome.status == "skipped" and "spans several repos" in deps.state.flags()[-1]["detail"]

    # no resolver at all is also "unresolved", never a crash
    deps.resolver = None
    assert run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue2), deps).status == "skipped"


def test_build_preflight_skips(make_deps: Callable[..., Deps]) -> None:
    resolver = resolved()
    # an open PR already exists for the branch
    issue = build_issue(branch_name="")
    gh = FakeGh(open_prs={(SLUG, "factory/abc-1"): PullRequest(repo=SLUG, number=7, url=f"https://github.com/{SLUG}/pull/7")})
    deps = make_deps(tracker=FakeTracker([issue]), gh=gh, resolver=resolver, worker=no_worker)
    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)
    assert outcome.status == "skipped" and outcome.note == f"open PR already exists for factory/abc-1: https://github.com/{SLUG}/pull/7"
    assert gh.calls == [("find_open_pr_for_branch", SLUG, "factory/abc-1")]

    # an unfinished blocker (a completed one does not count)
    blocked = build_issue(blocked_by=[IssueRef(id="b1", key="ABC-0", state_type="started"), IssueRef(id="b2", key="ABC-8", state_type="completed")])
    deps.tracker = FakeTracker([blocked])
    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, blocked), deps)
    assert outcome.status == "skipped" and outcome.note == "blocked by ABC-0"

    # the label was removed / the issue was started meanwhile
    deps.tracker = FakeTracker([build_issue(labels=["bug"])])
    assert "label factory:build" in run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps).note
    deps.tracker = FakeTracker([build_issue(state_type="started")])
    assert "not unstarted" in run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps).note

    # no tracker configured
    deps.tracker = None
    assert run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps).status == "skipped"
    # only the open-PR case got as far as resolving; none of them flagged or spent anything
    assert resolver.calls == ["ABC-1"] and deps.state.flags() == []


ISSUE_KINDS_ON = [Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV, Kind.REFINE, Kind.PLAN, Kind.BUILD]


def enable(deps: Deps, kinds: list[Kind]) -> Deps:
    deps.profile = deps.profile.model_copy(update={"enabled_kinds": list(kinds)})
    return deps


@pytest.mark.parametrize("earlier", [["factory:refine"], ["factory:plan"], ["factory:refine", "factory:plan"]])
def test_build_waits_while_refine_or_plan_is_still_pending(make_deps: Callable[..., Deps], earlier: list[str]) -> None:
    """refine rewrites the text a build reads and plan splits the work: while either label is
    still on the issue (and its kind is enabled) the build is skipped -- before resolving or
    starting a worker, and without spending an attempt (``skipped`` is no attempt outcome). The
    human hears once which label holds it."""
    issue = build_issue(labels=["factory:build", *earlier])
    resolver = resolved()
    deps = enable(make_deps(tracker=FakeTracker([issue]), resolver=resolver, worker=no_worker), ISSUE_KINDS_ON)

    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    assert outcome.status == "skipped" and outcome.note == f"refine/plan run first: issue still carries {', '.join(earlier)}"
    assert not outcome.record_handled and resolver.calls == []
    assert deps.gh.calls == [] and deps.tracker.calls == []  # type: ignore[union-attr]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "build waits" and flags[0]["dedupe_key"] == "build-waits:ABC-1"
    for label in earlier:
        assert f"- `{label}`:" in flags[0]["detail"]
    assert notifies(deps)[-1]["title"] == "build waits" and notifies(deps)[-1]["level"] == "info"
    # flag-once: the next tick's skip says nothing new
    assert run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps).status == "skipped"
    assert len(deps.state.flags()) == 1 and len(notifies(deps)) == 1


def test_build_ignores_the_label_of_a_disabled_refine_or_plan(make_deps: Callable[..., Deps], git: Callable[..., str]) -> None:
    """Regression: with ``plan`` (or ``refine``) not in ``enabled_kinds`` its label never goes away
    (nothing runs to remove it or to record a verdict), so the build used to wait forever."""
    issue = build_issue(labels=["factory:build", "factory:plan", "factory:refine"])
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps = enable(make_deps(tracker=FakeTracker([issue]), resolver=resolved(), worker=worker), [Kind.BUILD])

    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    assert outcome.status == "done" and outcome.pushed and len(worker.calls) == 1
    assert "build waits" not in [f["reason"] for f in deps.state.flags()]
    assert build_mod.pending_earlier_kinds(issue, deps) == []
    # refine enabled again: its label holds the build (plan stays disabled and does not)
    enable(deps, [Kind.BUILD, Kind.REFINE])
    assert build_mod.pending_earlier_kinds(issue, deps) == ["factory:refine"]


def test_build_proceeds_once_plan_said_nothing_for_this_version(make_deps: Callable[..., Deps], git: Callable[..., str]) -> None:
    """Regression: plan never removes its label, so after "nothing to plan" (recorded at the issue's
    ``updated_at``) the build waited forever. The recorded verdict releases it -- for that version
    only: an edit of the issue makes the plan label pending again until plan has run on it."""
    issue = build_issue(labels=["factory:build", "factory:plan"])
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps = enable(make_deps(tracker=FakeTracker([issue]), resolver=resolved(), worker=worker), ISSUE_KINDS_ON)
    assert build_mod.pending_earlier_kinds(issue, deps, target="ABC-1") == ["factory:plan"]
    deps.state.mark_handled("ABC-1", "plan", "2026-09-29T00:00:00Z", "nothing")  # an older version: still pending
    assert build_mod.pending_earlier_kinds(issue, deps, target="ABC-1") == ["factory:plan"]
    deps.state.mark_handled("ABC-1", "plan", issue.updated_at, "nothing", "one PR's worth")

    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    assert outcome.status == "done" and outcome.pushed and len(worker.calls) == 1
    assert "build waits" not in [f["reason"] for f in deps.state.flags()]
    # an issue that already has children was planned: its plan label holds nothing either
    planned = build_issue(labels=["factory:build", "factory:plan"], updated_at="2026-10-01T00:00:00Z",
                          children=[IssueRef(id="c1", key="ABC-50", state_type="backlog")])  # fmt: skip
    assert build_mod.pending_earlier_kinds(planned, deps, target="ABC-1") == []


def test_build_evidence_names_changes_next_to_foreign_codeowners_areas(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]
) -> None:
    """The draft PR's evidence lists a changed file that sits in another team's CODEOWNERS area as
    sensitive-adjacent, while the push itself is judged per path (the file is mine by last match)."""
    (world["clone"] / ".github").mkdir(exist_ok=True)
    (world["clone"] / ".github" / "CODEOWNERS").write_text("/src/ @acme/core-team\n/src/feature.py @JaneDoe\n", encoding="utf-8")
    issue = build_issue()
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps = make_deps(tracker=FakeTracker([issue]), resolver=resolved(), worker=worker)

    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    assert outcome.status == "done" and outcome.pushed, outcome.note
    body = deps.gh.created[0]["body"]  # type: ignore[attr-defined]
    assert "Sensitive-adjacent paths (1):" in body and "  - `src/feature.py`" in body
    stored = kb.load_evidence(deps.state, SLUG, 42)
    assert stored is not None and stored.sensitive_adjacent == ["src/feature.py"]


def test_build_on_a_branch_that_exists_on_origin_starts_from_it_detached(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]
) -> None:
    """An earlier (human or factory) push of the issue branch: the worktree continues it and the
    prompt says so, still as a detached HEAD the engine pushes."""
    git(["checkout", "-q", "-b", "tester/abc-1-build-it"], world["clone"])
    start = commit_file(git, world["clone"], "src/started.py", "s = 1\n", "started by hand")
    git(["push", "-q", "origin", "HEAD:refs/heads/tester/abc-1-build-it"], world["clone"])
    git(["checkout", "-q", "main"], world["clone"])
    issue = build_issue()
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps = make_deps(tracker=FakeTracker([issue]), resolver=resolved(), worker=worker)

    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    assert outcome.status == "done" and outcome.pushed
    prompt = worker.calls[0]["prompt"]
    assert "- worktree: a detached HEAD at `origin/tester/abc-1-build-it`" in prompt
    assert git(["rev-parse", "refs/heads/tester/abc-1-build-it~1"], world["bare"]) == start  # continued, not replaced


def test_build_policy_refusal_keeps_the_worktree_and_opens_nothing(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths, git: Callable[..., str]
) -> None:
    issue = build_issue()
    tracker = FakeTracker([issue])
    worker = FakeWorker(build_output(push_ok=False, push_reasons=["touches billing logic"]), on_run=committing(git))
    deps = make_deps(tracker=tracker, resolver=resolved(), worker=worker)

    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    assert outcome.status == "kept" and not outcome.pushed and outcome.pr_url is None
    assert "touches billing logic" in outcome.note
    assert deps.gh.created == [] and tracker.calls == []  # type: ignore[attr-defined]
    assert git(["ls-remote", "--heads", "origin"], world["clone"]).count("refs/heads/") == 1  # only main
    kept = deps.state.kept_worktree("ABC-1")
    assert kept is not None and Path(kept["path"]).is_dir() and kept["kind"] == "build"
    assert [f["reason"] for f in deps.state.flags()] == ["push refused"]
    assert effect_events(deps) == []

    # a sensitive path is refused by the engine even when the worker says push_ok
    issue2 = build_issue(id="iss-2", key="ABC-2", branch_name="tester/abc-2")
    tracker.issues[issue2.id] = issue2
    worker = FakeWorker(build_output(), on_run=committing(git, "src/auth/login.py"))
    deps.run_worker = worker
    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue2), deps)
    assert outcome.status == "kept" and "sensitive globs: src/auth/login.py" in outcome.note
    assert deps.gh.created == []  # type: ignore[attr-defined]


def test_build_with_recording_effects_writes_nothing_outward(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths, git: Callable[..., str]
) -> None:
    """Deps built with dry_run=True (full mode): the handler runs, every outward write is recorded only."""
    issue = build_issue()
    tracker = FakeTracker([issue], raise_on_write=True)
    gh = FakeGh(raise_on_write=True)

    def work(call: dict[str, Any]) -> None:
        cwd = Path(call["cwd"])
        commit_file(git, cwd, "src/feature.py", "x = 1\n")
        (cwd / ".factory" / "evidence").mkdir(parents=True)
        (cwd / ".factory" / "evidence" / "after-home.png").write_bytes(b"png")

    evidence = build_output()["evidence"]
    evidence["screenshots"] = [{"path": ".factory/evidence/after-home.png", "caption": "home", "phase": "after"}]
    deps = make_deps(tracker=tracker, gh=gh, resolver=resolved(), worker=FakeWorker(build_output(evidence=evidence), on_run=work), recording=True)

    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    assert outcome.status == "done" and outcome.pushed is False and "dry-run" in outcome.note
    assert outcome.pr_url == f"dry-run://{SLUG}/pull/tester/abc-1-build-it"
    assert effect_events(deps) == ["would_push_evidence_assets", "would_push", "would_open_draft_pr", "would_transition_issue"]
    assert git(["ls-remote", "--heads", "origin"], world["clone"]).count("refs/heads/") == 1  # nothing reached origin
    assert gh.created == [] and tracker.calls == [] and deps.state.pushes() == []
    assert list(paths.worktrees.iterdir()) == []
    assert notifies(deps)[-1]["title"] == "draft PR opened"


def test_build_failures_spend_an_attempt_and_flag(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths, git: Callable[..., str]
) -> None:
    issue = build_issue()
    deps = make_deps(tracker=FakeTracker([issue]), resolver=resolved(), worker=FakeWorker(build_output()))  # done, no commits
    action = issue_action(Kind.BUILD, issue)

    outcome = run_handler(Kind.BUILD, action, deps)
    assert outcome.status == "failed" and "no commits" in outcome.note and not outcome.record_handled
    assert deps.state.flags()[-1]["reason"] == "no commits" and list(paths.worktrees.iterdir()) == []
    assert deps.gh.created == []  # type: ignore[attr-defined]

    deps.run_worker = FakeWorker(build_output(status="nothing", summary="already on the branch"))
    outcome = run_handler(Kind.BUILD, action, deps)
    assert outcome.status == "nothing" and outcome.record_handled and outcome.note == "already on the branch"

    deps.run_worker = FakeWorker(build_output(status="blocked", summary="AC 2 impossible", push_ok=False, push_reasons=["AC 2 impossible"]))
    outcome = run_handler(Kind.BUILD, action, deps)
    assert outcome.status == "failed" and "AC 2 impossible" in outcome.note and deps.state.flags()[-1]["reason"] == "worker blocked"

    deps.run_worker = FakeWorker(None, error="worker timed out after 420s", timed_out=True)
    outcome = run_handler(Kind.BUILD, action, deps)
    assert outcome.status == "failed" and "timed out" in outcome.note and deps.state.flags()[-1]["reason"] == "worker timed out"

    deps.run_worker = FakeWorker({"status": "done"})  # not a BuildOutput
    outcome = run_handler(Kind.BUILD, action, deps)
    assert outcome.status == "failed" and "invalid BuildOutput" in outcome.note
    assert list(paths.worktrees.iterdir()) == [] and deps.state.kept_worktree("ABC-1") is None


def test_build_resumes_a_pushed_branch_whose_draft_pr_failed_to_open(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths, git: Callable[..., str]
) -> None:
    """Tick 1 pushes the branch and ``gh pr create`` fails; tick 2 must open the PR for the pushed
    branch from the saved record, not run a second 30-minute worker session on it."""

    class FlakyGh(FakeGh):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def create_draft_pr(self, repo: str, head: str, base: str, title: str, body: str) -> PullRequest:
            if self.fail:
                raise RuntimeError("HTTP 502 from api.github.com")
            return super().create_draft_pr(repo, head, base, title, body)

    issue = build_issue()
    tracker = FakeTracker([issue])
    gh = FlakyGh()
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps = make_deps(tracker=tracker, gh=gh, resolver=resolved(), worker=worker)
    action = issue_action(Kind.BUILD, issue)

    first = run_handler(Kind.BUILD, action, deps)

    assert first.status == "failed" and first.pushed and "opening the draft PR failed" in first.note and "next run opens" in first.note
    assert deps.state.flags()[-1]["reason"] == "draft PR failed" and tracker.calls == [] and len(worker.calls) == 1
    pushed_sha = git(["rev-parse", "refs/heads/tester/abc-1-build-it"], world["bare"])
    pending = deps.state.cache_get(build_mod.PENDING_PR_CACHE, f"{SLUG}:tester/abc-1-build-it")
    assert pending is not None and pending["head_sha"] == pushed_sha and "Fixes ABC-1" in pending["body"]
    assert list(paths.worktrees.iterdir()) == []

    gh.fail = False
    deps.run_worker = no_worker
    second = run_handler(Kind.BUILD, action, deps)

    assert second.status == "done" and second.record_handled and second.pushed is False
    assert second.pr_url == f"https://github.com/{SLUG}/pull/42" and "pushed by an earlier run" in second.note
    created = gh.created[0]
    assert created["head"] == "tester/abc-1-build-it" and created["title"] == "Add the thing" and "Fixes ABC-1" in created["body"]
    assert created["body"] == pending["body"]
    assert deps.state.is_factory_pr(SLUG, 42) and deps.state.pushed_shas(SLUG, 42) == {pushed_sha}
    assert tracker.calls == [("move_to_state_type", ("iss-1", "started"))]
    assert notifies(deps)[-1]["title"] == "draft PR opened" and notifies(deps)[-1]["level"] == "warn"  # bare repo: low readiness
    assert deps.state.cache_get(build_mod.PENDING_PR_CACHE, f"{SLUG}:tester/abc-1-build-it") is None
    assert kb.load_evidence(deps.state, SLUG, 42) is not None  # upkeep refreshes merge into it
    assert [r["event"] for r in deps.audit.read() if r["event"].startswith("build.pending")] == ["build.pending_pr_resumed"]

    # the branch disappears from origin before the retry: the record is dropped and the worker runs
    deps.state.cache_set(build_mod.PENDING_PR_CACHE, f"{SLUG}:tester/abc-1-build-it", pending)
    git(["push", "-q", "origin", "--delete", "tester/abc-1-build-it"], world["clone"])
    gh.created.clear()
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps.run_worker = worker
    third = run_handler(Kind.BUILD, action, deps)
    assert third.status == "done" and third.pushed and len(worker.calls) == 1 and len(gh.created) == 1
    assert deps.audit.read()[-1]["event"] != "build.pending_pr_resumed"
    assert any(r["event"] == "build.pending_pr_dropped" for r in deps.audit.read())


def test_build_nothing_is_recorded_but_flagged_once(make_deps: Callable[..., Deps]) -> None:
    issue = build_issue()
    worker = FakeWorker(build_output(status="nothing", summary="the feature already exists on main"))
    deps = make_deps(tracker=FakeTracker([issue]), resolver=resolved(), worker=worker)
    action = issue_action(Kind.BUILD, issue)

    outcome = run_handler(Kind.BUILD, action, deps)

    assert outcome.status == "nothing" and outcome.record_handled
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "nothing to build" and flags[0]["dedupe_key"] == "build-nothing:ABC-1:iss-1"
    assert "already exists on main" in flags[0]["detail"] and "factory run build ABC-1" in flags[0]["detail"]
    assert notifies(deps)[-1]["title"] == "nothing to build" and notifies(deps)[-1]["level"] == "info"
    assert deps.gh.created == []  # type: ignore[attr-defined]
    run_handler(Kind.BUILD, action, deps)
    assert len(deps.state.flags()) == 1  # flag-once per target/version


def test_build_static_readiness_of_a_documented_repo_carries_no_low_warning(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]
) -> None:
    """Without a cached full run, build scores the static checks only; a documented repo with
    discovered commands must not be reported "50/100, low" for checks that were simply not run."""
    clone = world["clone"]
    (clone / "CLAUDE.md").write_text("# repo\n\nA test repo.\n\n## Verification\n\n- `uv run pytest -q`\n", encoding="utf-8")
    git(["add", "-A"], clone)
    git(["commit", "-q", "-m", "docs"], clone)
    git(["push", "-q", "origin", "main"], clone)
    issue = build_issue()
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps = make_deps(tracker=FakeTracker([issue]), resolver=resolved(), worker=worker)

    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    assert outcome.status == "done" and outcome.pushed
    body = deps.gh.created[0]["body"]  # type: ignore[attr-defined]
    assert "Low repo readiness" not in body and "not run" not in body
    note = notifies(deps)[-1]
    assert note["title"] == "draft PR opened" and note["level"] == "info" and "factory ready" not in note["body"]
    prompt = worker.calls[0]["prompt"]
    assert "Score: 100/100 - OK (static checks only; runtime checks were not run)" in prompt
    assert "| commands_discovered | ok |" in prompt and "commands_pass" not in prompt
    # a cached full run at this HEAD is used as-is, runtime results included
    seed_full_readiness(deps, world["repo"], 30)
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps.run_worker = worker
    deps.gh.open_prs.clear()  # type: ignore[attr-defined]
    issue2 = build_issue(id="iss-2", key="ABC-2", branch_name="tester/abc-2")
    deps.tracker = FakeTracker([issue2])
    run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue2), deps)
    assert "Score: 30/100 - LOW (minimum 60)" in worker.calls[0]["prompt"] and "static checks only" not in worker.calls[0]["prompt"]


def test_build_evidence_survives_a_later_upkeep_refresh(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    issue = build_issue()
    worker = FakeWorker(build_output(), on_run=committing(git))
    deps = make_deps(tracker=FakeTracker([issue]), resolver=resolved(), worker=worker)
    assert run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps).status == "done"
    opened = deps.gh.created[0]["body"]  # type: ignore[attr-defined]
    assert "| AC one |" in opened and "Low repo readiness: 0/100" in opened

    # a ci fix lands on the factory PR: the AC table, risk notes and readiness warning must survive
    wt = deps.wt_manager.for_new_branch(world["repo"], "tester/abc-1-build-it")
    commit_file(git, wt.path, "tests/test_feature.py", "ok\n")
    ci_out = CiOutput(status="done", summary="fixed the test", push_ok=True, head_verified=True,
                      verification=[VerificationRun(command="uv run pytest -q tests", exit_code=0, passed=True)])  # fmt: skip
    ci = Action(kind=Kind.CI, version="h", repo=SLUG, pr=42)
    result = asyncio.run(finish_code_worker(ci, deps, repo=world["repo"], wt=wt, output=ci_out, branch="tester/abc-1-build-it",
                                            pr_number=42, factory_pr=True, kind=Kind.CI))  # fmt: skip

    assert result.status == "done" and result.pushed
    body = deps.gh.edits[-1][2]  # type: ignore[attr-defined]
    assert "| AC one | `tests/test_x.py::test_one` | proven |" in body and "| AC two | none | UNPROVEN |" in body
    assert "Low repo readiness: 0/100" in body and "touches the config loader" in body
    assert "`src/feature.py`" in body and "`tests/test_feature.py`" in body and "Files touched: 2" in body
    assert "`uv run pytest -q`" in body and "`uv run pytest -q tests`" in body
    assert "kind ci, run run-1" in body and body.rstrip().endswith(FOOTER_LAST_LINE) and body.startswith("This adds the thing.")


def test_build_auth_failures_propagate(make_deps: Callable[..., Deps]) -> None:
    issue = build_issue()
    deps = make_deps(tracker=FakeTracker([issue], read_error=TrackerAuthError("bad key")), resolver=resolved(), worker=no_worker)
    with pytest.raises(TrackerAuthError):
        run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    class AuthGh(FakeGh):
        def find_open_pr_for_branch(self, repo: str, branch: str) -> PullRequest | None:
            raise GhAuthError("gh auth expired")

    deps = make_deps(tracker=FakeTracker([issue]), gh=AuthGh(), resolver=resolved(), worker=no_worker)
    with pytest.raises(GhAuthError):
        run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)

    deps = make_deps(tracker=FakeTracker([issue], read_error=TrackerError("boom")), resolver=resolved(), worker=no_worker)
    outcome = run_handler(Kind.BUILD, issue_action(Kind.BUILD, issue), deps)
    assert outcome.status == "failed" and "boom" in outcome.note and deps.state.flags()[-1]["reason"] == "tracker error"


class LimitedResolver:
    """The resolver whose model picker hit the provider's rate limit."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def resolve(self, issue: Issue) -> Resolution:
        self.calls.append(issue.key)
        raise ProviderUnavailable("rate limited (resolver)")


@pytest.mark.parametrize("kind", [Kind.BUILD, Kind.REFINE, Kind.PLAN])
def test_issue_kinds_propagate_provider_unavailable_without_flag_or_leftovers(
    make_deps: Callable[..., Deps], paths: ProfilePaths, kind: Kind
) -> None:
    """A rate-limited worker (or resolver picker) is no verdict on the issue: the handler raises
    ``ProviderUnavailable`` for the dispatcher (flag once a day, no attempt), flags nothing itself,
    writes nothing outward and leaves no checkout behind."""
    issue = {Kind.BUILD: build_issue(), Kind.REFINE: refine_issue(), Kind.PLAN: plan_issue()}[kind]
    worker = FakeWorker(None, error="usage limit reached")

    async def limited(**kwargs: Any) -> WorkerRun:
        worker.calls.append(kwargs)
        return WorkerRun(output=None, error="usage limit reached", rate_limited=True)

    tracker = FakeTracker([issue])
    deps = make_deps(tracker=tracker, resolver=resolved(), worker=limited)
    with pytest.raises(ProviderUnavailable, match="usage limit reached"):
        run_handler(kind, issue_action(kind, issue), deps)
    assert len(worker.calls) == 1 and deps.state.flags() == [] and tracker.calls == []
    assert list(paths.worktrees.iterdir()) == [] and deps.gh.created == []  # type: ignore[attr-defined]

    if kind == Kind.PLAN:
        return  # plan never resolves a repo
    resolver = LimitedResolver()
    deps = make_deps(tracker=FakeTracker([issue]), resolver=resolver, worker=no_worker)
    with pytest.raises(ProviderUnavailable, match="resolver"):
        run_handler(kind, issue_action(kind, issue), deps)
    assert resolver.calls == [issue.key] and deps.state.flags() == []


# ------------------------------------------------------------------------------------- ready
def test_ready_action_version_is_origin_default_head_after_a_fetch(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str], tmp_path: Path
) -> None:
    deps = make_deps()
    stale = git(["rev-parse", "origin/main"], world["clone"])
    other = tmp_path / "other clone"
    git(["clone", "-q", str(world["bare"]), str(other)], tmp_path)
    new_sha = commit_file(git, other, "NEW.md", "n\n", "someone else's commit")
    git(["push", "-q", "origin", "main"], other)
    assert git(["rev-parse", "origin/main"], world["clone"]) == stale  # the clone has not fetched yet

    action = ready_action(deps, "repo")  # name, slug or dirname all work

    assert action.kind == Kind.READY and action.repo == SLUG and action.target == SLUG and action.pr is None
    assert action.version == new_sha != stale
    assert action.context == {"default_branch": "main", "head_sha": new_sha}
    assert action.title == f"Improve factory readiness: {SLUG}"
    assert ready_action(deps, SLUG).version == new_sha
    with pytest.raises(ValueError, match="unknown repo 'nope'"):
        ready_action(deps, "nope")


def test_ready_opens_a_draft_pr_on_the_dated_branch_without_an_issue_key(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths, git: Callable[..., str]
) -> None:
    output = {
        "status": "done", "summary": "documented the verification commands", "push_ok": True,
        "verification": [{"command": "npm test", "exit_code": 0, "passed": True}],
        "pr_title": "Document verification commands", "changes": ["added a verification section to CLAUDE.md"],
    }  # fmt: skip
    worker = FakeWorker(output, on_run=committing(git, "CLAUDE.md", "# repo\n\n## Verification\n\n- `npm test`\n"))
    deps = make_deps(worker=worker)
    action = ready_action(deps, SLUG)

    outcome = run_handler(Kind.READY, action, deps)

    assert outcome.status == "done" and outcome.pushed and outcome.pr_url == f"https://github.com/{SLUG}/pull/42"
    assert git(["ls-remote", "--heads", "origin", "factory/ready-20261001"], world["clone"]).endswith("refs/heads/factory/ready-20261001")
    created = deps.gh.created[0]  # type: ignore[attr-defined]
    assert created["head"] == "factory/ready-20261001" and created["base"] == "main"
    assert created["title"] == "Document verification commands"
    body = created["body"]
    assert "Fixes" not in body
    assert body.startswith("documented the verification commands") and "## Changes" in body
    assert "- added a verification section to CLAUDE.md" in body
    assert START in body and "`CLAUDE.md`" in body and "`npm test`" in body and body.rstrip().endswith(FOOTER_LAST_LINE)
    assert "kind ready, run run-1" in body
    registered = deps.state.factory_prs()[0]
    assert registered["issue_key"] is None and registered["kind"] == "ready" and registered["branch"] == "factory/ready-20261001"
    note = notifies(deps)[-1]
    assert note["title"] == "readiness PR opened" and note["target"] == SLUG and outcome.pr_url in note["body"]
    assert list(paths.worktrees.iterdir()) == []

    call = worker.calls[0]
    assert call["kind"] == "ready" and call["schema"] is ReadyOutput and call["policy"].read_only is False
    prompt = call["prompt"]
    positions = heading_positions(prompt, "## Action", "## Repo readiness", "## Verification commands")
    assert positions == sorted(positions)
    assert "`factory ready`" in prompt and "branch: `factory/ready-20261001`" in prompt
    assert "a detached HEAD at `origin/main`" in prompt and "Do not create, switch or rename branches" in prompt
    assert "Score: 0/100 - LOW (minimum 60)" in prompt and "| commands_discovered | FAIL |" in prompt

    # a second run the same day finds the open PR and skips
    deps.gh.open_prs[(SLUG, "factory/ready-20261001")] = PullRequest(repo=SLUG, number=42, url=outcome.pr_url or "")  # type: ignore[attr-defined]
    deps.run_worker = no_worker
    again = run_handler(Kind.READY, ready_action(deps, SLUG), deps)
    assert again.status == "skipped" and "open PR already exists for factory/ready-20261001" in again.note
    assert run_handler(Kind.READY, Action(kind=Kind.READY, version="x", repo="o/unknown"), deps).status == "skipped"


# ------------------------------------------------------------------------------------ refine
def refine_issue(**over: Any) -> Issue:
    base: dict[str, Any] = dict(
        id="iss-2", key="ABC-2", title="Fix login", description="Login breaks sometimes.\nSee https://example.com/x",
        url="https://linear.app/x/issue/ABC-2", labels=["factory:refine"], state_type="backlog", team_key="ABC",
        updated_at="2026-09-30T09:00:00Z",
    )  # fmt: skip
    base.update(over)
    return Issue(**base)


def refine_output(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": "done",
        "title": "Fix login validation message",
        "description": (
            "## Context\n\nLogin breaks sometimes.\n\n## Acceptance criteria\n\n1. Empty name shows the message\n\n"
            "## Links\n\n- https://example.com/x"
        ),
        "acceptance_criteria": ["Empty name shows the message"],
        "open_questions": ["Which users are affected?"],
        "suggested_split": ["Split out the API change"],
        "change_summary": "Tightened the AC and kept the link.",
    }
    base.update(over)
    return base


def record_statuses(deps: Deps) -> list[str]:
    statuses: list[str] = []
    original = deps.state.set_refinement_status

    def wrapped(key: str, status: str) -> None:
        statuses.append(status)
        original(key, status)

    deps.state.set_refinement_status = wrapped  # type: ignore[method-assign]
    return statuses


def test_refine_persists_then_notifies_then_edits_then_removes_the_label(
    make_deps: Callable[..., Deps], paths: ProfilePaths
) -> None:
    issue = refine_issue()
    tracker = FakeTracker([issue])
    box: dict[str, Deps] = {}
    toaster = FakeToaster(
        probe=lambda: (box["deps"].state.get_refinement("ABC-2"), paths.refinement_file("ABC-2").is_file(), list(tracker.calls))
    )
    worker = FakeWorker(refine_output())
    deps = box["deps"] = make_deps(tracker=tracker, worker=worker, toaster=toaster)
    statuses = record_statuses(deps)

    outcome = run_handler(Kind.REFINE, issue_action(Kind.REFINE, issue), deps)

    assert outcome.status == "done" and outcome.record_handled and outcome.cost_usd == 0.5
    assert outcome.note == "refinement applied (notified -> edited -> applied)"

    # state + refinements/<KEY>.md existed before the notification, and nothing had been edited yet
    assert len(toaster.seen) == 1
    record, file_existed, calls_at_notify = toaster.seen[0]
    assert record is not None and record["status"] == "pending" and file_existed and calls_at_notify == []
    assert record["original_title"] == "Fix login" and record["original_description"] == issue.description
    assert record["new_title"] == "Fix login validation message"

    # the notification: diff summary, open questions, split, the revert command
    title, body = toaster.sent[0]
    assert title == "refined ABC-2: Fix login validation message"
    assert 'Title: "Fix login" -> "Fix login validation message"' in body
    assert "Description: +10 / -1 lines" in body and str(paths.refinement_file("ABC-2")) in body
    assert "Tightened the AC and kept the link." in body
    assert "Open questions:\n- Which users are affected?" in body and "Suggested split:\n- Split out the API change" in body
    assert body.rstrip().endswith("Revert with: factory revert-refine ABC-2")

    # effects in order, nothing else (no comments, no labels added)
    assert tracker.calls == [
        ("update_issue_text", ("iss-2", "Fix login validation message", refine_output()["description"])),
        ("remove_label", ("iss-2", "factory:refine")),
    ]
    assert effect_events(deps) == ["effect.update_issue_text", "effect.remove_label"]
    assert statuses == ["notified", "edited", "applied"]
    assert deps.state.get_refinement("ABC-2")["status"] == "applied"  # type: ignore[index]

    text = paths.refinement_file("ABC-2").read_text(encoding="utf-8")
    for needle in ("# Refinement of ABC-2", "## Original title", "Fix login\n", "## Original description", "Login breaks sometimes.",
                   "## New description", "```diff", "-See https://example.com/x", "+## Context", "## Open questions",
                   "- Which users are affected?", "## Suggested split", "- Split out the API change", "factory revert-refine ABC-2"):
        assert needle in text, needle

    # the worker: read-only, scratch cwd (no resolver), the issue context under ## Issue
    call = worker.calls[0]
    assert call["kind"] == "refine" and call["schema"] is RefineOutput and call["policy"].read_only is True
    assert call["cwd"] == paths.root / "scratch"
    prompt = call["prompt"]
    assert heading_positions(prompt, "## Action", "## Issue") == sorted(heading_positions(prompt, "## Action", "## Issue"))
    assert "# ABC-2: Fix login" in prompt and "https://example.com/x" in prompt and "no repository resolved" in prompt


def test_refine_reads_the_resolved_repo_in_a_detached_worktree_and_removes_it(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths
) -> None:
    issue = refine_issue()
    seen: dict[str, Any] = {}

    def look(call: dict[str, Any]) -> None:
        cwd = Path(call["cwd"])
        seen["cwd"] = cwd
        seen["readme"] = (cwd / "README.md").is_file()
        seen["detached"] = cwd.parent == paths.worktrees

    worker = FakeWorker(refine_output(), on_run=look)
    deps = make_deps(tracker=FakeTracker([issue]), resolver=resolved(), worker=worker)

    outcome = run_handler(Kind.REFINE, issue_action(Kind.REFINE, issue), deps)

    assert outcome.status == "done"
    assert seen["readme"] and seen["detached"] and not seen["cwd"].exists()
    assert list(paths.worktrees.iterdir()) == []
    call = worker.calls[0]
    assert call["policy"].read_only is True and not call["policy"].check("Edit", {"file_path": str(seen["cwd"] / "README.md")}).allow
    assert f"repository for reading only: {SLUG}" in call["prompt"]


def test_refine_without_a_user_facing_notification_never_edits_and_resumes_later(
    make_deps: Callable[..., Deps], paths: ProfilePaths
) -> None:
    issue = refine_issue()
    tracker = FakeTracker([issue])
    deps = make_deps(tracker=tracker, worker=FakeWorker(refine_output()), toaster=FakeToaster(fail=True))
    action = issue_action(Kind.REFINE, issue)

    outcome = run_handler(Kind.REFINE, action, deps)

    assert outcome.status == "failed" and "not notified" in outcome.note and not outcome.record_handled
    assert tracker.calls == [] and effect_events(deps) == []
    record = deps.state.get_refinement("ABC-2")
    assert record is not None and record["status"] == "pending" and paths.refinement_file("ABC-2").is_file()
    assert [f["reason"] for f in deps.state.flags()] == ["refine not notified"]
    assert "no desktop session" in deps.state.flags()[0]["detail"]

    # next tick: the saved record resumes at the notification step; the worker does not run again
    deps.run_worker = no_worker
    deps.notifier = Notifier(deps.profile, deps.audit, toaster=FakeToaster())
    statuses = record_statuses(deps)
    outcome = run_handler(Kind.REFINE, action, deps)
    assert outcome.status == "done" and statuses == ["notified", "edited", "applied"]
    assert [c[0] for c in tracker.calls] == ["update_issue_text", "remove_label"]


@pytest.mark.parametrize(
    ("status", "expected_calls", "expected_statuses"),
    [
        ("notified", ["update_issue_text", "remove_label"], ["edited", "applied"]),
        ("edited", ["remove_label"], ["applied"]),
    ],
)
def test_refine_resumes_a_partial_state_without_a_worker_or_a_second_notification(
    make_deps: Callable[..., Deps], status: str, expected_calls: list[str], expected_statuses: list[str]
) -> None:
    issue = refine_issue()
    tracker = FakeTracker([issue])
    deps = make_deps(tracker=tracker, worker=no_worker)
    deps.state.save_refinement("iss-2", "ABC-2", "Fix login", issue.description, "New title", "new description")
    deps.state.set_refinement_status("ABC-2", status)
    statuses = record_statuses(deps)

    outcome = run_handler(Kind.REFINE, issue_action(Kind.REFINE, issue), deps)

    assert outcome.status == "done" and outcome.record_handled
    assert [c[0] for c in tracker.calls] == expected_calls and statuses == expected_statuses
    if "update_issue_text" in expected_calls:
        assert tracker.calls[0] == ("update_issue_text", ("iss-2", "New title", "new description"))
    assert notifies(deps) == []  # the human was already told
    assert deps.state.get_refinement("ABC-2")["status"] == "applied"  # type: ignore[index]

    # an applied record does not block a fresh refinement later (the label was re-added): the worker runs
    deps.run_worker = FakeWorker(refine_output(status="nothing", change_summary="already crisp"))
    outcome = run_handler(Kind.REFINE, issue_action(Kind.REFINE, issue), deps)
    assert outcome.status == "nothing" and outcome.note == "already crisp"


def test_refine_effect_failure_keeps_the_step_for_the_next_tick(make_deps: Callable[..., Deps]) -> None:
    class Flaky(FakeTracker):
        def remove_label(self, issue_id: str, label: str) -> None:
            raise TrackerError("Linear 500")

    issue = refine_issue()
    tracker = Flaky([issue])
    deps = make_deps(tracker=tracker, worker=FakeWorker(refine_output()))
    outcome = run_handler(Kind.REFINE, issue_action(Kind.REFINE, issue), deps)
    assert outcome.status == "failed" and "remove_label failed" in outcome.note
    assert deps.state.get_refinement("ABC-2")["status"] == "edited"  # type: ignore[index]
    assert [c[0] for c in tracker.calls] == ["update_issue_text"]
    assert deps.state.flags()[-1]["reason"] == "refine effect failed"


def test_revert_refine_restores_the_original_text(make_deps: Callable[..., Deps]) -> None:
    issue = refine_issue()
    tracker = FakeTracker([issue])
    deps = make_deps(tracker=tracker, worker=FakeWorker(refine_output()))
    assert run_handler(Kind.REFINE, issue_action(Kind.REFINE, issue), deps).status == "done"
    assert deps.state.get_refinement("ABC-2")["status"] == "applied"  # type: ignore[index]

    outcome = asyncio.run(revert_refine(deps, "ABC-2"))

    assert outcome.status == "done" and "restored" in outcome.note
    assert tracker.calls[-1] == ("update_issue_text", ("iss-2", "Fix login", issue.description))
    assert deps.state.get_refinement("ABC-2")["status"] == "reverted"  # type: ignore[index]
    assert all(c[0] in ("update_issue_text", "remove_label") for c in tracker.calls)  # no labels added, no comments
    note = notifies(deps)[-1]
    assert note["title"] == "reverted refinement ABC-2" and "Fix login" in note["body"]

    assert asyncio.run(revert_refine(deps, "ABC-2")).status == "nothing"
    assert asyncio.run(revert_refine(deps, "ABC-404")).status == "failed"

    # a record that never reached the edit step is just marked reverted (nothing to restore)
    calls_before = len(tracker.calls)
    deps.state.save_refinement("iss-5", "ABC-5", "t", "d", "T", "D")
    outcome = asyncio.run(revert_refine(deps, "ABC-5"))
    assert outcome.status == "done" and "never edited" in outcome.note
    assert len(tracker.calls) == calls_before and deps.state.get_refinement("ABC-5")["status"] == "reverted"  # type: ignore[index]


def test_refine_nothing_blocked_and_empty_outputs(make_deps: Callable[..., Deps], paths: ProfilePaths) -> None:
    issue = refine_issue()
    tracker = FakeTracker([issue])
    deps = make_deps(tracker=tracker, worker=FakeWorker(refine_output(status="nothing", change_summary="already a crisp spec")))
    action = issue_action(Kind.REFINE, issue)

    outcome = run_handler(Kind.REFINE, action, deps)
    assert outcome.status == "nothing" and outcome.record_handled and outcome.note == "already a crisp spec"
    assert deps.state.get_refinement("ABC-2") is None and not paths.refinement_file("ABC-2").exists()
    # the label comes off (or every later touch of the issue would start another session) and I am told
    assert tracker.calls == [("remove_label", ("iss-2", "factory:refine"))] and effect_events(deps) == ["effect.remove_label"]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "nothing to refine" and "label was removed" in flags[0]["detail"]
    assert "already a crisp spec" in flags[0]["detail"] and flags[0]["dedupe_key"] == f"refine-nothing:ABC-2:{issue.updated_at}"
    assert notifies(deps)[-1]["title"] == "nothing to refine" and notifies(deps)[-1]["level"] == "info"
    tracker.calls.clear()

    deps.run_worker = FakeWorker(refine_output(status="blocked", change_summary="too thin to refine"))
    outcome = run_handler(Kind.REFINE, action, deps)
    assert outcome.status == "failed" and "too thin" in outcome.note and deps.state.flags()[-1]["reason"] == "refine blocked"

    deps.run_worker = FakeWorker(refine_output(description="   "))
    outcome = run_handler(Kind.REFINE, action, deps)
    assert outcome.status == "failed" and "empty" in outcome.note
    assert tracker.calls == [] and notifies(deps)[-1]["title"] != "refined ABC-2: Fix login validation message"

    deps.tracker = FakeTracker([refine_issue(labels=[])])
    assert run_handler(Kind.REFINE, action, deps).status == "skipped"


def test_refine_stale_pending_proposal_is_discarded_and_the_worker_reruns(make_deps: Callable[..., Deps]) -> None:
    """A proposal saved but not applied (notification failed) must not overwrite the human's later
    edits of the issue: it is discarded and the worker runs again on the current text."""
    issue = refine_issue()
    tracker = FakeTracker([issue])
    deps = make_deps(tracker=tracker, worker=FakeWorker(refine_output()), toaster=FakeToaster(fail=True))
    assert run_handler(Kind.REFINE, issue_action(Kind.REFINE, issue), deps).status == "failed"
    assert deps.state.get_refinement("ABC-2")["status"] == "pending"  # type: ignore[index]

    edited = refine_issue(description=issue.description + "\n\nAlso broken on mobile.", updated_at="2026-09-30T11:00:00Z")
    tracker.issues[edited.id] = edited
    new_description = "## Context\n\nLogin breaks, also on mobile.\n\n## Acceptance criteria\n\n1. Mobile too"
    worker = FakeWorker(refine_output(description=new_description, change_summary="covers mobile"))
    deps.run_worker = worker
    deps.notifier = Notifier(deps.profile, deps.audit, toaster=FakeToaster())

    outcome = run_handler(Kind.REFINE, issue_action(Kind.REFINE, edited), deps)

    assert outcome.status == "done" and len(worker.calls) == 1  # the worker ran again on the current text
    assert "Also broken on mobile." in worker.calls[0]["prompt"]
    assert [c for c in tracker.calls if c[0] == "update_issue_text"] == [
        ("update_issue_text", ("iss-2", "Fix login validation message", new_description))
    ]
    record = deps.state.get_refinement("ABC-2")
    assert record is not None and record["status"] == "applied" and record["original_description"] == edited.description
    stale = [f for f in deps.state.flags() if f["reason"] == "refinement discarded"]
    assert len(stale) == 1 and "title or description changed" in stale[0]["detail"]
    assert "refine.stale" in [r["event"] for r in deps.audit.read()]

    # the label removed by hand while a proposal waits: discarded too, nothing edited, no worker
    deps.state.save_refinement("iss-9", "ABC-9", "T", "D", "T2", "D2")
    deps.tracker = FakeTracker([refine_issue(id="iss-9", key="ABC-9", title="T", description="D", labels=[])])
    deps.run_worker = no_worker
    outcome = run_handler(Kind.REFINE, issue_action(Kind.REFINE, refine_issue(id="iss-9", key="ABC-9")), deps)
    assert outcome.status == "skipped" and deps.state.get_refinement("ABC-9")["status"] == "reverted"  # type: ignore[index]
    assert not any(c[1][0] == "iss-9" for c in tracker.calls)


def test_refine_with_recording_effects_records_the_edits_only(make_deps: Callable[..., Deps]) -> None:
    issue = refine_issue()
    tracker = FakeTracker([issue], raise_on_write=True)
    deps = make_deps(tracker=tracker, worker=FakeWorker(refine_output()), recording=True)
    outcome = run_handler(Kind.REFINE, issue_action(Kind.REFINE, issue), deps)
    assert outcome.status == "dry_run" and "recorded only" in outcome.note
    assert effect_events(deps) == ["would_update_issue_text", "would_remove_label"] and tracker.calls == []
    # the record stops at ``notified``: a full-mode run later applies the edit without a second notice
    assert deps.state.get_refinement("ABC-2")["status"] == "notified"  # type: ignore[index]


def test_revert_refine_in_observe_mode_keeps_the_record_restorable(make_deps: Callable[..., Deps], paths: ProfilePaths) -> None:
    issue = refine_issue()
    tracker = FakeTracker([issue])
    deps = make_deps(tracker=tracker, worker=FakeWorker(refine_output()))
    assert run_handler(Kind.REFINE, issue_action(Kind.REFINE, issue), deps).status == "done"
    live_effects = deps.effects
    deps.effects = RecordingEffects(deps.profile, paths, deps.state, deps.gh, tracker, deps.audit, raising_git)  # type: ignore[arg-type]

    outcome = asyncio.run(revert_refine(deps, "ABC-2"))

    assert outcome.status == "dry_run" and "NOT restored" in outcome.note and "mode = \"full\"" in outcome.note
    assert deps.state.get_refinement("ABC-2")["status"] == "applied"  # type: ignore[index]
    assert [c[0] for c in tracker.calls].count("update_issue_text") == 1 and effect_events(deps)[-1] == "would_update_issue_text"
    # back in full mode the original can still be restored
    deps.effects = live_effects
    outcome = asyncio.run(revert_refine(deps, "ABC-2"))
    assert outcome.status == "done" and "restored" in outcome.note
    assert deps.state.get_refinement("ABC-2")["status"] == "reverted"  # type: ignore[index]
    assert tracker.calls[-1] == ("update_issue_text", ("iss-2", "Fix login", issue.description))


# -------------------------------------------------------------------------------------- plan
def plan_issue(**over: Any) -> Issue:
    base: dict[str, Any] = dict(
        id="iss-3", key="ABC-3", title="Big feature", description="Do a big thing across the API and the client.",
        url="https://linear.app/x/issue/ABC-3", labels=["factory:plan"], state_type="backlog", team_id="team-1",
        team_key="ABC", updated_at="2026-09-30T08:00:00Z",
    )  # fmt: skip
    base.update(over)
    return Issue(**base)


def plan_output(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": "done",
        "summary": "API first, then the client.",
        "sub_issues": [
            {"title": "Add the API endpoint", "description": "Add POST /things in src/api.py",
             "acceptance_criteria": ["POST /things returns 201"], "repo": SLUG, "blocked_by": []},
            {"title": "Wire the client", "description": "Call the endpoint from the form",
             "acceptance_criteria": ["the button creates a thing"], "repo": "repo", "blocked_by": [0]},
        ],
        "open_questions": ["Should drafts be allowed?"],
    }  # fmt: skip
    base.update(over)
    return base


def test_plan_creates_sub_issues_with_repo_labels_and_relations(
    make_deps: Callable[..., Deps], paths: ProfilePaths, repos_root: Path
) -> None:
    parent = plan_issue()
    tracker = FakeTracker([parent], existing_labels=("repo:repo", "factory:build"))
    worker = FakeWorker(plan_output())
    deps = make_deps(tracker=tracker, worker=worker)

    outcome = run_handler(Kind.PLAN, issue_action(Kind.PLAN, parent), deps)

    assert outcome.status == "done" and outcome.record_handled and outcome.cost_usd == 0.5
    assert outcome.note == "2 sub-issue(s) created: ABC-101, ABC-102"
    creates = [c for c in tracker.calls if c[0] == "create_sub_issue"]
    assert [c[1][1] for c in creates] == ["Add the API endpoint", "Wire the client"]
    assert [c[1][3] for c in creates] == [["repo:repo"], ["repo:repo"]]  # never factory:build
    assert "## Acceptance criteria\n\n1. POST /things returns 201" in creates[0][1][2]
    assert REPO_MARKER.format(slug=SLUG) not in creates[0][1][2]  # the label exists, no marker needed
    assert ("add_blocked_by", ("child-2", "child-1")) in tracker.calls
    assert tracker.calls[-1] == ("move_to_state_type", ("iss-3", "started"))
    assert [c[0] for c in tracker.calls] == ["create_sub_issue", "create_sub_issue", "add_blocked_by", "move_to_state_type"]
    assert effect_events(deps) == ["effect.create_sub_issue", "effect.create_sub_issue", "effect.add_blocked_by", "effect.transition_issue"]

    # the children's repo is cached for the resolver (no chain, no picker, for them)
    for child_id in ("child-1", "child-2"):
        resolution = deps.state.get_resolution(child_id)
        assert resolution is not None and resolution["repo"] == SLUG and resolution["source"] == "plan"
    child = tracker.get_issue("child-2")
    resolver = Resolver(deps.profile, tracker, deps.repos, deps.state)  # type: ignore[arg-type]
    assert asyncio.run(resolver.resolve(child)) == Resolution(repo=SLUG, confidence=1.0, source="plan", candidates=[SLUG])

    note = notifies(deps)[-1]
    assert note["title"] == "plan ready for review" and note["target"] == "ABC-3"
    body = note["body"]
    assert "ABC-101: Add the API endpoint [JaneDoe/repo]" in body
    assert "ABC-102: Wire the client [JaneDoe/repo] (blocked by ABC-101)" in body
    assert "Should drafts be allowed?" in body and "`factory:build`" in body and "API first" in body

    call = worker.calls[0]
    assert call["kind"] == "plan" and call["schema"] is PlanOutput and call["policy"].read_only is True
    assert call["cwd"] == paths.root / "scratch" and call["policy"].extra_read_dirs == [repos_root.resolve()]
    assert call["policy"].check("Read", {"file_path": str(repos_root / "repo" / "README.md")}).allow
    prompt = call["prompt"]
    positions = heading_positions(prompt, "## Action", "## Issue", "## Candidate repos")
    assert positions == sorted(positions)
    assert "# ABC-3: Big feature" in prompt and f"### {SLUG}" in prompt and str(repos_root) in prompt


def test_plan_uses_the_repo_marker_when_the_label_does_not_exist(make_deps: Callable[..., Deps]) -> None:
    parent = plan_issue()
    tracker = FakeTracker([parent], existing_labels=())
    deps = make_deps(tracker=tracker, worker=FakeWorker(plan_output()))
    assert run_handler(Kind.PLAN, issue_action(Kind.PLAN, parent), deps).status == "done"
    creates = [c for c in tracker.calls if c[0] == "create_sub_issue"]
    assert [c[1][3] for c in creates] == [[], []]
    assert all(REPO_MARKER.format(slug=SLUG) in c[1][2] for c in creates)


@pytest.mark.parametrize(
    ("patch", "problem"),
    [
        ({"sub_issues": [{"title": "x", "description": "d", "repo": "nope/x", "blocked_by": []}]}, "not a managed repo"),
        ({"sub_issues": [{"title": "x", "description": "d", "repo": SLUG, "blocked_by": [5]}]}, "out of range"),
        ({"sub_issues": [{"title": "x", "description": "d", "repo": SLUG, "blocked_by": [0]}]}, "blocked by itself"),
        (
            {"sub_issues": [
                {"title": "a", "description": "d", "repo": SLUG, "blocked_by": [1]},
                {"title": "b", "description": "d", "repo": SLUG, "blocked_by": [2]},
                {"title": "c", "description": "d", "repo": SLUG, "blocked_by": [0]},
            ]},
            "cycle: 0 -> 1 -> 2 -> 0",
        ),
        ({"sub_issues": []}, "no sub-issues"),
    ],
)
def test_plan_invalid_specs_fail_flag_and_create_nothing(make_deps: Callable[..., Deps], patch: dict, problem: str) -> None:
    parent = plan_issue()
    tracker = FakeTracker([parent])
    deps = make_deps(tracker=tracker, worker=FakeWorker(plan_output(**patch)))

    outcome = run_handler(Kind.PLAN, issue_action(Kind.PLAN, parent), deps)

    assert outcome.status == "failed" and not outcome.record_handled and problem in outcome.note
    assert tracker.calls == [] and effect_events(deps) == []
    assert deps.state.get_resolution("child-1") is None
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "invalid plan" and problem in flags[0]["detail"]
    assert "nothing was created" in flags[0]["detail"]


def test_plan_children_already_exist_is_nothing(make_deps: Callable[..., Deps]) -> None:
    parent = plan_issue(children=[IssueRef(id="c1", key="ABC-50", state_type="backlog", title="first")])
    tracker = FakeTracker([parent])
    deps = make_deps(tracker=tracker, worker=no_worker)
    outcome = run_handler(Kind.PLAN, issue_action(Kind.PLAN, parent), deps)
    assert outcome.status == "nothing" and outcome.record_handled and "ABC-50" in outcome.note
    assert tracker.calls == [] and notifies(deps) == []

    deps.tracker = FakeTracker([plan_issue(labels=["factory:build"])])
    assert run_handler(Kind.PLAN, issue_action(Kind.PLAN, parent), deps).status == "skipped"


def test_plan_worker_nothing_and_blocked(make_deps: Callable[..., Deps]) -> None:
    parent = plan_issue()
    tracker = FakeTracker([parent])
    deps = make_deps(tracker=tracker, worker=FakeWorker(plan_output(status="nothing", sub_issues=[], summary="one PR's worth")))
    action = issue_action(Kind.PLAN, parent)
    outcome = run_handler(Kind.PLAN, action, deps)
    assert outcome.status == "nothing" and outcome.note == "one PR's worth" and tracker.calls == []
    # the label stays (plan never removes labels): say so once instead of re-planning on every edit
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "nothing to plan" and flags[0]["dedupe_key"] == "plan-nothing:ABC-3"
    assert "one PR's worth" in flags[0]["detail"] and "`factory:build`" in flags[0]["detail"]
    # the order matters: labelling for build edits the issue, so a plan label left on would re-plan it first
    assert "remove the `factory:plan` label, then label `factory:build` to build it as one PR" in flags[0]["detail"]
    assert notifies(deps)[-1]["title"] == "nothing to plan" and notifies(deps)[-1]["level"] == "info"
    run_handler(Kind.PLAN, issue_action(Kind.PLAN, plan_issue(updated_at="2026-09-30T09:00:00Z")), deps)
    assert len(deps.state.flags()) == 1  # a later edit of the parent does not re-notify

    deps.run_worker = FakeWorker(plan_output(status="blocked", sub_issues=[], summary="", open_questions=["Which API?"]))
    outcome = run_handler(Kind.PLAN, action, deps)
    assert outcome.status == "failed" and "Which API?" in outcome.note and deps.state.flags()[-1]["reason"] == "plan blocked"
    assert tracker.calls == []


def test_plan_with_recording_effects_creates_nothing(make_deps: Callable[..., Deps]) -> None:
    parent = plan_issue()
    tracker = FakeTracker([parent], raise_on_write=True)
    deps = make_deps(tracker=tracker, worker=FakeWorker(plan_output()), recording=True)
    outcome = run_handler(Kind.PLAN, issue_action(Kind.PLAN, parent), deps)
    assert outcome.status == "done" and "dry-run" in outcome.note and tracker.calls == []
    assert effect_events(deps) == ["would_create_sub_issues", "would_transition_issue"]
    assert deps.state.get_resolution("dry-run:iss-3:0")["repo"] == SLUG  # type: ignore[index]
    assert notifies(deps)[-1]["title"] == "plan ready for review"


def test_validate_specs_and_find_cycle(world: dict) -> None:
    repos = RepoIndex([world["repo"]])
    specs = [
        SubIssueSpec(title="a", description="d", repo="repo", blocked_by=[]),
        SubIssueSpec(title="b", description="d", repo=SLUG, blocked_by=[0, 0]),
    ]
    normalised, problems = validate_specs(specs, repos)
    assert problems == [] and [s.repo for s in normalised] == [SLUG, SLUG] and normalised[1].blocked_by == [0]
    assert find_cycle([[], [0], [1]]) == [] and find_cycle([[1], [0]]) == [0, 1, 0]
    assert find_cycle([[], [2], [1]]) == [1, 2, 1] and find_cycle([]) == []
    _, problems = validate_specs([SubIssueSpec(title=" ", description="d", repo="", blocked_by=[-1])], repos)
    assert any("empty title" in p for p in problems) and any("not a managed repo" in p for p in problems)
    assert any("out of range" in p for p in problems)
