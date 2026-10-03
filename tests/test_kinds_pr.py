"""PR kind handlers (``kinds/ci.py``, ``conflict.py``, ``rc.py``, ``rv.py``) and their shared
pre-flight (``kinds/_pr.py``).

Everything runs against a real tmp git world (bare ``origin`` + clone with a ``feat`` branch as the
PR head and a diverged ``main``), real ``WorktreeManager`` worktrees, a fake ``gh`` (canned
``pr_details`` / ``failed_log``), a fake ``run_worker`` that returns canned structured output (and
may commit into the worktree first), and a recording fake ``Effects``. No network, no real worker.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import run_git  # pytest puts tests/ on sys.path (no package)
from pydantic import BaseModel

from software_factory import dispatch
from software_factory.actions import Action, Kind
from software_factory.config import Profile, RepoOverride
from software_factory.effects import LiveEffects, RecordingEffects
from software_factory.kinds import _pr as pr_mod
from software_factory.kinds import get_handler
from software_factory.kinds.base import AuthFailure, Deps, Outcome, ProviderUnavailable
from software_factory.kinds.ci import HANDLER as CI
from software_factory.kinds.conflict import HANDLER as CONFLICT
from software_factory.kinds.rc import HANDLER as RC
from software_factory.kinds.rc import select_feedback
from software_factory.kinds.rv import HANDLER as RV
from software_factory.notify import Audit, Notifier
from software_factory.paths import ProfilePaths
from software_factory.repos import RepoIndex, RepoInfo
from software_factory.runner import RunSummary, WorkerRun
from software_factory.sources.github import GhAuthError, GhError
from software_factory.sources.models import Check, Feedback, PullRequest, ReviewThread
from software_factory.state import State
from software_factory.workers.schemas import (
    CiFailure,
    CiOutput,
    CommentDisposition,
    ConflictOutput,
    HunkResolution,
    RcOutput,
    ReviewFinding,
    RvOutput,
    VerificationRun,
)
from software_factory.worktree import Worktree, WorktreeManager

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

SLUG = "JaneDoe/repo"
PR_NUMBER = 7
TARGET = f"{SLUG}#{PR_NUMBER}"
PR_URL = f"https://github.com/{SLUG}/pull/{PR_NUMBER}"
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
RUN_ID = 11
LOG_TAIL = "FAIL tests/test_feature.py::test_adds - assert 2 == 3\n1 failed, 12 passed in 1.2s\n"
CLAUDE_MD = "# repo\n\nA test repo.\n\n## Verification\n\n- `uv run pytest -q`\n"
FEEDBACK_AT = "2026-09-30T10:00:00Z"


# ------------------------------------------------------------------------------------- world
@dataclass
class World:
    bare: Path
    clone: Path
    repo: RepoInfo
    head: str  # tip of ``feat`` (the PR head)
    base: str  # tip of ``main`` (diverged from ``feat`` on src/app.py)


@pytest.fixture
def world(bare_repo_factory: Callable[..., Path], git_repo_factory: Callable[..., Path]) -> World:
    bare = bare_repo_factory("origin")
    clone = git_repo_factory(
        "repo", origin=bare, files={"README.md": "# repo\n", "CLAUDE.md": CLAUDE_MD, "src/app.py": "x = 1\n"}
    )
    run_git(["checkout", "-q", "-b", "feat"], clone)
    (clone / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    (clone / "src" / "feature.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    run_git(["add", "-A"], clone)
    run_git(["commit", "-q", "-m", "feature"], clone)
    head = run_git(["rev-parse", "HEAD"], clone)
    run_git(["push", "-q", "origin", "feat"], clone)
    run_git(["checkout", "-q", "main"], clone)
    (clone / "src" / "app.py").write_text("x = 3\n", encoding="utf-8")
    run_git(["commit", "-q", "-am", "main moves"], clone)
    base = run_git(["rev-parse", "HEAD"], clone)
    run_git(["push", "-q", "origin", "main"], clone)
    repo = RepoInfo(slug=SLUG, name="repo", dirname="repo", path=clone, default_branch="main", override=RepoOverride())
    return World(bare=bare, clone=clone, repo=repo, head=head, base=base)


def failing_check(name: str = "test (3.13)", run_id: int | None = RUN_ID, job_id: int | None = 22) -> Check:
    url = f"https://github.com/{SLUG}/actions/runs/{run_id}/job/{job_id}" if run_id is not None else "https://codecov.io/x"
    return Check(name=name, state="failure", workflow="CI" if run_id is not None else "", details_url=url, run_id=run_id, job_id=job_id)


def make_pr(world: World, **over: Any) -> PullRequest:
    fields: dict[str, Any] = dict(
        repo=SLUG, number=PR_NUMBER, title="Add feature", url=PR_URL, author="JaneDoe", state="OPEN",
        head_ref_name="feat", head_ref_oid=world.head, base_ref_name="main", base_ref_oid=world.base,
        mergeable="MERGEABLE", body="Adds `add`.", updated_at="2026-09-30T09:00:00Z",
    )  # fmt: skip
    fields.update(over)
    return PullRequest(**fields)


def feedback_items() -> tuple[list[Feedback], list[ReviewThread]]:
    items = [
        Feedback(id="c1", author="alice", body="Please rename add to plus", created_at="2026-09-30T09:30:00Z", kind="review", url="https://x/c1"),
        Feedback(id="c2", author="cursor[bot]", author_is_bot=True, body="Possible overflow", created_at=FEEDBACK_AT, kind="thread",
                 path="src/feature.py", line=2, thread_id="t1", url="https://x/c2"),
        Feedback(id="c3", author="bob", body="older comment", created_at="2026-09-29T09:00:00Z", kind="comment"),
    ]  # fmt: skip
    threads = [ReviewThread(id="t1", is_resolved=False, is_outdated=False, path="src/feature.py", line=2, comments=[items[1]])]
    return items, threads


# ------------------------------------------------------------------------------------- fakes
class FakeGitHub:
    """``pr_details`` / ``pr_state`` return the canned PR; ``failed_log`` serves ``logs`` by run id."""

    def __init__(self, pr: PullRequest, *, logs: dict[int, str] | None = None, error: Exception | None = None) -> None:
        self.pr = pr
        self.logs = logs if logs is not None else {RUN_ID: "[... first 1200 bytes omitted ...]\n" + LOG_TAIL}
        self.error = error
        self.calls: list[tuple[Any, ...]] = []

    def login(self) -> str:
        return "JaneDoe"

    def auth_check(self) -> None:
        pass

    def pr_details(self, repo: str, number: int) -> PullRequest:
        self.calls.append(("pr_details", repo, number))
        if self.error is not None:
            raise self.error
        return self.pr

    def pr_state(self, repo: str, number: int) -> PullRequest:
        self.calls.append(("pr_state", repo, number))
        return self.pr

    def failed_log(self, repo: str, run_id: int, max_bytes: int = 60_000) -> str:
        self.calls.append(("failed_log", repo, run_id))
        if run_id not in self.logs:
            raise GhError(f"gh run view {run_id} failed (1): run not found")
        return self.logs[run_id]


class FakeEffects:
    """Records pushes / evidence edits; ``write_review`` really writes the local file."""

    def __init__(self, paths: ProfilePaths) -> None:
        self.paths = paths
        self.pushes: list[tuple[Worktree, str, str, int | None, str]] = []  # + the pushed sha
        self.evidence: list[tuple[str, int, str]] = []
        self.reviews: list[Path] = []

    def push(self, wt: Worktree, branch: str, *, kind: str, pr: int | None) -> str:
        sha = run_git(["rev-parse", "HEAD"], wt.path)
        self.pushes.append((wt, branch, kind, pr, sha))
        return sha

    def update_pr_evidence(self, repo: str, number: int, section: str) -> None:
        self.evidence.append((repo, number, section))

    def write_review(self, repo: str, number: int, markdown: str) -> Path:
        path = self.paths.review_file(repo.split("/")[-1], number)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown if markdown.endswith("\n") else markdown + "\n", encoding="utf-8")
        self.reviews.append(path)
        return path


class NoPushEffects(FakeEffects):
    def push(self, wt: Worktree, branch: str, *, kind: str, pr: int | None) -> str:
        raise AssertionError("push must never be called by this kind")


class FakeWorker:
    """Returns canned structured output; ``script(cwd)`` runs first (to commit into the worktree)."""

    def __init__(
        self,
        output: BaseModel | dict | None = None,
        *,
        script: Callable[[Path], None] | None = None,
        error: str | None = None,
        auth_failed: bool = False,
        timed_out: bool = False,
        rate_limited: bool = False,
    ) -> None:
        self.output = output
        self.script = script
        self.error = error
        self.auth_failed = auth_failed
        self.timed_out = timed_out
        self.rate_limited = rate_limited
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> WorkerRun:
        self.calls.append(kwargs)
        if self.script is not None:
            self.script(Path(kwargs["cwd"]))
        out = self.output.model_dump(mode="json") if isinstance(self.output, BaseModel) else self.output
        return WorkerRun(
            output=out, error=self.error, auth_failed=self.auth_failed, timed_out=self.timed_out,
            rate_limited=self.rate_limited, summary=RunSummary(cost_usd=0.25, usage={"input_tokens": 10}, duration_ms=42),
        )  # fmt: skip

    @property
    def cwd(self) -> Path:
        return Path(self.calls[0]["cwd"])

    @property
    def prompt(self) -> str:
        return str(self.calls[0]["prompt"])


def commit_files(files: dict[str, str], message: str = "fix") -> Callable[[Path], None]:
    def script(cwd: Path) -> None:
        for name, content in files.items():
            target = cwd / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        run_git(["add", "-A"], cwd)
        run_git(["commit", "-q", "-m", message], cwd)

    return script


def merge_main(resolve: str | None) -> Callable[[Path], None]:
    """``git merge origin/main`` in the worktree (conflicts on src/app.py); resolve + commit when
    ``resolve`` is given, else leave the markers exactly as git left them."""

    def script(cwd: Path) -> None:
        run_git(["fetch", "-q", "origin", "main"], cwd)
        run_git(["merge", "origin/main"], cwd, check=False)
        assert "<<<<<<<" in (cwd / "src" / "app.py").read_text(encoding="utf-8")
        if resolve is not None:
            (cwd / "src" / "app.py").write_text(resolve, encoding="utf-8")
            run_git(["add", "src/app.py"], cwd)
            run_git(["commit", "-q", "-m", "merge origin/main into feat"], cwd)

    return script


# ---------------------------------------------------------------------------------- fixtures
@pytest.fixture
def paths(tmp_path: Path) -> ProfilePaths:
    return ProfilePaths.for_profile("personal", root=tmp_path / "data root").ensure()


@pytest.fixture
def quiet_profile(make_profile: Callable[..., Path], load: Callable[[str], Profile]) -> Callable[..., Profile]:
    def _build(mode: str = "full") -> Profile:
        make_profile(
            "personal",
            {"profile": {"mode": mode, "session_timeout_min": 5}, "notify": {"desktop": False, "slack": False},
             "github": {"owners": ["JaneDoe"], "login": "JaneDoe"}},
            {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test"},
        )  # fmt: skip
        return load("personal")

    return _build


@pytest.fixture
def make_deps(paths: ProfilePaths, quiet_profile: Callable[..., Profile], world: World) -> Iterator[Callable[..., Deps]]:
    states: list[State] = []

    def _make(
        pr: PullRequest,
        *,
        worker: FakeWorker | None = None,
        effects: Any = None,
        gh: Any = None,
        mode: str = "full",
        state: State | None = None,
    ) -> Deps:
        profile = quiet_profile(mode)
        state = state or State(paths.state_db)
        states.append(state)
        audit = Audit(paths.audit_log, profile.name)
        return Deps(
            profile=profile, paths=paths, state=state, gh=gh or FakeGitHub(pr), tracker=None,
            repos=RepoIndex([world.repo]), effects=effects or FakeEffects(paths), notifier=Notifier(profile, audit),
            audit=audit, wt_manager=WorktreeManager(paths.worktrees), resolver=None,
            run_worker=worker or FakeWorker(), now=lambda: NOW, run_id="run-1",
        )  # type: ignore[arg-type]  # fmt: skip

    yield _make
    for st in states:
        st.close()


# ----------------------------------------------------------------------------------- helpers
def ci_action(world: World, *, version: str | None = None, number: int = PR_NUMBER, repo: str = SLUG) -> Action:
    failing = [{"name": "test (3.13)", "run_id": RUN_ID, "job_id": 22, "workflow": "CI", "details_url": "https://x"}]
    return Action(kind=Kind.CI, version=version or world.head, repo=repo, pr=number, title="Add feature", url=PR_URL,
                  context={"head_ref_name": "feat", "failing": failing})  # fmt: skip


def conflict_action(world: World, *, version: str | None = None) -> Action:
    return Action(kind=Kind.CONFLICT, version=version or f"{world.head}:{world.base}", repo=SLUG, pr=PR_NUMBER, title="Add feature", url=PR_URL)


def rc_action(world: World, *, version: str = FEEDBACK_AT, feedback_ids: list[str] | None = None) -> Action:
    ids = ["c1", "c2"] if feedback_ids is None else feedback_ids
    return Action(kind=Kind.RC, version=version, repo=SLUG, pr=PR_NUMBER, title="Add feature", url=PR_URL, context={"feedback_ids": ids})


def rv_action(world: World, *, version: str | None = None) -> Action:
    return Action(kind=Kind.RV, version=version or world.head, repo=SLUG, pr=PR_NUMBER, title="Add feature", url=PR_URL, context={"author": "alice"})


def passing() -> list[VerificationRun]:
    return [VerificationRun(command="uv run pytest -q tests/test_feature.py", exit_code=0, passed=True)]


def ci_output(**over: Any) -> CiOutput:
    base: dict[str, Any] = dict(
        status="done", summary="The assertion in test_adds was wrong; fixed the expected value.", push_ok=True,
        head_verified=True, verification=passing(),
        failures=[CiFailure(check="test (3.13)", classification="caused_by_pr", cause="the PR changed add()", fixed=True)],
    )  # fmt: skip
    base.update(over)
    return CiOutput(**base)


def conflict_output(**over: Any) -> ConflictOutput:
    base: dict[str, Any] = dict(
        status="done", summary="Merged origin/main; one trivial hunk in src/app.py.", push_ok=True, head_verified=True,
        all_resolved=True, verification=passing(),
        hunks=[HunkResolution(file="src/app.py", classification="trivial", resolution="kept the PR's value")],
    )  # fmt: skip
    base.update(over)
    return ConflictOutput(**base)


def rc_output(**over: Any) -> RcOutput:
    base: dict[str, Any] = dict(
        status="done", summary="Renamed add to plus as asked; the overflow claim is not real.", push_ok=True,
        head_verified=True, verification=passing(),
        comments=[
            CommentDisposition(comment_id="c1", author="alice", disposition="addressed", note="renamed"),
            CommentDisposition(comment_id="c2", author="cursor[bot]", disposition="disagree", note="ints do not overflow"),
        ],
    )  # fmt: skip
    base.update(over)
    return RcOutput(**base)


def rv_output(**over: Any) -> RvOutput:
    base: dict[str, Any] = dict(
        status="done", summary="Approve; one minor edge case.", head_verified=True, verdict="approve",
        findings=[ReviewFinding(severity="minor", file="src/feature.py", line=2, claim="no type check", evidence="read the diff")],
        review_markdown="## Review\n\nLooks good overall.\n\n- minor: `src/feature.py:2` has no type check\n",
    )  # fmt: skip
    base.update(over)
    return RvOutput(**base)


def run(handler: Any, action: Action, deps: Deps) -> Outcome:
    return asyncio.run(handler.run(action, deps))


def worktree_dirs(paths: ProfilePaths) -> list[Path]:
    return sorted(p for p in paths.worktrees.iterdir()) if paths.worktrees.is_dir() else []


def notifies(deps: Deps) -> list[dict]:
    return [r for r in deps.audit.read() if r["event"] == "notify"]


# ------------------------------------------------------------------------------- registration
def test_handlers_are_registered_for_every_pr_kind() -> None:
    for kind, handler in ((Kind.CI, CI), (Kind.CONFLICT, CONFLICT), (Kind.RC, RC), (Kind.RV, RV)):
        assert get_handler(kind) is handler and handler.kind == kind


# ----------------------------------------------------------------------------------------- ci
def test_ci_happy_path_pushes_and_is_recorded_handled(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    worker = FakeWorker(ci_output(), script=commit_files({"tests/test_feature.py": "def test_adds():\n    assert 1 + 1 == 2\n"}))
    deps = make_deps(make_pr(world, checks=[failing_check(), failing_check("lint", RUN_ID, 23), failing_check("codecov", None, None)]), worker=worker)
    action = ci_action(world)

    outcome = asyncio.run(dispatch.run_one(deps.profile, paths, action, deps=deps, print_fn=lambda _l: None, force=True))

    assert outcome.status == "done" and outcome.pushed is True and outcome.record_handled
    assert outcome.cost_usd == 0.25 and outcome.usage == {"input_tokens": 10} and outcome.duration_ms == 42
    assert outcome.pr_url == PR_URL
    assert deps.state.is_handled(TARGET, "ci", world.head)  # the dispatcher recorded it
    effects: FakeEffects = deps.effects  # type: ignore[assignment]
    assert [(p[1], p[2], p[3]) for p in effects.pushes] == [("feat", "ci", PR_NUMBER)]
    assert effects.pushes[0][0].path == worker.cwd and effects.evidence == []  # not a factory PR
    assert not worker.cwd.exists() and worktree_dirs(paths) == []
    assert deps.state.kept_worktree(TARGET) is None and deps.state.flags() == []
    # one failed-log fetch per distinct Actions run; the codecov check has no run
    assert deps.gh.calls.count(("failed_log", SLUG, RUN_ID)) == 1  # type: ignore[attr-defined]
    call = worker.calls[0]
    assert call["kind"] == "ci" and call["schema"] is CiOutput and call["policy"].read_only is False
    assert call["cwd"] == worker.cwd and call["policy"].worktree == worker.cwd.resolve()


def test_ci_prompt_has_the_headings_pr_fields_log_tail_and_verification(make_deps: Callable[..., Deps], world: World) -> None:
    worker = FakeWorker(ci_output(), script=commit_files({"tests/test_feature.py": "ok\n"}))
    checks = [failing_check(), failing_check("lint", RUN_ID, 23), failing_check("codecov/patch", None, None)]
    deps = make_deps(make_pr(world, checks=checks), worker=worker)

    run(CI, ci_action(world), deps)

    prompt = worker.prompt
    positions = [prompt.index(h) for h in ("## Action", "## Pull request", "## Failing checks", "## Verification commands")]
    assert positions == sorted(positions)
    assert "- kind: `ci`" in prompt and f"- target: {TARGET}" in prompt
    assert f"headRefOid: `{world.head}`" in prompt and "headRefName: `feat`" in prompt and f"baseRefOid: `{world.base}`" in prompt
    assert "### test (3.13) (workflow: CI)" in prompt and f"run_id: {RUN_ID}, job_id: 22" in prompt
    assert LOG_TAIL.strip() in prompt and prompt.count("FAIL tests/test_feature.py::test_adds") == 2  # once per check on the run
    assert "### codecov/patch" in prompt and "not a GitHub Actions run" in prompt and "https://codecov.io/x" in prompt
    assert "- test: `uv run pytest -q` (from CLAUDE.md)" in prompt
    assert call_prompt_sections(prompt) == ["## Action", "## Pull request", "## Failing checks", "## Verification commands"]


def call_prompt_sections(prompt: str) -> list[str]:
    return [line for line in prompt.splitlines() if line.startswith("## ")]


def test_ci_live_push_lands_on_the_local_bare_remote(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    worker = FakeWorker(ci_output(), script=commit_files({"tests/test_feature.py": "ok\n"}))
    deps = make_deps(make_pr(world, checks=[failing_check()]), worker=worker)
    deps.effects = LiveEffects(deps.profile, paths, deps.state, None, None, deps.audit)

    outcome = run(CI, ci_action(world), deps)

    assert outcome.status == "done" and outcome.pushed is True
    remote = run_git(["ls-remote", "--heads", "origin", "feat"], world.clone).split()[0]
    pushes = deps.state.pushes(SLUG, PR_NUMBER)
    assert len(pushes) == 1 and pushes[0]["after_sha"] == remote != world.head and pushes[0]["before_sha"] == world.head
    assert pushes[0]["branch"] == "feat" and pushes[0]["kind"] == "ci" and len(pushes[0]["commit_shas"]) == 1
    events = [r["event"] for r in deps.audit.read()]
    assert "effect.push" in events and not any(e.startswith("would_") for e in events)
    assert run_git(["log", "-1", "--format=%s", remote], world.clone) == "fix"


def test_ci_factory_pr_refreshes_evidence_and_names_the_issue(make_deps: Callable[..., Deps], world: World) -> None:
    worker = FakeWorker(ci_output(), script=commit_files({"tests/test_feature.py": "ok\n"}))
    deps = make_deps(make_pr(world, checks=[failing_check()]), worker=worker)
    deps.state.register_pr(SLUG, PR_NUMBER, "feat", "build", "ABC-1", PR_URL)

    outcome = run(CI, ci_action(world), deps)

    assert outcome.status == "done" and outcome.pushed
    effects: FakeEffects = deps.effects  # type: ignore[assignment]
    assert len(effects.evidence) == 1 and effects.evidence[0][:2] == (SLUG, PR_NUMBER)
    assert "`tests/test_feature.py`" in effects.evidence[0][2]
    assert "- linked issue: ABC-1" in worker.prompt


def test_ci_all_flaky_is_done_without_push_and_notifies_once(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    output = ci_output(
        push_ok=False, push_reasons=["nothing to push"], verification=[],
        summary="Both failures are unrelated to the diff.",
        failures=[CiFailure(check="test (3.13)", classification="flaky", cause="timeout in an unrelated socket test"),
                  CiFailure(check="lint", classification="infra", cause="runner lost network while installing")],
    )  # fmt: skip
    worker = FakeWorker(output)
    deps = make_deps(make_pr(world, checks=[failing_check(), failing_check("lint", RUN_ID, 23)]), worker=worker)

    outcome = run(CI, ci_action(world), deps)

    assert outcome.status == "done" and outcome.pushed is False and outcome.record_handled
    assert "test (3.13) flaky" in outcome.note and "lint infra" in outcome.note
    assert deps.effects.pushes == [] and not worker.cwd.exists() and worktree_dirs(paths) == []  # type: ignore[attr-defined]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "ci failures not caused by the pr"
    assert "test (3.13): flaky - timeout" in flags[0]["detail"] and "lint: infra" in flags[0]["detail"]
    assert flags[0]["dedupe_key"] == f"ci-classified:{TARGET}:{world.head}"
    assert notifies(deps)[0]["level"] == "info" and notifies(deps)[0]["target"] == TARGET
    assert deps.state.kept_worktree(TARGET) is None

    run(CI, ci_action(world), deps)  # e.g. ``factory run ci`` again at the same head: no second notification

    assert len(deps.state.flags()) == 1 and len(notifies(deps)) == 1 and worktree_dirs(paths) == []


def test_ci_flaky_classification_with_a_blocked_status_is_not_done(make_deps: Callable[..., Deps], world: World) -> None:
    output = ci_output(status="blocked", push_ok=False, push_reasons=["could not verify head"], head_verified=False,
                       failures=[CiFailure(check="test (3.13)", classification="flaky", cause="?")])  # fmt: skip
    deps = make_deps(make_pr(world, checks=[failing_check()]), worker=FakeWorker(output))
    outcome = run(CI, ci_action(world), deps)
    assert outcome.status == "failed" and "could not verify head" in outcome.note
    assert deps.state.flags()[0]["reason"] == "worker blocked"


@pytest.mark.parametrize(
    ("label", "files", "over", "needle"),
    [
        ("sensitive file", {"src/auth/login.py": "x\n"}, {}, "sensitive globs: src/auth/login.py"),
        ("no passing verification", {"tests/t.py": "x\n"}, {"verification": [VerificationRun(command="uv run pytest", exit_code=1, passed=False)]}, "no passing verification run"),
        ("head not verified", {"tests/t.py": "x\n"}, {"head_verified": False}, "did not verify the worktree HEAD"),
        ("worker push_ok false", {"tests/t.py": "x\n"}, {"push_ok": False, "push_reasons": ["touches prod logic"]}, "touches prod logic"),
    ],
)
def test_ci_policy_refusal_keeps_the_worktree_and_flags(
    make_deps: Callable[..., Deps], world: World, label: str, files: dict[str, str], over: dict, needle: str
) -> None:
    worker = FakeWorker(ci_output(**over), script=commit_files(files))
    deps = make_deps(make_pr(world, checks=[failing_check()]), worker=worker)

    outcome = run(CI, ci_action(world), deps)

    assert outcome.status == "kept" and outcome.pushed is False and not outcome.record_handled, label
    assert needle in outcome.note and str(worker.cwd) in outcome.note
    assert deps.effects.pushes == []  # type: ignore[attr-defined]
    assert worker.cwd.exists() and (worker.cwd / next(iter(files))).is_file()
    kept = deps.state.kept_worktree(TARGET)
    assert kept is not None and kept["path"] == str(worker.cwd) and kept["kind"] == "ci" and needle in kept["reason"]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "push refused" and needle in flags[0]["detail"]
    assert notifies(deps)[0]["title"] == "push refused"


def test_ci_no_failing_checks_now_is_nothing_or_pending_skip(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    green = Check(name="test (3.13)", state="success", run_id=RUN_ID)
    worker = FakeWorker(ci_output())
    deps = make_deps(make_pr(world, checks=[green]), worker=worker)
    outcome = run(CI, ci_action(world), deps)
    assert outcome.status == "nothing" and outcome.record_handled and worker.calls == [] and worktree_dirs(paths) == []

    deps = make_deps(make_pr(world, checks=[green, Check(name="lint", state="pending")]), worker=worker)
    outcome = run(CI, ci_action(world), deps)
    assert outcome.status == "skipped" and "pending" in outcome.note and worker.calls == [] and worktree_dirs(paths) == []


def test_ci_failed_log_fetch_error_is_reported_in_the_prompt_not_fatal(make_deps: Callable[..., Deps], world: World) -> None:
    worker = FakeWorker(ci_output(status="nothing", failures=[], push_ok=False))
    pr = make_pr(world, checks=[failing_check()])
    deps = make_deps(pr, worker=worker, gh=FakeGitHub(pr, logs={}))
    outcome = run(CI, ci_action(world), deps)
    assert outcome.status == "nothing"
    assert "(failed log unavailable: gh run view 11 failed" in worker.prompt


class RunListGitHub(FakeGitHub):
    """``FakeGitHub`` with the injectable runner ``GitHub`` has: serves ``gh run list`` from ``runs``
    (keyed by workflow name) and records every argv."""

    def __init__(self, pr: PullRequest, *, runs: dict[str, list[dict]] | None = None, run_error: Exception | None = None) -> None:
        super().__init__(pr)
        self.runs = runs or {}
        self.run_error = run_error
        self.argvs: list[list[str]] = []

    def _run(self, args: list[str], *, cwd: Any = None, input: str | None = None, timeout: float | None = None) -> str:
        self.argvs.append(list(args))
        if self.run_error is not None:
            raise self.run_error
        return json.dumps(self.runs.get(args[args.index("--workflow") + 1], []))


BASE_RUN = {"conclusion": "failure", "headSha": "b" * 40, "url": f"https://github.com/{SLUG}/actions/runs/99", "createdAt": "2026-09-30T08:00:00Z"}


def test_ci_prompt_names_the_base_branch_run_of_each_failing_workflow(make_deps: Callable[..., Deps], world: World) -> None:
    """Regression: the ci prompt told the worker to run ``gh run list --branch <base>`` itself to
    classify a failure as ``inherited``, but workers run gh unauthenticated. The engine looks the base
    run up with its own gh (one read per workflow) and puts it under ``## Failing checks``."""
    checks = [failing_check(), failing_check("lint", RUN_ID, 23), failing_check("codecov/patch", None, None)]
    pr = make_pr(world, checks=checks)
    gh = RunListGitHub(pr, runs={"CI": [BASE_RUN]})
    worker = FakeWorker(ci_output(status="nothing", failures=[], push_ok=False))
    deps = make_deps(pr, worker=worker, gh=gh)

    outcome = run(CI, ci_action(world), deps)

    assert outcome.status == "nothing"
    line = f"- base branch: failure at {'b' * 40} (https://github.com/{SLUG}/actions/runs/99)"
    section = worker.prompt.split("## Failing checks", 1)[1].split("## Verification commands", 1)[0]
    assert section.count(line) == 2  # test + lint share the CI workflow; codecov is no Actions run
    assert gh.argvs == [["run", "list", "-R", SLUG, "--branch", "main", "--workflow", "CI", "--limit", "1",
                         "--json", "conclusion,headSha,url,createdAt"]]  # fmt: skip
    # no run on the base yet / a run still going
    gh.runs = {}
    run(CI, ci_action(world), deps)
    assert "- base branch: no CI run found on `main`" in deps.run_worker.calls[-1]["prompt"]  # type: ignore[attr-defined]
    gh.runs = {"CI": [{**BASE_RUN, "conclusion": ""}]}
    run(CI, ci_action(world), deps)
    assert "- base branch: no conclusion yet (still running) at " in deps.run_worker.calls[-1]["prompt"]  # type: ignore[attr-defined]


def test_ci_base_branch_lookup_failure_is_a_note_and_auth_errors_stop_the_wave(
    make_deps: Callable[..., Deps], world: World, paths: ProfilePaths
) -> None:
    pr = make_pr(world, checks=[failing_check()])
    worker = FakeWorker(ci_output(status="nothing", failures=[], push_ok=False))
    deps = make_deps(pr, worker=worker, gh=RunListGitHub(pr, run_error=GhError("gh run list failed (1): HTTP 502")))
    assert run(CI, ci_action(world), deps).status == "nothing"
    assert "- base branch: not checked (gh run list failed (1): HTTP 502)" in worker.prompt

    worker = FakeWorker(ci_output())
    deps = make_deps(pr, worker=worker, gh=RunListGitHub(pr, run_error=GhAuthError("gh run list failed (1): HTTP 401")))
    with pytest.raises(GhAuthError):
        run(CI, ci_action(world), deps)
    assert worker.calls == [] and worktree_dirs(paths) == []  # no session, no leaked checkout


def test_ci_base_branch_lookup_prefers_the_clients_own_method(make_deps: Callable[..., Deps], world: World) -> None:
    class MethodGitHub(RunListGitHub):
        def latest_workflow_run(self, repo: str, branch: str, workflow: str) -> dict:
            self.calls.append(("latest_workflow_run", repo, branch, workflow))
            return {**BASE_RUN, "conclusion": "success"}

    pr = make_pr(world, checks=[failing_check()])
    gh = MethodGitHub(pr)
    worker = FakeWorker(ci_output(status="nothing", failures=[], push_ok=False))
    run(CI, ci_action(world), make_deps(pr, worker=worker, gh=gh))
    assert ("latest_workflow_run", SLUG, "main", "CI") in gh.calls and gh.argvs == []
    assert f"- base branch: success at {'b' * 40} (" in worker.prompt


# ------------------------------------------------------------------------------------ pre-flight
@pytest.mark.parametrize(
    ("handler", "make_action"),
    [(CI, lambda w: ci_action(w, version="deadbeef")), (CONFLICT, lambda w: conflict_action(w, version="dead:beef")),
     (RC, lambda w: rc_action(w, version="2020-01-01T00:00:00Z")), (RV, lambda w: rv_action(w, version="deadbeef"))],
    ids=["ci", "conflict", "rc", "rv"],
)  # fmt: skip
def test_stale_version_is_skipped_before_any_worktree(
    make_deps: Callable[..., Deps], world: World, paths: ProfilePaths, handler: Any, make_action: Callable[[World], Action]
) -> None:
    items, threads = feedback_items()
    worker = FakeWorker(ci_output())
    deps = make_deps(make_pr(world, checks=[failing_check()], mergeable="CONFLICTING", feedback=items, threads=threads), worker=worker)
    outcome = run(handler, make_action(world), deps)
    assert outcome.status == "skipped" and outcome.note.startswith("stale:") and not outcome.record_handled
    assert worker.calls == [] and worktree_dirs(paths) == [] and deps.state.flags() == []


def test_rc_version_compares_as_an_instant(make_deps: Callable[..., Deps], world: World) -> None:
    items, threads = feedback_items()
    deps = make_deps(make_pr(world, feedback=items, threads=threads), worker=FakeWorker(rc_output(status="nothing", push_ok=False)))
    outcome = run(RC, rc_action(world, version="2026-09-30T12:00:00+02:00"), deps)  # same instant as FEEDBACK_AT
    assert outcome.status == "nothing"


@pytest.mark.parametrize("state", ["MERGED", "CLOSED"])
def test_closed_or_merged_pr_is_skipped(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths, state: str) -> None:
    worker = FakeWorker(ci_output())
    deps = make_deps(make_pr(world, state=state, checks=[failing_check()]), worker=worker)
    for handler, action in ((CI, ci_action(world)), (CONFLICT, conflict_action(world)), (RV, rv_action(world))):
        outcome = run(handler, action, deps)
        assert outcome.status == "skipped" and outcome.note == f"pr {state.lower()}"
    assert worker.calls == [] and worktree_dirs(paths) == []


def test_fork_head_is_flagged_and_skipped_for_pushing_kinds_but_reviewed(
    make_deps: Callable[..., Deps], world: World, paths: ProfilePaths
) -> None:
    items, threads = feedback_items()
    pr = make_pr(world, head_repo="someone/repo", checks=[failing_check()], mergeable="CONFLICTING", feedback=items, threads=threads)
    worker = FakeWorker(ci_output())
    deps = make_deps(pr, worker=worker)
    for handler, action in ((CI, ci_action(world)), (CONFLICT, conflict_action(world)), (RC, rc_action(world))):
        outcome = run(handler, action, deps)
        assert outcome.status == "skipped" and "fork" in outcome.note and "cannot push" in outcome.note
    assert worker.calls == [] and worktree_dirs(paths) == []
    flags = deps.state.flags()
    assert [f["reason"] for f in flags] == ["pull request from a fork"] * 3
    assert {f["dedupe_key"] for f in flags} == {f"fork:{TARGET}:{k}:{v}" for k, v in (("ci", world.head), ("conflict", f"{world.head}:{world.base}"), ("rc", FEEDBACK_AT))}
    assert "someone/repo" in flags[0]["detail"]
    run(CI, ci_action(world), deps)
    assert len(deps.state.flags()) == 3  # flag-once per target/kind/version

    # rv reads a fork through pull/<n>/head and is fine with it
    run_git(["update-ref", f"refs/pull/{PR_NUMBER}/head", world.head], world.bare)
    rv_worker = FakeWorker(rv_output())
    deps = make_deps(pr, worker=rv_worker, effects=NoPushEffects(paths))
    outcome = run(RV, rv_action(world), deps)
    assert outcome.status == "done" and len(rv_worker.calls) == 1 and "head repo (fork): someone/repo" in rv_worker.prompt


def test_unmanaged_repo_and_non_pr_actions_are_skipped(make_deps: Callable[..., Deps], world: World) -> None:
    deps = make_deps(make_pr(world))
    assert run(CI, ci_action(world, repo="JaneDoe/other"), deps).status == "skipped"
    outcome = run(CI, Action(kind=Kind.CI, version="v", issue_id="i", issue_key="ABC-1"), deps)
    assert outcome.status == "skipped" and "not a pull request" in outcome.note
    assert deps.gh.calls == []  # type: ignore[attr-defined]


def test_pr_fetch_errors(make_deps: Callable[..., Deps], world: World) -> None:
    pr = make_pr(world)
    deps = make_deps(pr, gh=FakeGitHub(pr, error=GhError("gh pr view failed (1): rate limited")))
    outcome = run(CI, ci_action(world), deps)
    assert outcome.status == "failed" and "pr fetch failed" in outcome.note and "rate limited" in outcome.note
    deps = make_deps(pr, gh=FakeGitHub(pr, error=GhAuthError("HTTP 401")))
    with pytest.raises(GhAuthError):
        run(RV, rv_action(world), deps)


def test_worktree_failure_is_flagged_and_failed(make_deps: Callable[..., Deps], world: World) -> None:
    worker = FakeWorker(ci_output())
    deps = make_deps(make_pr(world, head_ref_oid="0" * 40, checks=[failing_check()]), worker=worker)
    outcome = run(CI, ci_action(world, version="0" * 40), deps)
    assert outcome.status == "failed" and outcome.note.startswith("worktree:") and worker.calls == []
    assert deps.state.flags()[0]["reason"] == "worktree failed"


# -------------------------------------------------------------------------------------- conflict
def test_conflict_happy_path_merges_base_and_pushes(make_deps: Callable[..., Deps], world: World) -> None:
    worker = FakeWorker(conflict_output(), script=merge_main("x = 5\n"))
    deps = make_deps(make_pr(world, mergeable="CONFLICTING"), worker=worker)

    outcome = run(CONFLICT, conflict_action(world), deps)

    assert outcome.status == "done" and outcome.pushed is True and outcome.record_handled
    effects: FakeEffects = deps.effects  # type: ignore[assignment]
    assert len(effects.pushes) == 1 and effects.pushes[0][1:4] == ("feat", "conflict", PR_NUMBER)
    merge_sha = effects.pushes[0][4]
    parents = run_git(["rev-list", "--parents", "-n", "1", merge_sha], world.clone).split()[1:]
    assert sorted(parents) == sorted([world.head, world.base])  # a merge of origin/main, not a rebase
    assert not worker.cwd.exists()
    prompt = worker.prompt
    assert call_prompt_sections(prompt) == ["## Action", "## Pull request", "## Verification commands"]
    assert "Base branch: `main`" in prompt and "git merge origin/main" in prompt and "never a rebase" in prompt
    assert "mergeable: CONFLICTING" in prompt and "baseRefName: `main`" in prompt
    assert worker.calls[0]["kind"] == "conflict" and worker.calls[0]["schema"] is ConflictOutput


def test_conflict_unresolved_keeps_the_worktree_with_markers_and_flags(make_deps: Callable[..., Deps], world: World) -> None:
    output = conflict_output(
        status="blocked", push_ok=False, all_resolved=False, verification=[],
        push_reasons=["src/app.py: both sides changed x"],
        summary="Left src/app.py for a human.",
        hunks=[HunkResolution(file="src/app.py", classification="unsafe", resolution="both sides changed the value of x")],
    )  # fmt: skip
    worker = FakeWorker(output, script=merge_main(None))
    deps = make_deps(make_pr(world, mergeable="CONFLICTING"), worker=worker)

    outcome = run(CONFLICT, conflict_action(world), deps)

    assert outcome.status == "kept" and outcome.pushed is False and not outcome.record_handled
    assert "not all conflicts resolved" in outcome.note and "unsafe hunks: src/app.py" in outcome.note
    assert deps.effects.pushes == []  # type: ignore[attr-defined]
    assert worker.cwd.exists()
    wt = Worktree(path=worker.cwd, repo=SLUG, clone=world.clone, base_sha=world.head)
    assert deps.wt_manager.has_conflict_markers(wt) and deps.wt_manager.new_commits(wt) == []  # the partial merge survives
    kept = deps.state.kept_worktree(TARGET)
    assert kept is not None and kept["path"] == str(worker.cwd) and kept["kind"] == "conflict"
    assert "not all conflicts resolved" in kept["reason"] and "both sides changed x" in kept["reason"]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "conflict needs a human"
    assert "both sides changed the value of x" in flags[0]["detail"] and str(worker.cwd) in flags[0]["detail"]
    # the worktree is a detached HEAD: the hint names the PR head branch to continue on
    assert (
        "detached HEAD: `git switch -c feat-factory` to continue by hand, "
        "or review and `git push origin HEAD:refs/heads/feat` yourself"
    ) in flags[0]["detail"]


def test_conflict_markers_in_a_commit_refuse_the_push(make_deps: Callable[..., Deps], world: World) -> None:
    markers = "<<<<<<< HEAD\nx = 2\n=======\nx = 3\n>>>>>>> origin/main\n"
    worker = FakeWorker(conflict_output(), script=commit_files({"src/app.py": markers}, "merge"))
    deps = make_deps(make_pr(world, mergeable="CONFLICTING"), worker=worker)

    outcome = run(CONFLICT, conflict_action(world), deps)

    assert outcome.status == "kept" and "conflict markers remain" in outcome.note
    assert deps.effects.pushes == [] and worker.cwd.exists()  # type: ignore[attr-defined]
    assert deps.state.flags()[0]["reason"] == "push refused" and deps.state.kept_worktree(TARGET) is not None


def test_conflict_unsafe_hunk_refuses_even_when_all_resolved(make_deps: Callable[..., Deps], world: World) -> None:
    output = conflict_output(hunks=[HunkResolution(file="src/app.py", classification="unsafe", resolution="guessed")])
    worker = FakeWorker(output, script=merge_main("x = 5\n"))
    deps = make_deps(make_pr(world, mergeable="CONFLICTING"), worker=worker)
    outcome = run(CONFLICT, conflict_action(world), deps)
    assert outcome.status == "kept" and "unsafe hunks: src/app.py" in outcome.note and deps.effects.pushes == []  # type: ignore[attr-defined]


def test_conflict_nothing_when_the_branch_merges_cleanly(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    worker = FakeWorker(conflict_output(status="nothing", push_ok=False, all_resolved=False, hunks=[], verification=[], summary="merges cleanly"))
    deps = make_deps(make_pr(world, mergeable="CONFLICTING"), worker=worker)
    outcome = run(CONFLICT, conflict_action(world), deps)
    assert outcome.status == "nothing" and outcome.record_handled and outcome.note == "merges cleanly"
    assert not worker.cwd.exists() and worktree_dirs(paths) == [] and deps.state.flags() == []


# -------------------------------------------------------------------------------------------- rc
def test_rc_happy_path_renders_selected_feedback_and_pushes(make_deps: Callable[..., Deps], world: World) -> None:
    items, threads = feedback_items()
    worker = FakeWorker(rc_output(), script=commit_files({"src/feature.py": "def plus(a, b):\n    return a + b\n"}, "rename add to plus"))
    deps = make_deps(make_pr(world, feedback=items, threads=threads), worker=worker)

    outcome = run(RC, rc_action(world), deps)

    assert outcome.status == "done" and outcome.pushed is True and outcome.record_handled
    effects: FakeEffects = deps.effects  # type: ignore[assignment]
    assert [(p[1], p[2], p[3]) for p in effects.pushes] == [("feat", "rc", PR_NUMBER)]
    assert not worker.cwd.exists()
    prompt = worker.prompt
    assert call_prompt_sections(prompt) == ["## Action", "## Pull request", "## Review feedback", "## Verification commands"]
    assert "### c1 by alice" in prompt and "Please rename add to plus" in prompt
    assert "### c2 by cursor[bot] (bot)" in prompt and "location: `src/feature.py` line 2" in prompt and "thread: t1 (unresolved)" in prompt
    assert "### c3" not in prompt  # not in feedback_ids
    assert worker.calls[0]["kind"] == "rc" and worker.calls[0]["schema"] is RcOutput
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "review feedback dispositions"
    assert "c1 (alice): addressed - renamed" in flags[0]["detail"] and "c2 (cursor[bot]): disagree" in flags[0]["detail"]
    assert notifies(deps)[-1]["level"] == "info"


def test_rc_falls_back_to_open_feedback_when_no_id_matches(make_deps: Callable[..., Deps], world: World) -> None:
    items, threads = feedback_items()
    worker = FakeWorker(rc_output(status="nothing", push_ok=False, comments=[]))
    deps = make_deps(make_pr(world, feedback=items, threads=threads), worker=worker)
    run(RC, rc_action(world, feedback_ids=["gone"]), deps)
    assert "### c3 by bob" in worker.prompt and "### c1 by alice" in worker.prompt
    deps = make_deps(make_pr(world, feedback=items, threads=threads), worker=FakeWorker(rc_output(status="nothing", push_ok=False, comments=[])))
    run(RC, rc_action(world, feedback_ids=[]), deps)  # ``factory run rc`` without scan context
    assert "### c3 by bob" in deps.run_worker.calls[0]["prompt"]  # type: ignore[attr-defined]


def test_rc_fallback_without_matching_ids_leaves_out_comments_on_resolved_threads(make_deps: Callable[..., Deps], world: World) -> None:
    """``factory run rc`` (no scan context) works on ``open_feedback``: a comment on a thread the
    reviewer resolved stays in ``pr.feedback`` (version key only) but must not be handed to the
    worker as work to do."""
    items, threads = feedback_items()
    resolved = Feedback(id="c4", author="carol", body="resolved nit", created_at="2026-09-30T09:45:00Z", kind="thread",
                        path="src/feature.py", line=1, thread_id="t2")  # fmt: skip
    threads = [*threads, ReviewThread(id="t2", is_resolved=True, is_outdated=False, path="src/feature.py", line=1, comments=[resolved])]
    pr = make_pr(world, feedback=[*items, resolved], threads=threads)
    assert [f.id for f in select_feedback(rc_action(world, feedback_ids=[]), pr)] == ["c1", "c2", "c3"]
    assert [f.id for f in select_feedback(rc_action(world, feedback_ids=["c4"]), pr)] == ["c4"]  # named explicitly: kept
    worker = FakeWorker(rc_output(status="nothing", push_ok=False, comments=[]))
    deps = make_deps(pr, worker=worker)
    run(RC, rc_action(world, feedback_ids=["gone"]), deps)
    assert "### c1 by alice" in worker.prompt and "resolved nit" not in worker.prompt and "### c4 by carol" not in worker.prompt


def test_rc_without_open_feedback_is_nothing_before_any_worktree(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    """Regression: a manual ``factory run rc`` on a PR whose only feedback sits on resolved threads
    selected no item, then still created a worktree and started a session with an empty
    ``## Review feedback``. No open feedback is ``nothing``: recorded, no worktree, no worker."""
    resolved = Feedback(id="c4", author="carol", body="resolved nit", created_at=FEEDBACK_AT, kind="thread",
                        path="src/feature.py", line=1, thread_id="t2")  # fmt: skip
    pr = make_pr(world, feedback=[resolved], threads=[ReviewThread(id="t2", is_resolved=True, is_outdated=False, comments=[resolved])])
    assert pr.open_feedback == [] and select_feedback(rc_action(world, feedback_ids=[]), pr) == []
    worker = FakeWorker(rc_output())
    deps = make_deps(pr, worker=worker, effects=NoPushEffects(paths))

    outcome = run(RC, rc_action(world, feedback_ids=[]), deps)

    assert outcome.status == "nothing" and outcome.record_handled and outcome.note == "no open feedback"
    assert outcome.pr_url == PR_URL
    assert worker.calls == [] and worktree_dirs(paths) == [] and deps.state.flags() == []
    # a stale scan id that no longer matches anything falls back to the (empty) open feedback too
    assert run(RC, rc_action(world, feedback_ids=["gone"]), deps).status == "nothing" and worker.calls == []


def test_rc_skipped_comment_keeps_the_worktree_and_never_pushes(make_deps: Callable[..., Deps], world: World) -> None:
    output = rc_output(
        push_ok=False, push_reasons=["c2 skipped"],
        comments=[CommentDisposition(comment_id="c1", author="alice", disposition="addressed", note="renamed"),
                  CommentDisposition(comment_id="c2", author="cursor[bot]", disposition="skipped", note="changes the public API")],
    )  # fmt: skip
    items, threads = feedback_items()
    worker = FakeWorker(output, script=commit_files({"src/feature.py": "def plus(a, b):\n    return a + b\n"}))
    deps = make_deps(make_pr(world, feedback=items, threads=threads), worker=worker)

    outcome = run(RC, rc_action(world), deps)

    assert outcome.status == "kept" and outcome.pushed is False and not outcome.record_handled
    assert "comments skipped: c2" in outcome.note and str(worker.cwd) in outcome.note
    assert deps.effects.pushes == [] and worker.cwd.exists()  # type: ignore[attr-defined]
    kept = deps.state.kept_worktree(TARGET)
    assert kept is not None and kept["kind"] == "rc" and "changes the public API" in kept["reason"]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "review feedback needs a human"
    assert "c2 (cursor[bot]): changes the public API" in flags[0]["detail"]
    assert (
        "detached HEAD: `git switch -c feat-factory` to continue by hand, "
        "or review and `git push origin HEAD:refs/heads/feat` yourself"
    ) in flags[0]["detail"]
    # a skipped comment with ``push_ok: true`` is still refused: the engine applies its own policy
    worker = FakeWorker(rc_output(comments=output.comments), script=commit_files({"src/feature.py": "ok\n"}))
    deps = make_deps(make_pr(world, feedback=items, threads=threads), worker=worker)
    assert run(RC, rc_action(world), deps).status == "kept" and deps.effects.pushes == []  # type: ignore[attr-defined]


def test_rc_nothing_is_recorded_and_the_dispositions_are_notified(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    output = rc_output(
        status="nothing", push_ok=False, verification=[], summary="Already handled on this head.",
        comments=[CommentDisposition(comment_id="c1", author="alice", disposition="already_done", note="renamed earlier"),
                  CommentDisposition(comment_id="c2", author="cursor[bot]", disposition="not_actionable")],
    )  # fmt: skip
    items, threads = feedback_items()
    worker = FakeWorker(output)
    deps = make_deps(make_pr(world, feedback=items, threads=threads), worker=worker)

    outcome = run(RC, rc_action(world), deps)

    assert outcome.status == "nothing" and outcome.record_handled and outcome.pushed is False
    assert outcome.note == "Already handled on this head." and outcome.cost_usd == 0.25
    assert deps.effects.pushes == [] and not worker.cwd.exists() and worktree_dirs(paths) == []  # type: ignore[attr-defined]
    assert deps.state.kept_worktree(TARGET) is None
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "review feedback dispositions"
    assert "c1 (alice): already_done - renamed earlier" in flags[0]["detail"] and "c2 (cursor[bot]): not_actionable" in flags[0]["detail"]
    assert flags[0]["dedupe_key"] == f"rc-dispositions:{TARGET}:{FEEDBACK_AT}" and PR_URL in flags[0]["detail"]


def test_rc_policy_refusal_without_skips_is_kept(make_deps: Callable[..., Deps], world: World) -> None:
    items, threads = feedback_items()
    worker = FakeWorker(rc_output(head_verified=False), script=commit_files({"src/feature.py": "ok\n"}))
    deps = make_deps(make_pr(world, feedback=items, threads=threads), worker=worker)
    outcome = run(RC, rc_action(world), deps)
    assert outcome.status == "kept" and "did not verify the worktree HEAD" in outcome.note
    assert [f["reason"] for f in deps.state.flags()] == ["push refused"]  # no dispositions notice on a refusal


# -------------------------------------------------------------------------------------------- rv
def test_rv_writes_the_review_locally_notifies_and_never_pushes(
    make_deps: Callable[..., Deps], world: World, paths: ProfilePaths
) -> None:
    def raising_git(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
        raise AssertionError(f"git invoked through effects: {args}")

    worker = FakeWorker(rv_output())
    deps = make_deps(make_pr(world, author="alice"), worker=worker, mode="observe")
    deps.effects = RecordingEffects(deps.profile, paths, deps.state, None, None, deps.audit, raising_git)

    outcome = run(RV, rv_action(world), deps)

    path = paths.review_file("repo", PR_NUMBER)
    assert outcome.status == "done" and outcome.pushed is False and outcome.record_handled
    assert str(path) in outcome.note and "approve" in outcome.note and "1 finding(s)" in outcome.note
    assert path.is_file() and path.read_text(encoding="utf-8") == rv_output().review_markdown
    assert not worker.cwd.exists() and worktree_dirs(paths) == []
    assert deps.state.kept_worktree(TARGET) is None and deps.state.pushes() == []
    events = [r["event"] for r in deps.audit.read()]
    assert "effect.write_review" in events and not any(e in ("would_push", "effect.push") for e in events)
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == f"review draft ready: {path}"
    assert "verdict: approve (1 minor)" in flags[0]["detail"] and PR_URL in flags[0]["detail"]
    assert flags[0]["dedupe_key"] == f"rv:{TARGET}:{world.head}"
    assert notifies(deps)[-1]["title"].startswith("review draft ready: ")
    call = worker.calls[0]
    assert call["kind"] == "rv" and call["schema"] is RvOutput and call["policy"].read_only is True
    assert not call["policy"].check("Edit", {"file_path": str(worker.cwd / "src" / "app.py")}).allow
    assert call["policy"].check("Read", {"file_path": str(worker.cwd / "src" / "app.py")}).allow
    assert not call["policy"].check("Bash", {"command": "git push origin HEAD:refs/heads/feat"}).allow
    prompt = worker.prompt
    assert call_prompt_sections(prompt) == ["## Action", "## Pull request", "## Verification commands"]
    assert "- author: alice" in prompt and f"headRefOid: `{world.head}`" in prompt and "read-only" in prompt


def test_rv_with_fake_effects_never_calls_push(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    worker = FakeWorker(rv_output(verdict="request_changes", findings=[ReviewFinding(severity="blocker", claim="breaks", evidence="ran tests")]))
    effects = NoPushEffects(paths)
    deps = make_deps(make_pr(world), worker=worker, effects=effects)
    outcome = run(RV, rv_action(world), deps)
    assert outcome.status == "done" and effects.reviews == [paths.review_file("repo", PR_NUMBER)]
    assert "verdict: request_changes (1 blocker)" in deps.state.flags()[0]["detail"]


def test_rv_nothing_and_blocked(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    worker = FakeWorker(rv_output(status="nothing", summary="PR already merged elsewhere", review_markdown=""))
    deps = make_deps(make_pr(world), worker=worker, effects=NoPushEffects(paths))
    outcome = run(RV, rv_action(world), deps)
    assert outcome.status == "nothing" and outcome.record_handled and outcome.note == "PR already merged elsewhere"
    assert not worker.cwd.exists() and not paths.review_file("repo", PR_NUMBER).exists() and deps.state.flags() == []

    worker = FakeWorker(rv_output(status="blocked", head_verified=False, summary="HEAD did not match", review_markdown=""))
    deps = make_deps(make_pr(world), worker=worker, effects=NoPushEffects(paths))
    outcome = run(RV, rv_action(world), deps)
    assert outcome.status == "failed" and "HEAD did not match" in outcome.note and not outcome.record_handled
    assert not worker.cwd.exists() and not paths.review_file("repo", PR_NUMBER).exists()
    assert deps.state.flags()[0]["reason"] == "review draft blocked" and "head_verified: False" in deps.state.flags()[0]["detail"]


# ------------------------------------------------------------------------------- worker failures
@pytest.mark.parametrize(
    ("handler", "make_action"),
    [(CI, ci_action), (CONFLICT, conflict_action), (RC, rc_action), (RV, rv_action)],
    ids=["ci", "conflict", "rc", "rv"],
)
def test_auth_failure_propagates_from_an_auth_failed_worker_run(
    make_deps: Callable[..., Deps], world: World, paths: ProfilePaths, handler: Any, make_action: Callable[[World], Action]
) -> None:
    items, threads = feedback_items()
    worker = FakeWorker(None, error="authentication failed", auth_failed=True)
    deps = make_deps(make_pr(world, checks=[failing_check()], mergeable="CONFLICTING", feedback=items, threads=threads), worker=worker)
    with pytest.raises(AuthFailure, match="authentication failed"):
        run(handler, make_action(world), deps)
    assert len(worker.calls) == 1 and deps.effects.pushes == []  # type: ignore[attr-defined]
    assert deps.state.flags() == [] and deps.state.kept_worktree(TARGET) is None  # no attempt, no flag: the wave stops
    assert not worker.cwd.exists() and worktree_dirs(paths) == []  # and no checkout leaks per tick


@pytest.mark.parametrize(
    ("handler", "make_action"),
    [(CI, ci_action), (CONFLICT, conflict_action), (RC, rc_action), (RV, rv_action)],
    ids=["ci", "conflict", "rc", "rv"],
)
def test_provider_unavailable_propagates_from_a_rate_limited_worker_run(
    make_deps: Callable[..., Deps], world: World, paths: ProfilePaths, handler: Any, make_action: Callable[[World], Action]
) -> None:
    items, threads = feedback_items()
    worker = FakeWorker(None, error="rate limited: usage limit reached", rate_limited=True)
    deps = make_deps(make_pr(world, checks=[failing_check()], mergeable="CONFLICTING", feedback=items, threads=threads), worker=worker)
    with pytest.raises(ProviderUnavailable, match="usage limit reached"):
        run(handler, make_action(world), deps)
    assert len(worker.calls) == 1 and deps.effects.pushes == []  # type: ignore[attr-defined]
    assert deps.state.flags() == [] and deps.state.kept_worktree(TARGET) is None  # the dispatcher flags once a day
    assert not worker.cwd.exists() and worktree_dirs(paths) == []


def test_ci_failed_log_auth_error_does_not_leak_the_worktree(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    class AuthLogGh(FakeGitHub):
        def failed_log(self, repo: str, run_id: int, max_bytes: int = 60_000) -> str:
            raise GhAuthError("gh run view failed (1): HTTP 401")

    pr = make_pr(world, checks=[failing_check()])
    worker = FakeWorker(ci_output())
    deps = make_deps(pr, worker=worker, gh=AuthLogGh(pr))
    with pytest.raises(GhAuthError):
        run(CI, ci_action(world), deps)
    assert worker.calls == [] and worktree_dirs(paths) == [] and deps.state.flags() == []


@pytest.mark.parametrize(
    ("label", "failures"),
    [
        ("owned but not fixed", [CiFailure(check="test (3.13)", classification="caused_by_pr", cause="the PR broke add()", fixed=False)]),
        ("unknown", [CiFailure(check="test (3.13)", classification="unknown", cause="could not tell")]),
        ("no classification at all", []),
    ],
)
def test_ci_done_without_commits_for_an_owned_failure_is_failed_and_flagged(
    make_deps: Callable[..., Deps], world: World, paths: ProfilePaths, label: str, failures: list[CiFailure]
) -> None:
    """Only the all-flaky/inherited/infra case may end ``done`` without a commit; anything else is a
    worker that fixed nothing on a red head, and recording that head handled would hide it."""
    worker = FakeWorker(ci_output(failures=failures, push_ok=False, push_reasons=["nothing committed"]))
    deps = make_deps(make_pr(world, checks=[failing_check()]), worker=worker)

    outcome = run(CI, ci_action(world), deps)

    assert outcome.status == "failed" and not outcome.record_handled and "no commits" in outcome.note, label
    assert not worker.cwd.exists() and worktree_dirs(paths) == [] and deps.effects.pushes == []  # type: ignore[attr-defined]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "no commits" and flags[0]["dedupe_key"] == f"no-commits:{TARGET}:ci:{world.head}"
    assert "reported done but committed nothing on feat" in flags[0]["detail"]
    for failure in failures:
        assert f"- {failure.check}: {failure.classification} - {failure.cause}" in flags[0]["detail"]
    assert notifies(deps)[0]["title"] == "no commits"


def test_conflict_done_without_a_merge_commit_is_failed_and_flagged(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    worker = FakeWorker(conflict_output())  # claims all_resolved, commits nothing
    deps = make_deps(make_pr(world, mergeable="CONFLICTING"), worker=worker)
    outcome = run(CONFLICT, conflict_action(world), deps)
    assert outcome.status == "failed" and not outcome.record_handled and "no commits" in outcome.note
    assert not worker.cwd.exists() and worktree_dirs(paths) == [] and deps.effects.pushes == []  # type: ignore[attr-defined]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "no commits" and "- src/app.py: trivial - kept the PR's value" in flags[0]["detail"]


def test_conflict_merge_of_a_base_that_touched_sensitive_files_still_pushes(
    make_deps: Callable[..., Deps], world: World, paths: ProfilePaths
) -> None:
    """main also changed a sensitive file after the PR branched: the merge brings it in, the worker
    never touched it, and the push must not be refused for it (K3)."""
    (world.clone / "src" / "auth").mkdir(parents=True)
    (world.clone / "src" / "auth" / "login.py").write_text("secret = 1\n", encoding="utf-8")
    run_git(["add", "-A"], world.clone)
    run_git(["commit", "-q", "-m", "auth on main"], world.clone)
    run_git(["push", "-q", "origin", "main"], world.clone)
    base = run_git(["rev-parse", "HEAD"], world.clone)
    worker = FakeWorker(conflict_output(), script=merge_main("x = 5\n"))
    deps = make_deps(make_pr(world, mergeable="CONFLICTING", base_ref_oid=base), worker=worker)
    deps.state.register_pr(SLUG, PR_NUMBER, "feat", "build", "ABC-1")

    outcome = run(CONFLICT, conflict_action(world, version=f"{world.head}:{base}"), deps)

    assert outcome.status == "done" and outcome.pushed, outcome.note
    effects: FakeEffects = deps.effects  # type: ignore[assignment]
    assert len(effects.pushes) == 1 and effects.pushes[0][1:4] == ("feat", "conflict", PR_NUMBER)
    section = effects.evidence[0][2]
    assert "`src/app.py`" in section and "login.py" not in section and "Files touched: 1" in section
    assert deps.state.kept_worktree(TARGET) is None and deps.state.flags() == []

    # the worker's own edit of the sensitive file inside the merge commit is still refused
    def merge_and_touch_auth(cwd: Path) -> None:
        merge_main(None)(cwd)  # markers left, nothing committed
        (cwd / "src" / "app.py").write_text("x = 5\n", encoding="utf-8")
        (cwd / "src" / "auth" / "login.py").write_text("secret = 2\n", encoding="utf-8")
        run_git(["add", "-A"], cwd)
        run_git(["commit", "-q", "-m", "merge origin/main into feat"], cwd)

    worker = FakeWorker(conflict_output(), script=merge_and_touch_auth)
    deps = make_deps(make_pr(world, mergeable="CONFLICTING", base_ref_oid=base), worker=worker, state=State(paths.root / "other.db"))
    outcome = run(CONFLICT, conflict_action(world, version=f"{world.head}:{base}"), deps)
    assert outcome.status == "kept" and "sensitive globs: src/auth/login.py" in outcome.note
    assert deps.effects.pushes == []  # type: ignore[attr-defined]


def test_worker_without_usable_output_is_failed_and_flagged(make_deps: Callable[..., Deps], world: World, paths: ProfilePaths) -> None:
    worker = FakeWorker(None, error="worker timed out after 300s", timed_out=True)
    deps = make_deps(make_pr(world, checks=[failing_check()]), worker=worker)
    outcome = run(CI, ci_action(world), deps)
    assert outcome.status == "failed" and "timed out" in outcome.note and outcome.cost_usd == 0.25
    assert not worker.cwd.exists() and deps.state.flags()[0]["reason"] == "worker timed out"

    worker = FakeWorker({"status": "done", "summary": "no head_verified field"})  # does not validate as CiOutput
    deps = make_deps(make_pr(world, checks=[failing_check()]), worker=worker)
    outcome = run(CI, ci_action(world), deps)
    assert outcome.status == "failed" and "does not match CiOutput" in outcome.note and not worker.cwd.exists()

    worker = FakeWorker(None, error="bad json")
    deps = make_deps(make_pr(world), worker=worker, effects=NoPushEffects(paths))
    outcome = run(RV, rv_action(world), deps)
    assert outcome.status == "failed" and "bad json" in outcome.note and not worker.cwd.exists()
    assert worktree_dirs(paths) == []


# ------------------------------------------------------------------------------------ _pr helpers
def test_version_helpers(world: World) -> None:
    items, _threads = feedback_items()
    pr = make_pr(world, feedback=items)
    assert pr_mod.current_version(Kind.CI, pr) == world.head and pr_mod.current_version(Kind.RV, pr) == world.head
    assert pr_mod.current_version(Kind.CONFLICT, pr) == f"{world.head}:{world.base}"
    assert pr_mod.current_version(Kind.RC, pr) == FEEDBACK_AT and pr_mod.latest_feedback_at(make_pr(world)) == ""
    assert pr_mod.versions_match(Kind.RC, "2026-09-30T12:00:00+02:00", FEEDBACK_AT)
    assert not pr_mod.versions_match(Kind.RC, "2026-09-30T10:00:01Z", FEEDBACK_AT)
    assert pr_mod.versions_match(Kind.RC, "garbage", "garbage") and not pr_mod.versions_match(Kind.CI, "a", "b")
    assert pr_mod.PUSH_KINDS == (Kind.CI, Kind.CONFLICT, Kind.RC)
