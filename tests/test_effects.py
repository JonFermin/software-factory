"""Effects tests: the only module that performs outward writes.

Live pushes run against a real local bare remote (no network); GitHub and the tracker are fakes.
``RecordingEffects`` is proven write-free with fakes that raise on any write.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from software_factory import providers as providers_mod
from software_factory.config import Profile, RepoOverride
from software_factory.effects import (
    REPO_MARKER,
    EffectError,
    EffectRefused,
    GitCommandError,
    LiveEffects,
    PushRejected,
    RecordingEffects,
    default_git_runner,
    is_auth_failure,
    is_recording,
    link_pushes,
    make_effects,
    push_refspec,
    validate_asset_path,
    validate_branch,
)
from software_factory.evidence import END, EVIDENCE_BRANCH, START, EvidenceData, evidence_url, render_evidence
from software_factory.notify import Audit
from software_factory.paths import ProfilePaths
from software_factory.providers import unplantable_hooks_path
from software_factory.repos import RepoInfo
from software_factory.sources.models import PullRequest
from software_factory.state import State
from software_factory.trackers.base import Issue, TrackerError
from software_factory.workers.schemas import SubIssueSpec
from software_factory.worktree import Worktree, WorktreeManager

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

SLUG = "JaneDoe/repo"
FORBIDDEN_PUSH_ARGS = {"--force", "-f", "--force-with-lease", "--no-verify", "--mirror", "--delete"}


def is_push(args: list[str]) -> bool:
    """``git [-c ...] push origin <refspec>``: the hooks override precedes the subcommand."""
    args = strip_hooks(args)
    return args[:2] == ["push", "origin"]


def hooks_prefix() -> list[str]:
    return ["-c", f"core.hooksPath={unplantable_hooks_path()}"]


def strip_hooks(args: list[str]) -> list[str]:
    """Drop the ``-c core.hooksPath=<unplantable>`` prefix every engine git in effects carries."""
    return list(args[2:]) if list(args[:2]) == hooks_prefix() else list(args)


def plant_hook(clone: Path, name: str, marker: Path) -> Path:
    """What a worker's shell can do from a linked worktree: write the clone's shared hooks dir."""
    hooks = clone / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    hook = hooks / name
    hook.write_text(f'#!/bin/sh\necho {name} >> "{marker.as_posix()}"\nexit 1\n', encoding="utf-8")
    hook.chmod(0o755)
    return hook


# ------------------------------------------------------------------------------------- fakes
class FakeGh:
    """Reads answer from memory; writes are recorded (or raise when ``raise_on_write``)."""

    def __init__(self, body: str = "", *, raise_on_write: bool = False) -> None:
        self.body = body
        self.raise_on_write = raise_on_write
        self.edits: list[tuple[str, int, str]] = []
        self.created: list[dict[str, Any]] = []

    def pr_body(self, repo: str, number: int) -> str:
        return self.body

    def edit_pr_body(self, repo: str, number: int, body: str) -> None:
        if self.raise_on_write:
            raise AssertionError("gh write attempted: edit_pr_body")
        self.edits.append((repo, number, body))
        self.body = body

    def create_draft_pr(self, repo: str, head: str, base: str, title: str, body: str) -> PullRequest:
        if self.raise_on_write:
            raise AssertionError("gh write attempted: create_draft_pr")
        self.created.append({"repo": repo, "head": head, "base": base, "title": title, "body": body})
        return PullRequest(
            repo=repo, number=42, url=f"https://github.com/{repo}/pull/42", title=title, body=body,
            is_draft=True, head_ref_name=head, base_ref_name=base,
        )  # fmt: skip


class FakeTracker:
    name = "fake"

    def __init__(self, *, existing_labels: set[str] | None = None, raise_on_write: bool = False) -> None:
        self.existing_labels = {lbl.lower() for lbl in (existing_labels or set())}
        self.raise_on_write = raise_on_write
        self.calls: list[tuple[str, Any]] = []
        self._counter = 0

    def _write(self, name: str, *args: Any) -> None:
        if self.raise_on_write:
            raise AssertionError(f"tracker write attempted: {name}")
        self.calls.append((name, args))

    def labels_exist(self, names: list[str]) -> dict[str, bool]:
        return {n: n.lower() in self.existing_labels for n in names}

    def create_sub_issue(self, parent: Issue, title: str, description: str, labels: list[str]) -> Issue:
        self._write("create_sub_issue", parent.key, title, description, list(labels))
        self._counter += 1
        return Issue(
            id=f"child-{self._counter}", key=f"{parent.key.split('-')[0]}-{100 + self._counter}", title=title,
            description=description, labels=list(labels), team_id=parent.team_id,
        )  # fmt: skip

    def add_blocked_by(self, issue_id: str, blocker_id: str) -> None:
        self._write("add_blocked_by", issue_id, blocker_id)

    def move_to_state_type(self, issue_id: str, state_type: str) -> None:
        self._write("move_to_state_type", issue_id, state_type)

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None:
        self._write("update_issue_text", issue_id, title, description)

    def remove_label(self, issue_id: str, label: str) -> None:
        self._write("remove_label", issue_id, label)


def raising_git(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
    raise AssertionError(f"git invoked in recording mode: {args}")


# ---------------------------------------------------------------------------------- fixtures
@pytest.fixture
def paths(tmp_path: Path) -> ProfilePaths:
    return ProfilePaths.for_profile("personal", root=tmp_path / "data root").ensure()


@pytest.fixture
def audit(paths: ProfilePaths) -> Audit:
    return Audit(paths.audit_log, "personal")


@pytest.fixture
def world(bare_repo_factory: Callable[..., Path], git_repo_factory: Callable[..., Path], paths: ProfilePaths) -> dict:
    bare = bare_repo_factory("origin")
    clone = git_repo_factory("repo", origin=bare)
    repo = RepoInfo(slug=SLUG, name="repo", dirname="repo", path=clone, default_branch="main", override=RepoOverride())
    return {"bare": bare, "clone": clone, "repo": repo, "manager": WorktreeManager(paths.worktrees)}


def live(profile: Profile, paths: ProfilePaths, state: State, audit: Audit, **kw: Any) -> LiveEffects:
    kw.setdefault("gh", None)
    kw.setdefault("tracker", None)
    return LiveEffects(profile, paths, state, kw["gh"], kw["tracker"], audit, kw.get("git_runner"))


def commit_file(git: Callable[..., str], cwd: Path, name: str, content: str, message: str) -> str:
    target = cwd / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(["add", "-A"], cwd)
    git(["commit", "-q", "-m", message], cwd)
    return git(["rev-parse", "HEAD"], cwd)


def events(audit: Audit) -> list[str]:
    return [r["event"] for r in audit.read()]


# --------------------------------------------------------------------------------- refspecs
def test_push_refspec_is_the_one_safe_shape() -> None:
    assert push_refspec("feature/x-1") == "HEAD:refs/heads/feature/x-1"
    assert validate_branch(" factory/ABC-12 ") == "factory/ABC-12"


@pytest.mark.parametrize(
    "bad",
    ["", "+main", "-x", "--force", "refs/heads/x", "a b", "a:b", "a..b", "x~1", "x^", "x?", "x*", "x[1]",
     "feature/", "/feature", "a//b", "x.lock", ".hidden", "a/.b", "x@{1}", "back\\slash"],
)  # fmt: skip
def test_push_refspec_rejects_unsafe_branch_names(bad: str) -> None:
    with pytest.raises(ValueError):
        push_refspec(bad)


@pytest.mark.parametrize("bad", ["", "/abs/a.png", "../x.png", "a/../b.png", "C:/x.png", ".", "a/.."])
def test_validate_asset_path_rejects_escapes(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_asset_path(bad)


def test_validate_asset_path_normalises() -> None:
    assert validate_asset_path("repo\\pr-1\\shot.png") == "repo/pr-1/shot.png"
    assert validate_asset_path("./repo/pr-1/./shot.png") == "repo/pr-1/shot.png"


# -------------------------------------------------------------------------------- live push
def test_live_push_uses_exact_safe_refspec_and_records_shas(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str]
) -> None:
    calls: list[list[str]] = []

    def spy(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
        calls.append(list(args))
        return default_git_runner(args, cwd=cwd, env=env)

    effects = live(profile, paths, state, audit, git_runner=spy)
    wt = world["manager"].for_new_branch(world["repo"], "factory/abc-1")
    head = commit_file(git, wt.path, "new.txt", "hi\n", "add new")

    after = effects.push(wt, "factory/abc-1", kind="build", pr=None)

    assert after == head
    push_calls = [c for c in calls if is_push(c)]
    assert len(push_calls) == 1 and push_calls[0][-3:] == ["push", "origin", "HEAD:refs/heads/factory/abc-1"]
    assert push_calls[0][:2] == hooks_prefix()
    assert all(call[:2] == hooks_prefix() for call in calls), calls  # every engine git, not only the push
    for call in calls:
        assert not (set(call) & FORBIDDEN_PUSH_ARGS), call
        assert not any(arg.startswith("+") for arg in call), call
    assert git(["rev-parse", "refs/heads/factory/abc-1"], world["bare"]) == head
    pushes = state.pushes(SLUG)
    assert len(pushes) == 1
    assert pushes[0]["before_sha"] == "" and pushes[0]["after_sha"] == head
    assert pushes[0]["commit_shas"] == [head] and pushes[0]["kind"] == "build" and pushes[0]["number"] is None
    record = next(r for r in audit.read() if r["event"] == "effect.push")
    assert record["refspec"] == "HEAD:refs/heads/factory/abc-1" and record["after_sha"] == head
    assert git(["status", "--porcelain"], world["clone"]) == ""  # the user's checkout is untouched

    # a second push on the same branch records the previous remote sha as before_sha
    head2 = commit_file(git, wt.path, "more.txt", "more\n", "more")
    assert effects.push(wt, "factory/abc-1", kind="ci", pr=9) == head2
    second = state.pushes(SLUG, 9)[0]
    assert second["before_sha"] == head and second["after_sha"] == head2
    assert second["commit_shas"] == [head, head2]  # base_sha..HEAD
    assert state.pushed_shas(SLUG, 9) == {head, head2}


def test_live_push_non_fast_forward_raises_push_rejected_and_records_nothing(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str]
) -> None:
    effects = live(profile, paths, state, audit)
    wt = world["manager"].for_new_branch(world["repo"], "feature/x")
    commit_file(git, wt.path, "a.txt", "a\n", "A")
    effects.push(wt, "feature/x", kind="build", pr=None)

    # Someone else rewrites the remote branch meanwhile (test-only force push from the user's clone).
    commit_file(git, world["clone"], "other.txt", "b\n", "B")
    git(["push", "-q", "--force", "origin", "HEAD:refs/heads/feature/x"], world["clone"])
    remote_before = git(["rev-parse", "refs/heads/feature/x"], world["bare"])

    commit_file(git, wt.path, "c.txt", "c\n", "C")
    with pytest.raises(PushRejected, match="non-fast-forward"):
        effects.push(wt, "feature/x", kind="ci", pr=7)
    assert git(["rev-parse", "refs/heads/feature/x"], world["bare"]) == remote_before  # never forced
    assert state.pushes(SLUG, 7) == []
    assert "effect.push_rejected" in events(audit)


def test_live_push_never_runs_hooks_planted_in_the_shared_git_dir(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str], tmp_path: Path
) -> None:
    """A linked worktree's hooks are the clone's ``.git/hooks``, reachable from a worker's shell;
    a planted ``pre-push`` must never execute inside the engine's authenticated push (it would
    block it here, and could run anything)."""
    marker = tmp_path / "hook marker.txt"
    plant_hook(world["clone"], "pre-push", marker)
    effects = live(profile, paths, state, audit)
    wt = world["manager"].for_new_branch(world["repo"], "factory/abc-1")
    head = commit_file(git, wt.path, "new.txt", "hi\n", "add new")

    assert effects.push(wt, "factory/abc-1", kind="build", pr=None) == head

    assert git(["rev-parse", "refs/heads/factory/abc-1"], world["bare"]) == head
    assert not marker.exists()
    # the evidence push runs from the user's clone itself: same hooks, same override
    asset = tmp_path / "a.png"
    asset.write_text("a", encoding="utf-8")
    effects.push_evidence_assets(world["repo"], [(asset, "repo/pr-1/a.png")])
    assert git(["rev-parse", f"refs/heads/{EVIDENCE_BRANCH}"], world["bare"]) and not marker.exists()
    assert not (paths.root / "no-hooks").exists()  # no hooks dir is made (nothing to plant into)


def test_live_push_fails_closed_when_hooks_cannot_be_disabled(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unplantable path's parent must be a file / character device; when it is a directory (or
    gone) every engine git is refused before it runs: no push, no state, no audit record."""
    effects = live(profile, paths, state, audit)
    wt = world["manager"].for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "new.txt", "hi\n", "add new")
    fake_hosts = tmp_path / "hosts"
    fake_hosts.mkdir()
    monkeypatch.setattr(providers_mod, "_unplantable_parent", lambda: str(fake_hosts))

    with pytest.raises(EffectError, match="refusing to run git with hooks enabled"):
        effects.push(wt, "factory/abc-1", kind="build", pr=None)
    asset = tmp_path / "a.png"
    asset.write_text("a", encoding="utf-8")
    with pytest.raises(EffectError, match="refusing to run git with hooks enabled"):
        effects.push_evidence_assets(world["repo"], [(asset, "repo/pr-1/a.png")])
    assert git(["ls-remote", "--heads", "origin"], world["clone"]).count("refs/heads/") == 1  # main only
    assert state.pushes(SLUG) == [] and "effect.push" not in events(audit)


def test_link_pushes_delegates_to_the_state_layer() -> None:
    class SpyState:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int]] = []

        def link_pushes(self, repo: str, branch: str, number: int) -> int:
            self.calls.append((repo, branch, number))
            return 2

    spy = SpyState()
    assert link_pushes(spy, "o/r", "factory/abc-1", 42) == 2  # type: ignore[arg-type]
    assert spy.calls == [("o/r", "factory/abc-1", 42)]


def test_rev_list_after_a_merge_of_base_records_only_the_workers_commits(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str]
) -> None:
    effects = live(profile, paths, state, audit)
    clone = world["clone"]
    wt = world["manager"].for_new_branch(world["repo"], "feature/x")
    first = commit_file(git, wt.path, "feature.txt", "f\n", "feature")
    effects.push(wt, "feature/x", kind="build", pr=None)
    base_commits = [commit_file(git, clone, f"base{i}.txt", f"{i}\n", f"base {i}") for i in range(3)]
    git(["push", "-q", "origin", "main"], clone)
    git(["fetch", "-q", "origin", "main"], wt.path)
    git(["merge", "-q", "--no-edit", "origin/main"], wt.path)
    merge = git(["rev-parse", "HEAD"], wt.path)

    effects.push(wt, "feature/x", kind="conflict", pr=5)

    row = state.pushes(SLUG, 5)[0]
    assert row["commit_shas"] == [first, merge] and row["after_sha"] == merge
    assert not set(base_commits) & state.pushed_shas(SLUG, 5)


def test_live_push_refuses_bad_branch_before_touching_git(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit
) -> None:
    calls: list[list[str]] = []

    def spy(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
        calls.append(list(args))
        return default_git_runner(args, cwd=cwd, env=env)

    effects = live(profile, paths, state, audit, git_runner=spy)
    wt = world["manager"].for_new_branch(world["repo"], "ok")
    calls.clear()
    with pytest.raises(ValueError):
        effects.push(wt, "+main", kind="ci", pr=1)
    assert calls == []


# ------------------------------------------------------------------------------------ git LFS
class LfsRunner:
    """Fake git for ``LiveEffects.push``: answers the reads, records every call, and plays the
    ``lfs push`` step as installed / not installed / failing."""

    HEAD = "h" * 40

    def __init__(self, changed: list[str], lfs: set[str], *, lfs_error: str | None = None) -> None:
        self.changed = changed
        self.lfs = lfs
        self.lfs_error = lfs_error
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], *, cwd: Any = None, env: Any = None) -> str:
        self.calls.append(list(args))
        sub = strip_hooks(args)
        if sub[:1] == ["ls-remote"]:
            return ""
        if sub[:2] == ["rev-parse", "HEAD"]:
            return self.HEAD + "\n"
        if sub[:1] == ["rev-list"]:
            return self.HEAD + "\n"
        if sub[:1] == ["diff"]:
            return "".join(f"{path}\0" for path in self.changed)
        if sub[:1] == ["check-attr"]:
            paths = sub[sub.index("--") + 1 :]
            return "".join(f"{p}\0filter\0{'lfs' if p in self.lfs else 'unspecified'}\0" for p in paths)
        if sub[:2] == ["lfs", "push"]:
            if self.lfs_error is not None:
                raise GitCommandError(args, 1, self.lfs_error)
            return ""
        if sub[:2] == ["push", "origin"]:
            return ""
        raise AssertionError(f"unexpected git call {args}")

    def subcommands(self) -> list[list[str]]:
        return [strip_hooks(c) for c in self.calls]


def fake_wt(tmp_path: Path) -> Worktree:
    path = tmp_path / "wt"
    path.mkdir(exist_ok=True)
    return Worktree(path=path, repo=SLUG, clone=tmp_path, base_sha="b" * 40, branch="factory/abc-1")


def test_live_push_uploads_lfs_objects_before_the_ref_push(
    profile: Profile, paths: ProfilePaths, state: State, audit: Audit, tmp_path: Path
) -> None:
    """Regression: engine pushes run with hooks disabled, so git-lfs's ``pre-push`` never uploaded
    the objects and a push touching ``*.glb`` (LFS-tracked) would publish pointers whose objects
    exist only locally. ``git lfs push origin <HEAD sha>`` now runs first, hooks disabled, no force."""
    runner = LfsRunner(changed=["assets/ship.glb", "src/main.ts", "docs/a b.md"], lfs={"assets/ship.glb"})
    effects = live(profile, paths, state, audit, git_runner=runner)

    assert effects.push(fake_wt(tmp_path), "factory/abc-1", kind="build", pr=None) == LfsRunner.HEAD

    subs = runner.subcommands()
    lfs_index = subs.index(["lfs", "push", "origin", LfsRunner.HEAD])
    push_index = subs.index(["push", "origin", "HEAD:refs/heads/factory/abc-1"])
    assert lfs_index < push_index
    diff = next(c for c in subs if c[:1] == ["diff"])
    assert diff == ["diff", "--name-only", "-z", "--no-renames", "--diff-filter=d", f"{'b' * 40}..HEAD"]
    check = next(c for c in subs if c[:1] == ["check-attr"])
    assert check == ["check-attr", "-z", "filter", "--", "assets/ship.glb", "src/main.ts", "docs/a b.md"]
    assert all(call[:2] == hooks_prefix() for call in runner.calls), runner.calls
    for call in runner.calls:
        assert not (set(call) & FORBIDDEN_PUSH_ARGS) and "--all" not in call, call
    assert state.pushes(SLUG)[0]["after_sha"] == LfsRunner.HEAD
    record = next(r for r in audit.read() if r["event"] == "effect.push_lfs")
    assert record["lfs_files"] == ["assets/ship.glb"] and record["sha"] == LfsRunner.HEAD


def test_live_push_without_lfs_paths_never_calls_git_lfs(
    profile: Profile, paths: ProfilePaths, state: State, audit: Audit, tmp_path: Path
) -> None:
    runner = LfsRunner(changed=["src/main.ts"], lfs=set(), lfs_error="git: 'lfs' is not a git command.")
    effects = live(profile, paths, state, audit, git_runner=runner)
    assert effects.push(fake_wt(tmp_path), "factory/abc-1", kind="ci", pr=3) == LfsRunner.HEAD
    assert not [c for c in runner.subcommands() if c[:1] == ["lfs"]]
    # many changed paths go to check-attr in chunks (Windows command-line length)
    many = [f"f{i}.txt" for i in range(450)]
    runner = LfsRunner(changed=many, lfs={"f449.txt"})
    live(profile, paths, state, audit, git_runner=runner).push(fake_wt(tmp_path), "factory/abc-1", kind="ci", pr=3)
    checks = [c for c in runner.subcommands() if c[:1] == ["check-attr"]]
    assert len(checks) == 3 and sum(len(c) - 4 for c in checks) == 450
    assert ["lfs", "push", "origin", LfsRunner.HEAD] in runner.subcommands()


@pytest.mark.parametrize(
    "error",
    ["git: 'lfs' is not a git command. See 'git --help'.", "batch response: Repository or object not found"],
    ids=["not-installed", "upload-failed"],
)
def test_live_push_refuses_when_lfs_objects_cannot_be_uploaded(
    profile: Profile, paths: ProfilePaths, state: State, audit: Audit, tmp_path: Path, error: str
) -> None:
    """git-lfs missing or the upload failing: the ref push never happens (pointers are never pushed
    without their objects); ``EffectError`` makes the caller keep the worktree and flag it."""
    runner = LfsRunner(changed=["assets/ship.glb"], lfs={"assets/ship.glb"}, lfs_error=error)
    effects = live(profile, paths, state, audit, git_runner=runner)

    with pytest.raises(EffectError, match="Git LFS") as info:
        effects.push(fake_wt(tmp_path), "factory/abc-1", kind="build", pr=None)

    assert not isinstance(info.value, PushRejected) and error in str(info.value)
    assert not is_auth_failure(str(info.value))  # a refusal to flag, not an auth stop
    assert not [c for c in runner.subcommands() if c[:2] == ["push", "origin"]]
    assert state.pushes(SLUG) == [] and "effect.push" not in events(audit)
    refused = next(r for r in audit.read() if r["event"] == "effect.push_refused_lfs")
    assert refused["lfs_files"] == ["assets/ship.glb"] and error[:20] in refused["error"]


def test_live_push_lfs_detection_with_real_git_attributes(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str]
) -> None:
    """``check-attr -z`` / ``diff -z`` parsing against real git: ``.gitattributes`` marks ``*.glb``
    as LFS; only the real ``lfs push`` is faked (no LFS server here)."""
    clone = world["clone"]
    (clone / ".gitattributes").write_text("*.glb filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    git(["add", "-A"], clone)
    git(["commit", "-q", "-m", "track glb with lfs"], clone)
    git(["push", "-q", "origin", "main"], clone)
    lfs_calls: list[list[str]] = []

    def runner(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
        if strip_hooks(args)[:1] == ["lfs"]:
            lfs_calls.append(strip_hooks(args))
            return ""
        return default_git_runner(args, cwd=cwd, env=env)

    effects = live(profile, paths, state, audit, git_runner=runner)
    wt = world["manager"].for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "notes.txt", "n\n", "plain file")
    effects.push(wt, "factory/abc-1", kind="build", pr=None)
    assert lfs_calls == []

    head = commit_file(git, wt.path, "models/space ship.glb", "glTF-binary\n", "add a model")
    assert effects.push(wt, "factory/abc-1", kind="build", pr=None) == head
    assert lfs_calls == [["lfs", "push", "origin", head]]
    assert git(["rev-parse", "refs/heads/factory/abc-1"], world["bare"]) == head


# ---------------------------------------------------------------------------- evidence branch
def test_push_evidence_assets_orphan_then_existing_branch(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str], tmp_path: Path
) -> None:
    effects = live(profile, paths, state, audit)
    clone, bare, repo = world["clone"], world["bare"], world["repo"]
    head_before = git(["rev-parse", "HEAD"], clone)
    index_before = git(["ls-files", "-s"], clone)
    shot = tmp_path / "shot.png"
    shot.write_text("png-bytes-1", encoding="utf-8")

    urls = effects.push_evidence_assets(repo, [(shot, "repo/pr-1/shot.png")])

    assert urls == [evidence_url(SLUG, "repo/pr-1/shot.png")]
    sha1 = git(["rev-parse", f"refs/heads/{EVIDENCE_BRANCH}"], bare)
    commit = git(["cat-file", "-p", sha1], bare)
    assert "parent " not in commit  # orphan root commit
    assert git(["ls-tree", "-r", "--name-only", sha1], bare).splitlines() == ["repo/pr-1/shot.png"]
    assert git(["show", f"{sha1}:repo/pr-1/shot.png"], bare) == "png-bytes-1"
    # the user's clone: same HEAD, same index, clean tree, no local evidence branch
    assert git(["rev-parse", "HEAD"], clone) == head_before
    assert git(["ls-files", "-s"], clone) == index_before
    assert git(["status", "--porcelain"], clone) == ""
    assert git(["branch", "--list", EVIDENCE_BRANCH], clone) == ""

    gif = tmp_path / "two.gif"
    gif.write_text("gif-bytes", encoding="utf-8")
    urls2 = effects.push_evidence_assets(repo, [(gif, "repo/pr-1/two.gif")])
    assert urls2 == [evidence_url(SLUG, "repo/pr-1/two.gif")]
    sha2 = git(["rev-parse", f"refs/heads/{EVIDENCE_BRANCH}"], bare)
    assert f"parent {sha1}" in git(["cat-file", "-p", sha2], bare)
    assert git(["rev-list", "--count", sha2], bare) == "2"
    assert sorted(git(["ls-tree", "-r", "--name-only", sha2], bare).splitlines()) == [
        "repo/pr-1/shot.png",
        "repo/pr-1/two.gif",
    ]
    assert git(["ls-files", "-s"], clone) == index_before
    records = [r for r in audit.read() if r["event"] == "effect.push_evidence_assets"]
    assert len(records) == 2 and records[1]["parent"] == sha1


def test_push_evidence_assets_retries_once_after_non_fast_forward(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str], tmp_path: Path
) -> None:
    clone, bare, repo = world["clone"], world["bare"], world["repo"]
    pushes = {"n": 0}

    def runner(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
        if is_push(args) and args[-1].endswith(f"refs/heads/{EVIDENCE_BRANCH}"):
            pushes["n"] += 1
            if pushes["n"] == 1:
                # someone created the branch meanwhile: publish the clone's HEAD there, then reject ours
                default_git_runner(["push", "-q", "origin", f"HEAD:refs/heads/{EVIDENCE_BRANCH}"], cwd=clone)
                raise GitCommandError(args, 1, "! [rejected] (fetch first)\nerror: failed to push some refs")
        return default_git_runner(args, cwd=cwd, env=env)

    effects = live(profile, paths, state, audit, git_runner=runner)
    asset = tmp_path / "a.png"
    asset.write_text("a", encoding="utf-8")
    effects.push_evidence_assets(repo, [(asset, "repo/pr-2/a.png")])

    assert pushes["n"] == 2
    sha = git(["rev-parse", f"refs/heads/{EVIDENCE_BRANCH}"], bare)
    assert git(["rev-list", "--count", sha], bare) == "2"  # built on top of the branch that appeared
    names = git(["ls-tree", "-r", "--name-only", sha], bare).splitlines()
    assert "repo/pr-2/a.png" in names and "README.md" in names


def test_push_evidence_assets_gives_up_after_second_rejection(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, tmp_path: Path
) -> None:
    def runner(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
        if is_push(args):
            raise GitCommandError(args, 1, "! [rejected] non-fast-forward")
        return default_git_runner(args, cwd=cwd, env=env)

    effects = live(profile, paths, state, audit, git_runner=runner)
    asset = tmp_path / "a.png"
    asset.write_text("a", encoding="utf-8")
    with pytest.raises(PushRejected):
        effects.push_evidence_assets(world["repo"], [(asset, "repo/pr-2/a.png")])


def test_evidence_parent_survives_a_concurrent_fetch_rewriting_fetch_head(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str], tmp_path: Path
) -> None:
    """Parallel groups fetch PR branches in the same clone; ``FETCH_HEAD`` is theirs as often as
    ours, so the evidence parent must come from the remote tip, never from ``FETCH_HEAD``."""
    clone, bare, repo = world["clone"], world["bare"], world["repo"]
    first = tmp_path / "one.png"
    first.write_text("1", encoding="utf-8")
    live(profile, paths, state, audit).push_evidence_assets(repo, [(first, "repo/pr-1/one.png")])
    tip = git(["rev-parse", f"refs/heads/{EVIDENCE_BRANCH}"], bare)

    def racing(args: list[str], *, cwd: Any = None, env: Any = None) -> str:
        out = default_git_runner(args, cwd=cwd, env=env)
        if strip_hooks(args)[:2] == ["fetch", "--no-tags"] and args[-1] == f"refs/heads/{EVIDENCE_BRANCH}":
            default_git_runner(["fetch", "--no-tags", "origin", "main"], cwd=cwd)  # another group's for_pr fetch
        return out

    effects = live(profile, paths, state, audit, git_runner=racing)
    second = tmp_path / "two.png"
    second.write_text("2", encoding="utf-8")
    effects.push_evidence_assets(repo, [(second, "repo/pr-1/two.png")])

    sha = git(["rev-parse", f"refs/heads/{EVIDENCE_BRANCH}"], bare)
    assert git(["rev-parse", f"{sha}^"], bare) == tip  # parented on the evidence tip, not on main
    assert sorted(git(["ls-tree", "-r", "--name-only", sha], bare).splitlines()) == ["repo/pr-1/one.png", "repo/pr-1/two.png"]
    assert git(["branch", "--list", EVIDENCE_BRANCH], clone) == ""


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("fatal: Authentication failed for 'https://github.com/o/r.git/'", True),
        ("fatal: could not read Username for 'https://github.com': terminal prompts disabled", True),
        ("git@github.com: Permission denied (publickey).\nfatal: Could not read from remote repository.", True),
        ("remote: Permission to o/r.git denied to bob.\nfatal: unable to access '...': The requested URL returned error: 403", True),
        ("! [rejected] HEAD -> feat (non-fast-forward)\nerror: failed to push some refs", False),
        ("fatal: unable to access 'https://github.com/o/r.git/': Could not resolve host: github.com", False),
        ("remote: error: GH006: Protected branch update failed", False),
    ],
)
def test_is_auth_failure_classifies_git_credential_errors(message: str, expected: bool) -> None:
    assert is_auth_failure(message) is expected


def test_push_evidence_assets_stores_bytes_exactly(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str], tmp_path: Path
) -> None:
    """Assets go in with --no-filters: no autocrlf / attribute conversion, whatever the clone's config."""
    effects = live(profile, paths, state, audit)
    asset = tmp_path / "notes.txt"
    asset.write_bytes(b"line one\r\nline two\n")
    effects.push_evidence_assets(world["repo"], [(asset, "repo/pr-3/notes.txt")])
    sha = git(["rev-parse", f"refs/heads/{EVIDENCE_BRANCH}"], world["bare"])
    expected_blob = git(["hash-object", "--no-filters", str(asset)], world["clone"])
    assert git(["rev-parse", f"{sha}:repo/pr-3/notes.txt"], world["bare"]) == expected_blob


def test_push_evidence_assets_validates_inputs(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, tmp_path: Path
) -> None:
    effects = live(profile, paths, state, audit, git_runner=raising_git)
    assert effects.push_evidence_assets(world["repo"], []) == []
    with pytest.raises(EffectError):
        effects.push_evidence_assets(world["repo"], [(tmp_path / "missing.png", "r/pr-1/m.png")])
    asset = tmp_path / "a.png"
    asset.write_text("a", encoding="utf-8")
    with pytest.raises(ValueError):
        effects.push_evidence_assets(world["repo"], [(asset, "../escape.png")])


# ---------------------------------------------------------------------------------- PR body
def section() -> str:
    return render_evidence(EvidenceData(kind="ci", run_id="r1", head_sha="abc"))


def test_update_pr_evidence_refuses_non_factory_prs_and_edits_only_when_changed(
    profile: Profile, paths: ProfilePaths, state: State, audit: Audit
) -> None:
    gh = FakeGh("Human text.\n")
    effects = live(profile, paths, state, audit, gh=gh)
    with pytest.raises(EffectRefused, match="not opened by the factory"):
        effects.update_pr_evidence("o/r", 1, section())
    assert gh.edits == []

    state.register_pr("o/r", 1, "factory/x", "build")
    effects.update_pr_evidence("o/r", 1, section())
    assert len(gh.edits) == 1
    body = gh.edits[0][2]
    assert body.startswith("Human text.\n") and START in body and END in body
    effects.update_pr_evidence("o/r", 1, section())  # idempotent -> no second edit
    assert len(gh.edits) == 1
    records = [r for r in audit.read() if r["event"] == "effect.update_pr_evidence"]
    assert [r["changed"] for r in records] == [True, False]


def test_open_draft_pr_registers_the_pr(profile: Profile, paths: ProfilePaths, state: State, audit: Audit) -> None:
    gh = FakeGh()
    effects = live(profile, paths, state, audit, gh=gh)
    pr = effects.open_draft_pr("o/r", "factory/abc-1", "main", "Title", "Body", kind="build", issue_key="ABC-1")
    assert pr.number == 42 and pr.is_draft
    assert gh.created == [{"repo": "o/r", "head": "factory/abc-1", "base": "main", "title": "Title", "body": "Body"}]
    assert state.is_factory_pr("o/r", 42)
    row = state.factory_prs()[0]
    assert row["issue_key"] == "ABC-1" and row["kind"] == "build" and row["url"] == pr.url
    assert "effect.open_draft_pr" in events(audit)
    with pytest.raises(ValueError):
        effects.open_draft_pr("o/r", "+bad", "main", "t", "b", kind="build", issue_key=None)


def test_open_draft_pr_links_the_pushes_made_before_the_pr_existed(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, git: Callable[..., str]
) -> None:
    """build/ready push first and open the PR second: the push rows start with ``number`` NULL and
    must be attached to the PR, or the metrics never see a factory commit on factory-opened PRs."""
    gh = FakeGh()
    effects = live(profile, paths, state, audit, gh=gh)
    wt = world["manager"].for_new_branch(world["repo"], "factory/abc-1")
    head = commit_file(git, wt.path, "new.txt", "hi\n", "add new")
    effects.push(wt, "factory/abc-1", kind="build", pr=None)
    other = world["manager"].for_new_branch(world["repo"], "factory/abc-2")
    commit_file(git, other.path, "other.txt", "o\n", "other")
    effects.push(other, "factory/abc-2", kind="build", pr=None)
    assert state.pushed_shas(SLUG, 42) == set() and all(p["number"] is None for p in state.pushes(SLUG))

    pr = effects.open_draft_pr(SLUG, "factory/abc-1", "main", "T", "B", kind="build", issue_key="ABC-1")

    assert pr.number == 42
    linked = state.pushes(SLUG, 42)
    assert len(linked) == 1 and linked[0]["commit_shas"] == [head] and state.pushed_shas(SLUG, 42) == {head}
    row = next(r for r in state.touched_prs() if r["number"] == 42)
    assert row["pushes"] == 1 and row["factory_opened"] is True
    assert next(r for r in audit.read() if r["event"] == "effect.open_draft_pr")["linked_pushes"] == 1
    assert [p["number"] for p in state.pushes(SLUG) if p["branch"] == "factory/abc-2"] == [None]  # other branch untouched


# ---------------------------------------------------------------------------------- tracker
def parent_issue() -> Issue:
    return Issue(id="p1", key="ABC-1", title="Parent", team_id="team1", project_id="proj1")


def test_create_sub_issues_labels_or_marker_and_relations(
    profile: Profile, paths: ProfilePaths, state: State, audit: Audit
) -> None:
    tracker = FakeTracker(existing_labels={"repo:repo"})
    effects = live(profile, paths, state, audit, tracker=tracker)
    specs = [
        SubIssueSpec(title="API", description="Add endpoint", acceptance_criteria=["returns 200"], repo="o/repo"),
        SubIssueSpec(title="Client", description="Call it", repo="o/other", blocked_by=[0]),
        SubIssueSpec(title="Docs", description="Write", repo="o/repo", blocked_by=[0, 1, 7, 2]),
    ]
    children = effects.create_sub_issues(parent_issue(), specs)
    assert [c.key for c in children] == ["ABC-101", "ABC-102", "ABC-103"]
    creates = [c for c in tracker.calls if c[0] == "create_sub_issue"]
    assert creates[0][1][3] == ["repo:repo"]  # label exists -> label, no marker
    assert REPO_MARKER.format(slug="o/repo") not in creates[0][1][2]
    assert "## Acceptance criteria\n\n1. returns 200" in creates[0][1][2]
    assert creates[1][1][3] == [] and REPO_MARKER.format(slug="o/other") in creates[1][1][2]  # marker fallback
    assert all("factory:build" not in c[1][3] for c in creates)
    relations = [c[1] for c in tracker.calls if c[0] == "add_blocked_by"]
    assert relations == [("child-2", "child-1"), ("child-3", "child-1"), ("child-3", "child-2")]
    assert events(audit).count("effect.create_sub_issue") == 3 and events(audit).count("effect.add_blocked_by") == 3


def test_create_sub_issues_decides_repo_labels_per_team(profile: Profile, paths: ProfilePaths, state: State, audit: Audit) -> None:
    """``labels_exist`` is workspace-wide; ``create_sub_issue`` applies only the parent team's (or
    workspace) labels. A ``repo:`` label owned by another team must fall back to the marker, or
    the child gets neither hint."""

    class TeamTracker(FakeTracker):
        def ensure_label(self, team_id: str, name: str) -> str:
            if team_id == "team1" and name.lower() in self.existing_labels:
                return "lbl-1"
            raise TrackerError(f"label {name!r} does not exist for team {team_id}")

    tracker = TeamTracker(existing_labels={"repo:repo"})
    effects = live(profile, paths, state, audit, tracker=tracker)
    specs = [SubIssueSpec(title="A", description="d", repo="o/repo")]

    effects.create_sub_issues(parent_issue(), specs)  # team1 owns the label
    other = Issue(id="p2", key="XYZ-1", title="Parent", team_id="team2", project_id="proj1")
    effects.create_sub_issues(other, specs)  # team2 cannot apply it although labels_exist says it exists

    creates = [c[1] for c in tracker.calls if c[0] == "create_sub_issue"]
    assert creates[0][3] == ["repo:repo"] and REPO_MARKER.format(slug="o/repo") not in creates[0][2]
    assert creates[1][3] == [] and REPO_MARKER.format(slug="o/repo") in creates[1][2]
    # a tracker without the lookup still gets the workspace-wide answer
    plain = FakeTracker(existing_labels={"repo:repo"})
    live(profile, paths, state, audit, tracker=plain).create_sub_issues(other, specs)
    assert [c[1] for c in plain.calls if c[0] == "create_sub_issue"][0][3] == ["repo:repo"]


def test_tracker_effects_delegate_and_audit(profile: Profile, paths: ProfilePaths, state: State, audit: Audit) -> None:
    tracker = FakeTracker()
    effects = live(profile, paths, state, audit, tracker=tracker)
    effects.transition_issue("i1", "started")
    effects.update_issue_text("i1", "New title", "New body")
    effects.remove_label("i1", "factory:refine")
    assert tracker.calls == [
        ("move_to_state_type", ("i1", "started")),
        ("update_issue_text", ("i1", "New title", "New body")),
        ("remove_label", ("i1", "factory:refine")),
    ]
    assert events(audit)[-3:] == ["effect.transition_issue", "effect.update_issue_text", "effect.remove_label"]


def test_tracker_effects_without_tracker_raise(profile: Profile, paths: ProfilePaths, state: State, audit: Audit) -> None:
    effects = live(profile, paths, state, audit)
    with pytest.raises(EffectError, match="tracker"):
        effects.transition_issue("i1", "started")


def test_write_review_is_local_in_every_mode(profile: Profile, paths: ProfilePaths, state: State, audit: Audit) -> None:
    for effects in (live(profile, paths, state, audit), RecordingEffects(profile, paths, state, None, None, audit, raising_git)):
        path = effects.write_review("o/repo-name", 12, "# Review\n\nfine")
        assert path == paths.review_file("repo-name", 12)
        assert path.read_text(encoding="utf-8") == "# Review\n\nfine\n"
    assert events(audit).count("effect.write_review") == 2


# ---------------------------------------------------------------------------- recording mode
def test_recording_effects_make_zero_writes(
    world: dict, profile: Profile, paths: ProfilePaths, state: State, audit: Audit, tmp_path: Path
) -> None:
    gh = FakeGh("body", raise_on_write=True)
    tracker = FakeTracker(raise_on_write=True)
    effects = RecordingEffects(profile, paths, state, gh, tracker, audit, raising_git)
    wt = world["manager"].for_new_branch(world["repo"], "factory/dry")
    asset = tmp_path / "a.png"
    asset.write_text("a", encoding="utf-8")
    state.register_pr("o/r", 1, "b", "build")  # so update_pr_evidence passes the policy gate

    assert effects.push(wt, "factory/dry", kind="build", pr=None) == "dry-run"
    pr = effects.open_draft_pr("o/r", "factory/dry", "main", "T", "B", kind="build", issue_key="ABC-1")
    assert pr.number == 0 and pr.url.startswith("dry-run://") and pr.is_draft
    effects.update_pr_evidence("o/r", 1, section())
    with pytest.raises(EffectRefused):
        effects.update_pr_evidence("o/r", 2, section())  # policy still applies
    assert effects.push_evidence_assets(world["repo"], [(asset, "repo/pr-1/a.png")]) == [
        evidence_url(SLUG, "repo/pr-1/a.png")
    ]
    effects.transition_issue("i1", "started")
    children = effects.create_sub_issues(parent_issue(), [SubIssueSpec(title="A", description="d", repo="o/repo")])
    assert len(children) == 1 and children[0].id.startswith("dry-run:")
    effects.update_issue_text("i1", "t", "d")
    effects.remove_label("i1", "factory:refine")
    review = effects.write_review("o/r", 3, "draft")
    assert review.is_file()

    assert gh.edits == [] and gh.created == [] and tracker.calls == []
    assert state.pushes() == [] and [r for r in state.factory_prs() if r["number"] != 1] == []
    recorded = [e for e in events(audit) if e.startswith("would_")]
    assert recorded == [
        "would_push",
        "would_open_draft_pr",
        "would_update_pr_evidence",
        "would_push_evidence_assets",
        "would_transition_issue",
        "would_create_sub_issues",
        "would_update_issue_text",
        "would_remove_label",
    ]
    push_record = next(r for r in audit.read() if r["event"] == "would_push")
    assert push_record["refspec"] == "HEAD:refs/heads/factory/dry" and push_record["branch"] == "factory/dry"
    assert is_recording(effects)


def test_make_effects_selects_implementation(profile: Profile, paths: ProfilePaths, state: State, audit: Audit) -> None:
    full = make_effects(profile, dry_run=False, paths=paths, state=state, gh=None, audit=audit)
    assert isinstance(full, LiveEffects)
    assert isinstance(make_effects(profile, dry_run=True, paths=paths, state=state, gh=None, audit=audit), RecordingEffects)
    observe = profile.model_copy(update={"mode": "observe"})
    assert isinstance(make_effects(observe, dry_run=False, paths=paths, state=state, gh=None, audit=audit), RecordingEffects)
    # audit defaults to the profile's audit log
    auto = make_effects(profile, dry_run=True, paths=paths, state=state)
    assert isinstance(auto, RecordingEffects) and auto.audit.path == paths.audit_log
