"""Worktree tests against real git: a bare origin, a seed clone that publishes branches and a
fork-style ``refs/pull/<n>/head``, and the user's clone the manager works from."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from software_factory import providers as providers_mod
from software_factory.providers import unplantable_hooks_path
from software_factory.sources.models import PullRequest
from software_factory.worktree import (
    FACTORY_EXCLUDE,
    Worktree,
    WorktreeError,
    WorktreeManager,
    clone_lock,
    default_git,
)


def _strip_no_hooks(args: list[str]) -> list[str]:
    """Drop the ``-c core.hooksPath=<dir>`` prefix the manager puts on clone-level commands."""
    if len(args) >= 2 and args[0] == "-c" and args[1].startswith("core.hooksPath="):
        return args[2:]
    return args


def _has_no_hooks_prefix(args: list[str], manager: WorktreeManager) -> bool:
    return args[:2] == ["-c", f"core.hooksPath={unplantable_hooks_path()}"] and manager.hooks_path == unplantable_hooks_path()


def _plant_hook(hooks_dir: Path, name: str, marker: Path) -> None:
    """A hook that records that it ran. Executable bit for POSIX; git on Windows runs any file."""
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook = hooks_dir / name
    hook.write_text(f"#!/bin/sh\necho ran > '{marker.as_posix()}'\n", encoding="utf-8")
    hook.chmod(hook.stat().st_mode | 0o755)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


class Repo:
    """Duck-typed stand-in for repos.RepoInfo."""

    def __init__(self, path: Path, default_branch: str = "main"):
        self.slug = "o/r"
        self.name = "r"
        self.dirname = path.name
        self.path = path
        self.default_branch = default_branch


@pytest.fixture(autouse=True)
def git_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from the user's global git config (signing, hooksPath, ...) and fix the identity."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "Factory Test")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "factory@example.com")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    origin = tmp_path / "origin repo.git"  # spaces on purpose
    git(["init", "--bare", "-b", "main", str(origin)], cwd=tmp_path)
    seed = tmp_path / "seed"
    git(["clone", str(origin), str(seed)], cwd=tmp_path)
    (seed / "README.md").write_text("# r\n", encoding="utf-8")
    git(["add", "."], cwd=seed)
    git(["commit", "-q", "-m", "init"], cwd=seed)
    git(["push", "-q", "origin", "HEAD:main"], cwd=seed)
    main_sha = git(["rev-parse", "HEAD"], cwd=seed)
    git(["checkout", "-q", "-b", "feature"], cwd=seed)
    (seed / "feature.txt").write_text("feature\n", encoding="utf-8")
    git(["add", "."], cwd=seed)
    git(["commit", "-q", "-m", "feature work"], cwd=seed)
    feature_sha = git(["rev-parse", "HEAD"], cwd=seed)
    git(["push", "-q", "origin", "feature"], cwd=seed)
    # fork-style PR: only reachable via refs/pull/7/head on origin
    git(["checkout", "-q", "-b", "fork-only", "main"], cwd=seed)
    (seed / "fork.txt").write_text("from a fork\n", encoding="utf-8")
    git(["add", "."], cwd=seed)
    git(["commit", "-q", "-m", "fork work"], cwd=seed)
    fork_sha = git(["rev-parse", "HEAD"], cwd=seed)
    git(["push", "-q", "origin", "fork-only:refs/pull/7/head"], cwd=seed)
    git(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=origin)

    clone = tmp_path / "my clone"  # the user's checkout, with a space in the path
    git(["clone", "-q", str(origin), str(clone)], cwd=tmp_path)
    return {"origin": origin, "clone": clone, "main": main_sha, "feature": feature_sha, "fork": fork_sha}


@pytest.fixture
def manager(tmp_path: Path) -> WorktreeManager:
    return WorktreeManager(tmp_path / "wt dir")


@pytest.fixture
def repo(world: dict[str, Any]) -> Repo:
    return Repo(world["clone"])


def _pr(world: dict[str, Any], **kw: Any) -> PullRequest:
    base = dict(repo="o/r", number=5, head_ref_name="feature", head_ref_oid=world["feature"], base_ref_name="main")
    base.update(kw)
    return PullRequest(**base)


def _worktree_paths(clone: Path) -> list[str]:
    out = git(["worktree", "list", "--porcelain"], cwd=clone)
    return [line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("worktree ")]


def _user_checkout_untouched(world: dict[str, Any]) -> None:
    clone = world["clone"]
    assert git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=clone) == "main"
    assert git(["rev-parse", "HEAD"], cwd=clone) == world["main"]
    assert git(["status", "--porcelain"], cwd=clone) == ""


# -------------------------------------------------------------------------------------- for_pr
def test_for_pr_detached_at_head_oid(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    wt = manager.for_pr(repo, _pr(world))
    assert isinstance(wt, Worktree)
    assert wt.path.parent == manager.root and wt.path.is_dir()
    assert len(wt.path.name) <= 20, "short names keep Windows MAX_PATH headroom"
    assert wt.detached is True and wt.branch is None
    assert wt.base_sha == world["feature"] and wt.repo == "o/r" and wt.clone == world["clone"]
    assert manager.head(wt) == world["feature"]
    assert (wt.path / "feature.txt").read_text(encoding="utf-8") == "feature\n"
    assert git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=wt.path) == "HEAD"
    assert str(wt.path.resolve()) in [str(Path(p).resolve()) for p in _worktree_paths(world["clone"])]
    _user_checkout_untouched(world)


def test_for_pr_adds_factory_dir_to_shared_exclude_once(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    wt1 = manager.for_pr(repo, _pr(world))
    wt2 = manager.for_pr(repo, _pr(world, number=6))
    exclude = world["clone"] / ".git" / "info" / "exclude"
    lines = exclude.read_text(encoding="utf-8").splitlines()
    assert lines.count(FACTORY_EXCLUDE) == 1
    for wt in (wt1, wt2):
        (wt.path / ".factory").mkdir()
        (wt.path / ".factory" / "scratch.png").write_bytes(b"\x89PNG")
        assert git(["status", "--porcelain"], cwd=wt.path) == ""  # ignored


def test_for_pr_fork_fetches_pull_ref(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    pr = _pr(world, number=7, head_repo="someone/r", head_ref_name="branch-not-on-origin", head_ref_oid=world["fork"])
    assert pr.is_fork
    wt = manager.for_pr(repo, pr)
    assert manager.head(wt) == world["fork"]
    assert (wt.path / "fork.txt").is_file()


def test_for_pr_non_fork_with_unknown_branch_fails(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    with pytest.raises(WorktreeError):
        manager.for_pr(repo, _pr(world, head_ref_name="does-not-exist", head_ref_oid=world["fork"]))
    assert not manager.root.exists() or list(manager.root.iterdir()) == []


def test_for_pr_requires_head_oid(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    with pytest.raises(WorktreeError, match="head_ref_oid"):
        manager.for_pr(repo, _pr(world, head_ref_oid=""))


def test_for_pr_unknown_oid_raises_and_leaves_nothing(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    with pytest.raises(WorktreeError):
        manager.for_pr(repo, _pr(world, head_ref_oid="0123456789abcdef0123456789abcdef01234567"))
    assert not manager.root.exists() or list(manager.root.iterdir()) == []
    assert len(_worktree_paths(world["clone"])) == 1


def test_for_pr_head_verification_failure_removes_worktree(tmp_path: Path, repo: Repo, world: dict[str, Any]) -> None:
    def lying_git(args: list[str], *, cwd: Path | None = None, **kw: Any) -> str:
        if args[:2] == ["rev-parse", "HEAD"]:
            return "f" * 40 + "\n"
        return default_git(args, cwd=cwd)

    manager = WorktreeManager(tmp_path / "wt", git=lying_git)
    with pytest.raises(WorktreeError, match="does not match"):
        manager.for_pr(repo, _pr(world))
    assert list(manager.root.iterdir()) == []
    assert len(_worktree_paths(world["clone"])) == 1  # pruned
    _user_checkout_untouched(world)


# ------------------------------------------------------------------------------ for_new_branch
def _local_branches(clone: Path) -> list[str]:
    return [line.strip().lstrip("* ") for line in git(["branch", "--list"], cwd=clone).splitlines() if line.strip()]


def test_for_new_branch_from_default(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    wt = manager.for_new_branch(repo, "factory/abc-123-do-thing")
    assert wt.branch == "factory/abc-123-do-thing" and wt.detached is True  # branch = the push target only
    assert wt.base_sha == world["main"]
    assert manager.head(wt) == world["main"]
    assert git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=wt.path) == "HEAD"  # detached on purpose
    _user_checkout_untouched(world)
    assert _local_branches(world["clone"]) == ["main"]  # no local branch created in the user's clone
    assert "origin/factory/abc-123-do-thing" not in git(["branch", "-r"], cwd=world["clone"])  # nothing pushed


def test_for_new_branch_bases_on_existing_origin_branch(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    wt = manager.for_new_branch(repo, "feature")
    assert wt.base_sha == world["feature"] and wt.branch == "feature" and wt.detached is True
    assert manager.head(wt) == world["feature"]
    assert git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=wt.path) == "HEAD"
    _user_checkout_untouched(world)
    assert _local_branches(world["clone"]) == ["main"]


def test_for_new_branch_refuses_the_default_branch(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    with pytest.raises(WorktreeError, match="default branch"):
        manager.for_new_branch(repo, "main")
    _user_checkout_untouched(world)


def test_for_new_branch_never_touches_a_pre_existing_local_branch(
    manager: WorktreeManager, repo: Repo, world: dict[str, Any]
) -> None:
    """Regression: ``git worktree add -B <branch>`` reset the user's own local branch (Linear's
    branchName, started locally, unpushed) to origin/main and its commits vanished from the branch."""
    clone = world["clone"]
    git(["branch", "tester/abc-1-build-it", "main"], cwd=clone)
    git(["worktree", "add", "-q", "--detach", str(clone.parent / "scratch"), "tester/abc-1-build-it"], cwd=clone)
    scratch = clone.parent / "scratch"
    git(["checkout", "-q", "tester/abc-1-build-it"], cwd=scratch)
    (scratch / "wip.txt").write_text("unpushed work\n", encoding="utf-8")
    git(["add", "."], cwd=scratch)
    git(["commit", "-q", "-m", "wip"], cwd=scratch)
    wip_sha = git(["rev-parse", "HEAD"], cwd=scratch)
    git(["checkout", "-q", "--detach"], cwd=scratch)  # branch exists locally, is not checked out anywhere

    wt = manager.for_new_branch(repo, "tester/abc-1-build-it")

    assert git(["rev-parse", "refs/heads/tester/abc-1-build-it"], cwd=clone) == wip_sha  # untouched
    assert manager.head(wt) == world["main"] and wt.base_sha == world["main"]  # factory work starts from origin
    assert git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=wt.path) == "HEAD"
    assert "wip" in git(["log", "--oneline", "tester/abc-1-build-it"], cwd=clone)
    _user_checkout_untouched(world)


def test_for_new_branch_is_not_blocked_by_a_kept_worktree_for_the_same_branch(
    manager: WorktreeManager, repo: Repo, world: dict[str, Any]
) -> None:
    """Regression: a worktree kept for the human (push refused) held the branch checked out, so after
    ``factory release`` without --remove every retry failed with 'already checked out'."""
    kept = manager.for_new_branch(repo, "tester/abc-1")
    (kept.path / "x.txt").write_text("x\n", encoding="utf-8")
    git(["add", "."], cwd=kept.path)
    git(["commit", "-q", "-m", "refused push"], cwd=kept.path)
    assert kept.path.is_dir()  # not removed: the human may still be looking at it

    again = manager.for_new_branch(repo, "tester/abc-1")

    assert again.path != kept.path and again.path.is_dir() and kept.path.is_dir()
    assert manager.head(again) == world["main"]
    assert manager.head(kept) != world["main"]  # the kept work is left alone
    _user_checkout_untouched(world)


@pytest.mark.parametrize("name", ["", "-x", "a b", "a~b", "a^b", "a:b", "a?b", "a*b", "a[b", "a\\b"])
def test_for_new_branch_rejects_invalid_names(manager: WorktreeManager, repo: Repo, name: str) -> None:
    with pytest.raises(WorktreeError, match="invalid branch"):
        manager.for_new_branch(repo, name)


def test_for_default_branch(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    wt = manager.for_default_branch(repo)
    assert wt.detached and wt.branch is None and wt.base_sha == world["main"]
    assert manager.head(wt) == world["main"]
    assert (wt.path / "README.md").is_file() and not (wt.path / "feature.txt").exists()


# ------------------------------------------------------------------------------------ inspect
def test_new_commits_and_changed_files(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    wt = manager.for_pr(repo, _pr(world))
    assert manager.new_commits(wt) == [] and manager.changed_files(wt) == []
    (wt.path / "src").mkdir()
    (wt.path / "src" / "b.py").write_text("b = 1\n", encoding="utf-8")
    git(["add", "."], cwd=wt.path)
    git(["commit", "-q", "-m", "one"], cwd=wt.path)
    first = git(["rev-parse", "HEAD"], cwd=wt.path)
    (wt.path / "feature.txt").write_text("changed\n", encoding="utf-8")
    git(["commit", "-q", "-am", "two"], cwd=wt.path)
    second = git(["rev-parse", "HEAD"], cwd=wt.path)
    assert manager.new_commits(wt) == [first, second]  # oldest first
    assert sorted(manager.changed_files(wt)) == ["feature.txt", "src/b.py"]
    assert manager.head(wt) == second
    _user_checkout_untouched(world)


def test_has_conflict_markers(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    wt = manager.for_pr(repo, _pr(world))
    assert manager.has_conflict_markers(wt) is False
    # a setext heading underline is not a conflict marker
    (wt.path / "notes.md").write_text("Title\n=======\nbody\n", encoding="utf-8")
    assert manager.has_conflict_markers(wt) is False
    # binary file containing marker bytes is skipped
    (wt.path / "blob.bin").write_bytes(b"\x00\x01<<<<<<< HEAD\n")
    assert manager.has_conflict_markers(wt) is False
    (wt.path / "feature.txt").write_text("<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> main\n", encoding="utf-8")
    assert manager.has_conflict_markers(wt) is True  # dirty, uncommitted
    git(["commit", "-q", "-am", "oops committed markers"], cwd=wt.path)
    assert manager.has_conflict_markers(wt) is True  # committed
    (wt.path / "feature.txt").write_text("resolved\n", encoding="utf-8")
    git(["commit", "-q", "-am", "resolved"], cwd=wt.path)
    assert manager.has_conflict_markers(wt) is False


# -------------------------------------------------------------------------------------- hooks
def test_clone_level_git_never_runs_hooks_from_the_users_clone(
    manager: WorktreeManager, repo: Repo, world: dict[str, Any], tmp_path: Path
) -> None:
    """Regression: ``git worktree add`` fires ``post-checkout`` from the clone's shared
    ``.git/hooks`` (writable by any worker's shell), so a planted hook ran with the engine's
    environment every time a worktree was created. Every clone-level command now runs with an empty
    ``core.hooksPath``."""
    clone = world["clone"]
    checkout_marker = tmp_path / "post-checkout.ran"
    ref_marker = tmp_path / "reference-transaction.ran"
    _plant_hook(clone / ".git" / "hooks", "post-checkout", checkout_marker)
    _plant_hook(clone / ".git" / "hooks", "reference-transaction", ref_marker)
    # sanity: the planted hook is live on this platform when git runs without the manager
    git(["worktree", "add", "-q", "--detach", str(tmp_path / "plain"), world["main"]], cwd=clone)
    assert checkout_marker.is_file(), "post-checkout did not fire at all; the test cannot prove anything"
    checkout_marker.unlink()
    ref_marker.unlink(missing_ok=True)

    wt_pr = manager.for_pr(repo, _pr(world))
    wt_br = manager.for_new_branch(repo, "factory/abc-1")
    wt_def = manager.for_default_branch(repo)
    for wt in (wt_pr, wt_br, wt_def):
        manager.remove(wt)
    manager.cleanup(set())

    assert not checkout_marker.exists(), "post-checkout ran inside an engine worktree add"
    assert not ref_marker.exists(), "reference-transaction ran inside an engine fetch"
    assert manager.head(Worktree(path=clone, repo="o/r", clone=clone, base_sha=world["main"])) == world["main"]
    _user_checkout_untouched(world)


def test_the_manager_creates_no_hooks_dir_and_points_git_at_the_unplantable_path(
    tmp_path: Path, repo: Repo, world: dict[str, Any]
) -> None:
    """Regression: the manager made ``<root>/../no-hooks`` and emptied it before each command; any
    process could refill it between the emptying and git's exec, and emptying followed junctions.
    Now nothing is created: ``core.hooksPath`` is a path under a regular file / character device."""
    root = tmp_path / "data" / "wt"
    manager = WorktreeManager(root)
    assert manager.hooks_path == unplantable_hooks_path()
    assert not os.path.isdir(manager.hooks_path)
    assert not (tmp_path / "data").exists()  # neither a hooks dir nor the (lazily created) root
    assert manager.no_hooks_args() == ["-c", f"core.hooksPath={unplantable_hooks_path()}"]
    wt = manager.for_default_branch(repo)
    assert manager.head(wt) == world["main"]
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["wt"]


def test_the_manager_fails_closed_when_hooks_cannot_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup sanity check plus a check before every clone-level command: when the unplantable
    path is not available (its parent became a directory, or is gone) no git runs at all."""
    calls: list[list[str]] = []

    def fake_git(args: list[str], *, cwd: Path | None = None) -> str:
        calls.append(list(args))
        return ""

    manager = WorktreeManager(tmp_path / "wt", git=fake_git)
    fake_hosts = tmp_path / "hosts"
    fake_hosts.mkdir()  # a directory: something could be planted under it
    monkeypatch.setattr(providers_mod, "_unplantable_parent", lambda: str(fake_hosts))
    with pytest.raises(WorktreeError, match="hooks enabled"):
        WorktreeManager(tmp_path / "wt2", git=fake_git)
    clone = tmp_path / "clone"
    clone.mkdir()
    with pytest.raises(WorktreeError, match="hooks enabled"):
        manager.for_new_branch(Repo(clone), "factory/x")
    with pytest.raises(WorktreeError, match="hooks enabled"):
        manager.no_hooks_args()
    assert calls == []


def test_every_clone_level_command_carries_the_no_hooks_prefix(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_git(args: list[str], *, cwd: Path | None = None) -> str:
        calls.append(list(args))
        args = _strip_no_hooks(args)
        if args[:1] == ["rev-parse"] and args[-1].endswith("^{commit}"):
            return "c" * 40 + "\n"
        if args[:2] == ["rev-parse", "HEAD"]:
            return "c" * 40 + "\n"
        if args[:3] == ["rev-parse", "--git-path", "info/exclude"]:
            return str(tmp_path / "exclude.txt") + "\n"
        if args[:2] == ["worktree", "add"]:
            Path(args[-2]).mkdir(parents=True, exist_ok=True)
        return ""

    manager = WorktreeManager(tmp_path / "wt", git=fake_git)
    clone = tmp_path / "clone"
    clone.mkdir()
    wt = manager.for_new_branch(Repo(clone), "factory/x")
    manager.remove(wt)
    clone_level = [c for c in calls if _strip_no_hooks(c)[0] in ("fetch", "worktree") or "^{commit}" in c[-1]
                   or "--verify" in c]  # fmt: skip
    assert len(clone_level) >= 5  # fetch, rev-parse --verify, rev-parse ^{commit}, worktree add, remove, prune
    assert all(_has_no_hooks_prefix(c, manager) for c in clone_level), clone_level
    # commands inside the worktree (rev-parse HEAD, the exclude lookup) are passed through unchanged
    assert ["rev-parse", "HEAD"] in calls and ["rev-parse", "--git-path", "info/exclude"] in calls


# ------------------------------------------------------------------------------------ removal
def test_remove(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    wt = manager.for_pr(repo, _pr(world))
    (wt.path / "untracked.txt").write_text("x", encoding="utf-8")  # --force handles dirty trees
    manager.remove(wt)
    assert not wt.path.exists()
    assert len(_worktree_paths(world["clone"])) == 1
    manager.remove(wt)  # idempotent
    _user_checkout_untouched(world)


def test_cleanup_keeps_only_listed_paths(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    keep = manager.for_pr(repo, _pr(world))
    drop = manager.for_pr(repo, _pr(world, number=6))
    stray = manager.root / "stray-dir"
    stray.mkdir()
    (stray / "f").write_text("x", encoding="utf-8")
    removed = manager.cleanup({str(keep.path)})
    assert sorted(removed) == sorted([drop.path, stray])
    assert keep.path.is_dir() and not drop.path.exists() and not stray.exists()
    paths = [str(Path(p).resolve()) for p in _worktree_paths(world["clone"])]
    assert str(keep.path.resolve()) in paths and str(drop.path.resolve()) not in paths
    assert manager.cleanup({str(keep.path)}) == []
    assert WorktreeManager(manager.root / "missing").cleanup(set()) == []


# ----------------------------------------------------------------------------------- runners
def test_default_git_errors_and_output(world: dict[str, Any]) -> None:
    assert default_git(["rev-parse", "HEAD"], cwd=world["clone"]).strip() == world["main"]
    with pytest.raises(WorktreeError, match="rev-parse"):
        default_git(["rev-parse", "--verify", "refs/heads/nope"], cwd=world["clone"])


def test_injected_runner_receives_args_and_cwd(tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path | None]] = []
    canned = {
        ("rev-parse", "HEAD"): "abc\n",
        ("rev-list", "--reverse", "base..HEAD"): "c1\nc2\n",
        ("diff", "--name-only", "base..HEAD"): "a.py\nsub\\b.py\n",
    }

    def fake_git(args: list[str], *, cwd: Path | None = None) -> str:
        calls.append((args, cwd))
        return canned[tuple(args)]

    manager = WorktreeManager(tmp_path / "wt", git=fake_git)
    wt = Worktree(path=tmp_path / "wt" / "x", repo="o/r", clone=tmp_path / "clone", base_sha="base")
    assert manager.head(wt) == "abc"
    assert manager.new_commits(wt) == ["c1", "c2"]
    assert manager.changed_files(wt) == ["a.py", "sub/b.py"]
    assert all(cwd == wt.path for _, cwd in calls)


def test_paths_with_spaces_are_quoted_safely(manager: WorktreeManager, repo: Repo, world: dict[str, Any]) -> None:
    assert " " in str(repo.path) and " " in str(manager.root)
    wt = manager.for_default_branch(repo)
    assert wt.path.is_dir() and os.path.isfile(wt.path / "README.md")


# --------------------------------------------------------------------------------- clone lock
class _SlowGit:
    """Fake git that sleeps inside clone-level commands and records how many ran at once."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self._guard = threading.Lock()
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], *, cwd: Path | None = None) -> str:
        args = _strip_no_hooks(args)
        with self._guard:
            self.calls.append(list(args))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if args[0] in ("fetch", "worktree"):
                time.sleep(self.delay)
            if args[:1] == ["rev-parse"] and args[-1].endswith("^{commit}"):
                return "a" * 40 + "\n"
            if args[:2] == ["rev-parse", "HEAD"]:
                return "a" * 40 + "\n"
            if args[:3] == ["rev-parse", "--git-path", "info/exclude"]:
                return str((cwd or Path(".")) / "exclude.txt") + "\n"
            if args[:2] == ["worktree", "add"]:
                Path(args[-2]).mkdir(parents=True, exist_ok=True)
            return ""
        finally:
            with self._guard:
                self.in_flight -= 1


def test_clone_level_git_operations_are_serialised_per_clone(tmp_path: Path) -> None:
    """Regression: two groups on the same repo ran ``git fetch`` concurrently in one clone and git
    failed one of them with 'cannot lock ref' -> a spurious 'worktree failed' attempt."""
    fake = _SlowGit()
    manager = WorktreeManager(tmp_path / "wt", git=fake)
    clone = tmp_path / "clone"
    clone.mkdir()
    repo = Repo(clone)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            manager.for_default_branch(repo)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    assert fake.max_in_flight == 1, "clone-level git commands overlapped"
    assert sum(1 for c in fake.calls if c[0] == "fetch") == 4
    # a different clone has its own lock; the same clone (any spelling) shares one
    assert clone_lock(clone) is clone_lock(Path(str(clone).upper()) if os.name == "nt" else clone)
    assert clone_lock(clone) is not clone_lock(tmp_path / "other")


def test_fetch_retries_a_ref_lock_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def flaky_git(args: list[str], *, cwd: Path | None = None) -> str:
        args = _strip_no_hooks(args)
        if args[0] == "fetch":
            attempts.append(1)
            if len(attempts) < 3:
                raise WorktreeError(
                    "git fetch --no-tags origin failed (1): error: cannot lock ref 'refs/remotes/origin/b1': "
                    "is at 1111 but expected 2222"
                )
            return ""
        if args[:1] == ["rev-parse"] and args[-1].endswith("^{commit}"):
            return "b" * 40 + "\n"
        if args[:2] == ["rev-parse", "HEAD"]:
            return "b" * 40 + "\n"
        if args[:3] == ["rev-parse", "--git-path", "info/exclude"]:
            return str(tmp_path / "exclude.txt") + "\n"
        if args[:2] == ["worktree", "add"]:
            Path(args[-2]).mkdir(parents=True, exist_ok=True)
        return ""

    monkeypatch.setattr("software_factory.worktree.FETCH_RETRY_DELAY_S", 0.0)
    manager = WorktreeManager(tmp_path / "wt", git=flaky_git)
    wt = manager.for_default_branch(Repo(tmp_path / "clone"))
    assert wt.base_sha == "b" * 40 and len(attempts) == 3

    attempts.clear()

    def hard_fail(args: list[str], *, cwd: Path | None = None) -> str:
        attempts.append(1)
        raise WorktreeError("git fetch failed (128): fatal: could not read from remote repository")

    with pytest.raises(WorktreeError, match="remote repository"):
        WorktreeManager(tmp_path / "wt2", git=hard_fail).for_default_branch(Repo(tmp_path / "clone"))
    assert len(attempts) == 1  # only ref-lock races are retried
