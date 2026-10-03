"""Regression tests for the unplantable ``core.hooksPath`` (real git, local bare origins only).

The engine used to point ``core.hooksPath`` at ``<profile root>/no-hooks`` (and workers at
``no-hooks-worker``), emptying the directory right before each git command. Another process -- a
second group's worker, or a background loop a worker left running -- could re-plant ``pre-push``
between the emptying and git's exec (the hook ran inside ``LiveEffects.push``), emptying followed
junctions into their targets, and two engine fetches (the evidence branch's parent fetch and
``ready_action``'s fetch in the user's clone) ran with no override at all. Every git the engine runs
in a clone or worktree, and every worker git, now uses ``providers.unplantable_hooks_path()``: a path
under a regular file (the hosts file) or a character device (``/dev/null``), which no process can
ever create.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from software_factory import providers as providers_mod
from software_factory.config import Profile, ProviderConfig, RepoOverride
from software_factory.effects import EffectError, LiveEffects
from software_factory.evidence import EVIDENCE_BRANCH
from software_factory.kinds.ready import ready_action
from software_factory.notify import Audit
from software_factory.paths import ProfilePaths
from software_factory.providers import ProviderError, unplantable_hooks_path, worker_env
from software_factory.repos import RepoIndex, RepoInfo
from software_factory.sources.models import PullRequest
from software_factory.state import State
from software_factory.worktree import WorktreeError, WorktreeManager

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

SLUG = "JaneDoe/repo"
# Every hook a clone-level / push / commit command could fire.
HOOK_NAMES = (
    "pre-push", "post-checkout", "reference-transaction", "pre-commit", "prepare-commit-msg", "commit-msg",
    "post-commit", "post-merge", "post-rewrite", "pre-auto-gc", "post-index-change",
)  # fmt: skip


def plant_hooks(hooks_dir: Path, marker: Path, names: tuple[str, ...] = HOOK_NAMES) -> None:
    """Hooks that append their name to ``marker`` (exit 0: an unwanted run shows only there)."""
    hooks_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        hook = hooks_dir / name
        hook.write_text(f'#!/bin/sh\necho {name} >> "{marker.as_posix()}"\nexit 0\n', encoding="utf-8")
        hook.chmod(0o755)


def ran(marker: Path) -> list[str]:
    return marker.read_text(encoding="utf-8").split() if marker.exists() else []


def plain_git(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60, env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})}, check=False,
    )  # fmt: skip


def commit_file(git: Callable[..., str], cwd: Path, name: str, content: str, message: str) -> str:
    target = cwd / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(["add", "-A"], cwd)
    git(["commit", "-q", "-m", message], cwd)
    return git(["rev-parse", "HEAD"], cwd)


@pytest.fixture
def paths(tmp_path: Path) -> ProfilePaths:
    return ProfilePaths.for_profile("personal", root=tmp_path / "data root").ensure()


@pytest.fixture
def world(
    bare_repo_factory: Callable[..., Path], git_repo_factory: Callable[..., Path], paths: ProfilePaths,
    git: Callable[..., str], tmp_path: Path,
) -> dict[str, Any]:
    """origin (bare), the user's clone, a second clone that moves origin's refs, a feature branch."""
    bare = bare_repo_factory("origin")
    clone = git_repo_factory("repo", origin=bare)
    other = tmp_path / "other clone"
    git(["clone", "-q", str(bare), str(other)], tmp_path)
    git(["checkout", "-q", "-b", "feature"], other)
    feature = commit_file(git, other, "feature.txt", "f\n", "feature work")
    git(["push", "-q", "origin", "feature"], other)
    git(["checkout", "-q", "main"], other)
    repo = RepoInfo(slug=SLUG, name="repo", dirname="repo", path=clone, default_branch="main", override=RepoOverride())
    return {"bare": bare, "clone": clone, "other": other, "repo": repo, "feature": feature,
            "manager": WorktreeManager(paths.worktrees)}  # fmt: skip


def move_remote(git: Callable[..., str], other: Path, branch: str, name: str) -> str:
    """Someone else pushes a new commit to ``origin/<branch>`` (fast-forward)."""
    git(["fetch", "-q", "origin"], other)
    git(["checkout", "-q", "-B", f"tmp-{branch.replace('/', '-')}", f"origin/{branch}"], other)
    sha = commit_file(git, other, name, f"{name}\n", f"move {branch}")
    git(["push", "-q", "origin", f"HEAD:refs/heads/{branch}"], other)
    git(["checkout", "-q", "main"], other)
    return sha


def live(profile: Profile, paths: ProfilePaths, state: State) -> LiveEffects:
    return LiveEffects(profile, paths, state, None, None, Audit(paths.audit_log, "personal"))


# ----------------------------------------------------------------------------- worktree manager
def test_planted_hooks_never_run_for_worktree_setup_fetch_or_removal(
    world: dict[str, Any], git: Callable[..., str], tmp_path: Path
) -> None:
    clone, manager, repo = world["clone"], world["manager"], world["repo"]
    marker = tmp_path / "hooks ran.txt"
    plant_hooks(clone / ".git" / "hooks", marker)
    # sanity: these hooks are live in this clone when git runs without the override
    move_remote(git, world["other"], "main", "m1.txt")
    assert plain_git(["fetch", "-q", "origin"], clone).returncode == 0
    assert plain_git(["worktree", "add", "-q", "--detach", str(tmp_path / "plain wt"), "HEAD"], clone).returncode == 0
    assert {"reference-transaction", "post-checkout"} <= set(ran(marker)), ran(marker)
    marker.unlink()

    move_remote(git, world["other"], "main", "m2.txt")  # the next fetches really update refs
    move_remote(git, world["other"], "feature", "f2.txt")
    feature = git(["rev-parse", "refs/heads/feature"], world["bare"])
    pr = PullRequest(repo=SLUG, number=5, head_ref_name="feature", head_ref_oid=feature, base_ref_name="main")
    wt_pr = manager.for_pr(repo, pr)
    wt_br = manager.for_new_branch(repo, "factory/abc-1")
    wt_def = manager.for_default_branch(repo)
    for wt in (wt_pr, wt_br, wt_def):
        manager.remove(wt)
    manager.cleanup(set())

    assert ran(marker) == []
    assert git(["rev-parse", "refs/remotes/origin/feature"], clone) == feature  # the fetch did happen


# ------------------------------------------------------------------------------------- effects
def test_planted_hooks_never_run_inside_live_effects_push(
    world: dict[str, Any], profile: Profile, paths: ProfilePaths, state: State, git: Callable[..., str], tmp_path: Path
) -> None:
    clone = world["clone"]
    wt = world["manager"].for_new_branch(world["repo"], "factory/abc-1")
    marker = tmp_path / "hooks ran.txt"
    plant_hooks(clone / ".git" / "hooks", marker)
    head = commit_file(git, wt.path, "new.txt", "hi\n", "worker commit")
    marker.unlink(missing_ok=True)  # the test's own commit fired pre-commit etc. (sanity: hooks are live)

    assert live(profile, paths, state).push(wt, "factory/abc-1", kind="build", pr=None) == head
    head2 = commit_file(git, wt.path, "more.txt", "more\n", "second")
    marker.unlink(missing_ok=True)
    assert live(profile, paths, state).push(wt, "factory/abc-1", kind="ci", pr=4) == head2

    assert ran(marker) == []
    assert git(["rev-parse", "refs/heads/factory/abc-1"], world["bare"]) == head2


def test_planted_hooks_never_run_for_evidence_pushes_even_when_the_parent_fetch_moves_a_ref(
    world: dict[str, Any], profile: Profile, paths: ProfilePaths, state: State, git: Callable[..., str], tmp_path: Path
) -> None:
    """Regression: ``_fetch_evidence_parent`` fetched without the hooks override; when another
    machine had moved ``factory/evidence`` the fetch updated ``refs/remotes/origin/factory/evidence``
    and the clone's ``reference-transaction`` hook fired (twice: prepared + committed)."""
    clone = world["clone"]
    effects = live(profile, paths, state)
    first = tmp_path / "one.png"
    first.write_text("1", encoding="utf-8")
    effects.push_evidence_assets(world["repo"], [(first, "repo/pr-1/one.png")])
    moved = move_remote(git, world["other"], EVIDENCE_BRANCH, "repo/pr-9/theirs.png")
    marker = tmp_path / "hooks ran.txt"
    plant_hooks(clone / ".git" / "hooks", marker)

    second = tmp_path / "two.png"
    second.write_text("2", encoding="utf-8")
    effects.push_evidence_assets(world["repo"], [(second, "repo/pr-1/two.png")])

    assert ran(marker) == []
    tip = git(["rev-parse", f"refs/heads/{EVIDENCE_BRANCH}"], world["bare"])
    assert git(["rev-parse", f"{tip}^"], world["bare"]) == moved  # built on the moved tip, via the fetch
    assert git(["rev-parse", f"refs/remotes/origin/{EVIDENCE_BRANCH}"], clone) in (moved, tip)
    # sanity: the same fetch without the override does fire the planted hook
    move_remote(git, world["other"], EVIDENCE_BRANCH, "repo/pr-9/again.png")
    assert plain_git(["fetch", "-q", "--no-tags", "origin", f"refs/heads/{EVIDENCE_BRANCH}"], clone).returncode == 0
    assert "reference-transaction" in ran(marker)


# ---------------------------------------------------------------------------------- ready_action
def test_ready_action_fetch_in_the_users_clone_runs_no_hooks(
    world: dict[str, Any], git: Callable[..., str], tmp_path: Path
) -> None:
    """Regression: ``ready_action`` fetched the default branch in the user's clone without the
    override, so a moved ``origin/<default>`` fired the clone's ``reference-transaction`` hook."""
    clone = world["clone"]
    marker = tmp_path / "hooks ran.txt"
    plant_hooks(clone / ".git" / "hooks", marker)
    new_sha = move_remote(git, world["other"], "main", "ready.txt")
    deps = types.SimpleNamespace(repos=RepoIndex([world["repo"]]), wt_manager=world["manager"])

    action = ready_action(deps, SLUG)  # type: ignore[arg-type]

    assert action.version == new_sha
    assert ran(marker) == []
    move_remote(git, world["other"], "main", "ready2.txt")
    assert plain_git(["fetch", "-q", "--no-tags", "origin", "main"], clone).returncode == 0
    assert "reference-transaction" in ran(marker)  # sanity: live without the override


# ------------------------------------------------------------------------------------ worker env
def test_planted_hooks_never_run_for_a_commit_under_the_worker_env(
    world: dict[str, Any], git: Callable[..., str], tmp_path: Path
) -> None:
    clone = world["clone"]
    wt = world["manager"].for_new_branch(world["repo"], "factory/abc-1")
    marker = tmp_path / "hooks ran.txt"
    plant_hooks(clone / ".git" / "hooks", marker)
    profile = Profile(name="personal", provider=ProviderConfig(kind="api_key"), secrets={"ANTHROPIC_API_KEY": "k"},
                      disclosure_footer="footer")  # fmt: skip
    git_env = {k: v for k, v in worker_env(profile).items() if k.startswith("GIT_")}
    assert git_env[f"GIT_CONFIG_VALUE_{int(git_env['GIT_CONFIG_COUNT']) - 1}"] == unplantable_hooks_path()

    (wt.path / "a.txt").write_text("a\n", encoding="utf-8")
    assert plain_git(["add", "-A"], wt.path, git_env).returncode == 0
    result = plain_git(["commit", "-q", "-m", "worker commit"], wt.path, git_env)
    assert result.returncode == 0, result.stderr
    assert ran(marker) == []
    (wt.path / "b.txt").write_text("b\n", encoding="utf-8")
    plain_git(["add", "-A"], wt.path)
    assert plain_git(["commit", "-q", "-m", "plain commit"], wt.path).returncode == 0
    assert {"pre-commit", "commit-msg", "post-commit"} <= set(ran(marker))  # sanity: live without it


# ------------------------------------------------------------------------------ fail closed
def test_every_entry_point_fails_closed_when_the_parent_is_a_directory(
    world: dict[str, Any], profile: Profile, paths: ProfilePaths, state: State, git: Callable[..., str],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    wt = world["manager"].for_new_branch(world["repo"], "factory/abc-1")
    commit_file(git, wt.path, "new.txt", "hi\n", "worker commit")
    hosts_dir = tmp_path / "hosts"
    hosts_dir.mkdir()  # a directory under which a hook could be planted
    monkeypatch.setattr(providers_mod, "_unplantable_parent", lambda: str(hosts_dir))

    with pytest.raises(WorktreeError, match="hooks enabled"):
        WorktreeManager(paths.worktrees)
    with pytest.raises(WorktreeError, match="hooks enabled"):
        world["manager"].for_new_branch(world["repo"], "factory/abc-2")
    with pytest.raises(EffectError, match="hooks enabled"):
        live(profile, paths, state).push(wt, "factory/abc-1", kind="build", pr=None)
    deps = types.SimpleNamespace(repos=RepoIndex([world["repo"]]), wt_manager=world["manager"])
    with pytest.raises(WorktreeError, match="hooks enabled"):
        ready_action(deps, SLUG)  # type: ignore[arg-type]
    worker_profile = Profile(name="personal", provider=ProviderConfig(kind="api_key"),
                             secrets={"ANTHROPIC_API_KEY": "k"}, disclosure_footer="footer")  # fmt: skip
    with pytest.raises(ProviderError):
        worker_env(worker_profile)
    assert git(["ls-remote", "--heads", "origin", "refs/heads/factory/abc-1"], world["clone"]) == ""
    assert list(hosts_dir.iterdir()) == []  # nothing was created there either


# ------------------------------------------------------------------------------- concurrency
def test_a_concurrent_planter_cannot_get_a_hook_into_a_push(
    world: dict[str, Any], profile: Profile, paths: ProfilePaths, state: State, git: Callable[..., str], tmp_path: Path
) -> None:
    """Regression for the race the critic reproduced: a process re-planting ``pre-push`` into the
    hooks dir between the engine's emptying and git's exec. A planter thread now hammers every
    plausible directory -- the clone's shared hooks, the old ``no-hooks`` / ``no-hooks-worker`` dirs,
    and the hooks path itself -- while the engine pushes; nothing it writes can run, because the
    hooks path is not (and cannot become) a directory."""
    clone = world["clone"]
    wt = world["manager"].for_new_branch(world["repo"], "factory/abc-1")
    marker = tmp_path / "planted hook ran.txt"
    hooks_path = Path(unplantable_hooks_path())
    targets = [clone / ".git" / "hooks", paths.root / "no-hooks", paths.root / "no-hooks-worker", hooks_path]
    stop = threading.Event()
    planted = dict.fromkeys(map(str, targets), 0)
    refused = dict.fromkeys(map(str, targets), 0)

    def planter() -> None:
        while not stop.is_set():
            for target in targets:
                try:
                    plant_hooks(target, marker, ("pre-push", "reference-transaction"))
                    planted[str(target)] += 1
                except OSError:  # the hooks path always; a hook file busy being executed sometimes
                    refused[str(target)] += 1
            time.sleep(0.001)

    thread = threading.Thread(target=planter, daemon=True)
    effects = live(profile, paths, state)
    heads: list[str] = []
    thread.start()
    try:
        for i in range(4):
            heads.append(commit_file(git, wt.path, f"f{i}.txt", f"{i}\n", f"commit {i}"))
            marker.unlink(missing_ok=True)  # the test's own commits fire planted hooks in the clone
            assert effects.push(wt, "factory/abc-1", kind="build", pr=None) == heads[-1]
            assert ran(marker) == [], f"a planted hook ran inside push {i}: {ran(marker)}"
    finally:
        stop.set()
        thread.join(timeout=30)

    for target in targets[:-1]:  # the planter really wrote hooks into every directory it could reach
        assert planted[str(target)] > 0, target
    assert planted[str(hooks_path)] == 0 and refused[str(hooks_path)] > 0  # ... but never into the hooks path
    assert not hooks_path.exists() and not hooks_path.parent.is_dir()
    assert git(["rev-parse", "refs/heads/factory/abc-1"], world["bare"]) == heads[-1]
