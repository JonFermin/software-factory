"""Cross-module regression tests for the safety-core fixes that need real git: the worker-env
push block at the git level and concurrent clone-level operations in one clone."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from conftest import init_bare_repo, init_git_repo, run_git  # pytest puts tests/ on sys.path (no package)

from software_factory.config import Profile, ProviderConfig
from software_factory.policy import classify_shell
from software_factory.providers import git_no_push_env, worker_env
from software_factory.worktree import WorktreeManager

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


class _Repo:
    def __init__(self, path: Path, default_branch: str = "main"):
        self.slug = "o/r"
        self.name = "r"
        self.dirname = path.name
        self.path = path
        self.default_branch = default_branch


def _git(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60, env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})}, check=False,
    )


def test_worker_git_env_blocks_github_pushes_but_not_fetches(tmp_path: Path) -> None:
    """``git push`` to any GitHub URL (via a remote name or the URL itself) fails inside the worker
    env before touching the network, from a script as much as from the shell; fetching from origin
    keeps working; the engine (without the env) can still push."""
    origin = init_bare_repo(tmp_path / "origin.git")
    clone = init_git_repo(tmp_path / "clone", origin=origin)
    run_git(["remote", "add", "github", "https://github.com/o/r.git"], clone)
    run_git(["remote", "add", "github-ssh", "git@github.com:o/r.git"], clone)
    (clone / "b.txt").write_text("b\n", encoding="utf-8")
    run_git(["add", "-A"], clone)
    run_git(["commit", "-q", "-m", "two"], clone)
    env = git_no_push_env()

    for target in ("github", "github-ssh", "https://github.com/o/r.git", "git@github.com:o/r.git"):
        result = _git(["push", target, "HEAD:refs/heads/main"], clone, env)
        assert result.returncode != 0, target
        assert "factory-no-push" in (result.stderr + result.stdout), target
    assert _git(["fetch", "origin"], clone, env).returncode == 0
    assert _git(["push", "origin", "HEAD:refs/heads/main"], clone).returncode == 0  # the engine's own push
    assert run_git(["rev-parse", "main"], origin) == run_git(["rev-parse", "HEAD"], clone)


def _plant_hook(hooks_dir: Path, name: str, marker: Path) -> None:
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook = hooks_dir / name
    hook.write_text(f"#!/bin/sh\necho ran > '{marker.as_posix()}'\n", encoding="utf-8")
    hook.chmod(hook.stat().st_mode | 0o755)


def test_a_hook_planted_in_the_shared_clone_runs_for_nobody(tmp_path: Path) -> None:
    """Three layers, one scenario: a worker tries ``echo 'git push' > .git/hooks/pre-commit`` (the
    policy denies the write); had it landed anyway in the clone's shared ``.git/hooks``, the engine's
    worktree commands run with ``core.hooksPath`` = ``providers.unplantable_hooks_path()`` and the
    worker's own git (``worker_env``) does too, so neither the engine's ``worktree add`` nor the
    worker's ``git commit`` runs it."""
    planting = "echo 'git push origin HEAD:main' > .git/hooks/pre-commit"
    assert not classify_shell(planting).allow
    assert not classify_shell("Set-Content .git/hooks/pre-commit 'git push'", powershell=True).allow

    origin = init_bare_repo(tmp_path / "origin.git")
    clone = init_git_repo(tmp_path / "clone", origin=origin)
    hooks = clone / ".git" / "hooks"
    checkout_marker = tmp_path / "post-checkout.ran"
    commit_marker = tmp_path / "pre-commit.ran"
    _plant_hook(hooks, "post-checkout", checkout_marker)
    _plant_hook(hooks, "pre-commit", commit_marker)

    manager = WorktreeManager(tmp_path / "data" / "wt")
    wt = manager.for_new_branch(_Repo(clone), "factory/abc-1")
    assert not checkout_marker.exists(), "post-checkout ran inside the engine's worktree add"

    profile = Profile(name="t", provider=ProviderConfig(kind="api_key"), secrets={"ANTHROPIC_API_KEY": "k"},
                      disclosure_footer="footer")  # fmt: skip
    git_env = {k: v for k, v in worker_env(profile).items() if k.startswith("GIT_")}
    (wt.path / "a.txt").write_text("a\n", encoding="utf-8")
    assert _git(["add", "-A"], wt.path, git_env).returncode == 0
    result = _git(["commit", "-q", "-m", "worker commit"], wt.path, git_env)
    assert result.returncode == 0, result.stderr
    assert not commit_marker.exists(), "pre-commit ran inside the worker's git commit"
    assert manager.new_commits(wt) and manager.changed_files(wt) == ["a.txt"]

    # sanity: both hooks are live on this platform outside those two environments
    _git(["worktree", "add", "-q", "--detach", str(tmp_path / "plain"), "HEAD"], clone)
    assert checkout_marker.is_file()
    (wt.path / "b.txt").write_text("b\n", encoding="utf-8")
    _git(["add", "-A"], wt.path)
    assert _git(["commit", "-q", "-m", "plain commit"], wt.path).returncode == 0
    assert commit_marker.is_file()


def test_concurrent_worktrees_in_one_clone_do_not_race_on_fetch(tmp_path: Path) -> None:
    """Regression: parallel groups on the same repo fetched concurrently in one clone and git failed
    one of them with 'cannot lock ref'. The per-clone lock serialises them."""
    origin = init_bare_repo(tmp_path / "origin.git")
    seed = init_git_repo(tmp_path / "seed", origin=origin)
    for name in ("b1", "b2", "b3"):
        run_git(["checkout", "-q", "-b", name, "main"], seed)
        (seed / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
        run_git(["add", "-A"], seed)
        run_git(["commit", "-q", "-m", name], seed)
        run_git(["push", "-q", "origin", name], seed)
    clone = tmp_path / "clone"
    run_git(["clone", "-q", str(origin), str(clone)], tmp_path)
    manager = WorktreeManager(tmp_path / "wt")
    repo = _Repo(clone)
    errors: list[BaseException] = []
    made: list[Path] = []

    def work(branch: str) -> None:
        try:
            for _ in range(3):
                wt = manager.for_new_branch(repo, branch)
                made.append(wt.path)
                manager.remove(wt)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(b,)) for b in ("b1", "b2", "b3")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not errors, errors
    assert len(made) == 9
    assert run_git(["worktree", "list", "--porcelain"], clone).count("worktree ") == 1
