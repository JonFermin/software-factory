"""Throwaway git worktrees under the profile's data dir.

Every worker runs inside a linked worktree of the user's local clone, so the user's own checkout is
never modified: PR work is detached at the PR head (verified against ``head_ref_oid``), new-branch
work is detached at ``origin/<branch>`` (or ``origin/<default>``) and the engine pushes
``HEAD:refs/heads/<branch>`` -- no local branch is ever created, reset or deleted in the user's
clone (``git worktree add -B`` would silently reset an existing local branch, and a branch left
checked out in a kept worktree would block every later run). Worktrees share the clone's common git
dir, so ``.factory/`` (worker scratch + evidence images) is added to the shared ``info/exclude``.

Clone-level git operations (fetch, worktree add/remove/prune, the exclude file) are serialised per
clone with ``clone_lock``: concurrent fetches in one clone fail with ``cannot lock ref`` and would
turn a healthy action into a "worktree failed" attempt. Other modules that touch a clone from a
worker thread (evidence pushes) should take the same lock.

Every clone-level command also runs with ``-c core.hooksPath=<unplantable path>``
(``providers.unplantable_hooks_path``: a path under a regular file / character device, which git
treats as "no hooks" and under which nothing can ever be created): ``git worktree add`` fires
``post-checkout`` and ``git fetch`` fires ``reference-transaction`` from the clone's shared
``.git/hooks`` (and ``core.hooksPath`` in its config), both writable by a worker's shell, so a
planted hook would otherwise execute with the engine's environment every time a worktree is
prepared. The path is checked when the manager is created and before every command; when it is not
available git does not run (``WorktreeError``).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .providers import HooksPathError, git_no_hooks_args, unplantable_hooks_path
from .sources.models import PullRequest

if TYPE_CHECKING:
    from .repos import RepoInfo

GitRunner = Callable[..., str]
"""``git(args: list[str], *, cwd: Path | None = None) -> str`` (stdout); raises ``WorktreeError``."""

FACTORY_EXCLUDE = ".factory/"
_CONFLICT_RE = re.compile(rb"^(<{7}|>{7}|\|{7})( |$)", re.MULTILINE)
_DEFAULT_TIMEOUT_S = 300
# git's wording when two processes update the same ref (another fetch, the user's own git).
_REF_LOCK_RE = re.compile(r"(cannot lock ref|unable to update local ref|failed to lock|could not lock|\.lock': File exists)", re.IGNORECASE)
FETCH_RETRIES = 2
FETCH_RETRY_DELAY_S = 1.0

_CLONE_LOCKS: dict[str, threading.RLock] = {}
_CLONE_LOCKS_GUARD = threading.Lock()


def clone_lock(clone: Path) -> threading.RLock:
    """Process-wide re-entrant lock for clone-level git operations on ``clone``."""
    key = os.path.normcase(str(Path(clone).resolve())) if Path(clone).exists() else os.path.normcase(str(clone))
    with _CLONE_LOCKS_GUARD:
        lock = _CLONE_LOCKS.get(key)
        if lock is None:
            lock = _CLONE_LOCKS[key] = threading.RLock()
        return lock


class WorktreeError(RuntimeError):
    pass


def hooks_disabled_args() -> list[str]:
    """``["-c", "core.hooksPath=<unplantable path>"]`` for an engine git command in a clone or
    worktree; ``WorktreeError`` (git must not run) when hooks cannot be disabled on this host."""
    try:
        return git_no_hooks_args()
    except HooksPathError as exc:
        raise WorktreeError(f"refusing to run git with hooks enabled: {exc}") from exc


class Worktree(BaseModel):
    path: Path
    repo: str
    clone: Path
    base_sha: str
    branch: str | None = None
    detached: bool = True


def default_git(args: list[str], *, cwd: Path | None = None, timeout: float = _DEFAULT_TIMEOUT_S) -> str:
    """Run ``git <args>`` without prompts; return stdout; raise ``WorktreeError`` on failure."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"}
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        raise WorktreeError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise WorktreeError(f"git {' '.join(args)} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise WorktreeError(f"git {' '.join(args)} failed ({proc.returncode}): {detail}")
    return proc.stdout


def _short_hash(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


def _rmtree(path: Path) -> None:
    """``shutil.rmtree`` that copes with read-only git objects on Windows."""

    def onexc(func: Any, p: str, _exc: BaseException) -> None:
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onexc=onexc)


class WorktreeManager:
    def __init__(self, root: Path, git: GitRunner | None = None):
        self.root = Path(root)
        self._git = git or default_git
        # Startup sanity check: a host where hooks cannot be disabled gets no manager (fail closed).
        try:
            self.hooks_path = unplantable_hooks_path()
        except HooksPathError as exc:
            raise WorktreeError(f"refusing to run git with hooks enabled: {exc}") from exc

    # ------------------------------------------------------------------------------ helpers
    def git(self, args: list[str], *, cwd: Path | None = None) -> str:
        return self._git(args, cwd=cwd)

    def no_hooks_args(self) -> list[str]:
        """``-c core.hooksPath=<unplantable path>`` for a clone-level git command (re-checked on
        every call; ``WorktreeError`` when hooks cannot be disabled)."""
        return hooks_disabled_args()

    def _clone_git(self, args: list[str], clone: Path) -> str:
        """Run ``git <args>`` in the clone with hooks disabled."""
        return self.git([*self.no_hooks_args(), *args], cwd=clone)

    def _new_path(self, repo: RepoInfo, tag: str, target: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = str(time.time_ns())
        for attempt in range(5):
            digest = _short_hash(repo.slug, target, stamp, str(attempt))
            path = self.root / f"{tag}-{digest}"
            if not path.exists():
                return path
        raise WorktreeError(f"could not pick a free worktree path under {self.root}")

    def _add_exclude(self, wt_path: Path) -> None:
        raw = self.git(["rev-parse", "--git-path", "info/exclude"], cwd=wt_path).strip()
        exclude = Path(raw)
        if not exclude.is_absolute():
            exclude = wt_path / exclude
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
        if FACTORY_EXCLUDE in {line.strip() for line in existing.splitlines()}:
            return
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        with exclude.open("a", encoding="utf-8") as fh:
            fh.write(f"{prefix}{FACTORY_EXCLUDE}\n")

    def _origin_ref_exists(self, clone: Path, ref: str) -> bool:
        try:
            self._clone_git(["rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{ref}^{{commit}}"], clone)
            return True
        except WorktreeError:
            return False

    def _fetch(self, clone: Path, *refspecs: str, sleep: Callable[[float], None] = time.sleep) -> None:
        """``git fetch --no-tags origin <refspecs>``; a ref-lock race (another process updating the
        same remote-tracking ref) is retried a couple of times instead of failing the action."""
        for attempt in range(FETCH_RETRIES + 1):
            try:
                self._clone_git(["fetch", "--no-tags", "origin", *refspecs], clone)
                return
            except WorktreeError as exc:
                if attempt >= FETCH_RETRIES or not _REF_LOCK_RE.search(str(exc)):
                    raise
                sleep(FETCH_RETRY_DELAY_S * (attempt + 1))

    def _finish(self, wt: Worktree, expected_head: str | None = None) -> Worktree:
        head = self.head(wt)
        if expected_head is not None and head != expected_head:
            self.remove(wt)
            raise WorktreeError(
                f"worktree HEAD {head} does not match expected {expected_head} for {wt.repo}"
            )
        self._add_exclude(wt.path)
        return wt

    # ------------------------------------------------------------------------------ creation
    def for_pr(self, repo: RepoInfo, pr: PullRequest) -> Worktree:
        """Detached worktree at ``pr.head_ref_oid`` (fetched via ``pull/<n>/head`` for forks)."""
        oid = (pr.head_ref_oid or "").strip()
        if not oid:
            raise WorktreeError(f"{pr.target}: head_ref_oid is empty")
        clone = Path(repo.path)
        refspec = f"pull/{pr.number}/head" if pr.is_fork else pr.head_ref_name
        if not refspec:
            raise WorktreeError(f"{pr.target}: no head branch to fetch")
        with clone_lock(clone):
            self._fetch(clone, refspec)
            path = self._new_path(repo, f"pr{pr.number}", f"{pr.target}@{oid}")
            self._clone_git(["worktree", "add", "--detach", str(path), oid], clone)
            wt = Worktree(path=path, repo=repo.slug, clone=clone, base_sha=oid, branch=None, detached=True)
            return self._finish(wt, expected_head=oid)

    def for_new_branch(self, repo: RepoInfo, branch: str) -> Worktree:
        """Detached worktree for work the engine will push as ``HEAD:refs/heads/<branch>``.

        Starts at ``origin/<branch>`` when the branch exists on origin, else ``origin/<default>``.
        ``wt.branch`` records the push target; the user's clone gets no local branch of that name
        (a pre-existing local branch, pushed or not, is left exactly as it is).
        """
        if not branch or branch.startswith("-") or any(c in branch for c in " ~^:?*[\\"):
            raise WorktreeError(f"invalid branch name {branch!r}")
        if branch == repo.default_branch:
            raise WorktreeError(f"refusing to prepare work for the default branch {branch!r}")
        clone = Path(repo.path)
        with clone_lock(clone):
            self._fetch(clone)
            start = f"origin/{branch}" if self._origin_ref_exists(clone, branch) else f"origin/{repo.default_branch}"
            base_sha = self._clone_git(["rev-parse", f"{start}^{{commit}}"], clone).strip()
            path = self._new_path(repo, "br", f"{branch}@{base_sha}")
            self._clone_git(["worktree", "add", "--detach", str(path), base_sha], clone)
            wt = Worktree(path=path, repo=repo.slug, clone=clone, base_sha=base_sha, branch=branch, detached=True)
            return self._finish(wt, expected_head=base_sha)

    def for_default_branch(self, repo: RepoInfo) -> Worktree:
        """Detached worktree at ``origin/<default>`` (readiness runs, refine/plan reading)."""
        clone = Path(repo.path)
        with clone_lock(clone):
            self._fetch(clone, repo.default_branch)
            start = f"origin/{repo.default_branch}"
            base_sha = self._clone_git(["rev-parse", f"{start}^{{commit}}"], clone).strip()
            path = self._new_path(repo, "def", f"{start}@{base_sha}")
            self._clone_git(["worktree", "add", "--detach", str(path), base_sha], clone)
            wt = Worktree(path=path, repo=repo.slug, clone=clone, base_sha=base_sha, branch=None, detached=True)
            return self._finish(wt, expected_head=base_sha)

    # ----------------------------------------------------------------------------- inspection
    def head(self, wt: Worktree) -> str:
        return self.git(["rev-parse", "HEAD"], cwd=wt.path).strip()

    def new_commits(self, wt: Worktree) -> list[str]:
        """Commit SHAs in ``base_sha..HEAD``, oldest first."""
        out = self.git(["rev-list", "--reverse", f"{wt.base_sha}..HEAD"], cwd=wt.path)
        return [line.strip() for line in out.splitlines() if line.strip()]

    def changed_files(self, wt: Worktree) -> list[str]:
        """Paths changed in ``base_sha..HEAD`` (name-only, forward slashes)."""
        out = self.git(["diff", "--name-only", f"{wt.base_sha}..HEAD"], cwd=wt.path)
        return [line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()]

    def _dirty_files(self, wt: Worktree) -> list[str]:
        out = self.git(["status", "--porcelain", "--untracked-files=all"], cwd=wt.path)
        files: list[str] = []
        for line in out.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            files.append(path.strip('"'))
        return files

    def has_conflict_markers(self, wt: Worktree) -> bool:
        """True when any changed (committed or dirty) text file contains ``<<<<<<<``/``>>>>>>>`` markers."""
        candidates = dict.fromkeys(self.changed_files(wt) + self._dirty_files(wt))
        for rel in candidates:
            path = wt.path / rel
            if not path.is_file():
                continue
            try:
                with path.open("rb") as fh:
                    data = fh.read(4 * 1024 * 1024)
            except OSError:
                continue
            if b"\0" in data[:8192]:
                continue  # binary
            if _CONFLICT_RE.search(data):
                return True
        return False

    # -------------------------------------------------------------------------------- removal
    def remove(self, wt: Worktree) -> None:
        self._remove_path(wt.path, wt.clone)

    def _remove_path(self, path: Path, clone: Path | None) -> None:
        if clone is None:
            clone = _clone_of(path)
        if clone is None:
            if path.exists():
                _rmtree(path)
            return
        with clone_lock(clone):
            removed = False
            if path.exists():
                try:
                    self._clone_git(["worktree", "remove", "--force", str(path)], clone)
                    removed = True
                except WorktreeError:
                    removed = False
            if not removed and path.exists():
                _rmtree(path)
            with contextlib.suppress(WorktreeError):
                self._clone_git(["worktree", "prune"], clone)

    def cleanup(self, keep_paths: set[str]) -> list[Path]:
        """Remove every worktree under ``root`` whose path is not in ``keep_paths``."""
        if not self.root.is_dir():
            return []
        keep = {os.path.normcase(str(Path(p).resolve())) for p in keep_paths}
        removed: list[Path] = []
        for child in sorted(self.root.iterdir()):
            if os.path.normcase(str(child.resolve())) in keep:
                continue
            self._remove_path(child, None)
            removed.append(child)
        return removed


def _clone_of(path: Path) -> Path | None:
    """Find the clone a linked worktree belongs to via its ``.git`` file (``gitdir: <clone>/.git/worktrees/x``)."""
    dotgit = path / ".git"
    if not dotgit.is_file():
        return None
    try:
        text = dotgit.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text[len("gitdir:") :].strip())
    # <clone>/.git/worktrees/<name> -> <clone>
    if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
        return gitdir.parent.parent.parent
    return None
