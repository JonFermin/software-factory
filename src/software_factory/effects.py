"""Effects: every outward write the factory performs, in one place (architecture invariant 2).

Workers propose, the engine acts. The kind handlers never touch GitHub, git remotes or the
tracker directly; they call an ``Effects`` implementation:

* ``LiveEffects`` performs the write (``git push`` with the one safe refspec, ``gh pr create
  --draft``, PR evidence edits on factory-opened PRs only, the evidence branch, tracker writes) and
  appends an audit record for each (``effect.<name>``).
* ``RecordingEffects`` (``mode = "observe"`` and ``--dry-run``) writes a ``would_<name>`` audit
  record with the arguments and returns placeholders. It never calls a ``gh`` or tracker write and
  never runs ``git push``; tests prove it with fakes that raise on any write.

``write_review`` is a *local* file write and is allowed in every mode (the review draft is never
posted). Push refspecs are built here as ``HEAD:refs/heads/<branch>`` -- never ``+``, never
``--force*``, never ``--no-verify`` -- and the pushed commit SHAs are recorded in state so metrics
can tell factory commits from human rework.

Every engine git command here (pushes, the evidence branch's fetch and plumbing, the LFS upload)
runs with ``-c core.hooksPath=<unplantable path>`` (``providers.unplantable_hooks_path``): the hooks
of a linked worktree live in the user's clone's shared ``.git/hooks`` (and ``core.hooksPath`` in the
shared config), both writable by a worker's shell, so a planted ``pre-push`` /
``reference-transaction`` would otherwise execute inside the engine's authenticated git (invariant
2). The path lies under a regular file (Windows: the hosts file; POSIX: ``/dev/null``), so no
process can ever plant a hook there; when it is not available the command is refused
(``EffectError``). The recorded commit SHAs follow the first-parent line of the worktree, so a
merge of the base branch records the merge commit, not every base commit it brought in.

Hooks disabled means git-lfs's ``pre-push`` never uploads LFS objects either: ``push`` uploads them
itself (``git lfs push origin <HEAD sha>``) when the pushed change set touches LFS-tracked paths,
and refuses the push (``EffectError``) when that is not possible -- LFS pointers are never pushed
without their objects.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

from .config import Profile
from .evidence import EVIDENCE_BRANCH, evidence_url, upsert_evidence
from .notify import Audit
from .paths import ProfilePaths
from .policy import may_edit_pr_body
from .providers import HooksPathError, git_no_hooks_args
from .sources.models import PullRequest
from .state import State
from .trackers.base import Issue, TrackerAuthError, TrackerError
from .workers.schemas import SubIssueSpec
from .worktree import Worktree

if TYPE_CHECKING:
    from .repos import RepoInfo
    from .sources.github import GitHub
    from .trackers.base import Tracker

GitRunner = Callable[..., str]
"""``git(args, *, cwd=None, env=None) -> stdout``; raises on a non-zero exit. ``env`` holds extra
variables (the temporary ``GIT_INDEX_FILE`` for the evidence branch)."""

GIT_TIMEOUT_S = 300.0
REPO_MARKER = "<!-- factory:repo {slug} -->"
LFS_FILTER = "lfs"  # ``filter`` attribute value of Git LFS tracked paths
CHECK_ATTR_CHUNK = 200  # paths per ``git check-attr`` call (Windows command-line length)
_REJECTED_MARKERS = ("non-fast-forward", "fetch first", "[rejected]", "failed to push some refs", "stale info")
# git's own credential failures (not covered by ``gh auth status``): invariant 5 treats them as an
# auth stop, never as a per-PR "push failed" attempt.
_GIT_AUTH_MARKERS = (
    "authentication failed",
    "could not read username",
    "could not read password",
    "terminal prompts disabled",
    "permission denied (publickey)",
    "invalid username or password",
    "invalid username or token",
    "invalid credentials",
    "bad credentials",
    "token has expired",
    "returned error: 401",
    "returned error: 403",
    "http 401",
    "http 403",
    "permission to ",
)
_BRANCH_FORBIDDEN = re.compile(r"[\s~^:?*\[\\]|\.\.|@\{|//|^/|/$|\.lock$|^\.|/\.|\.$")


class EffectError(RuntimeError):
    pass


class PushRejected(EffectError):
    """The remote refused the push (someone pushed to the branch meanwhile). Never force: the caller
    keeps the worktree and flags the human."""


class EffectRefused(EffectError):
    """Policy refused the effect (e.g. editing a PR the factory did not open)."""


class GitCommandError(EffectError):
    def __init__(self, args: list[str], returncode: int, stderr: str) -> None:
        self.args_list = list(args)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"git {' '.join(args)} failed ({returncode}): {stderr.strip()}")


# ---------------------------------------------------------------------------------- git runner
def default_git_runner(
    args: list[str],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = GIT_TIMEOUT_S,
) -> str:
    """``git <args>`` with list args (never a shell), no prompts, utf-8, explicit timeout."""
    full_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"}
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=full_env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise EffectError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise EffectError(f"git {' '.join(args)} timed out after {timeout:g}s") from exc
    if proc.returncode != 0:
        raise GitCommandError(args, proc.returncode, proc.stderr or proc.stdout or "")
    return proc.stdout


def _clone_lock(clone: Path) -> Any:
    """The worktree manager's per-clone lock when the module provides one, else a no-op."""
    from . import worktree as worktree_mod

    lock = getattr(worktree_mod, "clone_lock", None)
    return lock(clone) if callable(lock) else contextlib.nullcontext()


def validate_branch(branch: str) -> str:
    """A plain branch name usable in ``HEAD:refs/heads/<branch>``: no refspec syntax, no ``+``,
    no leading ``-`` (option injection), no ``refs/`` prefix."""
    name = (branch or "").strip()
    if not name or name.startswith(("-", "+", "refs/")) or _BRANCH_FORBIDDEN.search(name):
        raise ValueError(f"invalid branch name for push: {branch!r}")
    return name


def push_refspec(branch: str) -> str:
    """The only refspec the factory ever pushes."""
    return f"HEAD:refs/heads/{validate_branch(branch)}"


def is_rejected_push(message: str) -> bool:
    low = (message or "").lower()
    return any(marker in low for marker in _REJECTED_MARKERS)


def is_auth_failure(message: str) -> bool:
    """True when a git error message is a credential/permission failure (``git push`` with an
    expired PAT, a credential helper that cannot prompt, a key the remote refuses)."""
    low = (message or "").lower()
    return any(marker in low for marker in _GIT_AUTH_MARKERS)


def link_pushes(state: State, repo: str, branch: str, number: int) -> int:
    """Attach the ``pushes`` rows recorded for ``(repo, branch)`` before the PR existed (``number``
    NULL: build/ready push the branch first, then open the PR) to ``number``, so ``pushed_shas`` /
    ``touched_prs`` and the metrics see the factory's own commits on factory-opened PRs."""
    return int(state.link_pushes(repo, branch, number) or 0)


def validate_asset_path(dest: str) -> str:
    """Destination on the evidence branch: relative, POSIX, no ``..`` / absolute / drive segments."""
    text = (dest or "").strip().replace("\\", "/")
    pure = PurePosixPath(text)
    parts = [p for p in pure.parts if p not in ("", ".")]
    if not text or text.startswith("/") or ".." in parts or ":" in text or not parts:
        raise ValueError(f"invalid evidence asset path: {dest!r}")
    return "/".join(parts)


# ------------------------------------------------------------------------------------ protocol
class Effects(Protocol):
    def push(self, wt: Worktree, branch: str, *, kind: str, pr: int | None) -> str: ...

    def open_draft_pr(
        self, repo: str, branch: str, base: str, title: str, body: str, *, kind: str, issue_key: str | None
    ) -> PullRequest: ...

    def update_pr_evidence(self, repo: str, number: int, section: str) -> None: ...

    def push_evidence_assets(self, repo: RepoInfo, files: list[tuple[Path, str]]) -> list[str]: ...

    def transition_issue(self, issue_id: str, state_type: str) -> None: ...

    def create_sub_issues(self, parent: Issue, specs: list[SubIssueSpec]) -> list[Issue]: ...

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None: ...

    def remove_label(self, issue_id: str, label: str) -> None: ...

    def write_review(self, repo: str, number: int, markdown: str) -> Path: ...


class _Base:
    """Shared constructor + the local-only effects (identical in both modes)."""

    def __init__(
        self,
        profile: Profile,
        paths: ProfilePaths,
        state: State,
        gh: GitHub | None,
        tracker: Tracker | None,
        audit: Audit,
        git_runner: GitRunner | None = None,
    ) -> None:
        self.profile = profile
        self.paths = paths
        self.state = state
        self.gh = gh
        self.tracker = tracker
        self.audit = audit
        self._git: GitRunner = git_runner or default_git_runner

    # ------------------------------------------------------------------------- local-only
    def write_review(self, repo: str, number: int, markdown: str) -> Path:
        """Review draft to ``reviews/<repo-name>-<pr>.md``. Local, allowed in every mode."""
        path = self.paths.review_file(repo.split("/")[-1], number)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = markdown if markdown.endswith("\n") else markdown + "\n"
        path.write_text(text, encoding="utf-8")
        self.audit.write("effect.write_review", repo=repo, pr=number, path=str(path), chars=len(text))
        return path

    # ----------------------------------------------------------------------------- helpers
    def no_hooks_args(self) -> list[str]:
        """``-c core.hooksPath=<unplantable path>``: no git hook runs inside an engine command.

        Checked on every call; when hooks cannot be disabled on this host the command is refused
        (``EffectError``): fail closed."""
        try:
            return git_no_hooks_args()
        except HooksPathError as exc:
            raise EffectError(f"refusing to run git with hooks enabled: {exc}") from exc

    def _engine_git(self, args: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
        """``git -c core.hooksPath=<unplantable> <args>`` in ``cwd`` (a worktree or the user's clone)."""
        full = [*self.no_hooks_args(), *args]
        return self._git(full, cwd=cwd) if env is None else self._git(full, cwd=cwd, env=env)

    def _require_tracker(self) -> Tracker:
        if self.tracker is None:
            raise EffectError("no tracker configured for this profile (linear.enabled / LINEAR_API_KEY)")
        return self.tracker

    def _require_gh(self) -> GitHub:
        if self.gh is None:
            raise EffectError("no GitHub client configured")
        return self.gh

    def _sub_issue_description(self, spec: SubIssueSpec, label_exists: bool) -> str:
        parts = [spec.description.strip()]
        if spec.acceptance_criteria:
            parts.append(
                "## Acceptance criteria\n\n"
                + "\n".join(f"{i}. {ac.strip()}" for i, ac in enumerate(spec.acceptance_criteria, 1))
            )
        if not label_exists:
            parts.append(REPO_MARKER.format(slug=spec.repo))
        return "\n\n".join(p for p in parts if p)

    def _repo_label(self, slug: str) -> str:
        return f"{self.profile.linear.repo_label_prefix}{slug.split('/')[-1]}"


# ---------------------------------------------------------------------------------------- live
class LiveEffects(_Base):
    """Performs the writes. Every method appends an ``effect.<name>`` audit record."""

    # -------------------------------------------------------------------------------- push
    def push(self, wt: Worktree, branch: str, *, kind: str, pr: int | None) -> str:
        """``git push origin HEAD:refs/heads/<branch>`` from the worktree. Returns the pushed sha.

        Raises ``PushRejected`` on a non-fast-forward rejection (never retried with force). Every
        git runs with hooks disabled (``no_hooks_args``): the worktree's hooks are the clone's shared
        ones. When the change set touches Git LFS paths their objects are uploaded first
        (``_push_lfs_objects``); a failed upload refuses the push (``EffectError``).
        """
        refspec = push_refspec(branch)
        name = validate_branch(branch)
        before = self._remote_sha(wt.path, name)
        head = self._engine_git(["rev-parse", "HEAD"], wt.path).strip()
        commit_shas = self._rev_list(wt)
        self._push_lfs_objects(wt, head, branch=name, kind=kind, pr=pr)
        try:
            self._engine_git(["push", "origin", refspec], wt.path)
        except Exception as exc:
            if is_rejected_push(str(exc)):
                self.audit.write(
                    "effect.push_rejected", repo=wt.repo, pr=pr, branch=name, kind=kind,
                    before_sha=before, local_sha=head, error=str(exc)[:500],
                )  # fmt: skip
                raise PushRejected(
                    f"push of {wt.repo} HEAD:refs/heads/{name} rejected (remote at {before or 'none'} "
                    f"moved; non-fast-forward): {exc}"
                ) from exc
            raise
        self.state.record_push(wt.repo, pr, name, kind, before, head, commit_shas)
        self.audit.write(
            "effect.push", repo=wt.repo, pr=pr, branch=name, kind=kind, refspec=refspec,
            before_sha=before, after_sha=head, commits=commit_shas,
        )  # fmt: skip
        return head

    def _push_lfs_objects(self, wt: Worktree, head: str, *, branch: str, kind: str, pr: int | None) -> None:
        """``git lfs push origin <head>`` when the change set has LFS-tracked paths.

        Engine pushes run with hooks disabled, so git-lfs's own ``pre-push`` never uploads the
        objects; pushing the ref alone would publish pointers whose objects exist only on this
        machine. git-lfs missing, or an upload that fails, refuses the push (``EffectError``: the
        caller keeps the worktree and flags it). ``lfs push`` has no force mode."""
        lfs_paths = self._lfs_paths(wt)
        if not lfs_paths:
            return
        try:
            self._engine_git(["lfs", "push", "origin", head], wt.path)
        except Exception as exc:
            self.audit.write(
                "effect.push_refused_lfs", repo=wt.repo, pr=pr, branch=branch, kind=kind, local_sha=head,
                lfs_files=lfs_paths[:20], error=str(exc)[:500],
            )  # fmt: skip
            raise EffectError(
                f"refusing to push {wt.repo} HEAD:refs/heads/{branch}: it changes Git LFS files "
                f"({', '.join(lfs_paths[:5])}) and `git lfs push` failed (is git-lfs installed?): {exc}"
            ) from exc
        self.audit.write(
            "effect.push_lfs", repo=wt.repo, pr=pr, branch=branch, kind=kind, sha=head, lfs_files=lfs_paths[:20]
        )

    def _lfs_paths(self, wt: Worktree) -> list[str]:
        """Paths added or modified in ``base_sha..HEAD`` whose ``filter`` attribute is ``lfs``
        (``git check-attr``: ``.gitattributes`` at every level plus ``info/attributes``). Not being
        able to tell refuses the push (``EffectError``)."""
        try:
            out = self._engine_git(
                ["diff", "--name-only", "-z", "--no-renames", "--diff-filter=d", f"{wt.base_sha}..HEAD"], wt.path
            )
        except Exception as exc:
            raise EffectError(f"refusing to push {wt.repo}: cannot list the changed files for the LFS check: {exc}") from exc
        paths = [path for path in out.split("\0") if path]
        found: list[str] = []
        for start in range(0, len(paths), CHECK_ATTR_CHUNK):
            chunk = paths[start : start + CHECK_ATTR_CHUNK]
            try:
                attrs = self._engine_git(["check-attr", "-z", "filter", "--", *chunk], wt.path)
            except Exception as exc:
                raise EffectError(f"refusing to push {wt.repo}: cannot read the LFS attributes: {exc}") from exc
            fields = attrs.split("\0")
            for i in range(0, len(fields) - 2, 3):
                path, attribute, value = fields[i : i + 3]
                if attribute == "filter" and value == LFS_FILTER:
                    found.append(path)
        return found

    def _remote_sha(self, cwd: Path, branch: str) -> str:
        out = self._engine_git(["ls-remote", "--heads", "origin", f"refs/heads/{branch}"], cwd)
        for line in out.splitlines():
            sha, _, ref = line.strip().partition("\t")
            if ref.strip() == f"refs/heads/{branch}" and sha:
                return sha.strip()
        return ""

    def _rev_list(self, wt: Worktree) -> list[str]:
        """The worker's own commits, oldest first: the first-parent line from HEAD down to
        ``base_sha`` (a merge of the base branch contributes the merge commit only)."""
        try:
            out = self._engine_git(["rev-list", "--reverse", "--first-parent", f"{wt.base_sha}..HEAD"], wt.path)
        except Exception:
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]

    # ------------------------------------------------------------------------------- PRs
    def open_draft_pr(
        self, repo: str, branch: str, base: str, title: str, body: str, *, kind: str, issue_key: str | None
    ) -> PullRequest:
        gh = self._require_gh()
        pr = gh.create_draft_pr(repo, head=validate_branch(branch), base=base, title=title, body=body)
        self.state.register_pr(repo, pr.number, branch, kind, issue_key, pr.url)
        linked = link_pushes(self.state, repo, branch, pr.number)
        self.audit.write(
            "effect.open_draft_pr", repo=repo, pr=pr.number, url=pr.url, branch=branch, base=base,
            title=title, kind=kind, issue_key=issue_key, linked_pushes=linked,
        )  # fmt: skip
        return pr

    def update_pr_evidence(self, repo: str, number: int, section: str) -> None:
        """Replace the evidence block of a PR the factory opened; refuses every other PR."""
        if not may_edit_pr_body(self.state, repo, number):
            raise EffectRefused(f"{repo}#{number} was not opened by the factory; its body is not editable")
        gh = self._require_gh()
        body = gh.pr_body(repo, number)
        new_body = upsert_evidence(body, section)
        changed = new_body != body
        if changed:
            gh.edit_pr_body(repo, number, new_body)
        self.audit.write("effect.update_pr_evidence", repo=repo, pr=number, changed=changed, chars=len(section))

    # ------------------------------------------------------------------- evidence branch
    def push_evidence_assets(self, repo: RepoInfo, files: list[tuple[Path, str]]) -> list[str]:
        """Commit ``files`` onto ``origin/factory/evidence`` (orphan root when absent) using a
        temporary index, then push ``<sha>:refs/heads/factory/evidence``. The user's working tree
        and index are never touched. Returns one raw URL per file."""
        if not files:
            return []
        clone = Path(repo.path)
        dests = [(Path(local), validate_asset_path(dest)) for local, dest in files]
        for local, _dest in dests:
            if not local.is_file():
                raise EffectError(f"evidence asset not found: {local}")
        with _clone_lock(clone):  # serialised with the worktree manager's fetches in this clone
            parent = self._fetch_evidence_parent(clone)
            sha = self._build_evidence_commit(clone, parent, dests)
            try:
                self._push_sha(clone, sha)
            except PushRejected:
                parent = self._fetch_evidence_parent(clone)
                sha = self._build_evidence_commit(clone, parent, dests)
                self._push_sha(clone, sha)
        urls = [evidence_url(repo.slug, dest) for _local, dest in dests]
        self.audit.write(
            "effect.push_evidence_assets", repo=repo.slug, branch=EVIDENCE_BRANCH, parent=parent or "",
            sha=sha, files=[dest for _l, dest in dests],
        )  # fmt: skip
        return urls

    def _fetch_evidence_parent(self, clone: Path) -> str | None:
        """The remote tip of the evidence branch, with its objects fetched into the clone. The sha
        comes from ``ls-remote``, never from ``FETCH_HEAD``: other groups fetch PR branches in the
        same clone concurrently and rewrite ``FETCH_HEAD`` between two commands. The fetch runs with
        hooks disabled like every engine git: it updates ``refs/remotes/origin/factory/evidence`` and
        would fire the clone's ``reference-transaction`` hook otherwise."""
        for _attempt in range(2):
            remote = self._remote_sha(clone, EVIDENCE_BRANCH)
            if not remote:
                return None
            self._engine_git(["fetch", "--no-tags", "origin", f"refs/heads/{EVIDENCE_BRANCH}"], clone)
            try:
                self._engine_git(["cat-file", "-e", f"{remote}^{{commit}}"], clone)
                return remote
            except Exception:  # noqa: BLE001 - the branch moved (rewritten) between the two calls
                continue
        raise EffectError(f"could not fetch the tip of {EVIDENCE_BRANCH} from origin")

    def _build_evidence_commit(self, clone: Path, parent: str | None, dests: list[tuple[Path, str]]) -> str:
        tmp = Path(tempfile.mkdtemp(prefix="sf-evidence-"))
        env = {"GIT_INDEX_FILE": str(tmp / "index")}
        try:
            if parent:
                self._engine_git(["read-tree", parent], clone, env)
            else:
                self._engine_git(["read-tree", "--empty"], clone, env)
            for local, dest in dests:
                # --no-filters: evidence assets (images, logs) are stored byte for byte, whatever
                # the clone's autocrlf / attributes would otherwise do to them.
                blob = self._engine_git(["hash-object", "-w", "--no-filters", str(local)], clone).strip()
                self._engine_git(["update-index", "--add", "--cacheinfo", f"100644,{blob},{dest}"], clone, env)
            tree = self._engine_git(["write-tree"], clone, env).strip()
            message = f"factory evidence: {len(dests)} file(s)\n\n" + "\n".join(d for _l, d in dests)
            args = ["commit-tree", tree, "-m", message]
            if parent:
                args += ["-p", parent]
            return self._engine_git(args, clone, env).strip()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _push_sha(self, clone: Path, sha: str) -> None:
        try:
            self._engine_git(["push", "origin", f"{sha}:refs/heads/{EVIDENCE_BRANCH}"], clone)
        except Exception as exc:
            if is_rejected_push(str(exc)):
                raise PushRejected(f"push of {EVIDENCE_BRANCH} rejected (non-fast-forward): {exc}") from exc
            raise

    # ----------------------------------------------------------------------------- tracker
    def transition_issue(self, issue_id: str, state_type: str) -> None:
        self._require_tracker().move_to_state_type(issue_id, state_type)  # type: ignore[arg-type]
        self.audit.write("effect.transition_issue", issue_id=issue_id, state_type=state_type)

    def create_sub_issues(self, parent: Issue, specs: list[SubIssueSpec]) -> list[Issue]:
        """Children carry ``repo:<name>`` when that label exists, else the ``factory:repo`` marker
        in their description; ``blocked_by`` indices become ``blocks`` relations. Never adds the
        build label (the human approves each child by labelling it)."""
        tracker = self._require_tracker()
        labels = {spec.repo: self._repo_label(spec.repo) for spec in specs}
        existing = self._labels_usable_by(tracker, parent, sorted(set(labels.values())))
        children: list[Issue] = []
        for spec in specs:
            label = labels[spec.repo]
            has_label = bool(existing.get(label))
            child = tracker.create_sub_issue(
                parent, spec.title, self._sub_issue_description(spec, has_label), [label] if has_label else []
            )
            children.append(child)
            applied = any(lbl.lower() == label.lower() for lbl in child.labels)
            self.audit.write(
                "effect.create_sub_issue", parent=parent.key, issue_id=child.id, key=child.key,
                title=spec.title, repo=spec.repo, labels=child.labels,
                label_dropped=(has_label and not applied) or None,
            )  # fmt: skip
        for index, spec in enumerate(specs):
            for blocker in spec.blocked_by:
                if not (0 <= blocker < len(children)) or blocker == index:
                    continue
                tracker.add_blocked_by(children[index].id, children[blocker].id)
                self.audit.write(
                    "effect.add_blocked_by", issue=children[index].key, blocker=children[blocker].key
                )
        return children

    def _labels_usable_by(self, tracker: Tracker, parent: Issue, names: list[str]) -> dict[str, bool]:
        """Which of ``names`` the parent's *team* can apply. ``labels_exist`` is workspace-wide, so a
        ``repo:<name>`` label owned by another team would count as present while
        ``create_sub_issue`` (team-scoped) silently drops it -- and the child would get neither the
        label nor the description marker. ``ensure_label`` (a lookup) decides per team when the
        tracker offers it and the parent has a team id."""
        if not names:
            return {}
        lookup = getattr(tracker, "ensure_label", None)
        if not callable(lookup) or not parent.team_id:
            return tracker.labels_exist(names)
        usable: dict[str, bool] = {}
        for name in names:
            try:
                usable[name] = bool(lookup(parent.team_id, name))
            except TrackerAuthError:
                raise
            except TrackerError:
                usable[name] = False
        return usable

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None:
        self._require_tracker().update_issue_text(issue_id, title, description)
        self.audit.write("effect.update_issue_text", issue_id=issue_id, title=title, chars=len(description))

    def remove_label(self, issue_id: str, label: str) -> None:
        self._require_tracker().remove_label(issue_id, label)
        self.audit.write("effect.remove_label", issue_id=issue_id, label=label)


# ----------------------------------------------------------------------------------- recording
class RecordingEffects(_Base):
    """Observe mode / ``--dry-run``: audit ``would_<effect>`` and return placeholders. No ``gh`` or
    tracker write, no ``git push``, no state writes."""

    def push(self, wt: Worktree, branch: str, *, kind: str, pr: int | None) -> str:
        refspec = push_refspec(branch)
        self.audit.write("would_push", repo=wt.repo, pr=pr, branch=branch, kind=kind, refspec=refspec, worktree=str(wt.path))
        return "dry-run"

    def open_draft_pr(
        self, repo: str, branch: str, base: str, title: str, body: str, *, kind: str, issue_key: str | None
    ) -> PullRequest:
        self.audit.write(
            "would_open_draft_pr", repo=repo, branch=branch, base=base, title=title, kind=kind,
            issue_key=issue_key, body_chars=len(body),
        )  # fmt: skip
        return PullRequest(
            repo=repo, number=0, title=title, url=f"dry-run://{repo}/pull/{branch}", body=body,
            is_draft=True, head_ref_name=branch, base_ref_name=base,
        )  # fmt: skip

    def update_pr_evidence(self, repo: str, number: int, section: str) -> None:
        if not may_edit_pr_body(self.state, repo, number):
            raise EffectRefused(f"{repo}#{number} was not opened by the factory; its body is not editable")
        self.audit.write("would_update_pr_evidence", repo=repo, pr=number, chars=len(section))

    def push_evidence_assets(self, repo: RepoInfo, files: list[tuple[Path, str]]) -> list[str]:
        dests = [validate_asset_path(dest) for _local, dest in files]
        self.audit.write("would_push_evidence_assets", repo=repo.slug, branch=EVIDENCE_BRANCH, files=dests)
        return [evidence_url(repo.slug, dest) for dest in dests]

    def transition_issue(self, issue_id: str, state_type: str) -> None:
        self.audit.write("would_transition_issue", issue_id=issue_id, state_type=state_type)

    def create_sub_issues(self, parent: Issue, specs: list[SubIssueSpec]) -> list[Issue]:
        self.audit.write(
            "would_create_sub_issues", parent=parent.key,
            sub_issues=[{"title": s.title, "repo": s.repo, "blocked_by": list(s.blocked_by)} for s in specs],
        )  # fmt: skip
        return [
            Issue(
                id=f"dry-run:{parent.id}:{i}", key=f"{parent.key}-dry{i}", title=spec.title,
                description=self._sub_issue_description(spec, label_exists=False),
                labels=[self._repo_label(spec.repo)], state_type="backlog", team_id=parent.team_id,
                team_key=parent.team_key, project_id=parent.project_id, project_name=parent.project_name,
            )  # fmt: skip
            for i, spec in enumerate(specs)
        ]

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None:
        self.audit.write("would_update_issue_text", issue_id=issue_id, title=title, chars=len(description))

    def remove_label(self, issue_id: str, label: str) -> None:
        self.audit.write("would_remove_label", issue_id=issue_id, label=label)


# ------------------------------------------------------------------------------------- factory
def make_effects(
    profile: Profile,
    *,
    dry_run: bool,
    paths: ProfilePaths,
    state: State,
    gh: GitHub | None = None,
    tracker: Tracker | None = None,
    audit: Audit | None = None,
    git_runner: GitRunner | None = None,
) -> Effects:
    """``LiveEffects`` only in ``mode = "full"`` without ``--dry-run``; ``RecordingEffects`` otherwise."""
    audit = audit or Audit(paths.audit_log, profile.name)
    cls: type[_Base] = LiveEffects if (profile.mode == "full" and not dry_run) else RecordingEffects
    return cls(profile, paths, state, gh, tracker, audit, git_runner)  # type: ignore[return-value]


def is_recording(effects: Any) -> bool:
    return isinstance(effects, RecordingEffects)
