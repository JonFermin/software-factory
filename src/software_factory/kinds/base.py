"""Shared layer for the kind handlers (``kinds/<kind>.py``).

A handler receives one ``Action`` and the engine ``Deps`` and returns an ``Outcome``. Everything a
handler needs that is not kind-specific lives here: the prompt section renderers (headings match
what ``workers/*.md`` reference verbatim), ``run_code_worker`` (policy + prompt + timeout +
transcript, auth failure -> ``AuthFailure``), ``finish_code_worker`` (the shared push tail:
commits -> push policy -> ``effects.push`` -> evidence refresh -> remove worktree, or keep the
worktree + flag) and ``flag`` (flag-once semantics through ``state.add_flag``).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel

from ..actions import Action, Kind
from ..config import Profile
from ..effects import Effects, PushRejected, is_auth_failure, is_recording, make_effects
from ..evidence import EvidenceData, render_evidence, sensitive_adjacent_paths
from ..notify import Audit, Delivered, Notifier
from ..paths import ProfilePaths
from ..policy import ToolPolicy, evaluate_push, foreign_owned_changes, sensitive_globs_for
from ..readiness import Readiness
from ..runner import WorkerRun, load_prompt
from ..runner import run_worker as default_run_worker
from ..sources.github import GhAuthError, GitHub
from ..sources.models import Check, Feedback, PullRequest, ReviewThread
from ..state import State
from ..trackers.base import Issue, Tracker, TrackerAuthError
from ..workers.schemas import CodeWorkerOutput, RepoPick
from ..worktree import Worktree, WorktreeManager

if TYPE_CHECKING:
    from ..discover import Verification
    from ..repos import RepoIndex, RepoInfo
    from ..resolve import Resolver

# Headings the worker prompts reference verbatim (tests/test_prompts.py pins them).
H_ACTION = "## Action"
H_PR = "## Pull request"
H_CHECKS = "## Failing checks"
H_FEEDBACK = "## Review feedback"
H_ISSUE = "## Issue"
H_VERIFICATION = "## Verification commands"
H_READINESS = "## Repo readiness"
H_CANDIDATES = "## Candidate repos"

RESOLVE_TIMEOUT_S = 600.0
EVIDENCE_CACHE = "evidence"  # state cache namespace: the evidence data of every factory-opened PR
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_BODY_CLIP = 6000
_LOG_CLIP = 60_000


class AuthFailure(RuntimeError):
    """Raised by handlers / the dispatcher when gh, the tracker or the model provider rejected our
    credentials. Stops the wave; no attempt is recorded."""


class ProviderUnavailable(RuntimeError):
    """Raised by handlers when the model provider refused or could not serve a session (rate
    limit, overloaded, billing: ``runner.WorkerRun.rate_limited``). An infrastructure failure, not
    a verdict on the target: the dispatcher stops launching new actions, flags once per day and
    records no attempt (invariant 5 semantics), so the next tick retries the same actions."""


OutcomeStatus = Literal["done", "nothing", "kept", "flagged", "failed", "skipped", "dry_run"]


class Outcome(BaseModel):
    status: OutcomeStatus
    pushed: bool = False
    note: str = ""
    pr_url: str | None = None
    cost_usd: float | None = None
    usage: dict | None = None
    duration_ms: int | None = None

    @property
    def record_handled(self) -> bool:
        """Invariant 4: only ``done`` / ``nothing`` mark the ``(target, kind, version)`` handled."""
        return self.status in ("done", "nothing")


def new_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


@dataclass(kw_only=True)
class Deps:
    """Everything a handler may touch. Built by ``Deps.build`` for a tick; tests construct it with
    fakes (every field is a plain attribute)."""

    profile: Profile
    paths: ProfilePaths
    state: State
    gh: GitHub
    tracker: Tracker | None = None
    repos: RepoIndex
    effects: Effects
    notifier: Notifier
    audit: Audit
    wt_manager: WorktreeManager
    resolver: Resolver | None = None
    run_worker: Callable[..., Awaitable[WorkerRun]] = default_run_worker
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    run_id: str = field(default_factory=new_run_id)

    @classmethod
    def build(
        cls,
        profile: Profile,
        paths: ProfilePaths,
        *,
        dry_run: bool,
        gh: GitHub | None = None,
        tracker: Tracker | None = None,
        state: State | None = None,
        repos: RepoIndex | None = None,
        effects: Effects | None = None,
        notifier: Notifier | None = None,
        audit: Audit | None = None,
        wt_manager: WorktreeManager | None = None,
        resolver: Resolver | None = None,
        run_worker: Callable[..., Awaitable[WorkerRun]] | None = None,
        now: Callable[[], datetime] | None = None,
        run_id: str | None = None,
        gh_runner: Callable[..., str] | None = None,
        git_runner: Callable[..., str] | None = None,
        http: Any | None = None,
        toaster: Any | None = None,
    ) -> Deps:
        """Construct the real objects for ``profile``; every piece can be overridden (tests)."""
        from ..repos import RepoIndex as _RepoIndex
        from ..resolve import Resolver as _Resolver

        paths.ensure()
        state = state or State(paths.state_db)
        audit = audit or Audit(paths.audit_log, profile.name)
        gh = gh or GitHub.from_profile(profile, run=gh_runner)
        if tracker is None and profile.linear.enabled and profile.secret("LINEAR_API_KEY"):
            from ..trackers.linear import LinearTracker

            tracker = LinearTracker(profile, http=http)
        repos = repos or _RepoIndex.discover(profile)
        effects = effects or make_effects(
            profile, dry_run=dry_run, paths=paths, state=state, gh=gh, tracker=tracker, audit=audit,
            git_runner=git_runner,
        )  # fmt: skip
        notifier = notifier or Notifier(profile, audit, http=http, toaster=toaster)
        wt_manager = wt_manager or WorktreeManager(paths.worktrees, git=git_runner)
        run_worker = run_worker or default_run_worker
        now = now or (lambda: datetime.now(UTC))
        run_id = run_id or new_run_id(now())
        if resolver is None and tracker is not None:
            picker = make_picker(profile, paths, repos, tracker, run_worker, run_id)
            resolver = _Resolver(profile, tracker, repos, state, gh, picker)
        return cls(
            profile=profile, paths=paths, state=state, gh=gh, tracker=tracker, repos=repos,
            effects=effects, notifier=notifier, audit=audit, wt_manager=wt_manager, resolver=resolver,
            run_worker=run_worker, now=now, run_id=run_id,
        )  # fmt: skip


class KindHandler(Protocol):
    kind: Kind

    async def run(self, action: Action, deps: Deps) -> Outcome: ...


# ------------------------------------------------------------------------------- repo picker
def make_picker(
    profile: Profile,
    paths: ProfilePaths,
    repos: RepoIndex,
    tracker: Tracker | None,
    run_worker: Callable[..., Awaitable[WorkerRun]],
    run_id: str,
) -> Callable[[Issue, list[RepoInfo]], Awaitable[RepoPick]]:
    """The resolver's last step: a tool-less model call with ``## Issue`` + ``## Candidate repos``."""

    async def picker(issue: Issue, candidates: list[RepoInfo]) -> RepoPick:
        prompt = user_prompt(
            [(H_ISSUE, render_issue_section(issue, tracker)), (H_CANDIDATES, render_candidates_section(repos, candidates))]
        )
        cwd = paths.root / "scratch"
        cwd.mkdir(parents=True, exist_ok=True)
        run = await run_worker(
            kind="resolve",
            prompt=prompt,
            cwd=cwd,
            schema=RepoPick,
            profile=profile,
            policy=None,
            system_append=load_prompt("resolve"),
            timeout_s=min(float(profile.session_timeout_min * 60), RESOLVE_TIMEOUT_S),
            transcript_path=transcript_path(paths, run_id, "resolve", issue.key or issue.id),
            tools_enabled=False,
        )
        if run.auth_failed:
            raise AuthFailure(run.error or "authentication failed (resolver)")
        if run.rate_limited and run.output is None:
            # only a session that produced nothing is a provider outage; a finished pick stands
            raise ProviderUnavailable(run.error or "provider unavailable (resolver)")
        if run.output is None:
            return RepoPick(repo=None, confidence=0.0, reasoning=f"resolver worker failed: {run.error}")
        return RepoPick.model_validate(run.output)

    return picker


# ------------------------------------------------------------------------------- prompt text
def user_prompt(sections: list[tuple[str, str]]) -> str:
    """``## Heading\\n\\nbody`` blocks separated by blank lines. Headings are used verbatim when
    they already start with ``#``; otherwise ``## `` is prepended."""
    blocks: list[str] = []
    for heading, body in sections:
        head = heading.strip()
        if not head.startswith("#"):
            head = f"## {head}"
        text = (body or "").strip() or "_(none)_"
        blocks.append(f"{head}\n\n{text}")
    return "\n\n".join(blocks) + "\n"


def render_action_section(action: Action, extra: str = "") -> str:
    lines = [f"- kind: `{action.kind.value}`", f"- target: {action.target}", f"- version: `{action.version}`"]
    if action.title:
        lines.append(f"- title: {action.title}")
    if action.url:
        lines.append(f"- url: {action.url}")
    if extra.strip():
        lines += ["", extra.strip()]
    return "\n".join(lines)


def render_pr_section(pr: PullRequest, *, issue_key: str | None = None, include_body: bool = True) -> str:
    """The PR as the prompts expect it: title/number/url/author, ``headRefName``/``headRefOid`` and
    base equivalents (the worker compares ``git rev-parse HEAD`` with ``headRefOid``), state, body."""
    lines = [
        f"- repo: {pr.repo}",
        f"- number: {pr.number}",
        f"- title: {pr.title}",
        f"- url: {pr.url}" if pr.url else "- url: (unknown)",
        f"- author: {pr.author}",
        f"- state: {pr.state}" + (" (draft)" if pr.is_draft else ""),
        f"- headRefName: `{pr.head_ref_name}`",
        f"- headRefOid: `{pr.head_ref_oid}`",
        f"- baseRefName: `{pr.base_ref_name}`",
        f"- baseRefOid: `{pr.base_ref_oid}`",
        f"- mergeable: {pr.mergeable}",
    ]
    if pr.is_fork:
        lines.append(f"- head repo (fork): {pr.head_repo}")
    if issue_key:
        lines.append(f"- linked issue: {issue_key}")
    if pr.updated_at:
        lines.append(f"- updated: {pr.updated_at}")
    if include_body:
        body = (pr.body or "").strip()
        lines += ["", "### Body", "", _clip(body, _BODY_CLIP) if body else "_(empty)_"]
    return "\n".join(lines)


def render_checks_section(
    failing: list[Check | dict], logs: dict[Any, str] | None = None, *, base_runs: dict[str, str] | None = None
) -> str:
    """Failing checks (``action.context["failing"]`` dicts or ``Check`` models) with the failed log
    of each Actions run (``logs`` keyed by ``run_id``); non-Actions checks list ``details_url`` only.
    ``base_runs`` (keyed by workflow name) adds the ``base branch: ...`` line the ci prompt uses to
    recognise an inherited failure."""
    logs = logs or {}
    base_runs = base_runs or {}
    if not failing:
        return "_(no failing checks)_"
    blocks: list[str] = []
    for item in failing:
        data = item.model_dump() if isinstance(item, Check) else dict(item)
        name = str(data.get("name") or "(unnamed check)")
        workflow = data.get("workflow") or ""
        run_id = data.get("run_id")
        lines = [f"### {name}" + (f" (workflow: {workflow})" if workflow else "")]
        if run_id is not None:
            job = data.get("job_id")
            lines.append(f"- run_id: {run_id}" + (f", job_id: {job}" if job is not None else ""))
        if data.get("details_url"):
            lines.append(f"- details: {data['details_url']}")
        if data.get("state"):
            lines.append(f"- state: {data['state']}")
        if workflow and base_runs.get(str(workflow)):
            lines.append(f"- {base_runs[str(workflow)]}")
        log = logs.get(run_id) if run_id is not None else None
        if log is None and run_id is not None:
            log = logs.get(str(run_id))
        if log:
            lines += ["", "```text", _clip(log.strip(), _LOG_CLIP), "```"]
        elif run_id is not None:
            lines.append("- log: (not fetched; use `gh run view <run_id> --log-failed`)")
        else:
            lines.append("- log: not a GitHub Actions run; see the details URL")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_feedback_section(feedback: list[Feedback], threads: list[ReviewThread] | None = None) -> str:
    """One block per feedback item: id, author (bot or not), kind, timestamp, file:line + thread
    (for thread comments), url, body. ``threads`` adds the resolved/outdated state of each thread."""
    if not feedback:
        return "_(no feedback)_"
    thread_state = {t.id: t for t in (threads or [])}
    blocks: list[str] = []
    for item in feedback:
        who = f"{item.author} (bot)" if item.author_is_bot else item.author
        lines = [f"### {item.id} by {who}", f"- kind: {item.kind}", f"- created: {item.created_at}"]
        if item.path:
            where = f"`{item.path}`" + (f" line {item.line}" if item.line is not None else "")
            lines.append(f"- location: {where}")
        if item.thread_id:
            state = ""
            thread = thread_state.get(item.thread_id)
            if thread is not None:
                state = " (resolved)" if thread.is_resolved else " (unresolved)"
                if thread.is_outdated:
                    state += " (outdated)"
            lines.append(f"- thread: {item.thread_id}{state}")
        if item.url:
            lines.append(f"- url: {item.url}")
        lines += ["", _clip((item.body or "").strip(), _BODY_CLIP) or "_(empty)_"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_issue_section(issue: Issue, tracker: Tracker | None = None) -> str:
    """The tracker's own markdown context when it is a ``ContextSource``; else the Linear renderer."""
    context_for = getattr(tracker, "context_for", None)
    if callable(context_for):
        return str(context_for(issue))
    from ..trackers.linear import render_context

    return render_context(issue)


def render_verification_section(verification: Verification) -> str:
    lines: list[str] = []
    if verification.setup:
        lines += ["Setup:", ""] + [f"- `{c.command}` (from {c.source})" for c in verification.setup] + [""]
    if verification.verify:
        lines += ["Verify (run the targeted ones for what you changed):", ""]
        lines += [f"- {c.name}: `{c.command}` (from {c.source})" for c in verification.verify]
    else:
        lines.append(
            "No verification commands were discovered for this repo. Look for them in CLAUDE.md / "
            "AGENTS.md or the package manifests before running anything; report what you ran."
        )
    if verification.dev_command:
        port = f" on port {verification.dev_port}" if verification.dev_port else ""
        lines += ["", f"Dev server: `{verification.dev_command}`{port}"]
    return "\n".join(lines)


def render_readiness_section(readiness: Readiness | None, min_readiness: int = 60, *, note: str = "") -> str:
    if readiness is None:
        return "_(readiness not assessed)_"
    score = int(getattr(readiness, "score", 0))
    verdict = "OK" if score >= min_readiness else f"LOW (minimum {min_readiness})"
    if note.strip():
        verdict += f" ({note.strip()})"
    lines = [f"Score: {score}/100 - {verdict}", "", "| Check | Result | Detail |", "|---|---|---|"]
    for check in getattr(readiness, "checks", None) or []:
        ok = "ok" if getattr(check, "ok", False) else "FAIL"
        lines.append(f"| {check.name} | {ok} | {_cell(getattr(check, 'detail', ''))} |")
    missing = list(getattr(readiness, "missing", None) or [])
    flaky = list(getattr(readiness, "flaky", None) or [])
    lines.append("")
    lines.append("Missing: " + (", ".join(f"`{m}`" for m in missing) if missing else "none"))
    if flaky:
        lines.append("Flaky commands: " + ", ".join(f"`{c}`" for c in flaky))
    runtime = getattr(readiness, "runtime_s", None)
    if runtime:
        lines.append(f"Runtime of one verify pass: {float(runtime):.1f}s")
    return "\n".join(lines)


def render_candidates_section(repos: RepoIndex, candidates: list[RepoInfo] | None = None) -> str:
    """Slug + summary (first section of CLAUDE.md/AGENTS.md/README) for each candidate repo."""
    items = candidates if candidates is not None else repos.all()
    if not items:
        return "_(no managed repos)_"
    blocks: list[str] = []
    for repo in items:
        summary = repos.summary(repo.slug).strip() or "_(no CLAUDE.md / AGENTS.md / README summary)_"
        lines = [f"### {repo.slug}", f"- local dir: `{repo.dirname}` (default branch `{repo.default_branch}`)", "", summary]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ------------------------------------------------------------------------------- worker runs
def transcript_path(paths: ProfilePaths, run_id: str, kind: str, target: str) -> Path:
    return paths.transcripts / f"{run_id}-{kind}-{slugify(target)}.jsonl"


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", (text or "").strip()).strip("-") or "target"


def run_stats(run: WorkerRun | None) -> dict[str, Any]:
    """``cost_usd`` / ``usage`` / ``duration_ms`` for an ``Outcome`` from a worker run."""
    if run is None or run.summary is None:
        return {}
    return {"cost_usd": run.summary.cost_usd, "usage": run.summary.usage, "duration_ms": run.summary.duration_ms}


async def run_code_worker(
    action: Action,
    deps: Deps,
    *,
    repo: RepoInfo | None,
    wt: Worktree | None,
    kind: Kind | str,
    schema: type[BaseModel],
    prompt: str,
    read_only: bool = False,
    cwd: Path | None = None,
    extra_read_dirs: list[Path] | None = None,
) -> WorkerRun:
    """One worker session in the worktree (or ``cwd``) under this profile's tool policy.

    The shared git dir of the worktree (the clone's ``config`` / ``hooks``, writable through the
    worker's shell) is snapshotted before the session and verified after it: a worker that planted
    a hook or rewrote the config gets its changes reverted, a flag, and no usable output (every
    handler then treats the run as failed; nothing of it is pushed). An ``auth_failed`` run raises
    ``AuthFailure`` and a ``rate_limited`` run *without output* ``ProviderUnavailable``, both after
    the worktree is removed (invariant 5: no attempt, no leak). A session that finished with valid
    structured output is returned even when a rate-limit signal was seen along the way: its work
    (commits in the worktree, a review, a plan) must never be thrown away over a limit blip."""
    kind_value = kind.value if isinstance(kind, Kind) else str(kind)
    workdir = Path(cwd) if cwd is not None else (Path(wt.path) if wt is not None else None)
    if workdir is None:
        raise ValueError("run_code_worker needs a worktree or an explicit cwd")
    workdir.mkdir(parents=True, exist_ok=True)
    policy = ToolPolicy(workdir, deps.profile, extra_read_dirs=extra_read_dirs, read_only=read_only)
    timeout_s = float(deps.profile.session_timeout_min * 60)
    path = transcript_path(deps.paths, deps.run_id, kind_value, action.target)
    guard = GitDirGuard.capture(deps, wt) if wt is not None else None
    deps.audit.write(
        "worker.start", kind=kind_value, target=action.target, version=action.version, cwd=str(workdir),
        read_only=read_only, transcript=str(path), repo=getattr(repo, "slug", None),
    )  # fmt: skip
    run = await deps.run_worker(
        kind=kind_value,
        prompt=prompt,
        cwd=workdir,
        schema=schema,
        profile=deps.profile,
        policy=policy,
        system_append=load_prompt(kind_value),
        timeout_s=timeout_s,
        transcript_path=path,
        tools_enabled=True,
    )
    deps.audit.write(
        "worker.end", kind=kind_value, target=action.target, error=run.error, auth_failed=run.auth_failed,
        timed_out=run.timed_out, rate_limited=run.rate_limited, has_output=run.output is not None, **run_stats(run),
    )  # fmt: skip
    finished = run.output is not None  # judged before a tampered run's output is discarded below
    tampered = guard.restore(deps) if guard is not None else []
    if tampered:
        detail = (
            f"the {kind_value} worker changed the shared git dir of {wt.repo if wt else '?'} "
            f"({'; '.join(tampered)}); the changes were reverted and the run's output is discarded"
        )
        deps.audit.write("worker.git_dir_tampered", kind=kind_value, target=action.target, changes=tampered)
        await flag_async(
            deps, action, "worker modified the git dir", detail, level="error",
            dedupe_key=f"git-dir:{action.target}:{kind_value}:{action.version}",
        )  # fmt: skip
        run = run.model_copy(update={"output": None, "error": detail})
    if run.auth_failed:
        if wt is not None:
            _remove_worktree(deps, wt)
        raise AuthFailure(run.error or f"authentication failed ({kind_value} worker)")
    if run.rate_limited and not finished:
        if wt is not None:
            _remove_worktree(deps, wt)
        raise ProviderUnavailable(run.error or f"provider unavailable ({kind_value} worker)")
    if run.rate_limited:
        deps.audit.write("worker.rate_limited_with_output", kind=kind_value, target=action.target, error=run.error)
    return run


# ----------------------------------------------------------------------------- git dir guard
@dataclass
class GitDirGuard:
    """Snapshot of what a worker must not touch in the worktree's git dir: the clone's shared
    ``config``, the per-worktree ``config.worktree`` and every file in the hooks dir. Edit/Write
    tools are path-checked by the policy, shell redirections are not, and the hooks dir of a linked
    worktree is the user's clone's ``.git/hooks``: a planted ``pre-commit`` runs at the worker's
    next commit, a planted ``pre-push`` would run inside the engine's push."""

    configs: dict[Path, bytes | None]
    hooks_dir: Path | None
    hooks: dict[str, bytes]

    @classmethod
    def capture(cls, deps: Deps, wt: Worktree) -> GitDirGuard | None:
        try:
            common = _git_path(deps, wt, "--git-common-dir")
            hooks_dir = _git_path(deps, wt, "--git-path", "hooks")
            worktree_config = _git_path(deps, wt, "--git-path", "config.worktree")
        except Exception as exc:  # noqa: BLE001 - a fake worktree in tests, or git unavailable
            deps.audit.write("worker.git_dir_guard_failed", path=str(wt.path), error=f"{type(exc).__name__}: {exc}")
            return None
        configs = {p: _read_bytes(p) for p in (common / "config", worktree_config)}
        hooks: dict[str, bytes] = {}
        if hooks_dir.is_dir():
            for child in sorted(hooks_dir.iterdir()):
                if child.is_file():
                    data = _read_bytes(child)
                    if data is not None:
                        hooks[child.name] = data
        return cls(configs=configs, hooks_dir=hooks_dir, hooks=hooks)

    def restore(self, deps: Deps) -> list[str]:
        """Put every changed file back; returns one description per change (empty = untouched)."""
        changes: list[str] = []
        for path, before in self.configs.items():
            after = _read_bytes(path)
            if after == before:
                continue
            changes.append(f"{'added' if before is None else 'modified'} {path.name}")
            _restore_file(deps, path, before)
        if self.hooks_dir is None:
            return changes
        current: dict[str, Path] = {}
        if self.hooks_dir.is_dir():
            current = {c.name: c for c in self.hooks_dir.iterdir() if c.is_file()}
        for name, path in current.items():
            before = self.hooks.get(name)
            if before is None:
                changes.append(f"added hook {name}")
                _restore_file(deps, path, None)
            elif _read_bytes(path) != before:
                changes.append(f"modified hook {name}")
                _restore_file(deps, path, before)
        for name, before in self.hooks.items():
            if name not in current:
                changes.append(f"removed hook {name}")
                _restore_file(deps, self.hooks_dir / name, before)
        return changes


def _git_path(deps: Deps, wt: Worktree, *args: str) -> Path:
    raw = deps.wt_manager.git(["rev-parse", *args], cwd=wt.path).strip()
    path = Path(raw)
    return (path if path.is_absolute() else Path(wt.path) / path).resolve()


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def _restore_file(deps: Deps, path: Path, content: bytes | None) -> None:
    try:
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    except OSError as exc:
        deps.audit.write("worker.git_dir_restore_failed", path=str(path), error=f"{type(exc).__name__}: {exc}")


def worker_failed(action: Action, deps: Deps, run: WorkerRun, *, wt: Worktree | None = None) -> Outcome:
    """A run without usable output: flag it, drop the worktree, return a ``failed`` outcome."""
    error = run.error or "worker produced no structured output"
    reason = "worker timed out" if run.timed_out else "worker failed"
    flag(deps, action, reason, error, dedupe_key=f"worker-failed:{action.target}:{action.kind.value}:{action.version}")
    if wt is not None:
        _remove_worktree(deps, wt)
    return Outcome(status="failed", note=_clip(error, 500), **run_stats(run))


# -------------------------------------------------------------------------------- push tail
async def finish_code_worker(
    action: Action,
    deps: Deps,
    *,
    repo: RepoInfo,
    wt: Worktree,
    output: CodeWorkerOutput,
    branch: str,
    pr_number: int | None,
    factory_pr: bool,
    run: WorkerRun | None = None,
    kind: Kind | str | None = None,
    conflict_markers: bool | None = None,
    screenshots: list[tuple[str, str]] | None = None,
    readiness: Readiness | None = None,
    base_ref: str | None = None,
) -> Outcome:
    """Shared tail for ci / conflict / rc / build / ready after the worker returned ``output``.

    No commits -> ``nothing`` when the worker says so; ``done`` without a commit is a worker bug
    (``failed`` + flag: the PR stays red / conflicting and nobody would know) and ``blocked`` is
    the worker's own report (``failed`` + flag); worktree removed. Commits -> ``evaluate_push`` on
    the worker's *own* changes (a merge of ``origin/<base_ref>`` contributes only the files whose
    merge result differs from both parents, never every base-branch change); allowed ->
    ``effects.push`` (a rejected push keeps the worktree + flags; a git credential failure is an
    ``AuthFailure``), evidence refresh on factory-opened PRs merged with the PR's stored evidence,
    worktree removed; refused -> ``state.keep_worktree`` + flag with the reasons and the path,
    ``kept``.
    """
    kind_value = (kind.value if isinstance(kind, Kind) else str(kind)) if kind else action.kind.value
    return await asyncio.to_thread(
        _finish_sync, action, deps, repo, wt, output, branch, pr_number, factory_pr, run, kind_value,
        conflict_markers, screenshots or [], readiness, base_ref,
    )  # fmt: skip


def _finish_sync(
    action: Action,
    deps: Deps,
    repo: RepoInfo,
    wt: Worktree,
    output: CodeWorkerOutput,
    branch: str,
    pr_number: int | None,
    factory_pr: bool,
    run: WorkerRun | None,
    kind_value: str,
    conflict_markers: bool | None,
    screenshots: list[tuple[str, str]],
    readiness: Readiness | None,
    base_ref: str | None = None,
) -> Outcome:
    stats = run_stats(run)
    target = action.target
    commits = deps.wt_manager.new_commits(wt)
    summary = _clip((output.summary or "").strip(), 400)

    if not commits:
        _remove_worktree(deps, wt)
        if output.status == "nothing":
            return Outcome(status="nothing", note=summary or "nothing to do", **stats)
        if output.status == "done":
            detail = f"the {kind_value} worker reported done but committed nothing on {branch}"
            extra = describe_output(output)
            if extra:
                detail += "\n" + extra
            if summary:
                detail += f"\nworker: {summary}"
            flag(deps, action, "no commits", detail, dedupe_key=f"no-commits:{target}:{kind_value}:{action.version}")
            return Outcome(status="failed", note=_clip("worker reported done but made no commits", 500), **stats)
        detail = "; ".join(output.push_reasons) or summary or "worker reported blocked"
        flag(deps, action, "worker blocked", detail, dedupe_key=f"blocked:{target}:{kind_value}:{action.version}")
        return Outcome(status="failed", note=_clip(f"blocked: {detail}", 500), **stats)

    changes = worker_changes(deps, wt, base_ref=base_ref, base_oid=str(action.context.get("base_ref_oid") or ""))
    changed = changes.files
    globs, foreign = push_policy_inputs(deps, repo, changed)
    markers = conflict_markers
    if markers is None:
        markers = deps.wt_manager.has_conflict_markers(wt) if kind_value == Kind.CONFLICT.value else False
    decision = evaluate_push(kind_value, output, changed, globs, conflict_markers=markers, foreign_owned=foreign)
    if not decision.allowed:
        return _keep(deps, action, wt, kind_value, decision.reasons, stats, summary, branch=branch)

    try:
        after = deps.effects.push(wt, branch, kind=kind_value, pr=pr_number)
    except PushRejected as exc:
        return _keep(deps, action, wt, kind_value, [f"push rejected: {exc}"], stats, summary, reason="push rejected", branch=branch)
    except (AuthFailure, GhAuthError, TrackerAuthError):
        _remove_worktree(deps, wt)
        raise
    except Exception as exc:  # noqa: BLE001 - keep the work; the human decides
        if is_auth_failure(str(exc)):
            # git's own credential failure: an auth stop (flag once, no attempt), not a per-PR flag
            _remove_worktree(deps, wt)
            raise AuthFailure(f"git push to {repo.slug} rejected our credentials: {exc}") from exc
        return _keep(deps, action, wt, kind_value, [f"push failed: {type(exc).__name__}: {exc}"], stats, summary,
                     reason="push failed", status="failed", branch=branch)  # fmt: skip

    # A recorder (observe mode, or Deps built with dry_run=True) only audits "would_push".
    pushed = deps.profile.mode == "full" and not is_recording(deps.effects)
    note = summary
    if factory_pr and pr_number is not None:
        data = EvidenceData(
            changed_files=changed,
            sensitive_adjacent=sensitive_adjacent_paths(changed, evidence_globs(deps, repo, globs)),
            additions=changes.additions,
            deletions=changes.deletions,
            ac_proofs=list(output.evidence.ac_proofs),
            verification=list(output.verification),
            screenshots=screenshots,
            risk_notes=list(output.evidence.risk_notes),
            readiness=readiness,
            min_readiness=repo.override.min_readiness or deps.profile.policy.min_readiness,
            kind=kind_value,
            run_id=deps.run_id,
            head_sha=after if after and after != "dry-run" else "",
        )
        data = merge_evidence(load_evidence(deps.state, repo.slug, pr_number), data)
        try:
            deps.effects.update_pr_evidence(repo.slug, pr_number, render_evidence(data))
        except (AuthFailure, GhAuthError, TrackerAuthError):
            _remove_worktree(deps, wt)
            raise
        except Exception as exc:  # noqa: BLE001 - the push already happened; report, do not fail
            deps.audit.write("evidence.refresh_failed", target=target, error=f"{type(exc).__name__}: {exc}")
            note = f"{note} (evidence refresh failed: {exc})".strip()
        else:
            save_evidence(deps.state, repo.slug, pr_number, data)
    _remove_worktree(deps, wt)
    if not pushed:
        why = "observe mode" if deps.profile.mode != "full" else "dry-run"
        note = f"{note} ({why}: push recorded only)".strip()
    return Outcome(status="done", pushed=pushed, note=_clip(note, 500), **stats)


def describe_output(output: CodeWorkerOutput) -> str:
    """Kind-specific lines for a flag: ci classifications, conflict hunks, rc dispositions."""
    lines: list[str] = []
    for failure in getattr(output, "failures", None) or []:
        cause = str(getattr(failure, "cause", "") or "").strip()
        lines.append(f"- {failure.check}: {failure.classification}" + (f" - {cause}" if cause else ""))
    for hunk in getattr(output, "hunks", None) or []:
        lines.append(f"- {hunk.file}: {hunk.classification} - {hunk.resolution.strip()}".rstrip(" -"))
    for comment in getattr(output, "comments", None) or []:
        note = str(getattr(comment, "note", "") or "").strip()
        lines.append(f"- {comment.comment_id} ({comment.author}): {comment.disposition}" + (f" - {note}" if note else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------------- worker change set
@dataclass
class WorkerChanges:
    """What the worker itself changed in the worktree (the input of the push policy and the
    evidence section)."""

    files: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    merges_of_base: list[str] = field(default_factory=list)  # merge commits that only pulled in the base


def worker_changes(deps: Deps, wt: Worktree, *, base_ref: str | None = None, base_oid: str = "") -> WorkerChanges:
    """The worker's own change set over ``base_sha..HEAD``.

    Without a merge commit in the range this is ``changed_files`` + ``diff --numstat`` of the
    range. With merges (``conflict``: the worker merged ``origin/<base>``), ``base_sha..HEAD`` also
    spans every base-branch commit the merge brought in, so the sensitive-glob check would refuse
    files the PR author never touched (and the evidence would report the base branch's blast
    radius). Instead the first-parent line is walked: a non-merge commit contributes its own diff; a
    merge whose other parents are on the base branch (``origin/<base_ref>`` / ``base_oid``)
    contributes only the files whose result differs from *every* parent (conflict resolutions and
    any extra edit made in the merge); a merge of anything else contributes its whole diff against
    the first parent (conservative: that content is new to the branch)."""
    git = deps.wt_manager.git
    cwd = Path(wt.path)
    merges = _lines(git(["rev-list", "--merges", f"{wt.base_sha}..HEAD"], cwd=cwd))
    if not merges:
        additions, deletions = _numstat(git(["diff", "--numstat", f"{wt.base_sha}..HEAD"], cwd=cwd))
        return WorkerChanges(files=deps.wt_manager.changed_files(wt), additions=additions, deletions=deletions)

    base_tips = [oid for oid in (base_oid.strip(),) if oid]
    if base_ref:
        # no remote-tracking ref for the base: every merge counts as "foreign" (conservative)
        with contextlib.suppress(Exception):
            base_tips.append(git(["rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{base_ref}^{{commit}}"], cwd=cwd).strip())

    def on_base(oid: str) -> bool:
        for tip in base_tips:
            if oid == tip:
                return True
            try:
                git(["merge-base", "--is-ancestor", oid, tip], cwd=cwd)
                return True
            except Exception:  # noqa: BLE001 - non-zero exit: not an ancestor (or unknown)
                continue
        return False

    files: dict[str, None] = {}
    additions = deletions = 0
    merges_of_base: list[str] = []
    for commit in _lines(git(["rev-list", "--reverse", "--first-parent", f"{wt.base_sha}..HEAD"], cwd=cwd)):
        parents = git(["rev-list", "--parents", "-n", "1", commit], cwd=cwd).split()[1:]
        if len(parents) <= 1:
            names = _lines(git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit], cwd=cwd))
            stats = git(["diff", "--numstat", f"{commit}^", commit], cwd=cwd) if parents else ""
        elif all(on_base(p) for p in parents[1:]):
            per_parent = [set(_lines(git(["diff", "--name-only", p, commit], cwd=cwd))) for p in parents]
            names = sorted(set.intersection(*per_parent)) if per_parent else []
            stats = git(["diff", "--numstat", parents[0], commit, "--", *names], cwd=cwd) if names else ""
            merges_of_base.append(commit)
        else:
            names = _lines(git(["diff", "--name-only", parents[0], commit], cwd=cwd))
            stats = git(["diff", "--numstat", parents[0], commit], cwd=cwd)
        for name in names:
            files.setdefault(name.replace("\\", "/"), None)
        add, dele = _numstat(stats)
        additions += add
        deletions += dele
    return WorkerChanges(files=list(files), additions=additions, deletions=deletions, merges_of_base=merges_of_base)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _numstat(out: str) -> tuple[int, int]:
    additions = deletions = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        if parts[0].isdigit():
            additions += int(parts[0])
        if parts[1].isdigit():
            deletions += int(parts[1])
    return additions, deletions


# ------------------------------------------------------------------------- evidence memory
def load_evidence(state: State, repo: str, number: int) -> EvidenceData | None:
    """The evidence data last written to ``repo#number``'s body by the factory, if any."""
    return evidence_from_dict(state.cache_get(EVIDENCE_CACHE, f"{repo}#{number}"))


def evidence_from_dict(raw: Any) -> EvidenceData | None:
    """``EvidenceData`` back from its ``model_dump(mode="json")`` (the readiness is rebuilt as a
    ``Readiness``: the field is duck-typed, a plain dict would render as score 0)."""
    if not isinstance(raw, dict):
        return None
    data = dict(raw)
    if isinstance(data.get("readiness"), dict):
        try:
            data["readiness"] = Readiness.model_validate(data["readiness"])
        except Exception:  # noqa: BLE001 - an unreadable readiness is dropped, not fatal
            data["readiness"] = None
    try:
        return EvidenceData.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


def save_evidence(state: State, repo: str, number: int, data: EvidenceData) -> None:
    if number:
        state.cache_set(EVIDENCE_CACHE, f"{repo}#{number}", data.model_dump(mode="json"))


def merge_evidence(previous: EvidenceData | None, new: EvidenceData) -> EvidenceData:
    """An upkeep refresh (ci / rc / conflict) must not erase what the build recorded: the AC ->
    test map, the screenshots and the readiness warning stay, files and risk notes are unioned,
    verification runs accumulate, the blast radius adds up; kind / run / head are the new push's."""
    if previous is None:
        return new
    proofs = {p.criterion: p for p in previous.ac_proofs}
    for proof in new.ac_proofs:
        proofs[proof.criterion] = proof
    shots = list(previous.screenshots)
    seen = {url for _c, url in shots}
    shots += [(c, u) for c, u in new.screenshots if u not in seen]
    return new.model_copy(
        update={
            "changed_files": list(dict.fromkeys([*previous.changed_files, *new.changed_files])),
            "sensitive_adjacent": sorted(dict.fromkeys([*previous.sensitive_adjacent, *new.sensitive_adjacent])),
            "additions": previous.additions + new.additions,
            "deletions": previous.deletions + new.deletions,
            "ac_proofs": list(proofs.values()),
            "verification": [*previous.verification, *new.verification],
            "screenshots": shots,
            "risk_notes": list(dict.fromkeys([*previous.risk_notes, *new.risk_notes])),
            "readiness": new.readiness if new.readiness is not None else previous.readiness,
        }
    )


def detached_head_hint(branch: str | None, *, pr: bool = False) -> str:
    """The line every kept-worktree flag carries: the checkout is a detached HEAD (the engine
    pushes ``HEAD:refs/heads/<branch>``; no local branch exists), so the human who continues by
    hand must put it on a branch first or lose the commits at the next checkout.

    The suggested local branch is ``<branch>-factory``, never ``<branch>`` itself: the user's clone
    usually has a local ``<branch>`` already (``git switch -c`` would refuse) or has it checked
    out in the main worktree (git refuses to check one branch out twice). For a PR kind (``pr``)
    the hint also names the direct route: review the kept commits and push them to the PR head
    with the same refspec the engine uses."""
    name = branch or "<branch>"
    hint = f"detached HEAD: `git switch -c {name}-factory` to continue by hand"
    if pr:
        hint += f", or review and `git push origin HEAD:refs/heads/{name}` yourself"
    return hint


def _keep(
    deps: Deps,
    action: Action,
    wt: Worktree,
    kind_value: str,
    reasons: list[str],
    stats: dict[str, Any],
    summary: str,
    *,
    reason: str = "push refused",
    status: OutcomeStatus = "kept",
    branch: str | None = None,
) -> Outcome:
    joined = "; ".join(reasons) or "no reason given"
    detail = f"{joined}\nworktree: {wt.path}\n{detached_head_hint(branch or wt.branch, pr=action.pr is not None)}"
    if summary:
        detail += f"\nworker: {summary}"
    detail = keep_worktree_record(deps, action.target, kind_value, wt, joined, detail)
    flag(deps, action, reason, detail, dedupe_key=f"{reason}:{action.target}:{kind_value}:{action.version}")
    return Outcome(status=status, note=_clip(f"{reason}: {joined} (worktree kept: {wt.path})", 500), **stats)


def keep_worktree_record(deps: Deps, target: str, kind_value: str, wt: Worktree, reason: str, detail: str) -> str:
    """``state.keep_worktree`` unless the target already has a kept worktree at another path that
    still exists: that record is a human handoff another action (or ``factory run``) must not
    overwrite (``factory status`` would stop listing it and ``factory cleanup`` would delete it).
    The newer worktree is then only named in the returned flag detail."""
    existing = deps.state.kept_worktree(target)
    if existing is not None and existing.get("path") != str(wt.path) and Path(str(existing.get("path"))).exists():
        deps.audit.write(
            "worktree.keep_refused", target=target, kind=kind_value, path=str(wt.path),
            kept_path=existing.get("path"), kept_kind=existing.get("kind"),
        )  # fmt: skip
        return (
            f"{detail}\nnote: {target} already has a kept worktree from {existing.get('kind')} at "
            f"{existing.get('path')} (still recorded); this {kind_value} worktree is not recorded and "
            "`factory cleanup` will remove it"
        )
    deps.state.keep_worktree(target, kind_value, str(wt.path), reason)
    return detail


def _remove_worktree(deps: Deps, wt: Worktree) -> None:
    try:
        deps.wt_manager.remove(wt)
    except Exception as exc:  # noqa: BLE001 - a stuck worktree must not fail the action
        deps.audit.write("worktree.remove_failed", path=str(wt.path), error=f"{type(exc).__name__}: {exc}")


# -------------------------------------------------------------------------------------- flags
def my_login(deps: Deps) -> str:
    """The real GitHub login (``gh api user``), falling back to ``github.login`` on a non-auth gh
    error. Auth errors propagate: they stop the wave like every other auth failure."""
    from ..sources.github import GhAuthError, GhError

    login_fn = getattr(deps.gh, "login", None)
    if not callable(login_fn):
        return deps.profile.github.login
    try:
        return login_fn() or deps.profile.github.login
    except GhAuthError:
        raise
    except (GhError, OSError):
        return deps.profile.github.login


def push_policy_inputs(deps: Deps, repo: RepoInfo, changed: list[str]) -> tuple[list[str], list[str]]:
    """``(sensitive globs, changed files CODEOWNERS assigns to someone else)`` for ``evaluate_push``."""
    login = my_login(deps)
    globs = sensitive_globs_for(deps.profile, repo, my_login=login)
    return globs, foreign_owned_changes(deps.profile, repo, changed, my_login=login)


def evidence_globs(deps: Deps, repo: RepoInfo, sensitive_globs: list[str]) -> list[str]:
    """The globs the evidence section's *sensitive-adjacent* list is computed with: the push
    policy's sensitive globs plus every CODEOWNERS pattern owned by someone else
    (``codeowners.foreign_owned_globs``), so a reviewer also sees the changed files that sit next to
    another team's code. Evidence only: the push policy applies CODEOWNERS per changed path
    (``foreign_owned_changes``) and never receives these pattern-level globs (a catch-all ``*``
    rule would refuse every push). Informational, so a CODEOWNERS that cannot be read or a login
    that cannot be resolved here falls back to the sensitive globs / the profile's login."""
    from .. import codeowners

    try:
        login = my_login(deps)
    except Exception:  # noqa: BLE001 - the push already ran its own login check
        login = deps.profile.github.login
    try:
        foreign = codeowners.foreign_owned_globs(Path(repo.path), list(deps.profile.policy.my_teams), login or "")
    except Exception as exc:  # noqa: BLE001 - evidence is informational
        deps.audit.write("evidence.codeowners_failed", repo=repo.slug, error=f"{type(exc).__name__}: {exc}")
        foreign = []
    return list(dict.fromkeys([*sensitive_globs, *foreign]))


def flag(
    deps: Deps,
    action: Action | None,
    reason: str,
    detail: str = "",
    *,
    level: str = "warn",
    dedupe_key: str | None = None,
) -> int | None:
    """``state.add_flag`` + ``notifier.notify``; the notification goes out only when the flag is
    new (``add_flag`` returns an id), so a repeated reason with the same ``dedupe_key`` notifies once.
    Returns the flag id, or ``None`` when deduped."""
    target = action.target if action is not None else f"profile:{deps.profile.name}"
    kind = action.kind.value if action is not None else "tick"
    flag_id = deps.state.add_flag(target, kind, reason, detail, dedupe_key)
    if flag_id is not None:
        deps.notifier.notify(reason, detail, target=target, kind=kind, level=level)
    return flag_id


async def flag_async(
    deps: Deps,
    action: Action | None,
    reason: str,
    detail: str = "",
    *,
    level: str = "warn",
    dedupe_key: str | None = None,
) -> int | None:
    """``flag`` for coroutine context: the notifier blocks (Slack HTTP, toast) for up to tens of
    seconds, and on the event loop that would stall the other workers' permission/hook callbacks."""
    return await asyncio.to_thread(flag, deps, action, reason, detail, level=level, dedupe_key=dedupe_key)


async def notify_async(
    deps: Deps, title: str, body: str, *, target: str = "", kind: str = "", level: str = "info"
) -> Delivered:
    """``deps.notifier.notify`` off the event loop (see ``flag_async``)."""
    return await asyncio.to_thread(deps.notifier.notify, title, body, target=target, kind=kind, level=level)


# ------------------------------------------------------------------------------------ helpers
def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + f"... [{len(text) - limit} more chars]"


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")
