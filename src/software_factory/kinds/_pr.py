"""Shared pre-flight for the pull-request kinds (``ci`` / ``conflict`` / ``rc`` / ``rv``).

Every PR handler starts the same way (architecture: "PR kinds common pre-flight"): resolve the
managed repo, re-fetch the PR with ``gh.pr_details``, skip when it is closed or merged, skip when
the action's version no longer matches the PR's current state, refuse fork heads for the kinds
that push, then open a detached worktree at the PR head and discover the repo's verification
commands. ``pr_preflight`` returns either a ``PrContext`` to continue with or the ``Outcome`` to
return as-is.

The small helpers below (prompt sections, output validation, keep-for-human, worktree removal)
are what the four handlers share beyond ``kinds/base.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from ..actions import Action, Kind
from ..discover import Verification, discover_cached
from ..runner import WorkerRun
from ..sources.github import GhAuthError, GhError, parse_iso
from ..sources.models import Feedback, PullRequest
from ..worktree import Worktree, WorktreeError
from .base import (
    H_ACTION,
    H_PR,
    H_VERIFICATION,
    Deps,
    Outcome,
    detached_head_hint,
    flag,
    flag_async,
    keep_worktree_record,
    render_action_section,
    render_pr_section,
    render_verification_section,
    run_stats,
)

if TYPE_CHECKING:
    from ..repos import RepoInfo

PUSH_KINDS: tuple[Kind, ...] = (Kind.CI, Kind.CONFLICT, Kind.RC)
"""PR kinds whose commits the engine may push; a fork head cannot be pushed to, so they skip it."""

NOTE_LIMIT = 500


@dataclass
class PrContext:
    """What a PR handler works with after pre-flight. ``wt`` / ``verification`` are filled by
    ``attach_worktree`` (``pr_preflight(..., worktree=True)`` does it for you)."""

    repo: RepoInfo
    pr: PullRequest
    factory_pr: bool
    issue_key: str | None = None
    wt: Worktree | None = None
    verification: Verification | None = None

    @property
    def worktree(self) -> Worktree:
        if self.wt is None:
            raise RuntimeError("worktree not attached; call attach_worktree first")
        return self.wt


# ---------------------------------------------------------------------------------- versions
def latest_feedback_at(pr: PullRequest) -> str:
    """ISO timestamp of the newest feedback item (the ``rc`` version key); ``""`` without feedback."""
    if not pr.feedback:
        return ""
    return max(pr.feedback, key=_feedback_ts).created_at


def current_version(kind: Kind, pr: PullRequest) -> str:
    """The version key the scan would compute for ``pr`` now (see the table in architecture.md)."""
    if kind == Kind.CONFLICT:
        return f"{pr.head_ref_oid}:{pr.base_ref_oid}"
    if kind == Kind.RC:
        return latest_feedback_at(pr)
    return pr.head_ref_oid


def versions_match(kind: Kind, expected: str, current: str) -> bool:
    """SHA keys compare as strings; the ``rc`` timestamp compares as an instant when both parse."""
    if kind == Kind.RC:
        a, b = parse_iso(expected), parse_iso(current)
        if a is not None and b is not None:
            return a == b
    return expected == current


def _feedback_ts(item: Feedback) -> datetime:
    parsed = parse_iso(item.created_at)
    return parsed if parsed is not None else datetime.min.replace(tzinfo=UTC)


# --------------------------------------------------------------------------------- pre-flight
async def pr_preflight(action: Action, deps: Deps, *, kind: Kind, worktree: bool = True) -> PrContext | Outcome:
    """The common pre-flight. Returns a ``PrContext`` or the ``Outcome`` to return immediately:

    * repo not managed / not a PR action -> ``skipped``
    * PR closed or merged -> ``skipped``; version moved on -> ``skipped("stale ...")``
    * fork head on a pushing kind -> flag once + ``skipped``
    * ``gh`` read failure (non-auth) -> ``failed``; worktree failure -> flag + ``failed``

    ``GhAuthError`` propagates (the dispatcher stops the wave).
    """
    if action.repo is None or action.pr is None:
        return Outcome(status="skipped", note="not a pull request action")
    repo = deps.repos.get(action.repo)
    if repo is None:
        return Outcome(status="skipped", note=f"repo {action.repo} is not managed by this profile")
    try:
        pr = await asyncio.to_thread(deps.gh.pr_details, repo.slug, action.pr)
    except GhAuthError:
        raise
    except GhError as exc:
        return Outcome(status="failed", note=clip(f"pr fetch failed: {exc}", NOTE_LIMIT))
    if pr.state != "OPEN":
        return Outcome(status="skipped", note=f"pr {pr.state.lower()}")
    current = current_version(kind, pr)
    if not versions_match(kind, action.version, current):
        return Outcome(status="skipped", note=f"stale: version {action.version} is now {current or '(none)'}")
    if kind in PUSH_KINDS and pr.is_fork:
        await flag_async(
            deps,
            action,
            "pull request from a fork",
            f"{pr.target} has its head in {pr.head_repo}; the factory cannot push to forks, so "
            f"{kind.value} is skipped. Handle it by hand.",
            dedupe_key=f"fork:{action.target}:{kind.value}:{action.version}",
        )
        return Outcome(status="skipped", note=f"fork head ({pr.head_repo}): cannot push")
    ctx = PrContext(
        repo=repo,
        pr=pr,
        factory_pr=bool(deps.state.is_factory_pr(repo.slug, pr.number)),
        issue_key=_issue_key(deps, repo.slug, pr.number),
    )
    if worktree:
        failure = await attach_worktree(action, deps, ctx)
        if failure is not None:
            return failure
    return ctx


async def attach_worktree(action: Action, deps: Deps, ctx: PrContext) -> Outcome | None:
    """``wt_manager.for_pr`` (detached at the PR head, verified) + ``discover_cached``. A worktree
    failure is flagged and returned as a ``failed`` outcome; ``None`` means ``ctx`` is ready."""
    try:
        ctx.wt = await asyncio.to_thread(deps.wt_manager.for_pr, ctx.repo, ctx.pr)
    except WorktreeError as exc:
        flag(
            deps,
            action,
            "worktree failed",
            f"could not create a worktree for {ctx.pr.target} at {ctx.pr.head_ref_oid}: {exc}",
            dedupe_key=f"worktree:{action.target}:{action.kind.value}:{action.version}",
        )
        return Outcome(status="failed", note=clip(f"worktree: {exc}", NOTE_LIMIT))
    ctx.verification = await asyncio.to_thread(discover_cached, deps.state, ctx.repo)
    return None


def _issue_key(deps: Deps, slug: str, number: int) -> str | None:
    """The tracker key the factory registered when it opened the PR, if any."""
    try:
        rows = deps.state.factory_prs()
    except Exception:  # noqa: BLE001 - informational only
        return None
    for row in rows:
        if row.get("repo") == slug and row.get("number") == number:
            key = row.get("issue_key")
            return str(key) if key else None
    return None


# ------------------------------------------------------------------------------ prompt pieces
def pr_sections(action: Action, ctx: PrContext, *, extra: str = "") -> list[tuple[str, str]]:
    """``## Action`` + ``## Pull request`` for ``user_prompt`` (headings verbatim from base)."""
    return [
        (H_ACTION, render_action_section(action, extra)),
        (H_PR, render_pr_section(ctx.pr, issue_key=ctx.issue_key)),
    ]


def verification_section(ctx: PrContext) -> tuple[str, str]:
    return (H_VERIFICATION, render_verification_section(ctx.verification or Verification()))


# ------------------------------------------------------------------------------ worker output
def validate_output[T: BaseModel](run: WorkerRun, model: type[T]) -> T | None:
    """``run.output`` as ``model``; ``None`` when the run produced nothing usable (``run.error`` then
    says why, for ``worker_failed``)."""
    if run.output is None:
        return None
    try:
        return model.model_validate(run.output)
    except ValidationError as exc:
        run.error = clip(f"structured output does not match {model.__name__}: {exc}", 1000)
        return None


# ------------------------------------------------------------------------- keep for a human
def keep_for_human(
    deps: Deps,
    action: Action,
    wt: Worktree,
    reasons: list[str],
    *,
    run: WorkerRun | None,
    summary: str = "",
    reason: str = "push refused",
    branch: str | None = None,
) -> Outcome:
    """``state.keep_worktree`` + flag (dedupe per target/kind/version) -> ``kept``. The kept-worktree
    filter blocks re-dispatch of the target until ``factory release``. ``branch`` (the PR head) goes
    into the flag's detached-HEAD hint. Blocking (state + notifier): call it through
    ``asyncio.to_thread`` from a handler."""
    joined = "; ".join(r for r in reasons if r) or "no reason given"
    detail = f"{joined}\nworktree: {wt.path}\n{detached_head_hint(branch or wt.branch, pr=True)}"
    if summary:
        detail += f"\nworker: {clip(summary, 400)}"
    detail = keep_worktree_record(deps, action.target, action.kind.value, wt, joined, detail)
    flag(deps, action, reason, detail, dedupe_key=f"{reason}:{action.target}:{action.kind.value}:{action.version}")
    return Outcome(
        status="kept",
        note=clip(f"{reason}: {joined} (worktree kept: {wt.path})", NOTE_LIMIT),
        **run_stats(run),
    )


def remove_worktree(deps: Deps, wt: Worktree | None) -> None:
    """Best-effort removal; a stuck worktree is audited, never fatal."""
    if wt is None:
        return
    try:
        deps.wt_manager.remove(wt)
    except Exception as exc:  # noqa: BLE001 - cleanup must not fail the action
        deps.audit.write("worktree.remove_failed", path=str(wt.path), error=f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------------------------ helpers
def clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + f"... [{len(text) - limit} more chars]"
