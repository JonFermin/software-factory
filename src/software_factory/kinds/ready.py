"""``ready``: a manual action (``factory ready <repo>``) that improves a repo's agent-readiness.

``ready_action`` builds the action the CLI dispatches: the target is the repo, the version is the
sha of ``origin/<default>`` read in the clone right after a fetch. The handler works on the branch
``factory/ready-<YYYYMMDD>`` from the default branch, gives the worker the *full* readiness report
(commands run, flakiness measured) and the discovered verification commands, and then publishes
through ``build.publish_branch``: push policy, push, evidence section, draft PR without an issue
key. No tracker is involved.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..actions import Action, Kind
from ..discover import Verification, discover_cached
from ..readiness import Readiness, readiness_cached
from ..sources.github import GhAuthError, GhError
from ..workers.schemas import ReadyOutput
from ..worktree import WorktreeError, clone_lock, hooks_disabled_args
from .base import (
    H_ACTION,
    H_READINESS,
    H_VERIFICATION,
    Deps,
    Outcome,
    flag_async,
    notify_async,
    render_action_section,
    render_readiness_section,
    render_verification_section,
    run_code_worker,
    user_prompt,
    worker_failed,
)
from .build import (
    SUMMARY_LIMIT,
    Published,
    branch_lines,
    clip,
    min_readiness_for,
    parse_output,
    publish_branch,
    resume_pending_pr,
    worktree_start_ref,
)

if TYPE_CHECKING:
    from ..repos import RepoInfo

READY_BRANCH_PREFIX = "factory/ready-"
DEFAULT_PR_TITLE = "Improve factory readiness"


def ready_branch(now: datetime) -> str:
    return f"{READY_BRANCH_PREFIX}{now.strftime('%Y%m%d')}"


def ready_action(deps: Deps, repo_key: str) -> Action:
    """The manual action for ``factory ready <repo>``: version = ``origin/<default>`` sha after a
    fetch in the clone. Raises ``ValueError`` for an unmanaged repo, ``WorktreeError`` when git fails."""
    repo = deps.repos.get(repo_key)
    if repo is None:
        managed = ", ".join(deps.repos.slugs()) or "none"
        raise ValueError(f"unknown repo {repo_key!r}; managed repos: {managed}")
    clone = Path(repo.path)
    with clone_lock(clone):  # serialised with the worktree manager's fetches in this clone
        # Hooks disabled like every engine git in a clone: the fetch updates refs/remotes/origin/*
        # and would otherwise fire the clone's (worker-writable) reference-transaction hook.
        fetch = [*hooks_disabled_args(), "fetch", "--no-tags", "origin", repo.default_branch]
        deps.wt_manager.git(fetch, cwd=clone)
        sha = deps.wt_manager.git(["rev-parse", f"origin/{repo.default_branch}^{{commit}}"], cwd=clone).strip()
    return Action(
        kind=Kind.READY,
        version=sha,
        title=f"{DEFAULT_PR_TITLE}: {repo.slug}",
        repo=repo.slug,
        context={"default_branch": repo.default_branch, "head_sha": sha},
    )


# --------------------------------------------------------------------------------------- ready
async def run_ready(action: Action, deps: Deps) -> Outcome:
    repo = deps.repos.get(action.repo or "")
    if repo is None:
        return Outcome(status="skipped", note=f"unknown repo {action.repo!r}")
    branch = ready_branch(deps.now())
    try:
        existing = await asyncio.to_thread(deps.gh.find_open_pr_for_branch, repo.slug, branch)
    except GhAuthError:
        raise
    except GhError as exc:
        return Outcome(status="failed", note=clip(f"gh: cannot check for an open PR on {branch}: {exc}"))
    if existing is not None:
        where = existing.url or f"{repo.slug}#{existing.number}"
        return Outcome(status="skipped", note=f"open PR already exists for {branch}: {where}")

    pending = await resume_pending_pr(action, deps, repo=repo, branch=branch, kind=Kind.READY, issue_key=None)
    if pending is not None:
        if pending.pr is None:
            return pending.outcome
        return await announce_ready_pr(deps, repo=repo, branch=branch, published=pending, summary=pending.summary, changes=[])

    try:
        wt = await asyncio.to_thread(deps.wt_manager.for_new_branch, repo, branch)
    except WorktreeError as exc:
        await flag_async(deps, action, "worktree failed", str(exc), dedupe_key=f"worktree:{action.target}:ready:{action.version}")
        return Outcome(status="failed", note=clip(f"worktree: {exc}"))

    verification = await asyncio.to_thread(discover_cached, deps.state, repo)
    readiness = await asyncio.to_thread(full_readiness, deps, repo, verification)
    start_ref = await asyncio.to_thread(worktree_start_ref, deps, repo, branch, wt)
    prompt = ready_prompt(
        action, deps, repo=repo, branch=branch, verification=verification, readiness=readiness, start_ref=start_ref
    )
    run = await run_code_worker(action, deps, repo=repo, wt=wt, kind=Kind.READY, schema=ReadyOutput, prompt=prompt)
    output, run = parse_output(run, ReadyOutput)
    if output is None:
        return await asyncio.to_thread(worker_failed, action, deps, run, wt=wt)

    published = await publish_branch(
        action, deps, repo=repo, wt=wt, output=output, branch=branch, run=run, kind=Kind.READY,
        issue_key=None, readiness=readiness, pr_title=output.pr_title.strip() or DEFAULT_PR_TITLE,
        pr_summary=pr_summary(output),
    )  # fmt: skip
    if published.pr is None:
        return published.outcome
    return await announce_ready_pr(
        deps, repo=repo, branch=branch, published=published, summary=output.summary,
        changes=[c for c in output.changes if c.strip()],
    )  # fmt: skip


async def announce_ready_pr(
    deps: Deps, *, repo: RepoInfo, branch: str, published: Published, summary: str, changes: list[str]
) -> Outcome:
    assert published.pr is not None
    pr, outcome = published.pr, published.outcome
    lines = [f"{repo.slug}: {pr.title}", pr.url, f"branch `{branch}` -> `{repo.default_branch}`", clip(summary, SUMMARY_LIMIT)]
    lines += [f"- {change}" for change in changes]
    await notify_async(deps, "readiness PR opened", "\n".join(line for line in lines if line), target=repo.slug, kind="ready")
    return outcome.model_copy(update={"pr_url": pr.url or None, "note": clip(f"draft PR {pr.url}; {outcome.note}")})


def full_readiness(deps: Deps, repo: RepoInfo, verification: Verification) -> Readiness | None:
    """The full readiness run (commands executed in a throwaway worktree under the profile's
    worktree root); a cached full run at the same HEAD is reused. Never fatal."""
    try:
        return readiness_cached(deps.state, repo, verification, run_commands=True, wt_manager=deps.wt_manager)
    except Exception as exc:  # noqa: BLE001 - readiness is informational
        deps.audit.write("readiness.failed", repo=repo.slug, error=f"{type(exc).__name__}: {exc}")
        return None


def ready_prompt(
    action: Action,
    deps: Deps,
    *,
    repo: RepoInfo,
    branch: str,
    verification: Verification,
    readiness: Readiness | None,
    start_ref: str | None = None,
) -> str:
    extra = [
        f"- repo: {repo.slug} (clone `{repo.dirname}`, default branch `{repo.default_branch}`)",
        *branch_lines(branch, start_ref or f"origin/{repo.default_branch}"),
        "- started by a human with `factory ready`; close the readiness gaps below without changing product behaviour",
    ]
    if repo.override.visual:
        command = repo.override.dev_command or verification.dev_command or "(not configured)"
        port = repo.override.dev_port or verification.dev_port or "(not configured)"
        extra.append(f"- visual repo: dev command `{command}`, port {port}")
    return user_prompt(
        [
            (H_ACTION, render_action_section(action, "\n".join(extra))),
            (H_READINESS, render_readiness_section(readiness, min_readiness_for(deps, repo))),
            (H_VERIFICATION, render_verification_section(verification)),
        ]
    )


def pr_summary(output: ReadyOutput) -> str:
    parts = [output.summary.strip()]
    changes = [c.strip() for c in output.changes if c.strip()]
    if changes:
        parts.append("## Changes\n\n" + "\n".join(f"- {c}" for c in changes))
    return "\n\n".join(p for p in parts if p) or "Readiness improvements."


# ------------------------------------------------------------------------------------- handler
class ReadyHandler:
    kind = Kind.READY

    async def run(self, action: Action, deps: Deps) -> Outcome:
        return await run_ready(action, deps)


HANDLER = ReadyHandler()
