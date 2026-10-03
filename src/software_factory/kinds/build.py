"""``build``: implement a tracker issue on its branch and open a draft pull request.

Pre-flight re-reads the issue (label still present, ``unstarted``, no unfinished blockers), resolves
the repo through the resolver chain (unresolved -> one flag carrying the best guess, ``skipped``, no
attempt spent), refuses when an open PR already exists for the branch, and creates the worktree on
``issue.branch_name`` (or ``factory/<key>``) from the default branch. The worker prompt carries
``## Action``, ``## Issue`` (the tracker's context), ``## Verification commands`` and
``## Repo readiness`` (static unless a full run is cached), plus the dev command, port and routes
when the repo is visual.

The tail is ``publish_branch``, shared with ``ready``: the push policy and the push itself go through
``finish_code_worker`` (refusals keep the worktree + flag), the worker's screenshots are published on
``factory/evidence``, the PR body is composed from the summary, ``Fixes KEY``, the evidence section
and the disclosure footer, and the draft PR is opened (and registered as a factory PR). ``build``
then moves the issue to ``started`` and notifies; a low readiness score shows up as a warning in the
evidence and the notification suggests ``factory ready <repo>``. Every outward write goes through
``deps.effects``; this module never touches ``gh`` or the tracker for a write.

A branch that was pushed but whose draft PR could not be opened (``gh`` down, rate limit) is
remembered in state (``PENDING_PR_CACHE``): the next run of the same action opens the PR from that
record instead of starting another worker session on the already-pushed branch.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from ..actions import Action, Kind
from ..discover import Verification, discover_cached
from ..effects import is_recording
from ..evidence import (
    EvidenceData,
    compose_pr_body,
    evidence_asset_path,
    render_evidence,
    screenshot_label,
    sensitive_adjacent_paths,
)
from ..policy import evaluate_push
from ..readiness import NOT_RUN, Readiness, readiness_cached
from ..resolve import Resolution
from ..runner import WorkerRun
from ..sources.github import GhAuthError, GhError
from ..sources.models import PullRequest
from ..trackers.base import Issue, TrackerAuthError, TrackerError
from ..workers.schemas import BuildOutput, CodeWorkerOutput
from ..worktree import Worktree, WorktreeError
from .base import (
    H_ACTION,
    H_ISSUE,
    H_READINESS,
    H_VERIFICATION,
    AuthFailure,
    Deps,
    Outcome,
    evidence_from_dict,
    evidence_globs,
    finish_code_worker,
    flag_async,
    notify_async,
    push_policy_inputs,
    render_action_section,
    render_issue_section,
    render_readiness_section,
    render_verification_section,
    run_code_worker,
    save_evidence,
    user_prompt,
    worker_changes,
    worker_failed,
)

if TYPE_CHECKING:
    from ..repos import RepoInfo

NOTE_LIMIT = 500
SUMMARY_LIMIT = 300
PENDING_PR_CACHE = "pr_pending"  # state cache namespace: branches pushed whose draft PR is still to open
STATIC_READINESS_NOTE = "static checks only; runtime checks were not run"


# ------------------------------------------------------------------------------ shared helpers
def clip(text: str, limit: int = NOTE_LIMIT) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def min_readiness_for(deps: Deps, repo: RepoInfo) -> int:
    return repo.override.min_readiness or deps.profile.policy.min_readiness


async def fetch_issue(action: Action, deps: Deps) -> Issue | Outcome:
    """The issue behind an issue action, freshly read from the tracker, or the ``Outcome`` that ends
    the action: ``skipped`` without a tracker, ``failed`` (+ flag) when the tracker cannot serve it.
    ``TrackerAuthError`` propagates so the dispatcher stops the wave."""
    if deps.tracker is None:
        return Outcome(status="skipped", note="no tracker configured for this profile")
    ref = action.issue_id or action.issue_key or ""
    if not ref:
        return Outcome(status="skipped", note="action has no issue id or key")
    try:
        return await asyncio.to_thread(deps.tracker.get_issue, ref)
    except TrackerAuthError:
        raise
    except TrackerError as exc:
        await flag_async(
            deps, action, "tracker error", f"get_issue({ref}): {exc}",
            dedupe_key=f"tracker-error:{action.target}:{action.kind.value}:{action.version}",
        )  # fmt: skip
        return Outcome(status="failed", note=clip(f"tracker: {exc}"))


def parse_output[T: BaseModel](run: WorkerRun, model: type[T]) -> tuple[T | None, WorkerRun]:
    """Validate ``run.output`` as ``model``. A validation failure comes back as ``(None, run)`` with
    the problem in ``run.error`` so ``worker_failed`` reports it."""
    if run.output is None:
        return None, run
    try:
        return model.model_validate(run.output), run
    except ValidationError as exc:
        error = clip(f"invalid {model.__name__} from the worker: {exc}", 2000)
        return None, run.model_copy(update={"output": None, "error": error})


def drop_worktree(deps: Deps, wt: Worktree) -> None:
    """Remove a worktree; a stuck one is audited, never fatal."""
    try:
        deps.wt_manager.remove(wt)
    except Exception as exc:  # noqa: BLE001 - a stuck worktree must not fail the action
        deps.audit.write("worktree.remove_failed", path=str(wt.path), error=f"{type(exc).__name__}: {exc}")


async def resolve_repo(issue: Issue, deps: Deps) -> tuple[RepoInfo | None, Resolution]:
    """The managed repo the resolver chain picks for ``issue`` (``None`` when unresolved)."""
    if deps.resolver is None:
        return None, Resolution(repo=None, confidence=0.0, source="none", candidates=deps.repos.slugs())
    resolution = await deps.resolver.resolve(issue)
    repo = deps.repos.get(resolution.repo) if resolution.repo else None
    return repo, resolution


def describe_resolution(resolution: Resolution) -> str:
    if resolution.multi_repo:
        head = "the issue spans several repos (" + ", ".join(resolution.candidates) + "); split it with the planner"
    elif resolution.candidates:
        head = (
            f"best guess {resolution.candidates[0]} via {resolution.source} "
            f"(confidence {resolution.confidence:.2f})"
        )
    else:
        head = "no candidate repo found"
    return head + ". Add a `repo:<name>` label to the issue to resolve it by hand."


# --------------------------------------------------------------------------------------- build
async def run_build(action: Action, deps: Deps) -> Outcome:
    issue = await fetch_issue(action, deps)
    if isinstance(issue, Outcome):
        return issue
    key = issue.key or action.issue_key or action.target
    label = deps.profile.linear.label_build
    if not issue.has_label(label):
        return Outcome(status="skipped", note=f"label {label} is no longer on the issue")
    earlier = pending_earlier_kinds(issue, deps, target=action.target)
    if earlier:
        await flag_async(
            deps, action, "build waits", build_waits_detail(deps, key, earlier, issue.url), level="info",
            dedupe_key=f"build-waits:{key}",
        )  # fmt: skip
        return Outcome(status="skipped", note=f"refine/plan run first: issue still carries {', '.join(earlier)}")
    if issue.state_type != "unstarted":
        return Outcome(status="skipped", note=f"issue state is {issue.state_type or 'unknown'}, not unstarted")
    blockers = issue.unfinished_blockers
    if blockers:
        return Outcome(status="skipped", note="blocked by " + ", ".join(b.key for b in blockers))

    repo, resolution = await resolve_repo(issue, deps)
    if repo is None:
        detail = describe_resolution(resolution)
        await flag_async(deps, action, "repo unresolved", f"{key}: {detail}", dedupe_key=f"resolve:{key}:{issue.updated_at}")
        return Outcome(status="skipped", note=clip(f"repo unresolved: {detail}"))

    branch = (issue.branch_name or "").strip() or f"factory/{key.lower()}"
    try:
        existing = await asyncio.to_thread(deps.gh.find_open_pr_for_branch, repo.slug, branch)
    except GhAuthError:
        raise
    except GhError as exc:
        return Outcome(status="failed", note=clip(f"gh: cannot check for an open PR on {branch}: {exc}"))
    if existing is not None:
        where = existing.url or f"{repo.slug}#{existing.number}"
        return Outcome(status="skipped", note=f"open PR already exists for {branch}: {where}")

    pending = await resume_pending_pr(action, deps, repo=repo, branch=branch, kind=Kind.BUILD, issue_key=key)
    if pending is not None:
        if pending.pr is None:
            return pending.outcome
        return await announce_build_pr(
            action, deps, issue=issue, repo=repo, branch=branch, key=key, published=pending,
            summary=pending.summary, readiness_score=pending.readiness_score,
        )  # fmt: skip

    try:
        wt = await asyncio.to_thread(deps.wt_manager.for_new_branch, repo, branch)
    except WorktreeError as exc:
        await flag_async(deps, action, "worktree failed", str(exc), dedupe_key=f"worktree:{action.target}:build:{action.version}")
        return Outcome(status="failed", note=clip(f"worktree: {exc}"))

    verification = await asyncio.to_thread(discover_cached, deps.state, repo)
    readiness, static = await asyncio.to_thread(static_readiness, deps, repo, verification)
    min_readiness = min_readiness_for(deps, repo)
    start_ref = await asyncio.to_thread(worktree_start_ref, deps, repo, branch, wt)
    prompt = build_prompt(
        action, deps, issue=issue, repo=repo, branch=branch, resolution=resolution,
        verification=verification, readiness=readiness, min_readiness=min_readiness,
        readiness_note=STATIC_READINESS_NOTE if static else "", start_ref=start_ref,
    )  # fmt: skip
    run = await run_code_worker(action, deps, repo=repo, wt=wt, kind=Kind.BUILD, schema=BuildOutput, prompt=prompt)
    output, run = parse_output(run, BuildOutput)
    if output is None:
        return await asyncio.to_thread(worker_failed, action, deps, run, wt=wt)

    pr_title = output.pr_title.strip() or issue.title
    published = await publish_branch(
        action, deps, repo=repo, wt=wt, output=output, branch=branch, run=run, kind=Kind.BUILD,
        issue_key=key, readiness=readiness, pr_title=pr_title, pr_summary=output.pr_summary,
    )  # fmt: skip
    if published.pr is None:
        return published.outcome
    return await announce_build_pr(
        action, deps, issue=issue, repo=repo, branch=branch, key=key, published=published,
        summary=output.summary, readiness_score=int(readiness.score) if readiness is not None else None,
    )  # fmt: skip


async def announce_build_pr(
    action: Action,
    deps: Deps,
    *,
    issue: Issue,
    repo: RepoInfo,
    branch: str,
    key: str,
    published: Published,
    summary: str,
    readiness_score: int | None,
) -> Outcome:
    """After the draft PR exists: move the issue to ``started`` (unless it is), notify, outcome."""
    assert published.pr is not None
    pr, outcome = published.pr, published.outcome
    min_readiness = min_readiness_for(deps, repo)
    notes: list[str] = []
    if issue.state_type != "started":
        try:
            await asyncio.to_thread(deps.effects.transition_issue, issue.id, "started")
        except (AuthFailure, GhAuthError, TrackerAuthError):
            raise
        except Exception as exc:  # noqa: BLE001 - the PR is open; report, do not fail
            deps.audit.write("build.transition_failed", target=key, error=f"{type(exc).__name__}: {exc}")
            notes.append(f"issue not moved to started: {exc}")

    low = readiness_score is not None and readiness_score < min_readiness
    lines = [
        f"{key}: {pr.title or action.title}",
        pr.url,
        f"branch `{branch}` -> `{repo.default_branch}` in {repo.slug}",
        clip(summary, SUMMARY_LIMIT),
    ]
    if low:
        lines.append(
            f"Repo readiness {readiness_score}/100 is below the minimum of {min_readiness}: review with "
            f"extra care and improve it with `factory ready {repo.slug}`."
        )
    lines += notes
    await notify_async(
        deps, "draft PR opened", "\n".join(line for line in lines if line), target=key, kind="build",
        level="warn" if low else "info",
    )  # fmt: skip
    note = f"draft PR {pr.url}; {outcome.note}" + (f" ({'; '.join(notes)})" if notes else "")
    return outcome.model_copy(update={"pr_url": pr.url or None, "note": clip(note)})


def static_readiness(deps: Deps, repo: RepoInfo, verification: Verification) -> tuple[Readiness | None, bool]:
    """``(readiness, static_only)``: the cached full run when there is one at this HEAD, else the
    static checks only -- rescaled over the checks that actually ran, so a documented repo with
    discovered commands is not reported "50/100, low" (and every build PR warned) merely because
    nobody ran its commands. Never fatal."""
    try:
        readiness = readiness_cached(deps.state, repo, verification, run_commands=False)
    except Exception as exc:  # noqa: BLE001 - readiness is informational
        deps.audit.write("readiness.failed", repo=repo.slug, error=f"{type(exc).__name__}: {exc}")
        return None, False
    return static_only(readiness)


def static_only(readiness: Readiness) -> tuple[Readiness, bool]:
    """Drop the checks a static run could not perform (``not run``) and rescore over the rest."""
    kept = [c for c in readiness.checks if c.ok or c.detail != NOT_RUN]
    if len(kept) == len(readiness.checks):
        return readiness, False
    total = sum(c.weight for c in kept)
    score = round(100 * sum(c.weight for c in kept if c.ok) / total) if total else 0
    return readiness.model_copy(update={"score": score, "checks": kept, "missing": [c.name for c in kept if not c.ok]}), True


def pending_earlier_kinds(issue: Issue, deps: Deps, *, target: str | None = None) -> list[str]:
    """The refine / plan labels on a build-labelled issue that still stand for work to come. Those
    kinds run first (they rewrite the text a build reads, or split the work into children), so a
    build is skipped while one is pending -- without spending an attempt: the issue is not wrong,
    just not ready.

    A label counts only when it can still lead somewhere: its kind is enabled (a disabled kind never
    runs, so its label would hold the build forever), and for ``plan`` only while the issue has no
    children and the planner has not already said ``nothing`` for the issue's current version (the
    ``handled`` row of the plan action, version = ``updated_at``; plan never removes its label, so a
    "nothing to plan" verdict leaves it on). ``target`` is the action target the dispatcher
    recorded the plan under (the issue key)."""
    linear = deps.profile.linear
    pending: list[str] = []
    if deps.profile.kind_enabled(Kind.REFINE) and issue.has_label(linear.label_refine):
        pending.append(linear.label_refine)
    if (
        deps.profile.kind_enabled(Kind.PLAN)
        and issue.has_label(linear.label_plan)
        and not issue.children
        and not plan_said_nothing(deps, target or issue.key, issue.updated_at)
    ):
        pending.append(linear.label_plan)
    return pending


def plan_said_nothing(deps: Deps, target: str, version: str) -> bool:
    """The plan action recorded ``nothing`` (too small to split) for this version of the issue."""
    if not target or not version:
        return False
    row = deps.state.handled(target, Kind.PLAN.value, version)
    return bool(row) and row.get("outcome") == "nothing"


def build_waits_detail(deps: Deps, key: str, labels: list[str], url: str = "") -> str:
    """Why the build is held, label by label, and what lets it proceed."""
    linear = deps.profile.linear
    lines = [f"{key}: the build waits while the issue carries {', '.join(f'`{lbl}`' for lbl in labels)}."]
    if linear.label_refine in labels:
        lines.append(
            f"- `{linear.label_refine}`: refine runs first (it rewrites the text the build reads) and removes the "
            "label when it applies; if it keeps failing, see its flags, or remove the label to build the issue as written."
        )
    if linear.label_plan in labels:
        lines.append(
            f"- `{linear.label_plan}`: plan runs first (it may split the issue into sub-issues); remove the label "
            "to build the issue as one PR."
        )
    if url:
        lines.append(url)
    return "\n".join(lines)


def worktree_start_ref(deps: Deps, repo: RepoInfo, branch: str, wt: Worktree) -> str:
    """The remote ref ``for_new_branch`` detached the worktree at: ``origin/<branch>`` when that
    branch already existed on origin and the checkout continues it, else ``origin/<default>``."""
    default = f"origin/{repo.default_branch}"
    try:
        sha = deps.wt_manager.git(
            ["rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}^{{commit}}"], cwd=Path(wt.path)
        ).strip()
    except Exception:  # noqa: BLE001 - no such remote branch (non-zero exit), or git unavailable
        return default
    return f"origin/{branch}" if sha and sha == wt.base_sha else default


def branch_lines(branch: str, start_ref: str) -> list[str]:
    """``## Action`` lines telling a build/ready worker where it stands: the worktree is a detached
    HEAD (no local branch exists or may be created) and the engine pushes ``HEAD`` to ``branch``."""
    return [
        f"- branch: `{branch}` (the engine pushes `HEAD` to it as-is)",
        f"- worktree: a detached HEAD at `{start_ref}`; commit on it as it is. Do not create, switch or "
        "rename branches: `HEAD` is what gets pushed.",
    ]


def build_prompt(
    action: Action,
    deps: Deps,
    *,
    issue: Issue,
    repo: RepoInfo,
    branch: str,
    resolution: Resolution,
    verification: Verification,
    readiness: Readiness | None,
    min_readiness: int,
    readiness_note: str = "",
    start_ref: str | None = None,
) -> str:
    extra = [
        f"- repo: {repo.slug} (clone `{repo.dirname}`, default branch `{repo.default_branch}`)",
        *branch_lines(branch, start_ref or f"origin/{repo.default_branch}"),
        f"- repo resolved via {resolution.source} (confidence {resolution.confidence:.2f})",
    ]
    if repo.override.visual:
        extra += ["", *visual_block(repo, verification)]
    return user_prompt(
        [
            (H_ACTION, render_action_section(action, "\n".join(extra))),
            (H_ISSUE, render_issue_section(issue, deps.tracker)),
            (H_VERIFICATION, render_verification_section(verification)),
            (H_READINESS, render_readiness_section(readiness, min_readiness, note=readiness_note)),
        ]
    )


def visual_block(repo: RepoInfo, verification: Verification) -> list[str]:
    """What the build prompt says for ``visual = true`` repos: dev command, port and routes."""
    command = repo.override.dev_command or verification.dev_command
    port = repo.override.dev_port or verification.dev_port
    if not command or not port:
        return [
            "This repo is marked visual, but no dev command and port are configured "
            "(repos.overrides.<repo>.dev_command / dev_port): skip the captures and say so in `summary`."
        ]
    routes = ", ".join(f"`{r}`" for r in (repo.override.dev_routes or ["/"]))
    return [
        "This repo is visual: capture before/after evidence with Playwright as your instructions describe.",
        f"- dev command: `{command}`",
        f"- dev port: {port}",
        f"- routes to capture: {routes}",
    ]


# --------------------------------------------------------------------------- publish the branch
@dataclass
class Published:
    """``publish_branch`` result: the outcome so far and the PR when one was opened. ``summary``
    and ``readiness_score`` carry what a resumed pending PR remembered for the notification."""

    outcome: Outcome
    pr: PullRequest | None = None
    summary: str = ""
    readiness_score: int | None = None


def pending_key(repo_slug: str, branch: str) -> str:
    return f"{repo_slug}:{branch}"


def remote_branch_sha(deps: Deps, repo: RepoInfo, branch: str) -> str:
    """The sha ``origin`` holds for ``branch`` (``""`` when the branch is not on the remote)."""
    out = deps.wt_manager.git(["ls-remote", "--heads", "origin", f"refs/heads/{branch}"], cwd=Path(repo.path))
    for line in out.splitlines():
        sha, _, ref = line.strip().partition("\t")
        if ref.strip() == f"refs/heads/{branch}" and sha:
            return sha.strip()
    return ""


async def resume_pending_pr(
    action: Action, deps: Deps, *, repo: RepoInfo, branch: str, kind: Kind, issue_key: str | None
) -> Published | None:
    """Open the draft PR of a branch an earlier run pushed but could not open a PR for (the record
    ``publish_branch`` left in ``PENDING_PR_CACHE``). ``None`` when there is nothing to resume (the
    worker path runs); a ``Published`` without ``pr`` when opening failed again (an attempt)."""
    key = pending_key(repo.slug, branch)
    record = deps.state.cache_get(PENDING_PR_CACHE, key)
    if not isinstance(record, dict) or not record.get("body"):
        return None
    try:
        remote = await asyncio.to_thread(remote_branch_sha, deps, repo, branch)
    except Exception as exc:  # noqa: BLE001 - cannot tell; the worker path will find out
        deps.audit.write("build.pending_pr_check_failed", target=action.target, branch=branch, error=f"{type(exc).__name__}: {exc}")
        return None
    if not remote:
        deps.state.cache_delete(PENDING_PR_CACHE, key)
        deps.audit.write("build.pending_pr_dropped", target=action.target, branch=branch, reason="branch no longer on origin")
        return None
    deps.audit.write("build.pending_pr_resumed", target=action.target, branch=branch, head_sha=remote, pushed_sha=record.get("head_sha"))
    try:
        pr = await asyncio.to_thread(
            deps.effects.open_draft_pr, repo.slug, branch, str(record.get("base") or repo.default_branch),
            str(record.get("title") or action.title or branch), str(record["body"]), kind=kind.value, issue_key=issue_key,
        )  # fmt: skip
    except (AuthFailure, GhAuthError, TrackerAuthError):
        raise
    except Exception as exc:  # noqa: BLE001 - still failing: an attempt, the record stays for the next tick
        detail = f"{branch} is on {repo.slug} (pushed earlier) but opening the draft PR failed again: {type(exc).__name__}: {exc}"
        await flag_async(deps, action, "draft PR failed", detail, dedupe_key=f"draft-pr:{action.target}:{kind.value}:{action.version}")
        return Published(Outcome(status="failed", note=clip(detail)))
    if not is_recording(deps.effects):
        deps.state.cache_delete(PENDING_PR_CACHE, key)
        evidence = evidence_from_dict(record.get("evidence"))
        if evidence is not None:
            save_evidence(deps.state, repo.slug, pr.number, evidence)
    note = f"draft PR opened for {branch}, pushed by an earlier run ({str(record.get('head_sha') or remote)[:10]})"
    outcome = Outcome(status="done", pushed=False, note=clip(note), pr_url=pr.url or None)
    score = record.get("readiness_score")
    return Published(outcome, pr, summary=str(record.get("summary") or ""), readiness_score=int(score) if score is not None else None)


async def publish_branch(
    action: Action,
    deps: Deps,
    *,
    repo: RepoInfo,
    wt: Worktree,
    output: CodeWorkerOutput,
    branch: str,
    run: WorkerRun | None,
    kind: Kind,
    issue_key: str | None,
    readiness: Readiness | None,
    pr_title: str,
    pr_summary: str,
) -> Published:
    """Shared tail of ``build`` and ``ready`` after the worker returned ``output``.

    No commits: ``nothing`` (recorded; ``build`` flags it once so the labelled issue is not left
    silently unbuilt) when the worker says so, else ``failed`` + flag (``done`` without commits is a
    bug, ``blocked`` is the worker's own report). Commits: the push policy and the push run through
    ``finish_code_worker`` (refusal -> worktree kept + flag, ``kept``); before that the worker's
    screenshots are published on ``factory/evidence``. Then the PR body is composed (summary,
    ``Fixes KEY`` when there is a key, evidence, disclosure footer) and the draft PR is opened
    through ``effects`` (which registers it as a factory PR); its evidence data is stored for the
    upkeep kinds to merge into. A pushed branch whose PR failed to open is remembered
    (``PENDING_PR_CACHE``) so the next run opens the PR without another worker session.
    """
    kind_value = kind.value
    commits = await asyncio.to_thread(deps.wt_manager.new_commits, wt)
    if not commits:
        # done without a commit -> failed + "no commits" flag; blocked -> failed + flag; nothing -> nothing
        outcome = await finish_code_worker(
            action, deps, repo=repo, wt=wt, output=output, branch=branch, pr_number=None, factory_pr=False,
            run=run, kind=kind, base_ref=repo.default_branch,
        )  # fmt: skip
        if outcome.status == "nothing" and kind == Kind.BUILD:
            detail = (
                f"{issue_key or action.target}: the build worker found nothing to build on {branch} "
                f"({clip(output.summary, SUMMARY_LIMIT) or 'no summary'}). The issue keeps its label but is "
                f"recorded as handled; fix the issue text and run `factory run build {issue_key or action.target}` "
                "to try again, or remove the label."
            )
            await flag_async(deps, action, "nothing to build", detail, level="info",
                             dedupe_key=f"build-nothing:{action.target}:{action.version}")  # fmt: skip
        return Published(outcome)

    changes = await asyncio.to_thread(worker_changes, deps, wt, base_ref=repo.default_branch)
    changed = changes.files
    globs, foreign = await asyncio.to_thread(push_policy_inputs, deps, repo, changed)
    if not evaluate_push(kind_value, output, changed, globs, foreign_owned=foreign).allowed:
        # finish_code_worker evaluates the same policy, keeps the worktree and flags the reasons.
        outcome = await finish_code_worker(
            action, deps, repo=repo, wt=wt, output=output, branch=branch, pr_number=None, factory_pr=False,
            run=run, kind=kind, base_ref=repo.default_branch,
        )  # fmt: skip
        return Published(outcome)

    head = await asyncio.to_thread(deps.wt_manager.head, wt)
    try:
        screenshots, notes = await asyncio.to_thread(publish_screenshots, deps, repo, wt, output, branch)
    except (AuthFailure, GhAuthError, TrackerAuthError):
        await asyncio.to_thread(drop_worktree, deps, wt)  # the wave stops; do not leak the checkout
        raise
    outcome = await finish_code_worker(
        action, deps, repo=repo, wt=wt, output=output, branch=branch, pr_number=None, factory_pr=False,
        run=run, kind=kind, screenshots=screenshots, readiness=readiness, base_ref=repo.default_branch,
    )  # fmt: skip
    if outcome.status != "done":
        return Published(outcome)

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
        min_readiness=min_readiness_for(deps, repo),
        kind=kind_value,
        run_id=deps.run_id,
        head_sha=head if outcome.pushed else "",
    )
    body = compose_pr_body(pr_summary, issue_key, render_evidence(data), deps.profile.disclosure_footer)
    key = pending_key(repo.slug, branch)
    if outcome.pushed:
        # Remembered until the PR exists: a failed ``gh pr create`` must not cost another session.
        deps.state.cache_set(
            PENDING_PR_CACHE, key,
            {
                "repo": repo.slug, "branch": branch, "base": repo.default_branch, "title": pr_title, "body": body,
                "kind": kind_value, "issue_key": issue_key, "head_sha": head, "target": action.target,
                "summary": output.summary, "readiness_score": int(readiness.score) if readiness is not None else None,
                "evidence": data.model_dump(mode="json"),
            },
        )  # fmt: skip
    try:
        pr = await asyncio.to_thread(
            deps.effects.open_draft_pr, repo.slug, branch, repo.default_branch, pr_title, body,
            kind=kind_value, issue_key=issue_key,
        )  # fmt: skip
    except (AuthFailure, GhAuthError, TrackerAuthError):
        raise
    except Exception as exc:  # noqa: BLE001 - the branch is pushed; the next run opens the PR from the record
        detail = (
            f"{branch} was pushed to {repo.slug} but opening the draft PR failed: {type(exc).__name__}: {exc}. "
            "The next run opens the PR for the pushed branch without a new worker session."
        )
        await flag_async(deps, action, "draft PR failed", detail, dedupe_key=f"draft-pr:{action.target}:{kind_value}:{action.version}")
        return Published(outcome.model_copy(update={"status": "failed", "note": clip(detail)}))
    if outcome.pushed:
        deps.state.cache_delete(PENDING_PR_CACHE, key)
        save_evidence(deps.state, repo.slug, pr.number, data)
    note = outcome.note + (f" ({'; '.join(notes)})" if notes else "")
    return Published(outcome.model_copy(update={"pr_url": pr.url or None, "note": clip(note)}), pr)


def diff_stats(deps: Deps, wt: Worktree) -> tuple[int, int]:
    """``(additions, deletions)`` the worker itself produced; ``(0, 0)`` when git cannot say."""
    try:
        changes = worker_changes(deps, wt)
    except Exception:  # noqa: BLE001 - stats are informational
        return 0, 0
    return changes.additions, changes.deletions


def publish_screenshots(
    deps: Deps, repo: RepoInfo, wt: Worktree, output: CodeWorkerOutput, branch: str
) -> tuple[list[tuple[str, str]], list[str]]:
    """Push the worker's screenshots (paths relative to the worktree) to ``factory/evidence`` and
    return ``(caption, url)`` pairs for the evidence section plus notes about anything skipped.
    A failed publish is a note, never a failed action."""
    shots = list(output.evidence.screenshots)
    if not shots:
        return [], []
    root = Path(wt.path).resolve()
    files: list[tuple[Path, str]] = []
    captions: list[str] = []
    notes: list[str] = []
    seen: set[str] = set()
    for shot in shots:
        local = (root / shot.path).resolve()
        try:
            local.relative_to(root)
        except ValueError:
            notes.append(f"screenshot outside the worktree skipped: {shot.path}")
            continue
        if not local.is_file():
            notes.append(f"screenshot not found: {shot.path}")
            continue
        name = local.name if local.name.lower().startswith(shot.phase) else f"{shot.phase}-{local.name}"
        dest = evidence_asset_path(repo.slug, branch, name)
        if dest in seen:
            stem, dot, ext = dest.rpartition(".")
            dest = f"{stem}-{len(seen)}{dot}{ext}" if dot else f"{dest}-{len(seen)}"
        seen.add(dest)
        files.append((local, dest))
        captions.append(screenshot_label(shot))
    if not files:
        return [], notes
    try:
        urls = deps.effects.push_evidence_assets(repo, files)
    except (AuthFailure, GhAuthError, TrackerAuthError):
        raise
    except Exception as exc:  # noqa: BLE001
        deps.audit.write("evidence.assets_failed", repo=repo.slug, error=f"{type(exc).__name__}: {exc}")
        notes.append(f"screenshots not published: {exc}")
        return [], notes
    return list(zip(captions, urls, strict=True)), notes


# ------------------------------------------------------------------------------------- handler
class BuildHandler:
    kind = Kind.BUILD

    async def run(self, action: Action, deps: Deps) -> Outcome:
        return await run_build(action, deps)


HANDLER = BuildHandler()
