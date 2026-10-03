"""``refine``: rewrite a tracker issue into a testable spec -- notify before edit (invariant 7).

The worker is read-only (an optional detached worktree of the resolved repo, else the profile's
scratch dir) and returns a new title + description. The engine then, in this order and with the
state machine persisted in ``State`` (``refinements``):

1. ``state.save_refinement`` (status ``pending``) + ``refinements/<KEY>.md`` (original, new, unified
   diff, open questions, suggested split) -- *before* anything leaves the machine;
2. ``notifier.notify`` with the diff summary, the open questions and ``factory revert-refine KEY``;
   no user-facing channel delivered -> flag, ``failed``, and the issue is **not** edited;
3. status ``notified`` -> ``effects.update_issue_text`` -> ``edited`` -> ``effects.remove_label``
   (the refine label only) -> ``applied`` -> ``done``.

A later tick that finds the record in ``pending`` / ``notified`` / ``edited`` resumes at that step
without re-running the worker -- unless the issue's title or description changed since the proposal
was saved (or the label is gone): a proposal computed against old text is discarded (``reverted``,
flagged) and the worker runs afresh, so a human edit is never overwritten by a stale rewrite.
``revert_refine`` (``factory revert-refine KEY``) restores the original text through ``effects``
and sets ``reverted``. Refine never adds labels or comments. A worker that finds nothing to refine
removes the label (an allowed write) and says so, so the label is not an inert trigger that costs
a session on every later edit of the issue.
"""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..actions import Action, Kind
from ..effects import is_recording
from ..sources.github import GhAuthError
from ..trackers.base import Issue, TrackerAuthError
from ..workers.schemas import RefineOutput
from ..worktree import Worktree, WorktreeError
from .base import (
    H_ACTION,
    H_ISSUE,
    AuthFailure,
    Deps,
    Outcome,
    ProviderUnavailable,
    flag,
    flag_async,
    render_action_section,
    render_issue_section,
    run_code_worker,
    run_stats,
    user_prompt,
    worker_failed,
)
from .build import clip, drop_worktree, fetch_issue, parse_output

if TYPE_CHECKING:
    from ..repos import RepoInfo

RESUMABLE = ("pending", "notified", "edited")
CACHE_NAMESPACE = "refine"  # worker extras (open questions, split, change summary) for resumes
DIFF_CONTEXT = 2


# -------------------------------------------------------------------------------------- refine
async def run_refine(action: Action, deps: Deps) -> Outcome:
    key = action.issue_key or action.target
    existing = deps.state.get_refinement(key)
    if existing is not None and existing["status"] == "edited":
        # The proposal is already on the issue; only the label removal is left.
        return await asyncio.to_thread(apply_refinement, action, deps, key)

    issue = await fetch_issue(action, deps)
    if isinstance(issue, Outcome):
        return issue
    key = issue.key or key
    label = deps.profile.linear.label_refine

    if existing is not None and existing["status"] in RESUMABLE:
        stale = stale_reason(existing, issue, label)
        if stale is None:
            return await asyncio.to_thread(apply_refinement, action, deps, key)
        deps.state.set_refinement_status(key, "reverted")
        deps.audit.write("refine.stale", target=key, issue_id=issue.id, previous_status=existing["status"], reason=stale)
        await flag_async(
            deps, action, "refinement discarded",
            f"{key}: the saved proposal (status {existing['status']}) was not applied because {stale}. "
            + ("The worker runs again on the current text." if issue.has_label(label) else "Re-add the label to refine it."),
            level="info", dedupe_key=f"refine-stale:{key}:{action.version}",
        )  # fmt: skip
    if not issue.has_label(label):
        return Outcome(status="skipped", note=f"label {label} is no longer on the issue")

    repo, wt = await reading_worktree(issue, deps)
    try:
        prompt = refine_prompt(action, deps, issue=issue, repo=repo, wt=wt)
        run = await run_code_worker(
            action, deps, repo=repo, wt=wt, kind=Kind.REFINE, schema=RefineOutput, prompt=prompt, read_only=True,
            cwd=None if wt is not None else deps.paths.root / "scratch",
        )  # fmt: skip
    finally:
        if wt is not None:
            await asyncio.to_thread(drop_worktree, deps, wt)
    output, run = parse_output(run, RefineOutput)
    if output is None:
        return await asyncio.to_thread(worker_failed, action, deps, run)
    stats = run_stats(run)
    if output.status == "nothing":
        return await refine_nothing(action, deps, issue=issue, key=key, label=label, output=output, stats=stats)
    if output.status == "blocked":
        detail = output.change_summary or "; ".join(output.open_questions) or "worker reported blocked"
        await flag_async(deps, action, "refine blocked", detail, dedupe_key=f"blocked:{action.target}:refine:{action.version}")
        return Outcome(status="failed", note=clip(f"blocked: {detail}"), **stats)
    new_title, new_description = output.title.strip(), output.description.strip()
    if not new_title or not new_description:
        await flag_async(deps, action, "refine produced an empty rewrite", "title or description came back empty; nothing changed",
                         dedupe_key=f"empty:{action.target}:refine:{action.version}")  # fmt: skip
        return Outcome(status="failed", note="worker returned an empty title or description", **stats)

    # Invariant 7, step 1: persist the original and the proposal before anything leaves the machine.
    deps.state.save_refinement(issue.id, key, issue.title, issue.description, new_title, new_description)
    deps.state.cache_set(
        CACHE_NAMESPACE, key,
        {
            "url": issue.url,
            "updated_at": issue.updated_at,
            "open_questions": list(output.open_questions),
            "suggested_split": list(output.suggested_split),
            "acceptance_criteria": list(output.acceptance_criteria),
            "change_summary": output.change_summary,
        },
    )  # fmt: skip
    path = write_refinement_file(deps, key, issue, output)
    deps.audit.write("refine.saved", target=key, issue_id=issue.id, path=str(path))
    outcome = await asyncio.to_thread(apply_refinement, action, deps, key)
    return outcome.model_copy(update=stats)


def stale_reason(record: dict[str, Any], issue: Issue, label: str) -> str | None:
    """Why a saved-but-unapplied proposal must not be applied to ``issue`` now (``None`` = fine)."""
    if not issue.has_label(label):
        return f"the {label} label is no longer on the issue"
    if issue.title != record["original_title"] or issue.description != record["original_description"]:
        return "the issue's title or description changed since the proposal was made"
    return None


async def refine_nothing(
    action: Action, deps: Deps, *, issue: Issue, key: str, label: str, output: RefineOutput, stats: dict[str, Any]
) -> Outcome:
    """``nothing``: the issue is already a crisp spec. The label comes off (allowed write; without it
    every later touch of the issue would start another session) and the human is told."""
    summary = clip(output.change_summary or "issue already a crisp spec")
    note = summary
    try:
        await asyncio.to_thread(deps.effects.remove_label, issue.id, label)
    except (AuthFailure, GhAuthError, TrackerAuthError):
        raise
    except Exception as exc:  # noqa: BLE001 - the verdict stands; the label stays for the human
        deps.audit.write("refine.remove_label_failed", target=key, issue_id=issue.id, error=f"{type(exc).__name__}: {exc}")
        note = f"{summary} (label {label} not removed: {exc})"
    else:
        if is_recording(deps.effects):
            note = f"{summary} (dry-run: label removal recorded only)"
    body = f"{key}: {issue.title}\n{summary}\nThe {label} label was removed; nothing was edited."
    if issue.url:
        body += f"\n{issue.url}"
    await flag_async(deps, action, "nothing to refine", body, level="info", dedupe_key=f"refine-nothing:{key}:{action.version}")
    return Outcome(status="nothing", note=clip(note), **stats)


async def reading_worktree(issue: Issue, deps: Deps) -> tuple[RepoInfo | None, Worktree | None]:
    """A detached worktree of the resolved repo for reading; resolution failure is fine (no repo)."""
    if deps.resolver is None:
        return None, None
    try:
        resolution = await deps.resolver.resolve(issue)
    except (AuthFailure, GhAuthError, TrackerAuthError, ProviderUnavailable):
        raise  # the picker's rate limit stops the wave like everywhere else, it is no "no repo"
    except Exception as exc:  # noqa: BLE001 - the repo is optional context for refine
        deps.audit.write("refine.resolve_failed", target=issue.key, error=f"{type(exc).__name__}: {exc}")
        return None, None
    repo = deps.repos.get(resolution.repo) if resolution.repo else None
    if repo is None:
        return None, None
    try:
        return repo, await asyncio.to_thread(deps.wt_manager.for_default_branch, repo)
    except WorktreeError as exc:
        deps.audit.write("refine.worktree_failed", target=issue.key, repo=repo.slug, error=str(exc))
        return repo, None


def refine_prompt(action: Action, deps: Deps, *, issue: Issue, repo: RepoInfo | None, wt: Worktree | None) -> str:
    if repo is not None and wt is not None:
        extra = (
            f"- repository for reading only: {repo.slug} (default branch `{repo.default_branch}`), "
            f"checked out at `{wt.path}`; do not edit or commit"
        )
    elif repo is not None:
        extra = f"- repository {repo.slug} could not be checked out; work from the issue text alone"
    else:
        extra = "- no repository resolved for this issue; work from the issue text alone"
    extra += "\n- the engine saves the original, notifies the human, then applies your title and description"
    return user_prompt(
        [(H_ACTION, render_action_section(action, extra)), (H_ISSUE, render_issue_section(issue, deps.tracker))]
    )


# -------------------------------------------------------------------------- the effect sequence
def apply_refinement(action: Action, deps: Deps, key: str) -> Outcome:
    """Run the remaining steps for ``key`` from its persisted status: notify (pending) -> edit the
    issue (notified) -> remove the label (edited) -> ``applied``. Each step is persisted before the
    next so a crash resumes exactly where it stopped; no notification means no edit."""
    record = deps.state.get_refinement(key)
    if record is None:
        return Outcome(status="failed", note=f"no refinement saved for {key}")
    meta = deps.state.cache_get(CACHE_NAMESPACE, key) or {}
    label = deps.profile.linear.label_refine
    status = str(record["status"])
    steps: list[str] = []

    if status == "pending":
        delivered = deps.notifier.notify(
            f"refined {key}: {record['new_title']}", notification_body(deps, key, record, meta), target=key, kind="refine"
        )
        if not delivered.user_facing:
            detail = (
                f"no user-facing channel (toast/Slack) delivered the refinement notice for {key}; "
                f"the issue was left unchanged and will be retried. Errors: {'; '.join(delivered.errors) or 'none'}"
            )
            flag(deps, action, "refine not notified", detail, dedupe_key=f"refine-notify:{key}")
            return Outcome(status="failed", note="refinement saved but not notified; issue left unchanged")
        deps.state.set_refinement_status(key, "notified")
        status = "notified"
        steps.append("notified")

    if is_recording(deps.effects):
        # Observe mode / dry-run: the edit and the label removal are only recorded, so the record
        # must not claim ``edited`` / ``applied`` -- a later full-mode run resumes from ``notified``.
        if status == "notified":
            deps.effects.update_issue_text(record["issue_id"], record["new_title"], record["new_description"])
            deps.effects.remove_label(record["issue_id"], label)
        deps.audit.write("refine.would_apply", target=key, issue_id=record["issue_id"], steps=steps, status=status)
        return Outcome(
            status="dry_run",
            note=f"refinement {status}; tracker edits recorded only (status unchanged until a full-mode run)",
        )

    if status == "notified":
        try:
            deps.effects.update_issue_text(record["issue_id"], record["new_title"], record["new_description"])
        except (AuthFailure, GhAuthError, TrackerAuthError):
            raise
        except Exception as exc:  # noqa: BLE001 - status stays 'notified'; the next tick retries the edit
            return _effect_failed(deps, action, key, "update_issue_text", exc)
        deps.state.set_refinement_status(key, "edited")
        status = "edited"
        steps.append("edited")

    if status == "edited":
        try:
            deps.effects.remove_label(record["issue_id"], label)
        except (AuthFailure, GhAuthError, TrackerAuthError):
            raise
        except Exception as exc:  # noqa: BLE001 - status stays 'edited'; the next tick retries
            return _effect_failed(deps, action, key, "remove_label", exc)
        deps.state.set_refinement_status(key, "applied")
        steps.append("applied")

    note = f"refinement applied ({' -> '.join(steps) or 'already applied'})"
    deps.audit.write("refine.applied", target=key, issue_id=record["issue_id"], steps=steps)
    return Outcome(status="done", note=note)


def _effect_failed(deps: Deps, action: Action, key: str, effect: str, exc: Exception) -> Outcome:
    detail = f"{effect} failed for {key}: {type(exc).__name__}: {exc}; the saved refinement resumes on the next tick"
    flag(deps, action, "refine effect failed", detail, dedupe_key=f"refine-effect:{key}:{effect}")
    return Outcome(status="failed", note=clip(detail))


async def revert_refine(deps: Deps, key: str) -> Outcome:
    """``factory revert-refine KEY``: restore the original title/description through ``effects`` and
    set the record to ``reverted``. The label is not re-added (refine never adds labels)."""
    return await asyncio.to_thread(_revert_sync, deps, key)


def _revert_sync(deps: Deps, key: str) -> Outcome:
    record = deps.state.get_refinement(key)
    if record is None:
        return Outcome(status="failed", note=f"no refinement saved for {key}")
    status = str(record["status"])
    if status == "reverted":
        return Outcome(status="nothing", note=f"{key} was already reverted")
    if status in ("pending", "notified"):
        deps.state.set_refinement_status(key, "reverted")
        note = f"{key}: the issue was never edited; refinement record marked reverted"
    elif is_recording(deps.effects):
        # Observe mode: the restore is only recorded, so the record keeps its status -- marking it
        # ``reverted`` here would make a later full-mode revert refuse ("already reverted") while
        # the tracker still shows the refined text.
        deps.effects.update_issue_text(record["issue_id"], record["original_title"], record["original_description"])
        deps.audit.write("refine.would_revert", target=key, issue_id=record["issue_id"], status=status)
        note = (
            f"{key}: original title and description NOT restored (observe mode: tracker edit recorded only; "
            f"run `factory revert-refine {key}` with mode = \"full\" to restore it)"
        )
        return Outcome(status="dry_run", note=note)
    else:
        deps.effects.update_issue_text(record["issue_id"], record["original_title"], record["original_description"])
        deps.state.set_refinement_status(key, "reverted")
        note = f"{key}: original title and description restored"
    deps.audit.write("refine.reverted", target=key, issue_id=record["issue_id"], previous_status=status)
    deps.notifier.notify(f"reverted refinement {key}", f"{note}\nOriginal title: {record['original_title']}", target=key, kind="refine")
    return Outcome(status="done", note=note)


# ------------------------------------------------------------------------------------ rendering
def unified_diff(original: str, new: str) -> str:
    lines = difflib.unified_diff(
        original.splitlines(), new.splitlines(), fromfile="original", tofile="refined", lineterm="", n=DIFF_CONTEXT
    )
    return "\n".join(lines)


def diff_counts(diff: str) -> tuple[int, int]:
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed


def notification_body(deps: Deps, key: str, record: dict[str, Any], meta: dict[str, Any]) -> str:
    """Diff summary + open questions + suggested split + the revert command."""
    diff = unified_diff(record["original_description"], record["new_description"])
    added, removed = diff_counts(diff)
    lines: list[str] = []
    if record["original_title"] != record["new_title"]:
        lines.append(f'Title: "{record["original_title"]}" -> "{record["new_title"]}"')
    else:
        lines.append(f"Title unchanged: {record['new_title']}")
    lines.append(f"Description: +{added} / -{removed} lines (full diff: {deps.paths.refinement_file(key)})")
    summary = str(meta.get("change_summary") or "").strip()
    if summary:
        lines.append(summary)
    questions = [str(q) for q in meta.get("open_questions") or [] if str(q).strip()]
    lines.append("Open questions:" if questions else "Open questions: none")
    lines += [f"- {q}" for q in questions]
    split = [str(s) for s in meta.get("suggested_split") or [] if str(s).strip()]
    if split:
        lines.append("Suggested split:")
        lines += [f"- {s}" for s in split]
    lines.append(f"Revert with: factory revert-refine {key}")
    return "\n".join(lines)


def write_refinement_file(deps: Deps, key: str, issue: Issue, output: RefineOutput) -> Path:
    """``refinements/<KEY>.md``: original, new, unified diff, open questions, suggested split."""
    diff = unified_diff(issue.description, output.description.strip())
    parts = [
        f"# Refinement of {key}",
        "",
        f"- issue: {issue.id}" + (f" ({issue.url})" if issue.url else ""),
        f"- refined at: {deps.now().isoformat(timespec='seconds')} (run {deps.run_id})",
        f"- revert: `factory revert-refine {key}`",
        "",
        "## Original title",
        "",
        issue.title,
        "",
        "## Original description",
        "",
        issue.description.strip() or "_(empty)_",
        "",
        "## New title",
        "",
        output.title.strip(),
        "",
        "## New description",
        "",
        output.description.strip(),
        "",
        "## Diff",
        "",
        "```diff",
        diff or "(no description change)",
        "```",
        "",
        "## Open questions",
        "",
        "\n".join(f"- {q}" for q in output.open_questions) or "_(none)_",
        "",
        "## Suggested split",
        "",
        "\n".join(f"- {s}" for s in output.suggested_split) or "_(none)_",
        "",
        "## Change summary",
        "",
        output.change_summary.strip() or "_(none)_",
        "",
    ]
    path = deps.paths.refinement_file(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------- handler
class RefineHandler:
    kind = Kind.REFINE

    async def run(self, action: Action, deps: Deps) -> Outcome:
        return await run_refine(action, deps)


HANDLER = RefineHandler()
