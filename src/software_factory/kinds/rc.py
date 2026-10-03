"""``rc``: reviewers (human or bot) left feedback on my pull request.

Pre-flight (``kinds/_pr.py``), then the feedback items the scan selected (``context["feedback_ids"]``;
every item of ``open_feedback`` when none of them is on the PR any more) go into the prompt under
``## Review feedback`` with the resolved / outdated state of their threads. No item at all (only
resolved-thread feedback left, e.g. on a manual run) is ``nothing`` before any worktree exists. The
worker makes the changes and gives every comment exactly one disposition (``RcOutput``); it never
replies -- the dispositions are for the human's notification.

Outcome rules: any ``skipped`` disposition keeps the worktree and flags the human (``kept``): a
partial push would tell the reviewer their feedback was handled. ``status == "nothing"`` is
recorded as handled and the dispositions are notified. Otherwise ``finish_code_worker`` (push
policy -> push -> evidence refresh, or keep + flag) and the dispositions are notified on ``done``.
"""

from __future__ import annotations

import asyncio

from ..actions import Action, Kind
from ..sources.models import Feedback, PullRequest
from ..workers.schemas import RcOutput
from ._pr import (
    NOTE_LIMIT,
    PrContext,
    attach_worktree,
    clip,
    keep_for_human,
    pr_preflight,
    pr_sections,
    remove_worktree,
    validate_output,
    verification_section,
)
from .base import (
    H_FEEDBACK,
    Deps,
    Outcome,
    finish_code_worker,
    flag,
    render_feedback_section,
    run_code_worker,
    run_stats,
    user_prompt,
    worker_failed,
)

KEEP_REASON = "review feedback needs a human"
ACTION_NOTE = (
    "Address the review feedback below with code changes only. Give every feedback item exactly one "
    "disposition in `comments` (copy `comment_id` and `author` as given); never reply, resolve or react. "
    "A `skipped` item blocks the whole push, so say in its `note` what the human has to decide."
)


class RcHandler:
    kind = Kind.RC

    async def run(self, action: Action, deps: Deps) -> Outcome:
        pre = await pr_preflight(action, deps, kind=Kind.RC, worktree=False)
        if isinstance(pre, Outcome):
            return pre
        items = select_feedback(action, pre.pr)
        if not items:
            # e.g. ``factory run rc`` on a PR whose only feedback sits on resolved threads: no
            # session, no worktree, recorded handled (the version key covers that feedback).
            return Outcome(status="nothing", note="no open feedback", pr_url=pre.pr.url or None)
        failure = await attach_worktree(action, deps, pre)
        if failure is not None:
            return failure
        wt = pre.worktree
        prompt = user_prompt(
            [
                *pr_sections(action, pre, extra=ACTION_NOTE),
                (H_FEEDBACK, render_feedback_section(items, pre.pr.threads)),
                verification_section(pre),
            ]
        )
        run = await run_code_worker(action, deps, repo=pre.repo, wt=wt, kind=Kind.RC, schema=RcOutput, prompt=prompt)
        output = validate_output(run, RcOutput)
        if output is None:
            return await asyncio.to_thread(worker_failed, action, deps, run, wt=wt)

        skipped = [c for c in output.comments if c.disposition == "skipped"]
        if skipped:
            reasons = ["comments skipped: " + ", ".join(c.comment_id for c in skipped)]
            reasons += [f"{c.comment_id} ({c.author}): {c.note.strip() or 'no note'}" for c in skipped]
            reasons.extend(output.push_reasons)
            return await asyncio.to_thread(
                keep_for_human, deps, action, wt, reasons, run=run, summary=output.summary, reason=KEEP_REASON,
                branch=pre.pr.head_ref_name,
            )  # fmt: skip

        if output.status == "nothing":
            await asyncio.to_thread(remove_worktree, deps, wt)
            await asyncio.to_thread(notify_dispositions, deps, action, pre, output)
            note = clip(output.summary.strip(), NOTE_LIMIT) or "nothing to change"
            return Outcome(status="nothing", note=note, pr_url=pre.pr.url or None, **run_stats(run))

        outcome = await finish_code_worker(
            action,
            deps,
            repo=pre.repo,
            wt=wt,
            output=output,
            branch=pre.pr.head_ref_name,
            pr_number=pre.pr.number,
            factory_pr=pre.factory_pr,
            run=run,
            kind=Kind.RC,
            base_ref=pre.pr.base_ref_name,
        )
        if outcome.status == "done":
            await asyncio.to_thread(notify_dispositions, deps, action, pre, output)
        if outcome.pr_url is None and pre.pr.url:
            outcome = outcome.model_copy(update={"pr_url": pre.pr.url})
        return outcome


def select_feedback(action: Action, pr: PullRequest) -> list[Feedback]:
    """The items named by ``context["feedback_ids"]``; when none match (a manual ``factory run rc``
    has no scan context) every item that can still call for work, i.e. ``open_feedback``: comments
    on a thread the reviewer resolved stay in ``pr.feedback`` for the version key only."""
    wanted = {str(i) for i in action.context.get("feedback_ids") or []}
    chosen = [f for f in pr.feedback if f.id in wanted] if wanted else []
    return chosen or list(pr.open_feedback)


def notify_dispositions(deps: Deps, action: Action, pre: PrContext, output: RcOutput) -> None:
    """One informational flag per target+version listing every disposition (the human's checklist)."""
    lines = [f"- {c.comment_id} ({c.author}): {c.disposition}" + (f" - {c.note.strip()}" if c.note.strip() else "") for c in output.comments]
    body = "\n".join(lines) or "_(no dispositions reported)_"
    summary = clip(output.summary.strip(), 400)
    if summary:
        body = f"{summary}\n\n{body}"
    if pre.pr.url:
        body += f"\n{pre.pr.url}"
    flag(
        deps,
        action,
        "review feedback dispositions",
        body,
        level="info",
        dedupe_key=f"rc-dispositions:{action.target}:{action.version}",
    )


HANDLER = RcHandler()
