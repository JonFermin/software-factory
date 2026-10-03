"""``conflict``: GitHub reports my pull request as ``CONFLICTING`` with its base.

Pre-flight (``kinds/_pr.py``), then the worker merges ``origin/<base>`` into the detached PR head
(never a rebase: the engine pushes ``HEAD:refs/heads/<branch>`` without force) and classifies every
hunk (``ConflictOutput``).

Outcome rules: ``all_resolved == False`` keeps the worktree, markers and all, and flags the human
(``kept``; the kept-worktree filter blocks re-dispatch until ``factory release``) -- this check
runs *before* the shared push tail so an uncommitted partial merge is never thrown away as a
"blocked" attempt. Otherwise ``finish_code_worker`` with ``conflict_markers`` taken from the
worktree itself: leftover markers refuse the push whatever the worker claimed.
"""

from __future__ import annotations

import asyncio

from ..actions import Action, Kind
from ..workers.schemas import ConflictOutput
from ._pr import (
    NOTE_LIMIT,
    clip,
    keep_for_human,
    pr_preflight,
    pr_sections,
    remove_worktree,
    validate_output,
    verification_section,
)
from .base import Deps, Outcome, finish_code_worker, run_code_worker, run_stats, user_prompt, worker_failed

KEEP_REASON = "conflict needs a human"


def action_note(base: str) -> str:
    return (
        f"Base branch: `{base}`. Run `git fetch origin {base}` and then `git merge origin/{base}` in this "
        "worktree (a merge commit, never a rebase, never `--abort`). Classify every hunk before resolving "
        "any; resolve only what is safe and leave the rest exactly as git left it, with `all_resolved: "
        "false`, so the kept worktree is a clean handoff."
    )


class ConflictHandler:
    kind = Kind.CONFLICT

    async def run(self, action: Action, deps: Deps) -> Outcome:
        pre = await pr_preflight(action, deps, kind=Kind.CONFLICT)
        if isinstance(pre, Outcome):
            return pre
        wt = pre.worktree
        prompt = user_prompt([*pr_sections(action, pre, extra=action_note(pre.pr.base_ref_name)), verification_section(pre)])
        run = await run_code_worker(
            action, deps, repo=pre.repo, wt=wt, kind=Kind.CONFLICT, schema=ConflictOutput, prompt=prompt
        )
        output = validate_output(run, ConflictOutput)
        if output is None:
            return await asyncio.to_thread(worker_failed, action, deps, run, wt=wt)

        commits = await asyncio.to_thread(deps.wt_manager.new_commits, wt)
        if output.status == "nothing" and not commits:
            await asyncio.to_thread(remove_worktree, deps, wt)
            note = clip(output.summary.strip(), NOTE_LIMIT) or "branch already merges cleanly"
            return Outcome(status="nothing", note=note, pr_url=pre.pr.url or None, **run_stats(run))
        if not output.all_resolved:
            unsafe = [h.file for h in output.hunks if h.classification == "unsafe"]
            reasons = ["not all conflicts resolved"]
            if unsafe:
                reasons.append("unsafe hunks: " + ", ".join(dict.fromkeys(unsafe)))
            reasons.extend(output.push_reasons)
            return await asyncio.to_thread(
                keep_for_human, deps, action, wt, reasons, run=run, summary=hunk_summary(output), reason=KEEP_REASON,
                branch=pre.pr.head_ref_name,
            )  # fmt: skip

        # ``base_ref``: the push policy and the evidence look at the worker's own changes (conflict
        # resolutions), not at every base-branch file the merge of origin/<base> brought in.
        markers = await asyncio.to_thread(deps.wt_manager.has_conflict_markers, wt)
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
            kind=Kind.CONFLICT,
            conflict_markers=markers,
            base_ref=pre.pr.base_ref_name,
        )
        if outcome.pr_url is None and pre.pr.url:
            outcome = outcome.model_copy(update={"pr_url": pre.pr.url})
        return outcome


def hunk_summary(output: ConflictOutput) -> str:
    """The worker's summary followed by one line per hunk (the human's guide through the handoff)."""
    lines = [output.summary.strip()] if output.summary.strip() else []
    lines += [f"- {h.file}: {h.classification} - {h.resolution.strip()}".rstrip(" -") for h in output.hunks]
    return "\n".join(lines)


HANDLER = ConflictHandler()
