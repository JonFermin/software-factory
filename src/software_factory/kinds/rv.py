"""``rv``: my review was requested on a pull request.

The only kind that runs for real in ``mode = "observe"``. Pre-flight (``kinds/_pr.py``; a fork head
is fine here, the worktree is fetched via ``pull/<n>/head``), then a *read-only* worker session
(``ToolPolicy(read_only=True)``: no Edit/Write, shell still classified) drafts the review
(``RvOutput``). The draft is written to ``reviews/<repo-name>-<pr>.md`` through
``effects.write_review`` -- a local file, allowed in every mode, never posted -- and the human is
told where it is. The worktree is always removed; nothing is ever pushed.
"""

from __future__ import annotations

import asyncio

from ..actions import Action, Kind
from ..workers.schemas import RvOutput
from ._pr import (
    NOTE_LIMIT,
    clip,
    pr_preflight,
    pr_sections,
    remove_worktree,
    validate_output,
    verification_section,
)
from .base import Deps, Outcome, flag_async, run_code_worker, run_stats, user_prompt, worker_failed

ACTION_NOTE = (
    "Draft a review of this pull request for me to read, edit and post myself. This is read-only: do "
    "not edit files or commit; run the repo's targeted commands when running something settles a "
    "question. Nothing you write is posted; `review_markdown` is saved to a local file."
)


class RvHandler:
    kind = Kind.RV

    async def run(self, action: Action, deps: Deps) -> Outcome:
        pre = await pr_preflight(action, deps, kind=Kind.RV)
        if isinstance(pre, Outcome):
            return pre
        wt = pre.worktree
        try:
            prompt = user_prompt([*pr_sections(action, pre, extra=ACTION_NOTE), verification_section(pre)])
            run = await run_code_worker(
                action, deps, repo=pre.repo, wt=wt, kind=Kind.RV, schema=RvOutput, prompt=prompt, read_only=True
            )
            output = validate_output(run, RvOutput)
            if output is None:
                return await asyncio.to_thread(worker_failed, action, deps, run)  # the worktree goes in ``finally``
            stats = run_stats(run)
            summary = clip(output.summary.strip(), NOTE_LIMIT)
            if output.status == "nothing":
                return Outcome(status="nothing", note=summary or "nothing to review", pr_url=pre.pr.url or None, **stats)
            if output.status == "blocked" or not output.review_markdown.strip():
                why = summary or ("worker returned an empty review" if output.status != "blocked" else "worker blocked")
                await flag_async(
                    deps,
                    action,
                    "review draft blocked",
                    f"{pre.pr.target}: {why}\nhead_verified: {output.head_verified}",
                    dedupe_key=f"rv-blocked:{action.target}:{action.version}",
                )
                return Outcome(status="failed", note=clip(f"blocked: {why}", NOTE_LIMIT), pr_url=pre.pr.url or None, **stats)

            path = await asyncio.to_thread(deps.effects.write_review, pre.repo.slug, pre.pr.number, output.review_markdown)
            await flag_async(
                deps,
                action,
                f"review draft ready: {path}",
                notification_body(pre.pr.title, pre.pr.url, output),
                level="info",
                dedupe_key=f"rv:{action.target}:{action.version}",
            )
            note = f"review draft: {path} ({output.verdict}, {len(output.findings)} finding(s))"
            if not output.head_verified:
                note += " [head not verified by the worker]"
            return Outcome(status="done", note=clip(note, NOTE_LIMIT), pr_url=pre.pr.url or None, **stats)
        finally:
            await asyncio.to_thread(remove_worktree, deps, wt)


def notification_body(title: str, url: str, output: RvOutput) -> str:
    counts: dict[str, int] = {}
    for finding in output.findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    tally = ", ".join(f"{n} {sev}" for sev, n in counts.items()) or "no findings"
    lines = [f"{title}".strip(), f"verdict: {output.verdict} ({tally})"]
    if output.summary.strip():
        lines.append(clip(output.summary.strip(), 400))
    if not output.head_verified:
        lines.append("note: the worker could not verify the worktree head against the PR head")
    if url:
        lines.append(url)
    return "\n".join(line for line in lines if line)


HANDLER = RvHandler()
