"""``ci``: my pull request has failing checks on its current head.

Pre-flight (``kinds/_pr.py``), then the failing checks with the failed log of every GitHub Actions
run behind them (one fetch per run id; checks that are not Actions runs carry their details URL
only) go into the prompt under ``## Failing checks``, each Actions check with a ``base branch:``
line: the latest run of its workflow on the PR's base branch, looked up by the engine's own
(authenticated, read-only) ``gh run list`` -- the evidence for an ``inherited`` classification,
which the worker cannot fetch itself when it runs ``gh`` unauthenticated. The worker classifies each
failure and fixes only what the PR caused (``CiOutput``).

Outcome rules: no commits and every failure classified ``flaky`` / ``inherited`` / ``infra`` ->
``done`` at this head (there is nothing to push; the human is told the classifications once per
target+version). Everything else goes through ``finish_code_worker``: push policy -> push ->
evidence refresh on factory-opened PRs, or keep the worktree and flag.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..actions import Action, Kind
from ..sources.github import GhAuthError, GhError
from ..sources.models import Check
from ..workers.schemas import CiOutput
from ._pr import (
    NOTE_LIMIT,
    attach_worktree,
    clip,
    pr_preflight,
    pr_sections,
    remove_worktree,
    validate_output,
    verification_section,
)
from .base import (
    H_CHECKS,
    Deps,
    Outcome,
    finish_code_worker,
    flag_async,
    render_checks_section,
    run_code_worker,
    run_stats,
    user_prompt,
    worker_failed,
)

NO_FIX_CLASSES: frozenset[str] = frozenset({"flaky", "inherited", "infra"})
"""Classifications that need no code change: nothing to push, the human reruns or chases the base."""

ACTION_NOTE = (
    "Classify every failing check below before touching code, then fix only the failures this pull "
    "request caused. Commit locally and report `push_ok`; the engine pushes `HEAD:refs/heads/<branch>` "
    "after its own policy check."
)


class CiHandler:
    kind = Kind.CI

    async def run(self, action: Action, deps: Deps) -> Outcome:
        pre = await pr_preflight(action, deps, kind=Kind.CI, worktree=False)
        if isinstance(pre, Outcome):
            return pre
        failing = pre.pr.failing_checks
        if not failing:
            # The head is unchanged, so the checks were re-run: green means nothing to do; still
            # running means come back later without spending a version on it.
            if any(c.state == "pending" for c in pre.pr.checks):
                return Outcome(status="skipped", note="no failing checks; some checks are still pending")
            return Outcome(status="nothing", note="no failing checks on this head")
        failure = await attach_worktree(action, deps, pre)
        if failure is not None:
            return failure
        wt = pre.worktree
        try:
            logs = await asyncio.to_thread(fetch_failed_logs, deps, pre.repo.slug, failing)
            base_runs = await asyncio.to_thread(fetch_base_branch_runs, deps, pre.repo.slug, pre.pr.base_ref_name, failing)
        except GhAuthError:
            await asyncio.to_thread(remove_worktree, deps, wt)  # the wave stops; do not leak the checkout
            raise
        prompt = user_prompt(
            [
                *pr_sections(action, pre, extra=ACTION_NOTE),
                (H_CHECKS, render_checks_section(failing, logs, base_runs=base_runs)),
                verification_section(pre),
            ]
        )
        run = await run_code_worker(action, deps, repo=pre.repo, wt=wt, kind=Kind.CI, schema=CiOutput, prompt=prompt)
        output = validate_output(run, CiOutput)
        if output is None:
            return await asyncio.to_thread(worker_failed, action, deps, run, wt=wt)

        commits = await asyncio.to_thread(deps.wt_manager.new_commits, wt)
        if not commits and nothing_to_fix(output):
            await asyncio.to_thread(remove_worktree, deps, wt)
            report = classification_report(output)
            await flag_async(
                deps,
                action,
                "ci failures not caused by the pr",
                f"{report}\n\n{clip(output.summary.strip(), 400)}".strip(),
                level="info",
                dedupe_key=f"ci-classified:{action.target}:{action.version}",
            )
            note = "no fix needed: " + ", ".join(f"{f.check} {f.classification}" for f in output.failures)
            return Outcome(status="done", note=clip(note, NOTE_LIMIT), pr_url=pre.pr.url or None, **run_stats(run))

        # ``done`` with no commits and a failure the PR owns (or no classification at all) is a
        # worker bug: finish_code_worker fails + flags it with the classifications instead of
        # recording the red head as handled.
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
            kind=Kind.CI,
            base_ref=pre.pr.base_ref_name,
        )
        if outcome.pr_url is None and pre.pr.url:
            outcome = outcome.model_copy(update={"pr_url": pre.pr.url})
        return outcome


def nothing_to_fix(output: CiOutput) -> bool:
    """Every failure is flaky / inherited / infra and the worker did not report itself blocked."""
    return (
        output.status in ("done", "nothing")
        and bool(output.failures)
        and all(f.classification in NO_FIX_CLASSES for f in output.failures)
    )


def fetch_failed_logs(deps: Deps, slug: str, failing: list[Check]) -> dict[int, str]:
    """``gh run view --log-failed`` once per distinct Actions run id. A non-auth ``gh`` failure
    becomes a placeholder so the worker knows the log was not available; auth errors propagate."""
    logs: dict[int, str] = {}
    for check in failing:
        run_id = check.run_id
        if run_id is None or run_id in logs:
            continue
        try:
            logs[run_id] = deps.gh.failed_log(slug, run_id)
        except GhAuthError:
            raise
        except GhError as exc:
            logs[run_id] = f"(failed log unavailable: {exc})"
    return logs


BASE_RUN_FIELDS = "conclusion,headSha,url,createdAt"


def fetch_base_branch_runs(deps: Deps, slug: str, base: str, failing: list[Check]) -> dict[str, str]:
    """``{workflow name: "base branch: ..." line}`` for every distinct workflow among the failing
    Actions checks: the latest run of that workflow on ``base`` (``gh run list``, read-only, through
    the engine's own ``gh``). A non-auth failure becomes a note line (the worker then knows the base
    state is unknown, not green); ``GhAuthError`` propagates (the dispatcher stops the wave). ``{}``
    when the ``gh`` client offers no way to run the query (a test fake) or there is no base."""
    lines: dict[str, str] = {}
    if not base:
        return lines
    for check in failing:
        workflow = (check.workflow or "").strip()
        if not workflow or check.run_id is None or workflow in lines:
            continue
        try:
            run = latest_base_run(deps.gh, slug, base, workflow)
        except GhAuthError:
            raise
        except _NoRunner:
            return {}
        except (GhError, OSError, ValueError, TypeError, KeyError) as exc:
            lines[workflow] = f"base branch: not checked ({_one_line(str(exc)) or type(exc).__name__})"
            continue
        lines[workflow] = render_base_run(run, base, workflow)
    return lines


class _NoRunner(LookupError):
    """The ``gh`` client exposes neither ``latest_workflow_run`` nor a runner."""


def latest_base_run(gh: Any, slug: str, base: str, workflow: str) -> dict[str, Any] | None:
    """The newest run of ``workflow`` on branch ``base`` as ``{conclusion, headSha, url,
    createdAt}``, ``None`` when there is none. Uses the client's own ``latest_workflow_run`` when it
    has one, else its injectable runner (``gh run list ... --limit 1 --json``): a read, never a write."""
    method = getattr(gh, "latest_workflow_run", None)
    if callable(method):
        result = method(slug, base, workflow)
        return dict(result) if result else None
    runner = getattr(gh, "_run", None)
    if not callable(runner):
        raise _NoRunner("gh client has no runner")
    out = runner(
        ["run", "list", "-R", slug, "--branch", base, "--workflow", workflow, "--limit", "1", "--json", BASE_RUN_FIELDS]
    )
    data = json.loads(out or "[]")
    if not isinstance(data, list):
        raise ValueError(f"unexpected `gh run list` output: {type(data).__name__}")
    return dict(data[0]) if data and isinstance(data[0], dict) else None


def render_base_run(run: dict[str, Any] | None, base: str, workflow: str) -> str:
    """``base branch: <conclusion> at <sha> (<url>)``; a run still going has no conclusion yet."""
    if not run:
        return f"base branch: no {workflow} run found on `{base}`"
    conclusion = str(run.get("conclusion") or "").strip().lower() or "no conclusion yet (still running)"
    sha = str(run.get("headSha") or "").strip() or "(unknown sha)"
    url = str(run.get("url") or "").strip() or "no url"
    return f"base branch: {conclusion} at {sha} ({url})"


def _one_line(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "..."


def classification_report(output: CiOutput) -> str:
    lines = [f"- {f.check}: {f.classification}" + (f" - {f.cause.strip()}" if f.cause.strip() else "") for f in output.failures]
    return "\n".join(lines) or "_(no failures reported)_"


HANDLER = CiHandler()
