"""``plan``: decompose a parent issue into sub-issues the factory can build one at a time.

The worker is read-only in the profile's scratch dir with the managed clones readable
(``extra_read_dirs=[repos.root]``) and gets ``## Issue`` + ``## Candidate repos`` (slug + summary of
each managed repo). Its ``sub_issues`` are validated before anything is created: every ``repo`` must
be managed (normalised to its slug), every ``blocked_by`` index in range and not self, and the
relations acyclic; any problem -> flag + ``failed`` and nothing is created. Then
``effects.create_sub_issues`` creates the children (``repo:<name>`` label or the ``factory:repo``
marker, blocked-by relations, never the build label: the human approves each child by labelling
it), the children's repo resolutions are cached, the parent moves to ``started`` and the human gets
"plan ready for review". A parent that already has children is ``nothing``.
"""

from __future__ import annotations

import asyncio

from ..actions import Action, Kind
from ..effects import is_recording
from ..repos import RepoIndex
from ..resolve import CACHE_NAMESPACE as RESOLVE_CACHE
from ..resolve import PLAN_SOURCE, Resolution
from ..sources.github import GhAuthError
from ..trackers.base import Issue, TrackerAuthError
from ..workers.schemas import PlanOutput, SubIssueSpec
from .base import (
    H_ACTION,
    H_CANDIDATES,
    H_ISSUE,
    AuthFailure,
    Deps,
    Outcome,
    flag_async,
    notify_async,
    render_action_section,
    render_candidates_section,
    render_issue_section,
    run_code_worker,
    run_stats,
    user_prompt,
    worker_failed,
)
from .build import SUMMARY_LIMIT, clip, fetch_issue, parse_output

RESOLUTION_SOURCE = PLAN_SOURCE  # the resolver honours it whatever the child's updated_at


# ---------------------------------------------------------------------------------------- plan
async def run_plan(action: Action, deps: Deps) -> Outcome:
    issue = await fetch_issue(action, deps)
    if isinstance(issue, Outcome):
        return issue
    key = issue.key or action.issue_key or action.target
    if issue.children:
        keys = ", ".join(c.key for c in issue.children if c.key)
        return Outcome(status="nothing", note=f"already has {len(issue.children)} sub-issue(s): {keys}")
    label = deps.profile.linear.label_plan
    if not issue.has_label(label):
        return Outcome(status="skipped", note=f"label {label} is no longer on the issue")

    repos_root = deps.profile.repos.root_path
    scratch = deps.paths.root / "scratch"
    prompt = plan_prompt(action, deps, issue=issue, repos_root=str(repos_root))
    run = await run_code_worker(
        action, deps, repo=None, wt=None, kind=Kind.PLAN, schema=PlanOutput, prompt=prompt, read_only=True,
        cwd=scratch, extra_read_dirs=[repos_root],
    )  # fmt: skip
    output, run = parse_output(run, PlanOutput)
    if output is None:
        return await asyncio.to_thread(worker_failed, action, deps, run)
    stats = run_stats(run)
    if output.status == "nothing":
        summary = clip(output.summary or "parent is small enough to build directly")
        # The label stays (plan never removes labels): say so once, or every later edit of the
        # parent would start another planning session that says nothing to anyone. Removing the
        # plan label first matters: labelling for build edits the issue (new updated_at), and a
        # plan label left on would re-plan the new version before the build may start.
        await flag_async(
            deps, action, "nothing to plan",
            f"{key}: {issue.title}\n{summary}\nThe `{label}` label is still on the issue and does nothing now: "
            f"remove the `{label}` label, then label `{deps.profile.linear.label_build}` to build it as one PR."
            + (f"\n{issue.url}" if issue.url else ""),
            level="info", dedupe_key=f"plan-nothing:{action.target}",
        )  # fmt: skip
        return Outcome(status="nothing", note=summary, **stats)
    if output.status == "blocked":
        detail = output.summary or "; ".join(output.open_questions) or "worker reported blocked"
        await flag_async(deps, action, "plan blocked", detail, dedupe_key=f"blocked:{action.target}:plan:{action.version}")
        return Outcome(status="failed", note=clip(f"blocked: {detail}"), **stats)

    specs, problems = validate_specs(output.sub_issues, deps.repos)
    if problems:
        detail = "\n".join(problems)
        await flag_async(deps, action, "invalid plan", f"{key}: nothing was created.\n{detail}",
                         dedupe_key=f"plan-invalid:{action.target}:{action.version}")  # fmt: skip
        return Outcome(status="failed", note=clip("invalid plan: " + "; ".join(problems)), **stats)

    try:
        children = await asyncio.to_thread(deps.effects.create_sub_issues, issue, specs)
    except (AuthFailure, GhAuthError, TrackerAuthError):
        raise
    except Exception as exc:  # noqa: BLE001 - some children may exist; the human sorts it out
        detail = f"creating sub-issues for {key} failed: {type(exc).__name__}: {exc}"
        await flag_async(deps, action, "sub-issue creation failed", detail, dedupe_key=f"plan-create:{action.target}:{action.version}")
        return Outcome(status="failed", note=clip(detail), **stats)
    for child, spec in zip(children, specs, strict=False):
        cache_resolution(deps, child, spec.repo)

    notes: list[str] = []
    if issue.state_type != "started":
        try:
            await asyncio.to_thread(deps.effects.transition_issue, issue.id, "started")
        except (AuthFailure, GhAuthError, TrackerAuthError):
            raise
        except Exception as exc:  # noqa: BLE001 - the children exist; report, do not fail
            deps.audit.write("plan.transition_failed", target=key, error=f"{type(exc).__name__}: {exc}")
            notes.append(f"parent not moved to started: {exc}")

    await notify_async(
        deps, "plan ready for review", notification_body(deps, issue, children, specs, output, notes), target=key, kind="plan"
    )
    note = f"{len(children)} sub-issue(s) created: " + ", ".join(c.key for c in children)
    if is_recording(deps.effects):
        note += " (dry-run: tracker writes recorded only)"
    if notes:
        note += f" ({'; '.join(notes)})"
    return Outcome(status="done", note=clip(note), **stats)


def plan_prompt(action: Action, deps: Deps, *, issue: Issue, repos_root: str) -> str:
    extra = (
        "- decompose the parent into ordered, independently shippable sub-issues; the engine creates them "
        "and leaves them unlabelled for build\n"
        f"- the managed clones are readable under `{repos_root}` (read-only; do not edit or commit)\n"
        "- `repo` must be one of the slugs under `## Candidate repos`"
    )
    return user_prompt(
        [
            (H_ACTION, render_action_section(action, extra)),
            (H_ISSUE, render_issue_section(issue, deps.tracker)),
            (H_CANDIDATES, render_candidates_section(deps.repos)),
        ]
    )


# --------------------------------------------------------------------------------- validation
def validate_specs(specs: list[SubIssueSpec], repos: RepoIndex) -> tuple[list[SubIssueSpec], list[str]]:
    """Normalise ``repo`` to the managed slug and ``blocked_by`` to sorted unique indices; return every
    problem found (unmanaged repo, empty title, index out of range, self-block, cycle)."""
    problems: list[str] = []
    if not specs:
        return [], ["the plan has no sub-issues"]
    count = len(specs)
    normalised: list[SubIssueSpec] = []
    for index, spec in enumerate(specs):
        label = f"sub-issue {index} ({spec.title.strip() or 'untitled'!r})"
        repo = repos.get(spec.repo) if spec.repo.strip() else None
        if repo is None:
            problems.append(f"{label}: repo {spec.repo!r} is not a managed repo (expected one of {', '.join(repos.slugs()) or 'none'})")
        if not spec.title.strip():
            problems.append(f"sub-issue {index}: empty title")
        valid: set[int] = set()
        for blocker in spec.blocked_by:
            if not 0 <= blocker < count:
                problems.append(f"{label}: blocked_by index {blocker} is out of range (0..{count - 1})")
            elif blocker == index:
                problems.append(f"{label}: is blocked by itself")
            else:
                valid.add(blocker)
        normalised.append(
            spec.model_copy(update={"repo": repo.slug if repo is not None else spec.repo, "blocked_by": sorted(valid)})
        )
    cycle = find_cycle([s.blocked_by for s in normalised])
    if cycle:
        problems.append("blocked_by relations form a cycle: " + " -> ".join(str(i) for i in cycle))
    return normalised, problems


def find_cycle(edges: list[list[int]]) -> list[int]:
    """A cycle in ``edges`` (``edges[i]`` = indices that must finish before ``i``) as the path that
    closes it, e.g. ``[0, 1, 0]``; empty when acyclic."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = [WHITE] * len(edges)
    stack: list[int] = []

    def visit(node: int) -> list[int]:
        colour[node] = GREY
        stack.append(node)
        for nxt in edges[node]:
            if not 0 <= nxt < len(edges):
                continue
            if colour[nxt] == GREY:
                return stack[stack.index(nxt) :] + [nxt]
            if colour[nxt] == WHITE:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        colour[node] = BLACK
        return []

    for start in range(len(edges)):
        if colour[start] == WHITE:
            found = visit(start)
            if found:
                return found
    return []


# -------------------------------------------------------------------------------- after effects
def cache_resolution(deps: Deps, child: Issue, slug: str) -> None:
    """The child's repo is known from creation: ``state.set_resolution`` + the resolver's cache entry
    so ``build`` never re-runs the chain (or the picker) for it."""
    deps.state.set_resolution(child.id, slug, 1.0, RESOLUTION_SOURCE)
    resolution = Resolution(repo=slug, confidence=1.0, source=RESOLUTION_SOURCE, candidates=[slug])
    deps.state.cache_set(RESOLVE_CACHE, child.id, {"updated_at": child.updated_at, "resolution": resolution.model_dump()})


def notification_body(
    deps: Deps, parent: Issue, children: list[Issue], specs: list[SubIssueSpec], output: PlanOutput, notes: list[str]
) -> str:
    lines = [f"{parent.key}: {parent.title}"]
    if parent.url:
        lines.append(parent.url)
    if output.summary.strip():
        lines.append(clip(output.summary, SUMMARY_LIMIT))
    lines.append(f"Sub-issues (label each with `{deps.profile.linear.label_build}` to approve it for build):")
    for index, (child, spec) in enumerate(zip(children, specs, strict=False)):
        blocked = ", ".join(children[b].key for b in spec.blocked_by if 0 <= b < len(children))
        suffix = f" (blocked by {blocked})" if blocked else ""
        lines.append(f"- {index}. {child.key}: {child.title} [{spec.repo}]{suffix}")
    questions = [q for q in output.open_questions if q.strip()]
    if questions:
        lines.append("Open questions:")
        lines += [f"- {q}" for q in questions]
    lines += notes
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- handler
class PlanHandler:
    kind = Kind.PLAN

    async def run(self, action: Action, deps: Deps) -> Outcome:
        return await run_plan(action, deps)


HANDLER = PlanHandler()
