"""Scan: external state (GitHub + tracker) -> ``Action[]``, then deterministic filters.

No LLM is involved anywhere here. Detection follows the table in ``docs/architecture.md`` ("Scan");
every filter produces a ``Skip`` with a human-readable reason so ``tick --dry-run`` can explain why
something was *not* dispatched. Non-auth GitHub / tracker failures become ``ScanResult.errors``;
auth failures (``GhAuthError``, ``TrackerAuthError``) propagate so the dispatcher can stop the wave.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .actions import Action, Kind
from .config import Profile
from .repos import RepoIndex
from .sources.github import GhAuthError, GhError, GitHub, parse_iso
from .sources.models import Feedback, PullRequest
from .state import State
from .trackers.base import Issue, Tracker, TrackerAuthError, TrackerError

_PR_URL = re.compile(r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/(\d+)")


class FlagSpec(BaseModel):
    """A flag the dispatcher should raise for a skip (``state.add_flag`` + notify, flag-once by
    ``dedupe_key``). The scan itself never writes state or notifies, so ``--dry-run`` stays pure."""

    reason: str
    detail: str = ""
    dedupe_key: str | None = None
    level: str = "warn"


class Skip(BaseModel):
    action: Action
    reason: str
    flag: FlagSpec | None = None  # set when the human must hear about this skip (attempts exhausted)


class ScanResult(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    skipped: list[Skip] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------------- github


def scan_github(
    profile: Profile,
    gh: GitHub,
    repos: RepoIndex,
    state: State,
    *,
    errors: list[str] | None = None,
) -> list[Action]:
    """``ci`` / ``conflict`` / ``rc`` for my open PRs in managed repos, ``rv`` for direct review
    requests in managed repos. Per-PR ``GhError``s are appended to ``errors`` (auth errors raise)."""
    errors = errors if errors is not None else []
    managed = repos.slugs()
    actions: list[Action] = []

    for basic in gh.my_open_prs(managed, profile.authored_max_age_days):
        slug = managed_slug(repos, basic.repo)
        try:
            pr = gh.pr_details(slug, basic.number)
        except GhAuthError:
            raise
        except GhError as exc:
            errors.append(f"github: {slug}#{basic.number}: {exc}")
            continue
        if pr.state != "OPEN":
            continue
        actions.extend(pr_actions(pr, gh=gh, state=state, errors=errors))

    for pr in gh.review_requested(profile.review_max_age_days):
        if repos.get(pr.repo) is None:
            continue  # no local clone to review in
        if pr.is_draft or not pr.review_requested_from_me or pr.state != "OPEN":
            continue
        slug = managed_slug(repos, pr.repo)
        if slug != pr.repo:
            pr = pr.model_copy(update={"repo": slug})
        actions.append(
            Action(
                kind=Kind.RV,
                version=pr.head_ref_oid,
                title=pr.title,
                repo=pr.repo,
                pr=pr.number,
                url=pr.url,
                context={**_pr_context(pr), "author": pr.author},
            )
        )
    return actions


def managed_slug(repos: RepoIndex, slug: str) -> str:
    """The managed repo's own slug for ``slug`` (GitHub reports its canonical casing, the local
    clone's origin URL may differ: GitHub slugs are case-insensitive). Every PR key the engine
    derives -- action targets, ``gh`` calls, state rows (pushes, factory PRs, kept worktrees) --
    uses the clone's spelling, which is also what ``factory run`` and the effects record under,
    so tick-derived and clone-derived keys agree. Unmanaged slugs pass through unchanged."""
    repo = repos.get(slug)
    return repo.slug if repo is not None else slug


def pr_actions(
    pr: PullRequest, *, gh: GitHub, state: State, errors: list[str] | None = None
) -> list[Action]:
    """Upkeep actions for one fully-populated PR (``gh.pr_details``): ``ci``, ``conflict``, ``rc``."""
    errors = errors if errors is not None else []
    actions: list[Action] = []
    base = {"title": pr.title, "repo": pr.repo, "pr": pr.number, "url": pr.url}

    failing = pr.failing_checks  # pending checks never trigger ci
    if failing:
        actions.append(
            Action(
                kind=Kind.CI,
                version=pr.head_ref_oid,
                context={
                    **_pr_context(pr),
                    "failing": [
                        {
                            "name": c.name,
                            "run_id": c.run_id,
                            "job_id": c.job_id,
                            "workflow": c.workflow,
                            "details_url": c.details_url,
                        }
                        for c in failing
                    ],
                },
                **base,
            )
        )
    if pr.mergeable == "CONFLICTING":
        actions.append(
            Action(
                kind=Kind.CONFLICT,
                version=f"{pr.head_ref_oid}:{pr.base_ref_oid}",
                context=_pr_context(pr),
                **base,
            )
        )
    rc = rc_action(pr, gh=gh, state=state, errors=errors)
    if rc is not None:
        actions.append(rc)
    return actions


def rc_action(
    pr: PullRequest, *, gh: GitHub, state: State, errors: list[str] | None = None
) -> Action | None:
    """``rc`` when there is feedback (not mine, not ignored, not on a resolved thread) newer than my
    last commit / the factory's last push, or an unresolved non-outdated thread that carries such
    feedback. Version = ISO timestamp of the latest feedback item over *all* of ``pr.feedback``
    (resolved-thread comments included): that set only grows, so the key never regresses to an
    older, never-recorded value when a reviewer resolves a thread, and ``kinds/_pr.py`` computes
    the same key from the same list for its stale check."""
    candidates = pr.open_feedback
    if not candidates:
        return None
    errors = errors if errors is not None else []
    feedback_ids = {f.id for f in candidates}
    last_mine = last_own_activity(pr, gh=gh, state=state, errors=errors)
    fresh = [f for f in candidates if last_mine is None or _ts(f) > last_mine]
    open_threads = [t for t in pr.unresolved_threads if any(c.id in feedback_ids for c in t.comments)]
    if not fresh and not open_threads:
        return None
    latest = max(pr.feedback, key=_ts)
    ids: list[str] = []
    for item in fresh + [c for t in open_threads for c in t.comments if c.id in feedback_ids]:
        if item.id not in ids:
            ids.append(item.id)
    return Action(
        kind=Kind.RC,
        version=latest.created_at,
        title=pr.title,
        repo=pr.repo,
        pr=pr.number,
        url=pr.url,
        context={
            **_pr_context(pr),
            "feedback_ids": ids,
            "unresolved_thread_ids": [t.id for t in open_threads],
            "latest_feedback_at": latest.created_at,
            "last_own_activity_at": last_mine.isoformat(timespec="seconds") if last_mine else None,
        },
    )


def last_own_activity(
    pr: PullRequest, *, gh: GitHub, state: State, errors: list[str] | None = None
) -> datetime | None:
    """``max(latest commit committedDate on the PR, latest factory push to the PR)``; ``None`` when
    neither is known."""
    errors = errors if errors is not None else []
    stamps: list[datetime] = []
    try:
        commits = gh.pr_commits(pr.repo, pr.number)
    except GhAuthError:
        raise
    except GhError as exc:
        errors.append(f"github: {pr.target}: commits: {exc}")
        commits = []
    for commit in commits:
        if (parsed := parse_iso(commit.committed_at)) is not None:
            stamps.append(parsed)
    for push in state.pushes(pr.repo, pr.number):
        if (parsed := parse_iso(push.get("ts"))) is not None:
            stamps.append(parsed)
    return max(stamps) if stamps else None


def _pr_context(pr: PullRequest) -> dict:
    return {
        "head_ref_name": pr.head_ref_name,
        "head_ref_oid": pr.head_ref_oid,
        "base_ref_name": pr.base_ref_name,
        "base_ref_oid": pr.base_ref_oid,
        "head_repo": pr.head_repo,
        "is_fork": pr.is_fork,
        "is_draft": pr.is_draft,
    }


def _ts(item: Feedback) -> datetime:
    parsed = parse_iso(item.created_at)
    return parsed if parsed is not None else datetime.min.replace(tzinfo=UTC)


# ------------------------------------------------------------------------------------ tracker


def scan_tracker(
    profile: Profile,
    tracker: Tracker,
    state: State,
    gh: GitHub | None = None,
    *,
    errors: list[str] | None = None,
) -> list[Action]:
    """``refine`` / ``plan`` / ``build`` from labeled issues. Blockers are *not* filtered here (they
    become a ``Skip`` in ``filter_actions``); an open linked PR suppresses ``build`` entirely."""
    errors = errors if errors is not None else []
    cfg = profile.linear
    if not cfg.enabled:
        return []
    actions: list[Action] = []

    for issue in _issues(tracker, cfg.label_refine, profile, errors):
        actions.append(_issue_action(Kind.REFINE, issue, version=issue.updated_at))

    for issue in _issues(tracker, cfg.label_plan, profile, errors):
        if issue.children:
            continue
        actions.append(_issue_action(Kind.PLAN, issue, version=issue.updated_at))

    for issue in _issues(tracker, cfg.label_build, profile, errors):
        if issue.state_type != "unstarted":
            continue
        linked = linked_open_pr(issue, gh, errors)
        if linked is not None:
            continue
        actions.append(
            _issue_action(
                Kind.BUILD,
                issue,
                version=issue.id,
                extra={
                    "branch_name": issue.branch_name,
                    "team_id": issue.team_id,
                    "unfinished_blockers": [b.key for b in issue.unfinished_blockers],
                },
            )
        )
    return actions


def linked_open_pr(issue: Issue, gh: GitHub | None, errors: list[str] | None = None) -> str | None:
    """``owner/name#n`` of an open PR attached to the issue, else ``None``. Without ``gh`` (or when
    ``gh`` fails) any PR attachment counts as open: never open a second PR for the same issue."""
    errors = errors if errors is not None else []
    for attachment in issue.attachments:
        match = _PR_URL.search(attachment.url or "")
        if not match:
            continue
        repo, number = f"{match.group(1)}/{match.group(2)}", int(match.group(3))
        target = f"{repo}#{number}"
        if gh is None:
            return target
        try:
            pr = gh.pr_state(repo, number)
        except GhAuthError:
            raise
        except GhError as exc:
            errors.append(f"github: {issue.key}: {target}: {exc}")
            return target
        if pr.state == "OPEN":
            return target
    return None


def _issues(tracker: Tracker, label: str, profile: Profile, errors: list[str]) -> list[Issue]:
    try:
        issues = tracker.list_issues_with_label(label)
    except TrackerAuthError:
        raise
    except TrackerError as exc:
        errors.append(f"tracker: list issues with label {label!r}: {exc}")
        return []
    teams = {k.lower() for k in profile.linear.team_keys}
    seen: set[str] = set()
    kept: list[Issue] = []
    for issue in sorted(issues, key=lambda i: i.key):
        if issue.id in seen or (teams and issue.team_key.lower() not in teams):
            continue
        seen.add(issue.id)
        kept.append(issue)
    return kept


def _issue_action(kind: Kind, issue: Issue, *, version: str, extra: dict | None = None) -> Action:
    return Action(
        kind=kind,
        version=version,
        title=issue.title,
        issue_id=issue.id,
        issue_key=issue.key,
        url=issue.url,
        context={"issue_id": issue.id, "labels": list(issue.labels), "state_type": issue.state_type, **(extra or {})},
    )


# ------------------------------------------------------------------------------------ filters


def filter_actions(
    profile: Profile,
    actions: list[Action],
    state: State,
    tracker: Tracker | None = None,
) -> tuple[list[Action], list[Skip]]:
    """Apply the dedupe / attempts / kept-worktree / enabled-kind / blocked filters. Duplicate
    ``(target, kind, version)`` entries within one scan are collapsed silently."""
    kept: list[Action] = []
    skipped: list[Skip] = []
    seen: set[tuple[str, str, str]] = set()
    for action in actions:
        if action.dedupe_key in seen:
            continue
        seen.add(action.dedupe_key)
        skip = skip_for(profile, action, state, tracker)
        if skip is None:
            kept.append(action)
        else:
            skipped.append(skip)
    return kept, skipped


def skip_reason(profile: Profile, action: Action, state: State, tracker: Tracker | None = None) -> str | None:
    """Why ``action`` must not be dispatched now, or ``None``."""
    skip = skip_for(profile, action, state, tracker)
    return skip.reason if skip is not None else None


def skip_for(profile: Profile, action: Action, state: State, tracker: Tracker | None = None) -> Skip | None:
    """The ``Skip`` for ``action`` (reason + the flag the human should get, if any), or ``None``."""
    target, kind, version = action.dedupe_key
    if profile.mode == "full" and not profile.kind_enabled(action.kind):
        return Skip(action=action, reason="kind disabled")  # observe mode keeps everything: non-rv kinds dry-run
    if state.is_handled(target, kind, version):
        return Skip(action=action, reason="handled at this version")
    attempts = state.attempts(target, kind, version)
    if attempts >= profile.max_attempts:
        return Skip(
            action=action,
            reason=f"attempts exhausted ({attempts}/{profile.max_attempts})",
            flag=exhausted_flag(profile, action, attempts, state.last_attempt(target, kind, version)),
        )
    kept = state.kept_worktree(target)
    if kept is not None:
        detail = f": {kept.get('reason')}" if kept.get("reason") else ""
        return Skip(action=action, reason=f"kept worktree awaiting me{detail}")
    if action.kind == Kind.BUILD:
        blockers = unfinished_blockers(action, tracker)
        if blockers:
            return Skip(action=action, reason="blocked by " + ", ".join(blockers))
    return None


EXHAUSTED_REASON = "attempts exhausted"


def exhausted_flag(profile: Profile, action: Action, attempts: int, last: dict | None = None) -> FlagSpec:
    """Flag-once (per target/kind/version) telling the human that the cap was hit, how the last
    attempt ended, what re-arms the action and the manual retry command. Without it the skip is
    only ever a line in the tick output, and a ``build`` issue would sit labelled and unstarted
    indefinitely: its version is the issue id, so editing the issue does not re-arm it."""
    target, kind, version = action.dedupe_key
    lines = [
        f"{kind} {target} failed {attempts} attempt(s) at version {version} "
        f"(max_attempts = {profile.max_attempts}); it is not retried at this version."
    ]
    if last:
        note = " ".join(str(last.get("note") or "").split())
        lines.append(f"last attempt ({last.get('ts', '?')}): {last.get('outcome', '?')}" + (f" - {note}" if note else ""))
    if action.kind == Kind.BUILD:
        lines.append(
            "The build version is the issue id, so editing the issue does not re-arm it: address the "
            f"blocker, then retry by hand with `factory run build {target}`."
        )
    elif action.kind in (Kind.REFINE, Kind.PLAN):
        lines.append(f"Editing the issue re-arms it (new updated_at); `factory run {kind} {target}` retries now.")
    else:
        lines.append(
            "A new head, base or feedback on the PR re-arms it (new version); "
            f"`factory run {kind} {target}` retries now."
        )
    if action.url:
        lines.append(action.url)
    return FlagSpec(
        reason=EXHAUSTED_REASON,
        detail="\n".join(lines),
        dedupe_key=f"exhausted:{target}:{kind}:{version}",
    )


def unfinished_blockers(action: Action, tracker: Tracker | None) -> list[str]:
    """Blocker keys from the scan context, else a fresh ``tracker.get_issue`` (``run build KEY``)."""
    if "unfinished_blockers" in action.context:
        return [str(k) for k in action.context["unfinished_blockers"]]
    if tracker is None or not (action.issue_id or action.issue_key):
        return []
    try:
        issue = tracker.get_issue(action.issue_id or action.issue_key or "")
    except TrackerAuthError:
        raise
    except TrackerError:
        return []
    return [b.key for b in issue.unfinished_blockers]


# ------------------------------------------------------------------------------------- facade


def scan(
    profile: Profile,
    *,
    gh: GitHub | None,
    tracker: Tracker | None,
    repos: RepoIndex,
    state: State,
) -> ScanResult:
    """Full scan + filters. ``gh`` / ``tracker`` may be ``None`` (that source is skipped)."""
    errors: list[str] = []
    actions: list[Action] = []
    if gh is not None:
        try:
            actions.extend(scan_github(profile, gh, repos, state, errors=errors))
        except GhAuthError:
            raise
        except GhError as exc:
            errors.append(f"github: {exc}")
    if tracker is not None:
        try:
            actions.extend(scan_tracker(profile, tracker, state, gh=gh, errors=errors))
        except TrackerAuthError:
            raise
        except TrackerError as exc:
            errors.append(f"tracker: {exc}")
    kept, skipped = filter_actions(profile, actions, state, tracker)
    kept.sort(key=lambda a: a.sort_key)
    return ScanResult(actions=kept, skipped=skipped, errors=errors)
