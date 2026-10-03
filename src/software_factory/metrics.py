"""Outcome tracking and the ``factory report`` / weekly digest.

``refresh_outcomes`` asks GitHub (duck-typed ``gh``: ``pr_state`` + ``pr_commits``) about every PR the
factory opened or pushed to and stores, per PR: state (merged / closed-unmerged / open), open ->
merge time, **rework** (human commits after the factory's first push and their line count), and
factory pushes that were later reverted. ``build_report`` folds those outcomes together with the
run log and flags into a ``Report``; ``render_report`` prints it as an aligned table and
``digest_text`` as compact Slack text. These numbers gate each rung of the rollout ladder.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .actions import Kind
from .sources.models import PrCommit, PullRequest
from .state import State, now_iso

TERMINAL_STATES: tuple[str, ...] = ("MERGED", "CLOSED")
MIN_SHA_PREFIX = 7
FAILED_OUTCOMES: tuple[str, ...] = ("failed", "blocked")
COUNTED_OUTCOMES: tuple[str, ...] = ("done", "nothing", "flagged", "kept", "dry_run")  # per-kind buckets
TOP_FLAG_REASONS = 5
KIND_ORDER: dict[str, int] = {k.value: i for i, k in enumerate(Kind)}

_REVERT_RE = re.compile(r"This reverts commit ([0-9a-fA-F]{7,40})", re.IGNORECASE)
_SPEC_RE = re.compile(r"^\s*(\d+)\s*([hdw])\s*$", re.IGNORECASE)


class GhLike(Protocol):
    """The slice of ``sources.github.GitHub`` metrics needs (tests pass a fake)."""

    def pr_state(self, repo: str, number: int) -> PullRequest: ...

    def pr_commits(self, repo: str, number: int) -> list[PrCommit]: ...


# --------------------------------------------------------------------------------- commit math
def rework(commits: list[PrCommit], factory_shas: set[str]) -> tuple[int, int]:
    """``(commits, lines)`` added after the factory's first push that the factory did not push.

    ``commits`` is the PR's commit list, oldest first. Merge commits (``Merge ...`` messages, e.g. a
    human merging the base branch) are not rework; reverts are."""
    first = next((i for i, c in enumerate(commits) if _factory_sha(c.sha, factory_shas)), None)
    if first is None:
        return 0, 0
    count = lines = 0
    for commit in commits[first + 1 :]:
        if _factory_sha(commit.sha, factory_shas) or _is_merge(commit.message):
            continue
        count += 1
        lines += int(commit.additions) + int(commit.deletions)
    return count, lines


def reverted_factory_pushes(commits: list[PrCommit], factory_shas: set[str]) -> list[str]:
    """Factory SHAs named by a ``This reverts commit <sha>`` message (full sha or >= 7-char prefix),
    in the order the reverts appear, de-duplicated."""
    reverted: list[str] = []
    for commit in commits:
        for ref in _REVERT_RE.findall(commit.message or ""):
            match = _factory_sha(ref, factory_shas)
            if match and match not in reverted:
                reverted.append(match)
    return reverted


def _factory_sha(sha: str, factory_shas: set[str]) -> str | None:
    """The factory sha that ``sha`` denotes (exact, or a >= 7-char prefix either way), else ``None``."""
    needle = sha.strip().lower()
    if len(needle) < MIN_SHA_PREFIX:
        return None
    for candidate in factory_shas:
        known = candidate.strip().lower()
        if len(known) >= MIN_SHA_PREFIX and (needle.startswith(known) or known.startswith(needle)):
            return candidate
    return None


def _is_merge(message: str) -> bool:
    return (message or "").lstrip().startswith("Merge ")


# ---------------------------------------------------------------------------- refresh outcomes
def refresh_outcomes(state: State, gh: GhLike, *, force: bool = False) -> int:
    """Refresh the stored outcome of every factory-touched PR; returns how many were refreshed.

    PRs already recorded as merged/closed are skipped unless ``force`` (their history cannot change).
    A failing ``gh`` call records ``refresh_error`` on that PR and moves on."""
    known = {(o["repo"], int(o["number"])): o for o in state.pr_outcomes()}
    refreshed = 0
    for row in _touched_rows(state):
        repo, number = row["repo"], int(row["number"])
        prior = known.get((repo, number))
        if prior is not None and not force and prior.get("state") in TERMINAL_STATES:
            continue
        try:
            pr = gh.pr_state(repo, number)
            commits = gh.pr_commits(repo, number)
        except Exception as exc:  # noqa: BLE001 - one broken PR must not stop the others
            state.upsert_pr_outcome(repo, number, refresh_error=f"{type(exc).__name__}: {exc}", refreshed_at=now_iso())
            continue
        factory_shas = state.pushed_shas(repo, number)
        pushes = state.pushes(repo, number)
        kinds: list[str] = []
        if row.get("factory_opened") and row.get("kind"):
            kinds.append(str(row["kind"]))
        for push in pushes:
            if push["kind"] not in kinds:
                kinds.append(str(push["kind"]))
        rework_commits, rework_lines = rework(commits, factory_shas)
        state.upsert_pr_outcome(
            repo,
            number,
            state=pr.state,
            title=pr.title,
            url=pr.url or row.get("url") or "",
            created_at=pr.created_at,
            merged_at=pr.merged_at,
            closed_at=pr.closed_at,
            hours_to_merge=_hours_between(pr.created_at, pr.merged_at),
            kind=kinds[0] if kinds else None,
            kinds=kinds,
            factory_opened=bool(row.get("factory_opened")),
            issue_key=row.get("issue_key"),
            factory_pushes=len(pushes),
            factory_commits=len(factory_shas),
            rework_commits=rework_commits,
            rework_lines=rework_lines,
            reverted=reverted_factory_pushes(commits, factory_shas),
            refreshed_at=now_iso(),
            refresh_error=None,
        )
        refreshed += 1
    return refreshed


def _touched_rows(state: State) -> list[dict]:
    """``touched_prs()`` already includes every factory-opened PR; union defensively anyway."""
    rows: dict[tuple[str, int], dict] = {}
    for row in state.touched_prs():
        rows[(row["repo"], int(row["number"]))] = dict(row)
    for pr in state.factory_prs():
        key = (pr["repo"], int(pr["number"]))
        rows.setdefault(key, {**pr, "factory_opened": True, "pushes": 0})
    return [rows[key] for key in sorted(rows)]


# ---------------------------------------------------------------------------------- report
class Report(BaseModel):
    since: str = ""
    runs: int = 0
    prs: int = 0  # factory-touched PRs in the window (open ones always count)
    # per kind: runs, done, nothing, failed, flagged, kept, dry_run, pushed, cost_usd,
    #           prs, merged, closed, merge_rate (merged / decided; None when nothing decided)
    per_kind: dict[str, dict] = Field(default_factory=dict)
    merged: int = 0
    closed_unmerged: int = 0
    open: int = 0
    median_hours_to_merge: float | None = None
    rework_rate: float | None = None  # PRs with >= 1 rework commit / factory-touched PRs
    reverted_pushes: int = 0
    cost_total: float = 0.0
    cost_per_merged_pr: float | None = None
    top_flag_reasons: list[tuple[str, int]] = Field(default_factory=list)


def _empty_kind() -> dict[str, Any]:
    return {
        "runs": 0,
        "done": 0,
        "nothing": 0,
        "failed": 0,
        "flagged": 0,
        "kept": 0,
        "dry_run": 0,
        "pushed": 0,
        "cost_usd": 0.0,
        "prs": 0,
        "merged": 0,
        "closed": 0,
        "merge_rate": None,
    }


def build_report(state: State, since_iso: str) -> Report:
    """Aggregate runs, PR outcomes and flags since ``since_iso`` (empty = everything)."""
    since = parse_iso(since_iso) if since_iso else None
    per_kind: dict[str, dict] = {}

    runs = state.runs(since_iso or None)
    cost_total = 0.0
    for run in runs:
        bucket = per_kind.setdefault(str(run.get("kind") or "unknown"), _empty_kind())
        bucket["runs"] += 1
        outcome = str(run.get("outcome") or "")
        if outcome in FAILED_OUTCOMES:
            bucket["failed"] += 1
        elif outcome in COUNTED_OUTCOMES:
            bucket[outcome] += 1
        if run.get("pushed"):
            bucket["pushed"] += 1
        cost = run.get("cost_usd")
        if cost is not None:
            bucket["cost_usd"] += float(cost)
            cost_total += float(cost)

    prs = [o for o in state.pr_outcomes() if _in_window(o, since)]
    merged = [o for o in prs if o.get("state") == "MERGED"]
    closed = [o for o in prs if o.get("state") == "CLOSED"]
    for outcome in prs:
        kinds = [str(k) for k in (outcome.get("kinds") or []) if k] or (
            [str(outcome["kind"])] if outcome.get("kind") else []
        )
        for kind in kinds:
            bucket = per_kind.setdefault(kind, _empty_kind())
            bucket["prs"] += 1
            if outcome.get("state") == "MERGED":
                bucket["merged"] += 1
            elif outcome.get("state") == "CLOSED":
                bucket["closed"] += 1
    for bucket in per_kind.values():
        decided = bucket["merged"] + bucket["closed"]
        bucket["merge_rate"] = bucket["merged"] / decided if decided else None
        bucket["cost_usd"] = round(bucket["cost_usd"], 4)

    hours = [float(o["hours_to_merge"]) for o in merged if o.get("hours_to_merge") is not None]
    flags = [f for f in state.flags(open_only=False) if since is None or _at_or_after(f.get("created_at"), since)]
    reasons = Counter(str(f.get("reason") or "") for f in flags)

    return Report(
        since=since_iso or "",
        runs=len(runs),
        prs=len(prs),
        per_kind={k: per_kind[k] for k in sorted(per_kind, key=lambda k: (KIND_ORDER.get(k, 99), k))},
        merged=len(merged),
        closed_unmerged=len(closed),
        open=len(prs) - len(merged) - len(closed),
        median_hours_to_merge=round(statistics.median(hours), 1) if hours else None,
        rework_rate=(sum(1 for o in prs if o.get("rework_commits")) / len(prs)) if prs else None,
        reverted_pushes=sum(len(o.get("reverted") or []) for o in prs),
        cost_total=round(cost_total, 4),
        cost_per_merged_pr=round(cost_total / len(merged), 4) if merged else None,
        top_flag_reasons=reasons.most_common(TOP_FLAG_REASONS),
    )


def _in_window(outcome: dict, since: datetime | None) -> bool:
    """Open PRs always count; merged/closed ones count when they were decided inside the window."""
    if since is None:
        return True
    state = outcome.get("state")
    if state == "MERGED":
        stamp = outcome.get("merged_at") or outcome.get("closed_at") or outcome.get("updated_at")
    elif state == "CLOSED":
        stamp = outcome.get("closed_at") or outcome.get("merged_at") or outcome.get("updated_at")
    else:
        return True
    return _at_or_after(stamp, since)


# ---------------------------------------------------------------------------------- text
def render_report(r: Report) -> str:
    """Aligned plain-text table plus the headline numbers (what ``factory report`` prints)."""
    headers = ["kind", "runs", "done", "failed", "flagged", "pushed", "cost", "PRs", "merged", "merge"]
    rows: list[list[str]] = []
    for kind, b in r.per_kind.items():
        rows.append(
            [
                kind,
                str(b["runs"]),
                str(b["done"]),
                str(b["failed"]),
                str(b["flagged"]),
                str(b["pushed"]),
                _money(b["cost_usd"]),
                str(b["prs"]),
                str(b["merged"]),
                _pct(b["merge_rate"]),
            ]
        )
    decided = r.merged + r.closed_unmerged
    rows.append(
        [
            "total",
            str(r.runs),
            str(sum(b["done"] for b in r.per_kind.values())),
            str(sum(b["failed"] for b in r.per_kind.values())),
            str(sum(b["flagged"] for b in r.per_kind.values())),
            str(sum(b["pushed"] for b in r.per_kind.values())),
            _money(r.cost_total),
            str(r.prs),
            str(r.merged),
            _pct(r.merged / decided if decided else None),
        ]
    )
    lines = [f"Factory report{' since ' + r.since if r.since else ''}", ""]
    lines += _table(headers, rows, left_aligned={0})
    reworked = round(r.rework_rate * r.prs) if r.rework_rate is not None else 0
    lines += [
        "",
        f"PRs touched: {r.prs} (merged {r.merged}, closed unmerged {r.closed_unmerged}, open {r.open})",
        "Median hours to merge: " + (f"{r.median_hours_to_merge:g}" if r.median_hours_to_merge is not None else "n/a"),
        "Rework rate: " + (f"{_pct(r.rework_rate)} ({reworked}/{r.prs} PRs)" if r.rework_rate is not None else "n/a"),
        f"Reverted factory pushes: {r.reverted_pushes}",
        f"Cost: {_money(r.cost_total)} total, "
        + (f"{_money(r.cost_per_merged_pr)} per merged PR" if r.cost_per_merged_pr is not None else "no merged PRs"),
    ]
    if r.top_flag_reasons:
        width = max(len(str(n)) for _, n in r.top_flag_reasons)
        lines.append("Top flag reasons:")
        lines += [f"  {n:>{width}}  {reason}" for reason, n in r.top_flag_reasons]
    else:
        lines.append("Top flag reasons: none")
    return "\n".join(lines) + "\n"


def digest_text(r: Report) -> str:
    """Compact Slack-friendly summary (mrkdwn bold, one line per topic)."""
    done = sum(b["done"] for b in r.per_kind.values())
    failed = sum(b["failed"] for b in r.per_kind.values())
    pushed = sum(b["pushed"] for b in r.per_kind.values())
    rates = ", ".join(
        f"{kind} {_pct(b['merge_rate'])} ({b['merged']}/{b['merged'] + b['closed']})"
        for kind, b in r.per_kind.items()
        if b["merge_rate"] is not None
    )
    merge_line = f"PRs: {r.merged} merged, {r.closed_unmerged} closed, {r.open} open"
    if r.median_hours_to_merge is not None:
        merge_line += f" | median {r.median_hours_to_merge:g}h to merge"
    if r.cost_per_merged_pr is not None:
        merge_line += f" | {_money(r.cost_per_merged_pr)} per merged PR"
    lines = [
        f"*Factory digest*{' since ' + r.since[:10] if r.since else ''}",
        f"Runs: {r.runs} ({done} done, {failed} failed, {pushed} pushed) | cost {_money(r.cost_total)}",
        merge_line,
        f"Merge rate: {rates or 'n/a'}",
        f"Rework: {_pct(r.rework_rate)} of touched PRs | {r.reverted_pushes} reverted pushes",
        "Top flags: " + (", ".join(f"{reason} ({n})" for reason, n in r.top_flag_reasons) or "none"),
    ]
    return "\n".join(lines)


def _table(headers: list[str], rows: list[list[str]], *, left_aligned: set[int]) -> list[str]:
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        return "  ".join(
            c.ljust(widths[i]) if i in left_aligned else c.rjust(widths[i]) for i, c in enumerate(cells)
        ).rstrip()

    return [fmt(headers), *(fmt(row) for row in rows)]


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{round(value * 100)}%"


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


# ------------------------------------------------------------------------------------ time
def parse_iso(value: str | None) -> datetime | None:
    """ISO-8601 (``Z`` or offset; naive = UTC) -> aware UTC datetime; ``None`` when unparseable."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def since_iso_from_spec(spec: str, now: datetime | None = None) -> str:
    """``"7d"`` / ``"36h"`` / ``"2w"`` -> ISO timestamp that long before ``now``; an ISO timestamp
    passes through normalised. Anything else raises ``ValueError``."""
    match = _SPEC_RE.match(spec or "")
    if match:
        amount, unit = int(match.group(1)), match.group(2).lower()
        delta = {"h": timedelta(hours=amount), "d": timedelta(days=amount), "w": timedelta(weeks=amount)}[unit]
        base = now if now is not None else datetime.now(UTC)
        return (base - delta).astimezone(UTC).isoformat(timespec="seconds")
    parsed = parse_iso(spec)
    if parsed is None:
        raise ValueError(f"invalid --since {spec!r}: use e.g. 7d, 36h, 2w or an ISO-8601 timestamp")
    return parsed.isoformat(timespec="seconds")


def _hours_between(start: str | None, end: str | None) -> float | None:
    a, b = parse_iso(start), parse_iso(end)
    if a is None or b is None:
        return None
    return round((b - a).total_seconds() / 3600, 2)


def _at_or_after(stamp: Any, since: datetime) -> bool:
    parsed = parse_iso(stamp if isinstance(stamp, str) else None)
    return parsed is not None and parsed >= since
