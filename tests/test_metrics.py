"""metrics.py: rework/revert detection from commit lists, outcome refresh with a fake gh, report
aggregation from a seeded State, text rendering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from software_factory.metrics import (
    Report,
    build_report,
    digest_text,
    parse_iso,
    refresh_outcomes,
    render_report,
    reverted_factory_pushes,
    rework,
    since_iso_from_spec,
)
from software_factory.sources.models import PrCommit, PullRequest

F1 = "aaaaaaaa11111111aaaaaaaa11111111aaaaaaaa"
F2 = "bbbbbbbb22222222bbbbbbbb22222222bbbbbbbb"
H1 = "cccccccc33333333cccccccc33333333cccccccc"
H2 = "dddddddd44444444dddddddd44444444dddddddd"
H3 = "eeeeeeee55555555eeeeeeee55555555eeeeeeee"


def _c(sha: str, message: str = "", *, add: int = 0, dele: int = 0, author: str = "human") -> PrCommit:
    return PrCommit(sha=sha, author_login=author, message=message, additions=add, deletions=dele)


def _iso(days_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------- rework
def test_rework_counts_only_human_commits_after_first_factory_push():
    commits = [
        _c(H1, "human before factory", add=100, dele=50),
        _c(F1, "factory: fix lint", add=3, dele=1, author="me"),
        _c(H2, "human fixes factory's work", add=10, dele=4),
        _c(F2, "factory: address review", add=2, dele=2, author="me"),
        _c(H3, "human polish", add=5, dele=0),
    ]
    assert rework(commits, {F1, F2}) == (2, 19)
    assert rework(commits, {F2}) == (1, 5)  # only commits after the first *known* factory sha
    assert rework(commits, set()) == (0, 0)
    assert rework([], {F1}) == (0, 0)


def test_rework_ignores_merge_commits_and_matches_sha_prefixes():
    commits = [
        _c(F1[:7], "factory"),
        _c(H1, "Merge branch 'main' into feat", add=400, dele=20),
        _c(H2, "Merge remote-tracking branch 'origin/main'", add=1, dele=1),
        _c(H3, "real rework", add=7, dele=3),
    ]
    assert rework(commits, {F1}) == (1, 10)  # full factory sha vs 7-char commit sha
    assert rework(commits, {F1[:7].upper()}) == (1, 10)  # prefix in state, case-insensitive
    assert rework(commits, {F1[:6]}) == (0, 0)  # too short to be trusted


def test_rework_counts_reverts_of_factory_commits_as_rework():
    commits = [_c(F1, "factory"), _c(H1, f'Revert "factory"\n\nThis reverts commit {F1}.', add=1, dele=3)]
    assert rework(commits, {F1}) == (1, 4)


# ---------------------------------------------------------------------------------- reverts
def test_reverted_factory_pushes_full_and_prefix_matches():
    commits = [
        _c(F1, "factory one"),
        _c(F2, "factory two"),
        _c(H1, f'Revert "factory one"\n\nThis reverts commit {F1}.'),
        _c(H2, f"Revert again\n\nthis reverts commit {F2[:7]}\nand This reverts commit {H1[:10]}."),
        _c(H3, f"Revert of a stranger\n\nThis reverts commit {H1}."),
    ]
    assert reverted_factory_pushes(commits, {F1, F2}) == [F1, F2]
    assert reverted_factory_pushes(commits, {F2}) == [F2]
    assert reverted_factory_pushes(commits, set()) == []
    assert reverted_factory_pushes(commits, {F1[:7]}) == [F1[:7]]  # returns the sha as stored in state


def test_reverted_factory_pushes_dedupes_and_ignores_short_refs():
    commits = [
        _c(H1, f"This reverts commit {F1}."),
        _c(H2, f"This reverts commit {F1[:12]}."),
        _c(H3, f"This reverts commit {F1[:5]}."),
    ]
    assert reverted_factory_pushes(commits, {F1}) == [F1]


# -------------------------------------------------------------------------- refresh outcomes
class FakeGh:
    def __init__(self, prs: dict[tuple[str, int], PullRequest], commits: dict[tuple[str, int], list[PrCommit]]):
        self.prs = prs
        self.commits = commits
        self.calls: list[tuple[str, str, int]] = []

    def pr_state(self, repo: str, number: int) -> PullRequest:
        self.calls.append(("pr_state", repo, number))
        if (repo, number) not in self.prs:
            raise RuntimeError(f"gh: no PR {repo}#{number}")
        return self.prs[(repo, number)]

    def pr_commits(self, repo: str, number: int) -> list[PrCommit]:
        self.calls.append(("pr_commits", repo, number))
        return self.commits.get((repo, number), [])


def _pr(repo: str, number: int, state: str, *, created: str, merged: str | None = None, closed: str | None = None) -> PullRequest:
    return PullRequest(
        repo=repo,
        number=number,
        title=f"PR {number}",
        url=f"https://github.com/{repo}/pull/{number}",
        state=state,
        created_at=created,
        merged_at=merged,
        closed_at=closed,
    )


def test_refresh_outcomes_records_state_rework_reverts_and_kinds(state):
    repo = "JaneDoe/game"
    state.register_pr(repo, 1, "feat/abc-1", "build", issue_key="ABC-1", url="https://x/1")
    state.record_push(repo, 1, "feat/abc-1", "build", "", F1, [F1])
    state.record_push(repo, 1, "feat/abc-1", "ci", F1, F2, [F2])
    state.record_push(repo, 5, "feat/human", "rc", "x", H3, [H3])  # touched-only PR
    state.record_push(repo, None, "factory/evidence", "build", "", "zzzzzzzz", ["zzzzzzzz"])
    gh = FakeGh(
        {
            (repo, 1): _pr(repo, 1, "MERGED", created="2026-09-01T00:00:00Z", merged="2026-09-02T12:00:00Z"),
            (repo, 5): _pr(repo, 5, "OPEN", created="2026-09-20T00:00:00Z"),
        },
        {
            (repo, 1): [
                _c(H1, "human start", add=50),
                _c(F1, "factory build", add=20, author="me"),
                _c(H2, "human tweak", add=4, dele=2),
                _c(F2, "factory ci fix", add=1, author="me"),
                _c(H3, f"Revert ci fix\n\nThis reverts commit {F2[:7]}.", add=0, dele=1),
            ],
            (repo, 5): [_c(H1, "human"), _c(H3, "factory rc", author="me")],
        },
    )

    assert refresh_outcomes(state, gh) == 2

    outcomes = {(o["repo"], o["number"]): o for o in state.pr_outcomes()}
    one = outcomes[(repo, 1)]
    assert one["state"] == "MERGED" and one["merged_at"] == "2026-09-02T12:00:00Z"
    assert one["hours_to_merge"] == 36.0
    assert one["kind"] == "build" and one["kinds"] == ["build", "ci"]
    assert one["factory_opened"] is True and one["issue_key"] == "ABC-1"
    assert one["factory_pushes"] == 2 and one["factory_commits"] == 2
    assert one["rework_commits"] == 2 and one["rework_lines"] == 7
    assert one["reverted"] == [F2]
    assert one["refresh_error"] is None and one["refreshed_at"]
    assert one["url"] == "https://github.com/JaneDoe/game/pull/1"
    five = outcomes[(repo, 5)]
    assert five["state"] == "OPEN" and five["hours_to_merge"] is None
    assert five["kind"] == "rc" and five["kinds"] == ["rc"] and five["factory_opened"] is False
    assert five["rework_commits"] == 0 and five["reverted"] == []
    assert (repo, None) not in outcomes and len(outcomes) == 2


def test_refresh_outcomes_skips_terminal_prs_unless_forced_and_tolerates_errors(state):
    repo = "o/r"
    state.register_pr(repo, 1, "b1", "build")
    state.register_pr(repo, 2, "b2", "build")
    state.register_pr(repo, 3, "b3", "ready")
    gh = FakeGh(
        {
            (repo, 1): _pr(repo, 1, "MERGED", created="2026-09-01T00:00:00Z", merged="2026-09-01T06:00:00Z"),
            (repo, 2): _pr(repo, 2, "OPEN", created="2026-09-01T00:00:00Z"),
            # PR 3 is missing: gh raises
        },
        {},
    )
    assert refresh_outcomes(state, gh) == 2
    three = next(o for o in state.pr_outcomes() if o["number"] == 3)
    assert three["refresh_error"].startswith("RuntimeError: gh: no PR o/r#3")
    assert "state" not in three

    gh.calls.clear()
    assert refresh_outcomes(state, gh) == 1  # merged PR skipped, open PR + failing PR retried
    assert ("pr_state", repo, 1) not in gh.calls
    assert ("pr_state", repo, 2) in gh.calls and ("pr_state", repo, 3) in gh.calls

    gh.calls.clear()
    gh.prs[(repo, 3)] = _pr(repo, 3, "CLOSED", created="2026-09-01T00:00:00Z", closed="2026-09-03T00:00:00Z")
    assert refresh_outcomes(state, gh, force=True) == 3
    assert ("pr_state", repo, 1) in gh.calls
    three = next(o for o in state.pr_outcomes() if o["number"] == 3)
    assert three["state"] == "CLOSED" and three["refresh_error"] is None and three["hours_to_merge"] is None


# ---------------------------------------------------------------------------------- report
def _seed(state) -> None:
    runs = [
        {"ts": _iso(1), "kind": "ci", "target": "o/r#1", "outcome": "done", "pushed": True, "cost_usd": 0.5},
        {"ts": _iso(2), "kind": "ci", "target": "o/r#2", "outcome": "done", "pushed": True, "cost_usd": 0.7},
        {"ts": _iso(2), "kind": "ci", "target": "o/r#3", "outcome": "failed", "pushed": False, "cost_usd": 0.2},
        {"ts": _iso(3), "kind": "build", "target": "ABC-1", "outcome": "done", "pushed": True, "cost_usd": 3.0},
        {"ts": _iso(3), "kind": "build", "target": "ABC-2", "outcome": "blocked", "pushed": False, "cost_usd": 1.0},
        {"ts": _iso(4), "kind": "rv", "target": "o/r#9", "outcome": "flagged", "pushed": False, "cost_usd": None},
        {"ts": _iso(5), "kind": "rc", "target": "o/r#1", "outcome": "nothing", "pushed": False, "cost_usd": 0.1},
        {"ts": _iso(30), "kind": "ci", "target": "o/r#8", "outcome": "done", "pushed": True, "cost_usd": 9.0},  # old
    ]
    for run in runs:
        state.record_run(run)
    state.upsert_pr_outcome(
        "o/r", 1, state="MERGED", kinds=["build", "ci", "rc"], merged_at=_iso(1), hours_to_merge=10.0, rework_commits=2, reverted=[F1]
    )
    state.upsert_pr_outcome("o/r", 2, state="MERGED", kinds=["ci"], merged_at=_iso(2), hours_to_merge=30.0, rework_commits=0)
    state.upsert_pr_outcome("o/r", 3, state="CLOSED", kinds=["ci"], closed_at=_iso(2), rework_commits=0)
    state.upsert_pr_outcome("o/r", 4, state="OPEN", kind="build", rework_commits=1)
    state.upsert_pr_outcome("o/r", 8, state="MERGED", kinds=["ci"], merged_at=_iso(30), hours_to_merge=500.0)  # old
    state.add_flag("o/r#9", "rv", "review drafted")
    state.add_flag("o/r#3", "ci", "needs me")
    state.add_flag("ABC-2", "build", "needs me")
    state.add_flag("o/r#1", "rc", "needs me")
    first = state.add_flag("ABC-9", "resolve", "repo unresolved")
    state.ack_flag(first)  # acked flags still count toward reasons


def test_build_report_aggregates_runs_prs_and_flags(state):
    _seed(state)
    report = build_report(state, _iso(7))

    assert report.runs == 7 and report.prs == 4
    assert list(report.per_kind) == ["build", "ci", "rc", "rv"]  # canonical kind order
    ci = report.per_kind["ci"]
    assert ci["runs"] == 3 and ci["done"] == 2 and ci["failed"] == 1 and ci["pushed"] == 2
    assert ci["cost_usd"] == pytest.approx(1.4)
    assert ci["prs"] == 3 and ci["merged"] == 2 and ci["closed"] == 1 and ci["merge_rate"] == pytest.approx(2 / 3)
    build = report.per_kind["build"]
    assert build["runs"] == 2 and build["done"] == 1 and build["failed"] == 1  # blocked counts as failed
    assert build["prs"] == 2 and build["merged"] == 1 and build["closed"] == 0 and build["merge_rate"] == 1.0
    assert report.per_kind["rv"]["flagged"] == 1 and report.per_kind["rv"]["merge_rate"] is None
    assert report.per_kind["rc"]["nothing"] == 1 and report.per_kind["rc"]["prs"] == 1

    assert (report.merged, report.closed_unmerged, report.open) == (2, 1, 1)
    assert report.median_hours_to_merge == 20.0
    assert report.rework_rate == pytest.approx(0.5)
    assert report.reverted_pushes == 1
    assert report.cost_total == pytest.approx(5.5)
    assert report.cost_per_merged_pr == pytest.approx(2.75)
    assert report.top_flag_reasons == [("needs me", 3), ("review drafted", 1), ("repo unresolved", 1)]


def test_build_report_without_since_includes_everything(state):
    _seed(state)
    report = build_report(state, "")
    assert report.runs == 8 and report.prs == 5 and report.merged == 3
    assert report.per_kind["ci"]["cost_usd"] == pytest.approx(10.4)
    assert report.median_hours_to_merge == 30.0


def test_build_report_on_empty_state(state):
    report = build_report(state, _iso(7))
    assert report == Report(since=report.since)
    text = render_report(report)
    assert "total" in text and "Top flag reasons: none" in text and "no merged PRs" in text
    assert digest_text(report).startswith("*Factory digest*")


# ------------------------------------------------------------------------------------ text
def test_render_report_is_an_aligned_table_with_headline_numbers(state):
    _seed(state)
    text = render_report(build_report(state, _iso(7)))
    lines = text.splitlines()
    assert lines[0].startswith("Factory report since ")
    header = lines[2]
    assert header.split() == ["kind", "runs", "done", "failed", "flagged", "pushed", "cost", "PRs", "merged", "merge"]
    table = lines[2:8]
    assert [row.split()[0] for row in table] == ["kind", "build", "ci", "rc", "rv", "total"]
    # numeric columns are right-aligned: a column's last character sits at the same offset in every row
    for offset in (header.index("runs") + len("runs"), header.index("cost") + len("cost"), len(header)):
        assert all(row[offset - 1] != " " and (len(row) == offset or row[offset] == " ") for row in table)
    assert all(len(row) == len(header) for row in table)  # last column flush right
    ci = next(row for row in table if row.startswith("ci"))
    assert ci.split() == ["ci", "3", "2", "1", "0", "2", "$1.40", "3", "2", "67%"]
    total = next(row for row in table if row.startswith("total"))
    assert total.split() == ["total", "7", "3", "2", "1", "3", "$5.50", "4", "2", "67%"]
    assert "PRs touched: 4 (merged 2, closed unmerged 1, open 1)" in text
    assert "Median hours to merge: 20" in text
    assert "Rework rate: 50% (2/4 PRs)" in text
    assert "Reverted factory pushes: 1" in text
    assert "Cost: $5.50 total, $2.75 per merged PR" in text
    assert "Top flag reasons:\n  3  needs me\n  1  review drafted\n  1  repo unresolved" in text


def test_digest_text_is_compact(state):
    _seed(state)
    report = build_report(state, _iso(7))
    text = digest_text(report)
    lines = text.splitlines()
    assert len(lines) == 6 and all(len(line) < 160 for line in lines)
    assert lines[0] == f"*Factory digest* since {report.since[:10]}"
    assert lines[1] == "Runs: 7 (3 done, 2 failed, 3 pushed) | cost $5.50"
    assert lines[2] == "PRs: 2 merged, 1 closed, 1 open | median 20h to merge | $2.75 per merged PR"
    assert lines[3] == "Merge rate: build 100% (1/1), ci 67% (2/3), rc 100% (1/1)"
    assert lines[4] == "Rework: 50% of touched PRs | 1 reverted pushes"
    assert lines[5] == "Top flags: needs me (3), review drafted (1), repo unresolved (1)"


# ------------------------------------------------------------------------------------ time
def test_since_iso_from_spec_and_parse_iso():
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    assert since_iso_from_spec("7d", now) == "2026-09-24T12:00:00+00:00"
    assert since_iso_from_spec(" 36H ", now) == "2026-09-30T00:00:00+00:00"
    assert since_iso_from_spec("2w", now) == "2026-09-17T12:00:00+00:00"
    assert since_iso_from_spec("2026-09-01T00:00:00Z") == "2026-09-01T00:00:00+00:00"
    assert since_iso_from_spec("2026-09-01") == "2026-09-01T00:00:00+00:00"
    with pytest.raises(ValueError):
        since_iso_from_spec("yesterday")
    assert parse_iso("2026-09-01T00:00:00Z") == datetime(2026, 9, 1, tzinfo=UTC)
    assert parse_iso("2026-09-01T02:00:00+02:00") == datetime(2026, 9, 1, tzinfo=UTC)
    assert parse_iso(None) is None and parse_iso("nope") is None
