"""scan.py: detection from fixture-built PRs and fake-tracker issues, version keys, every filter.

PRs are produced by the real ``GitHub`` parsers over ``tests/fixtures/github/`` (no subprocess), then
served by a duck-typed ``FakeGitHub``; the tracker is a ``FakeTracker`` satisfying the ``Tracker``
protocol (the real Linear client lives elsewhere).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from software_factory.actions import Action, Kind
from software_factory.config import Profile, RepoOverride
from software_factory.repos import RepoIndex, RepoInfo
from software_factory.scan import ScanResult, filter_actions, scan, scan_github, scan_tracker, skip_reason
from software_factory.sources.github import GhAuthError, GhError, GitHub, parse_commits, parse_pr
from software_factory.sources.models import Feedback, PrCommit, PullRequest, ReviewThread
from software_factory.trackers.base import (
    Attachment,
    Issue,
    IssueRef,
    StateType,
    Tracker,
    TrackerAuthError,
    TrackerError,
)

FIXTURES = Path(__file__).parent / "fixtures" / "github"
ME = "JaneDoe"
CADLIB = "JaneDoe/cadlib"
GAME = "JaneDoe/game"
WEBAPP = "JaneDoe/webapp"
HEAD_4 = "4444bbbb4444bbbb4444bbbb4444bbbb4444bbbb"
HEAD_12 = "bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222"
BASE_12 = "aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111"
LATEST_FEEDBACK_4 = "2026-09-30T11:00:00Z"
EMPTY_THREADS = json.dumps(
    {"data": {"repository": {"pullRequest": {"reviewThreads": {"pageInfo": {"hasNextPage": False}, "nodes": []}}}}}
)


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def details_from_fixtures(repo: str, number: int, view: str, threads: str | None = None) -> PullRequest:
    """Run the real ``GitHub.pr_details`` parsing over fixture files (no process is spawned)."""

    def run(args: list[str], **_: Any) -> str:
        if args[:2] == ["pr", "view"]:
            return (FIXTURES / view).read_text(encoding="utf-8")
        if args[:2] == ["api", "graphql"]:
            return (FIXTURES / threads).read_text(encoding="utf-8") if threads else EMPTY_THREADS
        raise AssertionError(f"unexpected gh call: {args}")

    return GitHub(login=ME, run=run).pr_details(repo, number)


def details_from_data(repo: str, number: int, view: dict, thread_nodes: list[dict]) -> PullRequest:
    """``GitHub.pr_details`` over inline ``gh pr view`` / review-thread JSON (real parsers, no process)."""
    threads = {"data": {"repository": {"pullRequest": {"reviewThreads": {"pageInfo": {"hasNextPage": False}, "nodes": thread_nodes}}}}}

    def run(args: list[str], **_: Any) -> str:
        if args[:2] == ["pr", "view"]:
            return json.dumps(view)
        if args[:2] == ["api", "graphql"]:
            return json.dumps(threads)
        raise AssertionError(f"unexpected gh call: {args}")

    return GitHub(login=ME, run=run).pr_details(repo, number)


def make_profile(**overrides: Any) -> Profile:
    return Profile(name="t", github={"login": ME, "owners": [ME]}, **overrides)


def index(*slugs: str) -> RepoIndex:
    repos = []
    for slug in slugs:
        name = slug.split("/", 1)[1]
        repos.append(
            RepoInfo(slug=slug, name=name, dirname=name, path=Path("repos") / name, default_branch="main", override=RepoOverride())
        )
    return RepoIndex(repos)


def action(kind: Kind, version: str = "v1", target_repo: str = CADLIB, pr: int = 4, **context: Any) -> Action:
    return Action(kind=kind, version=version, repo=target_repo, pr=pr, title="t", context=context)


def issue(
    key: str,
    labels: list[str],
    *,
    state_type: str = "unstarted",
    children: list[IssueRef] | None = None,
    blocked_by: list[IssueRef] | None = None,
    attachments: list[Attachment] | None = None,
    updated_at: str = "2026-10-01T10:00:00Z",
    team_key: str = "ABC",
) -> Issue:
    return Issue(
        id=f"id-{key}",
        key=key,
        title=f"Issue {key}",
        url=f"https://linear.app/x/issue/{key}",
        labels=labels,
        state_type=state_type,
        team_id="team-1",
        team_key=team_key,
        updated_at=updated_at,
        branch_name=f"{key.lower()}-branch",
        children=children or [],
        blocked_by=blocked_by or [],
        attachments=attachments or [],
    )


# -------------------------------------------------------------------------------------- fakes


class FakeGitHub:
    """Duck-typed stand-in for ``sources.github.GitHub`` serving pre-built models."""

    def __init__(
        self,
        *,
        open_prs: list[PullRequest] = (),
        review_requests: list[PullRequest] = (),
        commits: dict[tuple[str, int], Any] | None = None,
        states: dict[tuple[str, int], Any] | None = None,
        details_errors: dict[tuple[str, int], Exception] | None = None,
        list_error: Exception | None = None,
    ) -> None:
        self.open_prs = list(open_prs)
        self.review_requests = list(review_requests)
        self.commits = dict(commits or {})
        self.states = dict(states or {})
        self.details_errors = dict(details_errors or {})
        self.list_error = list_error
        self.calls: list[tuple] = []

    def login(self) -> str:
        return ME

    def my_open_prs(self, repos: list[str], max_age_days: int) -> list[PullRequest]:
        self.calls.append(("my_open_prs", tuple(repos), max_age_days))
        if self.list_error is not None:
            raise self.list_error
        managed = {r.lower() for r in repos}
        return [
            PullRequest(repo=p.repo, number=p.number, title=p.title, url=p.url, author=p.author, is_draft=p.is_draft)
            for p in self.open_prs
            if p.repo.lower() in managed
        ]

    def review_requested(self, max_age_days: int) -> list[PullRequest]:
        self.calls.append(("review_requested", max_age_days))
        return list(self.review_requests)

    @staticmethod
    def _lookup(table: dict[tuple[str, int], Any], repo: str, number: int, default: Any = KeyError) -> Any:
        """GitHub slugs are case-insensitive, like ``gh`` itself."""
        for (slug, n), value in table.items():
            if (slug.lower(), n) == (repo.lower(), number):
                return value
        if default is KeyError:
            raise KeyError((repo, number))
        return default

    def pr_details(self, repo: str, number: int) -> PullRequest:
        self.calls.append(("pr_details", repo, number))
        error = self._lookup(self.details_errors, repo, number, None)
        if error is not None:
            raise error
        found = next(p for p in self.open_prs if (p.repo.lower(), p.number) == (repo.lower(), number))
        return found.model_copy(update={"repo": repo})  # like ``parse_pr``: keyed by the slug asked for

    def pr_commits(self, repo: str, number: int) -> list[PrCommit]:
        self.calls.append(("pr_commits", repo, number))
        value = self._lookup(self.commits, repo, number, [])
        if isinstance(value, Exception):
            raise value
        return value

    def pr_state(self, repo: str, number: int) -> PullRequest:
        self.calls.append(("pr_state", repo, number))
        value = self._lookup(self.states, repo, number)
        if isinstance(value, Exception):
            raise value
        return value


class FakeTracker:
    name = "fake"

    def __init__(self, issues: list[Issue], *, fail: dict[str, Exception] | None = None) -> None:
        self.issues = list(issues)
        self.fail = dict(fail or {})
        self.calls: list[tuple] = []

    # reads
    def ping(self) -> str:
        return "me"

    def list_issues_with_label(self, label: str) -> list[Issue]:
        self.calls.append(("list", label))
        if label in self.fail:
            raise self.fail[label]
        return [i for i in self.issues if i.has_label(label)]

    def get_issue(self, id_or_key: str) -> Issue:
        self.calls.append(("get", id_or_key))
        for i in self.issues:
            if id_or_key in (i.id, i.key):
                return i
        raise TrackerError(f"no such issue {id_or_key}")

    def sibling_issues(self, issue: Issue, limit: int = 50) -> list[Issue]:
        return []

    def labels_exist(self, names: list[str]) -> dict[str, bool]:
        return dict.fromkeys(names, True)

    # writes: a scan must never reach these
    def create_sub_issue(self, parent: Issue, title: str, description: str, labels: list[str]) -> Issue:
        raise AssertionError("scan must never write to the tracker")

    def add_blocked_by(self, issue_id: str, blocker_id: str) -> None:
        raise AssertionError("scan must never write to the tracker")

    def move_to_state_type(self, issue_id: str, state_type: StateType) -> None:
        raise AssertionError("scan must never write to the tracker")

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None:
        raise AssertionError("scan must never write to the tracker")

    def remove_label(self, issue_id: str, label: str) -> None:
        raise AssertionError("scan must never write to the tracker")

    def ensure_label(self, team_id: str, name: str) -> str:
        raise AssertionError("scan must never write to the tracker")


@pytest.fixture
def cadlib_pr() -> PullRequest:
    return details_from_fixtures(CADLIB, 4, "pr_view_cadlib_4.json", "review_threads_cadlib_4.json")


@pytest.fixture
def game_pr() -> PullRequest:
    return details_from_fixtures(GAME, 12, "pr_view_game_12.json")


@pytest.fixture
def commits() -> dict[tuple[str, int], list[PrCommit]]:
    nodes = load("commits_cadlib_4.json")["data"]["repository"]["pullRequest"]["commits"]["nodes"]
    return {(CADLIB, 4): parse_commits(nodes)}  # latest committedDate 2026-09-29T19:47:08Z


def resolved(pr: PullRequest) -> PullRequest:
    return pr.model_copy(update={"threads": [t.model_copy(update={"is_resolved": True}) for t in pr.threads]})


NEWER_COMMIT = [PrCommit(sha="f" * 40, committed_at="2026-10-01T00:00:00Z")]


# ------------------------------------------------------------------------------- scan_github


def test_scan_github_detects_ci_conflict_rc_with_contract_versions(cadlib_pr, game_pr, commits, state):
    gh = FakeGitHub(open_prs=[cadlib_pr, game_pr], commits=commits)

    actions = scan_github(make_profile(), gh, index(CADLIB, GAME), state)

    by = {(a.target, a.kind): a for a in actions}
    assert set(by) == {(f"{CADLIB}#4", Kind.CI), (f"{CADLIB}#4", Kind.RC), (f"{GAME}#12", Kind.CONFLICT)}

    ci = by[(f"{CADLIB}#4", Kind.CI)]
    assert ci.version == HEAD_4 and ci.url.endswith("/pull/4") and ci.title.startswith("Add mech")
    assert [f["name"] for f in ci.context["failing"]] == ["test (3.13)", "Merge Check", "ci/external-failed"]
    assert ci.context["failing"][0] == {
        "name": "test (3.13)",
        "run_id": 100000001,
        "job_id": 200000002,
        "workflow": "CI",
        "details_url": "https://github.com/JaneDoe/cadlib/actions/runs/100000001/job/200000002",
    }
    assert ci.context["head_ref_name"] == "feat/mech-framework" and ci.context["head_ref_oid"] == HEAD_4
    assert ci.context["base_ref_name"] == "main" and ci.context["base_ref_oid"].startswith("9999aaaa")

    conflict = by[(f"{GAME}#12", Kind.CONFLICT)]
    assert conflict.version == f"{HEAD_12}:{BASE_12}"
    assert conflict.context["head_ref_oid"] == HEAD_12 and conflict.context["base_ref_oid"] == BASE_12

    rc = by[(f"{CADLIB}#4", Kind.RC)]
    assert rc.version == LATEST_FEEDBACK_4
    assert rc.context["feedback_ids"] == [
        "PRR_cursor",
        "PRR_reviewer_a",
        "PRRC_reviewer_a_thread",
        "IC_reviewer_a_1",  # newer than the last commit
        "PRRC_ghost",  # older, but on an unresolved non-outdated thread
    ]
    assert rc.context["unresolved_thread_ids"] == ["PRRT_open_current", "PRRT_ghost"]
    assert rc.context["last_own_activity_at"] == "2026-09-29T19:47:08+00:00"

    assert ("my_open_prs", (CADLIB, GAME), 30) in gh.calls and ("review_requested", 7) in gh.calls


def test_pending_checks_without_failures_never_produce_ci(game_pr, state):
    pr = game_pr.model_copy(update={"mergeable": "MERGEABLE"})
    assert [c.state for c in pr.checks] == ["success", "pending"]
    assert scan_github(make_profile(), FakeGitHub(open_prs=[pr]), index(GAME), state) == []


def test_rc_not_emitted_when_feedback_predates_my_last_commit_and_threads_are_resolved(cadlib_pr, state):
    gh = FakeGitHub(open_prs=[resolved(cadlib_pr)], commits={(CADLIB, 4): NEWER_COMMIT})
    kinds = {a.kind for a in scan_github(make_profile(), gh, index(CADLIB), state)}
    assert kinds == {Kind.CI}


def test_rc_emitted_for_unresolved_thread_even_when_feedback_is_old(cadlib_pr, state):
    gh = FakeGitHub(open_prs=[cadlib_pr], commits={(CADLIB, 4): NEWER_COMMIT})
    [rc] = [a for a in scan_github(make_profile(), gh, index(CADLIB), state) if a.kind == Kind.RC]
    assert rc.version == LATEST_FEEDBACK_4
    assert rc.context["feedback_ids"] == ["PRRC_reviewer_a_thread", "PRRC_ghost"]


def test_rc_suppressed_by_a_factory_push_newer_than_the_feedback(cadlib_pr, commits, state):
    pr = resolved(cadlib_pr)
    gh = FakeGitHub(open_prs=[pr], commits=commits)
    assert Kind.RC in {a.kind for a in scan_github(make_profile(), gh, index(CADLIB), state)}

    state.record_push(CADLIB, 4, "feat/mech-framework", "ci", "a" * 40, "b" * 40, ["b" * 40])  # ts = now
    assert Kind.RC not in {a.kind for a in scan_github(make_profile(), gh, index(CADLIB), state)}


def test_rc_with_only_my_own_comments_is_not_an_action(game_pr, state):
    mine = Feedback(id="c1", author=ME, body="note to self", created_at="2026-10-01T00:00:00Z", kind="thread")
    pr = game_pr.model_copy(
        update={
            "mergeable": "MERGEABLE",
            "feedback": [],  # GitHub.pr_details already dropped my own comments
            "threads": [ReviewThread(id="t1", is_resolved=False, is_outdated=False, comments=[mine])],
        }
    )
    gh = FakeGitHub(open_prs=[pr], commits={(GAME, 12): []})
    assert scan_github(make_profile(), gh, index(GAME), state) == []
    assert ("pr_commits", GAME, 12) not in gh.calls  # no feedback -> no commit lookup


def test_rc_version_does_not_regress_when_a_reviewer_resolves_a_thread(state):
    """Issue comment A (09-02) + thread comment B (09-03, unresolved), my last commit 09-01. Tick 1
    handles the pair at version 09-03 (worker: nothing to change). The reviewer then resolves B --
    the factory cannot -- and tick 2 must *not* re-dispatch A at the older, never-recorded version
    09-02. The version key is computed over a set that only grows, and the kind pre-flight agrees."""
    from software_factory.kinds._pr import current_version

    view = {
        "number": 4, "title": "t", "url": f"https://github.com/{CADLIB}/pull/4", "state": "OPEN",
        "author": {"login": ME}, "headRefName": "feat", "headRefOid": "a" * 40, "baseRefName": "main",
        "baseRefOid": "b" * 40, "headRepository": {"name": "cadlib", "nameWithOwner": ""},
        "headRepositoryOwner": {"login": ME}, "mergeable": "MERGEABLE", "reviews": [],
        "comments": [{"id": "IC_a", "author": {"login": "reviewer-a"}, "body": "A?", "createdAt": "2026-09-02T00:00:00Z"}],
    }  # fmt: skip
    thread_b = {
        "id": "PRRT_b", "isResolved": False, "isOutdated": False, "path": "x.py", "line": 1,
        "comments": {"nodes": [{"id": "PRRC_b", "author": {"login": "reviewer-a"}, "body": "B", "createdAt": "2026-09-03T00:00:00Z"}]},
    }  # fmt: skip
    own_commit = {(CADLIB, 4): [PrCommit(sha="c" * 40, committed_at="2026-09-01T00:00:00Z")]}

    tick1 = details_from_data(CADLIB, 4, view, [thread_b])
    [rc1] = scan_github(make_profile(), FakeGitHub(open_prs=[tick1], commits=own_commit), index(CADLIB), state)
    assert rc1.version == "2026-09-03T00:00:00Z" and rc1.context["feedback_ids"] == ["IC_a", "PRRC_b"]
    assert current_version(Kind.RC, tick1) == rc1.version
    state.mark_handled(rc1.target, "rc", rc1.version, "nothing", "both not actionable")

    tick2 = details_from_data(CADLIB, 4, view, [{**thread_b, "isResolved": True}])
    assert [f.id for f in tick2.open_feedback] == ["IC_a"]  # B is no longer actionable ...
    actions = scan_github(make_profile(), FakeGitHub(open_prs=[tick2], commits=own_commit), index(CADLIB), state)
    rc2 = [a for a in actions if a.kind == Kind.RC]
    assert rc2 == [] or rc2[0].version == rc1.version  # ... but the version key must not move backwards
    assert current_version(Kind.RC, tick2) == rc1.version  # and the pre-flight stale check agrees with the scan
    kept, skipped = filter_actions(make_profile(), actions, state)
    assert kept == [] and [s.reason for s in skipped] == ["handled at this version"] * len(rc2)

    # A reviewer replying on the resolved thread (and nothing else) is not a reason to dispatch.
    reply = {"id": "PRRC_b2", "author": {"login": "reviewer-a"}, "body": "thanks", "createdAt": "2026-09-04T00:00:00Z"}
    tick3 = details_from_data(CADLIB, 4, {**view, "comments": []}, [{**thread_b, "isResolved": True, "comments": {"nodes": [thread_b["comments"]["nodes"][0], reply]}}])
    assert scan_github(make_profile(), FakeGitHub(open_prs=[tick3], commits=own_commit), index(CADLIB), state) == []


def test_scan_keys_prs_by_the_managed_clone_slug_not_github_casing(cadlib_pr, commits, state):
    """GitHub reports ``JaneDoe/cadlib``; the clone's origin is ``janedoe/cadlib`` (GitHub
    slugs are case-insensitive). Everything the scan derives must use the clone's spelling, which
    is what the effects record pushes / factory PRs under and what ``factory run`` uses, and the
    canonical ``head_repo`` must not make the PR look like a fork."""
    local = "janedoe/cadlib"
    gh = FakeGitHub(open_prs=[resolved(cadlib_pr)], commits=commits)

    actions = scan_github(make_profile(), gh, index(local), state)

    assert {a.kind for a in actions} == {Kind.CI, Kind.RC}
    assert all(a.repo == local and a.target == f"{local}#4" for a in actions)
    assert all(a.context["is_fork"] is False and a.context["head_repo"] == CADLIB for a in actions)
    assert ("pr_details", local, 4) in gh.calls and ("pr_commits", local, 4) in gh.calls

    # A factory push is recorded under the clone's slug (``Worktree.repo``); the scan must see it.
    state.record_push(local, 4, "feat/mech-framework", "ci", "a" * 40, "b" * 40, ["b" * 40])
    assert {a.kind for a in scan_github(make_profile(), gh, index(local), state)} == {Kind.CI}

    direct = parse_pr(GAME, load("pr_view_game_21.json")).model_copy(update={"review_requested_from_me": True})
    [rv] = scan_github(make_profile(), FakeGitHub(review_requests=[direct]), index("janedoe/game"), state)
    assert rv.repo == "janedoe/game" and rv.target == "janedoe/game#21" and rv.context["is_fork"] is True


def test_rv_only_for_direct_requests_in_managed_repos(state):
    direct = parse_pr(GAME, load("pr_view_game_21.json")).model_copy(update={"review_requested_from_me": True})
    team_only = parse_pr(GAME, load("pr_view_game_22.json"))
    unmanaged = direct.model_copy(update={"repo": "other/repo", "number": 5})
    drafted = direct.model_copy(update={"number": 6, "is_draft": True})
    gh = FakeGitHub(review_requests=[direct, team_only, unmanaged, drafted])

    actions = scan_github(make_profile(), gh, index(GAME), state)

    [rv] = actions
    assert rv.kind == Kind.RV and rv.target == f"{GAME}#21"
    assert rv.version == "cccc3333cccc3333cccc3333cccc3333cccc3333"
    assert rv.context["author"] == "reviewer-a" and rv.context["is_fork"] is True
    assert rv.context["head_ref_name"] == "feat/direct-request" and rv.context["base_ref_name"] == "main"


def test_scan_github_collects_per_pr_errors_and_propagates_auth(cadlib_pr, game_pr, state):
    errors: list[str] = []
    gh = FakeGitHub(open_prs=[cadlib_pr, game_pr], details_errors={(CADLIB, 4): GhError("HTTP 502")})
    actions = scan_github(make_profile(), gh, index(CADLIB, GAME), state, errors=errors)
    assert [a.kind for a in actions] == [Kind.CONFLICT] and errors == [f"github: {CADLIB}#4: HTTP 502"]

    errors = []
    gh = FakeGitHub(open_prs=[cadlib_pr], commits={(CADLIB, 4): GhError("rate limited")})
    actions = scan_github(make_profile(), gh, index(CADLIB), state, errors=errors)
    assert {a.kind for a in actions} == {Kind.CI, Kind.RC}  # unknown last activity -> all feedback is fresh
    assert errors == [f"github: {CADLIB}#4: commits: rate limited"]

    gh = FakeGitHub(open_prs=[cadlib_pr], details_errors={(CADLIB, 4): GhAuthError("HTTP 401")})
    with pytest.raises(GhAuthError):
        scan_github(make_profile(), gh, index(CADLIB), state)


def test_closed_or_merged_pr_details_produce_nothing(cadlib_pr, state):
    gh = FakeGitHub(open_prs=[cadlib_pr.model_copy(update={"state": "MERGED"})])
    assert scan_github(make_profile(), gh, index(CADLIB), state) == []


# ----------------------------------------------------------------------------- filter_actions


def test_handled_at_same_version_is_skipped_and_new_version_re_emitted(state):
    profile = make_profile()
    state.mark_handled(f"{CADLIB}#4", "ci", "sha1", "done")

    kept, skipped = filter_actions(profile, [action(Kind.CI, "sha1")], state)
    assert kept == [] and [s.reason for s in skipped] == ["handled at this version"]
    assert skipped[0].action.version == "sha1"

    kept, skipped = filter_actions(profile, [action(Kind.CI, "sha2")], state)
    assert [a.version for a in kept] == ["sha2"] and skipped == []


def test_attempts_cap_per_target_kind_version(state):
    profile = make_profile(max_attempts=3)
    target = f"{CADLIB}#4"
    for _ in range(2):
        state.record_attempt(target, "ci", "sha1", "failed")
    assert skip_reason(profile, action(Kind.CI, "sha1"), state) is None

    state.record_attempt(target, "ci", "sha1", "failed")
    assert skip_reason(profile, action(Kind.CI, "sha1"), state) == "attempts exhausted (3/3)"
    assert skip_reason(profile, action(Kind.CI, "sha2"), state) is None  # a new version starts fresh
    assert skip_reason(profile, action(Kind.CONFLICT, "sha1"), state) is None  # another kind too


def test_exhausted_attempts_skip_carries_a_flag_for_the_dispatcher(state):
    """Hitting ``max_attempts`` is otherwise only a line in the tick output. The skip carries a
    flag-once spec (reason, detail with the last failure + the manual retry, dedupe key) that the
    dispatcher raises outside ``--dry-run``; the scan itself writes no state and notifies nothing."""
    profile = make_profile(max_attempts=2, enabled_kinds=[Kind.CI, Kind.BUILD])
    target = f"{CADLIB}#4"
    state.record_attempt(target, "ci", "sha1", "failed", "worker timed out")
    state.record_attempt(target, "ci", "sha1", "kept", "push refused: sensitive path\nworktree: C:/wt/x")

    kept, [skip] = filter_actions(profile, [action(Kind.CI, "sha1")], state)

    assert kept == [] and skip.reason == "attempts exhausted (2/2)"
    assert skip.flag is not None and skip.flag.reason == "attempts exhausted" and skip.flag.level == "warn"
    assert skip.flag.dedupe_key == f"exhausted:{target}:ci:sha1"
    assert "failed 2 attempt(s) at version sha1 (max_attempts = 2)" in skip.flag.detail
    assert "kept - push refused: sensitive path worktree: C:/wt/x" in skip.flag.detail  # the newest attempt
    assert f"`factory run ci {target}`" in skip.flag.detail and "re-arms it" in skip.flag.detail
    assert state.flags() == []  # the scan never writes flags itself (``--dry-run`` must stay pure)

    # build: the version is the issue id, so the human is told that editing the issue is not enough.
    build = Action(kind=Kind.BUILD, version="id-ABC-9", issue_id="id-ABC-9", issue_key="ABC-9", url="https://linear.app/x/issue/ABC-9")
    for _ in range(2):
        state.record_attempt("ABC-9", "build", "id-ABC-9", "failed", "blocked: tests missing")
    _, [bskip] = filter_actions(profile, [build], state)
    assert bskip.flag is not None and bskip.flag.dedupe_key == "exhausted:ABC-9:build:id-ABC-9"
    assert "editing the issue does not re-arm it" in bskip.flag.detail
    assert "`factory run build ABC-9`" in bskip.flag.detail and bskip.flag.detail.endswith("https://linear.app/x/issue/ABC-9")
    assert "failed - blocked: tests missing" in bskip.flag.detail

    # Other skips carry no flag; ``skip_reason`` keeps returning the plain reason.
    state.mark_handled(target, "ci", "sha2", "done")
    _, [handled] = filter_actions(profile, [action(Kind.CI, "sha2")], state)
    assert handled.reason == "handled at this version" and handled.flag is None
    assert skip_reason(profile, action(Kind.CI, "sha1"), state) == "attempts exhausted (2/2)"
    assert skip_reason(profile, action(Kind.CI, "sha3"), state) is None


def test_kept_worktree_blocks_every_kind_on_that_target(state):
    profile = make_profile()
    state.keep_worktree(f"{CADLIB}#4", "conflict", "C:/wt/x y", "conflict markers left for you")
    for kind in (Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV):
        assert skip_reason(profile, action(kind), state) == "kept worktree awaiting me: conflict markers left for you"
    assert skip_reason(profile, action(Kind.CI, target_repo=GAME, pr=12), state) is None


def test_disabled_kind_skipped_in_full_mode_but_kept_in_observe(state):
    actions = [action(Kind.CI, "s"), action(Kind.RC, "t"), action(Kind.RV, "u")]

    kept, skipped = filter_actions(make_profile(mode="full", enabled_kinds=[Kind.RV]), actions, state)
    assert [a.kind for a in kept] == [Kind.RV]
    assert [(s.action.kind, s.reason) for s in skipped] == [(Kind.CI, "kind disabled"), (Kind.RC, "kind disabled")]

    kept, skipped = filter_actions(make_profile(mode="observe", enabled_kinds=[Kind.RV]), actions, state)
    assert [a.kind for a in kept] == [Kind.CI, Kind.RC, Kind.RV] and skipped == []


def test_build_blocked_by_unfinished_relations(state):
    profile = make_profile(enabled_kinds=[Kind.BUILD])
    blocked = Action(kind=Kind.BUILD, version="id-ABC-9", issue_id="id-ABC-9", issue_key="ABC-9", context={"unfinished_blockers": ["ABC-1", "ABC-2"]})
    assert skip_reason(profile, blocked, state) == "blocked by ABC-1, ABC-2"

    free = blocked.model_copy(update={"context": {"unfinished_blockers": []}})
    assert skip_reason(profile, free, state) is None

    # No scan context (``factory run build ABC-9``): the tracker is consulted.
    bare = Action(kind=Kind.BUILD, version="id-ABC-9", issue_id="id-ABC-9", issue_key="ABC-9")
    tracker = FakeTracker(
        [issue("ABC-9", ["factory:build"], blocked_by=[IssueRef(id="id-ABC-1", key="ABC-1", state_type="started")])]
    )
    assert skip_reason(profile, bare, state, tracker) == "blocked by ABC-1"
    assert tracker.calls == [("get", "id-ABC-9")]
    assert skip_reason(profile, bare, state, FakeTracker([])) is None  # lookup failure -> not blocked
    assert skip_reason(profile, bare, state, None) is None


def test_duplicate_actions_within_one_scan_collapse(state):
    kept, skipped = filter_actions(make_profile(), [action(Kind.CI, "s"), action(Kind.CI, "s")], state)
    assert len(kept) == 1 and skipped == []


# ------------------------------------------------------------------------------ scan_tracker


def test_fake_tracker_satisfies_protocol():
    assert isinstance(FakeTracker([]), Tracker)


def test_scan_tracker_refine_plan_build_detection_and_versions(state):
    issues = [
        issue("ABC-1", ["factory:refine"], updated_at="2026-10-01T01:00:00Z"),
        issue("ABC-2", ["factory:plan"], updated_at="2026-10-01T02:00:00Z"),
        issue("ABC-3", ["factory:plan"], children=[IssueRef(id="c", key="ABC-30")]),
        issue("ABC-4", ["factory:build", "repo:cadlib"], state_type="unstarted"),
        issue("ABC-5", ["factory:build"], state_type="started"),
        issue("ABC-6", ["factory:build"], state_type="backlog"),
    ]
    tracker = FakeTracker(issues)

    actions = scan_tracker(make_profile(), tracker, state)

    assert [(a.kind, a.target, a.version) for a in actions] == [
        (Kind.REFINE, "ABC-1", "2026-10-01T01:00:00Z"),
        (Kind.PLAN, "ABC-2", "2026-10-01T02:00:00Z"),
        (Kind.BUILD, "ABC-4", "id-ABC-4"),
    ]
    build = actions[2]
    assert build.issue_id == "id-ABC-4" and build.url.endswith("/ABC-4") and build.title == "Issue ABC-4"
    assert build.context["branch_name"] == "abc-4-branch" and build.context["team_id"] == "team-1"
    assert build.context["unfinished_blockers"] == [] and build.context["labels"] == ["factory:build", "repo:cadlib"]
    assert actions[0].context["issue_id"] == "id-ABC-1" and actions[0].context["labels"] == ["factory:refine"]
    assert [c for c in tracker.calls if c[0] == "list"] == [
        ("list", "factory:refine"),
        ("list", "factory:plan"),
        ("list", "factory:build"),
    ]


def test_scan_tracker_build_blockers_flow_into_a_skip(state):
    blockers = [
        IssueRef(id="id-ABC-1", key="ABC-1", state_type="started"),
        IssueRef(id="id-ABC-0", key="ABC-0", state_type="completed"),
        IssueRef(id="id-ABC-2", key="ABC-2", state_type="canceled"),
    ]
    tracker = FakeTracker([issue("ABC-7", ["factory:build"], blocked_by=blockers)])

    [build] = scan_tracker(make_profile(), tracker, state)
    assert build.context["unfinished_blockers"] == ["ABC-1"]

    kept, skipped = filter_actions(make_profile(enabled_kinds=[Kind.BUILD]), [build], state, tracker)
    assert kept == [] and [s.reason for s in skipped] == ["blocked by ABC-1"]


def test_scan_tracker_build_suppressed_by_open_linked_pr_only(state):
    open_link = Attachment(url=f"https://github.com/{CADLIB}/pull/4", source_type="githubPr")
    merged_link = Attachment(url=f"https://github.com/{WEBAPP}/pull/10", source_type="githubPr")
    other_link = Attachment(url="https://example.invalid/design.fig", source_type="figma")
    tracker = FakeTracker(
        [
            issue("ABC-8", ["factory:build"], attachments=[other_link, open_link]),
            issue("ABC-9", ["factory:build"], attachments=[merged_link]),
        ]
    )
    gh = FakeGitHub(
        states={
            (CADLIB, 4): parse_pr(CADLIB, load("pr_view_cadlib_4.json")),
            (WEBAPP, 10): parse_pr(WEBAPP, load("pr_view_webapp_10_merged.json")),
        }
    )

    actions = scan_tracker(make_profile(), tracker, state, gh=gh)
    assert [a.target for a in actions] == ["ABC-9"]
    assert ("pr_state", CADLIB, 4) in gh.calls and ("pr_state", WEBAPP, 10) in gh.calls

    # Without gh any PR attachment counts as open (never open a second PR for the same issue).
    assert scan_tracker(make_profile(), tracker, state, gh=None) == []

    errors: list[str] = []
    failing = FakeGitHub(states={(CADLIB, 4): GhError("HTTP 500"), (WEBAPP, 10): GhError("HTTP 500")})
    assert scan_tracker(make_profile(), tracker, state, gh=failing, errors=errors) == []
    assert errors == [f"github: ABC-8: {CADLIB}#4: HTTP 500", f"github: ABC-9: {WEBAPP}#10: HTTP 500"]

    with pytest.raises(GhAuthError):
        scan_tracker(make_profile(), tracker, state, gh=FakeGitHub(states={(CADLIB, 4): GhAuthError("401")}))


def test_scan_tracker_errors_are_collected_auth_propagates_disabled_is_empty(state):
    issues = [issue("ABC-1", ["factory:refine"]), issue("ABC-2", ["factory:plan"]), issue("ABC-3", ["factory:build"])]
    errors: list[str] = []
    tracker = FakeTracker(issues, fail={"factory:plan": TrackerError("rate limited")})

    actions = scan_tracker(make_profile(), tracker, state, errors=errors)
    assert [a.kind for a in actions] == [Kind.REFINE, Kind.BUILD]
    assert errors == ["tracker: list issues with label 'factory:plan': rate limited"]

    with pytest.raises(TrackerAuthError):
        scan_tracker(make_profile(), FakeTracker(issues, fail={"factory:refine": TrackerAuthError("bad key")}), state)

    disabled = FakeTracker(issues)
    assert scan_tracker(make_profile(linear={"enabled": False}), disabled, state) == [] and disabled.calls == []


def test_scan_tracker_team_filter_and_dedupe(state):
    issues = [issue("ABC-1", ["factory:refine"], team_key="ABC"), issue("XYZ-1", ["factory:refine"], team_key="XYZ")]
    tracker = FakeTracker(issues + [issues[0]])  # the tracker returned the same issue twice
    assert [a.target for a in scan_tracker(make_profile(), tracker, state)] == ["ABC-1", "XYZ-1"]
    assert [a.target for a in scan_tracker(make_profile(linear={"team_keys": ["xyz"]}), tracker, state)] == ["XYZ-1"]


# ------------------------------------------------------------------------------------- facade


def test_scan_merges_sources_applies_filters_and_sorts(cadlib_pr, game_pr, commits, state):
    profile = make_profile(enabled_kinds=[Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV, Kind.REFINE, Kind.PLAN, Kind.BUILD])
    gh = FakeGitHub(open_prs=[cadlib_pr, game_pr], commits=commits)
    tracker = FakeTracker([issue("ABC-1", ["factory:refine"])])
    state.mark_handled(f"{CADLIB}#4", "rc", LATEST_FEEDBACK_4, "done")

    result = scan(profile, gh=gh, tracker=tracker, repos=index(CADLIB, GAME), state=state)

    assert isinstance(result, ScanResult)
    assert [(a.kind, a.target) for a in result.actions] == [
        (Kind.REFINE, "ABC-1"),
        (Kind.CI, f"{CADLIB}#4"),
        (Kind.CONFLICT, f"{GAME}#12"),
    ]
    assert [(s.action.kind, s.reason) for s in result.skipped] == [(Kind.RC, "handled at this version")]
    assert result.errors == []


def test_scan_observe_mode_keeps_non_rv_actions_for_dry_run(cadlib_pr, commits, state):
    profile = make_profile(mode="observe", enabled_kinds=[Kind.RV])
    result = scan(profile, gh=FakeGitHub(open_prs=[cadlib_pr], commits=commits), tracker=None, repos=index(CADLIB), state=state)
    assert {a.kind for a in result.actions} == {Kind.CI, Kind.RC} and result.skipped == []


def test_scan_without_sources_or_with_source_failures(cadlib_pr, state):
    empty = scan(make_profile(), gh=None, tracker=None, repos=index(), state=state)
    assert empty == ScanResult()

    gh = FakeGitHub(open_prs=[cadlib_pr], list_error=GhError("search failed"))
    tracker = FakeTracker([], fail={"factory:refine": TrackerError("down")})
    result = scan(make_profile(), gh=gh, tracker=tracker, repos=index(CADLIB), state=state)
    assert result.actions == [] and result.errors == [
        "github: search failed",
        "tracker: list issues with label 'factory:refine': down",
    ]

    with pytest.raises(GhAuthError):
        scan(make_profile(), gh=FakeGitHub(list_error=GhAuthError("401")), tracker=None, repos=index(), state=state)
