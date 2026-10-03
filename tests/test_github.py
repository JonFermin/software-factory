"""sources/github.py: the gh runner, parsers over real-shaped fixtures, reads, and the two writes.

Fixtures under ``tests/fixtures/github/`` were captured from real ``gh`` output (sanitised) and
extended with the documented ``StatusContext`` / ``Team`` shapes that did not occur in the sample.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from software_factory.sources import github as gh_mod
from software_factory.sources.github import (
    PR_FIELDS_BASIC,
    GhAuthError,
    GhError,
    GitHub,
    default_gh_runner,
    is_bot_login,
    parse_actions_url,
    parse_checks,
    parse_commits,
    parse_iso,
    parse_pr,
    parse_threads,
    tail_bytes,
)
from software_factory.sources.models import PullRequest

FIXTURES = Path(__file__).parent / "fixtures" / "github"
NOW = datetime(2026, 10, 2, tzinfo=UTC)
CADLIB = "JaneDoe/cadlib"
GAME = "JaneDoe/game"
HEAD_4 = "4444bbbb4444bbbb4444bbbb4444bbbb4444bbbb"
EMPTY_THREADS = json.dumps(
    {"data": {"repository": {"pullRequest": {"reviewThreads": {"pageInfo": {"hasNextPage": False}, "nodes": []}}}}}
)


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def load(name: str) -> Any:
    return json.loads(fixture(name))


# ------------------------------------------------------------------------------------ fake gh


def route(args: list[str]) -> str:
    """Stable key for a ``gh`` arg list (ignores ``--json`` field sets)."""
    head = args[:2]
    if head == ["api", "user"]:
        return "login"
    if head == ["auth", "status"]:
        return "auth"
    if head == ["search", "prs"]:
        return "search " + ("author" if "--author" in args else "review-requested")
    if head == ["pr", "view"]:
        return f"pr view {args[4]}#{args[2]}"
    if head == ["pr", "list"]:
        return f"pr list {args[3]} {args[5]}"
    if head == ["pr", "create"]:
        return f"pr create {args[3]}"
    if head == ["pr", "edit"]:
        return f"pr edit {args[4]}#{args[2]}"
    if head == ["run", "view"]:
        return f"run log {args[4]} {args[2]}"
    if head == ["api", "graphql"]:
        fields: dict[str, str] = {}
        for i, arg in enumerate(args):
            if arg in ("-F", "-f") and i + 1 < len(args):
                key, _, value = args[i + 1].partition("=")
                fields[key] = value
        kind = "threads" if "reviewThreads" in fields.get("query", "") else "commits"
        return f"graphql {kind} {fields.get('owner')}/{fields.get('name')}#{fields.get('number')}"
    return " ".join(args)


class FakeGh:
    """Routes ``gh`` arg lists to fixture text (or an exception / callable); records every call."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = dict(routes)
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, args: list[str], *, cwd: Any = None, input: str | None = None) -> str:
        self.calls.append((list(args), input))
        key = route(args)
        if key not in self.routes:
            raise AssertionError(f"unexpected gh call {key!r}: {args}")
        value = self.routes[key]
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value(args, input)
        return value

    def args_for(self, *head: str) -> list[list[str]]:
        return [args for args, _ in self.calls if args[: len(head)] == list(head)]


def standard_routes() -> dict[str, Any]:
    return {
        "login": "JaneDoe\n",
        "auth": "github.com\n  Logged in to github.com account JaneDoe (keyring)\n",
        "search author": fixture("search_my_open_prs.json"),
        "search review-requested": fixture("search_review_requested.json"),
        f"pr view {CADLIB}#4": fixture("pr_view_cadlib_4.json"),
        f"pr view {GAME}#12": fixture("pr_view_game_12.json"),
        f"pr view {GAME}#21": fixture("pr_view_game_21.json"),
        f"pr view {GAME}#22": fixture("pr_view_game_22.json"),
        "pr view JaneDoe/webapp#10": fixture("pr_view_webapp_10_merged.json"),
        f"graphql threads {CADLIB}#4": fixture("review_threads_cadlib_4.json"),
        f"graphql threads {GAME}#12": EMPTY_THREADS,
        f"graphql commits {CADLIB}#4": fixture("commits_cadlib_4.json"),
        f"run log {CADLIB} 100000001": fixture("run_log_failed.txt"),
        f"pr list {CADLIB} feat/mech-framework": fixture("pr_list_head.json"),
        f"pr list {CADLIB} nothing-here": "[]\n",
    }


@pytest.fixture
def fake() -> FakeGh:
    return FakeGh(standard_routes())


@pytest.fixture
def gh(fake: FakeGh, monkeypatch: pytest.MonkeyPatch) -> GitHub:
    monkeypatch.setattr(gh_mod, "_now", lambda: NOW)
    return GitHub(login="JaneDoe", run=fake)


# ------------------------------------------------------------------------------ default runner


def _completed(rc: int, stdout: str = "", stderr: str = "") -> Callable[..., subprocess.CompletedProcess]:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        fake_run.seen = {"cmd": cmd, "kwargs": kwargs}  # type: ignore[attr-defined]
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)

    return fake_run


def test_default_runner_uses_list_args_no_prompts_utf8_and_timeout(monkeypatch):
    fake_run = _completed(0, stdout="ok\n")
    monkeypatch.setattr(gh_mod.subprocess, "run", fake_run)
    monkeypatch.setenv("GH_PROMPT_DISABLED", "0")

    out = default_gh_runner(["pr", "view", "4", "-R", "o w n/n a m e"], input="body text")

    assert out == "ok\n"
    seen = fake_run.seen  # type: ignore[attr-defined]
    assert seen["cmd"] == ["gh", "pr", "view", "4", "-R", "o w n/n a m e"]  # list args, spaces intact
    kwargs = seen["kwargs"]
    assert "shell" not in kwargs
    assert kwargs["text"] is True and kwargs["encoding"] == "utf-8" and kwargs["errors"] == "replace"
    assert kwargs["timeout"] == gh_mod.GH_TIMEOUT_S
    assert kwargs["input"] == "body text"
    assert kwargs["env"]["GH_PROMPT_DISABLED"] == "1"
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"


@pytest.mark.parametrize(
    "rc, stderr, expected",
    [
        (1, "HTTP 404: Not Found (https://api.github.com/repos/x/y)", GhError),
        (4, "", GhAuthError),
        (1, "To get started with GitHub CLI, please run:  gh auth login", GhAuthError),
        (1, "HTTP 401: Bad credentials (https://api.github.com/user)", GhAuthError),
        (1, "The token in keyring is invalid; token is invalid", GhAuthError),
    ],
)
def test_default_runner_maps_failures(monkeypatch, rc, stderr, expected):
    monkeypatch.setattr(gh_mod.subprocess, "run", _completed(rc, stderr=stderr))
    with pytest.raises(expected) as info:
        default_gh_runner(["api", "user"])
    if expected is GhError:
        assert not isinstance(info.value, GhAuthError)
    assert "gh api user" in str(info.value)


def test_default_runner_missing_binary_and_timeout(monkeypatch):
    def missing(cmd, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(gh_mod.subprocess, "run", missing)
    with pytest.raises(GhError, match="not found"):
        default_gh_runner(["auth", "status"])

    def slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(gh_mod.subprocess, "run", slow)
    with pytest.raises(GhError, match="timed out"):
        default_gh_runner(["run", "view", "1"], timeout=5)


# --------------------------------------------------------------------------------- login/auth


def test_login_is_detected_once_and_cached(fake):
    gh = GitHub(run=fake)
    assert gh.login() == "JaneDoe"
    assert gh.login() == "JaneDoe"
    assert fake.args_for("api", "user") == [["api", "user", "--jq", ".login"]]


def test_explicit_login_never_calls_gh(fake):
    assert GitHub(login="someone", run=fake).login() == "someone"
    assert fake.calls == []


def test_auth_check_wraps_any_gh_failure_as_auth_error():
    with pytest.raises(GhAuthError, match="not authenticated"):
        GitHub(run=FakeGh({"auth": GhError("gh auth status failed (1): not logged in")})).auth_check()
    with pytest.raises(GhAuthError):
        GitHub(run=FakeGh({"auth": "ok", "login": GhAuthError("HTTP 401")})).auth_check()
    fake = FakeGh(standard_routes())
    GitHub(run=fake).auth_check()
    assert [route(a) for a, _ in fake.calls] == ["auth", "login"]


def test_from_profile_takes_login_and_lists_from_config(profile, fake):
    profile.github.ignored_logins = ["reviewer-a"]
    profile.github.bot_reviewers = []
    gh = GitHub.from_profile(profile, run=fake)
    assert gh.login() == "JaneDoe" and fake.calls == []
    pr = gh.pr_details(CADLIB, 4)
    assert all(f.author != "reviewer-a" for f in pr.feedback)
    assert not any(f.author_is_bot for f in pr.feedback if f.author == "copilot-pull-request-reviewer")


# ------------------------------------------------------------------------------------ parsers


def test_parse_pr_derives_head_repo_when_name_with_owner_is_empty():
    fork = parse_pr(GAME, load("pr_view_game_21.json"))
    assert fork.head_repo == "reviewer-a/game" and fork.is_fork
    own = parse_pr(CADLIB, load("pr_view_cadlib_4.json"))
    assert own.head_repo == "JaneDoe/cadlib" and not own.is_fork
    assert own.author == "JaneDoe" and own.head_ref_oid == HEAD_4 and own.base_ref_name == "main"
    assert own.created_at == "2026-09-29T19:32:35Z" and own.merged_at is None and own.closed_at is None


def test_is_fork_compares_slugs_case_insensitively():
    """``head_repo`` is rebuilt from GitHub's canonical ``headRepositoryOwner.login`` while ``repo``
    comes from the local clone's origin URL; GitHub slugs are case-insensitive, so a lowercase
    clone must not turn every PR into a 'fork' (which skips ci/conflict/rc and splits state keys)."""
    own = parse_pr("janedoe/CADLIB", load("pr_view_cadlib_4.json"))
    assert own.head_repo == "JaneDoe/cadlib" and own.repo == "janedoe/CADLIB" and not own.is_fork
    assert parse_pr("janedoe/game", load("pr_view_game_21.json")).is_fork  # a real fork still is one
    assert not PullRequest(repo="o/n", number=1, head_repo=None).is_fork


def test_parse_pr_merged_and_tolerant_of_odd_values():
    merged = parse_pr("JaneDoe/webapp", load("pr_view_webapp_10_merged.json"))
    assert merged.state == "MERGED" and merged.merged_at == "2026-09-15T12:30:00Z" and merged.closed_at
    odd = parse_pr("o/n", {"number": 3, "state": "WEIRD", "mergeable": "", "author": None, "headRepository": None})
    assert odd.state == "OPEN" and odd.mergeable == "UNKNOWN" and odd.author == "" and odd.head_repo is None


@pytest.mark.parametrize(
    "status, conclusion, expected",
    [
        ("COMPLETED", "SUCCESS", "success"),
        ("COMPLETED", "FAILURE", "failure"),
        ("COMPLETED", "TIMED_OUT", "failure"),
        ("COMPLETED", "STARTUP_FAILURE", "failure"),
        ("COMPLETED", "ACTION_REQUIRED", "failure"),
        ("COMPLETED", "ERROR", "failure"),
        ("COMPLETED", "CANCELLED", "cancelled"),
        ("COMPLETED", "SKIPPED", "skipped"),
        ("COMPLETED", "NEUTRAL", "neutral"),
        ("COMPLETED", "STALE", "unknown"),
        ("IN_PROGRESS", "", "pending"),
        ("QUEUED", "", "pending"),
        ("PENDING", "", "pending"),
        ("WAITING", "", "pending"),
        ("REQUESTED", "", "pending"),
    ],
)
def test_parse_checks_check_run_mapping(status, conclusion, expected):
    [check] = parse_checks([{"__typename": "CheckRun", "name": "x", "status": status, "conclusion": conclusion}])
    assert check.state == expected


@pytest.mark.parametrize(
    "state, expected",
    [("SUCCESS", "success"), ("FAILURE", "failure"), ("ERROR", "failure"), ("PENDING", "pending"), ("EXPECTED", "pending")],
)
def test_parse_checks_status_context_mapping(state, expected):
    [check] = parse_checks([{"__typename": "StatusContext", "context": "ci/ext", "state": state, "targetUrl": "u"}])
    assert check.state == expected and check.name == "ci/ext" and check.details_url == "u" and check.workflow == ""


def test_parse_checks_from_fixture_ids_and_pending_sentinel():
    checks = parse_checks(load("pr_view_cadlib_4.json")["statusCheckRollup"])
    assert len(checks) == 8
    assert [c.name for c in checks if c.failing] == ["test (3.13)", "Merge Check", "ci/external-failed"]
    by_name = {c.name: c for c in checks}
    assert (by_name["test (3.13)"].run_id, by_name["test (3.13)"].job_id) == (100000001, 200000002)
    assert by_name["test (3.13)"].workflow == "CI" and by_name["test (3.13)"].completed_at == "2026-10-01T21:18:48Z"
    assert by_name["build-docs"].state == "pending" and by_name["build-docs"].completed_at is None
    assert by_name["CodeQL"].run_id is None and by_name["CodeQL"].state == "neutral"
    assert by_name["Merge Check"].run_id is None  # detailsUrl is the PR page, not an Actions run
    assert by_name["ci/external-pending"].state == "pending"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/o/n/actions/runs/100000001/job/200000002", (100000001, 200000002)),
        ("https://github.com/o/n/actions/runs/100000001/jobs/5", (100000001, 5)),
        ("https://github.com/o/n/actions/runs/100000001", (100000001, None)),
        ("https://github.com/o/n/pull/4", (None, None)),
        ("https://github.com/o/n/runs/200000005", (None, None)),
        ("", (None, None)),
    ],
)
def test_parse_actions_url(url, expected):
    assert parse_actions_url(url) == expected


def test_parse_threads_from_fixture():
    nodes = load("review_threads_cadlib_4.json")["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    threads = parse_threads(nodes, bots=["copilot-pull-request-reviewer[bot]"])
    assert [t.id for t in threads] == ["PRRT_open_current", "PRRT_resolved_outdated", "PRRT_open_outdated", "PRRT_ghost"]
    first = threads[0]
    assert (first.is_resolved, first.is_outdated, first.path, first.line) == (False, False, "mech/solver.py", 42)
    assert [c.author for c in first.comments] == ["reviewer-a", "JaneDoe"]
    assert first.comments[0].kind == "thread" and first.comments[0].thread_id == "PRRT_open_current"
    assert first.comments[0].line == 42 and first.comments[0].path == "mech/solver.py"
    assert threads[1].comments[0].author_is_bot  # GraphQL login lacks the [bot] suffix
    assert threads[1].line is None
    assert threads[3].comments[0].author == ""  # deleted account -> null author


def test_parse_commits_from_fixture():
    nodes = load("commits_cadlib_4.json")["data"]["repository"]["pullRequest"]["commits"]["nodes"]
    commits = parse_commits(nodes + [{"commit": None}, "junk"])
    assert [c.sha[:7] for c in commits] == ["5555ccc", "6666ddd"]
    assert commits[0].author_login == "JaneDoe" and commits[0].author_email == "me@example.com"
    assert commits[1].author_login is None and (commits[1].additions, commits[1].deletions) == (410, 12)
    assert commits[1].committed_at == "2026-09-29T19:47:08Z" and commits[0].message.startswith("Add uv project")


@pytest.mark.parametrize(
    "login, bots, expected",
    [
        ("dependabot[bot]", [], True),
        ("Cursor[BOT]", [], True),
        ("copilot-pull-request-reviewer", ["copilot-pull-request-reviewer[bot]"], True),
        ("copilot-pull-request-reviewer", [], False),
        ("reviewer-a", ["cursor[bot]"], False),
        ("", ["cursor[bot]"], False),
    ],
)
def test_is_bot_login(login, bots, expected):
    assert is_bot_login(login, bots) is expected


def test_parse_iso_forms():
    assert parse_iso("2026-10-01T21:16:54Z") == datetime(2026, 10, 1, 21, 16, 54, tzinfo=UTC)
    assert parse_iso("2026-10-01T23:16:54+02:00") == datetime(2026, 10, 1, 21, 16, 54, tzinfo=UTC)
    assert parse_iso("2026-10-01T21:16:54") == datetime(2026, 10, 1, 21, 16, 54, tzinfo=UTC)
    assert parse_iso("") is None and parse_iso(None) is None and parse_iso("yesterday") is None


# -------------------------------------------------------------------------------------- reads


def test_my_open_prs_filters_to_managed_repos_and_window(gh, fake):
    prs = gh.my_open_prs([CADLIB, "janedoe/GAME", "JaneDoe/webapp"], max_age_days=30)
    assert [(p.repo, p.number) for p in prs] == [(CADLIB, 4), (GAME, 12)]
    assert prs[1].is_draft is True and prs[0].author == "JaneDoe" and prs[0].updated_at == "2026-10-01T21:16:54Z"
    [args] = fake.args_for("search", "prs")
    assert args[:4] == ["search", "prs", "--author", "@me"]
    assert args[args.index("--state") + 1] == "open" and args[args.index("--limit") + 1] == "200"


def test_my_open_prs_window_is_inclusive(gh):
    assert [p.number for p in gh.my_open_prs(["JaneDoe/webapp"], 93)] == [7]  # updated exactly 93 days ago
    assert gh.my_open_prs(["JaneDoe/webapp"], 92) == []


def test_review_requested_keeps_direct_non_draft_recent_requests_only(gh, fake):
    prs = gh.review_requested(max_age_days=7)
    assert [p.number for p in prs] == [21]
    assert prs[0].review_requested_from_me and prs[0].head_ref_oid == "cccc3333cccc3333cccc3333cccc3333cccc3333"
    assert prs[0].author == "reviewer-a" and prs[0].repo == GAME
    # the draft (#23) and the stale request (#24) are dropped before any pr view; #22 is team-only
    assert [args[2] for args in fake.args_for("pr", "view")] == ["21", "22"]
    [args] = fake.args_for("search", "prs")
    assert args[2:4] == ["--review-requested", "@me"]


def test_pr_details_feedback_threads_checks(gh):
    pr = gh.pr_details(CADLIB, 4)
    assert [f.id for f in pr.feedback] == [
        "PRRC_copilot",  # resolved-thread comment: kept so the rc version key never moves backwards
        "PRRC_reviewer_b_outdated",  # unresolved (outdated) thread comment
        "PRRC_ghost",  # unresolved thread, deleted author
        "PRR_cursor",  # bot review with a body
        "PRR_reviewer_a",  # review with a body
        "PRRC_reviewer_a_thread",  # unresolved thread comment
        "IC_reviewer_a_1",  # issue comment
    ]
    assert [f.id for f in pr.open_feedback] == [f.id for f in pr.feedback if f.id != "PRRC_copilot"]
    assert pr.resolved_thread_ids == {"PRRT_resolved_outdated"}
    by_id = {f.id: f for f in pr.feedback}
    assert by_id["PRR_cursor"].author_is_bot and by_id["PRR_cursor"].kind == "review"
    assert by_id["PRRC_copilot"].author_is_bot and by_id["PRRC_copilot"].thread_id == "PRRT_resolved_outdated"
    assert not by_id["PRR_reviewer_a"].author_is_bot and by_id["IC_reviewer_a_1"].kind == "comment"
    assert by_id["PRRC_reviewer_a_thread"].thread_id == "PRRT_open_current"
    # excluded: my own review and thread replies, the empty approval, dependabot (ignored)
    assert "PRR_me" not in by_id and "PRR_reviewer_b_empty" not in by_id
    assert "IC_dependabot" not in by_id and "PRRC_me_reply" not in by_id and "PRRC_me_addressed" not in by_id
    assert [t.id for t in pr.threads] == ["PRRT_open_current", "PRRT_resolved_outdated", "PRRT_open_outdated", "PRRT_ghost"]
    assert [t.id for t in pr.unresolved_threads] == ["PRRT_open_current", "PRRT_ghost"]
    assert len(pr.checks) == 8 and [c.name for c in pr.failing_checks] == ["test (3.13)", "Merge Check", "ci/external-failed"]
    assert pr.body.startswith("## Summary") and pr.mergeable == "MERGEABLE" and not pr.review_requested_from_me


def test_pr_details_with_no_feedback(gh):
    pr = gh.pr_details(GAME, 12)
    assert pr.feedback == [] and pr.threads == [] and pr.mergeable == "CONFLICTING" and pr.is_draft
    assert pr.failing_checks == [] and [c.state for c in pr.checks] == ["success", "pending"]


def test_failed_log_full_and_tail(gh, fake):
    full = gh.failed_log(CADLIB, 100000001)
    assert full == fixture("run_log_failed.txt")
    assert fake.args_for("run", "view") == [["run", "view", "100000001", "-R", CADLIB, "--log-failed"]]

    tail = gh.failed_log(CADLIB, 100000001, max_bytes=300)
    marker, _, body = tail.partition("\n")
    assert marker.startswith("[... first ") and "omitted" in marker
    assert len(body.encode("utf-8")) <= 300
    assert body.startswith("test (3.13)\t")  # cut on a line boundary
    assert body.endswith("##[error]Process completed with exit code 1.\n")
    assert "##[group]" not in body


def test_tail_bytes_edge_cases():
    assert tail_bytes("abc", 0) == "abc" and tail_bytes("abc", 3) == "abc" and tail_bytes("", 10) == ""
    assert tail_bytes("l1\nl2\nl3\n", 5).endswith("\nl3\n")
    assert tail_bytes("é" * 10, 5).endswith("é" * 2)  # multibyte cut never raises


def test_pr_commits_via_graphql(gh, fake):
    commits = gh.pr_commits(CADLIB, 4)
    assert [c.sha[:7] for c in commits] == ["5555ccc", "6666ddd"]
    [args] = fake.args_for("api", "graphql")
    assert args[2] == "-f" and args[3].startswith("query=") and "commits(first: 100" in args[3]
    assert args[4:] == ["-F", "owner=JaneDoe", "-F", "name=cadlib", "-F", "number=4"]


def test_pr_commits_paginates():
    def page(cursor: str | None, oid: str, next_cursor: str | None) -> str:
        return json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "commits": {
                                "pageInfo": {"hasNextPage": next_cursor is not None, "endCursor": next_cursor},
                                "nodes": [{"commit": {"oid": oid, "committedDate": "2026-01-01T00:00:00Z"}}],
                            }
                        }
                    }
                }
            }
        )

    def serve(args: list[str], _input: str | None) -> str:
        return page("C1", "b" * 40, None) if "after=C1" in args else page(None, "a" * 40, "C1")

    fake = FakeGh({f"graphql commits {CADLIB}#4": serve})
    commits = GitHub(login="JaneDoe", run=fake).pr_commits(CADLIB, 4)
    assert [c.sha for c in commits] == ["a" * 40, "b" * 40]
    assert len(fake.calls) == 2 and fake.calls[1][0][-2:] == ["-f", "after=C1"]


def test_graphql_errors_and_missing_pr_become_gh_error():
    fake = FakeGh({f"graphql threads {CADLIB}#4": json.dumps({"errors": [{"message": "nope"}]})})
    with pytest.raises(GhError, match="nope"):
        GitHub(login="me", run=fake)._review_threads(CADLIB, 4)
    fake = FakeGh({f"graphql commits {CADLIB}#4": json.dumps({"data": {"repository": {"pullRequest": None}}})})
    with pytest.raises(GhError, match="not found"):
        GitHub(login="me", run=fake).pr_commits(CADLIB, 4)


def test_pr_state_and_body(gh, fake):
    merged = gh.pr_state("JaneDoe/webapp", 10)
    assert merged.state == "MERGED" and merged.merged_at == "2026-09-15T12:30:00Z"
    assert fake.calls[-1][0] == ["pr", "view", "10", "-R", "JaneDoe/webapp", "--json", PR_FIELDS_BASIC]
    assert gh.pr_body(CADLIB, 4).startswith("## Summary")
    assert fake.calls[-1][0][-2:] == ["--json", "body"]


def test_find_open_pr_for_branch(gh, fake):
    found = gh.find_open_pr_for_branch(CADLIB, "feat/mech-framework")
    assert found is not None and found.number == 4 and found.is_draft and found.head_ref_oid == HEAD_4
    assert gh.find_open_pr_for_branch(CADLIB, "nothing-here") is None
    args = fake.args_for("pr", "list")[0]
    assert args[:6] == ["pr", "list", "-R", CADLIB, "--head", "feat/mech-framework"]
    assert args[args.index("--state") + 1] == "open" and args[args.index("--limit") + 1] == "1"


def test_invalid_json_is_a_gh_error():
    fake = FakeGh({"pr view o/n#1": "not json at all"})
    with pytest.raises(GhError, match="invalid JSON"):
        GitHub(login="me", run=fake).pr_state("o/n", 1)


def test_bad_slug_is_a_gh_error(gh):
    with pytest.raises(GhError, match="owner/name"):
        gh.pr_commits("no-slash", 1)


# ------------------------------------------------------------------------------------- writes


def test_create_draft_pr_pipes_body_via_stdin_and_returns_the_pr(fake):
    fake.routes[f"pr create {CADLIB}"] = "https://github.com/JaneDoe/cadlib/pull/4\n"
    gh = GitHub(login="JaneDoe", run=fake)
    pr = gh.create_draft_pr(CADLIB, "feat/mech-framework", "main", "Add mech", "Body\n\nFixes ABC-1\n")

    [(args, body)] = [(a, i) for a, i in fake.calls if a[:2] == ["pr", "create"]]
    assert args[:5] == ["pr", "create", "-R", CADLIB, "--draft"]
    assert args[args.index("--head") + 1] == "feat/mech-framework" and args[args.index("--base") + 1] == "main"
    assert args[args.index("--title") + 1] == "Add mech" and args[-2:] == ["--body-file", "-"]
    assert body == "Body\n\nFixes ABC-1\n"
    assert pr.number == 4 and pr.state == "OPEN" and pr.repo == CADLIB
    assert fake.calls[-1][0][:3] == ["pr", "view", "4"]


def test_create_draft_pr_falls_back_to_branch_lookup_when_stdout_has_no_url(fake):
    fake.routes[f"pr create {CADLIB}"] = "Creating pull request for feat/mech-framework into main\n"
    pr = GitHub(login="JaneDoe", run=fake).create_draft_pr(CADLIB, "feat/mech-framework", "main", "t", "b")
    assert pr.number == 4 and fake.calls[-1][0][:2] == ["pr", "list"]

    fake.routes[f"pr list {CADLIB} feat/mech-framework"] = "[]"
    with pytest.raises(GhError, match="no PR number"):
        GitHub(login="JaneDoe", run=fake).create_draft_pr(CADLIB, "feat/mech-framework", "main", "t", "b")


def test_edit_pr_body_pipes_body_via_stdin(fake):
    fake.routes[f"pr edit {CADLIB}#4"] = ""
    GitHub(login="JaneDoe", run=fake).edit_pr_body(CADLIB, 4, "new body <!-- factory:evidence -->")
    assert fake.calls == [(["pr", "edit", "4", "-R", CADLIB, "--body-file", "-"], "new body <!-- factory:evidence -->")]
