"""GitHub access, all through the ``gh`` CLI.

Every call goes through an injectable runner (``GhRunner``: args after ``gh`` -> stdout) so tests
feed fixture JSON instead of spawning processes. Reads use ``gh search`` / ``gh pr view --json`` and
one read-only ``gh api graphql`` query for review threads and commits (engine-side GraphQL reads are
fine; only *workers* are barred from graphql). The two writes (``create_draft_pr``, ``edit_pr_body``)
are only ever called from ``effects.py``.

Field names follow the ``gh --json`` output they come from; see ``tests/fixtures/github/`` for the
real shapes these parsers were written against.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from ..config import GitHubConfig, Profile
from .models import Check, CheckState, Feedback, PrCommit, PullRequest, ReviewThread

GH_TIMEOUT_S = 120.0
SEARCH_LIMIT = 200
THREADS_PAGE = 100
COMMITS_PAGE = 100

# `gh pr view --json` field sets. BASIC is enough for state / dedupe; DETAILS adds what scan needs.
PR_FIELDS_BASIC = (
    "number,title,url,author,state,isDraft,headRefName,headRefOid,baseRefName,baseRefOid,"
    "headRepository,headRepositoryOwner,mergeable,createdAt,updatedAt,mergedAt,closedAt"
)
PR_FIELDS_DETAILS = PR_FIELDS_BASIC + ",body,statusCheckRollup,reviews,comments,reviewRequests"
SEARCH_FIELDS = "repository,number,title,url,author,updatedAt,isDraft"

# gh exit code 4 = authentication required; the stderr markers cover `gh api` 401s and expired tokens.
_AUTH_EXIT_CODE = 4
_AUTH_MARKERS = (
    "gh auth login",
    "not logged in",
    "authentication required",
    "requires authentication",
    "http 401",
    "bad credentials",
    "token has expired",
    "token is invalid",
)

_ACTIONS_URL = re.compile(r"/actions/runs/(\d+)(?:/jobs?/(\d+))?")
_PR_URL = re.compile(r"/pull/(\d+)(?:[/#?].*)?$")
_GH_NULL_TIME = "0001-01-01T00:00:00Z"  # gh's zero time for checks that have not completed

THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id isResolved isOutdated path line
          comments(first: 50) { nodes { id author { login } body createdAt url } }
        }
      }
    }
  }
}
""".strip()

COMMITS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      commits(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          commit {
            oid committedDate message additions deletions
            author { email user { login } }
          }
        }
      }
    }
  }
}
""".strip()


class GhError(RuntimeError):
    pass


class GhAuthError(GhError):
    pass


GhRunner = Callable[..., str]
"""``run(args, *, cwd=None, input=None) -> stdout``; raises ``GhError`` / ``GhAuthError``. Reads call
``run(args)`` only, so a plain ``lambda args: ...`` fake works for them; the two writes pass
``input=`` (the PR body piped to ``--body-file -``)."""


def default_gh_runner(
    args: list[str], *, cwd: str | os.PathLike[str] | None = None, input: str | None = None, timeout: float | None = None
) -> str:
    """Run ``gh <args>`` (list args, never a shell) and return stdout; never prompts."""
    env = {
        **os.environ,
        "GH_PROMPT_DISABLED": "1",
        "GH_NO_UPDATE_NOTIFIER": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "NO_COLOR": "1",
    }
    try:
        result = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout or GH_TIMEOUT_S,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            input=input,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GhError("gh executable not found on PATH (install GitHub CLI and run `gh auth login`)") from exc
    except subprocess.TimeoutExpired as exc:
        raise GhError(f"gh {' '.join(args[:3])} timed out after {timeout or GH_TIMEOUT_S:g}s") from exc
    except OSError as exc:
        raise GhError(f"gh {' '.join(args[:3])} could not start: {exc}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        message = f"gh {' '.join(args[:3])} failed ({result.returncode}): {stderr or result.stdout.strip()}"
        if result.returncode == _AUTH_EXIT_CODE or _looks_like_auth_failure(stderr):
            raise GhAuthError(message)
        raise GhError(message)
    return result.stdout


def _looks_like_auth_failure(stderr: str) -> bool:
    text = stderr.lower()
    return any(marker in text for marker in _AUTH_MARKERS)


class GitHub:
    """``gh``-backed GitHub reads (+ the two writes effects.py performs).

    ``login`` is my GitHub login (detected with ``gh api user`` when omitted). ``ignored_logins`` /
    ``bot_reviewers`` come from the profile's ``[github]`` table (``from_profile``); feedback from
    me or an ignored login is dropped, feedback from a bot reviewer is marked ``author_is_bot``.
    """

    def __init__(
        self,
        login: str | None = None,
        run: GhRunner | None = None,
        *,
        ignored_logins: Iterable[str] | None = None,
        bot_reviewers: Iterable[str] | None = None,
    ) -> None:
        defaults = GitHubConfig()
        self._login: str | None = login or None
        self._run: GhRunner = run or default_gh_runner
        self._ignored = {_bare_login(x) for x in (ignored_logins if ignored_logins is not None else defaults.ignored_logins)}
        self._bots = {_bare_login(x) for x in (bot_reviewers if bot_reviewers is not None else defaults.bot_reviewers)}

    @classmethod
    def from_profile(cls, profile: Profile, run: GhRunner | None = None) -> GitHub:
        cfg = profile.github
        return cls(cfg.login or None, run, ignored_logins=cfg.ignored_logins, bot_reviewers=cfg.bot_reviewers)

    # ------------------------------------------------------------------------------ auth
    def login(self) -> str:
        """My login, cached (``gh api user --jq .login`` on first use)."""
        if not self._login:
            self._login = self._run(["api", "user", "--jq", ".login"]).strip()
            if not self._login:
                raise GhError("`gh api user` returned an empty login")
        return self._login

    def auth_check(self) -> None:
        """Raise ``GhAuthError`` unless ``gh`` is authenticated and my login resolves."""
        try:
            self._run(["auth", "status"])
            self.login()
        except GhAuthError:
            raise
        except GhError as exc:
            raise GhAuthError(f"gh is not authenticated: {exc}") from exc

    # ----------------------------------------------------------------------------- reads
    def my_open_prs(self, repos: list[str], max_age_days: int) -> list[PullRequest]:
        """My open PRs (basic fields) in ``repos`` updated within ``max_age_days``."""
        managed = {r.lower() for r in repos}
        cutoff = _now() - timedelta(days=max_age_days)
        found: list[PullRequest] = []
        for item in self._search(["--author", "@me"]):
            pr = _parse_search_item(item)
            if pr is None or pr.repo.lower() not in managed:
                continue
            if not _within(pr.updated_at, cutoff):
                continue
            found.append(pr)
        return found

    def review_requested(self, max_age_days: int) -> list[PullRequest]:
        """Open, non-draft PRs updated within the window where *I* (not a team) am a requested
        reviewer. Each returned PR has ``review_requested_from_me=True`` and basic fields populated."""
        me = self.login().lower()
        cutoff = _now() - timedelta(days=max_age_days)
        found: list[PullRequest] = []
        for item in self._search(["--review-requested", "@me"]):
            candidate = _parse_search_item(item)
            if candidate is None or candidate.is_draft or candidate.author.lower() == me:
                continue
            if not _within(candidate.updated_at, cutoff):
                continue
            data = self._pr_view(candidate.repo, candidate.number, PR_FIELDS_BASIC + ",reviewRequests")
            pr = parse_pr(candidate.repo, data)
            if pr.is_draft or pr.state != "OPEN":
                continue
            if not _requested_directly(data.get("reviewRequests") or [], me):
                continue
            found.append(pr.model_copy(update={"review_requested_from_me": True}))
        return found

    def pr_details(self, repo: str, number: int) -> PullRequest:
        """Full PR: checks, feedback (reviews + comments + every thread comment, resolved threads
        included -- see ``PullRequest.open_feedback`` for the actionable subset) and threads."""
        me = self.login()
        data = self._pr_view(repo, number, PR_FIELDS_DETAILS)
        pr = parse_pr(repo, data)
        threads = self._review_threads(repo, number)
        feedback = self._collect_feedback(data, threads, me)
        requested = _requested_directly(data.get("reviewRequests") or [], me.lower())
        return pr.model_copy(
            update={
                "checks": parse_checks(data.get("statusCheckRollup") or []),
                "feedback": feedback,
                "threads": threads,
                "review_requested_from_me": requested,
            }
        )

    def failed_log(self, repo: str, run_id: int, max_bytes: int = 60_000) -> str:
        """``gh run view --log-failed``, keeping the *last* ``max_bytes`` (failures are at the end)."""
        text = self._run(["run", "view", str(run_id), "-R", repo, "--log-failed"])
        return tail_bytes(text, max_bytes)

    def pr_body(self, repo: str, number: int) -> str:
        data = self._pr_view(repo, number, "body")
        return str(data.get("body") or "")

    def pr_commits(self, repo: str, number: int) -> list[PrCommit]:
        """Every commit on the PR (oldest first) with line counts, via GraphQL."""
        owner, name = _split_slug(repo)
        commits: list[PrCommit] = []
        cursor: str | None = None
        while True:
            payload = self._graphql(COMMITS_QUERY, owner, name, number, cursor)
            connection = _pr_node(payload, repo, number).get("commits") or {}
            commits.extend(parse_commits(connection.get("nodes") or []))
            page = connection.get("pageInfo") or {}
            if not page.get("hasNextPage") or not page.get("endCursor"):
                return commits
            cursor = page["endCursor"]

    def pr_state(self, repo: str, number: int) -> PullRequest:
        """State / timestamps / refs only (cheap; used by scan, metrics and effects)."""
        return parse_pr(repo, self._pr_view(repo, number, PR_FIELDS_BASIC))

    def find_open_pr_for_branch(self, repo: str, branch: str) -> PullRequest | None:
        out = self._run(
            ["pr", "list", "-R", repo, "--head", branch, "--state", "open", "--json", PR_FIELDS_BASIC, "--limit", "1"]
        )
        items = _json(out, "pr list")
        if not isinstance(items, list) or not items:
            return None
        return parse_pr(repo, items[0])

    # ------------------------------------------------------------ writes (effects.py only)
    def create_draft_pr(self, repo: str, head: str, base: str, title: str, body: str) -> PullRequest:
        """``gh pr create --draft`` with the body piped via stdin; returns the new PR's state."""
        out = self._run(
            [
                "pr", "create", "-R", repo, "--draft",
                "--head", head, "--base", base,
                "--title", title, "--body-file", "-",
            ],
            input=body,
        )  # fmt: skip
        number = _number_from_url(out)
        if number is None:
            existing = self.find_open_pr_for_branch(repo, head)
            if existing is None:
                raise GhError(f"gh pr create succeeded but no PR number could be determined from: {out.strip()!r}")
            return existing
        return self.pr_state(repo, number)

    def edit_pr_body(self, repo: str, number: int, body: str) -> None:
        self._run(["pr", "edit", str(number), "-R", repo, "--body-file", "-"], input=body)

    # -------------------------------------------------------------------------- plumbing
    def _search(self, qualifier: list[str]) -> list[dict]:
        out = self._run(
            ["search", "prs", *qualifier, "--state", "open", "--json", SEARCH_FIELDS, "--limit", str(SEARCH_LIMIT)]
        )
        items = _json(out, "search prs")
        return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []

    def _pr_view(self, repo: str, number: int, fields: str) -> dict:
        out = self._run(["pr", "view", str(number), "-R", repo, "--json", fields])
        data = _json(out, f"pr view {repo}#{number}")
        if not isinstance(data, dict):
            raise GhError(f"gh pr view {repo}#{number} returned {type(data).__name__}, expected an object")
        return data

    def _graphql(self, query: str, owner: str, name: str, number: int, cursor: str | None) -> dict:
        args = ["api", "graphql", "-f", f"query={query}", "-F", f"owner={owner}", "-F", f"name={name}", "-F", f"number={number}"]
        if cursor:
            args += ["-f", f"after={cursor}"]
        payload = _json(self._run(args), "api graphql")
        if not isinstance(payload, dict):
            raise GhError("gh api graphql returned a non-object payload")
        if payload.get("errors"):
            raise GhError(f"gh api graphql errors: {payload['errors']}")
        return payload

    def _review_threads(self, repo: str, number: int) -> list[ReviewThread]:
        owner, name = _split_slug(repo)
        threads: list[ReviewThread] = []
        cursor: str | None = None
        while True:
            payload = self._graphql(THREADS_QUERY, owner, name, number, cursor)
            connection = _pr_node(payload, repo, number).get("reviewThreads") or {}
            threads.extend(parse_threads(connection.get("nodes") or [], bots=self._bots))
            page = connection.get("pageInfo") or {}
            if not page.get("hasNextPage") or not page.get("endCursor"):
                return threads
            cursor = page["endCursor"]

    def _collect_feedback(self, data: dict, threads: list[ReviewThread], me: str) -> list[Feedback]:
        """Reviews with a body + issue comments + thread comments, minus me and ignored logins,
        oldest first.

        Comments on *resolved* threads stay in the list on purpose: the ``rc`` version key is the
        newest ``created_at`` over this list, and a reviewer resolving a thread (which the factory
        cannot do) must not make that key regress to an older, never-recorded value and re-dispatch
        feedback that was already dispositioned. ``PullRequest.open_feedback`` drops them for the
        callers that decide whether there is work to do."""
        items: list[Feedback] = []
        for review in data.get("reviews") or []:
            body = str(review.get("body") or "").strip()
            if not body:
                continue
            author = _login_of(review.get("author"))
            items.append(
                Feedback(
                    id=str(review.get("id") or ""),
                    author=author,
                    author_is_bot=self._is_bot(author),
                    body=body,
                    created_at=str(review.get("submittedAt") or ""),
                    kind="review",
                    url=str(review.get("url") or ""),
                )
            )
        for comment in data.get("comments") or []:
            author = _login_of(comment.get("author"))
            items.append(
                Feedback(
                    id=str(comment.get("id") or ""),
                    author=author,
                    author_is_bot=self._is_bot(author),
                    body=str(comment.get("body") or ""),
                    created_at=str(comment.get("createdAt") or ""),
                    kind="comment",
                    url=str(comment.get("url") or ""),
                )
            )
        for thread in threads:
            items.extend(thread.comments)
        excluded = {_bare_login(me)} | self._ignored
        kept = [f for f in items if _bare_login(f.author) not in excluded]
        kept.sort(key=lambda f: (_sort_ts(f.created_at), f.id))
        return kept

    def _is_bot(self, login: str) -> bool:
        return is_bot_login(login, self._bots)


# ----------------------------------------------------------------------------------- parsers
# Module-level so tests (and other modules) can turn fixture JSON into models without a runner.


def parse_pr(repo: str, data: dict) -> PullRequest:
    """``gh pr view/list --json`` object -> ``PullRequest`` (checks/feedback/threads left empty)."""
    head_repo_obj = data.get("headRepository") or {}
    head_owner = _login_of(data.get("headRepositoryOwner")) if data.get("headRepositoryOwner") else ""
    head_repo = str(head_repo_obj.get("nameWithOwner") or "")
    if not head_repo and head_owner and head_repo_obj.get("name"):
        head_repo = f"{head_owner}/{head_repo_obj['name']}"
    state = str(data.get("state") or "OPEN").upper()
    mergeable = str(data.get("mergeable") or "UNKNOWN").upper()
    return PullRequest(
        repo=repo,
        number=int(data["number"]),
        title=str(data.get("title") or ""),
        url=str(data.get("url") or ""),
        author=_login_of(data.get("author")),
        body=str(data.get("body") or ""),
        state=state if state in ("OPEN", "CLOSED", "MERGED") else "OPEN",
        is_draft=bool(data.get("isDraft")),
        head_ref_name=str(data.get("headRefName") or ""),
        head_ref_oid=str(data.get("headRefOid") or ""),
        base_ref_name=str(data.get("baseRefName") or ""),
        base_ref_oid=str(data.get("baseRefOid") or ""),
        head_repo=head_repo or None,
        mergeable=mergeable if mergeable in ("MERGEABLE", "CONFLICTING", "UNKNOWN") else "UNKNOWN",
        created_at=str(data.get("createdAt") or ""),
        updated_at=str(data.get("updatedAt") or ""),
        merged_at=data.get("mergedAt") or None,
        closed_at=data.get("closedAt") or None,
    )


_CHECKRUN_CONCLUSION: dict[str, CheckState] = {
    "SUCCESS": "success",
    "FAILURE": "failure",
    "TIMED_OUT": "failure",
    "STARTUP_FAILURE": "failure",
    "ACTION_REQUIRED": "failure",
    "ERROR": "failure",
    "CANCELLED": "cancelled",
    "SKIPPED": "skipped",
    "NEUTRAL": "neutral",
}
_STATUS_CONTEXT_STATE: dict[str, CheckState] = {
    "SUCCESS": "success",
    "FAILURE": "failure",
    "ERROR": "failure",
    "PENDING": "pending",
    "EXPECTED": "pending",
}


def parse_checks(rollup: list[dict]) -> list[Check]:
    """``statusCheckRollup`` items (``CheckRun`` or ``StatusContext``) -> ``Check`` list."""
    checks: list[Check] = []
    for item in rollup:
        if not isinstance(item, dict):
            continue
        typename = str(item.get("__typename") or ("StatusContext" if "context" in item else "CheckRun"))
        if typename == "StatusContext":
            state = _STATUS_CONTEXT_STATE.get(str(item.get("state") or "").upper(), "unknown")
            url = str(item.get("targetUrl") or "")
            run_id, job_id = parse_actions_url(url)
            checks.append(
                Check(
                    name=str(item.get("context") or ""),
                    state=state,
                    details_url=url,
                    run_id=run_id,
                    job_id=job_id,
                    completed_at=None,
                )
            )
            continue
        status = str(item.get("status") or "").upper()
        conclusion = str(item.get("conclusion") or "").upper()
        if status and status != "COMPLETED":
            state = "pending"
        elif conclusion in _CHECKRUN_CONCLUSION:
            state = _CHECKRUN_CONCLUSION[conclusion]
        elif not conclusion and status == "COMPLETED":
            state = "unknown"
        elif not conclusion:
            state = "pending"
        else:
            state = "unknown"  # STALE and anything new
        url = str(item.get("detailsUrl") or "")
        run_id, job_id = parse_actions_url(url)
        completed = item.get("completedAt") or None
        if completed == _GH_NULL_TIME:
            completed = None
        checks.append(
            Check(
                name=str(item.get("name") or ""),
                state=state,
                workflow=str(item.get("workflowName") or ""),
                details_url=url,
                run_id=run_id,
                job_id=job_id,
                completed_at=completed,
            )
        )
    return checks


def parse_actions_url(url: str) -> tuple[int | None, int | None]:
    """``.../actions/runs/<run>/job/<job>`` -> ``(run_id, job_id)``; ``(None, None)`` otherwise."""
    match = _ACTIONS_URL.search(url or "")
    if not match:
        return None, None
    run_id = int(match.group(1))
    job_id = int(match.group(2)) if match.group(2) else None
    return run_id, job_id


def parse_threads(nodes: list[dict], bots: Iterable[str] = ()) -> list[ReviewThread]:
    """GraphQL ``reviewThreads.nodes`` -> ``ReviewThread`` list (comments become thread ``Feedback``)."""
    bot_set = {_bare_login(b) for b in bots}
    threads: list[ReviewThread] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        thread_id = str(node.get("id") or "")
        path = node.get("path") or None
        line = node.get("line")
        comments: list[Feedback] = []
        for comment in (node.get("comments") or {}).get("nodes") or []:
            author = _login_of(comment.get("author"))
            comments.append(
                Feedback(
                    id=str(comment.get("id") or ""),
                    author=author,
                    author_is_bot=is_bot_login(author, bot_set),
                    body=str(comment.get("body") or ""),
                    created_at=str(comment.get("createdAt") or ""),
                    kind="thread",
                    path=path,
                    line=int(line) if isinstance(line, int) else None,
                    thread_id=thread_id,
                    url=str(comment.get("url") or ""),
                )
            )
        threads.append(
            ReviewThread(
                id=thread_id,
                is_resolved=bool(node.get("isResolved")),
                is_outdated=bool(node.get("isOutdated")),
                path=path,
                line=int(line) if isinstance(line, int) else None,
                comments=comments,
            )
        )
    return threads


def parse_commits(nodes: list[dict]) -> list[PrCommit]:
    """GraphQL ``commits.nodes`` (``{commit: {...}}``) -> ``PrCommit`` list."""
    commits: list[PrCommit] = []
    for node in nodes:
        commit = (node or {}).get("commit") if isinstance(node, dict) else None
        if not isinstance(commit, dict) or not commit.get("oid"):
            continue
        author = commit.get("author") or {}
        user = author.get("user") or {}
        commits.append(
            PrCommit(
                sha=str(commit["oid"]),
                author_login=str(user["login"]) if user.get("login") else None,
                author_email=str(author.get("email") or ""),
                message=str(commit.get("message") or ""),
                committed_at=str(commit.get("committedDate") or ""),
                additions=int(commit.get("additions") or 0),
                deletions=int(commit.get("deletions") or 0),
            )
        )
    return commits


def is_bot_login(login: str, bots: Iterable[str] = ()) -> bool:
    """``[bot]`` suffix (``gh pr view``) or a configured bot reviewer. GraphQL drops the suffix
    (``copilot-pull-request-reviewer``), so comparison is suffix-insensitive."""
    if not login:
        return False
    if login.lower().endswith("[bot]"):
        return True
    bare = _bare_login(login)
    return any(bare == _bare_login(b) for b in bots)


def tail_bytes(text: str, max_bytes: int) -> str:
    """Keep the last ``max_bytes`` of ``text`` (cut at a line boundary) with a truncation marker."""
    raw = text.encode("utf-8")
    if max_bytes <= 0 or len(raw) <= max_bytes:
        return text
    cut = raw[-max_bytes:]
    newline = cut.find(b"\n")
    if 0 <= newline < len(cut) - 1:
        cut = cut[newline + 1 :]
    dropped = len(raw) - len(cut)
    return f"[... first {dropped} bytes of the failed log omitted; showing the tail ...]\n" + cut.decode(
        "utf-8", errors="replace"
    )


# ------------------------------------------------------------------------------------ helpers


def _now() -> datetime:  # patched in tests
    return datetime.now(UTC)


def parse_iso(value: str | None) -> datetime | None:
    """ISO-8601 (``Z`` or offset) -> aware UTC datetime; ``None`` when unparseable."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _within(updated_at: str, cutoff: datetime) -> bool:
    parsed = parse_iso(updated_at)
    return parsed is not None and parsed >= cutoff


def _sort_ts(value: str) -> datetime:
    return parse_iso(value) or datetime.min.replace(tzinfo=UTC)


def _json(out: str, what: str) -> Any:
    try:
        return json.loads(out) if out.strip() else []
    except ValueError as exc:
        raise GhError(f"gh {what} returned invalid JSON: {exc}") from exc


def _parse_search_item(item: dict) -> PullRequest | None:
    repo = str((item.get("repository") or {}).get("nameWithOwner") or "")
    number = item.get("number")
    if not repo or not isinstance(number, int):
        return None
    return PullRequest(
        repo=repo,
        number=number,
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        author=_login_of(item.get("author")),
        is_draft=bool(item.get("isDraft")),
        updated_at=str(item.get("updatedAt") or ""),
    )


def _requested_directly(requests: list[dict], me_lower: str) -> bool:
    """True when a ``User`` review request (not a ``Team``) names me."""
    for request in requests:
        if not isinstance(request, dict):
            continue
        typename = str(request.get("__typename") or ("Team" if "slug" in request else "User"))
        if typename == "User" and str(request.get("login") or "").lower() == me_lower:
            return True
    return False


def _pr_node(payload: dict, repo: str, number: int) -> dict:
    node = ((payload.get("data") or {}).get("repository") or {}).get("pullRequest")
    if not isinstance(node, dict):
        raise GhError(f"gh api graphql: pull request {repo}#{number} not found")
    return node


def _split_slug(repo: str) -> tuple[str, str]:
    owner, sep, name = repo.partition("/")
    if not sep or not owner or not name:
        raise GhError(f"invalid repo slug {repo!r}; expected owner/name")
    return owner, name


def _login_of(author: Any) -> str:
    if isinstance(author, dict):
        return str(author.get("login") or "")
    return str(author or "")


def _bare_login(login: str) -> str:
    return login.strip().lower().removesuffix("[bot]")


def _number_from_url(out: str) -> int | None:
    for line in reversed(out.strip().splitlines()):
        match = _PR_URL.search(line.strip())
        if match:
            return int(match.group(1))
    return None
