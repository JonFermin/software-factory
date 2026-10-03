"""trackers/linear.py: GraphQL over httpx.MockTransport -- auth mapping, rate-limit retry, pagination,
issue parsing (blocked-by from inverse relations), every write's mutation name + variables, context."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from software_factory.config import Profile
from software_factory.trackers.base import (
    Attachment,
    Comment,
    ContextSource,
    Issue,
    IssueRef,
    Tracker,
    TrackerAuthError,
    TrackerError,
)
from software_factory.trackers.linear import LINEAR_API, LinearTracker, render_context

FIXTURES = Path(__file__).parent / "fixtures" / "linear"
_OPERATION = re.compile(r"\b(?:query|mutation)\s+(\w+)")


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def issue_node(**overrides: Any) -> dict:
    node = fixture("issue.json")
    node.update(overrides)
    return node


class LinearServer:
    """MockTransport handler routing on the GraphQL operation name. ``on(op, *responses)`` queues
    responses for an operation (the last one repeats); each may be ``data`` (dict), an
    ``httpx.Response``, or a callable ``variables -> data | Response``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._queues: dict[str, list[Any]] = {}

    def on(self, op: str, *responses: Any) -> LinearServer:
        self._queues[op] = list(responses)
        return self

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        match = _OPERATION.search(body["query"])
        op = match.group(1) if match else "?"
        self.calls.append({"op": op, "variables": body.get("variables") or {}, "headers": dict(request.headers), "url": str(request.url)})
        queue = self._queues.get(op)
        if not queue:
            return httpx.Response(200, json={"errors": [{"message": f"test server: no response queued for {op}"}]})
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(item):
            item = item(self.calls[-1]["variables"])
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, json={"data": item})

    def ops(self) -> list[str]:
        return [c["op"] for c in self.calls]

    def variables(self, index: int = -1) -> dict[str, Any]:
        return self.calls[index]["variables"]


def page(nodes: list[dict], *, next_cursor: str | None = None) -> dict:
    return {"nodes": nodes, "pageInfo": {"hasNextPage": next_cursor is not None, "endCursor": next_cursor}}


@pytest.fixture
def server() -> LinearServer:
    return LinearServer()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def make_tracker(server: LinearServer, sleeps: list[float]) -> Callable[..., LinearTracker]:
    def _make(profile: Profile | None = None, api_key: str | None = "lin_api_test") -> LinearTracker:
        profile = profile or Profile(name="t", secrets={"LINEAR_API_KEY": "lin_api_from_env"})
        http = httpx.Client(transport=httpx.MockTransport(server))
        return LinearTracker(profile, http, api_key=api_key, sleep=sleeps.append)

    return _make


@pytest.fixture
def tracker(make_tracker: Callable[..., LinearTracker]) -> LinearTracker:
    return make_tracker()


# ------------------------------------------------------------------------------- transport


def test_tracker_implements_both_protocols(tracker: LinearTracker):
    assert isinstance(tracker, Tracker) and isinstance(tracker, ContextSource)
    assert tracker.name == "linear"


def test_ping_posts_to_api_with_raw_api_key_header(tracker: LinearTracker, server: LinearServer):
    server.on("Viewer", {"viewer": {"id": "u1", "name": "Test User", "displayName": "tester"}})
    assert tracker.ping() == "tester"
    call = server.calls[0]
    assert call["url"] == LINEAR_API and call["op"] == "Viewer"
    assert call["headers"]["authorization"] == "lin_api_test"  # personal keys: no "Bearer"
    assert call["headers"]["content-type"] == "application/json"


def test_api_key_defaults_to_profile_secret(make_tracker, server: LinearServer):
    server.on("Viewer", {"viewer": {"name": "Test User"}})
    tracker = make_tracker(api_key=None)
    assert tracker.ping() == "Test User"
    assert server.calls[0]["headers"]["authorization"] == "lin_api_from_env"


def test_missing_api_key_is_an_auth_error_before_any_request(make_tracker, server: LinearServer):
    tracker = make_tracker(Profile(name="t"), api_key=None)
    with pytest.raises(TrackerAuthError, match="LINEAR_API_KEY"):
        tracker.ping()
    assert server.calls == []


@pytest.mark.parametrize("status", [401, 403])
def test_http_401_403_map_to_auth_error(tracker: LinearTracker, server: LinearServer, status: int):
    server.on("Viewer", httpx.Response(status, json=fixture("auth_error.json")))
    with pytest.raises(TrackerAuthError, match="authenticate"):
        tracker.ping()
    assert len(server.calls) == 1  # never retried


def test_graphql_authentication_error_with_http_200_maps_to_auth_error(tracker: LinearTracker, server: LinearServer):
    server.on("Viewer", httpx.Response(200, json=fixture("auth_error.json")))
    with pytest.raises(TrackerAuthError):
        tracker.ping()


def test_other_graphql_errors_raise_tracker_error_with_message(tracker: LinearTracker, server: LinearServer):
    body = {"errors": [{"message": "Entity not found: Issue", "extensions": {"type": "invalid input"}}, {"message": "second"}]}
    server.on("Issue", httpx.Response(200, json=body))
    with pytest.raises(TrackerError, match="Entity not found: Issue; second") as exc:
        tracker.get_issue("0a1b2c3d-0000-4000-8000-000000000000")
    assert not isinstance(exc.value, TrackerAuthError)


def test_non_json_http_failure_and_transport_errors_become_tracker_errors(make_tracker, server: LinearServer):
    server.on("Viewer", httpx.Response(500, text="<html>bad gateway</html>"))
    with pytest.raises(TrackerError, match="HTTP 500"):
        make_tracker().ping()

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    offline = LinearTracker(Profile(name="t"), httpx.Client(transport=httpx.MockTransport(boom)), api_key="k")
    with pytest.raises(TrackerError, match="ConnectError"):
        offline.ping()


def test_rate_limits_retry_with_bounded_backoff(tracker: LinearTracker, server: LinearServer, sleeps: list[float]):
    server.on(
        "Viewer",
        httpx.Response(429, headers={"Retry-After": "2"}, text="slow down"),
        httpx.Response(400, json=fixture("rate_limited.json")),  # Linear's documented shape: 400 + RATELIMITED
        {"viewer": {"name": "Test User"}},
    )
    assert tracker.ping() == "Test User"
    assert server.ops() == ["Viewer"] * 3
    assert sleeps == [2.0, 2.0]  # Retry-After honoured, then the default backoff ladder


def test_rate_limit_gives_up_after_max_retries(tracker: LinearTracker, server: LinearServer, sleeps: list[float]):
    server.on("Viewer", httpx.Response(429, text="slow down"))
    with pytest.raises(TrackerError, match="rate limit"):
        tracker.ping()
    assert len(server.calls) == 4 and len(sleeps) == 3


# ------------------------------------------------------------------------------- reads


def test_get_issue_parses_the_full_shape(tracker: LinearTracker, server: LinearServer):
    server.on("Issue", {"issue": issue_node()})
    issue = tracker.get_issue("FAC-12")

    assert server.variables() == {"id": "FAC-12"}
    assert issue.id == "iss-12" and issue.key == "FAC-12" and issue.title == "Add drawbridge lever"
    assert issue.url.endswith("/FAC-12/add-drawbridge-lever") and issue.updated_at == "2026-09-30T10:00:00.000Z"
    assert issue.branch_name == "tester/fac-12-add-drawbridge-lever"
    assert issue.state_type == "unstarted" and issue.state_name == "Todo"
    assert issue.labels == ["factory:build", "repo:castle-sim"] and issue.repo_label() == "castle-sim"
    assert issue.team_id == "team-1" and issue.team_key == "FAC"
    assert issue.project_id == "proj-1" and issue.project_name == "Castle"
    assert issue.parent == IssueRef(id="iss-10", key="FAC-10", state_type="started", title="Castle epic")
    assert [c.key for c in issue.children] == ["FAC-13", "FAC-16"]
    assert issue.attachments == [
        Attachment(url="https://github.com/JaneDoe/castle-sim/pull/7", title="PR #7", source_type="githubPr"),
        Attachment(url="https://www.figma.com/file/abc", title="Lever mock", source_type="figma"),
    ]
    # Comments come back oldest first regardless of API order; bots are credited by botActor name.
    assert issue.comments == [
        Comment(author="Bugbot", body="First comment, from a bot.", created_at="2026-09-28T12:00:00.000Z"),
        Comment(author="tester", body="Second comment, newest.", created_at="2026-09-29T12:00:00.000Z"),
    ]


def test_blocked_by_comes_from_inverse_blocks_relations_only(tracker: LinearTracker, server: LinearServer):
    server.on("Issue", {"issue": issue_node()})
    issue = tracker.get_issue("iss-12")
    # FAC-11 and FAC-14 "block" FAC-12 (inverseRelations); FAC-15 is blocked *by* FAC-12 (relations) and
    # "related" relations are not blockers.
    assert [(b.key, b.state_type) for b in issue.blocked_by] == [("FAC-11", "completed"), ("FAC-14", "started")]
    assert [b.key for b in issue.unfinished_blockers] == ["FAC-14"]


def test_blocked_by_honours_a_blocked_by_relation_type_if_the_api_exposes_it(tracker: LinearTracker, server: LinearServer):
    node = issue_node(
        inverseRelations={"nodes": []},
        relations={
            "nodes": [
                {"type": "blocked_by", "relatedIssue": {"id": "iss-30", "identifier": "FAC-30", "title": "x", "state": {"type": "started"}}},
                {"type": "blocked_by", "relatedIssue": {"id": "iss-30", "identifier": "FAC-30", "title": "x", "state": {"type": "started"}}},
            ]
        },
    )
    server.on("Issue", {"issue": node})
    assert [b.key for b in tracker.get_issue("iss-12").blocked_by] == ["FAC-30"]  # deduped


def test_get_issue_tolerates_sparse_nodes(tracker: LinearTracker, server: LinearServer):
    sparse = {"id": "iss-1", "identifier": "FAC-1", "title": "t", "state": {"type": "backlog"}, "team": {"id": "team-1", "key": "FAC"}}
    server.on("Issue", {"issue": sparse})
    issue = tracker.get_issue("FAC-1")
    assert issue.description == "" and issue.project_id is None and issue.parent is None
    assert issue.children == [] and issue.blocked_by == [] and issue.attachments == [] and issue.comments == []


def test_get_issue_by_key_falls_back_to_team_and_number_filter(tracker: LinearTracker, server: LinearServer):
    not_found = httpx.Response(200, json={"errors": [{"message": "Entity not found: Issue - Could not find referenced Issue."}]})
    server.on("Issue", not_found)
    server.on("IssueByKey", {"issues": page([issue_node()])})
    assert tracker.get_issue("FAC-12").key == "FAC-12"
    assert server.ops() == ["Issue", "IssueByKey"]
    assert server.variables() == {"filter": {"team": {"key": {"eqIgnoreCase": "FAC"}}, "number": {"eq": 12.0}}}

    server.on("IssueByKey", {"issues": page([])})
    with pytest.raises(TrackerError, match="FAC-12 not found"):
        tracker.get_issue("FAC-12")
    # A UUID is never retried by key.
    with pytest.raises(TrackerError, match="Entity not found"):
        tracker.get_issue("0a1b2c3d-0000-4000-8000-000000000000")
    assert server.ops()[-1] == "Issue"


def test_list_issues_with_label_paginates_and_filters(make_tracker, server: LinearServer):
    first = issue_node(id="iss-1", identifier="FAC-1")
    second = issue_node(id="iss-2", identifier="FAC-2")
    third = issue_node(id="iss-3", identifier="FAC-3")
    server.on("Issues", lambda v: {"issues": page([first, second], next_cursor="c1") if v["after"] is None else page([third])})
    tracker = make_tracker(Profile(name="t", linear={"team_keys": ["FAC", "OPS"]}))

    issues = tracker.list_issues_with_label("factory:build")

    assert [i.key for i in issues] == ["FAC-1", "FAC-2", "FAC-3"]
    assert server.ops() == ["Issues", "Issues"]
    assert [c["variables"]["after"] for c in server.calls] == [None, "c1"]
    assert all(c["variables"]["first"] == 50 for c in server.calls)
    assert server.variables(0)["filter"] == {
        "labels": {"name": {"eqIgnoreCase": "factory:build"}},
        "state": {"type": {"nin": ["completed", "canceled"]}},
        "team": {"key": {"in": ["FAC", "OPS"]}},
    }


def test_list_issues_without_team_keys_has_no_team_filter(tracker: LinearTracker, server: LinearServer):
    server.on("Issues", {"issues": page([])})
    assert tracker.list_issues_with_label("factory:plan") == []
    assert "team" not in server.variables()["filter"]


def test_sibling_issues_by_project_then_team_and_limit(tracker: LinearTracker, server: LinearServer):
    me = Issue(id="iss-12", key="FAC-12", title="t", project_id="proj-1", team_id="team-1")
    others = [issue_node(id=f"iss-{n}", identifier=f"FAC-{n}") for n in (20, 21, 22)]
    server.on("Issues", {"issues": page(others, next_cursor="more")})

    siblings = tracker.sibling_issues(me, limit=2)

    assert [s.key for s in siblings] == ["FAC-20", "FAC-21"]
    assert server.ops() == ["Issues"]  # limit reached: no second page
    assert server.variables()["first"] == 2
    assert server.variables()["filter"] == {"project": {"id": {"eq": "proj-1"}}, "id": {"neq": "iss-12"}}

    server.on("Issues", {"issues": page(others + [issue_node()])})
    no_project = Issue(id="iss-12", key="FAC-12", title="t", team_id="team-1")
    assert [s.key for s in tracker.sibling_issues(no_project)] == ["FAC-20", "FAC-21", "FAC-22"]  # self filtered
    assert server.variables()["filter"] == {"team": {"id": {"eq": "team-1"}}, "id": {"neq": "iss-12"}}

    assert tracker.sibling_issues(Issue(id="x", key="X-1", title="t")) == []
    assert len(server.calls) == 2  # no project and no team: no request


def test_labels_exist_is_one_case_insensitive_query(tracker: LinearTracker, server: LinearServer):
    server.on("IssueLabels", {"issueLabels": page([{"id": "l1", "name": "Factory:Build", "team": None}, {"id": "l2", "name": "repo:x", "team": {"id": "t"}}])})
    assert tracker.labels_exist(["factory:build", "repo:x", "factory:plan"]) == {"factory:build": True, "repo:x": True, "factory:plan": False}
    assert server.ops() == ["IssueLabels"]
    assert server.variables()["filter"] == {
        "or": [{"name": {"eqIgnoreCase": n}} for n in ("factory:build", "repo:x", "factory:plan")]
    }
    assert tracker.labels_exist([]) == {} and len(server.calls) == 1


def test_ensure_label_prefers_team_label_then_workspace_and_never_creates(tracker: LinearTracker, server: LinearServer):
    both = page([{"id": "ws-1", "name": "repo:x", "team": None}, {"id": "team-lbl", "name": "repo:x", "team": {"id": "team-1"}}])
    server.on("IssueLabels", {"issueLabels": both})
    assert tracker.ensure_label("team-1", "repo:x") == "team-lbl"
    assert tracker.ensure_label("team-2", "repo:x") == "ws-1"
    assert server.variables()["filter"] == {"name": {"eqIgnoreCase": "repo:x"}}

    server.on("IssueLabels", {"issueLabels": page([{"id": "other", "name": "repo:x", "team": {"id": "team-9"}}])})
    with pytest.raises(TrackerError, match="repo:x"):
        tracker.ensure_label("team-1", "repo:x")
    assert set(server.ops()) == {"IssueLabels"}  # lookups only; no label mutation ever


# ------------------------------------------------------------------------------- writes


def test_create_sub_issue_applies_only_existing_labels(tracker: LinearTracker, server: LinearServer):
    parent = Issue(id="iss-10", key="FAC-10", title="Epic", team_id="team-1", project_id="proj-1")

    def labels(variables: dict) -> dict:
        name = variables["filter"]["name"]["eqIgnoreCase"]
        nodes = [{"id": "lbl-repo", "name": "repo:castle-sim", "team": None}] if name == "repo:castle-sim" else []
        return {"issueLabels": page(nodes)}

    server.on("IssueLabels", labels)
    created = issue_node(id="iss-99", identifier="FAC-99", title="Lever", parent={"id": "iss-10", "identifier": "FAC-10", "title": "Epic", "state": {"type": "started"}})
    created["labels"] = {"nodes": [{"id": "lbl-repo", "name": "repo:castle-sim"}]}
    server.on("IssueCreate", {"issueCreate": {"success": True, "issue": created}})

    issue = tracker.create_sub_issue(parent, "Lever", "Body", ["repo:castle-sim", "repo:missing"])

    assert server.ops() == ["IssueLabels", "IssueLabels", "IssueCreate"]
    assert server.variables() == {
        "input": {
            "teamId": "team-1",
            "title": "Lever",
            "description": "Body",
            "parentId": "iss-10",
            "projectId": "proj-1",
            "labelIds": ["lbl-repo"],
        }
    }
    assert issue.key == "FAC-99" and issue.labels == ["repo:castle-sim"] and issue.parent.key == "FAC-10"


def test_create_sub_issue_without_labels_or_project_and_failure(tracker: LinearTracker, server: LinearServer):
    parent = Issue(id="iss-10", key="FAC-10", title="Epic", team_id="team-1")
    server.on("IssueCreate", {"issueCreate": {"success": True, "issue": issue_node(id="iss-98", identifier="FAC-98")}})
    tracker.create_sub_issue(parent, "T", "D", [])
    assert server.ops() == ["IssueCreate"]
    assert server.variables()["input"] == {"teamId": "team-1", "title": "T", "description": "D", "parentId": "iss-10"}

    server.on("IssueCreate", {"issueCreate": {"success": False, "issue": None}})
    with pytest.raises(TrackerError, match="issueCreate failed"):
        tracker.create_sub_issue(parent, "T", "D", [])
    with pytest.raises(TrackerError, match="no team"):
        tracker.create_sub_issue(Issue(id="p", key="P-1", title="t"), "T", "D", [])


def test_add_blocked_by_creates_a_blocks_relation_from_blocker_to_blocked(tracker: LinearTracker, server: LinearServer):
    server.on("IssueRelationCreate", {"issueRelationCreate": {"success": True, "issueRelation": {"id": "rel-1"}}})
    tracker.add_blocked_by("iss-blocked", "iss-blocker")
    assert server.ops() == ["IssueRelationCreate"]
    assert server.variables() == {"input": {"issueId": "iss-blocker", "relatedIssueId": "iss-blocked", "type": "blocks"}}

    server.on("IssueRelationCreate", {"issueRelationCreate": {"success": False}})
    with pytest.raises(TrackerError, match="issueRelationCreate failed"):
        tracker.add_blocked_by("a", "b")


def test_move_to_state_type_picks_lowest_position_state_of_the_issue_team(tracker: LinearTracker, server: LinearServer):
    server.on("IssueLabelsOnIssue", {"issue": {"id": "iss-12", "team": {"id": "team-1"}, "labels": {"nodes": []}}})
    states = [
        {"id": "st-review", "name": "In Review", "type": "started", "position": 2.0},
        {"id": "st-progress", "name": "In Progress", "type": "started", "position": 0.5},
    ]
    server.on("WorkflowStates", {"workflowStates": page(states)})
    server.on("IssueUpdate", {"issueUpdate": {"success": True}})

    tracker.move_to_state_type("iss-12", "started")

    assert server.ops() == ["IssueLabelsOnIssue", "WorkflowStates", "IssueUpdate"]
    assert server.variables(1)["filter"] == {"team": {"id": {"eq": "team-1"}}, "type": {"eq": "started"}}
    assert server.variables(2) == {"id": "iss-12", "input": {"stateId": "st-progress"}}

    server.on("WorkflowStates", {"workflowStates": page([])})
    with pytest.raises(TrackerError, match="no workflow state of type 'completed'"):
        tracker.move_to_state_type("iss-12", "completed")
    server.on("IssueLabelsOnIssue", {"issue": None})
    with pytest.raises(TrackerError, match="not found"):
        tracker.move_to_state_type("nope", "started")


def test_update_issue_text_uses_issue_update(tracker: LinearTracker, server: LinearServer):
    server.on("IssueUpdate", {"issueUpdate": {"success": True}})
    tracker.update_issue_text("iss-12", "New title", "New body")
    assert server.ops() == ["IssueUpdate"]
    assert server.variables() == {"id": "iss-12", "input": {"title": "New title", "description": "New body"}}

    server.on("IssueUpdate", {"issueUpdate": {"success": False}})
    with pytest.raises(TrackerError, match="issueUpdate failed"):
        tracker.update_issue_text("iss-12", "t", "d")


def test_remove_label_uses_issue_remove_label_and_is_a_noop_when_absent(tracker: LinearTracker, server: LinearServer):
    labels = {"nodes": [{"id": "lbl-refine", "name": "Factory:Refine"}, {"id": "lbl-build", "name": "factory:build"}]}
    server.on("IssueLabelsOnIssue", {"issue": {"id": "iss-12", "team": {"id": "team-1"}, "labels": labels}})
    server.on("IssueRemoveLabel", {"issueRemoveLabel": {"success": True}})

    tracker.remove_label("iss-12", "factory:refine")
    assert server.ops() == ["IssueLabelsOnIssue", "IssueRemoveLabel"]
    assert server.variables() == {"id": "iss-12", "labelId": "lbl-refine"}

    tracker.remove_label("iss-12", "factory:plan")  # not on the issue
    assert server.ops() == ["IssueLabelsOnIssue", "IssueRemoveLabel", "IssueLabelsOnIssue"]


# ------------------------------------------------------------------------------- context


def test_context_for_renders_the_issue_as_markdown(tracker: LinearTracker, server: LinearServer):
    server.on("Issue", {"issue": issue_node()})
    issue = tracker.get_issue("FAC-12")
    text = tracker.context_for(issue)

    assert text.startswith("# FAC-12: Add drawbridge lever\n")
    assert "- URL: https://linear.app/acme/issue/FAC-12/add-drawbridge-lever" in text
    assert "- State: Todo (unstarted)" in text
    assert "- Team: FAC" in text and "- Project: Castle" in text
    assert "- Branch: `tester/fac-12-add-drawbridge-lever`" in text
    assert "- Labels: `factory:build`, `repo:castle-sim`" in text
    assert "## Description\n\nThe drawbridge needs a lever" in text
    assert "## Parent\n\n- FAC-10 [started] Castle epic" in text
    assert "## Sub-issues\n\n- FAC-13 [completed] Lever art\n- FAC-16 [backlog] Lever sound" in text
    assert "## Blocked by\n\n- FAC-11 [completed] Gatehouse mesh\n- FAC-14 [started] Physics hooks" in text
    assert "- [PR #7](https://github.com/JaneDoe/castle-sim/pull/7) (githubPr)" in text
    assert "## Comments\n\n**Bugbot** — 2026-09-28T12:00:00.000Z\n\nFirst comment, from a bot." in text
    assert text.index("First comment") < text.index("Second comment")


def test_context_for_keeps_last_20_comments_and_truncates_bodies():
    comments = [Comment(author=f"u{i}", body=f"c{i} " + "x" * 2000, created_at=f"2026-01-{i + 1:02d}") for i in range(25)]
    issue = Issue(id="i", key="K-1", title="T", comments=comments)
    text = render_context(issue)
    assert "## Comments (last 20 of 25)" in text
    assert "**u4**" not in text and "**u5**" in text and "**u24**" in text
    assert "…" in text and "x" * 1300 not in text
    minimal = render_context(Issue(id="i", key="K-2", title="Bare"))
    assert minimal == "# K-2: Bare\n\n- Labels: (none)\n\n## Description\n\n_(empty)_\n"
