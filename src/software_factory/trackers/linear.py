"""Linear tracker: GraphQL over httpx with the profile's ``LINEAR_API_KEY``.

Names below were checked against Linear's public schema (``packages/sdk/src/schema.graphql``) and
developer docs:

* Personal API keys go in ``Authorization: <key>`` (no ``Bearer``; that prefix is for OAuth tokens).
* Rate limits come back as HTTP 400 with ``errors[].extensions.code == "RATELIMITED"`` (429 is
  handled too); auth failures as HTTP 401/403 or ``extensions.type == "authentication error"``.
* ``IssueRelation.type`` is ``blocks | duplicate | related | similar``. Issue *X* is blocked by *Y*
  when *Y* has a ``blocks`` relation whose ``relatedIssue`` is *X* -- i.e. ``X.inverseRelations``
  with ``type == "blocks"``. A ``blocked_by`` type is not in the schema today but is honoured on
  ``X.relations`` should it appear.
* Writes use ``issueCreate``, ``issueUpdate``, ``issueRelationCreate`` and ``issueRemoveLabel``.
  Label *creation* is not an allowed write, so ``ensure_label`` only looks labels up.

Reads are free; every write here is called only from ``effects.py``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

import httpx

from ..config import Profile
from .base import (
    Attachment,
    Comment,
    Issue,
    IssueRef,
    StateType,
    TrackerAuthError,
    TrackerError,
)

LINEAR_API = "https://api.linear.app/graphql"
PAGE_SIZE = 50
TIMEOUT_S = 30.0
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_S = (1.0, 2.0, 4.0)
MAX_RETRY_AFTER_S = 30.0
FINISHED_STATE_TYPES = ("completed", "canceled")
CONTEXT_COMMENTS = 20
CONTEXT_COMMENT_CHARS = 1200

_ISSUE_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9]*-\d+$")

# One fragment for every issue read so parsing happens in exactly one place (``_parse_issue``).
ISSUE_FRAGMENT = """
fragment IssueFields on Issue {
  id identifier title description url updatedAt branchName
  state { type name }
  labels(first: 50) { nodes { id name } }
  team { id key }
  project { id name }
  parent { id identifier title state { type } }
  children(first: 50) { nodes { id identifier title state { type } } }
  relations(first: 50) { nodes { type relatedIssue { id identifier title state { type } } } }
  inverseRelations(first: 50) { nodes { type issue { id identifier title state { type } } } }
  attachments(first: 50) { nodes { url title sourceType } }
  comments(first: 50) { nodes { body createdAt user { name displayName } botActor { name } } }
}
"""

Q_VIEWER = "query Viewer { viewer { id name displayName } }"
Q_ISSUE = ISSUE_FRAGMENT + "query Issue($id: String!) { issue(id: $id) { ...IssueFields } }"
Q_ISSUE_BY_KEY = ISSUE_FRAGMENT + (
    "query IssueByKey($filter: IssueFilter!) { issues(filter: $filter, first: 1) { nodes { ...IssueFields } } }"
)
Q_ISSUES = ISSUE_FRAGMENT + (
    "query Issues($filter: IssueFilter!, $first: Int!, $after: String) {"
    " issues(filter: $filter, first: $first, after: $after, orderBy: updatedAt) {"
    " nodes { ...IssueFields } pageInfo { hasNextPage endCursor } } }"
)
Q_ISSUE_LABELS = (
    "query IssueLabels($filter: IssueLabelFilter!, $first: Int!, $after: String) {"
    " issueLabels(filter: $filter, first: $first, after: $after) {"
    " nodes { id name team { id } } pageInfo { hasNextPage endCursor } } }"
)
Q_ISSUE_LABELS_ON_ISSUE = (
    "query IssueLabelsOnIssue($id: String!) { issue(id: $id) { id team { id } labels(first: 50) { nodes { id name } } } }"
)
Q_WORKFLOW_STATES = (
    "query WorkflowStates($filter: WorkflowStateFilter!, $first: Int!, $after: String) {"
    " workflowStates(filter: $filter, first: $first, after: $after) {"
    " nodes { id name type position } pageInfo { hasNextPage endCursor } } }"
)
M_ISSUE_CREATE = ISSUE_FRAGMENT + (
    "mutation IssueCreate($input: IssueCreateInput!) { issueCreate(input: $input) { success issue { ...IssueFields } } }"
)
M_ISSUE_UPDATE = (
    "mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) { issueUpdate(id: $id, input: $input) { success } }"
)
M_ISSUE_RELATION_CREATE = (
    "mutation IssueRelationCreate($input: IssueRelationCreateInput!) {"
    " issueRelationCreate(input: $input) { success issueRelation { id } } }"
)
M_ISSUE_REMOVE_LABEL = (
    "mutation IssueRemoveLabel($id: String!, $labelId: String!) { issueRemoveLabel(id: $id, labelId: $labelId) { success } }"
)


class LinearTracker:
    """``Tracker`` + ``ContextSource`` over Linear's GraphQL API.

    ``http`` is injectable (``httpx.MockTransport`` in tests); ``sleep`` is the rate-limit backoff
    sleeper. The API key comes from ``profile.secrets["LINEAR_API_KEY"]`` unless ``api_key`` is given.
    """

    name = "linear"

    def __init__(
        self,
        profile: Profile,
        http: httpx.Client | None = None,
        *,
        api_key: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.profile = profile
        self._api_key = api_key if api_key is not None else profile.secret("LINEAR_API_KEY")
        self._http = http
        self._sleep = sleep

    # ================================================================================ reads
    def ping(self) -> str:
        data = self._gql(Q_VIEWER)
        viewer = data.get("viewer") or {}
        return str(viewer.get("displayName") or viewer.get("name") or viewer.get("id") or "")

    def list_issues_with_label(self, label: str) -> list[Issue]:
        """Open (not completed/canceled) issues carrying ``label``, limited to ``linear.team_keys``."""
        filter_: dict[str, Any] = {
            "labels": {"name": {"eqIgnoreCase": label}},
            "state": {"type": {"nin": list(FINISHED_STATE_TYPES)}},
        }
        if self.profile.linear.team_keys:
            filter_["team"] = {"key": {"in": list(self.profile.linear.team_keys)}}
        nodes = self._paginate(Q_ISSUES, {"filter": filter_}, "issues")
        return [_parse_issue(n) for n in nodes]

    def get_issue(self, id_or_key: str) -> Issue:
        """By UUID or key (``ABC-123``). Linear resolves keys in ``issue(id:)``; a not-found on a key
        falls back to a team-key + number filter in case that ever changes."""
        try:
            data = self._gql(Q_ISSUE, {"id": id_or_key})
        except TrackerAuthError:
            raise
        except TrackerError as exc:
            if not (_ISSUE_KEY.match(id_or_key) and "not found" in str(exc).lower()):
                raise
            team_key, _, number = id_or_key.rpartition("-")
            filter_ = {"team": {"key": {"eqIgnoreCase": team_key}}, "number": {"eq": float(number)}}
            nodes = (self._gql(Q_ISSUE_BY_KEY, {"filter": filter_}).get("issues") or {}).get("nodes") or []
            if not nodes:
                raise TrackerError(f"issue {id_or_key} not found") from exc
            return _parse_issue(nodes[0])
        node = data.get("issue")
        if not node:
            raise TrackerError(f"issue {id_or_key} not found")
        return _parse_issue(node)

    def sibling_issues(self, issue: Issue, limit: int = 50) -> list[Issue]:
        """Other issues in the same project (or team when the issue has no project)."""
        if issue.project_id:
            filter_: dict[str, Any] = {"project": {"id": {"eq": issue.project_id}}}
        elif issue.team_id:
            filter_ = {"team": {"id": {"eq": issue.team_id}}}
        else:
            return []
        filter_["id"] = {"neq": issue.id}
        nodes = self._paginate(Q_ISSUES, {"filter": filter_}, "issues", limit=limit)
        return [_parse_issue(n) for n in nodes if n.get("id") != issue.id][:limit]

    def labels_exist(self, names: list[str]) -> dict[str, bool]:
        """Case-insensitive existence check (workspace or team labels), one query."""
        if not names:
            return {}
        filter_ = {"or": [{"name": {"eqIgnoreCase": n}} for n in names]}
        found = {str(n.get("name", "")).lower() for n in self._paginate(Q_ISSUE_LABELS, {"filter": filter_}, "issueLabels")}
        return {name: name.lower() in found for name in names}

    def ensure_label(self, team_id: str, name: str) -> str:
        """Id of the label ``name`` usable by ``team_id`` (team label preferred, else workspace label).
        Only a lookup: label creation is not an allowed write, so a missing label raises."""
        label_id = self._find_label_id(team_id, name)
        if label_id is None:
            raise TrackerError(f"label {name!r} does not exist for team {team_id} (create it in Linear)")
        return label_id

    # ====================================================================== writes (effects only)
    def create_sub_issue(self, parent: Issue, title: str, description: str, labels: list[str]) -> Issue:
        """``issueCreate`` under ``parent`` (same team/project). Labels that do not exist are skipped;
        the returned ``Issue.labels`` is what was actually applied."""
        if not parent.team_id:
            raise TrackerError(f"parent {parent.key} has no team id; cannot create a sub-issue")
        input_: dict[str, Any] = {
            "teamId": parent.team_id,
            "title": title,
            "description": description,
            "parentId": parent.id,
        }
        if parent.project_id:
            input_["projectId"] = parent.project_id
        label_ids = [lid for lid in (self._find_label_id(parent.team_id, lbl) for lbl in labels) if lid]
        if label_ids:
            input_["labelIds"] = label_ids
        payload = self._gql(M_ISSUE_CREATE, {"input": input_}).get("issueCreate") or {}
        if not payload.get("success") or not payload.get("issue"):
            raise TrackerError(f"issueCreate failed for {title!r}")
        return _parse_issue(payload["issue"])

    def add_blocked_by(self, issue_id: str, blocker_id: str) -> None:
        """``blocker`` blocks ``issue``: relation ``blocks`` from ``blocker`` to ``issue``."""
        input_ = {"issueId": blocker_id, "relatedIssueId": issue_id, "type": "blocks"}
        payload = self._gql(M_ISSUE_RELATION_CREATE, {"input": input_}).get("issueRelationCreate") or {}
        if not payload.get("success"):
            raise TrackerError(f"issueRelationCreate failed ({blocker_id} blocks {issue_id})")

    def move_to_state_type(self, issue_id: str, state_type: StateType) -> None:
        """Move to the lowest-position workflow state of ``state_type`` in the issue's team."""
        issue_data = self._gql(Q_ISSUE_LABELS_ON_ISSUE, {"id": issue_id}).get("issue") or {}
        team_id = (issue_data.get("team") or {}).get("id")
        if not team_id:
            raise TrackerError(f"issue {issue_id} not found or has no team")
        filter_ = {"team": {"id": {"eq": team_id}}, "type": {"eq": state_type}}
        states = self._paginate(Q_WORKFLOW_STATES, {"filter": filter_}, "workflowStates")
        if not states:
            raise TrackerError(f"team {team_id} has no workflow state of type {state_type!r}")
        target = min(states, key=lambda s: float(s.get("position") or 0.0))
        self._update_issue(issue_id, {"stateId": target["id"]})

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None:
        self._update_issue(issue_id, {"title": title, "description": description})

    def remove_label(self, issue_id: str, label: str) -> None:
        """``issueRemoveLabel``; a label the issue does not carry is a no-op."""
        issue_data = self._gql(Q_ISSUE_LABELS_ON_ISSUE, {"id": issue_id}).get("issue") or {}
        for node in (issue_data.get("labels") or {}).get("nodes") or []:
            if str(node.get("name", "")).lower() == label.lower() and node.get("id"):
                payload = self._gql(M_ISSUE_REMOVE_LABEL, {"id": issue_id, "labelId": node["id"]})
                if not (payload.get("issueRemoveLabel") or {}).get("success"):
                    raise TrackerError(f"issueRemoveLabel failed ({label!r} on {issue_id})")
                return

    # ============================================================================= context
    def context_for(self, issue: Issue) -> str:
        return render_context(issue)

    # ============================================================================ plumbing
    def _update_issue(self, issue_id: str, input_: dict[str, Any]) -> None:
        payload = self._gql(M_ISSUE_UPDATE, {"id": issue_id, "input": input_}).get("issueUpdate") or {}
        if not payload.get("success"):
            raise TrackerError(f"issueUpdate failed for {issue_id} ({', '.join(sorted(input_))})")

    def _find_label_id(self, team_id: str, name: str) -> str | None:
        filter_ = {"name": {"eqIgnoreCase": name}}
        workspace: str | None = None
        for node in self._paginate(Q_ISSUE_LABELS, {"filter": filter_}, "issueLabels"):
            label_team = (node.get("team") or {}).get("id")
            if label_team == team_id:
                return str(node["id"])
            if label_team is None and workspace is None:
                workspace = str(node["id"])
        return workspace

    def _paginate(
        self, query: str, variables: dict[str, Any], field: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Walk ``pageInfo`` forward collecting ``nodes`` (``first: 50`` pages) until exhausted or ``limit``."""
        nodes: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            first = PAGE_SIZE if limit is None else max(1, min(PAGE_SIZE, limit - len(nodes)))
            data = self._gql(query, {**variables, "first": first, "after": after})
            connection = data.get(field) or {}
            nodes.extend(connection.get("nodes") or [])
            page = connection.get("pageInfo") or {}
            after = page.get("endCursor")
            if not page.get("hasNextPage") or not after or (limit is not None and len(nodes) >= limit):
                return nodes

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=TIMEOUT_S)
        return self._http

    def _gql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """POST one operation; return ``data``. Maps auth problems to ``TrackerAuthError``, retries rate
        limits a bounded number of times, and turns every other failure into ``TrackerError``."""
        if not self._api_key:
            raise TrackerAuthError("LINEAR_API_KEY missing from the profile .env")
        headers = {"Authorization": self._api_key, "Content-Type": "application/json"}
        body = {"query": query, "variables": variables or {}}
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = self._client().post(LINEAR_API, json=body, headers=headers, timeout=TIMEOUT_S)
            except httpx.HTTPError as exc:
                raise TrackerError(f"Linear request failed: {type(exc).__name__}: {exc}") from exc
            payload = _json_or_none(response)
            errors = payload.get("errors") if isinstance(payload, dict) else None
            errors = errors if isinstance(errors, list) else []
            if response.status_code in (401, 403) or any(_is_auth_error(e) for e in errors):
                raise TrackerAuthError(f"Linear authentication failed: {_error_text(errors, response)}")
            if response.status_code == 429 or any(_is_rate_limited(e) for e in errors):
                if attempt >= MAX_RATE_LIMIT_RETRIES:
                    raise TrackerError(f"Linear rate limit exceeded after {attempt} retries")
                self._sleep(_retry_delay(response, attempt))
                continue
            if errors:
                raise TrackerError(f"Linear error: {_error_text(errors, response)}")
            if response.status_code != 200 or not isinstance(payload, dict):
                raise TrackerError(f"Linear HTTP {response.status_code}: {response.text[:200]}")
            data = payload.get("data")
            return data if isinstance(data, dict) else {}
        raise TrackerError("Linear request gave up")  # pragma: no cover - loop always returns/raises


# ------------------------------------------------------------------------------------- parsing


def _parse_issue(node: dict[str, Any]) -> Issue:
    state = node.get("state") or {}
    team = node.get("team") or {}
    project = node.get("project")
    blocked_by: list[IssueRef] = []
    seen: set[str] = set()
    # X is blocked by Y when Y "blocks" X: those relations hang off X.inverseRelations.
    for rel in _nodes(node.get("inverseRelations")):
        if str(rel.get("type", "")).lower() == "blocks" and rel.get("issue"):
            _add_ref(blocked_by, seen, rel["issue"])
    # Not in the public schema today, but honoured if it appears: X "blocked_by" Y.
    for rel in _nodes(node.get("relations")):
        if str(rel.get("type", "")).lower() in ("blocked_by", "blockedby") and rel.get("relatedIssue"):
            _add_ref(blocked_by, seen, rel["relatedIssue"])
    comments = sorted(
        (
            Comment(
                author=_comment_author(c),
                body=str(c.get("body") or ""),
                created_at=str(c.get("createdAt") or ""),
            )
            for c in _nodes(node.get("comments"))
        ),
        key=lambda c: c.created_at,
    )
    return Issue(
        id=str(node.get("id") or ""),
        key=str(node.get("identifier") or ""),
        title=str(node.get("title") or ""),
        description=str(node.get("description") or ""),
        url=str(node.get("url") or ""),
        labels=[str(lbl.get("name")) for lbl in _nodes(node.get("labels")) if lbl.get("name")],
        state_type=str(state.get("type") or ""),
        state_name=str(state.get("name") or ""),
        team_id=str(team.get("id") or ""),
        team_key=str(team.get("key") or ""),
        project_id=str(project["id"]) if project and project.get("id") else None,
        project_name=str(project["name"]) if project and project.get("name") else None,
        updated_at=str(node.get("updatedAt") or ""),
        branch_name=str(node.get("branchName") or ""),
        parent=_ref(node["parent"]) if node.get("parent") else None,
        children=[_ref(c) for c in _nodes(node.get("children"))],
        blocked_by=blocked_by,
        attachments=[
            Attachment(url=str(a.get("url") or ""), title=str(a.get("title") or ""), source_type=str(a.get("sourceType") or ""))
            for a in _nodes(node.get("attachments"))
            if a.get("url")
        ],
        comments=comments,
    )


def _nodes(connection: Any) -> list[dict[str, Any]]:
    if isinstance(connection, dict):
        nodes = connection.get("nodes")
        return [n for n in nodes if isinstance(n, dict)] if isinstance(nodes, list) else []
    return [n for n in connection if isinstance(n, dict)] if isinstance(connection, list) else []


def _ref(node: dict[str, Any]) -> IssueRef:
    return IssueRef(
        id=str(node.get("id") or ""),
        key=str(node.get("identifier") or ""),
        state_type=str((node.get("state") or {}).get("type") or ""),
        title=str(node.get("title") or ""),
    )


def _add_ref(refs: list[IssueRef], seen: set[str], node: dict[str, Any]) -> None:
    ref = _ref(node)
    if ref.id and ref.id not in seen:
        seen.add(ref.id)
        refs.append(ref)


def _comment_author(comment: dict[str, Any]) -> str:
    user = comment.get("user") or {}
    bot = comment.get("botActor") or {}
    return str(user.get("displayName") or user.get("name") or bot.get("name") or "unknown")


# -------------------------------------------------------------------------------------- errors


def _json_or_none(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _ext(error: Any) -> dict[str, Any]:
    ext = error.get("extensions") if isinstance(error, dict) else None
    return ext if isinstance(ext, dict) else {}


def _is_auth_error(error: Any) -> bool:
    ext = _ext(error)
    type_ = str(ext.get("type") or "").lower().replace("_", " ")
    code = str(ext.get("code") or "").upper()
    http = ext.get("http") if isinstance(ext.get("http"), dict) else {}
    status = ext.get("statusCode") or http.get("status")
    return (
        type_ in ("authentication error", "forbidden")
        or code in ("AUTHENTICATION_ERROR", "FORBIDDEN", "UNAUTHENTICATED")
        or status in (401, 403)
    )


def _is_rate_limited(error: Any) -> bool:
    ext = _ext(error)
    return str(ext.get("code") or "").upper() == "RATELIMITED" or str(ext.get("type") or "").lower() == "ratelimited"


def _error_text(errors: list[Any], response: httpx.Response) -> str:
    messages = []
    for error in errors:
        if isinstance(error, dict):
            ext = _ext(error)
            messages.append(str(ext.get("userPresentableMessage") or error.get("message") or "unknown error"))
    if messages:
        return "; ".join(messages)
    return f"HTTP {response.status_code}: {response.text[:200]}"


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("Retry-After")
    if header:
        try:
            return max(0.0, min(float(header), MAX_RETRY_AFTER_S))
        except ValueError:
            pass
    return RATE_LIMIT_BACKOFF_S[min(attempt, len(RATE_LIMIT_BACKOFF_S) - 1)]


# ------------------------------------------------------------------------------------- context


def render_context(issue: Issue) -> str:
    """Markdown context for a worker prompt: body, parent, children, blockers, links, recent comments."""
    lines = [f"# {issue.key}: {issue.title}", ""]
    if issue.url:
        lines.append(f"- URL: {issue.url}")
    state = issue.state_name or issue.state_type
    if state:
        lines.append(f"- State: {state}" + (f" ({issue.state_type})" if issue.state_name and issue.state_type else ""))
    if issue.team_key:
        lines.append(f"- Team: {issue.team_key}")
    if issue.project_name:
        lines.append(f"- Project: {issue.project_name}")
    if issue.branch_name:
        lines.append(f"- Branch: `{issue.branch_name}`")
    lines.append("- Labels: " + (", ".join(f"`{lbl}`" for lbl in issue.labels) if issue.labels else "(none)"))
    lines += ["", "## Description", "", issue.description.strip() or "_(empty)_"]
    if issue.parent:
        lines += ["", "## Parent", "", f"- {_ref_line(issue.parent)}"]
    if issue.children:
        lines += ["", "## Sub-issues", ""] + [f"- {_ref_line(c)}" for c in issue.children]
    if issue.blocked_by:
        lines += ["", "## Blocked by", ""] + [f"- {_ref_line(b)}" for b in issue.blocked_by]
    if issue.attachments:
        lines += ["", "## Attachments", ""]
        for att in issue.attachments:
            label = att.title or att.url
            suffix = f" ({att.source_type})" if att.source_type else ""
            lines.append(f"- [{label}]({att.url}){suffix}")
    if issue.comments:
        recent = issue.comments[-CONTEXT_COMMENTS:]
        header = f"## Comments (last {len(recent)} of {len(issue.comments)})" if len(issue.comments) > len(recent) else "## Comments"
        lines += ["", header]
        for comment in recent:
            body = comment.body.strip()
            if len(body) > CONTEXT_COMMENT_CHARS:
                body = body[: CONTEXT_COMMENT_CHARS - 1].rstrip() + "…"
            lines += ["", f"**{comment.author}** — {comment.created_at}", "", body]
    return "\n".join(lines).rstrip() + "\n"


def _ref_line(ref: IssueRef) -> str:
    state = f" [{ref.state_type}]" if ref.state_type else ""
    title = f" {ref.title}" if ref.title else ""
    return f"{ref.key}{state}{title}".strip()
