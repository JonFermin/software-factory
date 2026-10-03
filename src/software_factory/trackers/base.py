"""Tracker + ContextSource protocols.

Linear is the only v1 implementation (``trackers/linear.py``); Jira/Confluence slot in later behind
the same protocols. Tracker *reads* are free; tracker *writes* are only ever called from
``effects.py`` (never from workers, never from scan).
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

StateType = Literal["triage", "backlog", "unstarted", "started", "completed", "canceled"]


class IssueRef(BaseModel):
    id: str
    key: str
    state_type: str = ""
    title: str = ""


class Attachment(BaseModel):
    url: str
    title: str = ""
    source_type: str = ""  # e.g. "github", "githubPr"


class Comment(BaseModel):
    author: str
    body: str
    created_at: str


class Issue(BaseModel):
    id: str
    key: str
    title: str
    description: str = ""
    url: str = ""
    labels: list[str] = Field(default_factory=list)
    state_type: str = ""
    state_name: str = ""
    team_id: str = ""
    team_key: str = ""
    project_id: str | None = None
    project_name: str | None = None
    updated_at: str = ""
    branch_name: str = ""
    parent: IssueRef | None = None
    children: list[IssueRef] = Field(default_factory=list)
    blocked_by: list[IssueRef] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)

    def has_label(self, name: str) -> bool:
        return any(lbl.lower() == name.lower() for lbl in self.labels)

    def repo_label(self, prefix: str = "repo:") -> str | None:
        for lbl in self.labels:
            if lbl.lower().startswith(prefix):
                return lbl[len(prefix) :].strip()
        return None

    @property
    def unfinished_blockers(self) -> list[IssueRef]:
        return [b for b in self.blocked_by if b.state_type not in ("completed", "canceled")]


class TrackerError(RuntimeError):
    pass


class TrackerAuthError(TrackerError):
    pass


@runtime_checkable
class Tracker(Protocol):
    name: str

    # ------------------------------------------------------------------ reads
    def ping(self) -> str:
        """Return the authenticated user's display name; raise TrackerAuthError on bad creds."""
        ...

    def list_issues_with_label(self, label: str) -> list[Issue]: ...

    def get_issue(self, id_or_key: str) -> Issue: ...

    def sibling_issues(self, issue: Issue, limit: int = 50) -> list[Issue]:
        """Other issues in the same project (or team when the issue has no project)."""
        ...

    def labels_exist(self, names: list[str]) -> dict[str, bool]: ...

    # ----------------------------------------------------------------- writes (effects.py only)
    def create_sub_issue(
        self, parent: Issue, title: str, description: str, labels: list[str]
    ) -> Issue: ...

    def add_blocked_by(self, issue_id: str, blocker_id: str) -> None: ...

    def move_to_state_type(self, issue_id: str, state_type: StateType) -> None: ...

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None: ...

    def remove_label(self, issue_id: str, label: str) -> None: ...

    def ensure_label(self, team_id: str, name: str) -> str:
        """Return the label id for ``name``. May only *look up*; creating labels is not an allowed
        write in v1, so raise TrackerError when missing."""
        ...


@runtime_checkable
class ContextSource(Protocol):
    def context_for(self, issue: Issue) -> str:
        """Markdown context for a worker prompt: body, relations, comments, parent, siblings."""
        ...
