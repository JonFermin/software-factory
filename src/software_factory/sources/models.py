"""GitHub data models shared by scan, kinds, effects and metrics.

Populated by ``sources/github.py`` from ``gh`` JSON. Field names are snake_case versions of the
``gh --json`` / GraphQL fields they come from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CheckState = Literal["success", "failure", "pending", "skipped", "neutral", "cancelled", "unknown"]


class Check(BaseModel):
    name: str
    state: CheckState
    workflow: str = ""
    details_url: str = ""
    run_id: int | None = None  # GitHub Actions run id when the check is an Actions job
    job_id: int | None = None
    completed_at: str | None = None

    @property
    def failing(self) -> bool:
        return self.state in ("failure", "cancelled")


class Feedback(BaseModel):
    """One piece of reviewer feedback: a review body, an issue comment, or a review-thread comment."""

    id: str
    author: str
    author_is_bot: bool = False
    body: str
    created_at: str
    kind: Literal["review", "comment", "thread"]
    path: str | None = None
    line: int | None = None
    thread_id: str | None = None
    url: str = ""


class ReviewThread(BaseModel):
    id: str
    is_resolved: bool
    is_outdated: bool
    path: str | None = None
    line: int | None = None
    comments: list[Feedback] = Field(default_factory=list)


class PullRequest(BaseModel):
    repo: str  # owner/name
    number: int
    title: str = ""
    url: str = ""
    author: str = ""
    body: str = ""
    state: Literal["OPEN", "CLOSED", "MERGED"] = "OPEN"
    is_draft: bool = False
    head_ref_name: str = ""
    head_ref_oid: str = ""
    base_ref_name: str = ""
    base_ref_oid: str = ""
    head_repo: str | None = None  # owner/name of the head repo (differs for forks)
    mergeable: Literal["MERGEABLE", "CONFLICTING", "UNKNOWN"] = "UNKNOWN"
    created_at: str = ""
    updated_at: str = ""
    merged_at: str | None = None
    closed_at: str | None = None
    checks: list[Check] = Field(default_factory=list)
    feedback: list[Feedback] = Field(default_factory=list)
    threads: list[ReviewThread] = Field(default_factory=list)
    review_requested_from_me: bool = False

    @property
    def target(self) -> str:
        return f"{self.repo}#{self.number}"

    @property
    def failing_checks(self) -> list[Check]:
        return [c for c in self.checks if c.failing]

    @property
    def is_fork(self) -> bool:
        """GitHub slugs are case-insensitive: ``head_repo`` comes from GitHub's canonical casing
        while ``repo`` may come from a local clone's origin URL, so compare case-insensitively."""
        return bool(self.head_repo) and self.head_repo.lower() != (self.repo or "").lower()

    @property
    def unresolved_threads(self) -> list[ReviewThread]:
        return [t for t in self.threads if not t.is_resolved and not t.is_outdated]

    @property
    def resolved_thread_ids(self) -> set[str]:
        return {t.id for t in self.threads if t.is_resolved}

    @property
    def open_feedback(self) -> list[Feedback]:
        """``feedback`` minus the comments that sit on a resolved thread. ``feedback`` itself keeps
        them so that ``max(created_at)`` (the ``rc`` version key) never moves backwards when a
        reviewer resolves a thread; this is the list that can still call for work."""
        resolved = self.resolved_thread_ids
        return [f for f in self.feedback if f.thread_id is None or f.thread_id not in resolved]


class PrCommit(BaseModel):
    sha: str
    author_login: str | None = None
    author_email: str = ""
    message: str = ""
    committed_at: str = ""
    additions: int = 0
    deletions: int = 0
