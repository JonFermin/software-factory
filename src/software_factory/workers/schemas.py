"""Typed worker outputs.

Workers have no outward-write capability. They edit, test and commit inside a throwaway worktree and
return one of these models as structured output; the engine validates it and applies deterministic
policy before performing any outward write (see ``policy.py`` / ``effects.py``).

Every model here is converted to JSON Schema by ``runner.py`` (``model_json_schema()`` with ``$ref``
inlined) and passed as ``output_format={"type": "json_schema", "schema": ...}``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

WorkerStatus = Literal["done", "nothing", "blocked"]
"""done = work completed (possibly committed); nothing = nothing to do (stale/already fixed);
blocked = could not complete; ``summary``/``push_reasons`` say why. The engine treats blocked as a
failed attempt and flags it."""


class VerificationRun(BaseModel):
    command: str
    exit_code: int
    passed: bool
    duration_s: float | None = None
    notes: str = ""


class AcProof(BaseModel):
    """One acceptance criterion and the test/command that proves it."""

    criterion: str
    proof: str = ""  # test name, file::test, or command; empty when unproven
    status: Literal["proven", "unproven", "not_applicable"] = "unproven"


class Screenshot(BaseModel):
    path: str  # relative to the worktree root, under .factory/evidence/
    caption: str = ""
    phase: Literal["before", "after"] = "after"
    kind: Literal["png", "gif"] = "png"


class WorkerEvidence(BaseModel):
    ac_proofs: list[AcProof] = Field(default_factory=list)
    screenshots: list[Screenshot] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)


class CodeWorkerOutput(BaseModel):
    """Common shape for every worker that may produce commits."""

    status: WorkerStatus
    summary: str
    push_ok: bool = False
    push_reasons: list[str] = Field(
        default_factory=list,
        description="Why push is (or is not) safe. Required when push_ok is false.",
    )
    verification: list[VerificationRun] = Field(default_factory=list)
    evidence: WorkerEvidence = Field(default_factory=WorkerEvidence)


# ---------------------------------------------------------------------------------------- ci
CiClassification = Literal["caused_by_pr", "flaky", "inherited", "infra", "unknown"]


class CiFailure(BaseModel):
    check: str
    classification: CiClassification
    cause: str
    fixed: bool = False


class CiOutput(CodeWorkerOutput):
    head_verified: bool = Field(description="Worktree HEAD matched the PR headRefOid before work began.")
    failures: list[CiFailure] = Field(default_factory=list)


# ---------------------------------------------------------------------------------- conflict
HunkClass = Literal["trivial", "mechanical", "semantic", "unsafe"]


class HunkResolution(BaseModel):
    file: str
    classification: HunkClass
    resolution: str


class ConflictOutput(CodeWorkerOutput):
    head_verified: bool
    hunks: list[HunkResolution] = Field(default_factory=list)
    all_resolved: bool = False


# ---------------------------------------------------------------------------------------- rc
Disposition = Literal["addressed", "skipped", "not_actionable", "disagree", "already_done"]


class CommentDisposition(BaseModel):
    comment_id: str
    author: str
    disposition: Disposition
    note: str = ""


class RcOutput(CodeWorkerOutput):
    head_verified: bool
    comments: list[CommentDisposition] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------- rv
class ReviewFinding(BaseModel):
    severity: Literal["blocker", "major", "minor", "nit", "question"]
    file: str | None = None
    line: int | None = None
    claim: str
    evidence: str = ""


class RvOutput(BaseModel):
    status: WorkerStatus
    summary: str
    head_verified: bool
    verdict: Literal["approve", "comment", "request_changes"]
    findings: list[ReviewFinding] = Field(default_factory=list)
    review_markdown: str = Field(description="Full review draft. Written locally, never posted.")


# ------------------------------------------------------------------------------------- build
class BuildOutput(CodeWorkerOutput):
    pr_title: str
    pr_summary: str = Field(description="Markdown body summary (engine adds Fixes/evidence/footer).")
    tests_added: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------------- ready
class ReadyOutput(CodeWorkerOutput):
    pr_title: str = "Improve factory readiness"
    changes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------------ refine
class RefineOutput(BaseModel):
    status: WorkerStatus
    title: str
    description: str = Field(description="Full rewritten markdown description, including an AC list.")
    acceptance_criteria: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    suggested_split: list[str] = Field(default_factory=list)
    change_summary: str = ""


# -------------------------------------------------------------------------------------- plan
class SubIssueSpec(BaseModel):
    title: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    repo: str = Field(description="GitHub slug owner/name of the single repo this sub-issue touches.")
    blocked_by: list[int] = Field(
        default_factory=list, description="0-based indices into sub_issues that must finish first."
    )


class PlanOutput(BaseModel):
    status: WorkerStatus
    summary: str
    sub_issues: list[SubIssueSpec] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------- repo resolution
class RepoPick(BaseModel):
    repo: str | None = Field(description="GitHub slug owner/name, or null if no single repo fits.")
    confidence: float = Field(ge=0.0, le=1.0)
    multi_repo: bool = False
    reasoning: str = ""


OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "ci": CiOutput,
    "conflict": ConflictOutput,
    "rc": RcOutput,
    "rv": RvOutput,
    "build": BuildOutput,
    "ready": ReadyOutput,
    "refine": RefineOutput,
    "plan": PlanOutput,
    "resolve": RepoPick,
}
