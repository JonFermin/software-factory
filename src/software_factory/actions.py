"""Action model: one unit of work the engine may dispatch to one narrow worker.

An action is identified for dedupe purposes by ``(target, kind, version)``. The version key is
kind-specific (see ``docs/architecture.md`` -> "Action kinds") and changes whenever the external
state that triggered the action changes, so a handled action is re-dispatched only when there is
something new to do.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Kind(StrEnum):
    REFINE = "refine"
    PLAN = "plan"
    BUILD = "build"
    CI = "ci"
    CONFLICT = "conflict"
    RC = "rc"
    RV = "rv"
    READY = "ready"


# Kinds that operate on a pull request. Actions on the same PR share a branch, so they run
# sequentially in exactly this order.
PR_KINDS: tuple[Kind, ...] = (Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV)
PR_KIND_ORDER: dict[Kind, int] = {k: i for i, k in enumerate(PR_KINDS)}

# Kinds that operate on a tracker issue. Actions on the same issue run sequentially in exactly
# this order: a refine rewrites the text a plan reads, and a plan splits what a build would do.
ISSUE_KINDS: tuple[Kind, ...] = (Kind.REFINE, Kind.PLAN, Kind.BUILD)
ISSUE_KIND_ORDER: dict[Kind, int] = {k: i for i, k in enumerate(ISSUE_KINDS)}

# Position of a kind inside its group. A group never mixes PR and issue kinds (their targets
# differ), so the two orders share one table; manual kinds sort last.
KIND_ORDER: dict[Kind, int] = {**PR_KIND_ORDER, **ISSUE_KIND_ORDER}

# Kinds triggered only by a human command (never produced by a scan).
MANUAL_KINDS: tuple[Kind, ...] = (Kind.READY,)

# Kinds that may run for real in ``mode = "observe"``. Everything else is dry-run (recorded only).
OBSERVE_KINDS: tuple[Kind, ...] = (Kind.RV,)


class Action(BaseModel):
    """A dispatchable unit of work.

    Exactly one of (``repo`` + ``pr``), (``issue_id`` + ``issue_key``), or (``repo`` alone, for
    ``ready``) identifies the target.
    """

    kind: Kind
    version: str
    title: str = ""
    repo: str | None = None  # GitHub slug "owner/name"
    pr: int | None = None
    issue_id: str | None = None  # tracker-internal id (Linear UUID)
    issue_key: str | None = None  # human key, e.g. "ABC-123"
    url: str | None = None
    # Kind-specific scan data the handler needs (head/base SHAs, branch names, check names,
    # feedback ids...). Must be JSON-serialisable.
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def target(self) -> str:
        if self.pr is not None and self.repo:
            return f"{self.repo}#{self.pr}"
        if self.issue_key:
            return self.issue_key
        if self.repo:
            return self.repo
        raise ValueError("action has no target")

    @property
    def dedupe_key(self) -> tuple[str, str, str]:
        return (self.target, self.kind.value, self.version)

    @property
    def group(self) -> str:
        """Actions in the same group run sequentially (they share a branch / issue)."""
        return self.target

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.group, KIND_ORDER.get(self.kind, 100), self.kind.value)

    def describe(self) -> str:
        return f"{self.kind.value:<8} {self.target}  {self.title}".rstrip()


def order_group(actions: list[Action]) -> list[Action]:
    """Return one group's actions in execution order (``ci -> conflict -> rc -> rv`` on a pull
    request, ``refine -> plan -> build`` on an issue)."""
    return sorted(actions, key=lambda a: a.sort_key)


def group_actions(actions: list[Action]) -> dict[str, list[Action]]:
    """Group actions by target, each group ordered for sequential execution."""
    groups: dict[str, list[Action]] = {}
    for action in actions:
        groups.setdefault(action.group, []).append(action)
    return {g: order_group(acts) for g, acts in sorted(groups.items())}
