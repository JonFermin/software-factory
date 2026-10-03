"""Repo inference: which managed repo does a tracker issue belong to?

Chain (first *confident* hit wins; the result is cached in state per issue + ``updated_at``):

1. ``repo:<name>`` label, or the ``<!-- factory:repo owner/name -->`` marker the planner writes into
   descriptions (label creation is not an allowed write, so sub-issues may carry the marker instead).
   A manual override always ends the chain -- even when it names a repo the profile does not manage
   (that is reported as unresolved rather than second-guessed by later steps).
2. GitHub links on the issue: attachments plus ``github.com/<owner>/<name>/...`` URLs in the description.
3. Majority vote of the repos linked from sibling issues in the same project (state cache first, then
   their own labels/links); needs ``resolve.sibling_min_votes`` and a clear winner.
4. Repo names / dirnames / slugs or file paths in title + description, matched against ``RepoIndex``;
   only a unique match counts.
5. The ``picker`` (a model with structured ``RepoPick`` output) over every candidate repo.

Confidence: label/marker 1.0, attachment 0.95, siblings 0.6-0.9 (scaled by vote share), text 0.8,
picker = the model's own value. A step whose confidence is below ``resolve.min_confidence`` does not
win; the chain continues and, if nothing confident turns up, the best guess is returned with
``repo=None`` so the flag can show it. ``multi_repo`` ends the chain unresolved (the planner splits
such work). Unresolved results are cached too, so an undecidable issue does not re-run the picker
every tick until it changes.

A resolution the planner stored (``state.set_resolution(..., source="plan")``: the child's repo was
known when it was created) outlives the cache: it is honoured whatever the issue's ``updated_at``
(labelling the child ``factory:build`` edits it), second only to an explicit label / marker.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from .config import Profile
from .repos import RepoIndex, RepoInfo
from .state import State
from .trackers.base import Issue, Tracker
from .workers.schemas import RepoPick

Picker = Callable[[Issue, list[RepoInfo]], Awaitable[RepoPick]]
Step = Callable[[], Awaitable["Resolution | None"]]

CACHE_NAMESPACE = "resolve"
PLAN_SOURCE = "plan"  # what ``kinds/plan.py`` records for the children it creates
CONFIDENCE_LABEL = 1.0
CONFIDENCE_ATTACHMENT = 0.95
CONFIDENCE_SIBLINGS_MIN = 0.6
CONFIDENCE_SIBLINGS_MAX = 0.9
CONFIDENCE_TEXT = 0.8
MIN_NAME_LEN = 3  # repo names/dirnames shorter than this are too ambiguous for whole-word text matching

REPO_MARKER = re.compile(r"<!--\s*factory:repo\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s*-->", re.IGNORECASE)
GITHUB_URL = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?"
    r"(?=/(?:pull|issues|tree|blob|commit|compare|actions|releases)(?:[/?#]|$)|/?(?:[?#\s)>\]]|$))",
    re.IGNORECASE,
)
# A path-looking token: >=2 segments separated by "/", not part of a URL (no "://" or "/" before it).
FILE_PATH = re.compile(r"(?<![\w./:-])((?:[\w.-]+/)+[\w.-]+)")


class Resolution(BaseModel):
    repo: str | None  # GitHub slug owner/name of a managed repo, or None when unresolved
    confidence: float
    source: str  # label | marker | attachment | siblings | text | picker | none
    multi_repo: bool = False
    candidates: list[str] = Field(default_factory=list)


class Resolver:
    def __init__(
        self,
        profile: Profile,
        tracker: Tracker,
        repos: RepoIndex,
        state: State,
        gh: Any | None = None,
        picker: Picker | None = None,
    ) -> None:
        self.profile = profile
        self.tracker = tracker
        self.repos = repos
        self.state = state
        self.gh = gh  # reserved for PR-link lookups; unused by the current chain
        self.picker = picker

    # ---------------------------------------------------------------------------- public
    async def resolve(self, issue: Issue) -> Resolution:
        cached = self._cached(issue)
        if cached is not None:
            return cached
        resolution = await self._resolve_uncached(issue)
        self._store(issue, resolution)
        return resolution

    # ----------------------------------------------------------------------------- chain
    async def _resolve_uncached(self, issue: Issue) -> Resolution:
        override = self._from_label(issue)
        if override is not None:
            return override
        planned = self._from_plan(issue)
        if planned is not None:
            return planned
        best: Resolution | None = None
        for step in self._steps(issue):
            result = await step()
            if result is None:
                continue
            if result.multi_repo:
                return result
            if result.repo is not None and result.confidence >= self.profile.resolve.min_confidence:
                return result
            if best is None or result.confidence > best.confidence:
                best = result
        if best is not None:
            return best.model_copy(update={"repo": None})
        return Resolution(repo=None, confidence=0.0, source="none", candidates=self.repos.slugs())

    def _steps(self, issue: Issue) -> list[Step]:
        """Steps 2-5 as lazy callables: a step after an early win is never started."""
        steps: list[Step] = [partial(self._from_links, issue)]
        if issue.project_id:
            steps.append(partial(self._from_siblings, issue))
        steps.append(partial(self._from_text, issue))
        if self.picker is not None and len(self.repos):
            steps.append(partial(self._from_picker, issue))
        return steps

    def _from_label(self, issue: Issue) -> Resolution | None:
        """Step 1. An override naming an unmanaged repo is deliberately *unresolved*, not ignored."""
        prefix = self.profile.linear.repo_label_prefix
        name = issue.repo_label(prefix)
        source = "label"
        if name is None:
            match = REPO_MARKER.search(issue.description or "")
            if match is None:
                return None
            name, source = match.group(1), "marker"
        repo = self.repos.get(name)
        if repo is None:
            return Resolution(repo=None, confidence=0.0, source=source, candidates=[name])
        return Resolution(repo=repo.slug, confidence=CONFIDENCE_LABEL, source=source, candidates=[repo.slug])

    def _from_plan(self, issue: Issue) -> Resolution | None:
        """The planner's own record for a child it created (``source == "plan"``): the repo was
        chosen at creation, so later edits of the child (the human labelling it for build) must not
        send it back through links / siblings / text / the picker. A record naming a repo the
        profile no longer manages is ignored and the chain runs."""
        row = self.state.get_resolution(issue.id)
        if not row or row.get("source") != PLAN_SOURCE or not row.get("repo"):
            return None
        repo = self.repos.get(str(row["repo"]))
        if repo is None:
            return None
        confidence = float(row.get("confidence") or CONFIDENCE_LABEL)
        return Resolution(repo=repo.slug, confidence=confidence, source=PLAN_SOURCE, candidates=[repo.slug])

    async def _from_links(self, issue: Issue) -> Resolution | None:
        """Step 2. Links to two different managed repos mean the work spans repos."""
        slugs = self._linked_repos(issue)
        if not slugs:
            return None
        if len(slugs) > 1:
            return Resolution(
                repo=None, confidence=CONFIDENCE_ATTACHMENT, source="attachment", multi_repo=True, candidates=slugs
            )
        return Resolution(repo=slugs[0], confidence=CONFIDENCE_ATTACHMENT, source="attachment", candidates=slugs)

    async def _from_siblings(self, issue: Issue) -> Resolution | None:
        """Step 3. One vote per sibling that maps to exactly one managed repo."""
        siblings = await asyncio.to_thread(self.tracker.sibling_issues, issue)
        votes: Counter[str] = Counter()
        for sibling in siblings:
            if sibling.id == issue.id:
                continue
            repo = self._known_repo(sibling)
            if repo is not None:
                votes[repo] += 1
        if not votes:
            return None
        ranked = votes.most_common()
        winner, count = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if count < self.profile.resolve.sibling_min_votes or count <= runner_up:
            return None
        share = count / sum(votes.values())
        confidence = CONFIDENCE_SIBLINGS_MIN + (CONFIDENCE_SIBLINGS_MAX - CONFIDENCE_SIBLINGS_MIN) * share
        return Resolution(
            repo=winner, confidence=round(confidence, 3), source="siblings", candidates=[s for s, _ in ranked]
        )

    async def _from_text(self, issue: Issue) -> Resolution | None:
        """Step 4. Whole-word repo names/dirnames/slugs or file paths in title + description; unique only."""
        text = f"{issue.title}\n{issue.description or ''}"
        matched = set(self._repos_named_in(text)) | set(self._repos_with_paths_in(text))
        if len(matched) != 1:
            return None
        slug = next(iter(matched))
        return Resolution(repo=slug, confidence=CONFIDENCE_TEXT, source="text", candidates=[slug])

    async def _from_picker(self, issue: Issue) -> Resolution:
        """Step 5. The model's structured pick over every managed repo."""
        assert self.picker is not None
        candidates = self.repos.all()
        pick = await self.picker(issue, candidates)
        repo = self.repos.get(pick.repo) if pick.repo else None
        return Resolution(
            repo=repo.slug if repo is not None and not pick.multi_repo else None,
            confidence=float(pick.confidence),
            source="picker",
            multi_repo=pick.multi_repo,
            candidates=[r.slug for r in candidates],
        )

    # --------------------------------------------------------------------------- helpers
    def _known_repo(self, sibling: Issue) -> str | None:
        """A sibling's repo from the state cache, else its own label/marker/links (one repo only)."""
        cached = self.state.get_resolution(sibling.id)
        if cached and cached.get("repo"):
            repo = self.repos.get(str(cached["repo"]))
            return repo.slug if repo is not None else None
        direct = self._from_label(sibling)
        if direct is not None:
            return direct.repo
        linked = self._linked_repos(sibling)
        return linked[0] if len(linked) == 1 else None

    def _linked_repos(self, issue: Issue) -> list[str]:
        """Distinct managed repo slugs referenced by attachments or description URLs (first-seen order)."""
        texts = [att.url for att in issue.attachments] + [issue.description or ""]
        slugs: list[str] = []
        for text in texts:
            for owner, name in GITHUB_URL.findall(text):
                repo = self.repos.get(f"{owner}/{name}")
                if repo is not None and repo.slug not in slugs:
                    slugs.append(repo.slug)
        return slugs

    def _repos_named_in(self, text: str) -> list[str]:
        found: list[str] = []
        for repo in self.repos.all():
            keys = {repo.slug} | {k for k in (repo.name, repo.dirname) if len(k) >= MIN_NAME_LEN}
            if any(_whole_word(key).search(text) for key in keys) and repo.slug not in found:
                found.append(repo.slug)
        return found

    def _repos_with_paths_in(self, text: str) -> list[str]:
        """Each path-looking token votes for the single repo where its first two segments exist."""
        found: list[str] = []
        for token in FILE_PATH.findall(text):
            parts = [p for p in PurePosixPath(token).parts if p not in ("", ".", "..")]
            if len(parts) < 2:
                continue
            probe = PurePosixPath(*parts[:2])
            owners = [repo.slug for repo in self.repos.all() if (repo.path / probe).exists()]
            if len(owners) == 1 and owners[0] not in found:
                found.append(owners[0])
        return found

    # ----------------------------------------------------------------------------- cache
    def _cached(self, issue: Issue) -> Resolution | None:
        entry = self.state.cache_get(CACHE_NAMESPACE, issue.id)
        if not isinstance(entry, dict) or entry.get("updated_at") != issue.updated_at:
            return None
        if self.state.get_resolution(issue.id) is None:
            return None
        try:
            return Resolution.model_validate(entry.get("resolution"))
        except ValueError:
            return None

    def _store(self, issue: Issue, resolution: Resolution) -> None:
        self.state.set_resolution(issue.id, resolution.repo, resolution.confidence, resolution.source)
        self.state.cache_set(
            CACHE_NAMESPACE, issue.id, {"updated_at": issue.updated_at, "resolution": resolution.model_dump()}
        )


def _whole_word(key: str) -> re.Pattern[str]:
    """Case-insensitive match of ``key`` not glued to other name characters (``-`` and ``/`` count as
    name characters so ``game`` does not match inside ``game-client``)."""
    return re.compile(rf"(?<![\w/-]){re.escape(key)}(?![\w/-])", re.IGNORECASE)
