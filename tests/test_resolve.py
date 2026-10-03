"""resolve.py: chain order (label > marker > links > siblings > text > picker), thresholds, multi-repo,
caching keyed by issue ``updated_at``; fakes for the tracker and the picker, real RepoIndex + State."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from software_factory.config import Profile, RepoOverride
from software_factory.repos import RepoIndex, RepoInfo
from software_factory.resolve import Resolution, Resolver
from software_factory.state import State
from software_factory.trackers.base import Attachment, Issue
from software_factory.workers.schemas import RepoPick

# -------------------------------------------------------------------------------------- fakes


class FakeTracker:
    name = "fake"

    def __init__(self, siblings: list[Issue] | None = None) -> None:
        self.siblings = list(siblings or [])
        self.calls: list[str] = []

    def sibling_issues(self, issue: Issue, limit: int = 50) -> list[Issue]:
        self.calls.append(issue.id)
        return list(self.siblings)[:limit]


class FakePicker:
    def __init__(self, pick: RepoPick | None = None) -> None:
        self.pick = pick or RepoPick(repo=None, confidence=0.0)
        self.calls: list[tuple[str, list[str]]] = []

    async def __call__(self, issue: Issue, candidates: list[RepoInfo]) -> RepoPick:
        self.calls.append((issue.id, [r.slug for r in candidates]))
        return self.pick


def repo(tmp: Path, slug: str, dirname: str | None = None, dirs: tuple[str, ...] = ()) -> RepoInfo:
    name = slug.split("/", 1)[1]
    path = tmp / (dirname or name)
    for sub in dirs:
        (path / sub).mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    return RepoInfo(slug=slug, name=name, dirname=dirname or name, path=path, default_branch="main", override=RepoOverride())


def issue(**kw) -> Issue:
    defaults = {"id": "iss-1", "key": "FAC-1", "title": "Plain title", "updated_at": "2026-09-30T10:00:00Z", "project_id": "proj-1"}
    return Issue(**{**defaults, **kw})


def gh_attachment(slug: str, kind: str = "pull", number: int = 1) -> Attachment:
    return Attachment(url=f"https://github.com/{slug}/{kind}/{number}", title=f"#{number}", source_type="githubPr")


@pytest.fixture
def repos(tmp_path: Path) -> RepoIndex:
    return RepoIndex(
        [
            repo(tmp_path, "JaneDoe/castle-sim", dirs=("src/core", "src/world")),
            repo(tmp_path, "JaneDoe/game", dirname="game-client", dirs=("src/scenes",)),
            repo(tmp_path, "JaneDoe/blog", dirs=("app/models",)),
        ]
    )


@pytest.fixture
def profile() -> Profile:
    return Profile(name="t", resolve={"min_confidence": 0.7, "sibling_min_votes": 2})


@pytest.fixture
def make_resolver(profile: Profile, repos: RepoIndex, state: State) -> Callable[..., Resolver]:
    def _make(tracker: FakeTracker | None = None, picker: FakePicker | None = None, prof: Profile | None = None) -> Resolver:
        return Resolver(prof or profile, tracker or FakeTracker(), repos, state, gh=None, picker=picker)

    return _make


CASTLE, GAME, BLOG = "JaneDoe/castle-sim", "JaneDoe/game", "JaneDoe/blog"


# ------------------------------------------------------------------------------ step 1: label


async def test_repo_label_wins_over_every_other_signal(make_resolver):
    tracker = FakeTracker([issue(id="s1", attachments=[gh_attachment(BLOG)]), issue(id="s2", attachments=[gh_attachment(BLOG)])])
    picker = FakePicker(RepoPick(repo=BLOG, confidence=0.99))
    target = issue(labels=["factory:build", "repo:Game-Client"], title="blog post", attachments=[gh_attachment(CASTLE)])

    result = await make_resolver(tracker, picker).resolve(target)

    assert result == Resolution(repo=GAME, confidence=1.0, source="label", candidates=[GAME])
    assert tracker.calls == [] and picker.calls == []


async def test_factory_repo_marker_in_description_counts_as_an_override(make_resolver):
    picker = FakePicker(RepoPick(repo=BLOG, confidence=0.99))
    target = issue(description="Plan item.\n\n<!-- factory:repo JaneDoe/castle-sim -->\n", attachments=[gh_attachment(BLOG)])
    result = await make_resolver(picker=picker).resolve(target)
    assert result.repo == CASTLE and result.confidence == 1.0 and result.source == "marker"
    assert picker.calls == []


async def test_override_naming_an_unmanaged_repo_is_unresolved_not_second_guessed(make_resolver):
    picker = FakePicker(RepoPick(repo=CASTLE, confidence=0.99))
    result = await make_resolver(picker=picker).resolve(issue(labels=["repo:unknown-thing"], attachments=[gh_attachment(CASTLE)]))
    assert result == Resolution(repo=None, confidence=0.0, source="label", candidates=["unknown-thing"])
    assert picker.calls == []


# ------------------------------------------------------------------------------ step 2: links


async def test_attachment_beats_siblings_text_and_picker(make_resolver):
    tracker = FakeTracker([issue(id="s1", labels=["repo:castle-sim"]), issue(id="s2", labels=["repo:castle-sim"])])
    picker = FakePicker(RepoPick(repo=BLOG, confidence=0.99))
    target = issue(title="Fix the blog layout", attachments=[gh_attachment(GAME, "issues", 4)])

    result = await make_resolver(tracker, picker).resolve(target)

    assert result == Resolution(repo=GAME, confidence=0.95, source="attachment", candidates=[GAME])
    assert tracker.calls == [] and picker.calls == []


async def test_github_links_in_description_count_like_attachments(make_resolver):
    target = issue(description="Continue https://github.com/JaneDoe/castle-sim/tree/main/src/core please")
    result = await make_resolver().resolve(target)
    assert result.repo == CASTLE and result.source == "attachment"


async def test_links_to_unmanaged_repos_are_ignored(make_resolver):
    target = issue(attachments=[gh_attachment("SomeoneElse/theirs")], description="https://github.com/Other/thing/pull/2")
    result = await make_resolver().resolve(target)
    assert result.repo is None and result.source == "none"


async def test_links_to_two_managed_repos_is_multi_repo_and_stops_the_chain(make_resolver):
    picker = FakePicker(RepoPick(repo=CASTLE, confidence=0.99))
    target = issue(attachments=[gh_attachment(CASTLE), gh_attachment(GAME, "blob", 0)], title="castle-sim only")
    result = await make_resolver(picker=picker).resolve(target)
    assert result.repo is None and result.multi_repo is True and result.source == "attachment"
    assert result.candidates == [CASTLE, GAME]
    assert picker.calls == []


# --------------------------------------------------------------------------- step 3: siblings


async def test_sibling_majority_vote_prefers_state_cache_then_labels_then_links(make_resolver, state: State):
    state.set_resolution("s1", CASTLE, 0.95, "attachment")  # cache wins over s1's own (conflicting) attachment
    siblings = [
        issue(id="s1", attachments=[gh_attachment(GAME)]),
        issue(id="s2", labels=["repo:castle-sim"]),
        issue(id="s3", attachments=[gh_attachment(GAME)]),
        issue(id="s4"),  # no signal: no vote
        issue(id="iss-1", labels=["repo:blog"]),  # the issue itself, if the tracker echoes it back: ignored
    ]
    tracker = FakeTracker(siblings)
    picker = FakePicker(RepoPick(repo=BLOG, confidence=0.99))

    result = await make_resolver(tracker, picker).resolve(issue())

    assert result.repo == CASTLE and result.source == "siblings"
    assert result.confidence == pytest.approx(0.6 + 0.3 * (2 / 3), abs=1e-3)
    assert result.candidates == [CASTLE, GAME]
    assert tracker.calls == ["iss-1"] and picker.calls == []


async def test_sibling_vote_needs_min_votes_and_a_clear_winner(make_resolver, profile: Profile):
    tie = FakeTracker([issue(id="s1", labels=["repo:castle-sim"]), issue(id="s2", labels=["repo:game"])])
    assert (await make_resolver(tie).resolve(issue())).source == "none"

    profile.resolve.sibling_min_votes = 3
    two_votes = FakeTracker([issue(id="s1", labels=["repo:castle-sim"]), issue(id="s2", labels=["repo:castle-sim"])])
    assert (await make_resolver(two_votes).resolve(issue(id="iss-2"))).source == "none"

    profile.resolve.sibling_min_votes = 1
    assert (await make_resolver(two_votes).resolve(issue(id="iss-3"))).repo == CASTLE


async def test_siblings_are_not_consulted_without_a_project(make_resolver):
    tracker = FakeTracker([issue(id="s1", labels=["repo:castle-sim"]), issue(id="s2", labels=["repo:castle-sim"])])
    result = await make_resolver(tracker).resolve(issue(project_id=None, team_id="team-1"))
    assert tracker.calls == [] and result.repo is None


# ------------------------------------------------------------------------------- step 4: text


async def test_text_matches_repo_dirname_as_a_whole_word(make_resolver, repos: RepoIndex, tmp_path: Path):
    result = await make_resolver().resolve(issue(title="Fix the game-client crash on spawn"))
    assert result == Resolution(repo=GAME, confidence=0.8, source="text", candidates=[GAME])
    # "client" glued to "game-" must not count for a repo named "client"; slugs match too.
    with_client = RepoIndex([*repos.all(), repo(tmp_path, "JaneDoe/client")])
    resolver = Resolver(Profile(name="t"), FakeTracker(), with_client, make_resolver().state, picker=None)
    assert (await resolver.resolve(issue(id="i2", title="game-client crash"))).repo == GAME
    assert (await resolver.resolve(issue(id="i3", title="see JaneDoe/client"))).repo == "JaneDoe/client"


async def test_text_matches_a_file_path_that_exists_in_exactly_one_repo(make_resolver):
    unique = await make_resolver().resolve(issue(description="Change `src/core/config.ts` so the gate scales."))
    assert unique.repo == CASTLE and unique.source == "text"
    # src/ exists in two repos; "src/main.ts" alone is ambiguous and must not resolve.
    shared = await make_resolver().resolve(issue(id="iss-2", description="Edit src/main.ts"))
    assert shared.repo is None and shared.source == "none"


async def test_text_is_case_insensitive_and_ignores_short_names(make_resolver, repos: RepoIndex, tmp_path: Path):
    assert (await make_resolver().resolve(issue(title="BLOG comments broken"))).repo == BLOG
    with_short = RepoIndex([*repos.all(), repo(tmp_path, "JaneDoe/pi")])
    resolver = Resolver(Profile(name="t"), FakeTracker(), with_short, make_resolver().state, picker=None)
    assert (await resolver.resolve(issue(id="i2", title="raise pi to 4"))).repo is None
    assert (await resolver.resolve(issue(id="i3", title="JaneDoe/pi is wrong"))).repo == "JaneDoe/pi"


async def test_ambiguous_text_falls_through_to_the_picker(make_resolver):
    picker = FakePicker(RepoPick(repo=CASTLE, confidence=0.9, reasoning="castle"))
    result = await make_resolver(picker=picker).resolve(issue(title="Share the blog widget with castle-sim"))
    assert result.repo == CASTLE and result.source == "picker" and result.confidence == 0.9
    assert result.candidates == [BLOG, CASTLE, GAME]
    assert picker.calls == [("iss-1", [BLOG, CASTLE, GAME])]


# ----------------------------------------------------------------------------- step 5: picker


async def test_picker_below_min_confidence_is_unresolved_but_reported(make_resolver):
    picker = FakePicker(RepoPick(repo=CASTLE, confidence=0.5))
    result = await make_resolver(picker=picker).resolve(issue())
    assert result.repo is None and result.confidence == 0.5 and result.source == "picker"


async def test_picker_multi_repo_or_unknown_repo_is_unresolved(make_resolver):
    multi = await make_resolver(picker=FakePicker(RepoPick(repo=None, confidence=0.9, multi_repo=True))).resolve(issue())
    assert multi.repo is None and multi.multi_repo is True and multi.source == "picker"
    unknown = await make_resolver(picker=FakePicker(RepoPick(repo="Nope/nothing", confidence=0.95))).resolve(issue(id="iss-2"))
    assert unknown.repo is None and unknown.source == "picker" and unknown.confidence == 0.95


async def test_picker_is_skipped_when_there_are_no_repos(profile: Profile, state: State):
    picker = FakePicker(RepoPick(repo=CASTLE, confidence=0.9))
    result = await Resolver(profile, FakeTracker(), RepoIndex([]), state, picker=picker).resolve(issue())
    assert result == Resolution(repo=None, confidence=0.0, source="none", candidates=[])
    assert picker.calls == []


async def test_nothing_matches_and_no_picker_is_unresolved_with_candidates(make_resolver):
    result = await make_resolver().resolve(issue())
    assert result == Resolution(repo=None, confidence=0.0, source="none", candidates=[BLOG, CASTLE, GAME])


async def test_below_threshold_step_falls_through_and_best_guess_is_kept(make_resolver, profile: Profile):
    profile.resolve.min_confidence = 0.9
    tracker = FakeTracker([issue(id="s1", labels=["repo:castle-sim"]), issue(id="s2", labels=["repo:castle-sim"])])
    # Siblings vote castle at 0.9 (full share) -> meets 0.9; lower the share so it does not.
    tracker.siblings.append(issue(id="s3", labels=["repo:game"]))  # share 2/3 -> 0.8 < 0.9
    picker = FakePicker(RepoPick(repo=GAME, confidence=0.95))

    result = await make_resolver(tracker, picker).resolve(issue())
    assert result.repo == GAME and result.source == "picker"  # picker got its turn
    assert picker.calls

    without_picker = await make_resolver(tracker).resolve(issue(id="iss-2"))
    assert without_picker.repo is None and without_picker.source == "siblings"
    assert without_picker.confidence == pytest.approx(0.8, abs=1e-3)


# ------------------------------------------------------------------------------------ cache


async def test_cache_hit_skips_all_work_until_the_issue_changes(make_resolver, state: State):
    tracker = FakeTracker([issue(id="s1", labels=["repo:castle-sim"]), issue(id="s2", labels=["repo:castle-sim"])])
    picker = FakePicker(RepoPick(repo=BLOG, confidence=0.99))
    resolver = make_resolver(tracker, picker)

    first = await resolver.resolve(issue())
    assert first.repo == CASTLE and tracker.calls == ["iss-1"]
    assert state.get_resolution("iss-1") == {"repo": CASTLE, "confidence": first.confidence, "source": "siblings", "resolved_at": state.get_resolution("iss-1")["resolved_at"]}

    tracker.siblings = []  # would resolve differently now
    again = await resolver.resolve(issue())
    assert again == first and tracker.calls == ["iss-1"] and picker.calls == []

    changed = await resolver.resolve(issue(updated_at="2026-10-01T00:00:00Z"))
    assert tracker.calls == ["iss-1", "iss-1"] and changed.source == "picker" and changed.repo == BLOG
    assert state.get_resolution("iss-1")["repo"] == BLOG


async def test_unresolved_results_are_cached_too(make_resolver, state: State):
    picker = FakePicker(RepoPick(repo=None, confidence=0.2))
    resolver = make_resolver(picker=picker)
    first = await resolver.resolve(issue())
    second = await resolver.resolve(issue())
    assert first.repo is None and second == first
    assert len(picker.calls) == 1
    row = state.get_resolution("iss-1")
    assert row["repo"] is None and row["source"] == "picker" and row["confidence"] == 0.2


async def test_stale_or_partial_cache_entries_are_recomputed(make_resolver, state: State):
    state.cache_set("resolve", "iss-1", {"updated_at": "2026-09-30T10:00:00Z", "resolution": {"repo": CASTLE, "confidence": 1.0, "source": "label"}})
    # No row in the resolutions table -> not trusted.
    assert (await make_resolver().resolve(issue())).source == "none"
    state.cache_set("resolve", "iss-1", {"updated_at": "2026-09-30T10:00:00Z", "resolution": "garbage"})
    state.set_resolution("iss-1", CASTLE, 1.0, "label")
    assert (await make_resolver().resolve(issue())).source == "none"


# ------------------------------------------------------------------------- planner resolutions


async def test_plan_resolution_survives_an_updated_at_change(make_resolver, state: State):
    """A child the planner created has its repo recorded with source "plan" (confidence 1.0).
    Labelling it ``factory:build`` edits the child (new ``updated_at``), which used to miss the
    cache and send it back through links / siblings / text / the picker -- here the picker would
    pick another repo."""
    state.set_resolution("iss-1", GAME, 1.0, "plan")
    state.cache_set("resolve", "iss-1", {"updated_at": "2026-09-30T10:00:00Z", "resolution": {"repo": GAME, "confidence": 1.0, "source": "plan", "candidates": [GAME]}})
    tracker = FakeTracker([issue(id="s1", labels=["repo:blog"]), issue(id="s2", labels=["repo:blog"])])
    picker = FakePicker(RepoPick(repo=BLOG, confidence=0.99))
    resolver = make_resolver(tracker, picker)
    child = issue(title="touches blog", labels=["factory:build"], updated_at="2026-10-01T08:00:00Z", attachments=[gh_attachment(CASTLE)])

    result = await resolver.resolve(child)

    assert result == Resolution(repo=GAME, confidence=1.0, source="plan", candidates=[GAME])
    assert tracker.calls == [] and picker.calls == []
    row = state.get_resolution("iss-1")
    assert row["repo"] == GAME and row["source"] == "plan"  # still the planner's: survives the next edit too
    again = await resolver.resolve(child.model_copy(update={"updated_at": "2026-10-02T08:00:00Z"}))
    assert again.repo == GAME and again.source == "plan" and picker.calls == []


async def test_plan_resolution_yields_to_a_label_and_to_an_unmanaged_repo(make_resolver, state: State):
    state.set_resolution("iss-1", GAME, 1.0, "plan")
    # an explicit repo:<name> label (the human's override) still wins
    labelled = await make_resolver().resolve(issue(labels=["repo:blog"], updated_at="2026-10-01T08:00:00Z"))
    assert labelled.repo == BLOG and labelled.source == "label"

    # a planner record for a repo the profile no longer manages is ignored: the chain runs
    state.set_resolution("iss-2", "JaneDoe/gone", 1.0, "plan")
    picker = FakePicker(RepoPick(repo=CASTLE, confidence=0.9))
    result = await make_resolver(picker=picker).resolve(issue(id="iss-2", updated_at="2026-10-01T08:00:00Z"))
    assert result.repo == CASTLE and result.source == "picker" and len(picker.calls) == 1

    # other sources keep the updated_at-keyed cache semantics
    state.set_resolution("iss-3", CASTLE, 0.8, "siblings")
    fresh = await make_resolver().resolve(issue(id="iss-3", updated_at="2026-10-01T08:00:00Z"))
    assert fresh.source == "none" and fresh.repo is None
