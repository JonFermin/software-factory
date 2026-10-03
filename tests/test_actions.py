"""actions.py: targets, dedupe keys and the execution order inside a group.

A group (one target) runs sequentially: ``ci -> conflict -> rc -> rv`` on a pull request and
``refine -> plan -> build`` on an issue (a refine rewrites the text a plan reads, a plan splits
what a build would do). Before the explicit issue order the kinds sorted alphabetically, which ran
``build`` first.
"""

from __future__ import annotations

import itertools

import pytest

from software_factory.actions import ISSUE_KINDS, PR_KINDS, Action, Kind, group_actions, order_group


def issue_action(kind: Kind, key: str = "ABC-1") -> Action:
    return Action(kind=kind, version="v", issue_id=f"id-{key}", issue_key=key)


def pr_action(kind: Kind, number: int = 7, repo: str = "o/r") -> Action:
    return Action(kind=kind, version="v", repo=repo, pr=number)


@pytest.mark.parametrize("perm", list(itertools.permutations(ISSUE_KINDS)))
def test_issue_group_runs_refine_then_plan_then_build_whatever_the_scan_order(perm: tuple[Kind, ...]) -> None:
    ordered = order_group([issue_action(k) for k in perm])
    assert [a.kind for a in ordered] == [Kind.REFINE, Kind.PLAN, Kind.BUILD]


@pytest.mark.parametrize("perm", list(itertools.permutations(PR_KINDS)))
def test_pr_group_keeps_ci_conflict_rc_rv(perm: tuple[Kind, ...]) -> None:
    ordered = order_group([pr_action(k) for k in perm])
    assert [a.kind for a in ordered] == [Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV]


def test_sort_key_positions_and_manual_kinds_last() -> None:
    assert [issue_action(k).sort_key[1] for k in ISSUE_KINDS] == [0, 1, 2]
    assert [pr_action(k).sort_key[1] for k in PR_KINDS] == [0, 1, 2, 3]
    ready = Action(kind=Kind.READY, version="sha", repo="o/r")
    assert ready.sort_key == ("o/r", 100, "ready") and ready.target == "o/r"


def test_group_actions_groups_by_target_and_orders_each_group() -> None:
    actions = [
        issue_action(Kind.BUILD, "ABC-2"),
        pr_action(Kind.RV, 7),
        issue_action(Kind.BUILD, "ABC-1"),
        pr_action(Kind.CI, 7),
        issue_action(Kind.PLAN, "ABC-1"),
        issue_action(Kind.REFINE, "ABC-1"),
        pr_action(Kind.CONFLICT, 8),
    ]
    groups = group_actions(actions)
    assert list(groups) == ["ABC-1", "ABC-2", "o/r#7", "o/r#8"]
    assert [a.kind for a in groups["ABC-1"]] == [Kind.REFINE, Kind.PLAN, Kind.BUILD]
    assert [a.kind for a in groups["ABC-2"]] == [Kind.BUILD]
    assert [a.kind for a in groups["o/r#7"]] == [Kind.CI, Kind.RV]
    assert [a.kind for a in groups["o/r#8"]] == [Kind.CONFLICT]


def test_targets_and_dedupe_keys() -> None:
    assert pr_action(Kind.CI).target == "o/r#7" and pr_action(Kind.CI).dedupe_key == ("o/r#7", "ci", "v")
    assert issue_action(Kind.BUILD).target == "ABC-1" and issue_action(Kind.BUILD).group == "ABC-1"
    with pytest.raises(ValueError):
        _ = Action(kind=Kind.CI, version="v").target
