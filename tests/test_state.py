"""state.py: SQLite dedupe/attempts, flags, PRs/pushes, caches, migrations, thread-safety."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pytest

from software_factory import state as state_mod
from software_factory.state import State, now_iso

T = "JaneDoe/game#12"


# ---------------------------------------------------------------------------------- basics


def test_creates_db_with_wal_and_schema_version(tmp_path):
    db_path = tmp_path / "nested dir" / "deeper" / "state.db"
    db = State(db_path)
    try:
        assert db_path.is_file()
        assert db.schema_version == len(state_mod.MIGRATIONS) >= 1
        with db._lock:
            assert db._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        db.close()


def test_state_persists_across_reopen(tmp_path):
    path = tmp_path / "state.db"
    with State(path) as db:
        db.mark_handled(T, "ci", "sha1", "done")
    with State(path) as db:
        assert db.is_handled(T, "ci", "sha1")
        assert db.schema_version == len(state_mod.MIGRATIONS)


def test_now_iso_is_utc_iso8601():
    ts = now_iso()
    assert ts.endswith("+00:00")
    assert datetime.fromisoformat(ts).tzinfo is not None


# ------------------------------------------------------------------------ handled / attempts


def test_handled_is_keyed_by_target_kind_version(state):
    assert not state.is_handled(T, "ci", "v1")
    state.mark_handled(T, "ci", "v1", "done", "fixed lint")
    assert state.is_handled(T, "ci", "v1")
    assert not state.is_handled(T, "ci", "v2")
    assert not state.is_handled(T, "rc", "v1")
    assert not state.is_handled("other#1", "ci", "v1")
    row = state.handled(T, "ci", "v1")
    assert row["outcome"] == "done" and row["note"] == "fixed lint" and row["ts"]


def test_mark_handled_upserts(state):
    state.mark_handled(T, "ci", "v1", "done")
    state.mark_handled(T, "ci", "v1", "nothing", "stale")
    assert state.handled(T, "ci", "v1")["outcome"] == "nothing"


def test_attempts_count_per_version(state):
    assert state.attempts(T, "ci", "v1") == 0
    assert state.record_attempt(T, "ci", "v1", "failed", "boom") == 1
    assert state.record_attempt(T, "ci", "v1", "blocked") == 2
    assert state.attempts(T, "ci", "v1") == 2
    assert state.attempts(T, "ci", "v2") == 0
    assert state.attempts(T, "conflict", "v1") == 0
    assert state.record_attempt(T, "ci", "v2", "failed") == 1


def test_last_attempt_is_the_newest_row_for_that_version(state):
    assert state.last_attempt(T, "ci", "v1") is None
    state.record_attempt(T, "ci", "v1", "failed", "boom")
    state.record_attempt(T, "ci", "v1", "kept", "push refused")
    state.record_attempt(T, "ci", "v2", "failed", "other version")
    last = state.last_attempt(T, "ci", "v1")
    assert last is not None and (last["outcome"], last["note"]) == ("kept", "push refused")
    assert last["ts"].endswith("+00:00") and state.last_attempt(T, "conflict", "v1") is None


def test_record_attempt_is_thread_safe(state):
    def worker() -> None:
        for _ in range(25):
            state.record_attempt(T, "ci", "v", "failed")
            state.cache_set("ns", "k", {"n": 1})
            state.attempts(T, "ci", "v")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state.attempts(T, "ci", "v") == 200


# ---------------------------------------------------------------------------- kept worktrees


def test_kept_worktrees_roundtrip(state, tmp_path):
    path = str(tmp_path / "wt" / "a b")
    assert state.kept_worktree(T) is None
    state.keep_worktree(T, "conflict", path, "conflict markers left")
    kept = state.kept_worktree(T)
    assert kept["path"] == path and kept["kind"] == "conflict" and kept["reason"] == "conflict markers left"
    assert kept["kept_at"]
    state.keep_worktree(T, "ci", path, "updated reason")  # upsert by target
    assert state.kept_worktree(T)["kind"] == "ci"
    state.keep_worktree("X#1", "rc", "p", "r")
    assert [k["target"] for k in state.kept_worktrees()] == [T, "X#1"]
    state.release_worktree(T)
    assert state.kept_worktree(T) is None
    state.release_worktree(T)  # idempotent
    assert len(state.kept_worktrees()) == 1


# ----------------------------------------------------------------------------------- flags


def test_flags_dedupe_only_while_open(state):
    first = state.add_flag(T, "ci", "auth", "gh auth failed", dedupe_key="auth:gh")
    assert isinstance(first, int)
    assert state.add_flag(T, "ci", "auth", "again", dedupe_key="auth:gh") is None
    assert state.add_flag("other", "rv", "auth", "", dedupe_key="auth:linear") is not None
    assert [f["id"] for f in state.flags()] == [first, first + 1]
    state.ack_flag(first)
    assert [f["id"] for f in state.flags()] == [first + 1]
    reopened = state.add_flag(T, "ci", "auth", "after ack", dedupe_key="auth:gh")
    assert reopened is not None and reopened != first
    all_flags = state.flags(open_only=False)
    assert len(all_flags) == 3
    acked = next(f for f in all_flags if f["id"] == first)
    assert acked["status"] == "acked" and acked["acked_at"]
    assert next(f for f in all_flags if f["id"] == reopened)["status"] == "open"


def test_flags_without_dedupe_key_never_dedupe(state):
    a = state.add_flag(T, "rc", "needs me", "comment skipped")
    b = state.add_flag(T, "rc", "needs me", "comment skipped")
    assert a is not None and b is not None and a != b
    flag = state.flags()[0]
    assert flag["target"] == T and flag["reason"] == "needs me" and flag["dedupe_key"] is None
    assert flag["created_at"] and flag["acked_at"] is None


def test_ack_unknown_or_acked_flag_is_noop(state):
    state.ack_flag(999)
    fid = state.add_flag(T, "ci", "r")
    state.ack_flag(fid)
    acked_at = state.flags(open_only=False)[0]["acked_at"]
    state.ack_flag(fid)
    assert state.flags(open_only=False)[0]["acked_at"] == acked_at


# --------------------------------------------------------------------------- PRs and pushes


def test_factory_prs_registry(state):
    assert not state.is_factory_pr("o/r", 1)
    state.register_pr("o/r", 1, "feat/abc-1", "build", issue_key="ABC-1", url="https://x/1")
    state.register_pr("o/r", 2, "factory/ready-2026", "ready")
    assert state.is_factory_pr("o/r", 1) and state.is_factory_pr("o/r", 2)
    assert not state.is_factory_pr("o/other", 1)
    prs = state.factory_prs()
    assert [(p["repo"], p["number"]) for p in prs] == [("o/r", 1), ("o/r", 2)]
    assert prs[0]["issue_key"] == "ABC-1" and prs[0]["url"] == "https://x/1" and prs[0]["kind"] == "build"
    assert prs[1]["issue_key"] is None and prs[1]["url"] == ""
    state.register_pr("o/r", 1, "feat/abc-1", "build")  # re-register keeps issue_key/url
    assert state.factory_prs()[0]["issue_key"] == "ABC-1" and state.factory_prs()[0]["url"] == "https://x/1"


def test_pushes_and_pushed_shas(state):
    assert state.pushed_shas("o/r", 7) == set()
    state.record_push("o/r", 7, "feat", "ci", "aaa", "ccc", ["bbb", "ccc"])
    state.record_push("o/r", 7, "feat", "rc", "ccc", "ddd", ["ddd"])
    state.record_push("o/r", 8, "feat2", "ci", "111", "222", ["222"])
    state.record_push("o/r", None, "factory/evidence", "build", "", "eee", ["eee"])
    assert state.pushed_shas("o/r", 7) == {"bbb", "ccc", "ddd"}
    assert state.pushed_shas("o/r", 8) == {"222"}
    rows = state.pushes("o/r", 7)
    assert [r["kind"] for r in rows] == ["ci", "rc"] and rows[0]["commit_shas"] == ["bbb", "ccc"]
    assert len(state.pushes()) == 4 and len(state.pushes("o/r")) == 4


def test_link_pushes_attaches_only_the_branchs_unlinked_rows(state):
    """build/ready push first and open the PR second: ``link_pushes`` attaches the branch's rows
    that have no PR yet, and leaves other branches, other repos and already-linked rows alone."""
    state.record_push("o/r", None, "factory/abc-1", "build", "", "s1", ["s1"])
    state.record_push("o/r", None, "factory/abc-1", "build", "s1", "s2", ["s2"])
    state.record_push("o/r", 3, "factory/abc-1", "ci", "s0", "s0b", ["s0b"])  # an older PR of that branch
    state.record_push("o/r", None, "factory/abc-2", "build", "", "t1", ["t1"])
    state.record_push("o/other", None, "factory/abc-1", "build", "", "u1", ["u1"])

    assert state.link_pushes("o/r", "factory/abc-1", 42) == 2

    assert state.pushed_shas("o/r", 42) == {"s1", "s2"}
    assert state.pushed_shas("o/r", 3) == {"s0b"}
    assert [p["number"] for p in state.pushes("o/r") if p["branch"] == "factory/abc-2"] == [None]
    assert [p["number"] for p in state.pushes("o/other")] == [None]
    assert state.link_pushes("o/r", "factory/abc-1", 43) == 0  # nothing unlinked is left
    assert state.link_pushes("o/r", "no-such-branch", 44) == 0


def test_touched_prs_is_union_of_opened_and_pushed(state):
    state.register_pr("o/r", 1, "b1", "build", issue_key="K-1")
    state.record_push("o/r", 1, "b1", "build", "", "s1", ["s1"])
    state.record_push("o/r", 1, "b1", "ci", "s1", "s2", ["s2"])
    state.record_push("o/r", 5, "feat", "ci", "x", "y", ["y"])
    state.record_push("o/r", None, "factory/evidence", "build", "", "z", ["z"])
    state.register_pr("o/q", 3, "b3", "build")

    touched = state.touched_prs()

    assert [(t["repo"], t["number"]) for t in touched] == [("o/q", 3), ("o/r", 1), ("o/r", 5)]
    by_key = {(t["repo"], t["number"]): t for t in touched}
    assert by_key[("o/r", 1)]["factory_opened"] is True and by_key[("o/r", 1)]["pushes"] == 2
    assert by_key[("o/r", 1)]["issue_key"] == "K-1"
    assert by_key[("o/r", 5)]["factory_opened"] is False and by_key[("o/r", 5)]["pushes"] == 1
    assert by_key[("o/q", 3)]["factory_opened"] is True and by_key[("o/q", 3)]["pushes"] == 0


# ------------------------------------------------------------------------------- resolution


def test_resolution_cache(state):
    assert state.get_resolution("iss-1") is None
    state.set_resolution("iss-1", "o/r", 0.95, "label")
    assert state.get_resolution("iss-1") == {
        "repo": "o/r",
        "confidence": 0.95,
        "source": "label",
        "resolved_at": state.get_resolution("iss-1")["resolved_at"],
    }
    state.set_resolution("iss-1", None, 0.2, "model")  # unresolved is cached too
    res = state.get_resolution("iss-1")
    assert res["repo"] is None and res["confidence"] == 0.2 and res["source"] == "model"


# ------------------------------------------------------------------------------ refinements


def test_refinement_lifecycle(state):
    assert state.get_refinement("ABC-1") is None
    state.save_refinement("id-1", "ABC-1", "Old title", "Old body", "New title", "New body")
    ref = state.get_refinement("ABC-1")
    assert ref["status"] == "pending" and ref["issue_id"] == "id-1"
    assert ref["original_title"] == "Old title" and ref["original_description"] == "Old body"
    assert ref["new_title"] == "New title" and ref["new_description"] == "New body"
    for status in ("notified", "edited", "applied", "reverted"):
        state.set_refinement_status("ABC-1", status)
        assert state.get_refinement("ABC-1")["status"] == status
    with pytest.raises(ValueError):
        state.set_refinement_status("ABC-1", "bogus")
    with pytest.raises(KeyError):
        state.set_refinement_status("NOPE-1", "notified")
    state.save_refinement("id-1", "ABC-1", "Old title", "Old body", "Newer", "Newer body")  # re-refine resets
    assert state.get_refinement("ABC-1")["status"] == "pending"
    assert state.get_refinement("ABC-1")["new_title"] == "Newer"


# ------------------------------------------------------------------------------------- runs


def test_runs_recorded_and_filtered_by_since(state):
    old = (datetime.now(UTC) - timedelta(days=10)).isoformat(timespec="seconds")
    state.record_run({"ts": old, "kind": "ci", "target": T, "version": "v1", "outcome": "done", "pushed": True, "cost_usd": 0.5})
    state.record_run({"kind": "rv", "target": "o/r#2", "version": "v", "outcome": "flagged", "usage": {"in": 1}})
    state.record_run({"ts": "2020-01-01T00:00:00Z", "kind": "rc", "target": "o/r#3", "outcome": "failed", "duration_ms": 12})

    runs = state.runs()
    assert [r["kind"] for r in runs] == ["ci", "rv", "rc"]
    assert runs[0]["pushed"] is True and runs[0]["cost_usd"] == 0.5
    assert runs[1]["usage"] == {"in": 1} and runs[1]["ts"]  # ts filled in when missing
    assert all("id" in r for r in runs)

    since = (datetime.now(UTC) - timedelta(days=7)).isoformat(timespec="seconds")
    assert [r["kind"] for r in state.runs(since)] == ["rv"]
    since_z = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert [r["kind"] for r in state.runs(since_z)] == ["rv"]
    assert [r["kind"] for r in state.runs("2019-12-31T00:00:00Z")] == ["ci", "rv", "rc"]


# ------------------------------------------------------------------------------ pr outcomes


def test_pr_outcomes_merge_fields(state):
    assert state.pr_outcomes() == []
    state.upsert_pr_outcome("o/r", 1, state="OPEN", created_at="2026-09-01T00:00:00Z")
    state.upsert_pr_outcome("o/r", 1, state="MERGED", merged_at="2026-09-03T00:00:00Z", rework_commits=2)
    state.upsert_pr_outcome("o/r", 2, state="CLOSED")
    outcomes = state.pr_outcomes()
    assert [(o["repo"], o["number"]) for o in outcomes] == [("o/r", 1), ("o/r", 2)]
    first = outcomes[0]
    assert first["state"] == "MERGED" and first["created_at"] == "2026-09-01T00:00:00Z"
    assert first["merged_at"] == "2026-09-03T00:00:00Z" and first["rework_commits"] == 2
    assert first["updated_at"]


# ------------------------------------------------------------------------------------ cache


def test_cache_stores_json(state):
    assert state.cache_get("discover", "o/r@sha") is None
    value = {"verify": ["npm test"], "dev_port": 5174, "nested": {"ok": True, "list": [1, 2.5, None]}}
    state.cache_set("discover", "o/r@sha", value)
    assert state.cache_get("discover", "o/r@sha") == value
    state.cache_set("discover", "o/r@sha", ["replaced"])
    assert state.cache_get("discover", "o/r@sha") == ["replaced"]
    state.cache_set("votes", "o/r@sha", 3)
    assert state.cache_get("votes", "o/r@sha") == 3  # namespaces are independent
    assert state.cache_get("discover", "o/r@sha") == ["replaced"]
    state.cache_delete("discover", "o/r@sha")
    assert state.cache_get("discover", "o/r@sha") is None
    state.cache_set("votes", "b", 1)
    state.cache_delete("votes")
    assert state.cache_get("votes", "o/r@sha") is None and state.cache_get("votes", "b") is None


def test_cache_set_falls_back_to_str_for_non_json(state):
    state.cache_set("ns", "k", {"when": datetime(2026, 1, 1, tzinfo=UTC)})
    assert state.cache_get("ns", "k") == {"when": "2026-01-01 00:00:00+00:00"}


# ------------------------------------------------------------------------------- migrations


def test_migration_hook_applies_new_versions_on_open(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with State(path) as db:
        db.mark_handled(T, "ci", "v1", "done")
        assert db.schema_version == 1

    def _v2_add_table(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE extra (id INTEGER PRIMARY KEY, note TEXT)")

    monkeypatch.setattr(state_mod, "MIGRATIONS", [*state_mod.MIGRATIONS, _v2_add_table])
    with State(path) as db:
        assert db.schema_version == 2
        assert db.is_handled(T, "ci", "v1")  # data survived
        with db._lock:
            db._conn.execute("INSERT INTO extra (note) VALUES ('x')")
            db._conn.commit()
    with State(path) as db:  # already at v2: migration not re-run (would fail on CREATE TABLE)
        assert db.schema_version == 2


def test_failed_migration_rolls_back_and_raises(tmp_path, monkeypatch):
    def _broken(conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("nope")

    monkeypatch.setattr(state_mod, "MIGRATIONS", [*state_mod.MIGRATIONS, _broken])
    with pytest.raises(sqlite3.OperationalError):
        State(tmp_path / "state.db")
    monkeypatch.setattr(state_mod, "MIGRATIONS", state_mod.MIGRATIONS[:-1])
    with State(tmp_path / "state.db") as db:
        assert db.schema_version == 1
