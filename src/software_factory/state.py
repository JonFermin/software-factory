"""SQLite state, one file per profile.

A single connection (``check_same_thread=False``) guarded by a ``threading.Lock``: the dispatcher
calls into ``State`` from ``asyncio.to_thread`` workers. WAL mode keeps readers (``status``,
``flags``) from blocking a running tick. Schema versions live in ``PRAGMA user_version``; add a new
function to ``MIGRATIONS`` to evolve the schema (never edit an applied one). Timestamps are ISO-8601
UTC strings.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Migration = Callable[[sqlite3.Connection], None]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _norm_iso(value: str) -> str:
    """Normalise an ISO timestamp (``Z`` or offset) for string comparison against ``now_iso()``."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="seconds")


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _v1_initial(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS handled (
            target TEXT NOT NULL, kind TEXT NOT NULL, version TEXT NOT NULL,
            outcome TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', ts TEXT NOT NULL,
            PRIMARY KEY (target, kind, version)
        );
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target TEXT NOT NULL, kind TEXT NOT NULL, version TEXT NOT NULL,
            outcome TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', ts TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS attempts_key ON attempts (target, kind, version);
        CREATE TABLE IF NOT EXISTS kept_worktrees (
            target TEXT PRIMARY KEY, kind TEXT NOT NULL, path TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '', kept_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target TEXT NOT NULL, kind TEXT NOT NULL, reason TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '', dedupe_key TEXT,
            status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL, acked_at TEXT
        );
        CREATE INDEX IF NOT EXISTS flags_dedupe ON flags (dedupe_key, status);
        CREATE TABLE IF NOT EXISTS factory_prs (
            repo TEXT NOT NULL, number INTEGER NOT NULL, branch TEXT NOT NULL, kind TEXT NOT NULL,
            issue_key TEXT, url TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
            PRIMARY KEY (repo, number)
        );
        CREATE TABLE IF NOT EXISTS pushes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT NOT NULL, number INTEGER, branch TEXT NOT NULL, kind TEXT NOT NULL,
            before_sha TEXT NOT NULL, after_sha TEXT NOT NULL, commit_shas TEXT NOT NULL,
            ts TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS pushes_pr ON pushes (repo, number);
        CREATE TABLE IF NOT EXISTS resolutions (
            issue_id TEXT PRIMARY KEY, repo TEXT, confidence REAL NOT NULL,
            source TEXT NOT NULL, resolved_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS refinements (
            key TEXT PRIMARY KEY, issue_id TEXT NOT NULL,
            original_title TEXT NOT NULL, original_description TEXT NOT NULL,
            new_title TEXT NOT NULL, new_description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, kind TEXT, target TEXT, version TEXT, outcome TEXT,
            pushed INTEGER NOT NULL DEFAULT 0, cost_usd REAL, duration_ms INTEGER,
            data TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS runs_ts ON runs (ts);
        CREATE TABLE IF NOT EXISTS pr_outcomes (
            repo TEXT NOT NULL, number INTEGER NOT NULL, data TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY (repo, number)
        );
        CREATE TABLE IF NOT EXISTS cache (
            namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, ts TEXT NOT NULL,
            PRIMARY KEY (namespace, key)
        );
        """
    )


# Append new migrations here; index + 1 is the resulting ``PRAGMA user_version``.
MIGRATIONS: list[Migration] = [_v1_initial]

REFINEMENT_STATUSES = ("pending", "notified", "edited", "applied", "reverted")


class State:
    """All persistent engine state for one profile."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # ------------------------------------------------------------------------------ plumbing
    def _migrate(self) -> None:
        with self._tx() as conn:
            current = int(conn.execute("PRAGMA user_version").fetchone()[0])
            for index in range(current, len(MIGRATIONS)):
                MIGRATIONS[index](conn)
                conn.execute(f"PRAGMA user_version = {index + 1}")

    @property
    def schema_version(self) -> int:
        with self._lock:
            return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _row(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> State:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------- dedupe / attempts
    def is_handled(self, target: str, kind: str, version: str) -> bool:
        return (
            self._row(
                "SELECT 1 FROM handled WHERE target=? AND kind=? AND version=?", (target, kind, version)
            )
            is not None
        )

    def mark_handled(self, target: str, kind: str, version: str, outcome: str, note: str = "") -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO handled (target, kind, version, outcome, note, ts) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(target, kind, version) DO UPDATE SET outcome=excluded.outcome, "
                "note=excluded.note, ts=excluded.ts",
                (target, kind, version, outcome, note, now_iso()),
            )

    def handled(self, target: str, kind: str, version: str) -> dict | None:
        return self._row(
            "SELECT * FROM handled WHERE target=? AND kind=? AND version=?", (target, kind, version)
        )

    def attempts(self, target: str, kind: str, version: str) -> int:
        row = self._row(
            "SELECT COUNT(*) AS n FROM attempts WHERE target=? AND kind=? AND version=?",
            (target, kind, version),
        )
        return int(row["n"]) if row else 0

    def last_attempt(self, target: str, kind: str, version: str) -> dict | None:
        """The most recent failed attempt row for ``(target, kind, version)`` (outcome, note, ts)."""
        return self._row(
            "SELECT * FROM attempts WHERE target=? AND kind=? AND version=? ORDER BY id DESC LIMIT 1",
            (target, kind, version),
        )

    def record_attempt(self, target: str, kind: str, version: str, outcome: str, note: str = "") -> int:
        """Append one failed attempt; returns the new attempt count for ``(target, kind, version)``."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO attempts (target, kind, version, outcome, note, ts) VALUES (?,?,?,?,?,?)",
                (target, kind, version, outcome, note, now_iso()),
            )
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE target=? AND kind=? AND version=?",
                (target, kind, version),
            ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------- kept worktrees
    def keep_worktree(self, target: str, kind: str, path: str, reason: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO kept_worktrees (target, kind, path, reason, kept_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(target) DO UPDATE SET kind=excluded.kind, path=excluded.path, "
                "reason=excluded.reason, kept_at=excluded.kept_at",
                (target, kind, str(path), reason, now_iso()),
            )

    def kept_worktree(self, target: str) -> dict | None:
        return self._row("SELECT * FROM kept_worktrees WHERE target=?", (target,))

    def kept_worktrees(self) -> list[dict]:
        return self._rows("SELECT * FROM kept_worktrees ORDER BY kept_at, target")

    def release_worktree(self, target: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM kept_worktrees WHERE target=?", (target,))

    # ----------------------------------------------------------------------------- flags
    def add_flag(
        self, target: str, kind: str, reason: str, detail: str = "", dedupe_key: str | None = None
    ) -> int | None:
        """Insert a flag; returns its id, or ``None`` when an open flag with ``dedupe_key`` exists."""
        with self._tx() as conn:
            if dedupe_key is not None:
                existing = conn.execute(
                    "SELECT id FROM flags WHERE dedupe_key=? AND status='open' LIMIT 1", (dedupe_key,)
                ).fetchone()
                if existing is not None:
                    return None
            cursor = conn.execute(
                "INSERT INTO flags (target, kind, reason, detail, dedupe_key, status, created_at) "
                "VALUES (?,?,?,?,?,'open',?)",
                (target, kind, reason, detail, dedupe_key, now_iso()),
            )
            return int(cursor.lastrowid)

    def flags(self, open_only: bool = True) -> list[dict]:
        if open_only:
            return self._rows("SELECT * FROM flags WHERE status='open' ORDER BY id")
        return self._rows("SELECT * FROM flags ORDER BY id")

    def ack_flag(self, flag_id: int) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE flags SET status='acked', acked_at=? WHERE id=? AND status='open'",
                (now_iso(), flag_id),
            )

    # ------------------------------------------------------------- factory PRs and pushes
    def register_pr(
        self, repo: str, number: int, branch: str, kind: str, issue_key: str | None = None, url: str = ""
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO factory_prs (repo, number, branch, kind, issue_key, url, created_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(repo, number) DO UPDATE SET branch=excluded.branch, "
                "kind=excluded.kind, issue_key=COALESCE(excluded.issue_key, factory_prs.issue_key), "
                "url=CASE WHEN excluded.url='' THEN factory_prs.url ELSE excluded.url END",
                (repo, number, branch, kind, issue_key, url, now_iso()),
            )

    def is_factory_pr(self, repo: str, number: int) -> bool:
        return self._row("SELECT 1 FROM factory_prs WHERE repo=? AND number=?", (repo, number)) is not None

    def factory_prs(self) -> list[dict]:
        return self._rows("SELECT * FROM factory_prs ORDER BY created_at, repo, number")

    def record_push(
        self,
        repo: str,
        number: int | None,
        branch: str,
        kind: str,
        before_sha: str,
        after_sha: str,
        commit_shas: list[str],
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO pushes (repo, number, branch, kind, before_sha, after_sha, commit_shas, ts) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (repo, number, branch, kind, before_sha, after_sha, _dumps(list(commit_shas)), now_iso()),
            )

    def pushes(self, repo: str | None = None, number: int | None = None) -> list[dict]:
        sql, params = "SELECT * FROM pushes", []
        clauses = []
        if repo is not None:
            clauses.append("repo=?")
            params.append(repo)
        if number is not None:
            clauses.append("number=?")
            params.append(number)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        rows = self._rows(sql + " ORDER BY id", tuple(params))
        for row in rows:
            row["commit_shas"] = json.loads(row["commit_shas"])
        return rows

    def link_pushes(self, repo: str, branch: str, number: int) -> int:
        """Attach the ``pushes`` rows of ``(repo, branch)`` that have no PR yet (``number`` NULL:
        build/ready push the branch first and open the PR second) to ``number``; returns how many
        rows were linked. Rows already attached to a PR are left alone."""
        with self._tx() as conn:
            cursor = conn.execute(
                "UPDATE pushes SET number=? WHERE repo=? AND branch=? AND number IS NULL",
                (int(number), repo, branch),
            )
            return int(cursor.rowcount or 0)

    def pushed_shas(self, repo: str, number: int) -> set[str]:
        shas: set[str] = set()
        for row in self.pushes(repo, number):
            shas.update(row["commit_shas"])
            if row["after_sha"]:
                shas.add(row["after_sha"])
        return shas

    def touched_prs(self) -> list[dict]:
        """Every ``(repo, number)`` the factory opened or pushed to at least once."""
        rows = self._rows(
            """
            SELECT p.repo, p.number, 1 AS factory_opened, p.branch, p.kind, p.issue_key, p.url,
                   (SELECT COUNT(*) FROM pushes u WHERE u.repo=p.repo AND u.number=p.number) AS pushes
              FROM factory_prs p
            UNION
            SELECT u.repo, u.number, 0 AS factory_opened, MIN(u.branch), MIN(u.kind), NULL, '',
                   COUNT(*) AS pushes
              FROM pushes u
             WHERE u.number IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM factory_prs p WHERE p.repo=u.repo AND p.number=u.number)
             GROUP BY u.repo, u.number
             ORDER BY 1, 2
            """
        )
        for row in rows:
            row["factory_opened"] = bool(row["factory_opened"])
        return rows

    # --------------------------------------------------------------- repo resolution cache
    def get_resolution(self, issue_id: str) -> dict | None:
        return self._row(
            "SELECT repo, confidence, source, resolved_at FROM resolutions WHERE issue_id=?", (issue_id,)
        )

    def set_resolution(self, issue_id: str, repo: str | None, confidence: float, source: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO resolutions (issue_id, repo, confidence, source, resolved_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(issue_id) DO UPDATE SET repo=excluded.repo, confidence=excluded.confidence, "
                "source=excluded.source, resolved_at=excluded.resolved_at",
                (issue_id, repo, float(confidence), source, now_iso()),
            )

    # ------------------------------------------------------------------------ refinements
    def save_refinement(
        self,
        issue_id: str,
        key: str,
        original_title: str,
        original_description: str,
        new_title: str,
        new_description: str,
    ) -> None:
        """Persist the original + proposed text with ``status='pending'`` (before any notification/edit)."""
        ts = now_iso()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO refinements (key, issue_id, original_title, original_description, new_title, "
                "new_description, status, created_at, updated_at) VALUES (?,?,?,?,?,?,'pending',?,?) "
                "ON CONFLICT(key) DO UPDATE SET issue_id=excluded.issue_id, "
                "original_title=excluded.original_title, original_description=excluded.original_description, "
                "new_title=excluded.new_title, new_description=excluded.new_description, status='pending', "
                "updated_at=excluded.updated_at",
                (key, issue_id, original_title, original_description, new_title, new_description, ts, ts),
            )

    def set_refinement_status(self, key: str, status: str) -> None:
        if status not in REFINEMENT_STATUSES:
            raise ValueError(f"unknown refinement status {status!r}; expected one of {REFINEMENT_STATUSES}")
        with self._tx() as conn:
            cursor = conn.execute(
                "UPDATE refinements SET status=?, updated_at=? WHERE key=?", (status, now_iso(), key)
            )
            if cursor.rowcount == 0:
                raise KeyError(f"no refinement saved for {key!r}")

    def get_refinement(self, key: str) -> dict | None:
        return self._row("SELECT * FROM refinements WHERE key=?", (key,))

    # ------------------------------------------------------------------------------- runs
    def record_run(self, run: dict) -> None:
        data = dict(run)
        data.setdefault("ts", now_iso())
        cost = data.get("cost_usd")
        duration = data.get("duration_ms")
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO runs (ts, kind, target, version, outcome, pushed, cost_usd, duration_ms, data) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    _norm_iso(str(data["ts"])),
                    _str_or_none(data.get("kind")),
                    _str_or_none(data.get("target")),
                    _str_or_none(data.get("version")),
                    _str_or_none(data.get("outcome")),
                    1 if data.get("pushed") else 0,
                    float(cost) if cost is not None else None,
                    int(duration) if duration is not None else None,
                    _dumps(data),
                ),
            )

    def runs(self, since_iso: str | None = None) -> list[dict]:
        """Recorded runs (oldest first), optionally only those with ``ts >= since_iso``."""
        if since_iso:
            rows = self._rows("SELECT id, data FROM runs WHERE ts >= ? ORDER BY id", (_norm_iso(since_iso),))
        else:
            rows = self._rows("SELECT id, data FROM runs ORDER BY id")
        return [{"id": r["id"], **json.loads(r["data"])} for r in rows]

    # ------------------------------------------------------------------------ pr outcomes
    def upsert_pr_outcome(self, repo: str, number: int, **fields: Any) -> None:
        """Merge ``fields`` into the stored outcome record for ``(repo, number)``."""
        with self._tx() as conn:
            row = conn.execute("SELECT data FROM pr_outcomes WHERE repo=? AND number=?", (repo, number)).fetchone()
            data = json.loads(row["data"]) if row else {}
            data.update(fields)
            conn.execute(
                "INSERT INTO pr_outcomes (repo, number, data, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(repo, number) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                (repo, number, _dumps(data), now_iso()),
            )

    def pr_outcomes(self) -> list[dict]:
        rows = self._rows("SELECT repo, number, data, updated_at FROM pr_outcomes ORDER BY repo, number")
        return [
            {"repo": r["repo"], "number": r["number"], "updated_at": r["updated_at"], **json.loads(r["data"])}
            for r in rows
        ]

    # ------------------------------------------------------------------------------ cache
    def cache_get(self, namespace: str, key: str) -> Any | None:
        row = self._row("SELECT value FROM cache WHERE namespace=? AND key=?", (namespace, key))
        return json.loads(row["value"]) if row else None

    def cache_set(self, namespace: str, key: str, value: Any) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO cache (namespace, key, value, ts) VALUES (?,?,?,?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
                (namespace, key, _dumps(value), now_iso()),
            )

    def cache_delete(self, namespace: str, key: str | None = None) -> None:
        with self._tx() as conn:
            if key is None:
                conn.execute("DELETE FROM cache WHERE namespace=?", (namespace,))
            else:
                conn.execute("DELETE FROM cache WHERE namespace=? AND key=?", (namespace, key))


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)
