"""Runtime layout: every mutable file the factory owns lives outside the repo, per profile.

``data_root()/software-factory/<profile>/`` holds the SQLite state, worktrees, logs, reviews and
refinements for exactly one profile (invariant 6: profiles never cross). ``data_root()`` follows the
platform convention and may be overridden with ``SOFTWARE_FACTORY_DATA_DIR`` (tests, portable
installs).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path

APP_DIRNAME = "software-factory"
DATA_DIR_ENV = "SOFTWARE_FACTORY_DATA_DIR"


def data_root() -> Path:
    """Platform data directory: ``~/Library/Application Support`` on macOS, else ``$XDG_DATA_HOME``
    or ``~/.local/share`` (Windows included). ``SOFTWARE_FACTORY_DATA_DIR`` overrides all of them.

    Windows deliberately does not use ``%LOCALAPPDATA%``: packaged (MSIX) apps such as the Claude
    desktop app transparently redirect AppData writes into a per-package sandbox, so a tick started
    from such an app and one started by Task Scheduler or a terminal would see different state,
    logs and scheduler wrappers. The home directory is not virtualized."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg) if xdg else Path.home() / ".local" / "share"


def validate_profile_name(name: str) -> str:
    """Profile names are single path components (they become directory names)."""
    if not name or name in (".", "..") or any(sep in name for sep in ("/", "\\", "..")):
        raise ValueError(f"invalid profile name: {name!r}")
    return name


@dataclass(frozen=True)
class ProfilePaths:
    """All runtime paths of one profile, under ``data_root()/software-factory/<profile>/``."""

    root: Path
    state_db: Path  # state.db
    worktrees: Path  # wt/   (short name: Windows MAX_PATH)
    logs: Path  # logs/
    audit_log: Path  # logs/audit.jsonl
    runs_log: Path  # logs/runs.jsonl
    transcripts: Path  # logs/transcripts/
    reviews: Path  # reviews/<repo-name>-<pr>.md
    refinements: Path  # refinements/<KEY>.md
    lock_file: Path  # tick.lock
    scripts: Path  # scripts/ (scheduler wrapper scripts)

    @classmethod
    def for_profile(cls, name: str, root: Path | None = None) -> ProfilePaths:
        """Layout for ``name``. ``root`` replaces ``data_root()`` (the profile dir is still
        ``<root>/software-factory/<name>``)."""
        validate_profile_name(name)
        base = (root if root is not None else data_root()) / APP_DIRNAME / name
        logs = base / "logs"
        return cls(
            root=base,
            state_db=base / "state.db",
            worktrees=base / "wt",
            logs=logs,
            audit_log=logs / "audit.jsonl",
            runs_log=logs / "runs.jsonl",
            transcripts=logs / "transcripts",
            reviews=base / "reviews",
            refinements=base / "refinements",
            lock_file=base / "tick.lock",
            scripts=base / "scripts",
        )

    @property
    def directories(self) -> list[Path]:
        return [self.root, self.worktrees, self.logs, self.transcripts, self.reviews, self.refinements, self.scripts]

    def ensure(self) -> ProfilePaths:
        """``mkdir -p`` every directory (files are created lazily by their owners)."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def review_file(self, repo_name: str, pr: int) -> Path:
        return self.reviews / f"{repo_name}-{pr}.md"

    def refinement_file(self, key: str) -> Path:
        return self.refinements / f"{key}.md"

    def as_dict(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}
