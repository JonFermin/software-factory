"""paths.py: platform data root and per-profile layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from software_factory import paths
from software_factory.paths import ProfilePaths, data_root, validate_profile_name


def test_env_override_wins_everywhere(monkeypatch, tmp_path):
    monkeypatch.setenv("SOFTWARE_FACTORY_DATA_DIR", str(tmp_path / "my data"))
    for platform in ("win32", "darwin", "linux"):
        monkeypatch.setattr(paths.sys, "platform", platform)
        assert data_root() == tmp_path / "my data"


def test_windows_avoids_appdata_because_packaged_apps_virtualize_it(monkeypatch, tmp_path):
    # Found for real: from inside the Claude desktop app (MSIX), %LOCALAPPDATA% writes land in a
    # per-package sandbox that Task Scheduler and terminals never see. Home is not virtualized.
    monkeypatch.delenv("SOFTWARE_FACTORY_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert data_root() == Path.home() / ".local" / "share"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert data_root() == tmp_path / "xdg"


def test_macos_uses_application_support(monkeypatch):
    monkeypatch.delenv("SOFTWARE_FACTORY_DATA_DIR", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    assert data_root() == Path.home() / "Library" / "Application Support"


def test_linux_uses_xdg_or_local_share(monkeypatch, tmp_path):
    monkeypatch.delenv("SOFTWARE_FACTORY_DATA_DIR", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert data_root() == tmp_path / "xdg"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert data_root() == Path.home() / ".local" / "share"


def test_layout_under_data_root(tmp_path):
    p = ProfilePaths.for_profile("personal", root=tmp_path)
    base = tmp_path / "software-factory" / "personal"
    assert p.root == base
    assert p.state_db == base / "state.db"
    assert p.worktrees == base / "wt"
    assert p.logs == base / "logs"
    assert p.audit_log == base / "logs" / "audit.jsonl"
    assert p.runs_log == base / "logs" / "runs.jsonl"
    assert p.transcripts == base / "logs" / "transcripts"
    assert p.reviews == base / "reviews"
    assert p.refinements == base / "refinements"
    assert p.lock_file == base / "tick.lock"
    assert p.scripts == base / "scripts"
    assert p.review_file("game", 12) == base / "reviews" / "game-12.md"
    assert p.refinement_file("ABC-7") == base / "refinements" / "ABC-7.md"


def test_default_root_is_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SOFTWARE_FACTORY_DATA_DIR", str(tmp_path))
    assert ProfilePaths.for_profile("work").root == tmp_path / "software-factory" / "work"


def test_profiles_do_not_share_directories(tmp_path):
    a = ProfilePaths.for_profile("personal", root=tmp_path)
    b = ProfilePaths.for_profile("work", root=tmp_path)
    assert a.root != b.root
    assert not set(a.as_dict().values()) & set(b.as_dict().values())


def test_ensure_creates_every_directory_but_no_files(tmp_path):
    p = ProfilePaths.for_profile("personal", root=tmp_path / "with space").ensure()
    for directory in p.directories:
        assert directory.is_dir()
    assert not p.state_db.exists() and not p.audit_log.exists() and not p.lock_file.exists()
    assert p.ensure() is p  # idempotent


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "../x"])
def test_invalid_profile_names(name, tmp_path):
    with pytest.raises(ValueError):
        validate_profile_name(name)
    with pytest.raises(ValueError):
        ProfilePaths.for_profile(name, root=tmp_path)


def test_as_dict_lists_every_field(tmp_path):
    d = ProfilePaths.for_profile("personal", root=tmp_path).as_dict()
    assert set(d) == {
        "root", "state_db", "worktrees", "logs", "audit_log", "runs_log", "transcripts",
        "reviews", "refinements", "lock_file", "scripts",
    }  # fmt: skip
