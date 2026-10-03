"""Shared fixtures: isolated runtime env, profile-dir builder, tmp State, real git repo helpers.

Everything here is generic so other test modules can reuse it. Nothing touches the developer's real
data dir, profiles dir, secrets or git identity.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from software_factory.config import Profile, load_profile
from software_factory.state import State

SECRET_ENV_KEYS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "LINEAR_API_KEY",
    "SLACK_BOT_TOKEN",
    "SLACK_USER_ID",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)

# ------------------------------------------------------------------------------------------ env


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets a private data root, no profiles-dir env, no inherited secrets, hermetic git."""
    data = tmp_path_factory.mktemp("data")
    monkeypatch.setenv("SOFTWARE_FACTORY_DATA_DIR", str(data))
    monkeypatch.delenv("SOFTWARE_FACTORY_PROFILES_DIR", raising=False)
    for key in SECRET_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(data / "no-global-gitconfig"))
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "factory-test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "factory-test@example.invalid")


# ------------------------------------------------------------------------------------- profiles


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # JSON escapes are valid TOML basic-string escapes
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot render {type(value).__name__} as TOML")


def _toml_key(key: str) -> str:
    return key if key.replace("-", "").replace("_", "").isalnum() else json.dumps(key)


def dict_to_toml(data: dict[str, Any], prefix: str = "") -> str:
    """Minimal dict -> TOML renderer (scalars, lists, nested tables). Enough for test profiles."""
    lines: list[str] = []
    tables: list[tuple[str, dict]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    out = "\n".join(lines)
    for key, table in tables:
        name = f"{prefix}.{_toml_key(key)}" if prefix else _toml_key(key)
        out += f"\n\n[{name}]\n" + dict_to_toml(table, name)
    return out.strip() + "\n"


def dict_to_dotenv(data: dict[str, str]) -> str:
    return "".join(f"{k}={v}\n" for k, v in data.items())


ProfileBuilder = Callable[..., Path]


@pytest.fixture
def profiles_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "profiles dir"  # space on purpose: paths with spaces must work
    directory.mkdir()
    return directory


@pytest.fixture
def make_profile(profiles_dir: Path) -> ProfileBuilder:
    """``make_profile(name, toml=<str | dict>, env=<str | dict>) -> profile dir``.

    A dict ``toml`` is rendered to TOML with ``profile.name`` filled in; a str is written verbatim.
    """

    def _make(
        name: str = "personal",
        toml: str | dict[str, Any] | None = None,
        env: str | dict[str, str] | None = None,
        *,
        root: Path | None = None,
    ) -> Path:
        pdir = (root or profiles_dir) / name
        pdir.mkdir(parents=True, exist_ok=True)
        if toml is None:
            toml = {}
        if isinstance(toml, dict):
            data = dict(toml)
            data.setdefault("profile", {})
            data["profile"] = {"name": name, **data["profile"]}
            text = dict_to_toml(data)
        else:
            text = toml
        (pdir / "factory.toml").write_text(text, encoding="utf-8")
        if env is not None:
            env_text = dict_to_dotenv(env) if isinstance(env, dict) else env
            (pdir / ".env").write_text(env_text, encoding="utf-8")
        return pdir

    return _make


@pytest.fixture
def load(profiles_dir: Path) -> Callable[[str], Profile]:
    return lambda name="personal": load_profile(name, profiles_dir)


@pytest.fixture
def profile(make_profile: ProfileBuilder, load: Callable[[str], Profile]) -> Profile:
    """A plain personal oauth profile with a token and Slack secrets."""
    make_profile(
        "personal",
        {"profile": {"mode": "full"}, "github": {"owners": ["JaneDoe"], "login": "JaneDoe"}},
        {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-test", "SLACK_BOT_TOKEN": "xoxb-test", "SLACK_USER_ID": "U123"},
    )
    return load("personal")


# ---------------------------------------------------------------------------------------- state


@pytest.fixture
def state(tmp_path: Path) -> Iterator[State]:
    db = State(tmp_path / "data dir" / "state.db")
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------------------------ git


def run_git(args: list[str], cwd: Path, *, check: bool = True, timeout: float = 30) -> str:
    """``git -C cwd <args>`` with list args, utf-8, explicit timeout; returns stripped stdout."""
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def init_bare_repo(path: Path, default_branch: str = "main") -> Path:
    """A local bare repo usable as ``origin`` (push/fetch without any network)."""
    path.mkdir(parents=True, exist_ok=True)
    run_git(["init", "-q", "--bare", "-b", default_branch, "."], path)
    return path


def init_git_repo(
    path: Path,
    *,
    origin: str | Path | None = None,
    default_branch: str = "main",
    files: dict[str, str] | None = None,
    set_origin_head: bool = True,
) -> Path:
    """A real git repo with one commit. ``origin`` may be a GitHub URL (ssh/https; the remote branch
    refs are faked locally) or a local bare repo path (really pushed to). Returns ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    run_git(["init", "-q", "-b", default_branch, "."], path)
    for name, content in (files or {"README.md": f"# {path.name}\n"}).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    run_git(["add", "-A"], path)
    run_git(["commit", "-q", "-m", "init"], path)
    if origin is not None:
        run_git(["remote", "add", "origin", str(origin)], path)
        if isinstance(origin, Path) or Path(str(origin)).exists():
            run_git(["push", "-q", "origin", default_branch], path)
            if set_origin_head:
                run_git(["remote", "set-head", "origin", default_branch], path)
        elif set_origin_head:
            run_git(["update-ref", f"refs/remotes/origin/{default_branch}", "HEAD"], path)
            run_git(["symbolic-ref", "refs/remotes/origin/HEAD", f"refs/remotes/origin/{default_branch}"], path)
    return path


@pytest.fixture
def git() -> Callable[..., str]:
    return run_git


@pytest.fixture
def repos_root(tmp_path: Path) -> Path:
    root = tmp_path / "repos root"
    root.mkdir()
    return root


@pytest.fixture
def git_repo_factory(repos_root: Path) -> Callable[..., Path]:
    """``git_repo_factory(dirname, origin="git@github.com:Owner/name.git", default_branch="main", files=None)``
    creates ``<repos_root>/<dirname>``."""

    def _make(
        dirname: str,
        origin: str | Path | None = "git@github.com:JaneDoe/repo.git",
        *,
        default_branch: str = "main",
        files: dict[str, str] | None = None,
        set_origin_head: bool = True,
        root: Path | None = None,
    ) -> Path:
        return init_git_repo(
            (root or repos_root) / dirname,
            origin=origin,
            default_branch=default_branch,
            files=files,
            set_origin_head=set_origin_head,
        )

    return _make


@pytest.fixture
def bare_repo_factory(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "origin", default_branch: str = "main") -> Path:
        return init_bare_repo(tmp_path / "bare" / f"{name}.git", default_branch)

    return _make
