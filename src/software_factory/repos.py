"""Managed repositories: local clones under ``repos.root`` with a GitHub origin.

Discovery is deterministic and cheap: each child directory with a ``.git`` (directory or worktree
file) is inspected by reading git's config files directly, falling back to ``git`` subprocesses when
the files cannot be parsed. Local directory names may differ from GitHub repo names (``game-client``
-> ``owner/game``), so ``RepoIndex.get`` accepts slug, name or dirname.
"""

from __future__ import annotations

import configparser
import os
import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .config import Profile, RepoOverride

GIT_TIMEOUT_S = 15.0
SUMMARY_MAX_CHARS = 1500
SUMMARY_FILES = ("CLAUDE.md", "AGENTS.md", "README.md")

# git@github.com:o/n(.git) | ssh://git@github.com/o/n | https://[user@]github.com/o/n(.git)[/] | git://...
_GITHUB_REMOTE = re.compile(
    r"^(?:"
    r"(?:ssh|git|https?)://(?:[^@/\s]+@)?github\.com(?::\d+)?/"  # scheme form
    r"|(?:[^@/\s]+@)?github\.com:"  # scp-like form
    r")([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


def parse_github_remote(url: str) -> str | None:
    """``git@github.com:o/n(.git)`` | ``https://github.com/o/n(.git)`` -> ``"o/n"``; else ``None``."""
    match = _GITHUB_REMOTE.match(url.strip())
    if not match:
        return None
    owner, name = match.group(1), match.group(2)
    if not owner or not name or name == ".git":
        return None
    return f"{owner}/{name}"


class RepoInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str  # owner/name from the origin remote
    name: str  # GitHub repo name
    dirname: str  # local directory name (may differ from name)
    path: Path  # local clone
    default_branch: str  # override, else origin/HEAD, else the one of origin/main|master that exists, else "main"
    override: RepoOverride

    @property
    def owner(self) -> str:
        return self.slug.split("/", 1)[0]

    def keys(self) -> set[str]:
        """Every lower-cased key this repo answers to: slug, name, dirname."""
        return {self.slug.lower(), self.name.lower(), self.dirname.lower()}


class RepoIndex:
    def __init__(self, repos: list[RepoInfo] | None = None) -> None:
        self._repos: list[RepoInfo] = sorted(repos or [], key=lambda r: r.slug.lower())
        self._by_key: dict[str, RepoInfo] = {}
        # Slugs win over names, names over dirnames, so a dirname that equals another repo's name
        # cannot shadow it.
        for attr in ("dirname", "name", "slug"):
            for repo in self._repos:
                self._by_key[getattr(repo, attr).lower()] = repo

    @classmethod
    def discover(cls, profile: Profile) -> RepoIndex:
        """Scan ``repos.root`` children that have a ``.git`` and a GitHub origin whose owner is in
        ``github.owners`` (case-insensitive) or that are listed in ``repos.include``; then apply
        ``repos.exclude`` (dirname, name or slug)."""
        cfg = profile.repos
        root = cfg.root_path
        if not root.is_dir():
            return cls([])
        owners = {o.lower() for o in profile.github.owners}
        include = {i.lower() for i in cfg.include}
        exclude = {e.lower() for e in cfg.exclude}
        found: list[RepoInfo] = []
        seen_slugs: set[str] = set()
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir() or not (child / ".git").exists():
                continue
            url = git_origin_url(child)
            slug = parse_github_remote(url) if url else None
            if slug is None:
                continue
            owner, name = slug.split("/", 1)
            keys = {slug.lower(), name.lower(), child.name.lower()}
            if not (owner.lower() in owners or keys & include):
                continue
            if keys & exclude or slug.lower() in seen_slugs:
                continue
            seen_slugs.add(slug.lower())
            override = cfg.override_for(slug, child.name)
            found.append(
                RepoInfo(
                    slug=slug,
                    name=name,
                    dirname=child.name,
                    path=child,
                    default_branch=override.default_branch or git_default_branch(child) or guess_default_branch(child),
                    override=override,
                )
            )
        return cls(found)

    def all(self) -> list[RepoInfo]:
        return list(self._repos)

    def get(self, key: str) -> RepoInfo | None:
        """Look up by slug, name or dirname (case-insensitive)."""
        return self._by_key.get(key.strip().lower())

    def slugs(self) -> list[str]:
        return [r.slug for r in self._repos]

    def names(self) -> list[str]:
        return [r.name for r in self._repos]

    def summary(self, slug: str) -> str:
        """First section of CLAUDE.md, else AGENTS.md, else README.md (<= 1500 chars); ``""`` if none."""
        repo = self.get(slug)
        if repo is None:
            return ""
        for filename in SUMMARY_FILES:
            path = repo.path / filename
            if path.is_file():
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                section = first_section(text)
                if section:
                    return section
        return ""

    def __len__(self) -> int:
        return len(self._repos)

    def __iter__(self):
        return iter(self._repos)


def first_section(text: str, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """Text up to the second markdown heading. A leading title with no body of its own (``# Name``
    followed directly by ``## Overview``) is extended to the next heading so the summary has content."""
    lines = text.splitlines()
    out: list[str] = []
    headings = 0
    body_lines = 0
    for line in lines:
        if re.match(r"^\s{0,3}#{1,6}\s", line):
            headings += 1
            if headings >= 2 and body_lines > 0:
                break
        elif line.strip():
            body_lines += 1
        out.append(line)
    section = "\n".join(out).strip()
    if len(section) > max_chars:
        section = section[: max_chars - 1].rstrip() + "…"
    return section


# ----------------------------------------------------------------------------------- git helpers


def git_origin_url(path: Path) -> str | None:
    """Origin URL from the repo's config file, else ``git remote get-url origin``."""
    dirs = git_dirs(path)
    if dirs is not None:
        url = _origin_from_config(dirs[1] / "config")
        if url:
            return url
    return run_git(["remote", "get-url", "origin"], path)


def git_default_branch(path: Path) -> str | None:
    """Branch behind ``refs/remotes/origin/HEAD`` (file first, then ``git symbolic-ref``); ``None`` if unset."""
    dirs = git_dirs(path)
    if dirs is not None:
        head = dirs[1] / "refs" / "remotes" / "origin" / "HEAD"
        try:
            content = head.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            content = ""
        if content.startswith("ref: refs/remotes/origin/"):
            return content[len("ref: refs/remotes/origin/") :].strip() or None
    out = run_git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], path)
    if out:
        return out.removeprefix("origin/").strip() or None
    return None


def guess_default_branch(path: Path) -> str:
    """Default branch when ``origin/HEAD`` is not set locally (common for repos created locally and
    pushed later): ``main`` if ``origin/main`` exists, else ``master`` if ``origin/master`` exists,
    else ``main``. ``repos.overrides.<name>.default_branch`` or ``git remote set-head origin --auto``
    settles it for good."""
    for candidate in ("main", "master"):
        if run_git(["rev-parse", "--verify", "-q", f"refs/remotes/origin/{candidate}"], path):
            return candidate
    return "main"


def git_dirs(path: Path) -> tuple[Path, Path] | None:
    """``(gitdir, commondir)`` for a checkout: handles ``.git`` directories and ``.git`` files
    (worktrees, submodules). ``None`` when ``path`` is not a git checkout."""
    dot_git = path / ".git"
    if dot_git.is_dir():
        gitdir = dot_git
    elif dot_git.is_file():
        try:
            line = dot_git.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        if not line.startswith("gitdir:"):
            return None
        gitdir = Path(line[len("gitdir:") :].strip())
        if not gitdir.is_absolute():
            gitdir = (path / gitdir).resolve()
    else:
        return None
    commondir = gitdir
    common_file = gitdir / "commondir"
    if common_file.is_file():
        try:
            rel = common_file.read_text(encoding="utf-8", errors="replace").strip()
            common = Path(rel)
            commondir = common if common.is_absolute() else (gitdir / common).resolve()
        except OSError:
            pass
    return gitdir, commondir


def _origin_from_config(config_path: Path) -> str | None:
    if not config_path.is_file():
        return None
    parser = configparser.ConfigParser(strict=False, interpolation=None, delimiters=("=",))
    try:
        parser.read_string(config_path.read_text(encoding="utf-8", errors="replace"))
    except (configparser.Error, OSError):
        return None
    for section in parser.sections():
        if section.strip().lower() == 'remote "origin"':
            url = parser.get(section, "url", fallback="")
            return url.strip() or None
    return None


def run_git(args: list[str], cwd: Path, timeout: float = GIT_TIMEOUT_S) -> str | None:
    """``git -C <cwd> <args>`` -> stripped stdout, or ``None`` on any failure. Never prompts."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
