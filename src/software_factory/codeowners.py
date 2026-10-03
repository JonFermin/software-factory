"""CODEOWNERS -> paths owned by someone else.

A path owned by someone other than me (not ``my_login``, not one of ``policy.my_teams``) counts as
sensitive: the push policy refuses to push diffs that touch it. CODEOWNERS patterns use gitignore
syntax (minus negation), which is exactly what ``pathspec``'s ``gitignore`` pattern style
(``gitwildmatch``) implements.

GitHub resolves ownership per path with *last match wins*: ``owners_for`` gives that exact answer
for one path and ``foreign_owned_paths`` applies it to a list of changed files, which is what the
push policy must use. ``foreign_owned_globs`` is the older pattern-level approximation (a later
rule for the *same pattern* replaces the earlier one, every foreign pattern becomes a glob): it
over-refuses whenever a catch-all rule (``* @team``) is narrowed later for me (``/docs/ @me``),
because ``*`` stays a sensitive glob although GitHub assigns ``docs/guide.md`` to me. Negated globs
cannot fix that: ``policy.matches_any`` uses gitignore semantics where ``!/docs/`` does not undo a
``*`` match, and a negation merged into the policy glob list could cancel a policy glob.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pathspec

CODEOWNERS_LOCATIONS: tuple[str, ...] = ("CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS")

_SPLIT = re.compile(r"(?<!\\)\s+")


def parse_codeowners(text: str) -> list[tuple[str, list[str]]]:
    """``[(pattern, owners)]`` in file order. Comments (``#``) and blank lines are dropped; a rule
    with no owners is kept (it un-owns the pattern). Escaped spaces in patterns are preserved."""
    rules: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
        parts = [p for p in _SPLIT.split(line) if p]
        if not parts:
            continue
        pattern, owners = parts[0], parts[1:]
        rules.append((pattern, owners))
    return rules


def find_codeowners(repo_path: Path) -> Path | None:
    """First of ``CODEOWNERS``, ``.github/CODEOWNERS``, ``docs/CODEOWNERS`` that exists."""
    for rel in CODEOWNERS_LOCATIONS:
        path = Path(repo_path) / rel
        if path.is_file():
            return path
    return None


def to_glob(pattern: str) -> str:
    """Normalise a CODEOWNERS pattern into a gitwildmatch glob (``\\ `` -> space)."""
    return pattern.replace("\\ ", " ")


def is_mine(owner: str, my_teams: list[str], my_login: str) -> bool:
    """Owner matches me: ``@login`` / ``login`` or ``@org/team`` / ``org/team`` (case-insensitive).
    Email owners are never mine (CODEOWNERS emails cannot be mapped to a login offline)."""
    name = _normalise(owner)
    if not name or "@" in name:
        return False
    if my_login and name == _normalise(my_login):
        return True
    return any(name == _normalise(team) for team in my_teams)


def is_foreign(owners: list[str], my_teams: list[str], my_login: str) -> bool:
    """Owned, and by nobody that counts as me. Unowned (``[]``) is not foreign."""
    return bool(owners) and not any(is_mine(o, my_teams, my_login) for o in owners)


def load_rules(repo_path: Path) -> list[tuple[str, list[str]]]:
    """The repo's parsed CODEOWNERS rules in file order; ``[]`` when it has no CODEOWNERS file."""
    path = find_codeowners(repo_path)
    if path is None:
        return []
    return parse_codeowners(path.read_text(encoding="utf-8", errors="replace"))


def foreign_owned_paths(repo_path: Path, paths: Iterable[str], my_teams: list[str], my_login: str) -> list[str]:
    """The subset of ``paths`` (repo-relative) that GitHub assigns to someone other than me, using
    the exact per-path *last match wins* rule. Use this for changed files; it is what
    ``foreign_owned_globs`` approximates. ``[]`` without a CODEOWNERS file."""
    rules = load_rules(repo_path)
    if not rules:
        return []
    return [
        path
        for path in paths
        if is_foreign(owners_for(path.replace("\\", "/").lstrip("/"), rules), my_teams, my_login)
    ]


def foreign_owned_globs(repo_path: Path, my_teams: list[str], my_login: str) -> list[str]:
    """Globs for every pattern whose owners are all foreign (not me, not my teams).

    Last match wins per pattern: a later rule for the same pattern replaces an earlier one, so a
    pattern first given to another team and later to me is not foreign (and vice versa). Unowned
    patterns are not foreign. Pattern-level only: see the module docstring for why a catch-all rule
    makes this over-refuse, and prefer ``foreign_owned_paths`` for a concrete list of files.
    """
    effective: dict[str, list[str]] = {}
    for pattern, owners in load_rules(repo_path):
        effective.pop(pattern, None)  # re-insert so order reflects the last occurrence
        effective[pattern] = owners
    return [to_glob(pattern) for pattern, owners in effective.items() if is_foreign(owners, my_teams, my_login)]


def owners_for(path: str, rules: list[tuple[str, list[str]]]) -> list[str]:
    """Owners of ``path`` under GitHub semantics: the last matching rule wins; ``[]`` when unowned."""
    owners: list[str] = []
    for pattern, rule_owners in rules:
        spec = pathspec.PathSpec.from_lines("gitignore", [to_glob(pattern)])
        if spec.match_file(path):
            owners = rule_owners
    return owners


def _normalise(owner: str) -> str:
    return owner.strip().lstrip("@").lower()
