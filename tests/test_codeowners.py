"""codeowners.py: parsing, lookup, last-match-wins, my_teams/my_login filtering, pathspec compat."""

from __future__ import annotations

from pathlib import Path

import pathspec

from software_factory.codeowners import (
    find_codeowners,
    foreign_owned_globs,
    foreign_owned_paths,
    is_foreign,
    is_mine,
    load_rules,
    owners_for,
    parse_codeowners,
    to_glob,
)

FIXTURE = Path(__file__).parent / "fixtures" / "repos" / "codeowners"


def test_parse_codeowners_rules_in_order_with_comments_and_blank_lines_dropped():
    rules = parse_codeowners((FIXTURE / ".github" / "CODEOWNERS").read_text(encoding="utf-8"))
    assert rules[0] == ("*", ["@acme/core-team"])
    assert rules[1] == ("/docs/", ["@me-login"])
    assert rules[2] == ("docs/*.md", ["@acme/docs-team", "@me-login"])
    assert ("src/billing/**", ["@acme/payments", "bob@example.com"]) in rules
    assert ("/generated/", []) in rules  # unowned rule is kept
    assert ("path\\ with\\ spaces/**", ["@acme/weird"]) in rules
    assert len(rules) == 10


def test_parse_handles_tabs_trailing_comments_and_crlf():
    rules = parse_codeowners("*.js\t@a/b\t@c # trailing\r\n\r\n   # indented comment\r\n/x/ @d\r\n")
    assert rules == [("*.js", ["@a/b", "@c"]), ("/x/", ["@d"])]


def test_find_codeowners_precedence(tmp_path: Path):
    assert find_codeowners(FIXTURE) == FIXTURE / ".github" / "CODEOWNERS"
    assert find_codeowners(tmp_path) is None
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CODEOWNERS").write_text("* @x\n", encoding="utf-8")
    assert find_codeowners(tmp_path) == tmp_path / "docs" / "CODEOWNERS"
    (tmp_path / "CODEOWNERS").write_text("* @y\n", encoding="utf-8")
    assert find_codeowners(tmp_path) == tmp_path / "CODEOWNERS"  # root wins over docs/


def test_is_mine_accepts_login_and_team_forms_case_insensitively():
    assert is_mine("@Me-Login", [], "me-login")
    assert is_mine("me-login", [], "@me-login")
    assert is_mine("@acme/core-team", ["acme/core-team"], "")
    assert is_mine("@acme/core-team", ["@ACME/Core-Team"], "")
    assert not is_mine("@acme/other", ["@acme/core-team"], "me-login")
    assert not is_mine("me-login@example.com", [], "me-login")  # emails are never mine
    assert not is_mine("@someone", [], "")


def test_foreign_owned_globs_filters_my_teams_and_login_with_last_match_wins():
    globs = foreign_owned_globs(FIXTURE, ["@acme/core-team"], "me-login")
    assert globs == ["src/billing/**", "infra/**", "path with spaces/**"]
    # src/auth/** was foreign, then reassigned to me -> not foreign; infra/** the opposite -> foreign.
    assert "src/auth/**" not in globs
    assert "/generated/" not in globs  # unowned


def test_foreign_owned_globs_everything_foreign_when_i_am_nobody():
    globs = foreign_owned_globs(FIXTURE, [], "")
    assert globs == [
        "*", "/docs/", "docs/*.md", "src/billing/**", "src/auth/**", "infra/**", "path with spaces/**",
    ]  # fmt: skip


def test_foreign_owned_globs_without_file_is_empty(tmp_path: Path):
    assert foreign_owned_globs(tmp_path, ["@acme/core-team"], "me") == []


def test_globs_work_with_pathspec_gitwildmatch():
    globs = foreign_owned_globs(FIXTURE, ["@acme/core-team"], "me-login")
    spec = pathspec.PathSpec.from_lines("gitignore", globs)
    assert spec.match_file("src/billing/invoice.py")
    assert spec.match_file("infra/terraform/main.tf")
    assert spec.match_file("path with spaces/file.txt")
    assert not spec.match_file("src/auth/login.py")
    assert not spec.match_file("docs/guide.md")
    assert not spec.match_file("README.md")


def test_to_glob_unescapes_spaces():
    assert to_glob("path\\ with\\ spaces/**") == "path with spaces/**"
    assert to_glob("/docs/") == "/docs/"


def test_owners_for_path_level_last_match_wins():
    rules = parse_codeowners((FIXTURE / ".github" / "CODEOWNERS").read_text(encoding="utf-8"))
    assert owners_for("src/auth/login.py", rules) == ["@me-login"]
    assert owners_for("infra/main.tf", rules) == ["@acme/platform"]
    assert owners_for("docs/guide.md", rules) == ["@acme/docs-team", "@me-login"]
    assert owners_for("docs/img/logo.png", rules) == ["@me-login"]  # /docs/ (dir) but not docs/*.md
    assert owners_for("generated/out.js", rules) == []  # explicitly unowned overrides "*"
    assert owners_for("README.md", rules) == ["@acme/core-team"]


def test_load_rules_and_is_foreign(tmp_path: Path):
    assert load_rules(FIXTURE) == parse_codeowners((FIXTURE / ".github" / "CODEOWNERS").read_text(encoding="utf-8"))
    assert load_rules(tmp_path) == []
    assert not is_foreign([], [], "me")  # unowned is not foreign
    assert is_foreign(["@acme/core-team"], [], "me")
    assert not is_foreign(["@acme/core-team"], ["@acme/core-team"], "me")
    assert not is_foreign(["@acme/docs-team", "@me"], [], "me")  # co-owned with me counts as mine


def test_foreign_owned_paths_applies_last_match_wins_per_path():
    """A catch-all rule narrowed later for me: the glob approximation keeps ``*`` sensitive and
    would refuse every push, while GitHub assigns ``docs/guide.md`` to me."""
    assert "*" in foreign_owned_globs(FIXTURE, [], "me-login")
    files = [
        "docs/guide.md", "docs/img/logo.png", "README.md", "src/auth/login.py", "src/billing/invoice.py",
        "infra/main.tf", "generated/out.js", "path with spaces/file.txt", "docs\\sub\\nested.md", "/docs/rooted.md",
    ]  # fmt: skip
    assert foreign_owned_paths(FIXTURE, files, [], "me-login") == [
        "README.md", "src/billing/invoice.py", "infra/main.tf", "path with spaces/file.txt",
    ]  # fmt: skip
    # my team owns the catch-all: only the explicitly foreign paths remain
    assert foreign_owned_paths(FIXTURE, files, ["@acme/core-team"], "me-login") == [
        "src/billing/invoice.py", "infra/main.tf", "path with spaces/file.txt",
    ]  # fmt: skip
    # nobody is me: everything owned is foreign, unowned paths still are not
    assert "generated/out.js" not in foreign_owned_paths(FIXTURE, files, [], "")
    assert "docs/guide.md" in foreign_owned_paths(FIXTURE, files, [], "")


def test_foreign_owned_paths_without_codeowners_is_empty(tmp_path: Path):
    assert foreign_owned_paths(tmp_path, ["src/auth/login.py"], [], "me") == []
