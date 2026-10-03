"""repos.py: remote parsing, discovery over real git repos, lookup, summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from software_factory import repos as repos_mod
from software_factory.config import Profile, RepoOverride, ReposConfig
from software_factory.repos import (
    RepoIndex,
    RepoInfo,
    first_section,
    git_default_branch,
    git_dirs,
    git_origin_url,
    parse_github_remote,
)

# ------------------------------------------------------------------------ parse_github_remote


@pytest.mark.parametrize(
    "url, slug",
    [
        ("git@github.com:JaneDoe/game.git", "JaneDoe/game"),
        ("git@github.com:JaneDoe/game", "JaneDoe/game"),
        ("https://github.com/JaneDoe/word-game.git", "JaneDoe/word-game"),
        ("https://github.com/JaneDoe/word-game", "JaneDoe/word-game"),
        ("https://github.com/JaneDoe/word-game/", "JaneDoe/word-game"),
        ("https://jane@github.com/JaneDoe/x.git", "JaneDoe/x"),
        ("ssh://git@github.com/JaneDoe/x.git", "JaneDoe/x"),
        ("ssh://git@github.com:22/JaneDoe/x.git", "JaneDoe/x"),
        ("git://github.com/JaneDoe/x.git", "JaneDoe/x"),
        ("  https://GitHub.com/Acme/Repo.git\n", "Acme/Repo"),
        ("org-123@github.com:Acme/repo.git", "Acme/repo"),
    ],
)
def test_parse_github_remote_accepts_github_forms(url, slug):
    assert parse_github_remote(url) == slug


@pytest.mark.parametrize(
    "url",
    [
        "",
        "git@gitlab.com:JaneDoe/x.git",
        "https://bitbucket.org/JaneDoe/x.git",
        "https://github.com/JaneDoe",
        "https://github.com/JaneDoe/x/y.git",
        "https://github.com.evil.example/JaneDoe/x.git",
        "https://notgithub.com/JaneDoe/x.git",
        "C:\\Users\\me\\bare.git",
        "/tmp/bare.git",
    ],
)
def test_parse_github_remote_rejects_non_github(url):
    assert parse_github_remote(url) is None


# ---------------------------------------------------------------------------------- discover


def make_profile(root: Path, owners: list[str] | None = None, **repos) -> Profile:
    return Profile(
        name="t",
        github={"owners": owners if owners is not None else ["JaneDoe"]},
        repos=ReposConfig(root=str(root), **repos),
    )


def test_discover_ssh_and_https_remotes_with_dirname_differing_from_name(git_repo_factory, repos_root):
    git_repo_factory("game-client", "git@github.com:JaneDoe/game.git")
    git_repo_factory("word-game-v1", "https://github.com/JaneDoe/word-game.git")
    (repos_root / "not-a-repo").mkdir()
    (repos_root / "stray.txt").write_text("x", encoding="utf-8")

    index = RepoIndex.discover(make_profile(repos_root))

    assert index.slugs() == ["JaneDoe/game", "JaneDoe/word-game"]
    game = index.get("game-client")
    assert game is not None
    assert game.slug == "JaneDoe/game" and game.name == "game" and game.dirname == "game-client"
    assert game.path == repos_root / "game-client"
    assert game.owner == "JaneDoe"
    assert index.get("JaneDoe/game") is game
    assert index.get("game") is game
    assert index.get("JANEDOE/GAME") is game and index.get("  Game-Client ") is game
    assert index.get("word-game-v1").slug == "JaneDoe/word-game"
    assert index.get("word-game").dirname == "word-game-v1"
    assert index.get("nope") is None
    assert len(index) == 2 and [r.dirname for r in index] == ["game-client", "word-game-v1"]


def test_owner_filter_is_case_insensitive_and_excludes_others(git_repo_factory, repos_root):
    git_repo_factory("mine", "git@github.com:janedoe/mine.git")
    git_repo_factory("theirs", "git@github.com:SomeoneElse/theirs.git")
    git_repo_factory("gitlab", "git@gitlab.com:JaneDoe/gitlab.git")
    git_repo_factory("no-remote", None)

    index = RepoIndex.discover(make_profile(repos_root, owners=["JANEDOE"]))

    assert index.slugs() == ["janedoe/mine"]


def test_include_adds_foreign_owner_by_dirname_name_or_slug(git_repo_factory, repos_root):
    git_repo_factory("fork-dir", "git@github.com:Upstream/forked.git")
    git_repo_factory("other", "git@github.com:Upstream/other.git")
    git_repo_factory("third", "git@github.com:Upstream/third.git")
    git_repo_factory("fourth", "git@github.com:Upstream/fourth.git")

    index = RepoIndex.discover(make_profile(repos_root, include=["Fork-Dir", "upstream/other", "THIRD"]))

    assert index.slugs() == ["Upstream/forked", "Upstream/other", "Upstream/third"]


def test_no_owners_and_no_include_discovers_nothing(git_repo_factory, repos_root):
    git_repo_factory("mine", "git@github.com:JaneDoe/mine.git")
    assert RepoIndex.discover(make_profile(repos_root, owners=[])).all() == []


def test_exclude_by_dirname_name_or_slug(git_repo_factory, repos_root):
    git_repo_factory("notes", "git@github.com:JaneDoe/notes.git")
    git_repo_factory("assets-dir", "git@github.com:JaneDoe/art-assets.git")
    git_repo_factory("cheatsheet", "git@github.com:JaneDoe/algo-cheatsheet.git")
    git_repo_factory("keep", "git@github.com:JaneDoe/keep.git")

    index = RepoIndex.discover(
        make_profile(repos_root, exclude=["NOTES", "art-assets", "janedoe/algo-cheatsheet"])
    )

    assert index.slugs() == ["JaneDoe/keep"]


def test_exclude_beats_include(git_repo_factory, repos_root):
    git_repo_factory("x", "git@github.com:Other/x.git")
    assert RepoIndex.discover(make_profile(repos_root, include=["x"], exclude=["x"])).all() == []


def test_duplicate_clones_of_one_slug_keep_the_first(git_repo_factory, repos_root):
    git_repo_factory("b-copy", "git@github.com:JaneDoe/dup.git")
    git_repo_factory("a-copy", "https://github.com/JaneDoe/dup.git")
    index = RepoIndex.discover(make_profile(repos_root))
    assert [r.dirname for r in index.all()] == ["a-copy"]


def test_missing_root_yields_empty_index(tmp_path):
    index = RepoIndex.discover(make_profile(tmp_path / "does not exist"))
    assert index.all() == [] and index.slugs() == [] and index.get("x") is None


def test_default_branch_from_origin_head_override_or_main(git_repo_factory, repos_root):
    git_repo_factory("dev", "git@github.com:JaneDoe/dev.git", default_branch="develop")
    git_repo_factory("unset", "git@github.com:JaneDoe/unset.git", default_branch="trunk", set_origin_head=False)
    git_repo_factory("over", "git@github.com:JaneDoe/over.git", default_branch="develop")
    profile = make_profile(repos_root, overrides={"over": RepoOverride(default_branch="release")})

    index = RepoIndex.discover(profile)

    assert index.get("dev").default_branch == "develop"
    assert index.get("unset").default_branch == "main"
    assert index.get("over").default_branch == "release"
    assert index.get("over").override.default_branch == "release"


def test_default_branch_guessed_from_remote_refs_when_origin_head_is_unset(git_repo_factory, repos_root, git):
    # Older repos often default to master with no origin/HEAD set locally; assuming
    # "main" made readiness/build fetch a ref that does not exist.
    legacy = git_repo_factory("legacy", "git@github.com:JaneDoe/legacy.git", default_branch="master", set_origin_head=False)
    git(["update-ref", "refs/remotes/origin/master", "HEAD"], legacy)
    both = git_repo_factory("both", "git@github.com:JaneDoe/both.git", default_branch="master", set_origin_head=False)
    git(["update-ref", "refs/remotes/origin/master", "HEAD"], both)
    git(["update-ref", "refs/remotes/origin/main", "HEAD"], both)

    index = RepoIndex.discover(make_profile(repos_root))

    assert index.get("legacy").default_branch == "master"
    assert index.get("both").default_branch == "main"


def test_override_matched_by_dirname_or_slug(git_repo_factory, repos_root):
    git_repo_factory("game-client", "git@github.com:JaneDoe/game.git")
    git_repo_factory("arcade", "git@github.com:JaneDoe/arcade.git")
    profile = make_profile(
        repos_root,
        overrides={
            "game-client": RepoOverride(visual=True, dev_port=5173),
            "JaneDoe/arcade": RepoOverride(visual=True, dev_port=5174, dev_command="npm run dev"),
        },
    )
    index = RepoIndex.discover(profile)
    assert index.get("game").override.dev_port == 5173
    assert index.get("arcade").override.dev_command == "npm run dev"


def test_worktree_checkout_with_git_file_is_discovered(git_repo_factory, repos_root, git, tmp_path):
    main_clone = git_repo_factory("primary", "git@github.com:JaneDoe/primary.git", root=tmp_path / "elsewhere")
    git(["worktree", "add", "-q", "-b", "feature", str(repos_root / "primary-wt")], main_clone)
    assert (repos_root / "primary-wt" / ".git").is_file()

    index = RepoIndex.discover(make_profile(repos_root))

    assert index.slugs() == ["JaneDoe/primary"]
    assert index.get("primary-wt").default_branch == "main"
    dirs = git_dirs(repos_root / "primary-wt")
    assert dirs is not None and dirs[1] == (main_clone / ".git").resolve()


# -------------------------------------------------------------------------------- git helpers


def test_git_origin_url_falls_back_to_git_when_config_unparsable(git_repo_factory, monkeypatch):
    repo = git_repo_factory("r", "git@github.com:JaneDoe/r.git")
    monkeypatch.setattr(repos_mod, "_origin_from_config", lambda _path: None)
    assert git_origin_url(repo) == "git@github.com:JaneDoe/r.git"


def test_git_origin_url_reads_config_without_subprocess(git_repo_factory, monkeypatch):
    repo = git_repo_factory("r", "https://github.com/JaneDoe/r.git")
    monkeypatch.setattr(repos_mod, "run_git", lambda *a, **k: pytest.fail("subprocess used"))
    assert git_origin_url(repo) == "https://github.com/JaneDoe/r.git"


def test_git_helpers_on_non_repo(tmp_path):
    assert git_dirs(tmp_path) is None
    assert git_origin_url(tmp_path) is None
    assert git_default_branch(tmp_path) is None
    (tmp_path / ".git").write_text("garbage\n", encoding="utf-8")
    assert git_dirs(tmp_path) is None


def test_default_branch_via_symbolic_ref_fallback(git_repo_factory, monkeypatch):
    repo = git_repo_factory("r", "git@github.com:JaneDoe/r.git", default_branch="develop")
    # Hide the HEAD file so the subprocess path is exercised.
    monkeypatch.setattr(repos_mod, "git_dirs", lambda _path: None)
    assert git_default_branch(repo) == "develop"


# ---------------------------------------------------------------------------------- summary


def test_summary_prefers_claude_md_then_agents_then_readme(git_repo_factory, repos_root):
    git_repo_factory(
        "all",
        "git@github.com:JaneDoe/all.git",
        files={"CLAUDE.md": "# All\n\nClaude intro.\n\n## Commands\n\nnpm test\n", "AGENTS.md": "# A\nagents", "README.md": "# R\nreadme"},
    )
    git_repo_factory("agents", "git@github.com:JaneDoe/agents.git", files={"AGENTS.md": "# Agents\n\nAgents intro.\n\n## X\n\nmore"})
    git_repo_factory("readme", "git@github.com:JaneDoe/readme.git", files={"README.md": "Readme only."})
    git_repo_factory("none", "git@github.com:JaneDoe/none.git", files={"src/main.py": "print(1)\n"})

    index = RepoIndex.discover(make_profile(repos_root))

    assert index.summary("JaneDoe/all") == "# All\n\nClaude intro."
    assert index.summary("agents") == "# Agents\n\nAgents intro."
    assert index.summary("readme") == "Readme only."
    assert index.summary("none") == ""
    assert index.summary("unknown/repo") == ""


def test_summary_is_capped_at_1500_chars(git_repo_factory, repos_root):
    git_repo_factory("big", "git@github.com:JaneDoe/big.git", files={"README.md": "# Big\n\n" + "word " * 1000})
    index = RepoIndex.discover(make_profile(repos_root))
    summary = index.summary("big")
    assert len(summary) <= 1500 and summary.endswith("\u2026")


def test_first_section_skips_title_only_heading():
    text = "# CLAUDE.md\n\n## Workspace Overview\n\nMulti-project workspace.\n\n## Major Projects\n\n| a |\n"
    assert first_section(text) == "# CLAUDE.md\n\n## Workspace Overview\n\nMulti-project workspace."
    assert first_section("# Title\n\nBody.\n\n## Next\n\nMore") == "# Title\n\nBody."
    assert first_section("plain text, no headings\nsecond line") == "plain text, no headings\nsecond line"
    assert first_section("") == ""
    assert first_section("#not-a-heading\ntext\n# Real\nbody\n# Second\nx") == "#not-a-heading\ntext\n# Real\nbody"


# ---------------------------------------------------------------------------------- index API


def test_index_from_explicit_list_and_key_priority(tmp_path):
    a = RepoInfo(slug="O/alpha", name="alpha", dirname="beta", path=tmp_path, default_branch="main", override=RepoOverride())
    b = RepoInfo(slug="O/beta", name="beta", dirname="beta-dir", path=tmp_path, default_branch="main", override=RepoOverride())
    index = RepoIndex([b, a])
    assert index.slugs() == ["O/alpha", "O/beta"]
    assert index.names() == ["alpha", "beta"]
    # "beta" is a's dirname but b's name: the name wins over the dirname.
    assert index.get("beta") is b
    assert index.get("beta-dir") is b and index.get("O/alpha") is a
    assert a.keys() == {"o/alpha", "alpha", "beta"}
    assert RepoIndex().all() == []
