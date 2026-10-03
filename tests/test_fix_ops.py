"""Ops-bucket regressions that live in committed config and docs rather than in one module:
the example profile's dev ports and the operator docs that must agree with the code."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

from software_factory import cli, effects, policy, providers, runner
from software_factory.actions import Action, Kind
from software_factory.config import REPO_ROOT, Profile, load_profile
from software_factory.kinds import base as kinds_base
from software_factory.kinds.base import detached_head_hint
from software_factory.scan import exhausted_flag

EXAMPLE_PROFILE = REPO_ROOT / "profiles" / "example" / "factory.toml"


def _flat(path: Path) -> str:
    """The file's text with every whitespace run collapsed (docs wrap lines anywhere)."""
    return " ".join(path.read_text(encoding="utf-8").split())


def _row(path: Path, key: str) -> str:
    return next(line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(f"| `{key}`"))


def test_example_profile_gives_every_visual_repo_its_own_port() -> None:
    """Two visual repos on one port and max_parallel > 1: Vite auto-increments the second server and
    the build worker screenshots the other repo. The shipped example is what new profiles are copied
    from, so its ports must be unique and pinned strictly on the command line."""
    profile = load_profile("example", REPO_ROOT / "profiles")
    by_port: dict[int, list[str]] = {}
    visual = [name for name, override in profile.repos.overrides.items() if override.visual]
    assert len(visual) >= 2  # uniqueness is only demonstrated with more than one visual repo
    for name, override in profile.repos.overrides.items():
        if override.visual:
            assert override.dev_command and override.dev_port, f"{name}: visual repos need dev_command + dev_port"
            assert f"--port {override.dev_port}" in override.dev_command, name
            assert "--strictPort" in override.dev_command, name
        if override.dev_port:
            by_port.setdefault(override.dev_port, []).append(name)
    assert {port: names for port, names in by_port.items() if len(names) > 1} == {}
    assert profile.max_parallel > 1  # the reason uniqueness matters
    report = cli.DoctorReport()
    cli._doctor_dev_ports(profile, report)
    assert [c.status for c in report.checks] == ["ok"], report.checks


def test_shipped_profiles_are_generic_placeholders() -> None:
    """The example profile and factory.example.toml are templates: placeholder owner, generic root,
    upkeep kinds only, no secrets."""
    profile = load_profile("example", REPO_ROOT / "profiles")
    assert profile.name == "example" and profile.employer is False
    assert profile.github.owners == ["your-github-login"] and profile.github.login == ""
    assert profile.repos.root == "~/code" and profile.repos.exclude == []
    assert [k.value for k in profile.enabled_kinds] == ["ci", "conflict", "rc", "rv"]
    assert profile.secrets == {}
    example = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert example["github"]["owners"] == ["your-github-login"] and example["github"]["login"] == ""
    assert example["repos"]["root"] == "~/code"


def test_repo_profiles_dir_tracks_only_the_readme_and_the_example() -> None:
    """Real profiles live outside the repo; .gitignore keeps a stray one in <repo>/profiles out of git."""
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for rule in ("/profiles/*", "!/profiles/README.md", "!/profiles/example/"):
        assert rule in lines, rule
    assert lines.index("/profiles/*") < lines.index("!/profiles/README.md")
    assert lines.index("/profiles/*") < lines.index("!/profiles/example/")
    assert "**/.env" in lines  # a .env next to the example stays ignored too


def test_docs_describe_where_profiles_live_now() -> None:
    """The personal profile is no longer committed: every operator doc names the user config dir and
    the env override, and none still points at a committed personal profile."""
    for path in (README, PROFILES_README, EXAMPLE, ARCHITECTURE, REPO_ROOT / "CLAUDE.md"):
        text = _flat(path)
        assert "software-factory/profiles" in text, path.name
        assert "SOFTWARE_FACTORY_PROFILES_DIR" in text, path.name
        # Windows uses ~/.config too: packaged (MSIX) apps virtualize %APPDATA%.
        assert "XDG_CONFIG_HOME" in text and "Windows" in text, path.name
        assert r"%APPDATA%\software-factory" not in text, f"{path.name}: still points at %APPDATA%"
        for stale in ("committed `personal` profile", "`profiles/personal/factory.toml` is committed",
                      "committed for the personal profile"):
            assert stale not in text, f"{path.name}: {stale}"
    for path in (README, PROFILES_README):
        assert "cp profiles/example/factory.toml" in _flat(path), path.name
    assert "SOFTWARE_FACTORY_CONFIG_DIR" in _flat(PROFILES_README) and "SOFTWARE_FACTORY_CONFIG_DIR" in _flat(ARCHITECTURE)


def test_profiles_readme_matches_the_code_on_when_linear_api_key_is_needed() -> None:
    """profiles/README.md used to say the key is needed whenever linear.enabled = true; the code
    (doctor warns, precheck skips the tracker) only needs it once an issue kind is enabled."""
    text = (REPO_ROOT / "profiles" / "README.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| `LINEAR_API_KEY`"))
    assert "issue kind" in row and "enabled_kinds" in row and "warns" in row
    main = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    main_row = next(line for line in main.splitlines() if line.startswith("| `LINEAR_API_KEY`"))
    assert "issue kind" in main_row


def test_provider_cli_path_is_documented_where_operators_look() -> None:
    example = (REPO_ROOT / "factory.example.toml").read_text(encoding="utf-8")
    assert "# cli_path = " in example and "claude update" in example
    # the commented example is valid TOML once uncommented
    tomllib.loads('[provider]\n' + next(line for line in example.splitlines() if line.startswith("# cli_path = "))[2:])
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "provider.cli_path" in readme and "older than the CLI the SDK expects" in readme
    assert "gh is logged in as X but github.login = Y" in readme


# ------------------------------------------------------------- docs vs. the safety follow-ups
README = REPO_ROOT / "README.md"
PROFILES_README = REPO_ROOT / "profiles" / "README.md"
EXAMPLE = REPO_ROOT / "factory.example.toml"
ARCHITECTURE = REPO_ROOT / "docs" / "architecture.md"


def test_gh_worker_token_is_documented_wherever_operators_set_secrets() -> None:
    """Workers get no gh login; the optional read-only GH_WORKER_TOKEN is the only way they run gh
    authenticated. Every place that lists .env keys must name it, say it is read-only with the exact
    fine-grained permissions, and say that without it workers are unauthenticated by design."""
    key = providers.GH_WORKER_TOKEN_KEY
    for path in (README, PROFILES_README):
        row = " ".join(_row(path, key).split()).lower()
        assert "read-only" in row and "contents, pull requests" in row and "actions: read" in row, path
        assert "without it workers have no `gh` authentication" in row and "doctor" in row, path
    example = _flat(EXAMPLE)
    assert f"# {key} optional: a fine-grained READ-ONLY GitHub PAT" in example
    assert "Without it # workers have no gh authentication at all, by design" in example
    shipped = _flat(EXAMPLE_PROFILE)
    assert key in shipped and "read-only PAT" in shipped and "unauthenticated" in shipped
    # the doctor line the README points at is the one doctor prints
    readme = _flat(README)
    assert f"worker gh token: {key} not in the profile .env" in readme


def test_readme_troubleshooting_matches_the_attempt_cap_and_kept_worktree_behaviour() -> None:
    """The README used to say any change re-arms an exhausted action; build is keyed by the issue id,
    so only `factory run build KEY` retries it. The engine's own flag text is the reference."""
    readme = _flat(README)
    assert "flags it once when the cap is hit" in readme
    assert "PR kinds (`ci`, `conflict`, `rc`, `rv`) re-arm on a new head, base or review feedback" in readme
    assert "`refine` / `plan` re-arm when the issue is edited" in readme
    assert "editing the issue does **not** re-arm it" in readme and "`factory run build KEY`" in readme

    build = Action(kind=Kind.BUILD, version="uuid-1", issue_id="uuid-1", issue_key="ABC-1")
    profile = Profile(name="personal")
    assert "factory run build ABC-1" in exhausted_flag(profile, build, 3).detail
    assert "editing the issue does not re-arm it" in exhausted_flag(profile, build, 3).detail
    pr = Action(kind=Kind.CI, version="sha", repo="o/n", pr=1)
    assert "new head, base or feedback" in exhausted_flag(profile, pr, 3).detail

    # kept worktrees are detached HEADs; the README tells the human the same command the flag does
    # (a ``-factory`` local branch: ``<branch>`` itself usually exists in the clone already)
    assert "`git switch -c feat/x-factory`" in detached_head_hint("feat/x")
    assert "`git push origin HEAD:refs/heads/feat/x`" in detached_head_hint("feat/x", pr=True)
    assert "**detached HEAD**" in readme and "`git switch -c <branch>-factory`" in readme
    assert "`git push origin HEAD:refs/heads/<branch>`" in readme
    assert "`git switch -c <branch>-factory`" in _flat(PROFILES_README)
    assert "`git switch -c <branch>`" not in readme and "`git switch -c <branch>`" not in _flat(PROFILES_README)
    assert "without `--remove` the directory stays for you (until `factory cleanup`) and no longer blocks the next run" in readme


def test_readme_documents_rate_limits_hooks_settings_and_per_file_codeowners() -> None:
    readme = _flat(README)
    # provider limits stop the tick without spending attempts
    assert "2 auth failure or provider rate/usage limit" in readme
    assert "spends no attempt and flags once per day" in readme
    # below-the-tool-layer guarantees
    assert "**git hooks disabled**" in readme and "`.claude/settings.json`" in readme
    assert "`setting_sources=[]`" in readme
    assert "**every GitHub push URL rewritten**" in readme
    # CODEOWNERS per changed file, in the README and in the example config
    assert "evaluated **per changed file** with GitHub's last-match-wins rule" in readme
    assert "CODEOWNERS is applied per changed file with GitHub's last-match-wins rule" in _flat(EXAMPLE)
    # hooks / settings / gh are not knobs
    assert "Not configurable, by design: workers run git with hooks disabled" in _flat(EXAMPLE)


def test_architecture_contract_drops_what_round_one_made_false_and_names_the_new_seams() -> None:
    """docs/architecture.md wins over everything else when they disagree, so the statements the
    review rounds made false must be gone and the new public names must be in it (and exist)."""
    arch = _flat(ARCHITECTURE)
    for stale in (
        'setting_sources=["project"]',
        "(or the OS temp dir)",
        "policy + override + CODEOWNERS(not my_teams)",
        "the push policy should prefer it over `foreign_owned_globs`",
        "exit_code` (0 ok, 1 some action failed, 2 auth failure).",
    ):
        assert stale not in arch, stale
    assert "**`setting_sources=[]`**" in arch and "PreToolUse timeout 30 s" in arch
    assert policy.PRE_TOOL_USE_HOOK_TIMEOUT_S == 30.0
    assert "refine -> plan -> build" in arch and "ci -> conflict -> rc -> rv" in arch
    assert "deliberately NOT GH_TOKEN / GITHUB_TOKEN" in arch
    assert "GH_TOKEN" not in providers.SCRUB_ENV_VARS and "GITHUB_TOKEN" not in providers.SCRUB_ENV_VARS
    names = {
        "rate_limited": "rate_limited" in runner.WorkerRun.model_fields,
        "project_instructions": callable(runner.project_instructions),
        "kill_tree": "kill_tree" in runner.run_worker.__code__.co_varnames,
        "link_pushes": callable(effects.link_pushes),
        "is_auth_failure": callable(effects.is_auth_failure),
        "foreign_owned_changes": callable(policy.foreign_owned_changes),
        "worker_git_config_env": callable(providers.worker_git_config_env),
        "unplantable_hooks_path": callable(getattr(providers, "unplantable_hooks_path", None)),
        "evidence_globs": callable(kinds_base.evidence_globs),
    }
    for name, exists in names.items():
        assert exists, f"{name} is documented but missing from the code"
        assert name in arch, f"{name} exists but is not in the contract"
    for name in ("ProviderUnavailable", "GitDirGuard", "worker_changes", "push_policy_inputs", "managed_slug",
                 "open_feedback", "clone_lock", "gh-noauth", "FlagSpec", "pr_pending", "build-waits"):
        assert name in arch, name


def test_docs_describe_the_unplantable_hooks_path_not_the_old_hooks_dirs() -> None:
    """The empty ``no-hooks`` / ``no-hooks-worker`` dirs under the profile root are gone: every git
    (workers and engine) uses one ``core.hooksPath`` under a regular file, where nothing can be
    planted. The docs must not keep describing the dirs, nor the claim that the worker must not
    learn where the engine looks for hooks (the path is public and harmless now)."""
    docs = {path: _flat(path) for path in (README, PROFILES_README, ARCHITECTURE, REPO_ROOT / "CLAUDE.md")}
    for path, text in docs.items():
        for stale in ("no-hooks-worker", "no-hooks/", "must not learn where the engine looks for hooks",
                      "an empty per-profile directory", "`core.hooksPath` -> an empty directory",
                      "`core.hooksPath` -> empty dir", "emptied before every session"):
            assert stale not in text, f"{path.name}: {stale}"
    for path in (README, PROFILES_README, ARCHITECTURE):
        assert "/dev/null/no-hooks" in docs[path] and r"drivers\etc\hosts\no-hooks" in docs[path], path.name
    assert "unplantable_hooks_path()" in docs[ARCHITECTURE] and "unplantable_hooks_path()" in docs[REPO_ROOT / "CLAUDE.md"]
    # the end-of-session process-tree kill and the explicit LFS upload are part of the contract
    assert "at the end of *every* session" in docs[ARCHITECTURE] and "**whole process tree killed**" in docs[README]
    assert "LFS" in docs[ARCHITECTURE] and "LFS" in docs[README] and "LFS" in docs[REPO_ROOT / "CLAUDE.md"]


def test_profiles_readme_only_promises_to_revert_what_the_git_dir_guard_reverts() -> None:
    """``GitDirGuard`` restores the clone's shared config and hooks; it does not undo every shell
    write the policy missed (a write into another git/gh/ssh config file is denied, not reverted)."""
    text = _flat(PROFILES_README)
    assert "reverted when they happen anyway" not in text
    assert "changes to the clone's shared git config and hooks are reverted" in text


def test_example_profile_loads_and_its_schedule_validates(tmp_path: Path) -> None:
    """factory.example.toml is copied by hand into new profiles: it must load as-is (the schedule
    block is validated at load time now)."""
    (tmp_path / "example").mkdir()
    shutil.copy(EXAMPLE, tmp_path / "example" / "factory.toml")
    profile = load_profile("example", tmp_path)
    assert profile.schedule.interval_minutes == 20 and profile.schedule.digest_time == "09:00"
    assert profile.max_attempts == 3 and profile.provider.cli_path is None
