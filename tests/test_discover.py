"""discover.py: verification-command discovery from fixture repos, priority order, caching."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from software_factory.config import RepoOverride
from software_factory.discover import (
    GODOT_GUT_COMMAND,
    PRE_COMMIT_COMMAND,
    Verification,
    agent_doc,
    discover,
    discover_cached,
    git_head_sha,
    has_verification_section,
)

FIXTURES = Path(__file__).parent / "fixtures" / "repos"


def commands(cmds) -> list[str]:
    return [c.command for c in cmds]


def copy_fixture(name: str, tmp_path: Path) -> Path:
    dest = tmp_path / name
    shutil.copytree(FIXTURES / name, dest)
    return dest


# ------------------------------------------------------------------------------- CLAUDE.md
def test_claude_md_sections_fenced_blocks_and_inline_list_items():
    v = discover(FIXTURES / "claude_md_only")
    assert v.source == "CLAUDE.md"
    # Commands section (weak heading) keeps only verification-looking commands; Verification section
    # (strong heading) keeps every runner command that is not long-running or shell-prefixed.
    assert commands(v.verify) == ["npm run typecheck", "npm run lint", "npm test", "uv run pytest"]
    assert all(c.source == "CLAUDE.md" for c in v.verify)
    assert [c.name for c in v.verify] == ["typecheck", "lint", "test", "test"]
    # Explicit setup from the "## Setup" section.
    assert commands(v.setup) == ["npm ci"]
    # No package.json -> no dev command.
    assert v.dev_command is None and v.dev_port is None


def test_claude_md_excludes_dev_servers_migrations_prose_and_non_runner_lines():
    v = discover(FIXTURES / "claude_md_only")
    excluded = {"npm run dev", "npm run db:migrate", "npm run build", "cd apps/api && npm run smoke", "no runner here"}
    assert excluded.isdisjoint(commands(v.verify))


def test_placeholders_mutating_commands_and_e2e_suites_are_not_verification(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text(
        "# X\n\n## Verification\n\n```bash\n"
        "npm run headless -- <seed>\n"
        "pnpm test src/path/to/file.test.ts\n"
        "npm run lint:fix\n"
        "npm run fmt\n"
        "ruff format\n"
        "npm run init-precommit\n"
        "npm run test:e2e\n"
        "npx playwright test\n"
        "npm run check:format\n"
        "ruff format --check .\n"
        "npm run lint\n"
        "```\n",
        encoding="utf-8",
    )
    assert commands(discover(tmp_path).verify) == ["npm run check:format", "ruff format --check .", "npm run lint"]


def test_workflow_steps_apply_the_same_candidate_filter(tmp_path: Path):
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text(
        "jobs:\n  t:\n    steps:\n"
        "      - name: Format check\n        run: npm run lint:fix\n"
        "      - name: E2E tests\n        run: npx playwright test\n"
        "      - name: Tests\n        run: npm test\n",
        encoding="utf-8",
    )
    assert commands(discover(tmp_path).verify) == ["npm test"]


def test_agents_md_used_when_claude_md_absent():
    v = discover(FIXTURES / "agents_md_only")
    assert v.source == "AGENTS.md"
    assert commands(v.verify) == ["cargo test --all", "cargo clippy -- -D warnings"]
    assert [c.name for c in v.verify] == ["test", "lint"]
    assert v.setup == []


def test_agents_md_used_when_claude_md_has_no_commands(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text("# Thin\n\nSee AGENTS.md.\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# Rules\n\n## Verify\n\n- `make check`\n", encoding="utf-8")
    v = discover(tmp_path)
    assert v.source == "AGENTS.md"
    assert commands(v.verify) == ["make check"]


def test_headings_inside_code_fences_are_not_sections(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text(
        "# Doc\n\n```md\n## Testing\n```\n\nnpm test\n\n## Layout\n\n- `npm test`\n", encoding="utf-8"
    )
    v = discover(tmp_path)
    assert v.verify == []
    assert v.source == "none"


# ------------------------------------------------------------------------------- workflows
def test_workflow_run_steps_keep_verification_and_skip_install_deploy_upload_and_ci_expressions():
    v = discover(FIXTURES / "workflows_only")
    assert v.source == ".github/workflows/ci.yml"
    assert commands(v.verify) == [
        "npm run lint",
        "npm run typecheck",
        "npm test",
        "npx vitest run --dir integration",
        "cd apps/web && npm run build",
    ]
    assert [c.name for c in v.verify] == ["Lint", "Typecheck", "Unit and integration tests", "Unit and integration tests", "Build"]
    assert all(c.source == ".github/workflows/ci.yml" for c in v.verify)
    # install step -> explicit setup (deduped across ci.yml and release.yml)
    assert commands(v.setup) == ["npm ci"]
    assert v.setup[0].source == ".github/workflows/ci.yml"
    skipped = {"npm run deploy", "npx codecov", "npm publish --access public", "npm run test:release", "npm run test:extra"}
    assert skipped.isdisjoint(commands(v.verify))


def test_workflow_without_jobs_or_invalid_yaml_is_ignored(tmp_path: Path):
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "broken.yml").write_text("jobs: [unclosed\n  - :\n", encoding="utf-8")
    (wf / "empty.yml").write_text("name: nothing\non: push\n", encoding="utf-8")
    assert discover(tmp_path).verify == []


# ------------------------------------------------------------------------------ package.json
def test_package_json_scripts_in_order_and_placeholder_test_skipped():
    v = discover(FIXTURES / "package_json_only")
    assert v.source == "package.json"
    assert commands(v.verify) == ["npm run typecheck", "npm run lint", "npm run build"]
    assert [c.name for c in v.verify] == ["typecheck", "lint", "build"]
    assert commands(v.setup) == ["npm ci"]
    assert v.setup[0].source == "package-lock.json"


def test_dev_command_and_port_from_vite_config():
    v = discover(FIXTURES / "package_json_only")
    assert v.dev_command == "npm run dev"
    assert v.dev_port == 5177


def test_dev_port_from_script_flag_and_pnpm_from_lockfile(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"scripts": {"dev": "vite --port 4100", "test": "vitest run"}}', encoding="utf-8"
    )
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: 9\n", encoding="utf-8")
    v = discover(tmp_path)
    assert commands(v.verify) == ["pnpm run test"]
    assert commands(v.setup) == ["pnpm install --frozen-lockfile"]
    assert (v.dev_command, v.dev_port) == ("pnpm run dev", 4100)


def test_package_manager_field_wins_over_lockfile(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"packageManager": "bun@1.1.0", "scripts": {"test": "bun test"}}', encoding="utf-8"
    )
    (tmp_path / "yarn.lock").write_text("", encoding="utf-8")
    v = discover(tmp_path)
    assert commands(v.verify) == ["bun run test"]
    assert commands(v.setup) == ["yarn install --frozen-lockfile"]  # lockfile still drives setup


# ------------------------------------------------------------------------- Makefile / pyproject
def test_makefile_targets():
    v = discover(FIXTURES / "makefile_only")
    assert v.source == "Makefile"
    assert commands(v.verify) == ["make lint", "make test"]  # typecheck := ... is a variable


def test_pyproject_with_uv_lock_uses_uv_run():
    v = discover(FIXTURES / "pyproject_uv")
    assert v.source == "pyproject.toml"
    assert commands(v.verify) == ["uv run pytest"]
    assert commands(v.setup) == ["uv sync"]


def test_pyproject_without_uv_lock_uses_python_m_pytest(tmp_path: Path):
    repo = copy_fixture("pyproject_uv", tmp_path)
    (repo / "uv.lock").unlink()
    v = discover(repo)
    assert commands(v.verify) == ["python -m pytest"]
    assert v.setup == []


def test_pyproject_without_pytest_config_or_tests_dir_yields_nothing(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0"\n', encoding="utf-8")
    assert discover(tmp_path).verify == []


def test_justfile_recipes():
    v = discover(FIXTURES / "justfile_only")
    assert v.source == "justfile"
    assert commands(v.verify) == ["just test", "just check"]


def test_godot_with_gut():
    v = discover(FIXTURES / "godot_gut")
    assert v.source == "project.godot"
    assert commands(v.verify) == [GODOT_GUT_COMMAND]
    assert v.verify[0].name == "test"


def test_godot_without_gut_yields_nothing(tmp_path: Path):
    (tmp_path / "project.godot").write_text("[application]\n", encoding="utf-8")
    assert discover(tmp_path).verify == []


def test_empty_repo_is_harmless(tmp_path: Path):
    v = discover(tmp_path)
    assert v == Verification()
    assert v.source == "none"


# ------------------------------------------------------------------------------- priority
def test_priority_order_claude_md_workflows_package_json_makefile_pyproject_justfile(tmp_path: Path):
    repo = copy_fixture("all_sources", tmp_path)
    expectations = [
        (None, "CLAUDE.md", ["npm run verify"]),
        ("CLAUDE.md", ".github/workflows/ci.yml", ["npm run lint"]),
        (".github", "package.json", ["npm run test"]),
        ("package.json", "Makefile", ["make test"]),
        ("Makefile", "pyproject.toml", ["python -m pytest"]),
        ("pyproject.toml", "justfile", ["just test"]),
        ("justfile", "none", []),
    ]
    for remove, source, verify in expectations:
        if remove:
            target = repo / remove
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        v = discover(repo)
        assert (v.source, commands(v.verify)) == (source, verify), f"after removing {remove}"
        # setup is inferred from the lockfile whenever the winning tier has no explicit setup
        assert commands(v.setup) == ["npm ci"]


def test_dev_command_survives_tier_changes(tmp_path: Path):
    repo = copy_fixture("all_sources", tmp_path)
    assert (discover(repo).dev_command, discover(repo).dev_port) == ("npm run dev", 4321)
    (repo / "CLAUDE.md").unlink()
    assert discover(repo).dev_port == 4321


def test_explicit_doc_setup_beats_lockfile_inference(tmp_path: Path):
    repo = copy_fixture("all_sources", tmp_path)
    (repo / "CLAUDE.md").write_text("# X\n\n## Install\n\n```\npnpm install\n```\n\n## Test\n\n- `npm test`\n", encoding="utf-8")
    v = discover(repo)
    assert commands(v.verify) == ["npm test"]
    assert commands(v.setup) == ["pnpm install"]


# ------------------------------------------------------------------------------- pre-commit
PRE_COMMIT_YAML = "repos:\n  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: v0.6.0\n    hooks:\n      - id: ruff\n"


def test_pre_commit_config_adds_the_pre_commit_run_check(tmp_path: Path):
    """Regression: workers commit with hooks disabled, so a repo whose checks live in its
    pre-commit config had them skipped entirely. The config now yields a verification command."""
    (tmp_path / "CLAUDE.md").write_text("# X\n\n## Testing\n\n- `uv run pytest`\n", encoding="utf-8")
    (tmp_path / ".pre-commit-config.yaml").write_text(PRE_COMMIT_YAML, encoding="utf-8")
    v = discover(tmp_path)
    assert commands(v.verify) == ["uv run pytest", PRE_COMMIT_COMMAND]
    assert v.verify[-1].source == ".pre-commit-config.yaml" and v.verify[-1].name == "pre-commit"
    assert v.source == "CLAUDE.md"  # the winning tier still names the source

    # the config alone is enough for a repo with nothing else discoverable
    only = tmp_path / "only"
    only.mkdir()
    (only / ".pre-commit-config.yaml").write_text(PRE_COMMIT_YAML, encoding="utf-8")
    v = discover(only)
    assert commands(v.verify) == [PRE_COMMIT_COMMAND] and v.source == ".pre-commit-config.yaml"
    # no config, no command
    assert PRE_COMMIT_COMMAND not in commands(discover(FIXTURES / "claude_md_only").verify)


def test_pre_commit_run_already_listed_is_not_duplicated_and_installers_stay_excluded(tmp_path: Path):
    (tmp_path / ".pre-commit-config.yaml").write_text(PRE_COMMIT_YAML, encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(
        "# X\n\n## Lint\n\n```bash\n"
        "pre-commit install\n"
        "uv run pre-commit install --hook-type commit-msg\n"
        "pre-commit autoupdate\n"
        "npm run init-precommit\n"
        "uv run pre-commit run --all-files\n"
        "```\n",
        encoding="utf-8",
    )
    v = discover(tmp_path)
    assert commands(v.verify) == ["uv run pre-commit run --all-files"]  # the doc's own run covers it

    # a workflow step that runs pre-commit covers it too
    wf_repo = tmp_path / "wf"
    (wf_repo / ".github" / "workflows").mkdir(parents=True)
    (wf_repo / ".pre-commit-config.yaml").write_text(PRE_COMMIT_YAML, encoding="utf-8")
    (wf_repo / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  t:\n    steps:\n      - name: Lint (pre-commit)\n        run: pre-commit run --all-files --show-diff-on-failure\n"
        "      - name: Tests\n        run: npm test\n",
        encoding="utf-8",
    )
    assert commands(discover(wf_repo).verify) == [PRE_COMMIT_COMMAND, "npm test"]


def test_override_verify_is_not_extended_with_pre_commit(tmp_path: Path):
    (tmp_path / ".pre-commit-config.yaml").write_text(PRE_COMMIT_YAML, encoding="utf-8")
    assert commands(discover(tmp_path, RepoOverride(verify=["make check"])).verify) == ["make check"]
    assert discover(tmp_path, RepoOverride(verify=[])).verify == []


# ------------------------------------------------------------------------------- overrides
def test_override_verify_and_setup_win_over_everything():
    override = RepoOverride(verify=["uv run pytest -x", "uv run ruff check ."], setup=["uv sync --all-extras"])
    v = discover(FIXTURES / "claude_md_only", override)
    assert v.source == "override"
    assert commands(v.verify) == ["uv run pytest -x", "uv run ruff check ."]
    assert [c.name for c in v.verify] == ["test", "lint"]
    assert all(c.source == "override" for c in v.verify + v.setup)
    assert commands(v.setup) == ["uv sync --all-extras"]


def test_partial_override_keeps_discovered_fields():
    v = discover(FIXTURES / "package_json_only", RepoOverride(verify=["npm run check"], dev_port=9999))
    assert commands(v.verify) == ["npm run check"]
    assert commands(v.setup) == ["npm ci"]  # discovered setup kept
    assert v.dev_command == "npm run dev"  # discovered dev command kept
    assert v.dev_port == 9999  # overridden


def test_override_empty_verify_list_disables_discovery():
    v = discover(FIXTURES / "claude_md_only", RepoOverride(verify=[]))
    assert v.verify == [] and v.source == "override"


def test_override_dev_command():
    v = discover(FIXTURES / "claude_md_only", RepoOverride(dev_command="godot --path . --headless", dev_port=8080))
    assert (v.dev_command, v.dev_port) == ("godot --path . --headless", 8080)


# ------------------------------------------------------------------------- verification section
def test_agent_doc_and_verification_section_detection(tmp_path: Path):
    assert agent_doc(FIXTURES / "claude_md_only").name == "CLAUDE.md"
    assert agent_doc(FIXTURES / "agents_md_only").name == "AGENTS.md"
    assert agent_doc(FIXTURES / "package_json_only") is None
    assert has_verification_section(FIXTURES / "claude_md_only")
    assert has_verification_section(FIXTURES / "agents_md_only")
    assert not has_verification_section(FIXTURES / "package_json_only")
    # A "Commands" section counts only when it yields a verification command.
    (tmp_path / "CLAUDE.md").write_text("# X\n\n## Commands\n\n- `npm run dev`\n", encoding="utf-8")
    assert not has_verification_section(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("# X\n\n## Commands\n\n- `npm run dev`\n- `npm test`\n", encoding="utf-8")
    assert has_verification_section(tmp_path)


# -------------------------------------------------------------------------------- caching
class FakeState:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], object] = {}
        self.sets = 0

    def cache_get(self, namespace: str, key: str):
        return self.store.get((namespace, key))

    def cache_set(self, namespace: str, key: str, value) -> None:
        self.store[(namespace, key)] = value
        self.sets += 1


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=True,
    )
    return proc.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo with space"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "CLAUDE.md").write_text("# R\n\n## Test\n\n- `npm test`\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def repo_info(path: Path, override: RepoOverride | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        slug="octo/fixture", name="fixture", dirname=path.name, path=path, default_branch="main",
        override=override or RepoOverride(),
    )  # fmt: skip


def test_git_head_sha(git_repo: Path, tmp_path: Path):
    sha = git_head_sha(git_repo)
    assert len(sha) == 40
    assert git_head_sha(tmp_path / "not-a-repo") == ""


def test_discover_cached_keyed_by_head_sha(git_repo: Path):
    state = FakeState()
    repo = repo_info(git_repo)
    first = discover_cached(state, repo)
    assert commands(first.verify) == ["npm test"]
    assert state.sets == 1
    assert ("discover", "discover:octo/fixture") in state.store

    # Uncommitted edits do not invalidate (keyed by HEAD, not working tree).
    (git_repo / "CLAUDE.md").write_text("# R\n\n## Test\n\n- `npm run check`\n", encoding="utf-8")
    assert commands(discover_cached(state, repo).verify) == ["npm test"]
    assert state.sets == 1

    # A new commit does.
    git(git_repo, "commit", "-q", "-am", "change")
    assert commands(discover_cached(state, repo).verify) == ["npm run check"]
    assert state.sets == 2


def test_discover_cached_invalidates_on_override_change(git_repo: Path):
    state = FakeState()
    discover_cached(state, repo_info(git_repo))
    v = discover_cached(state, repo_info(git_repo, RepoOverride(verify=["make test"])))
    assert commands(v.verify) == ["make test"]
    assert state.sets == 2


def test_discover_cached_without_git_computes_each_time(tmp_path: Path):
    # The fixture tree lives inside this project's own git repo, so copy it somewhere git-free.
    repo = copy_fixture("makefile_only", tmp_path)
    state = FakeState()
    for _ in range(2):
        v = discover_cached(state, repo_info(repo))
        assert commands(v.verify) == ["make lint", "make test"]
    assert state.sets == 0
