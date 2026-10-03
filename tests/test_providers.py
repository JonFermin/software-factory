"""providers.py: worker env per provider, credential checks, env scrubbing."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import init_git_repo, run_git  # pytest puts tests/ on sys.path (no package)

from software_factory.config import Profile, ProviderConfig
from software_factory.paths import ProfilePaths
from software_factory.providers import (
    BASE_WORKER_ENV,
    GH_NOAUTH_CONFIG_NAME,
    GH_WORKER_TOKEN_KEY,
    NO_PUSH_PROTOCOL,
    NO_PUSH_URL_PREFIXES,
    SCRUB_ENV_VARS,
    UNPLANTABLE_HOOKS_CHILD,
    HooksPathError,
    ProviderError,
    check_credentials,
    fallback_model_id,
    gh_noauth_config_path,
    git_no_hooks_args,
    git_no_push_env,
    model_id,
    scrub_process_env,
    unplantable_hooks_path,
    worker_env,
)


def make(kind: str, secrets: dict[str, str] | None = None, employer: bool = False, **provider) -> Profile:
    return Profile(
        name="t",
        employer=employer,
        provider=ProviderConfig(kind=kind, **provider),
        secrets=secrets or {},
        disclosure_footer="footer",
    )


# -------------------------------------------------------------------------------- scrubbing


def test_scrub_process_env_pops_every_listed_var(monkeypatch):
    for name in SCRUB_ENV_VARS:
        monkeypatch.setenv(name, "leak")
    monkeypatch.setenv("PATH_KEEP_ME", "1")
    scrub_process_env()
    assert not any(name in os.environ for name in SCRUB_ENV_VARS)
    assert os.environ["PATH_KEEP_ME"] == "1"
    scrub_process_env()  # idempotent when nothing is set


def test_scrub_list_covers_the_contract_minimum():
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_OAUTH_SCOPES",
        "LINEAR_API_KEY",
        "SLACK_BOT_TOKEN",
    ):
        assert name in SCRUB_ENV_VARS


def test_scrub_keeps_the_engines_own_gh_token(monkeypatch):
    """Regression: scrubbing GH_TOKEN / GITHUB_TOKEN from os.environ broke `gh auth status` in every
    tick for a gh authenticated only through the env (doctor stayed green). Workers get their own
    override instead (``gh_isolation_env``)."""
    assert "GH_TOKEN" not in SCRUB_ENV_VARS and "GITHUB_TOKEN" not in SCRUB_ENV_VARS
    monkeypatch.setenv("GH_TOKEN", "ghp_engine")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_engine2")
    scrub_process_env()
    assert os.environ["GH_TOKEN"] == "ghp_engine" and os.environ["GITHUB_TOKEN"] == "ghp_engine2"


# ------------------------------------------------------------------------------- gh / git isolation


def test_worker_env_isolates_gh_from_the_users_login(monkeypatch):
    """A worker (or any script it runs) must not inherit the user's gh login: GH_CONFIG_DIR is a
    regular *file* (gh: not logged in; a hosts.yml cannot be created inside it, which is what would
    re-enable the keyring), and the inherited tokens are overridden with empty values (= unset)."""
    monkeypatch.setenv("GH_TOKEN", "ghp_engine")
    env = worker_env(make("api_key", {"ANTHROPIC_API_KEY": "k"}))
    config_dir = Path(env["GH_CONFIG_DIR"])
    assert config_dir.is_file() and config_dir.name == GH_NOAUTH_CONFIG_NAME
    assert env["GH_TOKEN"] == "" and env["GITHUB_TOKEN"] == ""
    assert env["GH_NO_UPDATE_NOTIFIER"] == "1"


def test_worker_env_forwards_the_read_only_worker_token():
    env = worker_env(make("api_key", {"ANTHROPIC_API_KEY": "k", GH_WORKER_TOKEN_KEY: "github_pat_readonly"}))
    assert env["GH_TOKEN"] == "github_pat_readonly"
    assert env["GITHUB_TOKEN"] == ""  # only gh's own variable carries it
    assert Path(env["GH_CONFIG_DIR"]).is_file()


def test_gh_noauth_config_path_replaces_a_planted_directory():
    profile = make("api_key", {"ANTHROPIC_API_KEY": "k"})
    path = gh_noauth_config_path(profile)
    assert path.is_file()
    path.unlink()
    path.mkdir()
    (path / "hosts.yml").write_text("github.com:\n    user: me\n", encoding="utf-8")
    again = gh_noauth_config_path(profile)
    assert again == path and again.is_file()


def test_worker_env_blocks_github_pushes_at_the_git_level():
    """``url.<x>.pushInsteadOf`` rewrites every GitHub push URL (remote name or explicit URL) to an
    unusable transport inside the worker's process tree; fetch URLs are untouched."""
    env = worker_env(make("api_key", {"ANTHROPIC_API_KEY": "k"}))
    block = git_no_push_env()
    # the push block is the head of the worker's GIT_CONFIG_* list (the hooks entry follows it)
    assert all(env[k] == v for k, v in block.items() if k != "GIT_CONFIG_COUNT")
    assert int(env["GIT_CONFIG_COUNT"]) >= int(block["GIT_CONFIG_COUNT"])
    count = int(block["GIT_CONFIG_COUNT"])
    assert count == len(NO_PUSH_URL_PREFIXES) >= 4
    for i in range(count):
        assert block[f"GIT_CONFIG_KEY_{i}"] == f"url.{NO_PUSH_PROTOCOL}.pushInsteadOf"
    values = {block[f"GIT_CONFIG_VALUE_{i}"] for i in range(count)}
    assert {"git@github.com:", "https://github.com/", "ssh://git@github.com/"} <= values


def _git_config_entries(env: dict[str, str]) -> dict[str, str]:
    count = int(env.get("GIT_CONFIG_COUNT", "0"))
    assert all(f"GIT_CONFIG_KEY_{i}" in env and f"GIT_CONFIG_VALUE_{i}" in env for i in range(count))
    assert f"GIT_CONFIG_KEY_{count}" not in env  # no dangling entries past the count
    return {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(count)}


def test_worker_env_points_git_hooks_at_the_unplantable_path():
    """``core.hooksPath`` in the worker's GIT_CONFIG_* block: a hook a worker plants in the shared
    ``.git/hooks`` (``echo 'git push' > .git/hooks/pre-commit``) never runs when the worker commits.
    Regression: it used to be ``<profile root>/no-hooks-worker``, an empty dir any process could
    refill between the emptying and git's exec; now it is a path under a regular file / character
    device that nobody can create, and no directory is made for it."""
    profile = make("api_key", {"ANTHROPIC_API_KEY": "k"})
    env = worker_env(profile)
    entries = _git_config_entries(env)
    hooks_path = entries["core.hooksPath"]
    assert hooks_path == unplantable_hooks_path()
    assert not os.path.isdir(hooks_path) and not os.path.isdir(os.path.dirname(hooks_path))
    root = ProfilePaths.for_profile(profile.name).root
    assert not (root / "no-hooks-worker").exists() and not (root / "no-hooks").exists()
    # the push block is still complete and in front
    assert [k for k in entries if k.endswith(".pushInsteadOf")] == [f"url.{NO_PUSH_PROTOCOL}.pushInsteadOf"]
    assert env["GIT_CONFIG_KEY_0"] == f"url.{NO_PUSH_PROTOCOL}.pushInsteadOf"
    assert env[f"GIT_CONFIG_KEY_{len(NO_PUSH_URL_PREFIXES)}"] == "core.hooksPath"


# --------------------------------------------------------------------- unplantable hooks path
def test_unplantable_hooks_path_lies_under_a_file_nobody_can_turn_into_a_directory():
    path = unplantable_hooks_path()
    parent = os.path.dirname(path)
    assert path.endswith(f"/{UNPLANTABLE_HOOKS_CHILD}") and "\\" not in path
    assert os.path.isabs(path) and os.path.exists(parent) and not os.path.isdir(parent)
    if os.name == "nt":
        expected = os.path.join(os.environ.get("SYSTEMROOT") or r"C:\Windows", "System32", "drivers", "etc", "hosts")
        assert os.path.normcase(parent) == os.path.normcase(expected.replace("\\", "/"))
        assert os.path.isfile(parent)
    else:
        assert parent == "/dev/null"
    with pytest.raises(OSError):
        os.makedirs(path)  # the whole point: no process can create the hooks dir
    assert git_no_hooks_args() == ["-c", f"core.hooksPath={path}"]


def test_unplantable_hooks_path_fails_closed_unless_the_parent_is_a_file(tmp_path: Path, monkeypatch):
    """Pointed at a directory, a missing path or a relative path, the check refuses: git must not run
    with a hooks path something could be planted under."""
    directory = tmp_path / "a dir"
    directory.mkdir()
    for bad in (directory, tmp_path / "missing", "relative/hosts"):
        with pytest.raises(HooksPathError):
            unplantable_hooks_path(parent=bad)
    regular = tmp_path / "a file"
    regular.write_text("", encoding="utf-8")
    assert unplantable_hooks_path(parent=regular).endswith(f"a file/{UNPLANTABLE_HOOKS_CHILD}")
    assert issubclass(HooksPathError, ProviderError)

    # the default parent too: a SystemRoot whose hosts is a directory (Windows), or a /dev/null
    # that is not a character device or file (POSIX), fails closed everywhere it is used
    fake_root = tmp_path / "fake windows"
    (fake_root / "System32" / "drivers" / "etc" / "hosts").mkdir(parents=True)
    monkeypatch.setenv("SYSTEMROOT", str(fake_root))
    if os.name != "nt":
        import software_factory.providers as providers_mod

        monkeypatch.setattr(providers_mod, "_unplantable_parent", lambda: str(fake_root / "System32" / "drivers" / "etc" / "hosts"))
    with pytest.raises(HooksPathError):
        unplantable_hooks_path()
    with pytest.raises(HooksPathError):
        git_no_hooks_args()
    with pytest.raises(ProviderError):  # no worker env (no worker) without disabled hooks
        worker_env(make("api_key", {"ANTHROPIC_API_KEY": "k"}))


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_worker_env_disables_a_planted_pre_commit_hook_with_real_git(tmp_path: Path):
    """A hook planted in ``.git/hooks`` fires for a plain commit but not for a commit made under the
    worker environment."""
    repo = init_git_repo(tmp_path / "repo")
    marker = tmp_path / "pre-commit.ran"
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\necho ran > '{marker.as_posix()}'\n", encoding="utf-8")
    hook.chmod(hook.stat().st_mode | 0o755)
    env = worker_env(make("api_key", {"ANTHROPIC_API_KEY": "k"}))
    git_env = {k: v for k, v in env.items() if k.startswith("GIT_")}

    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    run_git(["add", "-A"], repo)
    result = subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "under worker env"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60, env={**os.environ, **git_env}, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists(), "the planted pre-commit hook ran under the worker env"

    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    run_git(["add", "-A"], repo)
    run_git(["commit", "-q", "-m", "plain"], repo)
    assert marker.is_file(), "sanity: the hook is live on this platform without the worker env"


# ------------------------------------------------------------------------------- worker_env


def test_oauth_env():
    env = worker_env(make("oauth", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-1"}))
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-1"
    assert "ANTHROPIC_API_KEY" not in env
    for key, value in BASE_WORKER_ENV.items():
        assert env[key] == value
    assert env["GH_PROMPT_DISABLED"] == "1" and env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_api_key_env():
    env = worker_env(make("api_key", {"ANTHROPIC_API_KEY": "sk-ant-api"}))
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-api"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_bedrock_env_with_profile_and_passthrough():
    profile = make(
        "bedrock",
        {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "s3cr3t", "LINEAR_API_KEY": "never"},
        aws_profile="work",
        aws_region="us-east-1",
    )
    env = worker_env(profile)
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert env["AWS_REGION"] == "us-east-1"
    assert env["AWS_PROFILE"] == "work"
    assert env["AWS_ACCESS_KEY_ID"] == "AKIA" and env["AWS_SECRET_ACCESS_KEY"] == "s3cr3t"
    assert "LINEAR_API_KEY" not in env  # tracker secrets never reach workers
    assert "ANTHROPIC_API_KEY" not in env


def test_bedrock_env_without_aws_profile_relies_on_ambient_credentials():
    env = worker_env(make("bedrock", aws_region="eu-west-1"))
    assert env["AWS_REGION"] == "eu-west-1"
    assert "AWS_PROFILE" not in env and "AWS_ACCESS_KEY_ID" not in env


def test_vertex_env():
    env = worker_env(
        make("vertex", {"GOOGLE_APPLICATION_CREDENTIALS": "/c/sa.json"}, gcp_project="proj", gcp_region="us-east5")
    )
    assert env["CLAUDE_CODE_USE_VERTEX"] == "1"
    assert env["ANTHROPIC_VERTEX_PROJECT_ID"] == "proj"
    assert env["CLOUD_ML_REGION"] == "us-east5"
    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/c/sa.json"


def test_worker_env_never_reads_os_environ_for_secrets(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "from-environ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-environ")
    assert check_credentials(make("oauth")) != []
    with pytest.raises(ProviderError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        worker_env(make("oauth"))
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        worker_env(make("api_key"))


def test_worker_env_returns_a_fresh_dict_each_call():
    profile = make("api_key", {"ANTHROPIC_API_KEY": "k"})
    first = worker_env(profile)
    first["ANTHROPIC_API_KEY"] = "mutated"
    assert worker_env(profile)["ANTHROPIC_API_KEY"] == "k"
    assert BASE_WORKER_ENV["GH_PROMPT_DISABLED"] == "1"


# ------------------------------------------------------------------------ check_credentials


def test_check_credentials_ok_cases():
    assert check_credentials(make("oauth", {"CLAUDE_CODE_OAUTH_TOKEN": "t"})) == []
    assert check_credentials(make("api_key", {"ANTHROPIC_API_KEY": "k"})) == []
    assert check_credentials(make("bedrock", aws_region="us-east-1")) == []
    assert check_credentials(make("vertex", gcp_project="p", gcp_region="r")) == []


def test_check_credentials_reports_each_problem():
    assert any("CLAUDE_CODE_OAUTH_TOKEN" in p for p in check_credentials(make("oauth")))
    assert any("ANTHROPIC_API_KEY" in p for p in check_credentials(make("api_key")))
    assert any("aws_region" in p for p in check_credentials(make("bedrock")))
    vertex = check_credentials(make("vertex"))
    assert any("gcp_project" in p for p in vertex) and any("gcp_region" in p for p in vertex)
    assert len(vertex) == 2


def test_check_credentials_flags_oauth_on_employer_profile():
    # Profile validation already forbids this combination; bypass it to make sure doctor still says so.
    profile = make("api_key", {"CLAUDE_CODE_OAUTH_TOKEN": "t"}, employer=True)
    profile.provider.kind = "oauth"
    problems = check_credentials(profile)
    assert any("employer" in p and "oauth" in p for p in problems)
    with pytest.raises(ProviderError):
        worker_env(profile)


def test_empty_model_is_a_problem():
    assert any("provider.model" in p for p in check_credentials(make("api_key", {"ANTHROPIC_API_KEY": "k"}, model=" ")))


# -------------------------------------------------------------------------------- model ids


def test_model_id_uses_override_map():
    profile = make("bedrock", aws_region="r", model="claude-opus-5-5", model_ids={"claude-opus-5-5": "anthropic.opus"})
    assert model_id(profile) == "anthropic.opus"
    assert model_id(profile, "claude-opus-5-5") == "anthropic.opus"
    assert model_id(profile, "claude-sonnet-5") == "claude-sonnet-5"  # unmapped passes through


def test_fallback_model_id():
    assert fallback_model_id(make("api_key")) is None
    profile = make("api_key", fallback_model="claude-sonnet-5", model_ids={"claude-sonnet-5": "mapped"})
    assert fallback_model_id(profile) == "mapped"
    assert fallback_model_id(make("api_key", fallback_model="claude-sonnet-5")) == "claude-sonnet-5"
