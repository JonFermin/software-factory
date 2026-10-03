"""config.py: profile loading, isolation, employer rules, .env parsing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from software_factory import config
from software_factory.actions import Kind
from software_factory.config import (
    CONFIG_DIR_ENV,
    PROFILES_DIR_ENV,
    REPO_ROOT,
    ConfigError,
    RepoOverride,
    ReposConfig,
    config_root,
    list_profiles,
    load_profile,
    parse_dotenv,
    resolve_profiles_dir,
    user_profiles_dir,
)


@pytest.fixture(autouse=True)
def _no_real_user_config(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the config root at an empty tmp dir so no test here can pick up the developer's real
    ``<config root>/software-factory/profiles``."""
    root = tmp_path_factory.mktemp("config root")
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    return root

# ------------------------------------------------------------------------------- isolation


def test_two_profiles_side_by_side_never_cross(make_profile, load):
    """Loading A must not read B's toml (made invalid on purpose) nor B's .env (sentinel secret)."""
    make_profile("alpha", {"github": {"owners": ["Alpha"]}}, {"LINEAR_API_KEY": "lin_alpha"})
    make_profile("beta", toml="this is = not [valid toml", env={"LINEAR_API_KEY": "lin_beta", "BETA_ONLY": "1"})

    alpha = load("alpha")

    assert alpha.name == "alpha"
    assert alpha.github.owners == ["Alpha"]
    assert alpha.secrets == {"LINEAR_API_KEY": "lin_alpha"}
    assert "BETA_ONLY" not in alpha.secrets
    assert alpha.profile_dir is not None and alpha.profile_dir.name == "alpha"
    with pytest.raises(ConfigError):
        load("beta")  # beta really is broken; alpha loading fine proves it was never parsed


def test_secrets_never_land_in_os_environ(make_profile, load):
    make_profile("personal", env={"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-xyz", "SLACK_BOT_TOKEN": "xoxb-1"})
    before = dict(os.environ)

    profile = load("personal")

    assert profile.secrets["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-xyz"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ
    assert "SLACK_BOT_TOKEN" not in os.environ
    assert dict(os.environ) == before


def test_secrets_excluded_from_model_dump_and_repr(make_profile, load):
    make_profile("personal", env={"LINEAR_API_KEY": "lin_secret"})
    profile = load("personal")
    assert "secrets" not in profile.model_dump()
    assert "lin_secret" not in repr(profile)
    assert "lin_secret" not in profile.model_dump_json()


def test_profile_secret_accessor_treats_empty_as_missing(make_profile, load):
    make_profile("personal", env={"LINEAR_API_KEY": ""})
    profile = load("personal")
    assert profile.secret("LINEAR_API_KEY") is None
    assert profile.secret("NOPE") is None


# -------------------------------------------------------------------------------- employer


def test_employer_profile_rejects_oauth(make_profile, load):
    make_profile("work", {"profile": {"employer": True}, "provider": {"kind": "oauth"}})
    with pytest.raises(ConfigError, match="oauth"):
        load("work")


def test_employer_profile_defaults_to_observe(make_profile, load):
    make_profile("work", {"profile": {"employer": True}, "provider": {"kind": "api_key"}})
    profile = load("work")
    assert profile.mode == "observe"
    assert profile.employer is True


def test_employer_explicit_mode_is_respected(make_profile, load):
    make_profile("work", {"profile": {"employer": True, "mode": "full"}, "provider": {"kind": "bedrock"}})
    assert load("work").mode == "full"


def test_employer_requires_disclosure_footer(make_profile, load):
    make_profile(
        "work", {"profile": {"employer": True, "disclosure_footer": "  "}, "provider": {"kind": "api_key"}}
    )
    with pytest.raises(ConfigError, match="disclosure_footer"):
        load("work")


def test_personal_profile_defaults(make_profile, load):
    make_profile("personal")
    profile = load("personal")
    assert profile.mode == "full"
    assert profile.employer is False
    assert profile.enabled_kinds == [Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV]
    assert profile.provider.kind == "oauth"
    assert profile.max_attempts == 3 and profile.max_parallel == 3
    assert profile.kind_enabled(Kind.CI) and not profile.kind_enabled(Kind.BUILD)
    assert profile.disclosure_footer.startswith("---")


# ------------------------------------------------------------------------- names and lookup


@pytest.mark.parametrize("name", ["", "../personal", "a/b", "a\\b", "..", "x/../y"])
def test_bad_profile_names_rejected(name, profiles_dir):
    with pytest.raises(ConfigError, match="invalid profile name"):
        load_profile(name, profiles_dir)


def test_missing_profile_is_a_config_error(profiles_dir):
    with pytest.raises(ConfigError, match="not found"):
        load_profile("ghost", profiles_dir)


def test_name_mismatch_with_directory_rejected(make_profile, load):
    make_profile("personal", toml='[profile]\nname = "other"\n')
    with pytest.raises(ConfigError, match="does not match directory"):
        load("personal")


def test_name_defaults_to_directory_when_omitted(make_profile, profiles_dir):
    make_profile("personal", toml="[provider]\nkind = \"api_key\"\n")
    assert load_profile("personal", profiles_dir).name == "personal"


def test_invalid_toml_and_invalid_values_are_config_errors(make_profile, load):
    make_profile("personal", toml="[profile\n")
    with pytest.raises(ConfigError):
        load("personal")
    make_profile("personal", {"profile": {"enabled_kinds": ["ci", "bogus"]}})
    with pytest.raises(ConfigError, match="enabled_kinds"):
        load("personal")
    make_profile("personal", {"profile": {"mode": "yolo"}})
    with pytest.raises(ConfigError, match="mode"):
        load("personal")


@pytest.mark.parametrize(
    ("schedule", "field"),
    [
        ({"digest_time": "9:00"}, "digest_time"),
        ({"digest_time": "25:00"}, "digest_time"),
        ({"digest_time": "09:60"}, "digest_time"),
        ({"interval_minutes": 0}, "interval_minutes"),
        ({"interval_minutes": 1440}, "interval_minutes"),
        ({"interval_minutes": -5}, "interval_minutes"),
    ],
)
def test_schedule_values_are_validated_at_load(make_profile, load, schedule, field):
    """Regression: an invalid [schedule] value loaded fine and then crashed `doctor` / `schedule show`
    with a pydantic traceback out of ScheduleSpec. It is a ConfigError at load time now."""
    make_profile("personal", {"schedule": schedule})
    with pytest.raises(ConfigError, match=field):
        load("personal")


def test_schedule_valid_values_load(make_profile, load):
    make_profile("personal", {"schedule": {"digest_time": "23:59", "interval_minutes": 1439, "digest_weekday": "FRI"}})
    profile = load("personal")
    assert profile.schedule.digest_time == "23:59" and profile.schedule.interval_minutes == 1439
    make_profile("personal", {"schedule": {"digest_time": "00:00", "interval_minutes": 1}})
    assert load("personal").schedule.interval_minutes == 1


def test_top_level_keys_outside_profile_table_are_rejected_only_if_unknown_models(make_profile, load):
    # Sub-model tables map by name; [profile] scalars are flattened.
    make_profile(
        "personal",
        {
            "profile": {"max_parallel": 5, "enabled_kinds": ["rv"]},
            "provider": {"kind": "api_key", "model": "claude-opus-5-5", "model_ids": {"claude-opus-5-5": "x"}},
            "repos": {"root": "~/Code", "overrides": {"game": {"visual": True, "dev_port": 5174}}},
        },
        {"ANTHROPIC_API_KEY": "k"},
    )
    profile = load("personal")
    assert profile.max_parallel == 5
    assert profile.enabled_kinds == [Kind.RV]
    assert profile.provider.model_ids == {"claude-opus-5-5": "x"}
    assert profile.repos.overrides["game"].dev_port == 5174
    assert profile.repos.root_path == Path("~/Code").expanduser()


def test_profiles_dir_resolution_order(tmp_path, monkeypatch, _no_real_user_config):
    """--profiles-dir, then $SOFTWARE_FACTORY_PROFILES_DIR, then the user config dir when it exists,
    then <repo>/profiles."""
    explicit = tmp_path / "explicit"
    env_dir = tmp_path / "from env"
    user_dir = _no_real_user_config / "software-factory" / "profiles"
    user_dir.mkdir(parents=True)
    monkeypatch.setenv(PROFILES_DIR_ENV, str(env_dir))
    assert resolve_profiles_dir(explicit) == explicit.resolve()
    assert resolve_profiles_dir(None) == env_dir.resolve()
    monkeypatch.delenv(PROFILES_DIR_ENV)
    assert user_profiles_dir() == user_dir
    assert resolve_profiles_dir(None) == user_dir.resolve()
    user_dir.rmdir()
    assert resolve_profiles_dir(None) == REPO_ROOT / "profiles"


def test_user_profiles_dir_that_is_a_file_is_not_used(_no_real_user_config):
    user_dir = _no_real_user_config / "software-factory" / "profiles"
    user_dir.parent.mkdir(parents=True)
    user_dir.write_text("not a directory", encoding="utf-8")
    assert resolve_profiles_dir(None) == REPO_ROOT / "profiles"


def test_profiles_in_the_user_config_dir_load_without_flags(make_profile, _no_real_user_config):
    """The real profiles live outside the repo: with no --profiles-dir and no env var, load_profile and
    list_profiles find them in <config root>/software-factory/profiles."""
    user_dir = _no_real_user_config / "software-factory" / "profiles"
    make_profile("personal", {"github": {"owners": ["someone"]}}, {"LINEAR_API_KEY": "lin_user"}, root=user_dir)
    profile = load_profile("personal")
    assert profile.github.owners == ["someone"]
    assert profile.secrets == {"LINEAR_API_KEY": "lin_user"}
    assert profile.profile_dir == user_dir.resolve() / "personal"
    assert list_profiles() == ["personal"]


def test_config_root_follows_the_platform_convention(tmp_path, monkeypatch):
    monkeypatch.delenv(CONFIG_DIR_ENV)
    home = tmp_path / "home dir"
    monkeypatch.setenv("HOME", str(home))  # Path.home() on POSIX
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(config.sys, "platform", "win32")
    # %APPDATA% is ignored on purpose: packaged (MSIX) apps virtualize it (see paths.data_root).
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert config_root() == home / ".config"
    assert user_profiles_dir() == home / ".config" / "software-factory" / "profiles"

    monkeypatch.setattr(config.sys, "platform", "darwin")
    assert config_root() == home / "Library" / "Application Support"

    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config_root() == tmp_path / "xdg"
    assert user_profiles_dir() == tmp_path / "xdg" / "software-factory" / "profiles"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert config_root() == home / ".config"


def test_config_dir_env_overrides_every_platform(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "my config"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for platform in ("win32", "darwin", "linux"):
        monkeypatch.setattr(config.sys, "platform", platform)
        assert config_root() == tmp_path / "my config"
        assert user_profiles_dir() == tmp_path / "my config" / "software-factory" / "profiles"


def test_default_repos_root_is_generic():
    assert ReposConfig().root == "~/code"
    assert ReposConfig().root_path == Path("~/code").expanduser()


def test_list_profiles_only_counts_dirs_with_factory_toml(make_profile, profiles_dir):
    make_profile("zed")
    make_profile("alpha")
    (profiles_dir / "junk").mkdir()
    (profiles_dir / "stray.txt").write_text("x", encoding="utf-8")
    assert list_profiles(profiles_dir) == ["alpha", "zed"]
    assert list_profiles(profiles_dir / "missing") == []


def test_override_lookup_order_slug_then_name_then_dirname():
    cfg = ReposConfig(
        overrides={
            "someone/arcade": RepoOverride(dev_port=1),
            "arcade": RepoOverride(dev_port=2),
            "arcade-web": RepoOverride(dev_port=3),
        }
    )
    assert cfg.override_for("someone/arcade", "arcade-web").dev_port == 1
    assert cfg.override_for("Other/arcade", "arcade-web").dev_port == 2
    assert cfg.override_for("Other/other", "arcade-web").dev_port == 3
    assert cfg.override_for("Other/other", "nothing") == RepoOverride()


# ----------------------------------------------------------------------------------- dotenv


def test_parse_dotenv_edge_cases(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "PLAIN=value",
                "  SPACED  =  padded  ",
                'DQ="quoted # not a comment"',
                "SQ='single quoted'",
                "export EXPORTED=yes",
                "INLINE=value # trailing comment",
                "URLISH=https://x.y/z?a=b=c",
                "EMPTY=",
                "NOEQUALS",
                "HASH_NO_SPACE=abc#def",
                "CRLF=windows\r",
                "LAST=1",
            ]
        ),
        encoding="utf-8",
    )
    values = parse_dotenv(env)
    assert values == {
        "PLAIN": "value",
        "SPACED": "padded",
        "DQ": "quoted # not a comment",
        "SQ": "single quoted",
        "EXPORTED": "yes",
        "INLINE": "value",
        "URLISH": "https://x.y/z?a=b=c",
        "EMPTY": "",
        "HASH_NO_SPACE": "abc#def",
        "CRLF": "windows",
        "LAST": "1",
    }


def test_parse_dotenv_missing_file_is_empty(tmp_path):
    assert parse_dotenv(tmp_path / "nope.env") == {}


def test_profile_without_dotenv_has_no_secrets(make_profile, load):
    make_profile("personal")
    assert load("personal").secrets == {}
