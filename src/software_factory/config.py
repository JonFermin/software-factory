"""Profiles: ``<profiles_dir>/<name>/factory.toml`` + ``<profiles_dir>/<name>/.env``.

A tick only ever sees one profile. Secrets from ``.env`` are parsed into ``Profile.secrets`` and are
**never** written into ``os.environ`` -- workers therefore cannot read Linear/Slack credentials, and
two profiles can never see each other's secrets.

Profiles dir resolution (first hit wins): explicit argument / ``--profiles-dir``, env
``SOFTWARE_FACTORY_PROFILES_DIR``, the user config dir ``<config root>/software-factory/profiles``
when it exists (``user_profiles_dir()``), else ``<repo root>/profiles`` (which only ships
``example/``). Real profiles live outside the repo; doctor warns when an ``employer = true`` profile
sits inside a git checkout.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .actions import Kind
from .paths import APP_DIRNAME

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR_ENV = "SOFTWARE_FACTORY_PROFILES_DIR"
CONFIG_DIR_ENV = "SOFTWARE_FACTORY_CONFIG_DIR"
DEFAULT_PROFILE = "personal"
DEFAULT_DISCLOSURE = (
    "---\n"
    "_This pull request was prepared by an AI coding agent (software-factory, Claude) and is a draft "
    "for human review._"
)
UPKEEP_KINDS = [Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV]


class ConfigError(RuntimeError):
    pass


class ProviderConfig(BaseModel):
    kind: Literal["oauth", "api_key", "bedrock", "vertex"] = "oauth"
    model: str = "claude-opus-5-5"
    fallback_model: str | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_turns: int = 200
    # Claude Code CLI the Agent SDK should launch. Empty = the SDK's bundled CLI, else `claude` on
    # PATH. Windows SDK wheels ship no bundled CLI, so the PATH one must be new enough for `model`.
    cli_path: str | None = None
    # Per-provider model id overrides, e.g. {"claude-opus-5-5": "anthropic.claude-opus-5-5"} on Bedrock.
    model_ids: dict[str, str] = Field(default_factory=dict)
    aws_profile: str | None = None
    aws_region: str | None = None
    gcp_project: str | None = None
    gcp_region: str | None = None


class GitHubConfig(BaseModel):
    login: str = ""  # my GitHub login; empty = detect via `gh api user`
    owners: list[str] = Field(default_factory=list)  # owners whose repos are managed
    ignored_logins: list[str] = Field(default_factory=lambda: ["dependabot[bot]", "github-actions[bot]"])
    bot_reviewers: list[str] = Field(
        default_factory=lambda: ["cursor[bot]", "copilot-pull-request-reviewer[bot]", "coderabbitai[bot]"]
    )


class RepoOverride(BaseModel):
    visual: bool = False
    dev_command: str | None = None
    dev_port: int | None = None
    dev_routes: list[str] = Field(default_factory=lambda: ["/"])
    setup: list[str] | None = None  # overrides discovered setup commands
    verify: list[str] | None = None  # overrides discovered verification commands
    sensitive_globs: list[str] = Field(default_factory=list)  # added to policy.sensitive_globs
    min_readiness: int | None = None
    default_branch: str | None = None


class ReposConfig(BaseModel):
    root: str = "~/code"
    include: list[str] = Field(default_factory=list)  # dir names or owner/name; empty = auto-discover
    exclude: list[str] = Field(default_factory=list)
    overrides: dict[str, RepoOverride] = Field(default_factory=dict)  # keyed by dir name or owner/name

    @property
    def root_path(self) -> Path:
        return Path(os.path.expandvars(self.root)).expanduser()

    def override_for(self, slug: str, dirname: str | None = None) -> RepoOverride:
        for key in (slug, slug.split("/")[-1], dirname or ""):
            if key and key in self.overrides:
                return self.overrides[key]
        return RepoOverride()


class LinearConfig(BaseModel):
    enabled: bool = True
    team_keys: list[str] = Field(default_factory=list)  # empty = all teams
    label_refine: str = "factory:refine"
    label_plan: str = "factory:plan"
    label_build: str = "factory:build"
    repo_label_prefix: str = "repo:"

    @property
    def factory_labels(self) -> list[str]:
        return [self.label_refine, self.label_plan, self.label_build]


class ResolveConfig(BaseModel):
    min_confidence: float = 0.7
    sibling_min_votes: int = 2


class PolicyConfig(BaseModel):
    sensitive_globs: list[str] = Field(
        default_factory=lambda: [
            "**/auth/**",
            "**/authentication/**",
            "**/security/**",
            "**/billing/**",
            "**/payments/**",
            "**/migrations/**",
            "**/migrate/**",
            "**/*.pem",
            "**/*.key",
            "**/.env*",
            ".github/workflows/**",
            "**/CODEOWNERS",
        ]
    )
    my_teams: list[str] = Field(default_factory=list)  # CODEOWNERS teams that count as "mine"
    allow_web_tools: bool = False  # WebFetch / WebSearch for workers
    extra_allowed_tools: list[str] = Field(default_factory=list)
    min_readiness: int = 60


class NotifyConfig(BaseModel):
    desktop: bool = True
    slack: bool = True


_DIGEST_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
MAX_INTERVAL_MINUTES = 1439  # the scheduler backends take a daily-or-shorter interval


class ScheduleConfig(BaseModel):
    """Validated here (not only in ``scheduler.ScheduleSpec``) so a typo is a ``ConfigError`` from
    ``load_profile`` instead of a pydantic traceback out of ``doctor`` / ``schedule show``."""

    interval_minutes: int = 20
    digest_weekday: Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"] = "MON"
    digest_time: str = "09:00"

    @field_validator("interval_minutes")
    @classmethod
    def _interval_range(cls, value: int) -> int:
        if not 1 <= value <= MAX_INTERVAL_MINUTES:
            raise ValueError(f"schedule.interval_minutes must be between 1 and {MAX_INTERVAL_MINUTES}, got {value}")
        return value

    @field_validator("digest_time")
    @classmethod
    def _digest_time_format(cls, value: str) -> str:
        if not _DIGEST_TIME_RE.match(value.strip()):
            raise ValueError(f"schedule.digest_time must be HH:MM in 24-hour form (e.g. 09:00), got {value!r}")
        return value.strip()


class Profile(BaseModel):
    name: str
    employer: bool = False
    mode: Literal["observe", "full"] = "full"
    enabled_kinds: list[Kind] = Field(default_factory=lambda: list(UPKEEP_KINDS))
    max_parallel: int = 3
    max_attempts: int = 3
    session_timeout_min: int = 30
    authored_max_age_days: int = 30
    review_max_age_days: int = 7
    disclosure_footer: str = DEFAULT_DISCLOSURE

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    repos: ReposConfig = Field(default_factory=ReposConfig)
    linear: LinearConfig = Field(default_factory=LinearConfig)
    resolve: ResolveConfig = Field(default_factory=ResolveConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)

    # Not from TOML: populated by the loader.
    secrets: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)
    profile_dir: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _employer_rules(self) -> Profile:
        if self.employer:
            if self.provider.kind == "oauth":
                raise ValueError(
                    "employer profiles must not use provider.kind = 'oauth' (subscription token); "
                    "use api_key, bedrock or vertex"
                )
            if not self.disclosure_footer.strip():
                raise ValueError("employer profiles require a non-empty disclosure_footer")
        return self

    def kind_enabled(self, kind: Kind) -> bool:
        return kind in self.enabled_kinds

    def secret(self, key: str) -> str | None:
        value = self.secrets.get(key)
        return value if value else None


def config_root() -> Path:
    """Platform config directory: ``~/Library/Application Support`` on macOS, else
    ``$XDG_CONFIG_HOME`` or ``~/.config`` (Windows included; see ``paths.data_root`` for why not
    ``%APPDATA%``). ``SOFTWARE_FACTORY_CONFIG_DIR`` overrides all of them."""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return Path(xdg) if xdg else Path.home() / ".config"


def user_profiles_dir() -> Path:
    """Per-user profiles dir, ``<config root>/software-factory/profiles`` (it may not exist)."""
    return config_root() / APP_DIRNAME / "profiles"


def resolve_profiles_dir(explicit: str | Path | None = None) -> Path:
    """First hit wins: ``explicit`` (``--profiles-dir``), ``$SOFTWARE_FACTORY_PROFILES_DIR``,
    ``user_profiles_dir()`` when it is a directory, else ``<repo root>/profiles``."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(PROFILES_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    user = user_profiles_dir()
    if user.is_dir():
        return user.resolve()
    return REPO_ROOT / "profiles"


def parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, optional quotes, ``#`` comments, optional ``export``."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_profile(name: str = DEFAULT_PROFILE, profiles_dir: str | Path | None = None) -> Profile:
    """Load exactly one profile. Reads only ``<profiles_dir>/<name>/``."""
    if not name or any(sep in name for sep in ("/", "\\", "..")):
        raise ConfigError(f"invalid profile name: {name!r}")
    root = resolve_profiles_dir(profiles_dir)
    pdir = root / name
    toml_path = pdir / "factory.toml"
    if not toml_path.is_file():
        raise ConfigError(f"profile {name!r} not found: {toml_path} does not exist")
    try:
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{toml_path}: {exc}") from exc

    # The [profile] table holds top-level scalar settings; other tables map to sub-models.
    flat: dict = dict(data.pop("profile", {}))
    flat.update(data)
    flat.setdefault("name", name)
    if flat["name"] != name:
        raise ConfigError(f"{toml_path}: profile.name {flat['name']!r} does not match directory {name!r}")
    # Employer profiles default to observe unless mode is explicitly set.
    if flat.get("employer") and "mode" not in flat:
        flat["mode"] = "observe"
    try:
        profile = Profile.model_validate(flat)
    except ValidationError as exc:
        raise ConfigError(f"{toml_path}: {exc}") from exc
    profile.secrets = parse_dotenv(pdir / ".env")
    profile.profile_dir = pdir
    return profile


def list_profiles(profiles_dir: str | Path | None = None) -> list[str]:
    root = resolve_profiles_dir(profiles_dir)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "factory.toml").is_file())
