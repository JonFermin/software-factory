"""Model providers: turn a profile's ``provider.*`` + secrets into the worker environment.

Secrets come only from ``Profile.secrets`` (parsed from the profile's ``.env``), never from
``os.environ``; the Claude Agent SDK merges ``os.environ`` into every worker, so inherited credential
variables are scrubbed first (``scrub_process_env``). Bedrock / Vertex may rely on ambient AWS / GCP
credentials (profiles, instance roles, ADC) -- those are not Anthropic secrets and are left alone.

GitHub credentials are handled per worker rather than by scrubbing: the engine itself needs
``GH_TOKEN`` / ``GITHUB_TOKEN`` when that is how ``gh`` is authenticated, so they stay in the
process; the worker env overrides them (``GH_TOKEN`` = the optional read-only ``GH_WORKER_TOKEN``
from the profile ``.env``, else empty = unset for gh) and points ``GH_CONFIG_DIR`` at a regular
*file*, so the user's ``hosts.yml`` / keyring login is unreachable even from a script the worker
writes (gh falls back to the keyring as soon as a ``hosts.yml`` names the host, and a file cannot
receive one). Pushes are blocked at the git level too: ``url.<x>.pushInsteadOf`` rewrites every
GitHub push URL (by remote name or explicit URL) to an unusable protocol while fetches keep working,
and ``core.hooksPath`` points at ``unplantable_hooks_path()`` so a hook a worker plants in the
clone's shared ``.git/hooks`` (``echo 'git push' > .git/hooks/pre-commit``, which the tool policy
also denies) never runs when the worker commits. Both travel as ``GIT_CONFIG_COUNT`` entries, which
every git in the worker's process tree honours (and which the policy forbids workers to override).

``unplantable_hooks_path()`` is also the ``core.hooksPath`` of every git the engine itself runs in a
clone or worktree (worktree setup, fetches, pushes): a path *under a regular file* (Windows: the
hosts file) or a character device (POSIX: ``/dev/null``). Git treats it as "no hooks" (exit 0), and
nothing can ever be created there, so unlike an empty directory under the profile root there is no
window between "checked empty" and "git ran" in which another process could plant a hook, and
nothing for the engine to empty (emptying followed junctions into their targets).
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from .config import Profile
from .paths import ProfilePaths

# Inherited variables that must not leak into workers (a worker must authenticate only with the
# profile's own credentials, and must never see tracker / chat tokens). GH_TOKEN / GITHUB_TOKEN are
# deliberately absent: the engine's own ``gh`` may depend on them (doctor and tick must agree);
# ``gh_isolation_env`` overrides them for workers instead.
SCRUB_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_SCOPES",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
    "CLAUDE_CODE_SKIP_VERTEX_AUTH",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SSE_PORT",
    "LINEAR_API_KEY",
    "SLACK_BOT_TOKEN",
    "SLACK_USER_ID",
)

# Always set for workers: no telemetry/update checks, no interactive prompts (the headless failure mode).
BASE_WORKER_ENV: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "GH_PROMPT_DISABLED": "1",
    "GH_NO_UPDATE_NOTIFIER": "1",
    "GIT_TERMINAL_PROMPT": "0",
}

# Optional profile secret: a fine-grained read-only GitHub token for workers (contents / pull
# requests / actions: read). Without it workers have no gh authentication at all (reads fail with
# "gh auth login"; the prompt context already carries what they need).
GH_WORKER_TOKEN_KEY = "GH_WORKER_TOKEN"
GH_NOAUTH_CONFIG_NAME = "gh-noauth"
# Last component of ``unplantable_hooks_path()``; its parent is never a directory.
UNPLANTABLE_HOOKS_CHILD = "no-hooks"
# Every push URL with one of these prefixes is rewritten to this (non-existent) transport.
NO_PUSH_PROTOCOL = "factory-no-push://"
NO_PUSH_URL_PREFIXES: tuple[str, ...] = (
    "git@github.com:",
    "https://github.com/",
    "ssh://git@github.com/",
    "git://github.com/",
    "http://github.com/",
)

# Secrets forwarded verbatim when the profile's .env defines them (ambient creds work without them).
_AWS_PASSTHROUGH = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK")
_GCP_PASSTHROUGH = ("GOOGLE_APPLICATION_CREDENTIALS",)


class ProviderError(RuntimeError):
    pass


class HooksPathError(ProviderError):
    """``unplantable_hooks_path()`` is not available on this host: git must not run (fail closed)."""


def scrub_process_env() -> None:
    """Pop ``SCRUB_ENV_VARS`` from ``os.environ`` (the SDK merges ``os.environ`` into workers)."""
    for name in SCRUB_ENV_VARS:
        os.environ.pop(name, None)


def model_id(profile: Profile, logical: str | None = None) -> str:
    """Provider-specific id for a logical model name (``provider.model_ids`` override, else as-is)."""
    name = logical or profile.provider.model
    return profile.provider.model_ids.get(name, name)


def fallback_model_id(profile: Profile) -> str | None:
    fallback = profile.provider.fallback_model
    return model_id(profile, fallback) if fallback else None


def check_credentials(profile: Profile) -> list[str]:
    """Human-readable problems with the profile's provider configuration; ``[]`` means ok."""
    provider = profile.provider
    problems: list[str] = []
    if provider.kind == "oauth":
        if profile.employer:
            problems.append("employer profiles must not use provider.kind = 'oauth'")
        if not profile.secret("CLAUDE_CODE_OAUTH_TOKEN"):
            problems.append(
                "CLAUDE_CODE_OAUTH_TOKEN missing from the profile .env "
                "(run `claude setup-token` and paste the token)"
            )
    elif provider.kind == "api_key":
        if not profile.secret("ANTHROPIC_API_KEY"):
            problems.append("ANTHROPIC_API_KEY missing from the profile .env")
    elif provider.kind == "bedrock":
        if not provider.aws_region:
            problems.append("provider.aws_region is required for provider.kind = 'bedrock'")
    elif provider.kind == "vertex":
        if not provider.gcp_project:
            problems.append("provider.gcp_project is required for provider.kind = 'vertex'")
        if not provider.gcp_region:
            problems.append("provider.gcp_region is required for provider.kind = 'vertex'")
    else:  # pragma: no cover - pydantic Literal already rejects this
        problems.append(f"unknown provider.kind {provider.kind!r}")
    if not provider.model.strip():
        problems.append("provider.model must not be empty")
    return problems


def gh_noauth_config_path(profile: Profile) -> Path:
    """A regular file under the profile's data dir used as the workers' ``GH_CONFIG_DIR``.

    gh treats a missing ``hosts.yml`` as "not logged in" and never consults the keyring for a
    host that ``hosts.yml`` does not name; because the path is a file, nothing (not even a script
    the worker writes) can drop a ``hosts.yml`` into it. A directory found there is replaced.
    """
    path = ProfilePaths.for_profile(profile.name).root / GH_NOAUTH_CONFIG_NAME
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return path


def gh_isolation_env(profile: Profile) -> dict[str, str]:
    """``GH_CONFIG_DIR`` -> the no-auth file; ``GH_TOKEN`` -> the read-only worker token or empty
    (gh treats an empty value as unset); ``GITHUB_TOKEN`` emptied so an inherited one never applies."""
    token = profile.secret(GH_WORKER_TOKEN_KEY) or ""
    return {"GH_CONFIG_DIR": str(gh_noauth_config_path(profile)), "GH_TOKEN": token, "GITHUB_TOKEN": ""}


def _unplantable_parent() -> str:
    """Windows: ``%SystemRoot%/System32/drivers/etc/hosts`` (a regular file only an administrator can
    replace); elsewhere ``/dev/null`` (a character device)."""
    if os.name == "nt":
        root = os.environ.get("SYSTEMROOT") or r"C:\Windows"  # os.environ is case-insensitive on Windows
        return os.path.join(root, "System32", "drivers", "etc", "hosts")
    return "/dev/null"


def unplantable_hooks_path(*, parent: str | os.PathLike[str] | None = None) -> str:
    """The ``core.hooksPath`` that disables every git hook and cannot be planted into.

    ``<parent>/no-hooks`` where ``parent`` (default: the hosts file on Windows, ``/dev/null``
    elsewhere) exists and is a regular file or a character device: git finds no hooks there and
    exits 0, and creating anything under a file fails for every process, so -- unlike an empty
    directory -- there is no race with a worker (or a background loop it left running) re-planting a
    ``pre-push`` between a check and git's exec. Not ``NUL`` (a real ``NUL`` directory can be
    created through the extended-length path prefix) and not ``/dev/null`` on Windows (a non-MSYS
    process resolves it under the Git install). Returned with forward slashes (git config values).

    Raises ``HooksPathError`` (fail closed) when ``parent`` is relative, missing, a directory or
    anything but a regular file / character device: git must then not run at all.
    """
    base = os.fspath(parent) if parent is not None else _unplantable_parent()
    if not os.path.isabs(base):
        raise HooksPathError(f"cannot disable git hooks: {base!r} is not an absolute path")
    try:
        mode = os.stat(base).st_mode
    except OSError as exc:
        raise HooksPathError(f"cannot disable git hooks: {base} is not accessible ({exc})") from exc
    if stat.S_ISDIR(mode) or not (stat.S_ISREG(mode) or stat.S_ISCHR(mode)):
        raise HooksPathError(
            f"cannot disable git hooks: {base} must be a regular file or a character device (a directory "
            "could receive a planted hook)"
        )
    path = os.path.join(base, UNPLANTABLE_HOOKS_CHILD)
    return path.replace("\\", "/") if os.name == "nt" else path


def git_config_env(entries: list[tuple[str, str]]) -> dict[str, str]:
    """``GIT_CONFIG_COUNT`` / ``GIT_CONFIG_KEY_i`` / ``GIT_CONFIG_VALUE_i`` for ``entries`` in order."""
    env = {"GIT_CONFIG_COUNT": str(len(entries))}
    for i, (key, value) in enumerate(entries):
        env[f"GIT_CONFIG_KEY_{i}"] = key
        env[f"GIT_CONFIG_VALUE_{i}"] = value
    return env


def git_no_push_entries() -> list[tuple[str, str]]:
    """``url.<no-push>.pushInsteadOf = <github prefix>`` for every GitHub URL form."""
    return [(f"url.{NO_PUSH_PROTOCOL}.pushInsteadOf", prefix) for prefix in NO_PUSH_URL_PREFIXES]


def git_no_hooks_entries(hooks_path: str | None = None) -> list[tuple[str, str]]:
    """``core.hooksPath = <hooks_path>`` (default: ``unplantable_hooks_path()``)."""
    return [("core.hooksPath", hooks_path if hooks_path is not None else unplantable_hooks_path())]


def git_no_hooks_args() -> list[str]:
    """``["-c", "core.hooksPath=<unplantable>"]`` for an engine git command (checked on every call;
    ``HooksPathError`` when hooks cannot be disabled)."""
    return ["-c", f"core.hooksPath={unplantable_hooks_path()}"]


def git_no_push_env() -> dict[str, str]:
    """``GIT_CONFIG_COUNT`` config that makes every GitHub push URL unusable (fetch unaffected).

    Applies to ``git push origin ...`` and ``git push <url> ...`` alike, from any process in the
    worker's tree, so a scripted push fails even though the engine's own push (which runs outside
    this env) still works.
    """
    return git_config_env(git_no_push_entries())


def worker_git_config_env(profile: Profile) -> dict[str, str]:
    """The worker's whole ``GIT_CONFIG_*`` block: the push block first, then ``core.hooksPath`` =
    ``unplantable_hooks_path()`` (the same for every profile and for the engine: nothing can be
    planted there, so the worker knowing the path gives it nothing)."""
    del profile  # kept for the contract; the block no longer depends on the profile
    return git_config_env([*git_no_push_entries(), *git_no_hooks_entries()])


def worker_env(profile: Profile) -> dict[str, str]:
    """Environment passed to every worker session for this profile.

    Raises ``ProviderError`` when the provider cannot be configured (``check_credentials`` reports
    the same problems without raising; the dispatcher runs it in the precheck).
    """
    problems = check_credentials(profile)
    if problems:
        raise ProviderError("; ".join(problems))
    provider = profile.provider
    env = dict(BASE_WORKER_ENV)
    env.update(gh_isolation_env(profile))
    env.update(worker_git_config_env(profile))
    if provider.kind == "oauth":
        env["CLAUDE_CODE_OAUTH_TOKEN"] = profile.secrets["CLAUDE_CODE_OAUTH_TOKEN"]
    elif provider.kind == "api_key":
        env["ANTHROPIC_API_KEY"] = profile.secrets["ANTHROPIC_API_KEY"]
    elif provider.kind == "bedrock":
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        env["AWS_REGION"] = provider.aws_region or ""
        if provider.aws_profile:
            env["AWS_PROFILE"] = provider.aws_profile
        env.update(_passthrough(profile, _AWS_PASSTHROUGH))
    elif provider.kind == "vertex":
        env["CLAUDE_CODE_USE_VERTEX"] = "1"
        env["ANTHROPIC_VERTEX_PROJECT_ID"] = provider.gcp_project or ""
        env["CLOUD_ML_REGION"] = provider.gcp_region or ""
        env.update(_passthrough(profile, _GCP_PASSTHROUGH))
    return env


def _passthrough(profile: Profile, keys: tuple[str, ...]) -> dict[str, str]:
    return {k: v for k in keys if (v := profile.secret(k))}
