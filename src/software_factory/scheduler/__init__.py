"""OS-agnostic ``factory schedule install|uninstall|show``.

Two jobs per profile: ``tick`` every ``interval_minutes`` and a weekly digest
(``factory report --digest --since 7d``). Each job runs
``<uv> run --project <repo root> factory --profile <name> <command>`` with
``SOFTWARE_FACTORY_PROFILES_DIR`` set when the profile lives outside the default profiles dir and
``SOFTWARE_FACTORY_DATA_DIR`` set when it was set in the shell that ran ``install`` (scheduled jobs
inherit no shell environment; without it a scheduled tick would use a different state.db, lock and
logs than the interactive CLI).

Backends (``windows.py`` -> schtasks + ``.cmd`` wrapper, ``macos.py`` -> launchd plist,
``linux.py`` -> crontab block) expose pure ``render_*`` functions plus thin ``install`` /
``uninstall`` / ``show`` that take a :class:`ScheduleSpec` and an injectable :class:`Runner`, so
tests never touch the real scheduler.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from ..config import REPO_ROOT, Profile, resolve_profiles_dir
from ..paths import DATA_DIR_ENV

if TYPE_CHECKING:
    from collections.abc import Sequence

OsName = Literal["windows", "macos", "linux"]
JobName = Literal["tick", "digest"]
Weekday = Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

JOBS: tuple[JobName, ...] = ("tick", "digest")
LOG_FILENAME = "scheduler.log"
DIGEST_SINCE = "7d"
PROFILES_DIR_ENV = "SOFTWARE_FACTORY_PROFILES_DIR"
# Appears in `schtasks /Query /V` output (English locale) when a task keeps the schtasks default of
# not starting on battery; `windows.install` relaxes that, doctor warns when it is still present.
POWER_RESTRICTED_MARKER = "no start on batteries"
_TIME_RE = re.compile(r"^(?P<h>[01]\d|2[0-3]):(?P<m>[0-5]\d)$")


class SchedulerError(RuntimeError):
    pass


class SchedulerPaths(Protocol):
    """The subset of ``paths.ProfilePaths`` the scheduler needs (tests pass a namespace)."""

    scripts: Path
    logs: Path


# ------------------------------------------------------------------------------------ runner
class RunResult(BaseModel):
    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        return (self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr).strip()


class Runner(Protocol):
    def __call__(self, argv: Sequence[str], *, input: str | None = None) -> RunResult: ...


def default_runner(argv: Sequence[str], *, input: str | None = None, timeout: float = 60.0) -> RunResult:
    """Run a scheduler CLI (schtasks / launchctl / crontab). Never ``shell=True``."""
    args = [str(a) for a in argv]
    try:
        proc = subprocess.run(
            args,
            input=input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SchedulerError(f"{args[0]} not found on PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SchedulerError(f"{args[0]} timed out after {timeout:.0f}s") from exc
    return RunResult(argv=args, returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


# -------------------------------------------------------------------------------------- spec
class ScheduleSpec(BaseModel):
    """Everything a backend needs, resolved once at install time (uv path embedded absolute)."""

    profile: str
    repo_root: Path  # the software-factory checkout (`uv run --project`)
    uv_path: Path  # absolute, from shutil.which at install time
    profiles_dir: Path | None = None  # only set when non-default -> SOFTWARE_FACTORY_PROFILES_DIR
    data_dir: Path | None = None  # only set when SOFTWARE_FACTORY_DATA_DIR was set at install time
    scripts_dir: Path  # paths.scripts
    logs_dir: Path  # paths.logs
    interval_minutes: int = 20
    digest: bool = True
    digest_weekday: Weekday = "MON"
    digest_time: str = "09:00"

    @field_validator("interval_minutes")
    @classmethod
    def _interval_range(cls, value: int) -> int:
        if not 1 <= value <= 1439:
            raise ValueError(f"interval_minutes must be 1..1439, got {value}")
        return value

    @field_validator("digest_time")
    @classmethod
    def _time_format(cls, value: str) -> str:
        if not _TIME_RE.match(value):
            raise ValueError(f"digest_time must be HH:MM (24h), got {value!r}")
        return value

    # names ---------------------------------------------------------------------------
    @property
    def task_prefix(self) -> str:
        return f"software-factory-{self.profile}"

    def task_name(self, job: JobName) -> str:
        """Windows task name / generic job id: ``software-factory-<profile>-<job>``."""
        return f"{self.task_prefix}-{job}"

    @property
    def tick_task_name(self) -> str:
        return self.task_name("tick")

    @property
    def digest_task_name(self) -> str:
        return self.task_name("digest")

    def label(self, job: JobName) -> str:
        """launchd label: ``com.software-factory.<profile>.<job>``."""
        return f"com.software-factory.{self.profile}.{job}"

    @property
    def log_path(self) -> Path:
        return self.logs_dir / LOG_FILENAME

    @property
    def jobs(self) -> tuple[JobName, ...]:
        return JOBS if self.digest else ("tick",)

    # commands ------------------------------------------------------------------------
    def base_argv(self) -> list[str]:
        return [str(self.uv_path), "run", "--project", str(self.repo_root), "factory", "--profile", self.profile]

    def tick_argv(self) -> list[str]:
        return [*self.base_argv(), "tick"]

    def digest_argv(self) -> list[str]:
        return [*self.base_argv(), "report", "--digest", "--since", DIGEST_SINCE]

    def argv(self, job: JobName) -> list[str]:
        return self.tick_argv() if job == "tick" else self.digest_argv()

    def env(self) -> dict[str, str]:
        """Environment every job carries (scheduled jobs inherit none from the shell)."""
        env: dict[str, str] = {}
        if self.profiles_dir:
            env[PROFILES_DIR_ENV] = str(self.profiles_dir)
        if self.data_dir:
            env[DATA_DIR_ENV] = str(self.data_dir)
        return env

    def digest_hour_minute(self) -> tuple[int, int]:
        match = _TIME_RE.match(self.digest_time)
        assert match is not None  # validated
        return int(match["h"]), int(match["m"])

    # construction --------------------------------------------------------------------
    @classmethod
    def from_profile(
        cls,
        profile: Profile,
        paths: SchedulerPaths,
        *,
        interval_minutes: int | None = None,
        digest: bool = True,
        uv_path: Path | None = None,
        profiles_dir: Path | None = None,
        data_dir: Path | None = None,
        repo_root: Path | None = None,
        require_uv: bool = True,
    ) -> ScheduleSpec:
        """Build from ``Profile.schedule`` + runtime paths.

        ``profiles_dir`` defaults to the directory the profile was loaded from; it is embedded only
        when it differs from ``<repo root>/profiles`` (the loader's default). ``data_dir`` defaults
        to ``SOFTWARE_FACTORY_DATA_DIR`` from the current environment (resolved absolute) and is
        embedded whenever it is set, so the scheduled jobs use the same runtime data dir as the
        shell that installed them.
        """
        resolved_uv = uv_path or find_uv()
        if resolved_uv is None:
            if require_uv:
                raise SchedulerError("uv not found on PATH; install uv or pass uv_path explicitly")
            resolved_uv = Path("uv")
        root = repo_root or REPO_ROOT
        pdir = profiles_dir
        if pdir is None:
            pdir = profile.profile_dir.parent if profile.profile_dir else resolve_profiles_dir()
        if _same_path(pdir, root / "profiles"):
            pdir = None
        return cls(
            profile=profile.name,
            repo_root=root,
            uv_path=resolved_uv,
            profiles_dir=pdir,
            data_dir=data_dir if data_dir is not None else env_data_dir(),
            scripts_dir=paths.scripts,
            logs_dir=paths.logs,
            interval_minutes=interval_minutes or profile.schedule.interval_minutes,
            digest=digest,
            digest_weekday=profile.schedule.digest_weekday,
            digest_time=profile.schedule.digest_time,
        )


class ScheduleReport(BaseModel):
    """What ``install`` / ``uninstall`` did: job ids touched, files written or removed, argv run."""

    os: OsName
    profile: str
    action: Literal["install", "uninstall"]
    jobs: list[str] = Field(default_factory=list)
    files: list[Path] = Field(default_factory=list)
    commands: list[list[str]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{self.action} ({self.os}) profile {self.profile}"]
        lines += [f"  job   {j}" for j in self.jobs]
        lines += [f"  file  {f}" for f in self.files]
        lines += [f"  note  {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------------- helpers
def find_uv() -> Path | None:
    found = shutil.which("uv")
    return Path(found).resolve() if found else None


def env_data_dir() -> Path | None:
    """``SOFTWARE_FACTORY_DATA_DIR`` from the current environment, resolved absolute; ``None`` when
    unset (the jobs then use the platform default, like the CLI)."""
    value = os.environ.get(DATA_DIR_ENV)
    if not value:
        return None
    path = Path(os.path.expandvars(value)).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def install_drift(spec: ScheduleSpec, status_text: str) -> list[str]:
    """How the installed jobs, as the backend's ``show`` reports them, differ from what ``install``
    would embed now: the profile's data dir (every backend's status names a wrapper or log path
    under it) and the environment the jobs carry. Empty when consistent. A non-empty result means
    the scheduled ticks use another state.db / lock / logs than this shell's CLI."""
    text = os.path.normcase(status_text)
    problems: list[str] = []
    root = spec.scripts_dir.parent
    if os.path.normcase(str(root)) not in text:
        problems.append(f"they do not run from the current data dir {root}")
    for key, value in spec.env().items():
        if key not in status_text or os.path.normcase(value) not in text:
            problems.append(f"they do not carry {key}={value}")
    return problems


def _same_path(a: Path, b: Path) -> bool:
    try:
        ra, rb = a.expanduser().resolve(), b.expanduser().resolve()
    except OSError:
        ra, rb = a, b
    return os.path.normcase(str(ra)) == os.path.normcase(str(rb))


def current_os() -> OsName:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    raise SchedulerError(f"unsupported platform for scheduling: {sys.platform}")


def _backend(os_name: OsName | None):
    name = os_name or current_os()
    if name not in ("windows", "macos", "linux"):
        raise SchedulerError(f"unknown scheduler backend: {name!r}")
    # Lazy: backends import SchedulerError/RunResult from this package.
    return importlib.import_module(f".{name}", __name__)


def _default_paths(profile: Profile) -> SchedulerPaths:
    try:
        from ..paths import ProfilePaths  # lazy: paths.py is optional for this package
    except ImportError as exc:  # pragma: no cover - only when paths.py is absent
        raise SchedulerError("paths argument is required (software_factory.paths unavailable)") from exc
    return ProfilePaths.for_profile(profile.name)


# --------------------------------------------------------------------------------- dispatch
def install(
    profile: Profile,
    paths: SchedulerPaths,
    *,
    interval_minutes: int | None = None,
    digest: bool = True,
    runner: Runner | None = None,
    os_name: OsName | None = None,
    uv_path: Path | None = None,
) -> ScheduleReport:
    """Install the tick job (and the weekly digest when ``digest``) for one profile."""
    spec = ScheduleSpec.from_profile(
        profile, paths, interval_minutes=interval_minutes, digest=digest, uv_path=uv_path
    )
    return _backend(os_name).install(spec, runner or default_runner)


def uninstall(
    profile: Profile,
    paths: SchedulerPaths | None = None,
    *,
    runner: Runner | None = None,
    os_name: OsName | None = None,
) -> ScheduleReport:
    """Remove both jobs (missing ones are ignored) and any wrapper/plist files."""
    spec = ScheduleSpec.from_profile(profile, paths or _default_paths(profile), require_uv=False)
    return _backend(os_name).uninstall(spec, runner or default_runner)


def show(
    profile: Profile,
    paths: SchedulerPaths | None = None,
    *,
    runner: Runner | None = None,
    os_name: OsName | None = None,
) -> str:
    """Human-readable status of the profile's jobs as the OS scheduler reports them."""
    spec = ScheduleSpec.from_profile(profile, paths or _default_paths(profile), require_uv=False)
    return _backend(os_name).show(spec, runner or default_runner)


__all__ = [
    "DATA_DIR_ENV",
    "DIGEST_SINCE",
    "JOBS",
    "LOG_FILENAME",
    "POWER_RESTRICTED_MARKER",
    "PROFILES_DIR_ENV",
    "JobName",
    "OsName",
    "RunResult",
    "Runner",
    "ScheduleReport",
    "ScheduleSpec",
    "SchedulerError",
    "SchedulerPaths",
    "Weekday",
    "current_os",
    "default_runner",
    "env_data_dir",
    "find_uv",
    "install",
    "install_drift",
    "show",
    "uninstall",
]
