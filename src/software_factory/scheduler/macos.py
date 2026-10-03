"""macOS launchd backend.

One LaunchAgent plist per job in ``~/Library/LaunchAgents``: ``StartInterval`` for the tick,
``StartCalendarInterval`` for the weekly digest. Loaded with ``launchctl bootstrap gui/<uid>``
(falls back to the legacy ``launchctl load -w``), removed with ``launchctl bootout``.
"""

from __future__ import annotations

import os
import plistlib
from pathlib import Path
from typing import TYPE_CHECKING

from . import JOBS, JobName, ScheduleReport, SchedulerError, Weekday

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from . import Runner, ScheduleSpec

# launchd StartCalendarInterval Weekday: 0 and 7 are Sunday.
WEEKDAYS: dict[Weekday, int] = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6}


def launch_agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path(spec: ScheduleSpec, job: JobName, agents_dir: Path | None = None) -> Path:
    return (agents_dir or launch_agents_dir()) / f"{spec.label(job)}.plist"


def current_uid() -> int:
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        raise SchedulerError("launchd scheduling requires a POSIX host")
    return int(getuid())


# ------------------------------------------------------------------------------ render
def weekly_calendar(weekday: Weekday, time_hhmm: str) -> dict[str, int]:
    if weekday not in WEEKDAYS:
        raise SchedulerError(f"weekday must be one of {tuple(WEEKDAYS)}, got {weekday!r}")
    hour, _, minute = time_hhmm.partition(":")
    try:
        h, m = int(hour), int(minute)
    except ValueError as exc:
        raise SchedulerError(f"time must be HH:MM, got {time_hhmm!r}") from exc
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise SchedulerError(f"time out of range: {time_hhmm!r}")
    return {"Weekday": WEEKDAYS[weekday], "Hour": h, "Minute": m}


def render_plist(
    label: str,
    program_args: Sequence[str],
    *,
    interval_seconds: int | None = None,
    calendar: Mapping[str, int] | None = None,
    working_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
) -> str:
    """XML plist text. Exactly one of ``interval_seconds`` / ``calendar`` must be given."""
    if (interval_seconds is None) == (calendar is None):
        raise SchedulerError("render_plist needs exactly one of interval_seconds or calendar")
    data: dict[str, object] = {
        "Label": label,
        "ProgramArguments": [str(a) for a in program_args],
        "RunAtLoad": False,
    }
    if interval_seconds is not None:
        if interval_seconds < 1:
            raise SchedulerError(f"interval_seconds must be >= 1, got {interval_seconds}")
        data["StartInterval"] = int(interval_seconds)
    else:
        data["StartCalendarInterval"] = dict(calendar or {})
    if working_dir is not None:
        data["WorkingDirectory"] = str(working_dir)
    if env:
        data["EnvironmentVariables"] = dict(env)
    if log_path is not None:
        data["StandardOutPath"] = str(log_path)
        data["StandardErrorPath"] = str(log_path)
    return plistlib.dumps(data, sort_keys=True).decode("utf-8")


def render_job_plist(spec: ScheduleSpec, job: JobName, env: Mapping[str, str] | None = None) -> str:
    """Plist for one of the spec's jobs (env defaults to ``spec.env()``)."""
    kwargs: dict[str, object] = {
        "working_dir": spec.repo_root,
        "env": dict(spec.env() if env is None else env),
        "log_path": spec.log_path,
    }
    if job == "tick":
        return render_plist(spec.label(job), spec.tick_argv(), interval_seconds=spec.interval_minutes * 60, **kwargs)  # type: ignore[arg-type]
    calendar = weekly_calendar(spec.digest_weekday, spec.digest_time)
    return render_plist(spec.label(job), spec.digest_argv(), calendar=calendar, **kwargs)  # type: ignore[arg-type]


def render_launchctl_bootstrap(uid: int, plist: Path) -> list[str]:
    return ["launchctl", "bootstrap", f"gui/{uid}", str(plist)]


def render_launchctl_bootout(uid: int, label: str) -> list[str]:
    return ["launchctl", "bootout", f"gui/{uid}/{label}"]


def render_launchctl_load(plist: Path) -> list[str]:
    return ["launchctl", "load", "-w", str(plist)]


def render_launchctl_unload(plist: Path) -> list[str]:
    return ["launchctl", "unload", "-w", str(plist)]


def render_launchctl_print(uid: int, label: str) -> list[str]:
    return ["launchctl", "print", f"gui/{uid}/{label}"]


# ------------------------------------------------------------------------------ actions
def _agent_env(spec: ScheduleSpec) -> dict[str, str]:
    # launchd gives agents a minimal PATH; embed the install-time PATH so git/gh resolve.
    env = spec.env()
    path = os.environ.get("PATH")
    if path:
        env["PATH"] = path
    return env


def install(
    spec: ScheduleSpec, runner: Runner, *, uid: int | None = None, agents_dir: Path | None = None
) -> ScheduleReport:
    report = ScheduleReport(os="macos", profile=spec.profile, action="install")
    uid = current_uid() if uid is None else uid
    agents = agents_dir or launch_agents_dir()
    agents.mkdir(parents=True, exist_ok=True)
    spec.logs_dir.mkdir(parents=True, exist_ok=True)
    env = _agent_env(spec)
    for job in spec.jobs:
        label = spec.label(job)
        plist = plist_path(spec, job, agents)
        plist.write_text(render_job_plist(spec, job, env), encoding="utf-8")
        report.files.append(plist)
        # Re-install must not fail with "service already loaded": bootout first, ignore the result.
        bootout = render_launchctl_bootout(uid, label)
        report.commands.append(bootout)
        runner(bootout)
        bootstrap = render_launchctl_bootstrap(uid, plist)
        report.commands.append(bootstrap)
        result = runner(bootstrap)
        if not result.ok:
            load = render_launchctl_load(plist)
            report.commands.append(load)
            fallback = runner(load)
            if not fallback.ok:
                raise SchedulerError(
                    f"launchctl could not load {label}: bootstrap: {result.output or result.returncode}; "
                    f"load -w: {fallback.output or fallback.returncode}"
                )
            report.notes.append(f"{label}: bootstrap failed, loaded with legacy `launchctl load -w`")
        report.jobs.append(label)
    return report


def uninstall(
    spec: ScheduleSpec, runner: Runner, *, uid: int | None = None, agents_dir: Path | None = None
) -> ScheduleReport:
    report = ScheduleReport(os="macos", profile=spec.profile, action="uninstall")
    uid = current_uid() if uid is None else uid
    agents = agents_dir or launch_agents_dir()
    for job in JOBS:
        label = spec.label(job)
        plist = plist_path(spec, job, agents)
        bootout = render_launchctl_bootout(uid, label)
        report.commands.append(bootout)
        result = runner(bootout)
        if result.ok:
            report.jobs.append(label)
        elif plist.exists():
            unload = render_launchctl_unload(plist)
            report.commands.append(unload)
            if runner(unload).ok:
                report.jobs.append(label)
            else:
                report.notes.append(f"{label}: not loaded")
        else:
            report.notes.append(f"{label}: not installed")
        if plist.exists():
            plist.unlink()
            report.files.append(plist)
    return report


def show(spec: ScheduleSpec, runner: Runner, *, uid: int | None = None, agents_dir: Path | None = None) -> str:
    uid = current_uid() if uid is None else uid
    agents = agents_dir or launch_agents_dir()
    sections: list[str] = []
    for job in JOBS:
        label = spec.label(job)
        plist = plist_path(spec, job, agents)
        result = runner(render_launchctl_print(uid, label))
        if result.ok:
            sections.append(f"{label} ({plist})\n{result.stdout.strip()}")
        elif plist.exists():
            sections.append(f"{label}: plist present but not loaded ({plist})")
        else:
            sections.append(f"{label}: not installed")
    return "\n\n".join(sections)
