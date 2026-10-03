"""Linux cron backend.

The profile's jobs live in one block of the user's crontab between
``# software-factory:<profile> begin`` / ``# software-factory:<profile> end`` markers, so repeated
installs replace the block in place and uninstall removes exactly that block. Reads with
``crontab -l`` (no crontab = empty) and writes with ``crontab -`` (stdin).
"""

from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING

from . import ScheduleReport, SchedulerError, Weekday

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from . import Runner, ScheduleSpec

# cron: 0 = Sunday ... 6 = Saturday
WEEKDAYS: dict[Weekday, int] = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6}
_BEGIN_RE = re.compile(r"^# software-factory:(?P<profile>\S+) begin\s*$")


def begin_marker(profile: str) -> str:
    return f"# software-factory:{profile} begin"


def end_marker(profile: str) -> str:
    return f"# software-factory:{profile} end"


# ------------------------------------------------------------------------------ render
def cron_interval_expr(interval_minutes: int) -> str:
    """``*/N * * * *`` for N < 60; whole hours become ``0 */H * * *``; 1440 = daily."""
    n = interval_minutes
    if n < 1:
        raise SchedulerError(f"interval_minutes must be >= 1, got {n}")
    if n == 1:
        return "* * * * *"
    if n < 60:
        return f"*/{n} * * * *"
    if n == 1440:
        return "0 0 * * *"
    hours, rem = divmod(n, 60)
    if rem or hours > 23:
        raise SchedulerError(f"cron cannot express every {n} minutes; use <60 or a whole number of hours (<=23)")
    return "0 * * * *" if hours == 1 else f"0 */{hours} * * *"


def cron_weekly_expr(weekday: Weekday, time_hhmm: str) -> str:
    if weekday not in WEEKDAYS:
        raise SchedulerError(f"weekday must be one of {tuple(WEEKDAYS)}, got {weekday!r}")
    hour, _, minute = time_hhmm.partition(":")
    try:
        h, m = int(hour), int(minute)
    except ValueError as exc:
        raise SchedulerError(f"time must be HH:MM, got {time_hhmm!r}") from exc
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise SchedulerError(f"time out of range: {time_hhmm!r}")
    return f"{m} {h} * * {WEEKDAYS[weekday]}"


def cron_command(repo_root: Path, argv: Sequence[str], env: Mapping[str, str], log_path: Path) -> str:
    """The command part of a crontab line: cd to the repo, env prefix, command, log redirect."""
    prefix = "".join(f"{key}={shlex.quote(value)} " for key, value in env.items())
    cmd = " ".join(shlex.quote(str(a)) for a in argv)
    line = f"cd {shlex.quote(str(repo_root))} && {prefix}{cmd} >> {shlex.quote(str(log_path))} 2>&1"
    return line.replace("%", "\\%")  # literal % is a newline in cron commands


def render_cron_block(
    profile: str,
    *,
    repo_root: Path,
    tick_argv: Sequence[str],
    interval_minutes: int,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    digest_argv: Sequence[str] | None = None,
    digest_weekday: Weekday = "MON",
    digest_time: str = "09:00",
) -> str:
    """The marker-delimited block (always ends with a newline)."""
    env = dict(env or {})
    lines = [
        begin_marker(profile),
        f"{cron_interval_expr(interval_minutes)} {cron_command(repo_root, tick_argv, env, log_path)}",
    ]
    if digest_argv:
        lines.append(
            f"{cron_weekly_expr(digest_weekday, digest_time)} {cron_command(repo_root, digest_argv, env, log_path)}"
        )
    lines.append(end_marker(profile))
    return "\n".join(lines) + "\n"


def render_spec_block(spec: ScheduleSpec) -> str:
    return render_cron_block(
        spec.profile,
        repo_root=spec.repo_root,
        tick_argv=spec.tick_argv(),
        interval_minutes=spec.interval_minutes,
        log_path=spec.log_path,
        env=spec.env(),
        digest_argv=spec.digest_argv() if spec.digest else None,
        digest_weekday=spec.digest_weekday,
        digest_time=spec.digest_time,
    )


# ------------------------------------------------------------------------------ merge
def _block_span(lines: list[str], profile: str) -> tuple[int, int] | None:
    """(begin index, end index inclusive) of the profile's block; an unterminated block runs to EOF."""
    begin, end = begin_marker(profile), end_marker(profile)
    start = next((i for i, line in enumerate(lines) if line.rstrip() == begin), None)
    if start is None:
        return None
    stop = next((i for i in range(start + 1, len(lines)) if lines[i].rstrip() == end), len(lines) - 1)
    return start, stop


def extract_block(crontab_text: str, profile: str) -> str | None:
    lines = crontab_text.splitlines()
    span = _block_span(lines, profile)
    if span is None:
        return None
    return "\n".join(lines[span[0] : span[1] + 1]) + "\n"


def remove_block(crontab_text: str, profile: str) -> str:
    lines = crontab_text.splitlines()
    span = _block_span(lines, profile)
    if span is None:
        return crontab_text
    start, stop = span
    # Drop one blank separator line that install added before the block, if present.
    if start > 0 and lines[start - 1].strip() == "":
        start -= 1
    kept = lines[:start] + lines[stop + 1 :]
    return _finish(kept)


def merge_crontab(existing: str, block: str) -> str:
    """Replace the profile's block in place, or append it. Idempotent."""
    first = block.splitlines()[0] if block.strip() else ""
    match = _BEGIN_RE.match(first)
    if match is None:
        raise SchedulerError("block must start with a '# software-factory:<profile> begin' marker")
    profile = match["profile"]
    block_lines = block.rstrip("\n").splitlines()
    lines = existing.splitlines()
    span = _block_span(lines, profile)
    if span is None:
        if lines and lines[-1].strip():
            lines.append("")  # blank separator before an appended block
        lines += block_lines
    else:
        lines[span[0] : span[1] + 1] = block_lines
    return _finish(lines)


def _finish(lines: list[str]) -> str:
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


# ------------------------------------------------------------------------------ actions
def read_crontab(runner: Runner) -> str:
    result = runner(["crontab", "-l"])
    if result.ok:
        return result.stdout
    if "no crontab" in result.output.lower():
        return ""
    raise SchedulerError(f"crontab -l failed ({result.returncode}): {result.output}")


def write_crontab(runner: Runner, text: str) -> None:
    result = runner(["crontab", "-"], input=text)
    if not result.ok:
        raise SchedulerError(f"crontab - failed ({result.returncode}): {result.output}")


def install(spec: ScheduleSpec, runner: Runner) -> ScheduleReport:
    report = ScheduleReport(os="linux", profile=spec.profile, action="install")
    spec.logs_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_crontab(read_crontab(runner), render_spec_block(spec))
    report.commands += [["crontab", "-l"], ["crontab", "-"]]
    write_crontab(runner, merged)
    report.jobs += [spec.task_name(job) for job in spec.jobs]
    return report


def uninstall(spec: ScheduleSpec, runner: Runner) -> ScheduleReport:
    report = ScheduleReport(os="linux", profile=spec.profile, action="uninstall")
    existing = read_crontab(runner)
    report.commands.append(["crontab", "-l"])
    if extract_block(existing, spec.profile) is None:
        report.notes.append(f"{spec.task_prefix}: not installed")
        return report
    report.commands.append(["crontab", "-"])
    write_crontab(runner, remove_block(existing, spec.profile))
    report.jobs.append(spec.task_prefix)
    return report


def show(spec: ScheduleSpec, runner: Runner) -> str:
    block = extract_block(read_crontab(runner), spec.profile)
    return block.rstrip("\n") if block else f"{spec.task_prefix}: not installed"
