"""Scheduler: command/file generation per OS and the install/uninstall flows with a fake runner.

Never invokes schtasks / launchctl / crontab: every backend takes an injectable runner.
"""

from __future__ import annotations

import plistlib
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from software_factory import scheduler
from software_factory.config import REPO_ROOT, Profile
from software_factory.scheduler import RunResult, SchedulerError, ScheduleSpec, linux, macos, windows

UV = Path("C:/Users/me/.local/bin/uv.EXE")
NOT_FOUND = "ERROR: The system cannot find the file specified."


class FakeRunner:
    """Records argv/stdin; ``handler(argv, input) -> (returncode, stdout, stderr)``."""

    def __init__(self, handler=None):
        self.calls: list[tuple[list[str], str | None]] = []
        self.handler = handler or (lambda argv, input: (0, "", ""))

    def __call__(self, argv, *, input=None):
        argv = [str(a) for a in argv]
        self.calls.append((argv, input))
        rc, out, err = self.handler(argv, input)
        return RunResult(argv=argv, returncode=rc, stdout=out, stderr=err)

    @property
    def argvs(self) -> list[list[str]]:
        return [argv for argv, _ in self.calls]

    def argvs_with(self, token: str) -> list[list[str]]:
        return [argv for argv in self.argvs if token in argv]


@pytest.fixture
def tmp_paths(tmp_path: Path):
    return SimpleNamespace(scripts=tmp_path / "data" / "scripts", logs=tmp_path / "data" / "logs")


@pytest.fixture
def spec(tmp_path: Path, tmp_paths) -> ScheduleSpec:
    return ScheduleSpec(
        profile="work",
        repo_root=tmp_path / "dir with space" / "software-factory",
        uv_path=UV,
        profiles_dir=tmp_path / "my profiles",
        scripts_dir=tmp_paths.scripts,
        logs_dir=tmp_paths.logs,
        interval_minutes=20,
        digest=True,
        digest_weekday="MON",
        digest_time="09:00",
    )


# ================================================================================ ScheduleSpec
class TestScheduleSpec:
    def test_from_profile_non_default_profiles_dir_sets_env(self, tmp_path: Path, tmp_paths, monkeypatch):
        monkeypatch.delenv(scheduler.DATA_DIR_ENV)  # conftest sets it; this test is about the profiles dir
        profile = Profile(name="work")
        profile.profile_dir = tmp_path / "profiles here" / "work"
        s = ScheduleSpec.from_profile(profile, tmp_paths, uv_path=UV)
        assert s.profile == "work"
        assert s.repo_root == REPO_ROOT
        assert s.profiles_dir == tmp_path / "profiles here"
        assert s.env() == {"SOFTWARE_FACTORY_PROFILES_DIR": str(tmp_path / "profiles here")}
        assert s.interval_minutes == profile.schedule.interval_minutes == 20
        assert s.digest_weekday == "MON" and s.digest_time == "09:00"
        assert s.scripts_dir == tmp_paths.scripts and s.logs_dir == tmp_paths.logs

    def test_from_profile_default_profiles_dir_has_no_env(self, tmp_paths, monkeypatch):
        monkeypatch.delenv(scheduler.DATA_DIR_ENV)
        profile = Profile(name="personal")
        profile.profile_dir = REPO_ROOT / "profiles" / "personal"
        s = ScheduleSpec.from_profile(profile, tmp_paths, uv_path=UV)
        assert s.profiles_dir is None and s.data_dir is None
        assert s.env() == {}

    def test_from_profile_embeds_the_shell_data_dir_in_every_backend(self, tmp_path: Path, tmp_paths, monkeypatch):
        """Scheduled jobs inherit no shell env: SOFTWARE_FACTORY_DATA_DIR must travel with them or a
        scheduled tick uses another state.db / lock / logs than the CLI that installed it."""
        monkeypatch.setenv(scheduler.DATA_DIR_ENV, str(tmp_path / "factory data"))
        profile = Profile(name="personal")
        profile.profile_dir = REPO_ROOT / "profiles" / "personal"
        s = ScheduleSpec.from_profile(profile, tmp_paths, uv_path=UV)
        assert s.data_dir == (tmp_path / "factory data").resolve() and s.data_dir.is_absolute()
        assert s.env() == {scheduler.DATA_DIR_ENV: str(s.data_dir)}
        line = f'set "{scheduler.DATA_DIR_ENV}={s.data_dir}"'
        assert line in windows.render_wrapper_cmd(repo_root=s.repo_root, argv=s.tick_argv(), env=s.env(), log_path=s.log_path)
        plist = plistlib.loads(macos.render_job_plist(s, "tick").encode("utf-8"))
        assert plist["EnvironmentVariables"][scheduler.DATA_DIR_ENV] == str(s.data_dir)
        assert f"{scheduler.DATA_DIR_ENV}='{s.data_dir}'" in linux.render_spec_block(s)

        monkeypatch.delenv(scheduler.DATA_DIR_ENV)
        assert ScheduleSpec.from_profile(profile, tmp_paths, uv_path=UV).data_dir is None
        explicit = ScheduleSpec.from_profile(profile, tmp_paths, uv_path=UV, data_dir=tmp_path / "explicit")
        assert explicit.env() == {scheduler.DATA_DIR_ENV: str(tmp_path / "explicit")}

    def test_from_profile_without_profile_dir_uses_env_var(self, tmp_path: Path, tmp_paths, monkeypatch):
        monkeypatch.setenv("SOFTWARE_FACTORY_PROFILES_DIR", str(tmp_path / "elsewhere"))
        s = ScheduleSpec.from_profile(Profile(name="p"), tmp_paths, uv_path=UV)
        assert s.profiles_dir == (tmp_path / "elsewhere").resolve()

    def test_from_profile_overrides(self, tmp_paths):
        profile = Profile(name="p", schedule={"interval_minutes": 5, "digest_weekday": "FRI", "digest_time": "17:30"})
        s = ScheduleSpec.from_profile(profile, tmp_paths, uv_path=UV, interval_minutes=45, digest=False)
        assert s.interval_minutes == 45 and s.digest is False
        assert s.jobs == ("tick",)
        assert s.digest_weekday == "FRI" and s.digest_hour_minute() == (17, 30)

    def test_from_profile_requires_uv_on_path(self, tmp_paths, monkeypatch):
        monkeypatch.setattr(scheduler, "find_uv", lambda: None)
        with pytest.raises(SchedulerError, match="uv not found"):
            ScheduleSpec.from_profile(Profile(name="p"), tmp_paths)
        s = ScheduleSpec.from_profile(Profile(name="p"), tmp_paths, require_uv=False)
        assert s.uv_path == Path("uv")

    def test_from_profile_embeds_absolute_uv_from_which(self, tmp_path: Path, tmp_paths, monkeypatch):
        fake_uv = tmp_path / "bin" / "uv.exe"
        fake_uv.parent.mkdir()
        fake_uv.write_text("")
        monkeypatch.setattr(scheduler.shutil, "which", lambda name: str(fake_uv) if name == "uv" else None)
        s = ScheduleSpec.from_profile(Profile(name="p"), tmp_paths)
        assert s.uv_path == fake_uv.resolve() and s.uv_path.is_absolute()

    def test_commands_and_names(self, spec: ScheduleSpec):
        base = [str(UV), "run", "--project", str(spec.repo_root), "factory", "--profile", "work"]
        assert spec.tick_argv() == [*base, "tick"]
        assert spec.digest_argv() == [*base, "report", "--digest", "--since", "7d"]
        assert spec.argv("digest") == spec.digest_argv()
        assert spec.tick_task_name == "software-factory-work-tick"
        assert spec.digest_task_name == "software-factory-work-digest"
        assert spec.label("tick") == "com.software-factory.work.tick"
        assert spec.log_path == spec.logs_dir / "scheduler.log"
        assert spec.jobs == ("tick", "digest")

    @pytest.mark.parametrize("bad", [0, -1, 1440])
    def test_interval_range(self, spec: ScheduleSpec, bad: int):
        with pytest.raises(ValidationError):
            ScheduleSpec.model_validate({**spec.model_dump(), "interval_minutes": bad})

    @pytest.mark.parametrize("bad", ["9:00", "24:00", "09:60", "morning"])
    def test_digest_time_format(self, spec: ScheduleSpec, bad: str):
        with pytest.raises(ValidationError):
            ScheduleSpec.model_validate({**spec.model_dump(), "digest_time": bad})


# ===================================================================================== Windows
class TestWindowsRender:
    def test_wrapper_cmd(self, spec: ScheduleSpec):
        text = windows.render_wrapper_cmd(
            repo_root=spec.repo_root, argv=spec.tick_argv(), env=spec.env(), log_path=spec.log_path, job="tick"
        )
        assert text.endswith("\r\n") and "\r\r\n" not in text
        lines = text.split("\r\n")
        assert lines[0] == "@echo off"
        assert f'cd /d "{spec.repo_root}"' in lines
        assert f'set "SOFTWARE_FACTORY_PROFILES_DIR={spec.profiles_dir}"' in lines
        log = f'"{spec.log_path}"' if " " in str(spec.log_path) else str(spec.log_path)
        run_line = next(line for line in lines if line.startswith(str(UV)))
        assert run_line == (
            f'{UV} run --project "{spec.repo_root}" factory --profile work tick >> {log} 2>&1'
        )
        assert "exit /b %ERRORLEVEL%" in lines
        assert any(line.startswith("echo [%DATE% %TIME%] tick start>>") for line in lines)

    def test_wrapper_cmd_without_env_has_no_set_line(self, spec: ScheduleSpec):
        text = windows.render_wrapper_cmd(
            repo_root=spec.repo_root, argv=spec.digest_argv(), env={}, log_path=spec.log_path
        )
        assert "set " not in text
        assert "report --digest --since 7d >>" in text

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            ("plain", "plain"),
            ("C:\\no\\spaces.cmd", "C:\\no\\spaces.cmd"),
            ("C:\\has space\\x", '"C:\\has space\\x"'),
            ("a&b", '"a&b"'),
            ("", '""'),
        ],
    )
    def test_cmd_quote(self, arg: str, expected: str):
        assert windows.cmd_quote(arg) == expected

    def test_schtasks_create_minute(self):
        wrapper = Path("C:/data/scripts/software-factory-work-tick.cmd")
        argv = windows.render_schtasks_create("software-factory-work-tick", wrapper, 20)
        assert argv == [
            "schtasks", "/Create", "/TN", "software-factory-work-tick",
            "/SC", "MINUTE", "/MO", "20",
            "/TR", f'"{wrapper}"',
            "/RL", "LIMITED", "/F",
        ]  # fmt: skip
        # Default principal: no /RU, /RP, /NP or /IT -> "run only when user is logged on".
        assert not {"/RU", "/RP", "/NP", "/IT", "/S"} & set(argv)

    def test_schtasks_create_weekly(self):
        wrapper = Path("C:/data/scripts/software-factory-work-digest.cmd")
        argv = windows.render_schtasks_create_weekly("software-factory-work-digest", wrapper, "MON", "09:00")
        assert argv[:4] == ["schtasks", "/Create", "/TN", "software-factory-work-digest"]
        assert argv[4:10] == ["/SC", "WEEKLY", "/D", "MON", "/ST", "09:00"]
        assert argv[10:] == ["/TR", f'"{wrapper}"', "/RL", "LIMITED", "/F"]
        with pytest.raises(SchedulerError, match="weekday"):
            windows.render_schtasks_create_weekly("t", wrapper, "MONDAY", "09:00")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, 1440])
    def test_schtasks_create_interval_bounds(self, bad: int):
        with pytest.raises(SchedulerError, match="1..1439"):
            windows.render_schtasks_create("t", Path("C:/x.cmd"), bad)

    def test_schtasks_taskrun_limit(self):
        too_long = Path("C:/" + "a" * 300 + "/x.cmd")
        with pytest.raises(SchedulerError, match="/TR limit"):
            windows.render_schtasks_create("t", too_long, 5)

    def test_schtasks_delete_and_query(self):
        assert windows.render_schtasks_delete("x") == ["schtasks", "/Delete", "/TN", "x", "/F"]
        assert windows.render_schtasks_query("x") == ["schtasks", "/Query", "/TN", "x", "/FO", "LIST", "/V"]

    def test_power_settings_allow_running_on_battery(self):
        argv = windows.render_power_settings("software-factory-work-tick")
        assert argv[:6] == ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]
        command = argv[6]
        assert command.startswith("Set-ScheduledTask -TaskName 'software-factory-work-tick' -Settings (New-ScheduledTaskSettingsSet ")
        for flag in ("-AllowStartIfOnBatteries", "-DontStopIfGoingOnBatteries", "-MultipleInstances IgnoreNew", "-ExecutionTimeLimit (New-TimeSpan -Hours 6)"):
            assert flag in command
        assert windows.ps_quote("it's") == "'it''s'"

    def test_wrapper_env_reads_the_set_lines(self, spec: ScheduleSpec, tmp_path: Path):
        s = spec.model_copy(update={"data_dir": tmp_path / "factory data"})
        wrapper = tmp_path / "w.cmd"
        wrapper.write_text(
            windows.render_wrapper_cmd(repo_root=s.repo_root, argv=s.tick_argv(), env=s.env(), log_path=s.log_path),
            encoding="utf-8", newline="",
        )  # fmt: skip
        assert windows.wrapper_env(wrapper) == s.env() == {
            "SOFTWARE_FACTORY_PROFILES_DIR": str(s.profiles_dir),
            scheduler.DATA_DIR_ENV: str(tmp_path / "factory data"),
        }

    def test_is_not_found(self):
        assert windows.is_not_found(RunResult(argv=[], returncode=1, stderr=NOT_FOUND))
        assert not windows.is_not_found(RunResult(argv=[], returncode=0, stdout=NOT_FOUND))
        assert not windows.is_not_found(RunResult(argv=[], returncode=1, stderr="ERROR: Access is denied."))


class TestWindowsFlows:
    def test_install_writes_wrappers_and_creates_tasks(self, spec: ScheduleSpec):
        runner = FakeRunner()
        report = windows.install(spec, runner)
        tick = spec.scripts_dir / "software-factory-work-tick.cmd"
        digest = spec.scripts_dir / "software-factory-work-digest.cmd"
        assert report.files == [tick, digest]
        assert report.jobs == ["software-factory-work-tick", "software-factory-work-digest"]
        assert spec.logs_dir.is_dir()
        raw = tick.read_bytes()
        assert b"\r\n" in raw and b"\r\r\n" not in raw
        assert b"factory --profile work tick >>" in raw
        assert b"report --digest --since 7d" in digest.read_bytes()
        # Every /Create is followed by the power-settings rewrite (schtasks' defaults never tick on battery).
        assert runner.argvs == [
            windows.render_schtasks_create(spec.tick_task_name, tick, 20),
            windows.render_power_settings(spec.tick_task_name),
            windows.render_schtasks_create_weekly(spec.digest_task_name, digest, "MON", "09:00"),
            windows.render_power_settings(spec.digest_task_name),
        ]
        assert report.commands == runner.argvs
        assert report.notes == []

    def test_install_without_digest(self, spec: ScheduleSpec):
        runner = FakeRunner()
        report = windows.install(spec.model_copy(update={"digest": False}), runner)
        assert report.jobs == ["software-factory-work-tick"]
        assert len(runner.argvs) == 2 and "MINUTE" in runner.argvs[0] and runner.argvs[1][0] == "powershell"
        assert not (spec.scripts_dir / "software-factory-work-digest.cmd").exists()

    def test_install_failure_raises_with_output(self, spec: ScheduleSpec):
        runner = FakeRunner(lambda argv, _: (1, "", "ERROR: Access is denied."))
        with pytest.raises(SchedulerError, match="Access is denied"):
            windows.install(spec, runner)

    def test_install_notes_when_the_power_settings_cannot_be_applied(self, spec: ScheduleSpec):
        def handler(argv, _):
            if argv[0] == "powershell":
                return (1, "", "Set-ScheduledTask : Access is denied.\nAt line:1 char:1")
            return (0, "SUCCESS", "")

        report = windows.install(spec, FakeRunner(handler))
        assert report.jobs == ["software-factory-work-tick", "software-factory-work-digest"]  # tasks exist, on AC they run
        assert len(report.notes) == 2
        assert report.notes[0].startswith("software-factory-work-tick: could not allow running on battery (Set-ScheduledTask : Access is denied.)")
        assert "no start on battery" in report.notes[0]

        def missing(argv, *, input=None):
            if argv[0] == "powershell":
                raise SchedulerError("powershell not found on PATH")
            return RunResult(argv=[str(a) for a in argv], returncode=0)

        report = windows.install(spec.model_copy(update={"digest": False}), missing)
        assert report.jobs == ["software-factory-work-tick"]
        assert report.notes == [
            "software-factory-work-tick: could not allow running on battery (powershell not found on PATH); "
            "the task keeps the schtasks default (no start on battery, stop when going on battery)"
        ]

    def test_uninstall_ignores_missing_tasks_and_removes_wrappers(self, spec: ScheduleSpec):
        windows.install(spec, FakeRunner())
        tick = windows.wrapper_path(spec, "tick")
        digest = windows.wrapper_path(spec, "digest")

        def handler(argv, _):
            return (0, "SUCCESS", "") if "tick" in argv[3] else (1, "", NOT_FOUND)

        runner = FakeRunner(handler)
        report = windows.uninstall(spec, runner)
        assert runner.argvs == [
            windows.render_schtasks_delete(spec.tick_task_name),
            windows.render_schtasks_delete(spec.digest_task_name),
        ]
        assert report.jobs == ["software-factory-work-tick"]
        assert report.notes == ["software-factory-work-digest: not installed"]
        assert not tick.exists() and not digest.exists()
        assert report.files == [tick, digest]

    def test_uninstall_other_error_raises(self, spec: ScheduleSpec):
        runner = FakeRunner(lambda argv, _: (1, "", "ERROR: Access is denied."))
        with pytest.raises(SchedulerError, match="Access is denied"):
            windows.uninstall(spec, runner)

    def test_show(self, spec: ScheduleSpec):
        def handler(argv, _):
            if argv[3].endswith("tick"):
                return (0, "TaskName: \\software-factory-work-tick\nSchedule Type: One Time Only, Minute\n", "")
            return (1, "", NOT_FOUND)

        out = windows.show(spec, FakeRunner(handler))
        assert "Schedule Type: One Time Only, Minute" in out
        assert "software-factory-work-digest: not installed" in out
        # no wrapper under the current scripts dir: the task was installed from another data dir (the
        # line must not name the current dir, or install_drift would take it for the task's location)
        assert "Wrapper:                              missing (no wrapper for this task under the current data dir" in out
        assert str(spec.scripts_dir) not in out

    def test_show_prints_the_env_the_wrapper_sets(self, spec: ScheduleSpec, tmp_path: Path):
        s = spec.model_copy(update={"data_dir": tmp_path / "factory data"})
        windows.install(s, FakeRunner())
        out = windows.show(s, FakeRunner(lambda argv, _: (0, f"TaskName: {argv[3]}\nStatus: Ready\n", "")))
        assert f"Wrapper:                              {windows.wrapper_path(s, 'tick')}" in out
        assert (
            f"Wrapper Env:                          SOFTWARE_FACTORY_PROFILES_DIR={s.profiles_dir}, "
            f"{scheduler.DATA_DIR_ENV}={tmp_path / 'factory data'}"
        ) in out
        assert scheduler.install_drift(s, out) == []


class TestInstallDrift:
    """What doctor uses to tell that the installed jobs disagree with the current shell."""

    def test_consistent_status_has_no_drift(self, spec: ScheduleSpec):
        text = f'Task To Run: "{windows.wrapper_path(spec, "tick")}"\nWrapper Env: SOFTWARE_FACTORY_PROFILES_DIR={spec.profiles_dir}'
        assert scheduler.install_drift(spec, text) == []
        cron = linux.render_spec_block(spec)
        assert scheduler.install_drift(spec, cron) == []

    def test_jobs_from_another_data_dir(self, spec: ScheduleSpec, tmp_path: Path):
        other = tmp_path / "other" / "scripts" / "software-factory-work-tick.cmd"
        text = f'Task To Run: "{other}"\nWrapper Env: SOFTWARE_FACTORY_PROFILES_DIR={spec.profiles_dir}'
        assert scheduler.install_drift(spec, text) == [f"they do not run from the current data dir {spec.scripts_dir.parent}"]

    def test_jobs_missing_the_data_dir_env(self, spec: ScheduleSpec, tmp_path: Path):
        s = spec.model_copy(update={"data_dir": tmp_path / "factory data"})
        text = f'Task To Run: "{windows.wrapper_path(s, "tick")}"\nWrapper Env: SOFTWARE_FACTORY_PROFILES_DIR={s.profiles_dir}'
        assert scheduler.install_drift(s, text) == [f"they do not carry {scheduler.DATA_DIR_ENV}={tmp_path / 'factory data'}"]
        stale_value = text + f"\n{scheduler.DATA_DIR_ENV}={tmp_path / 'old data'}"
        assert scheduler.install_drift(s, stale_value) == [f"they do not carry {scheduler.DATA_DIR_ENV}={tmp_path / 'factory data'}"]

    def test_env_data_dir_resolves_and_handles_unset(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(scheduler.DATA_DIR_ENV, str(tmp_path / "rel" / ".." / "data"))
        assert scheduler.env_data_dir() == (tmp_path / "data").resolve()
        monkeypatch.setenv(scheduler.DATA_DIR_ENV, "")
        assert scheduler.env_data_dir() is None
        monkeypatch.delenv(scheduler.DATA_DIR_ENV)
        assert scheduler.env_data_dir() is None


# ======================================================================================= macOS
class TestMacosRender:
    def test_plist_interval(self, spec: ScheduleSpec):
        text = macos.render_plist(
            "com.software-factory.work.tick",
            spec.tick_argv(),
            interval_seconds=1200,
            working_dir=spec.repo_root,
            env=spec.env(),
            log_path=spec.log_path,
        )
        data = plistlib.loads(text.encode("utf-8"))
        assert data["Label"] == "com.software-factory.work.tick"
        assert data["ProgramArguments"] == spec.tick_argv()
        assert data["StartInterval"] == 1200
        assert "StartCalendarInterval" not in data
        assert data["WorkingDirectory"] == str(spec.repo_root)
        assert data["EnvironmentVariables"] == spec.env()
        assert data["StandardOutPath"] == data["StandardErrorPath"] == str(spec.log_path)
        assert data["RunAtLoad"] is False

    def test_plist_calendar(self):
        text = macos.render_plist("l", ["uv", "run"], calendar=macos.weekly_calendar("MON", "09:00"))
        data = plistlib.loads(text.encode("utf-8"))
        assert data["StartCalendarInterval"] == {"Weekday": 1, "Hour": 9, "Minute": 0}
        assert "StartInterval" not in data
        assert "EnvironmentVariables" not in data and "StandardOutPath" not in data

    def test_plist_requires_exactly_one_trigger(self):
        with pytest.raises(SchedulerError, match="exactly one"):
            macos.render_plist("l", ["x"])
        with pytest.raises(SchedulerError, match="exactly one"):
            macos.render_plist("l", ["x"], interval_seconds=60, calendar={"Hour": 1})
        with pytest.raises(SchedulerError, match=">= 1"):
            macos.render_plist("l", ["x"], interval_seconds=0)

    @pytest.mark.parametrize(("weekday", "num"), [("SUN", 0), ("MON", 1), ("SAT", 6)])
    def test_weekly_calendar(self, weekday: str, num: int):
        assert macos.weekly_calendar(weekday, "17:45") == {"Weekday": num, "Hour": 17, "Minute": 45}  # type: ignore[arg-type]

    def test_weekly_calendar_rejects_bad_input(self):
        with pytest.raises(SchedulerError):
            macos.weekly_calendar("MONDAY", "09:00")  # type: ignore[arg-type]
        with pytest.raises(SchedulerError):
            macos.weekly_calendar("MON", "25:00")

    def test_render_job_plist_digest(self, spec: ScheduleSpec):
        data = plistlib.loads(macos.render_job_plist(spec, "digest").encode("utf-8"))
        assert data["Label"] == "com.software-factory.work.digest"
        assert data["ProgramArguments"][-4:] == ["report", "--digest", "--since", "7d"]
        assert data["StartCalendarInterval"] == {"Weekday": 1, "Hour": 9, "Minute": 0}

    def test_launchctl_argv(self):
        p = Path("/Users/me/Library/LaunchAgents/com.software-factory.work.tick.plist")
        assert macos.render_launchctl_bootstrap(501, p) == ["launchctl", "bootstrap", "gui/501", str(p)]
        assert macos.render_launchctl_bootout(501, "lbl") == ["launchctl", "bootout", "gui/501/lbl"]
        assert macos.render_launchctl_load(p) == ["launchctl", "load", "-w", str(p)]
        assert macos.render_launchctl_unload(p) == ["launchctl", "unload", "-w", str(p)]
        assert macos.render_launchctl_print(501, "lbl") == ["launchctl", "print", "gui/501/lbl"]


class TestMacosFlows:
    def test_install_writes_plists_and_bootstraps(self, spec: ScheduleSpec, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/bin:/bin")
        agents = tmp_path / "LaunchAgents"
        runner = FakeRunner()
        report = macos.install(spec, runner, uid=501, agents_dir=agents)
        tick = agents / "com.software-factory.work.tick.plist"
        digest = agents / "com.software-factory.work.digest.plist"
        assert report.files == [tick, digest]
        assert report.jobs == ["com.software-factory.work.tick", "com.software-factory.work.digest"]
        data = plistlib.loads(tick.read_bytes())
        assert data["StartInterval"] == 20 * 60
        assert data["EnvironmentVariables"]["PATH"] == "/opt/homebrew/bin:/usr/bin:/bin"
        assert data["EnvironmentVariables"]["SOFTWARE_FACTORY_PROFILES_DIR"] == str(spec.profiles_dir)
        assert runner.argvs == [
            ["launchctl", "bootout", "gui/501/com.software-factory.work.tick"],
            ["launchctl", "bootstrap", "gui/501", str(tick)],
            ["launchctl", "bootout", "gui/501/com.software-factory.work.digest"],
            ["launchctl", "bootstrap", "gui/501", str(digest)],
        ]
        assert report.notes == []

    def test_install_falls_back_to_load(self, spec: ScheduleSpec, tmp_path: Path):
        def handler(argv, _):
            return (5, "", "Bootstrap failed: 5: Input/output error") if argv[1] == "bootstrap" else (0, "", "")

        runner = FakeRunner(handler)
        report = macos.install(spec.model_copy(update={"digest": False}), runner, uid=501, agents_dir=tmp_path)
        assert [a[1] for a in runner.argvs] == ["bootout", "bootstrap", "load"]
        assert runner.argvs[-1][:3] == ["launchctl", "load", "-w"]
        assert report.jobs == ["com.software-factory.work.tick"]
        assert any("legacy" in n for n in report.notes)

    def test_install_raises_when_both_fail(self, spec: ScheduleSpec, tmp_path: Path):
        runner = FakeRunner(lambda argv, _: (1, "", "nope") if argv[1] in ("bootstrap", "load") else (0, "", ""))
        with pytest.raises(SchedulerError, match="could not load"):
            macos.install(spec, runner, uid=501, agents_dir=tmp_path)

    def test_uninstall_bootouts_and_deletes(self, spec: ScheduleSpec, tmp_path: Path):
        macos.install(spec, FakeRunner(), uid=501, agents_dir=tmp_path)
        tick = tmp_path / "com.software-factory.work.tick.plist"
        digest = tmp_path / "com.software-factory.work.digest.plist"
        digest.unlink()  # digest never written -> "not installed"

        runner = FakeRunner(lambda argv, _: (0, "", "") if argv[2].endswith(".tick") else (3, "", "No such process"))
        report = macos.uninstall(spec, runner, uid=501, agents_dir=tmp_path)
        assert runner.argvs == [
            ["launchctl", "bootout", "gui/501/com.software-factory.work.tick"],
            ["launchctl", "bootout", "gui/501/com.software-factory.work.digest"],
        ]
        assert not tick.exists()
        assert report.files == [tick]
        assert report.jobs == ["com.software-factory.work.tick"]
        assert report.notes == ["com.software-factory.work.digest: not installed"]

    def test_uninstall_unload_fallback_when_plist_present(self, spec: ScheduleSpec, tmp_path: Path):
        tick = tmp_path / "com.software-factory.work.tick.plist"
        tick.write_text("<plist/>")
        runner = FakeRunner(lambda argv, _: (0, "", "") if argv[1] == "unload" else (3, "", "No such process"))
        report = macos.uninstall(spec, runner, uid=501, agents_dir=tmp_path)
        assert ["launchctl", "unload", "-w", str(tick)] in runner.argvs
        assert report.jobs == ["com.software-factory.work.tick"]
        assert not tick.exists()

    def test_show(self, spec: ScheduleSpec, tmp_path: Path):
        (tmp_path / "com.software-factory.work.digest.plist").write_text("<plist/>")
        runner = FakeRunner(lambda argv, _: (0, "state = running\n", "") if argv[2].endswith(".tick") else (113, "", ""))
        out = macos.show(spec, runner, uid=501, agents_dir=tmp_path)
        assert "state = running" in out
        assert "com.software-factory.work.digest: plist present but not loaded" in out

    def test_current_uid_requires_posix(self, monkeypatch):
        monkeypatch.delattr(macos.os, "getuid", raising=False)
        with pytest.raises(SchedulerError, match="POSIX"):
            macos.current_uid()


# ======================================================================================= Linux
class TestLinuxRender:
    @pytest.mark.parametrize(
        ("minutes", "expr"),
        [(1, "* * * * *"), (5, "*/5 * * * *"), (20, "*/20 * * * *"), (59, "*/59 * * * *"),
         (60, "0 * * * *"), (120, "0 */2 * * *"), (1440, "0 0 * * *")],
    )  # fmt: skip
    def test_interval_expr(self, minutes: int, expr: str):
        assert linux.cron_interval_expr(minutes) == expr

    @pytest.mark.parametrize("bad", [0, 90, 25 * 60])
    def test_interval_expr_rejects(self, bad: int):
        with pytest.raises(SchedulerError):
            linux.cron_interval_expr(bad)

    def test_weekly_expr(self):
        assert linux.cron_weekly_expr("MON", "09:00") == "0 9 * * 1"
        assert linux.cron_weekly_expr("SUN", "17:30") == "30 17 * * 0"
        with pytest.raises(SchedulerError):
            linux.cron_weekly_expr("MON", "9am")

    def test_cron_command_quotes_and_escapes(self):
        # PurePosixPath: the rendered line must be the same regardless of the host running the test.
        line = linux.cron_command(
            PurePosixPath("/home/me/dir with space/sf"),  # type: ignore[arg-type]
            ["/usr/local/bin/uv", "run", "--project", "/home/me/dir with space/sf", "factory", "tick"],
            {"SOFTWARE_FACTORY_PROFILES_DIR": "/p/100% mine"},
            PurePosixPath("/home/me/.local/share/sf/logs/scheduler.log"),  # type: ignore[arg-type]
        )
        assert line.startswith("cd '/home/me/dir with space/sf' && SOFTWARE_FACTORY_PROFILES_DIR='/p/100\\% mine' ")
        assert "/usr/local/bin/uv run --project '/home/me/dir with space/sf' factory tick" in line
        assert line.endswith(">> /home/me/.local/share/sf/logs/scheduler.log 2>&1")

    def test_render_cron_block(self, spec: ScheduleSpec):
        block = linux.render_spec_block(spec)
        lines = block.splitlines()
        assert block.endswith("\n")
        assert lines[0] == "# software-factory:work begin"
        assert lines[-1] == "# software-factory:work end"
        assert lines[1].startswith("*/20 * * * * cd ")
        assert "factory --profile work tick >>" in lines[1]
        assert lines[2].startswith("0 9 * * 1 cd ")
        assert "report --digest --since 7d >>" in lines[2]
        assert f"SOFTWARE_FACTORY_PROFILES_DIR='{spec.profiles_dir}'" in lines[1]

    def test_render_cron_block_without_digest(self, spec: ScheduleSpec):
        block = linux.render_spec_block(spec.model_copy(update={"digest": False}))
        assert len(block.splitlines()) == 3
        assert "--digest" not in block


class TestLinuxMerge:
    BLOCK = "# software-factory:work begin\n*/20 * * * * cd /sf && uv run tick\n# software-factory:work end\n"
    NEW = "# software-factory:work begin\n*/5 * * * * cd /sf && uv run tick\n# software-factory:work end\n"

    def test_append_to_empty(self):
        assert linux.merge_crontab("", self.BLOCK) == self.BLOCK

    def test_append_keeps_existing_with_separator(self):
        existing = "MAILTO=me\n0 * * * * /bin/other\n"
        merged = linux.merge_crontab(existing, self.BLOCK)
        assert merged == existing + "\n" + self.BLOCK

    def test_merge_is_idempotent(self):
        once = linux.merge_crontab("0 * * * * /bin/other\n", self.BLOCK)
        assert linux.merge_crontab(once, self.BLOCK) == once

    def test_replace_in_place_preserves_neighbours(self):
        existing = "# top\n" + self.BLOCK + "\n# software-factory:other begin\n* * * * * x\n# software-factory:other end\n"
        merged = linux.merge_crontab(existing, self.NEW)
        assert merged == "# top\n" + self.NEW + "\n# software-factory:other begin\n* * * * * x\n# software-factory:other end\n"
        assert merged.count("# software-factory:work begin") == 1

    def test_unterminated_block_is_replaced_to_eof(self):
        existing = "# keep\n# software-factory:work begin\n*/1 * * * * junk\n"
        assert linux.merge_crontab(existing, self.NEW) == "# keep\n" + self.NEW

    def test_merge_rejects_block_without_marker(self):
        with pytest.raises(SchedulerError, match="begin"):
            linux.merge_crontab("", "*/5 * * * * x\n")

    def test_remove_block(self):
        existing = "0 * * * * /bin/other\n\n" + self.BLOCK
        assert linux.remove_block(existing, "work") == "0 * * * * /bin/other\n"
        assert linux.remove_block(existing, "nope") == existing
        assert linux.remove_block(self.BLOCK, "work") == ""

    def test_extract_block(self):
        existing = "# top\n" + self.BLOCK + "# bottom\n"
        assert linux.extract_block(existing, "work") == self.BLOCK
        assert linux.extract_block(existing, "other") is None


class TestLinuxFlows:
    def test_install_reads_then_writes_merged(self, spec: ScheduleSpec):
        existing = "0 * * * * /bin/other\n"
        runner = FakeRunner(lambda argv, _: (0, existing, "") if argv == ["crontab", "-l"] else (0, "", ""))
        report = linux.install(spec, runner)
        assert runner.argvs == [["crontab", "-l"], ["crontab", "-"]]
        written = runner.calls[1][1]
        assert written == linux.merge_crontab(existing, linux.render_spec_block(spec))
        assert written.startswith(existing)
        assert report.jobs == ["software-factory-work-tick", "software-factory-work-digest"]
        assert spec.logs_dir.is_dir()

    def test_install_with_no_existing_crontab(self, spec: ScheduleSpec):
        runner = FakeRunner(
            lambda argv, _: (1, "", "no crontab for me") if argv == ["crontab", "-l"] else (0, "", "")
        )
        linux.install(spec, runner)
        assert runner.calls[1][1] == linux.render_spec_block(spec)

    def test_install_propagates_crontab_errors(self, spec: ScheduleSpec):
        with pytest.raises(SchedulerError, match="crontab -l failed"):
            linux.install(spec, FakeRunner(lambda argv, _: (1, "", "crontab: permission denied")))
        runner = FakeRunner(lambda argv, _: (0, "", "") if argv == ["crontab", "-l"] else (1, "", "bad crontab"))
        with pytest.raises(SchedulerError, match="crontab - failed"):
            linux.install(spec, runner)

    def test_uninstall_removes_block(self, spec: ScheduleSpec):
        existing = "0 * * * * /bin/other\n\n" + linux.render_spec_block(spec)
        runner = FakeRunner(lambda argv, _: (0, existing, "") if argv == ["crontab", "-l"] else (0, "", ""))
        report = linux.uninstall(spec, runner)
        assert runner.argvs == [["crontab", "-l"], ["crontab", "-"]]
        assert runner.calls[1][1] == "0 * * * * /bin/other\n"
        assert report.jobs == ["software-factory-work"]

    def test_uninstall_when_absent_does_not_write(self, spec: ScheduleSpec):
        runner = FakeRunner(lambda argv, _: (0, "0 * * * * /bin/other\n", ""))
        report = linux.uninstall(spec, runner)
        assert runner.argvs == [["crontab", "-l"]]
        assert report.notes == ["software-factory-work: not installed"]

    def test_show(self, spec: ScheduleSpec):
        block = linux.render_spec_block(spec)
        assert linux.show(spec, FakeRunner(lambda a, _: (0, "# x\n" + block, ""))) == block.rstrip("\n")
        assert linux.show(spec, FakeRunner(lambda a, _: (1, "", "no crontab for me"))) == "software-factory-work: not installed"


# ==================================================================================== dispatch
class TestDispatch:
    @pytest.mark.parametrize(
        ("platform", "expected"), [("win32", "windows"), ("darwin", "macos"), ("linux", "linux"), ("linux2", "linux")]
    )
    def test_current_os(self, monkeypatch, platform: str, expected: str):
        monkeypatch.setattr(scheduler.sys, "platform", platform)
        assert scheduler.current_os() == expected

    def test_current_os_unsupported(self, monkeypatch):
        monkeypatch.setattr(scheduler.sys, "platform", "freebsd14")
        with pytest.raises(SchedulerError, match="unsupported"):
            scheduler.current_os()

    def test_install_dispatches_linux(self, tmp_path: Path, tmp_paths):
        profile = Profile(name="work")
        profile.profile_dir = tmp_path / "profiles" / "work"
        runner = FakeRunner(lambda argv, _: (1, "", "no crontab for me") if argv == ["crontab", "-l"] else (0, "", ""))
        report = scheduler.install(
            profile, tmp_paths, interval_minutes=15, digest=True, runner=runner, os_name="linux", uv_path=UV
        )
        assert report.os == "linux" and report.action == "install"
        written = runner.calls[1][1]
        assert written.startswith("# software-factory:work begin\n*/15 * * * * ")
        assert f"SOFTWARE_FACTORY_PROFILES_DIR='{tmp_path / 'profiles'}'" in written
        assert "0 9 * * 1" in written

    def test_install_dispatches_windows(self, tmp_path: Path, tmp_paths):
        profile = Profile(name="work")
        runner = FakeRunner()
        report = scheduler.install(profile, tmp_paths, digest=False, runner=runner, os_name="windows", uv_path=UV)
        assert report.os == "windows"
        assert runner.argvs[0][:8] == ["schtasks", "/Create", "/TN", "software-factory-work-tick", "/SC", "MINUTE", "/MO", "20"]
        assert (tmp_paths.scripts / "software-factory-work-tick.cmd").is_file()

    def test_uninstall_dispatch_does_not_need_uv(self, tmp_paths, monkeypatch):
        monkeypatch.setattr(scheduler, "find_uv", lambda: None)
        runner = FakeRunner(lambda argv, _: (1, "", NOT_FOUND))
        report = scheduler.uninstall(Profile(name="work"), tmp_paths, runner=runner, os_name="windows")
        assert report.action == "uninstall"
        assert [a[3] for a in runner.argvs] == ["software-factory-work-tick", "software-factory-work-digest"]
        assert len(report.notes) == 2

    def test_show_dispatch(self, tmp_paths, monkeypatch):
        monkeypatch.setattr(scheduler, "find_uv", lambda: None)
        runner = FakeRunner(lambda argv, _: (1, "", "no crontab for me"))
        assert scheduler.show(Profile(name="work"), tmp_paths, runner=runner, os_name="linux") == "software-factory-work: not installed"

    def test_unknown_backend(self, tmp_paths):
        with pytest.raises(SchedulerError, match="unknown scheduler backend"):
            scheduler.install(Profile(name="w"), tmp_paths, runner=FakeRunner(), os_name="plan9", uv_path=UV)  # type: ignore[arg-type]

    def test_report_summary(self, tmp_paths):
        report = scheduler.install(Profile(name="w"), tmp_paths, runner=FakeRunner(), os_name="windows", uv_path=UV)
        text = report.summary()
        assert text.splitlines()[0] == "install (windows) profile w"
        assert "job   software-factory-w-tick" in text and "job   software-factory-w-digest" in text


# ================================================================================ default_runner
def test_default_runner_runs_a_real_process_with_list_args(tmp_path: Path):
    import sys

    script = tmp_path / "echo args.py"
    script.write_text("import sys; print(sys.argv[1:]); print(sys.stdin.read(), end=''); sys.exit(3)", encoding="utf-8")
    result = scheduler.default_runner([sys.executable, str(script), "a b", "c"], input="from stdin")
    assert result.returncode == 3 and not result.ok
    assert "['a b', 'c']" in result.stdout
    assert "from stdin" in result.stdout
    assert result.argv[1] == str(script)


def test_default_runner_missing_executable_raises():
    with pytest.raises(SchedulerError, match="not found"):
        scheduler.default_runner(["definitely-not-a-real-binary-xyz", "--version"])
