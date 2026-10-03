"""readiness.py: weighted scoring with an injected runner/workdir, caching, default providers."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from software_factory import readiness as readiness_mod
from software_factory.config import RepoOverride
from software_factory.discover import Cmd, Verification, discover
from software_factory.readiness import (
    NOT_RUN,
    TIMEOUT_EXIT_CODE,
    WEIGHTS,
    Readiness,
    ReadinessCheck,
    default_dev_probe,
    default_runner,
    descendants_from_table,
    port_in_use,
    readiness_cached,
    score_repo,
    shell_argv,
    split_command,
    worktree_workdir,
)
from software_factory.repos import RepoInfo

FIXTURES = Path(__file__).parent / "fixtures" / "repos"


class FakeRunner:
    """Exit codes per command: an int, or a list consumed one run at a time (last value repeats)."""

    def __init__(self, codes: dict[str, int | list[int]] | None = None, seconds: float = 0.5) -> None:
        self.codes = dict(codes or {})
        self.seconds = seconds
        self.calls: list[tuple[str, Path, float]] = []

    def __call__(self, cmd: str, cwd: Path, timeout: float) -> tuple[int, float, str]:
        self.calls.append((cmd, Path(cwd), timeout))
        spec = self.codes.get(cmd, 0)
        code = (spec.pop(0) if len(spec) > 1 else spec[0]) if isinstance(spec, list) else spec
        return code, self.seconds, f"output of {cmd}\nexit {code}"

    def runs_of(self, cmd: str) -> int:
        return sum(1 for c, _cwd, _t in self.calls if c == cmd)


class FakeWorkdir:
    def __init__(self, path: Path, fail: Exception | None = None) -> None:
        self.path = path
        self.fail = fail
        self.entered = 0
        self.exited = 0

    @contextmanager
    def __call__(self, repo):
        if self.fail:
            raise self.fail
        self.entered += 1
        try:
            yield self.path
        finally:
            self.exited += 1


def repo_info(path: Path, override: RepoOverride | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        slug="octo/fixture", name="fixture", dirname=path.name, path=path, default_branch="main",
        override=override or RepoOverride(),
    )  # fmt: skip


def verification(*cmds: str, setup: tuple[str, ...] = (), **kw) -> Verification:
    return Verification(
        setup=[Cmd(name="setup", command=c, source="test") for c in setup],
        verify=[Cmd(name=c.split()[-1], command=c, source="test") for c in cmds],
        source="test" if cmds else "none",
        **kw,
    )


def check(r: Readiness, name: str):
    c = r.check(name)
    assert c is not None, name
    return c


@pytest.fixture
def workdir(tmp_path: Path) -> FakeWorkdir:
    wd = tmp_path / "wt"
    wd.mkdir()
    return FakeWorkdir(wd)


# -------------------------------------------------------------------------------- scoring
def test_weights_sum_to_100_without_the_visual_check():
    assert sum(w for name, w in WEIGHTS.items() if name != "dev_server") == 100


def test_all_green_scores_100_and_runs_setup_once_and_each_command_twice(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only")
    v = discover(repo.path)
    runner = FakeRunner()
    r = score_repo(repo, v, runner=runner, workdir=workdir)
    assert r.score == 100
    assert r.missing == [] and r.flaky == []
    assert r.repo == "octo/fixture"
    assert all(c.ok for c in r.checks)
    assert [c.name for c in r.checks] == [
        "commands_discovered", "agent_doc_present", "verification_section",
        "commands_pass", "runtime_under_budget", "not_flaky",
    ]  # fmt: skip
    # setup once, then every verify command twice, all inside the throwaway workdir
    assert runner.calls[0][0] == "npm ci" and runner.runs_of("npm ci") == 1
    for cmd in v.verify_commands:
        assert runner.runs_of(cmd) == 2
    assert {cwd for _c, cwd, _t in runner.calls} == {workdir.path}
    assert workdir.entered == 1 and workdir.exited == 1
    assert r.runtime_s == pytest.approx(0.5 * len(v.verify))
    assert r.computed_at and "T" in r.computed_at


def test_missing_claude_md_loses_doc_and_section_points(tmp_path: Path, workdir: FakeWorkdir):
    repo = repo_info(tmp_path)  # no CLAUDE.md / AGENTS.md
    r = score_repo(repo, verification("npm test"), runner=FakeRunner(), workdir=workdir)
    assert r.missing == ["agent_doc_present", "verification_section"]
    assert r.score == 100 - WEIGHTS["agent_doc_present"] - WEIGHTS["verification_section"]
    assert "no CLAUDE.md" in check(r, "agent_doc_present").detail


def test_claude_md_without_verification_section(tmp_path: Path, workdir: FakeWorkdir):
    (tmp_path / "CLAUDE.md").write_text("# Repo\n\nJust prose.\n", encoding="utf-8")
    r = score_repo(repo_info(tmp_path), verification("npm test"), runner=FakeRunner(), workdir=workdir)
    assert r.missing == ["verification_section"]
    assert r.score == 100 - WEIGHTS["verification_section"]


def test_failing_command(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only")
    runner = FakeRunner({"npm test": 1})
    r = score_repo(repo, verification("npm run lint", "npm test"), runner=runner, workdir=workdir)
    assert r.missing == ["commands_pass"]
    assert r.score == 100 - WEIGHTS["commands_pass"]
    detail = check(r, "commands_pass").detail
    assert "`npm test` exit 1" in detail and "npm run lint" not in detail
    assert r.flaky == []  # fails both times: consistent, not flaky
    assert check(r, "not_flaky").ok


def test_flaky_command_alternating_exit_codes(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only")
    runner = FakeRunner({"npm test": [0, 1]})
    r = score_repo(repo, verification("npm run lint", "npm test"), runner=runner, workdir=workdir)
    assert r.flaky == ["npm test"]
    assert r.missing == ["not_flaky"]
    assert check(r, "commands_pass").ok  # first run passed
    assert "npm test" in check(r, "not_flaky").detail
    assert r.score == 100 - WEIGHTS["not_flaky"]


def test_slow_suite_exceeds_runtime_budget(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only")
    r = score_repo(repo, verification("npm run lint", "npm test"), runner=FakeRunner(seconds=70), workdir=workdir)
    assert r.runtime_s == pytest.approx(140.0)  # one pass, setup excluded
    assert r.missing == ["runtime_under_budget"]
    assert r.score == 100 - WEIGHTS["runtime_under_budget"]
    assert "140.0s" in check(r, "runtime_under_budget").detail
    # budget is configurable
    assert score_repo(repo, verification("npm test"), runner=FakeRunner(seconds=70), workdir=workdir, runtime_budget_s=100).score == 100


def test_static_only_marks_runtime_checks_not_run(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only")
    runner = FakeRunner()
    r = score_repo(repo, discover(repo.path), run_commands=False, runner=runner, workdir=workdir)
    assert runner.calls == [] and workdir.entered == 0
    for name in ("commands_pass", "runtime_under_budget", "not_flaky"):
        c = check(r, name)
        assert not c.ok and c.detail == NOT_RUN
    assert r.missing == ["commands_pass", "runtime_under_budget", "not_flaky"]
    assert r.score == WEIGHTS["commands_discovered"] + WEIGHTS["agent_doc_present"] + WEIGHTS["verification_section"]


def test_no_commands_discovered(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only")
    runner = FakeRunner()
    r = score_repo(repo, Verification(), runner=runner, workdir=workdir)
    assert runner.calls == []
    assert not check(r, "commands_discovered").ok
    assert check(r, "commands_pass").detail == "no commands to run"
    assert r.score == WEIGHTS["agent_doc_present"] + WEIGHTS["verification_section"]


def test_setup_failure_skips_verification_runs(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only")
    runner = FakeRunner({"npm ci": 7})
    r = score_repo(repo, verification("npm test", setup=("npm ci",)), runner=runner, workdir=workdir)
    assert runner.runs_of("npm test") == 0
    assert "setup failed: npm ci (exit 7)" in check(r, "commands_pass").detail
    assert check(r, "not_flaky").detail.startswith(NOT_RUN)
    assert r.missing == ["commands_pass", "runtime_under_budget", "not_flaky"]


def test_timed_out_command_is_not_rerun(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only")
    runner = FakeRunner({"npm test": TIMEOUT_EXIT_CODE})
    r = score_repo(repo, verification("npm run lint", "npm test"), runner=runner, workdir=workdir, command_timeout_s=5)
    assert runner.runs_of("npm test") == 1 and runner.runs_of("npm run lint") == 2
    assert all(t == 5 for _c, _cwd, t in runner.calls)
    assert r.missing == ["commands_pass"] and r.flaky == []


def test_workdir_failure_is_reported_not_raised(tmp_path: Path):
    repo = repo_info(FIXTURES / "claude_md_only")
    broken = FakeWorkdir(tmp_path, fail=RuntimeError("origin/main not fetched"))
    r = score_repo(repo, verification("npm test"), runner=FakeRunner(), workdir=broken)
    assert r.missing == ["commands_pass", "runtime_under_budget", "not_flaky"]
    assert "workdir unavailable: origin/main not fetched" in check(r, "commands_pass").detail


# ------------------------------------------------------------------------------ dev server
def test_visual_repo_dev_server_check(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only", RepoOverride(visual=True))
    v = verification("npm test", dev_command="npm run dev", dev_port=5177)
    probes: list[tuple[str, Path, int, float]] = []

    def probe_ok(cmd, cwd, port, timeout):
        probes.append((cmd, Path(cwd), port, timeout))
        return True, f"HTTP 200 on {port}"

    r = score_repo(repo, v, runner=FakeRunner(), workdir=workdir, dev_probe=probe_ok, dev_timeout_s=12)
    assert probes == [("npm run dev", workdir.path, 5177, 12)]
    assert check(r, "dev_server").ok and r.score == 100
    assert sum(c.weight for c in r.checks) == 100 + WEIGHTS["dev_server"]

    r = score_repo(repo, v, runner=FakeRunner(), workdir=workdir, dev_probe=lambda *_: (False, "no response"))
    assert r.missing == ["dev_server"]
    assert r.score == round(100 * 100 / (100 + WEIGHTS["dev_server"]))


def test_visual_repo_without_dev_command_fails_the_check(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only", RepoOverride(visual=True))
    r = score_repo(repo, verification("npm test"), runner=FakeRunner(), workdir=workdir, dev_probe=lambda *_: (True, ""))
    assert r.missing == ["dev_server"]
    assert "dev_command" in check(r, "dev_server").detail


def test_visual_repo_static_only_marks_dev_server_not_run(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only", RepoOverride(visual=True))
    v = verification("npm test", dev_command="npm run dev", dev_port=5177)
    r = score_repo(repo, v, run_commands=False, workdir=workdir, dev_probe=lambda *_: (True, ""))
    assert check(r, "dev_server").detail == NOT_RUN


def test_non_visual_repo_has_no_dev_server_check(workdir: FakeWorkdir):
    r = score_repo(repo_info(FIXTURES / "claude_md_only"), verification("npm test"), runner=FakeRunner(), workdir=workdir)
    assert r.check("dev_server") is None


def test_dev_probe_exception_is_reported(workdir: FakeWorkdir):
    repo = repo_info(FIXTURES / "claude_md_only", RepoOverride(visual=True))
    v = verification("npm test", dev_command="npm run dev", dev_port=5177)

    def boom(*_):
        raise OSError("port busy")

    r = score_repo(repo, v, runner=FakeRunner(), workdir=workdir, dev_probe=boom)
    assert "port busy" in check(r, "dev_server").detail and not check(r, "dev_server").ok


# -------------------------------------------------------------------------------- caching
class FakeState:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], object] = {}

    def cache_get(self, namespace: str, key: str):
        return self.store.get((namespace, key))

    def cache_set(self, namespace: str, key: str, value) -> None:
        self.store[(namespace, key)] = value


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=True,
    )  # fmt: skip


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "CLAUDE.md").write_text("# R\n\n## Test\n\n- `npm test`\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def test_readiness_cached_keyed_by_head_sha(git_repo: Path, workdir: FakeWorkdir):
    state, repo, runner = FakeState(), repo_info(git_repo), FakeRunner()
    first = readiness_cached(state, repo, runner=runner, workdir=workdir)
    assert first.score == 100 and len(first.head_sha) == 40
    assert ("readiness", "readiness:octo/fixture") in state.store
    assert ("discover", "discover:octo/fixture") in state.store  # verification discovered via discover_cached
    calls = len(runner.calls)

    assert readiness_cached(state, repo, runner=runner, workdir=workdir) == first
    assert len(runner.calls) == calls  # served from cache

    (git_repo / "CLAUDE.md").write_text("# R\n\n## Test\n\n- `npm run check`\n", encoding="utf-8")
    git(git_repo, "commit", "-q", "-am", "change")
    second = readiness_cached(state, repo, runner=runner, workdir=workdir)
    assert second.head_sha != first.head_sha and len(runner.calls) > calls
    assert runner.runs_of("npm run check") == 2


def test_readiness_cached_static_result_does_not_satisfy_a_full_request(git_repo: Path, workdir: FakeWorkdir):
    state, repo, runner = FakeState(), repo_info(git_repo), FakeRunner()
    static = readiness_cached(state, repo, run_commands=False, runner=runner, workdir=workdir)
    assert check(static, "commands_pass").detail == NOT_RUN and runner.calls == []
    full = readiness_cached(state, repo, runner=runner, workdir=workdir)
    assert full.score == 100 and runner.calls
    # ...but a cached full run satisfies a later static request, and force recomputes.
    assert readiness_cached(state, repo, run_commands=False, runner=runner, workdir=workdir) == full
    calls = len(runner.calls)
    readiness_cached(state, repo, force=True, runner=runner, workdir=workdir)
    assert len(runner.calls) > calls


def test_readiness_cached_accepts_explicit_verification(git_repo: Path, workdir: FakeWorkdir):
    state, runner = FakeState(), FakeRunner()
    r = readiness_cached(state, repo_info(git_repo), verification("make test"), runner=runner, workdir=workdir)
    assert runner.runs_of("make test") == 2 and r.score == 100


# ------------------------------------------------------------------------ default providers
def test_split_command_and_shell_argv():
    assert split_command('python -c "import x; x.y()" --flag') == ["python", "-c", "import x; x.y()", "--flag"]
    assert shell_argv("") is None
    assert shell_argv("definitely-not-a-real-executable-xyz --version") is None
    direct = shell_argv("git --version")
    assert direct is not None and direct[1:] == ["--version"] and Path(direct[0]).is_absolute()
    chained = shell_argv("git --version && git --version")
    assert chained is not None and chained[-1] == "git --version && git --version"
    assert chained[0].lower().endswith("cmd.exe") if os.name == "nt" else chained[0] == "/bin/sh"


def test_default_runner_runs_commands_and_captures_output(tmp_path: Path):
    code, secs, out = default_runner("git --version", tmp_path, 60)
    assert code == 0 and "git version" in out and secs >= 0
    code, _secs, out = default_runner("no-such-command-abc123 --x", tmp_path, 60)
    assert code == 127 and "not found" in out
    code, _secs, out = default_runner("git --version && git --no-such-flag", tmp_path, 60)
    assert code != 0 and "git version" in out  # shell operators go through the shell wrapper


def test_default_runner_times_out_and_kills_the_process(tmp_path: Path):
    cmd = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    started = time.monotonic()
    code, secs, out = default_runner(cmd, tmp_path, 1.5)
    assert code == TIMEOUT_EXIT_CODE
    assert "timed out" in out
    assert time.monotonic() - started < 15 and secs < 15


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_default_dev_probe_starts_polls_and_kills_a_real_server(tmp_path: Path):
    port = free_port()
    cmd = f'"{sys.executable}" -m http.server {port} --bind 127.0.0.1'
    ok, detail = default_dev_probe(cmd, tmp_path, port, 30)
    assert ok, detail
    assert "HTTP 200" in detail
    # the process tree was killed: the port is free again
    time.sleep(0.5)
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_default_dev_probe_reports_a_server_that_exits(tmp_path: Path):
    port = free_port()
    ok, detail = default_dev_probe(f'"{sys.executable}" -c "raise SystemExit(3)"', tmp_path, port, 10)
    assert not ok and "exited with 3" in detail
    ok, detail = default_dev_probe("no-such-dev-server-xyz", tmp_path, port, 10)
    assert not ok and "not found" in detail


def test_default_dev_probe_refuses_a_port_something_else_already_listens_on(tmp_path: Path):
    """The probe cannot tell whose server answered: an operator's own `npm run dev` (or another
    repo on the same port) must not make this repo's check pass."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(5)  # never accepted: Windows refuses connects once the backlog is full
        port = listener.getsockname()[1]
        assert port_in_use(port)
        ok, detail = default_dev_probe(f'"{sys.executable}" -c "raise SystemExit(3)"', tmp_path, port, 10)
    assert not ok, detail
    assert f"port {port} is already in use before the dev server was started" in detail
    assert "exited with 3" not in detail  # the command was never started
    assert not port_in_use(free_port())


def _ipv6_loopback_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="no IPv6 loopback")
def test_default_dev_probe_sees_a_server_listening_on_ipv6_loopback_only(tmp_path: Path):
    """Vite on Windows binds ``localhost`` -> ``::1`` only; probing 127.0.0.1 alone reported a running
    dev server as 'no response'."""
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
        probe.bind(("::1", 0))
        port = probe.getsockname()[1]
    cmd = f'"{sys.executable}" -m http.server {port} --bind ::1'
    ok, detail = default_dev_probe(cmd, tmp_path, port, 30)
    assert ok, detail
    assert f"[::1]:{port}" in detail


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="no IPv6 loopback")
def test_port_in_use_checks_both_loopback_families():
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
        listener.bind(("::1", 0))
        listener.listen(5)
        port = listener.getsockname()[1]
        assert port_in_use(port)
        assert not port_in_use(port, host="127.0.0.1")


def test_readiness_check_not_run_marks_only_deliberately_skipped_checks():
    assert ReadinessCheck(name="commands_pass", ok=False, detail=NOT_RUN, weight=25).not_run
    assert not ReadinessCheck(name="commands_pass", ok=False, detail=f"{NOT_RUN} (setup failed)", weight=25).not_run
    assert not ReadinessCheck(name="commands_pass", ok=False, detail="`npm test` exit 1: boom", weight=25).not_run
    assert not ReadinessCheck(name="commands_pass", ok=True, detail="1 command(s) passed", weight=25).not_run


# ------------------------------------------------------------------------- worktree workdir
def test_worktree_workdir_uses_its_own_temp_dir_and_creates_no_hooks_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bare_repo_factory, git_repo_factory
):
    """Without a manager the provider makes its own under a fresh temp dir, removed again on exit.
    No hooks dir is created anywhere (neither inside it nor in the shared OS temp root, where any
    local process could plant a hook): the manager disables hooks with the unplantable path."""
    clone = git_repo_factory("repo", origin=bare_repo_factory("origin"))
    repo = RepoInfo(slug="JaneDoe/repo", name="repo", dirname="repo", path=clone, default_branch="main", override=RepoOverride())
    temp_root = tmp_path / "os temp"
    temp_root.mkdir()
    made: list[Path] = []

    def mkdtemp(prefix: str = "tmp", **_kw) -> str:
        directory = temp_root / f"{prefix}{len(made)}"
        directory.mkdir()
        made.append(directory)
        return str(directory)

    monkeypatch.setattr(readiness_mod.tempfile, "mkdtemp", mkdtemp)

    with worktree_workdir()(repo) as path:
        assert len(made) == 1 and made[0].name.startswith("sf-readiness-")
        scratch = made[0]
        assert path.is_dir() and path.resolve().is_relative_to((scratch / "wt").resolve())
        assert not (scratch / "no-hooks").exists() and not (temp_root / "no-hooks").exists()
        assert (path / ".git").exists()

    assert not scratch.exists() and list(temp_root.iterdir()) == []
    listing = subprocess.run(["git", "-C", str(clone), "worktree", "list", "--porcelain"], capture_output=True, text=True, timeout=30)
    assert listing.stdout.count("worktree ") == 1  # only the clone itself: the readiness checkout is gone


# ---------------------------------------------------------------------------------- kill_tree
class _FakePosix:
    """Stand-ins for the POSIX process calls ``_kill_tree_posix`` makes (they do not exist on Windows)."""

    def __init__(self, groups: dict[int, int], engine_group: int) -> None:
        self.groups = groups
        self.engine_group = engine_group
        self.killpg_calls: list[int] = []
        self.kill_calls: list[int] = []

    def install(self, monkeypatch: pytest.MonkeyPatch, descendants: list[int]) -> None:
        monkeypatch.setattr(os, "getpgid", self.getpgid, raising=False)
        monkeypatch.setattr(os, "getpgrp", lambda: self.engine_group, raising=False)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: self.killpg_calls.append(pgid), raising=False)
        monkeypatch.setattr(os, "kill", lambda pid, sig: self.kill_calls.append(pid))
        monkeypatch.setattr(readiness_mod, "_posix_descendants", lambda pid: list(descendants))

    def getpgid(self, pid: int) -> int:
        if pid not in self.groups:
            raise ProcessLookupError(pid)
        return self.groups[pid]


def test_posix_kill_tree_never_kills_the_engines_own_process_group(monkeypatch: pytest.MonkeyPatch):
    """Regression: POSIX ``kill_tree`` was ``killpg(getpgid(pid))``. The Claude CLI the SDK spawns
    shares the engine's process group (no ``start_new_session``), so killing its tree -- which
    ``run_worker`` now does after every session -- would have SIGKILLed the engine itself. Such a
    pid is killed with its descendants, one by one; a pid leading its own group still takes the group."""
    fake = _FakePosix(groups={500: 100, 501: 100, 502: 777}, engine_group=100)
    fake.install(monkeypatch, descendants=[501, 502, os.getpid()])
    readiness_mod._kill_tree_posix(500)
    assert fake.killpg_calls == []  # never the engine's group
    assert fake.kill_calls == [500, 501, 502]  # the CLI, then its descendants (never the engine's own pid)

    leader = _FakePosix(groups={900: 900}, engine_group=100)
    leader.install(monkeypatch, descendants=[901])
    readiness_mod._kill_tree_posix(900)
    assert leader.killpg_calls == [900] and leader.kill_calls == [900, 901]

    gone = _FakePosix(groups={}, engine_group=100)
    gone.install(monkeypatch, descendants=[1])
    readiness_mod._kill_tree_posix(42)
    assert gone.killpg_calls == [] and gone.kill_calls == []


def test_descendants_from_table_walks_the_whole_tree_breadth_first():
    table = """
      1     0
     10     1
     11    10
     12    10
     13    11
     20     1
    abc   def
     14    13
    """
    assert descendants_from_table(10, table) == [11, 12, 13, 14]
    assert descendants_from_table(20, table) == []
    assert descendants_from_table(99, "") == []
    assert descendants_from_table(5, "5 6\n6 5\n") == [6]  # a ppid cycle (pid reuse) terminates
