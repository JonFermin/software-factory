"""Local end-to-end sandbox for software-factory.

Runs the REAL engine (scan -> filter -> dispatch -> handlers -> Claude Agent SDK workers -> policy ->
push policy -> effects) with REAL git, against a sandbox that never leaves this machine:

- "GitHub" is a local bare repository plus an in-process fake of the few ``gh`` reads/writes the
  engine uses. Its CI really runs the sandbox repo's tests on the head commit.
- "Linear" is an in-process fake tracker.
- Notifications go to the audit log and a fake toast.

It spends real model tokens (one worker session per scenario, a few dollars at most) and writes only
under a temp directory. Two ticks run: the first must do the work, the second must find nothing to
do. Then the refinement is reverted. Every outward call is checked against the allowed-writes list.

    uv run python scripts/e2e_local.py [--effort medium] [--cli-path PATH] [--profile NAME] [--keep]

Worker credentials: with ``--profile NAME`` the workers use that profile's provider settings and
secrets. Without it they fall back to this machine's existing Claude Code login (the profile token
is a placeholder that is stripped from the worker environment). Model id defaults to
``claude-opus-5-5``; the CLI must be new enough for it (``--cli-path`` or ``claude update``).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(tempfile.mkdtemp(prefix="sf-e2e-"))
os.environ["SOFTWARE_FACTORY_DATA_DIR"] = str(ROOT / "data")  # never touch the real data dir
os.environ.pop("CLAUDECODE", None)

from software_factory import dispatch, providers  # noqa: E402
from software_factory.actions import Kind  # noqa: E402
from software_factory.config import (  # noqa: E402
    GitHubConfig,
    NotifyConfig,
    Profile,
    ProviderConfig,
    RepoOverride,
    ReposConfig,
    load_profile,
)
from software_factory.effects import LiveEffects  # noqa: E402
from software_factory.kinds.base import Deps  # noqa: E402
from software_factory.kinds.refine import revert_refine  # noqa: E402
from software_factory.notify import Audit, Notifier  # noqa: E402
from software_factory.paths import ProfilePaths  # noqa: E402
from software_factory.repos import RepoIndex, RepoInfo  # noqa: E402
from software_factory.resolve import Resolver  # noqa: E402
from software_factory.sources.models import Check, Feedback, PrCommit, PullRequest, ReviewThread  # noqa: E402
from software_factory.state import State  # noqa: E402
from software_factory.trackers.base import Issue, IssueRef, TrackerError  # noqa: E402
from software_factory.worktree import WorktreeManager  # noqa: E402

SLUG = "sandbox/calc"
ME = "me"
VERIFY = "uv run --no-project --with pytest python -m pytest -q -p no:cacheprovider"
EPOCH = "2026-10-01T12:00:00+00:00"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------ git
def git(*args: str, cwd: Path, check: bool = True, env: dict | None = None) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, **(env or {})}, timeout=120,
    )  # fmt: skip
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def commit_all(repo: Path, message: str, when: str) -> str:
    git("add", "-A", cwd=repo)
    env = {"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    git("commit", "-q", "-m", message, cwd=repo, env=env)
    return git("rev-parse", "HEAD", cwd=repo)


def build_sandbox() -> tuple[Path, Path]:
    """Bare origin + the user's clone, with main and four PR branches."""
    bare = ROOT / "github" / "calc.git"
    clone = ROOT / "repos" / "calc"
    bare.parent.mkdir(parents=True)
    clone.parent.mkdir(parents=True)
    git("init", "-q", "--bare", "-b", "main", str(bare), cwd=ROOT)
    git("init", "-q", "-b", "main", str(clone), cwd=ROOT)
    for key, value in (("user.name", "Sandbox Me"), ("user.email", "me@example.com"), ("core.autocrlf", "false")):
        git("config", key, value, cwd=clone)
    git("remote", "add", "origin", str(bare), cwd=clone)

    write(clone, "CLAUDE.md", (
        "# calc\n\nA tiny calculator library used as a software-factory sandbox.\n\n"
        f"## Verification\n\n```\n{VERIFY}\n```\n"
    ))  # fmt: skip
    write(clone, ".gitignore", "__pycache__/\n.pytest_cache/\n")
    write(clone, "conftest.py", "")
    write(clone, "calc.py", '"""Calculator."""\n\n\ndef add(a, b):\n    return a + b\n')
    write(clone, "tests/test_calc.py", "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    commit_all(clone, "Initial calc", "2026-09-28T10:00:00+00:00")
    git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=clone)

    def branch(name: str, files: dict[str, str], message: str, when: str) -> None:
        git("switch", "-q", "-c", name, "main", cwd=clone)
        for rel, text in files.items():
            write(clone, rel, text)
        commit_all(clone, message, when)
        git("push", "-q", "origin", f"HEAD:refs/heads/{name}", cwd=clone)
        git("switch", "-q", "main", cwd=clone)

    # PR 1 (ci): multiply is wrong, its test fails.
    branch("feat/multiply", {
        "calc.py": '"""Calculator."""\n\n\ndef add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a + b\n',
        "tests/test_multiply.py": "from calc import multiply\n\n\ndef test_multiply():\n    assert multiply(2, 3) == 6\n",
    }, "Add multiply", "2026-09-30T10:00:00+00:00")  # fmt: skip
    # PR 2 (rc): a reviewer asks for sum() instead of the manual loop.
    branch("feat/total", {
        "stats.py": "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
        "tests/test_stats.py": "from stats import total\n\n\ndef test_total():\n    assert total([1, 2, 3]) == 6\n",
    }, "Add total", "2026-09-30T10:05:00+00:00")  # fmt: skip
    # PR 3 (conflict): both sides rewrite the module docstring.
    branch("feat/docstring", {
        "calc.py": '"""Simple calculator helpers."""\n\n\ndef add(a, b):\n    return a + b\n',
    }, "Reword docstring", "2026-09-30T10:10:00+00:00")  # fmt: skip
    # PR 4 (rv, a teammate's PR): divide has no zero handling.
    branch("feat/divide", {
        "calc.py": '"""Calculator."""\n\n\ndef add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n',
        "tests/test_divide.py": "from calc import divide\n\n\ndef test_divide():\n    assert divide(6, 3) == 2\n",
    }, "Add divide", "2026-09-30T10:15:00+00:00")  # fmt: skip
    # main moves on after the docstring branch was cut -> PR 3 conflicts.
    write(clone, "calc.py", '"""Calculator utilities for the sandbox."""\n\n\ndef add(a, b):\n    return a + b\n')
    commit_all(clone, "Reword module docstring on main", "2026-09-30T11:00:00+00:00")
    git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=clone)
    return bare, clone


# ----------------------------------------------------------------------------------- fake gh
class SandboxGitHub:
    """The ``gh`` surface the engine uses, backed by the local bare repo. CI runs the real tests."""

    def __init__(self, bare: Path) -> None:
        self.bare = bare
        self.calls: list[tuple[str, Any]] = []
        self._ci: dict[str, tuple[bool, str]] = {}
        self.prs: dict[int, dict[str, Any]] = {
            1: {"head": "feat/multiply", "title": "Add multiply", "author": ME},
            2: {"head": "feat/total", "title": "Add total", "author": ME, "feedback": [
                Feedback(id="c-1", author="reviewer", body="Please use the built-in sum() instead of the manual loop in total().",
                         created_at="2026-09-30T12:00:00+00:00", kind="thread", path="stats.py", line=2, thread_id="t-1"),
            ]},  # fmt: skip
            3: {"head": "feat/docstring", "title": "Reword docstring", "author": ME},
            4: {"head": "feat/divide", "title": "Add divide", "author": "teammate", "review_requested": True},
        }
        self.bodies: dict[int, str] = {n: f"PR {n}" for n in self.prs}

    # -- helpers
    def _sha(self, ref: str) -> str:
        return git("rev-parse", f"refs/heads/{ref}", cwd=self.bare)

    def ci_result(self, sha: str) -> tuple[bool, str]:
        if sha not in self._ci:
            tree = ROOT / "ci" / sha[:12]
            if tree.exists():
                shutil.rmtree(tree)
            tree.mkdir(parents=True)
            data = subprocess.run(["git", "archive", sha], cwd=self.bare, capture_output=True, check=True).stdout
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                tar.extractall(tree, filter="data")
            proc = subprocess.run(VERIFY.split(), cwd=tree, capture_output=True, text=True, timeout=300,
                                  encoding="utf-8", errors="replace")  # fmt: skip
            self._ci[sha] = (proc.returncode == 0, proc.stdout + proc.stderr)
        return self._ci[sha]

    def _run_id(self, sha: str) -> int:
        return zlib.crc32(sha.encode()) & 0x7FFFFFFF

    def _mergeable(self, head: str, base: str) -> str:
        if subprocess.run(["git", "merge-base", "--is-ancestor", base, head], cwd=self.bare).returncode == 0:
            return "MERGEABLE"
        mb = git("merge-base", base, head, cwd=self.bare)
        out = git("merge-tree", mb, base, head, cwd=self.bare, check=False)
        return "CONFLICTING" if "<<<<<<<" in out else "MERGEABLE"

    def _pr(self, number: int, *, full: bool = True) -> PullRequest:
        meta = self.prs[number]
        if meta.get("state", "OPEN") != "OPEN":
            return PullRequest(repo=SLUG, number=number, state=meta["state"], title=meta["title"])
        head, base = self._sha(meta["head"]), self._sha("main")
        pr = PullRequest(
            repo=SLUG, number=number, title=meta["title"], url=f"https://github.local/{SLUG}/pull/{number}",
            author=meta["author"], body=self.bodies.get(number, ""), is_draft=meta.get("draft", False),
            head_ref_name=meta["head"], head_ref_oid=head, base_ref_name="main", base_ref_oid=base, head_repo=SLUG,
            created_at="2026-09-30T10:00:00+00:00", updated_at=now_iso(),
            review_requested_from_me=bool(meta.get("review_requested")),
        )  # fmt: skip
        if full:
            passed, _ = self.ci_result(head)
            pr.mergeable = self._mergeable(head, base)  # type: ignore[assignment]
            pr.checks = [Check(name="tests", state="success" if passed else "failure", workflow="ci",
                               details_url=f"https://github.local/{SLUG}/actions/runs/{self._run_id(head)}",
                               run_id=self._run_id(head), completed_at=now_iso())]  # fmt: skip
            pr.feedback = list(meta.get("feedback", []))
            pr.threads = [ReviewThread(id=f.thread_id, is_resolved=False, is_outdated=False, path=f.path, line=f.line,
                                       comments=[f]) for f in pr.feedback if f.thread_id]  # fmt: skip
        return pr

    # -- reads
    def auth_check(self) -> None:
        self.calls.append(("auth_check", None))

    def login(self) -> str:
        return ME

    def my_open_prs(self, repos: list[str], max_age_days: int) -> list[PullRequest]:
        return [self._pr(n, full=False) for n, m in self.prs.items() if m["author"] == ME and m.get("state", "OPEN") == "OPEN"]

    def review_requested(self, max_age_days: int) -> list[PullRequest]:
        return [self._pr(n, full=False) for n, m in self.prs.items() if m.get("review_requested")]

    def pr_details(self, repo: str, number: int) -> PullRequest:
        self.calls.append(("pr_details", number))
        return self._pr(number)

    def pr_state(self, repo: str, number: int) -> PullRequest:
        return self._pr(number, full=False)

    def failed_log(self, repo: str, run_id: int, max_bytes: int = 60_000) -> str:
        for sha, (_, log) in self._ci.items():
            if self._run_id(sha) == run_id:
                return log[-max_bytes:]
        return ""

    def latest_workflow_run(self, repo: str, branch: str, workflow: str) -> dict | None:
        sha = self._sha(branch)
        passed, _ = self.ci_result(sha)
        return {"conclusion": "success" if passed else "failure", "headSha": sha, "url": f"https://ci.local/{sha}",
                "createdAt": now_iso()}  # fmt: skip

    def pr_commits(self, repo: str, number: int) -> list[PrCommit]:
        meta = self.prs[number]
        out = git("log", "--format=%H%x09%cI%x09%ae%x09%s", f"refs/heads/main..refs/heads/{meta['head']}", cwd=self.bare)
        commits = []
        for line in reversed([ln for ln in out.splitlines() if ln]):
            sha, when, email, subject = line.split("\t", 3)
            commits.append(PrCommit(sha=sha, committed_at=when, author_email=email, message=subject))
        return commits

    def pr_body(self, repo: str, number: int) -> str:
        return self.bodies.get(number, "")

    def find_open_pr_for_branch(self, repo: str, branch: str) -> PullRequest | None:
        for n, meta in self.prs.items():
            if meta["head"] == branch and meta.get("state", "OPEN") == "OPEN":
                return self._pr(n, full=False)
        return None

    # -- writes (only these two exist; anything else the engine tried would raise AttributeError)
    def create_draft_pr(self, repo: str, head: str, base: str, title: str, body: str) -> PullRequest:
        number = max(self.prs) + 1
        self.prs[number] = {"head": head, "title": title, "author": ME, "draft": True}
        self.bodies[number] = body
        self.calls.append(("create_draft_pr", {"number": number, "head": head, "base": base, "title": title}))
        return self._pr(number, full=False)

    def edit_pr_body(self, repo: str, number: int, body: str) -> None:
        self.calls.append(("edit_pr_body", number))
        self.bodies[number] = body


# ------------------------------------------------------------------------------ fake tracker
class SandboxTracker:
    name = "sandbox-linear"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.labels = {"factory:refine", "factory:plan", "factory:build", "repo:calc"}
        team = {"team_id": "team-1", "team_key": "ABC", "project_id": "proj-1", "project_name": "Calc"}
        self.issues: dict[str, Issue] = {
            "i-1": Issue(id="i-1", key="ABC-1", title="Add subtract(a, b)", **team, state_type="unstarted",
                         description="Add a `subtract(a, b)` function to calc.py.\n\nAcceptance criteria:\n"
                                     "- subtract(5, 3) returns 2\n- subtract(0, 4) returns -4\n- covered by a test",
                         labels=["factory:build", "repo:calc"], updated_at=EPOCH, branch_name="me/abc-1-add-subtract"),
            "i-2": Issue(id="i-2", key="ABC-2", title="make calc better", **team, state_type="backlog",
                         description="calc should handle more stuff, maybe errors? dunno", labels=["factory:refine", "repo:calc"],
                         updated_at=EPOCH),
            "i-3": Issue(id="i-3", key="ABC-3", title="Power and modulo operations", **team, state_type="backlog",
                         description="We want `power(a, b)` and `modulo(a, b)` in calc, each with tests and a short README "
                                     "section documenting both.", labels=["factory:plan"], updated_at=EPOCH),
        }  # fmt: skip
        self._seq = 10

    def _touch(self, issue: Issue) -> None:
        issue.updated_at = now_iso()

    def ping(self) -> str:
        return "Sandbox Me"

    def list_issues_with_label(self, label: str) -> list[Issue]:
        return [i.model_copy(deep=True) for i in self.issues.values()
                if i.has_label(label) and i.state_type not in ("completed", "canceled")]  # fmt: skip

    def get_issue(self, id_or_key: str) -> Issue:
        for issue in self.issues.values():
            if id_or_key in (issue.id, issue.key):
                return issue.model_copy(deep=True)
        raise TrackerError(f"no issue {id_or_key}")

    def sibling_issues(self, issue: Issue, limit: int = 50) -> list[Issue]:
        return [i.model_copy(deep=True) for i in self.issues.values() if i.id != issue.id][:limit]

    def labels_exist(self, names: list[str]) -> dict[str, bool]:
        return {n: n in self.labels for n in names}

    def ensure_label(self, team_id: str, name: str) -> str:
        if name not in self.labels:
            raise TrackerError(f"label {name} missing")
        return f"label-{name}"

    # writes (effects only)
    def create_sub_issue(self, parent: Issue, title: str, description: str, labels: list[str]) -> Issue:
        self._seq += 1
        applied = [lbl for lbl in labels if lbl in self.labels]
        child = Issue(id=f"i-{self._seq}", key=f"ABC-{self._seq}", title=title, description=description,
                      labels=applied, state_type="backlog", team_id=parent.team_id, team_key=parent.team_key,
                      project_id=parent.project_id, project_name=parent.project_name, updated_at=now_iso(),
                      parent=IssueRef(id=parent.id, key=parent.key, state_type=parent.state_type, title=parent.title))  # fmt: skip
        self.issues[child.id] = child
        stored = self.issues[parent.id]
        stored.children.append(IssueRef(id=child.id, key=child.key, state_type="backlog", title=title))
        self._touch(stored)
        self.calls.append(("create_sub_issue", {"parent": parent.key, "title": title, "labels": applied}))
        return child.model_copy(deep=True)

    def add_blocked_by(self, issue_id: str, blocker_id: str) -> None:
        blocker = self.issues[blocker_id]
        self.issues[issue_id].blocked_by.append(IssueRef(id=blocker.id, key=blocker.key, state_type=blocker.state_type))
        self.calls.append(("add_blocked_by", (self.issues[issue_id].key, blocker.key)))

    def move_to_state_type(self, issue_id: str, state_type: str) -> None:
        self.issues[issue_id].state_type = state_type
        self._touch(self.issues[issue_id])
        self.calls.append(("move_to_state_type", (self.issues[issue_id].key, state_type)))

    def update_issue_text(self, issue_id: str, title: str, description: str) -> None:
        issue = self.issues[issue_id]
        issue.title, issue.description = title, description
        self._touch(issue)
        self.calls.append(("update_issue_text", (issue.key, time.monotonic())))

    def remove_label(self, issue_id: str, label: str) -> None:
        issue = self.issues[issue_id]
        issue.labels = [lbl for lbl in issue.labels if lbl.lower() != label.lower()]
        self._touch(issue)
        self.calls.append(("remove_label", (issue.key, label)))


class SandboxToaster:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, float]] = []

    def send(self, title: str, message: str) -> None:
        self.sent.append((title, message, time.monotonic()))


# ---------------------------------------------------------------------------------- harness
class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((bool(ok), name, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail and not ok else ''}")

    @property
    def ok(self) -> bool:
        return all(r[0] for r in self.rows)


def make_profile(args: argparse.Namespace) -> Profile:
    if args.profile:
        base = load_profile(args.profile)
        provider = base.provider.model_copy()
        secrets = dict(base.secrets)
    else:
        provider = ProviderConfig(kind="oauth")
        secrets = {"CLAUDE_CODE_OAUTH_TOKEN": "placeholder-use-cli-login"}
        _strip_token_from_worker_env()
    provider.effort = args.effort
    if args.cli_path:
        provider.cli_path = args.cli_path
    profile = Profile(
        name="e2e", mode="full", enabled_kinds=[Kind.CI, Kind.CONFLICT, Kind.RC, Kind.RV, Kind.REFINE, Kind.PLAN, Kind.BUILD],
        max_parallel=3, session_timeout_min=args.timeout_min, provider=provider,
        github=GitHubConfig(login=ME, owners=["sandbox"]), repos=ReposConfig(root=str(ROOT / "repos")),
        notify=NotifyConfig(desktop=True, slack=False),
    )  # fmt: skip
    profile.secrets = secrets
    return profile


def _strip_token_from_worker_env() -> None:
    real = providers.worker_env

    def worker_env(profile: Profile) -> dict[str, str]:
        env = real(profile)
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)  # the CLI falls back to this machine's own login
        return env

    providers.worker_env = worker_env


def build_deps(profile: Profile, gh: SandboxGitHub, tracker: SandboxTracker, clone: Path, toaster: SandboxToaster) -> Deps:
    paths = ProfilePaths.for_profile(profile.name).ensure()
    state = State(paths.state_db)
    audit = Audit(paths.audit_log, profile.name)
    repo = RepoInfo(slug=SLUG, name="calc", dirname="calc", path=clone, default_branch="main", override=RepoOverride())
    repos = RepoIndex([repo])
    return Deps(
        profile=profile, paths=paths, state=state, gh=gh, tracker=tracker, repos=repos,  # type: ignore[arg-type]
        effects=LiveEffects(profile, paths, state, gh, tracker, audit),  # type: ignore[arg-type]
        notifier=Notifier(profile, audit, toaster=toaster), audit=audit, wt_manager=WorktreeManager(paths.worktrees),
        resolver=Resolver(profile, tracker, repos, state, gh=gh),  # type: ignore[arg-type]
    )  # fmt: skip


ALLOWED_GH_WRITES = {"create_draft_pr", "edit_pr_body"}
ALLOWED_TRACKER_WRITES = {"create_sub_issue", "add_blocked_by", "move_to_state_type", "update_issue_text", "remove_label"}


async def main(args: argparse.Namespace) -> int:
    print(f"sandbox: {ROOT}")
    bare, clone = build_sandbox()
    gh, tracker, toaster = SandboxGitHub(bare), SandboxTracker(), SandboxToaster()
    profile = make_profile(args)
    deps = build_deps(profile, gh, tracker, clone, toaster)
    heads_before = {n: gh._sha(m["head"]) for n, m in gh.prs.items()}
    print("baseline CI:", {n: gh.ci_result(h)[0] for n, h in heads_before.items()}, "main:", gh.ci_result(gh._sha("main"))[0])

    print("\n== tick 1 (real workers) ==")
    t0 = time.monotonic()
    report1 = await dispatch.run_tick(profile, deps.paths, deps=deps)
    print(f"tick 1 exit {report1.exit_code} in {time.monotonic() - t0:.0f}s")
    runs = deps.state.runs()

    print("\n== tick 2 (must be a no-op) ==")
    gh.calls.clear()
    report2 = await dispatch.run_tick(profile, deps.paths, deps=deps)
    print(f"tick 2 exit {report2.exit_code}")
    runs_after_tick2 = len(deps.state.runs())

    print("\n== revert-refine ABC-2 ==")
    revert = await revert_refine(deps, "ABC-2")
    print("revert:", revert.status, revert.note[:200])

    r = Results()
    print("\n== checks ==")
    r.check("tick 1 exits 0", report1.exit_code == 0, f"exit {report1.exit_code}")
    by = {(run["kind"], run["target"]): run for run in runs}
    for key in [("ci", f"{SLUG}#1"), ("rc", f"{SLUG}#2"), ("conflict", f"{SLUG}#3"), ("rv", f"{SLUG}#4"),
                ("build", "ABC-1"), ("refine", "ABC-2"), ("plan", "ABC-3")]:  # fmt: skip
        run = by.get(key)
        r.check(f"{key[0]} {key[1]} done", bool(run) and run.get("outcome") == "done",
                json.dumps({k: run.get(k) for k in ("outcome", "note", "pushed")} if run else "no run", default=str)[:300])  # fmt: skip

    # ci: pushed a fix, CI green on the new head
    head1 = gh._sha("feat/multiply")
    r.check("ci pushed a new head to feat/multiply", head1 != heads_before[1])
    r.check("ci: CI green on the pushed head", gh.ci_result(head1)[0], gh.ci_result(head1)[1][-300:])
    # rc: sum() used, pushed, CI green
    head2 = gh._sha("feat/total")
    stats = git("show", f"{head2}:stats.py", cwd=bare)
    r.check("rc pushed and stats.total uses sum()", head2 != heads_before[2] and "sum(" in stats, stats[:200])
    r.check("rc: CI green on the pushed head", gh.ci_result(head2)[0])
    # conflict: merge commit containing main, mergeable, CI green
    head3 = gh._sha("feat/docstring")
    parents = git("rev-list", "--parents", "-n", "1", head3, cwd=bare).split()
    r.check("conflict pushed a merge commit of main", head3 != heads_before[3] and len(parents) == 3)
    r.check("conflict: PR is mergeable now", gh._mergeable(head3, gh._sha("main")) == "MERGEABLE")
    r.check("conflict: CI green on the merge", gh.ci_result(head3)[0])
    # rv: local review only, nothing pushed
    review = deps.paths.reviews
    reviews = list(review.glob("*.md"))
    r.check("rv wrote a local review draft", bool(reviews), str(review))
    r.check("rv pushed nothing", gh._sha("feat/divide") == heads_before[4])
    # build: branch pushed, draft PR with Fixes/evidence/footer, issue started, CI green
    build_prs = [n for n, m in gh.prs.items() if m["head"] == "me/abc-1-add-subtract"]
    r.check("build opened exactly one draft PR", len(build_prs) == 1 and gh.prs[build_prs[0]].get("draft"), str(build_prs))
    if build_prs:
        body = gh.bodies[build_prs[0]]
        r.check("build PR body: Fixes ABC-1", "Fixes ABC-1" in body)
        r.check("build PR body: evidence section", "<!-- factory:evidence -->" in body and "<!-- /factory:evidence -->" in body)
        r.check("build PR body: disclosure footer last", body.rstrip().endswith(profile.disclosure_footer.strip().splitlines()[-1]))
        bhead = gh._sha("me/abc-1-add-subtract")
        r.check("build: CI green on the pushed branch", gh.ci_result(bhead)[0])
        r.check("build: subtract implemented", "def subtract" in git("show", f"{bhead}:calc.py", cwd=bare))
    r.check("build: issue moved to started", tracker.issues["i-1"].state_type == "started")
    # refine: notify before edit, label removed, original saved, then reverted
    notify_t = [t for title, body, t in toaster.sent if "ABC-2" in title or "ABC-2" in body]
    edit_t = [c[1][1] for c in tracker.calls if c[0] == "update_issue_text" and c[1][0] == "ABC-2"]
    r.check("refine: notification delivered before the edit", bool(notify_t) and bool(edit_t) and min(notify_t) < min(edit_t),
            f"toasts={[s[0] for s in toaster.sent]}")  # fmt: skip
    r.check("refine: original saved to refinements/ABC-2.md", (deps.paths.refinements / "ABC-2.md").exists())
    r.check("refine: factory:refine label removed", not tracker.issues["i-2"].has_label("factory:refine"))
    r.check("revert-refine restored the original title", tracker.issues["i-2"].title == "make calc better"
            and revert.status == "done", f"{revert.status}: {tracker.issues['i-2'].title}")  # fmt: skip
    # plan: children without factory:build, parent started
    children = [i for i in tracker.issues.values() if i.parent and i.parent.key == "ABC-3"]
    r.check("plan created sub-issues", len(children) >= 1, str(len(children)))
    r.check("plan: no sub-issue carries factory:build", all(not c.has_label("factory:build") for c in children))
    r.check("plan: parent moved to started", tracker.issues["i-3"].state_type == "started")
    # tick 2 + safety
    r.check("tick 2 dispatches nothing", runs_after_tick2 == len(runs), f"{runs_after_tick2} runs vs {len(runs)}")
    r.check("tick 2 exits 0", report2.exit_code == 0, f"exit {report2.exit_code}")
    gh_writes = {name for name, _ in gh.calls if name in ALLOWED_GH_WRITES}
    tracker_writes = {name for name, _ in tracker.calls}
    r.check("tracker writes are all on the allowed list", tracker_writes <= ALLOWED_TRACKER_WRITES, str(tracker_writes))
    r.check("no gh write other than draft PR / evidence edit", gh_writes <= ALLOWED_GH_WRITES, str(gh_writes))
    flags = [f for f in deps.state.flags() if "crash" in (f.get("reason") or "")]
    r.check("no handler crashed", not flags, json.dumps(flags, default=str)[:400])
    kept = deps.state.kept_worktrees()
    r.check("no kept worktrees", not kept, json.dumps(kept, default=str)[:400])
    cost = sum(float(run.get("cost_usd") or 0) for run in runs)

    print(f"\nworker cost (tick 1): ${cost:.2f}; open flags: {len(deps.state.flags())}")
    for f in deps.state.flags():
        print(f"  flag {f.get('target')} {f.get('reason')}: {(f.get('detail') or '')[:160]}")
    passed = sum(1 for row in r.rows if row[0])
    print(f"\nE2E {'PASS' if r.ok else 'FAIL'}: {passed}/{len(r.rows)} checks; sandbox {ROOT}")
    if not args.keep and r.ok:
        shutil.rmtree(ROOT, ignore_errors=True)
    return 0 if r.ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="use this profile's provider settings and secrets for the workers")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--cli-path", help="claude executable for the Agent SDK (else the profile's / PATH)")
    parser.add_argument("--timeout-min", type=int, default=20, help="per-worker session timeout")
    parser.add_argument("--keep", action="store_true", help="keep the sandbox directory even when everything passes")
    sys.exit(asyncio.run(main(parser.parse_args())))
