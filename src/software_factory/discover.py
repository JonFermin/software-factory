"""Verification-command discovery for a repo checkout.

Priority (first tier that yields verification commands wins; an override always wins):

1. ``RepoOverride.verify`` / ``setup`` / ``dev_command`` / ``dev_port`` from the profile.
2. ``CLAUDE.md`` then ``AGENTS.md``: sections whose heading mentions verify/verification/test/testing/
   build/check/lint/typecheck/commands; shell lines in fenced code blocks and the first inline
   ``code`` span of list items, kept when they start with a known runner.
3. ``.github/workflows/*.yml`` ``run:`` steps (test/lint/typecheck/build kept; install/setup/deploy/
   publish/upload skipped; install steps become explicit setup).
4. ``package.json`` scripts (typecheck, lint, test, build) -- placeholder ``test`` scripts skipped.
5. ``Makefile`` targets, then ``pyproject.toml`` (pytest), ``justfile`` recipes, Godot + GUT.

A repo with a ``.pre-commit-config.yaml`` also gets ``pre-commit run --all-files
--show-diff-on-failure`` (source ``.pre-commit-config.yaml``) unless a discovered command already
runs pre-commit: workers commit with git hooks disabled, so the checks the repo wired into its
pre-commit hook would otherwise never run. An explicit ``RepoOverride.verify`` stays exactly as
given. Hook *installers* (``pre-commit install``, ``husky``, ``npm run init-precommit``) are never
verification.

Setup commands come from the winning tier when it states them explicitly, else from CLAUDE.md /
AGENTS.md, else are inferred from lockfiles. Results are cached per repo keyed by HEAD sha via
``discover_cached``.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from .config import RepoOverride

if TYPE_CHECKING:
    from .repos import RepoInfo
    from .state import State


class Cmd(BaseModel):
    name: str
    command: str
    source: str  # "CLAUDE.md", ".github/workflows/ci.yml", "package.json", "override", ...


class Verification(BaseModel):
    setup: list[Cmd] = Field(default_factory=list)
    verify: list[Cmd] = Field(default_factory=list)
    source: str = "none"  # tier/file the verify commands came from; "none" when nothing was found
    dev_command: str | None = None
    dev_port: int | None = None

    @property
    def verify_commands(self) -> list[str]:
        return [c.command for c in self.verify]

    @property
    def setup_commands(self) -> list[str]:
        return [c.command for c in self.setup]


AGENT_DOCS: tuple[str, ...] = ("CLAUDE.md", "AGENTS.md")

# First token of a line must be one of these for the line to count as a verification command.
RUNNERS: frozenset[str] = frozenset(
    {
        "npm", "pnpm", "yarn", "npx", "bun", "uv", "python", "python3", "pytest", "make", "just",
        "cargo", "go", "godot", "ruff", "tsc", "vitest", "jest", "dotnet", "bundle", "rails", "rake",
        "lake", "pre-commit",
    }
)  # fmt: skip

# Lockfile -> inferred setup command (order matters; a repo may match several).
LOCKFILE_SETUP: tuple[tuple[str, str], ...] = (
    ("package-lock.json", "npm ci"),
    ("pnpm-lock.yaml", "pnpm install --frozen-lockfile"),
    ("yarn.lock", "yarn install --frozen-lockfile"),
    ("bun.lockb", "bun install"),
    ("bun.lock", "bun install"),
    ("uv.lock", "uv sync"),
)

PACKAGE_SCRIPT_ORDER: tuple[str, ...] = ("typecheck", "lint", "test", "build")
MAKE_TARGETS: tuple[str, ...] = ("typecheck", "lint", "test", "check")
GODOT_GUT_COMMAND = "godot --headless -s addons/gut/gut_cmdln.gd"
PRE_COMMIT_CONFIG = ".pre-commit-config.yaml"
PRE_COMMIT_COMMAND = "pre-commit run --all-files --show-diff-on-failure"

# Heading keywords. "commands" alone is a weak signal: such sections list everything (dev servers,
# migrations, REPLs), so only commands that look like verification are kept from them.
_STRONG_HEADING = re.compile(r"\b(verif\w*|tests?|testing|builds?|building|checks?|checking|lint\w*|type-?check\w*)\b", re.I)
_WEAK_HEADING = re.compile(r"\bcommands?\b", re.I)
_SETUP_HEADING = re.compile(r"\b(setup|set-up|install\w*|getting started|prerequisites?|bootstrap)\b", re.I)
_VERIFY_HINT = re.compile(
    r"\b(tests?|lint\w*|type-?check|check|build|tsc|pytest|vitest|jest|ruff|mypy|pyright|eslint|biome|prettier|"
    r"clippy|gut_cmdln|headless|compile|verify)\b",
    re.I,
)
_LONG_RUNNING = re.compile(
    r"^(?:(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|preview|serve|watch|storybook|console)(?::\S*)?(?:\s|$)"
    r"|npx\s+(?:vite|expo\s+start|serve|http-server)(?:\s|$)"
    r"|godot\s*$"
    r"|.*\s--watch(?:\s|$|=))",
    re.I,
)
# Documentation templates (``npm run x -- <seed>``, ``pnpm test src/path/to/file.test.ts``).
_PLACEHOLDER = re.compile(r"<[^>]*>|\bpath/to\b|\.\.\.|\{\{")
# Commands that rewrite files or install hooks are not verification (unless they run in check mode).
# ``pre-commit run`` is the one pre-commit invocation that verifies; every other one installs or
# updates hooks (``pre-commit install``, ``npm run init-precommit``).
_MUTATING = re.compile(r":fix\b|\bfix\b|\bfmt\b|\bformat\b|\bprettier\b|\bpre-?commit\b(?!\s+run\b)|\bhusky\b", re.I)
_PRE_COMMIT_RUN = re.compile(r"\bpre-?commit\s+run\b", re.I)
_CHECK_MODE = re.compile(r"\bcheck\b|--check\b|--dry-run\b", re.I)
# Browser/e2e suites need servers and browsers; they are opt-in via RepoOverride.verify.
_E2E = re.compile(r"\be2e\b|:e2e\b|\bplaywright\b|\bcypress\b", re.I)
_INSTALL = re.compile(
    r"^(?:(?:npm|pnpm|yarn|bun)\s+(?:ci|install|i)\b|uv\s+(?:sync|pip\s+install)\b|pip3?\s+install\b|"
    r"poetry\s+install\b|bundle\s+install\b|cargo\s+fetch\b|go\s+mod\s+download\b|lake\s+(?:update|exe\s+cache\s+get)\b)",
    re.I,
)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_STEP_SKIP = re.compile(
    r"\b(install|setup|set-up|deploy|publish|upload|release|checkout|cache|login|download|configure|"
    r"bump|tag|notify|comment|slack|codecov|artifact)\w*\b",
    re.I,
)
_STEP_KEEP = re.compile(
    r"\b(tests?|testing|lint\w*|type-?check|tsc|builds?|checks?|pytest|vitest|jest|ruff|mypy|eslint|biome|"
    r"prettier|format|verify|gut)\b",
    re.I,
)
_PLACEHOLDER_TEST = re.compile(r"no test specified|^echo\b.*\bexit\s+1\s*$", re.I)
_PORT_IN_SCRIPT = re.compile(r"(?:--port[ =]|-p )(\d{2,5})\b")
_PORT_IN_CONFIG = re.compile(r"server\s*:\s*\{[^}]*?\bport\s*:\s*(\d{2,5})\b", re.S)
_PORT_ANY = re.compile(r"\bport\s*:\s*(\d{2,5})\b")


# ---------------------------------------------------------------------------------- public API
def discover(repo_path: Path, override: RepoOverride | None = None) -> Verification:
    """Discover setup/verify/dev commands for ``repo_path`` (see module docstring for priority)."""
    repo_path = Path(repo_path)
    override = override or RepoOverride()

    verify: list[Cmd] = []
    explicit_setup: list[Cmd] = []
    source = "none"
    doc_setup: list[Cmd] = []
    for tier_source, tier in _tiers(repo_path):
        tier_verify, tier_setup = tier()
        if tier_source in AGENT_DOCS and tier_setup and not doc_setup:
            doc_setup = tier_setup
        if tier_verify:
            verify, explicit_setup = tier_verify, tier_setup
            source = tier_verify[0].source
            break

    setup = explicit_setup or doc_setup or _setup_from_lockfiles(repo_path)
    dev_command, dev_port = _dev_from_package_json(repo_path)
    pre_commit = _pre_commit_verify(repo_path, verify)
    if pre_commit is not None:
        verify = [*verify, pre_commit]
        if source == "none":
            source = pre_commit.source

    if override.verify is not None:
        verify = [Cmd(name=_cmd_name(c), command=c, source="override") for c in override.verify]
        source = "override"
    if override.setup is not None:
        setup = [Cmd(name="setup", command=c, source="override") for c in override.setup]
    if override.dev_command:
        dev_command = override.dev_command
    if override.dev_port:
        dev_port = override.dev_port
    return Verification(setup=setup, verify=verify, source=source, dev_command=dev_command, dev_port=dev_port)


def discover_cached(state: State, repo: RepoInfo) -> Verification:
    """``discover`` memoised in ``state`` (duck-typed ``cache_get``/``cache_set``) keyed by HEAD sha.

    The cached value also carries the override it was computed with, so a config change invalidates it.
    """
    sha = git_head_sha(repo.path)
    key = f"discover:{repo.slug}"
    override_dump = repo.override.model_dump(mode="json")
    cached = state.cache_get("discover", key) if sha else None
    if isinstance(cached, dict) and cached.get("head_sha") == sha and cached.get("override") == override_dump:
        try:
            return Verification.model_validate(cached["verification"])
        except (KeyError, ValidationError):
            pass
    verification = discover(repo.path, repo.override)
    if sha:
        state.cache_set(
            "discover",
            key,
            {"head_sha": sha, "override": override_dump, "verification": verification.model_dump(mode="json")},
        )
    return verification


def git_head_sha(repo_path: Path) -> str:
    """``git rev-parse HEAD`` in ``repo_path``; empty string when unavailable (not a repo, no commits)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def agent_doc(repo_path: Path) -> Path | None:
    """The repo's ``CLAUDE.md`` or ``AGENTS.md`` (first that exists)."""
    for name in AGENT_DOCS:
        path = Path(repo_path) / name
        if path.is_file():
            return path
    return None


def has_verification_section(repo_path: Path) -> bool:
    """True when CLAUDE.md/AGENTS.md has a verification-style heading, or a commands section that
    yields at least one verification command."""
    for name in AGENT_DOCS:
        path = Path(repo_path) / name
        if not path.is_file():
            continue
        text = _read(path)
        for heading, _body in _sections(text):
            if _STRONG_HEADING.search(heading):
                return True
        if _commands_from_markdown(text, name)[0]:
            return True
    return False


# -------------------------------------------------------------------------------------- tiers
def _tiers(repo_path: Path) -> list[tuple[str, Any]]:
    return [
        ("CLAUDE.md", lambda: _from_agent_doc(repo_path, "CLAUDE.md")),
        ("AGENTS.md", lambda: _from_agent_doc(repo_path, "AGENTS.md")),
        (".github/workflows", lambda: _from_workflows(repo_path)),
        ("package.json", lambda: _from_package_json(repo_path)),
        ("Makefile", lambda: _from_makefile(repo_path)),
        ("pyproject.toml", lambda: _from_pyproject(repo_path)),
        ("justfile", lambda: _from_justfile(repo_path)),
        ("project.godot", lambda: _from_godot(repo_path)),
    ]


def _from_agent_doc(repo_path: Path, name: str) -> tuple[list[Cmd], list[Cmd]]:
    path = repo_path / name
    if not path.is_file():
        return [], []
    return _commands_from_markdown(_read(path), name)


def _commands_from_markdown(text: str, source: str) -> tuple[list[Cmd], list[Cmd]]:
    verify: list[Cmd] = []
    setup: list[Cmd] = []
    seen: set[str] = set()
    for heading, body in _sections(text):
        strong = bool(_STRONG_HEADING.search(heading))
        weak = bool(_WEAK_HEADING.search(heading))
        setup_section = bool(_SETUP_HEADING.search(heading))
        if not (strong or weak or setup_section):
            continue
        for command in _shell_lines_in_markdown(body):
            if command in seen:
                continue
            if _INSTALL.match(command):
                seen.add(command)
                setup.append(Cmd(name="setup", command=command, source=source))
                continue
            if setup_section and not (strong or weak):
                continue
            if _first_token(command) not in RUNNERS or not _is_verification_candidate(command):
                continue
            if not strong and not _VERIFY_HINT.search(command):
                continue
            seen.add(command)
            verify.append(Cmd(name=_cmd_name(command), command=command, source=source))
    return verify, setup


def _from_workflows(repo_path: Path) -> tuple[list[Cmd], list[Cmd]]:
    wf_dir = repo_path / ".github" / "workflows"
    if not wf_dir.is_dir():
        return [], []
    verify: list[Cmd] = []
    setup: list[Cmd] = []
    seen: set[str] = set()
    for path in sorted(p for p in wf_dir.iterdir() if p.suffix in (".yml", ".yaml") and p.is_file()):
        try:
            data = yaml.safe_load(_read(path))
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
            continue
        source = f".github/workflows/{path.name}"
        wf_wd = _run_default(data, "working-directory")
        for job in data["jobs"].values():
            if not isinstance(job, dict):
                continue
            job_wd = _run_default(job, "working-directory") or wf_wd
            for step in job.get("steps") or []:
                if not isinstance(step, dict) or not isinstance(step.get("run"), str):
                    continue
                for kind, command in _classify_step(step, job_wd):
                    if command in seen:
                        continue
                    seen.add(command)
                    target = setup if kind == "setup" else verify
                    name = "setup" if kind == "setup" else str(step.get("name") or _cmd_name(command))
                    target.append(Cmd(name=name, command=command, source=source))
    return verify, setup


def _classify_step(step: dict, job_wd: str | None) -> list[tuple[str, str]]:
    """Return ``[(kind, command)]`` for one workflow step; ``kind`` is "setup" or "verify"."""
    run = step["run"]
    if "${{" in run or str(step.get("shell") or "").lower() == "python":
        return []
    lines = _clean_shell_lines(run)
    if not lines:
        return []
    wd = step.get("working-directory") or job_wd
    prefix = f"cd {wd} && " if wd else ""
    installs = [ln for ln in lines if _INSTALL.match(ln)]
    if installs:
        return [("setup", prefix + ln) for ln in installs]
    name = str(step.get("name") or step.get("id") or "")
    if _STEP_SKIP.search(name) or not _STEP_KEEP.search(name + "\n" + run):
        return []
    if len(lines) > 1 and not all(_first_token(ln) in RUNNERS for ln in lines):
        return []  # shell logic (if/for/pipes across lines): not reproducible outside CI
    return [("verify", prefix + ln) for ln in lines if _is_verification_candidate(ln)]


def _from_package_json(repo_path: Path) -> tuple[list[Cmd], list[Cmd]]:
    scripts, pm = _package_scripts(repo_path)
    verify: list[Cmd] = []
    for name in PACKAGE_SCRIPT_ORDER:
        script = scripts.get(name)
        if not isinstance(script, str) or not script.strip():
            continue
        if name == "test" and _PLACEHOLDER_TEST.search(script.strip()):
            continue
        verify.append(Cmd(name=name, command=f"{pm} run {name}", source="package.json"))
    return verify, []


def _from_makefile(repo_path: Path) -> tuple[list[Cmd], list[Cmd]]:
    path = _first_existing(repo_path, ("Makefile", "makefile", "GNUmakefile"))
    if path is None:
        return [], []
    text = _read(path)
    verify = [
        Cmd(name=target, command=f"make {target}", source=path.name)
        for target in MAKE_TARGETS
        if re.search(rf"^{target}\s*:(?!=)", text, re.M)
    ]
    return verify, []


def _from_pyproject(repo_path: Path) -> tuple[list[Cmd], list[Cmd]]:
    path = repo_path / "pyproject.toml"
    if not path.is_file():
        return [], []
    try:
        data = tomllib.loads(_read(path))
    except tomllib.TOMLDecodeError:
        data = {}
    pytest_configured = "pytest" in (data.get("tool") or {})
    if not pytest_configured and not (repo_path / "tests").is_dir():
        return [], []
    command = "uv run pytest" if (repo_path / "uv.lock").is_file() else "python -m pytest"
    return [Cmd(name="test", command=command, source="pyproject.toml")], []


def _from_justfile(repo_path: Path) -> tuple[list[Cmd], list[Cmd]]:
    path = _first_existing(repo_path, ("justfile", "Justfile", ".justfile"))
    if path is None:
        return [], []
    text = _read(path)
    verify = [
        Cmd(name=recipe, command=f"just {recipe}", source=path.name)
        for recipe in MAKE_TARGETS
        if re.search(rf"^@?{recipe}\b[^:\n]*:(?!=)", text, re.M)
    ]
    return verify, []


def _from_godot(repo_path: Path) -> tuple[list[Cmd], list[Cmd]]:
    if (repo_path / "project.godot").is_file() and (repo_path / "addons" / "gut").is_dir():
        return [Cmd(name="test", command=GODOT_GUT_COMMAND, source="project.godot")], []
    return [], []


def _pre_commit_verify(repo_path: Path, verify: list[Cmd]) -> Cmd | None:
    """``pre-commit run --all-files --show-diff-on-failure`` when the repo has a pre-commit config
    and none of the discovered commands already runs pre-commit."""
    if not (repo_path / PRE_COMMIT_CONFIG).is_file():
        return None
    if any(_PRE_COMMIT_RUN.search(c.command) for c in verify):
        return None
    return Cmd(name="pre-commit", command=PRE_COMMIT_COMMAND, source=PRE_COMMIT_CONFIG)


# --------------------------------------------------------------------------------- setup / dev
def _setup_from_lockfiles(repo_path: Path) -> list[Cmd]:
    setup: list[Cmd] = []
    seen: set[str] = set()
    for lockfile, command in LOCKFILE_SETUP:
        if (repo_path / lockfile).is_file() and command not in seen:
            seen.add(command)
            setup.append(Cmd(name="setup", command=command, source=lockfile))
    return setup


def _dev_from_package_json(repo_path: Path) -> tuple[str | None, int | None]:
    scripts, pm = _package_scripts(repo_path)
    script = scripts.get("dev")
    if not isinstance(script, str) or not script.strip():
        return None, None
    port: int | None = None
    if m := _PORT_IN_SCRIPT.search(script):
        port = int(m.group(1))
    if port is None:
        for config in sorted(repo_path.glob("vite.config.*")):
            text = _read(config)
            m = _PORT_IN_CONFIG.search(text) or _PORT_ANY.search(text)
            if m:
                port = int(m.group(1))
                break
    return f"{pm} run dev", port


def _package_scripts(repo_path: Path) -> tuple[dict[str, Any], str]:
    path = repo_path / "package.json"
    if not path.is_file():
        return {}, "npm"
    try:
        data = json.loads(_read(path))
    except json.JSONDecodeError:
        return {}, "npm"
    if not isinstance(data, dict):
        return {}, "npm"
    scripts = data.get("scripts") if isinstance(data.get("scripts"), dict) else {}
    return scripts, _package_manager(repo_path, data)


def _package_manager(repo_path: Path, data: dict) -> str:
    declared = str(data.get("packageManager") or "").split("@")[0].strip().lower()
    if declared in ("npm", "pnpm", "yarn", "bun"):
        return declared
    if (repo_path / "pnpm-lock.yaml").is_file():
        return "pnpm"
    if (repo_path / "yarn.lock").is_file():
        return "yarn"
    if (repo_path / "bun.lockb").is_file() or (repo_path / "bun.lock").is_file():
        return "bun"
    return "npm"


# ----------------------------------------------------------------------------------- helpers
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _first_existing(repo_path: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = repo_path / name
        if path.is_file():
            return path
    return None


def _run_default(node: dict, key: str) -> str | None:
    defaults = node.get("defaults")
    run = defaults.get("run") if isinstance(defaults, dict) else None
    value = run.get(key) if isinstance(run, dict) else None
    return str(value) if value else None


def _sections(markdown: str) -> list[tuple[str, str]]:
    """``[(heading text, body)]``; a body runs until the next heading of the same or higher level
    (sub-headings are included). Headings inside fenced code blocks are ignored."""
    headings: list[tuple[int, str, int]] = []  # (level, text, line index)
    lines = markdown.splitlines()
    in_fence = False
    for i, line in enumerate(lines):
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if m := _HEADING.match(line):
            headings.append((len(m.group(1)), m.group(2), i))
    sections: list[tuple[str, str]] = []
    for idx, (level, text, start) in enumerate(headings):
        end = len(lines)
        for next_level, _t, next_start in headings[idx + 1 :]:
            if next_level <= level:
                end = next_start
                break
        sections.append((text, "\n".join(lines[start + 1 : end])))
    return sections


def _shell_lines_in_markdown(body: str) -> list[str]:
    """Shell lines from fenced code blocks plus the first inline code span of each list item."""
    out: list[str] = []
    in_fence = False
    for raw in body.splitlines():
        stripped = raw.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            if command := _clean_shell_line(stripped):
                out.append(command)
            continue
        if m := _LIST_ITEM.match(raw):
            spans = _CODE_SPAN.findall(m.group(1))
            if spans and (command := _clean_shell_line(spans[0])):
                out.append(command)
    return out


def _clean_shell_lines(run: str) -> list[str]:
    """Workflow ``run:`` text -> cleaned lines, joining backslash continuations."""
    joined: list[str] = []
    for raw in run.splitlines():
        line = raw.strip()
        if joined and joined[-1].endswith("\\"):
            joined[-1] = joined[-1][:-1].rstrip() + " " + line
        else:
            joined.append(line)
    out: list[str] = []
    for line in joined:
        line = line[:-1].rstrip() if line.endswith("\\") else line
        if re.match(r"^set\s+[-+][a-zA-Z]+(\s+pipefail)?$", line):
            continue
        if command := _clean_shell_line(line):
            out.append(command)
    return out


def _clean_shell_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    if line.startswith(("$ ", "> ")):
        line = line[2:].strip()
    if " #" in line:
        line = line.split(" #", 1)[0].rstrip()
    return line


def _first_token(command: str) -> str:
    return command.split(maxsplit=1)[0] if command.strip() else ""


def _is_verification_candidate(command: str) -> bool:
    """False for dev servers/watchers, doc placeholders, file-rewriting commands and e2e suites."""
    if _LONG_RUNNING.match(command) or _PLACEHOLDER.search(command) or _E2E.search(command):
        return False
    return not (_MUTATING.search(command) and not _CHECK_MODE.search(command))


_NAME_HINTS: tuple[tuple[str, str], ...] = (
    (r"\btype-?check\b|\btsc\b|\bmypy\b|\bpyright\b", "typecheck"),
    (r"\blint\b|\bruff\s+check\b|\beslint\b|\bclippy\b|\bbiome\b", "lint"),
    (r"\bformat\b|\bprettier\b|\bruff\s+format\b", "format"),
    (r"\btests?\b|\bpytest\b|\bvitest\b|\bjest\b|\bgut_cmdln\b", "test"),
    (r"\bbuild\b", "build"),
    (r"\bcheck\b", "check"),
    (r"\bverify\b", "verify"),
)


def _cmd_name(command: str) -> str:
    """Short label for a command: typecheck/lint/test/build/check, else its first token."""
    for pattern, name in _NAME_HINTS:
        if re.search(pattern, command, re.I):
            return name
    return _first_token(command) or "verify"
