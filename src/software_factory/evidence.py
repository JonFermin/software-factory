"""PR evidence: the machine-maintained block between ``<!-- factory:evidence -->`` markers.

Every factory-opened PR body carries one evidence section built from the worker's structured
output: a risk summary (files touched, sensitive-adjacent paths, blast radius), the acceptance
criteria with the test/command that proves each one, the verification commands that ran, optional
before/after screenshots pushed to the orphan ``factory/evidence`` branch, and a low-readiness
warning when the repo scores under ``min_readiness``. ``upsert_evidence`` replaces only the marked
region, so everything a human wrote around it survives byte for byte and re-rendering is idempotent.

The renderer is deliberately emoji-free (statuses are words) so the section reads the same in every
client and in plain-text diffs.
"""

from __future__ import annotations

import posixpath
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import pathspec
from pydantic import BaseModel, Field

from .workers.schemas import AcProof, Screenshot, VerificationRun

if TYPE_CHECKING:
    from .readiness import Readiness
else:
    # ``readiness.py`` is only needed for type checking. At runtime the field is duck-typed (an
    # object with ``repo``, ``score``, ``checks``, ``missing`` and ``flaky``), which keeps this module
    # importable on its own and lets tests pass simple stand-ins.
    Readiness = Any

START = "<!-- factory:evidence -->"
END = "<!-- /factory:evidence -->"
HEADING = "## Factory evidence"
EVIDENCE_BRANCH = "factory/evidence"
MAX_INLINE_FILES = 15  # longer file lists collapse into <details>
SENSITIVE_KEYWORDS: tuple[str, ...] = ("auth", "security", "billing", "payment", "migration", "secret", "token")
STATUS_TEXT: dict[str, str] = {"proven": "proven", "unproven": "UNPROVEN", "not_applicable": "n/a"}

# "before: x" | "[after] x" | "after - x" | "before"; never "Afternoon view"
_PHASE_PREFIX = re.compile(r"^\s*\[?(before|after)\]?(?:\s*[:\-]\s*|\s+|\s*$)(.*)$", re.IGNORECASE)
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class EvidenceData(BaseModel):
    """Everything ``render_evidence`` needs; assembled by the kind handler after the worker ran."""

    changed_files: list[str] = Field(default_factory=list)
    sensitive_adjacent: list[str] = Field(default_factory=list)  # see sensitive_adjacent_paths()
    additions: int = 0
    deletions: int = 0
    ac_proofs: list[AcProof] = Field(default_factory=list)
    verification: list[VerificationRun] = Field(default_factory=list)
    screenshots: list[tuple[str, str]] = Field(default_factory=list)  # (caption, url); see screenshot_label()
    risk_notes: list[str] = Field(default_factory=list)  # worker's free-form risk notes
    readiness: Readiness | None = None
    min_readiness: int = 60
    kind: str = ""
    run_id: str = ""
    head_sha: str = ""


# --------------------------------------------------------------------------------------- render
def render_evidence(data: EvidenceData) -> str:
    """The full section, ``START`` through ``END`` (no trailing newline)."""
    lines: list[str] = [START, HEADING, ""]
    lines += _risk_summary(data)
    lines += _ac_table(data.ac_proofs)
    lines += _verification_table(data.verification)
    lines += _screenshots(data.screenshots)
    lines += _readiness_warning(data)
    lines += [_footer(data), END]
    return "\n".join(lines)


def _risk_summary(data: EvidenceData) -> list[str]:
    files = [_posix(f) for f in data.changed_files]
    dirs = sorted({f.split("/", 1)[0] if "/" in f else "(root)" for f in files})
    lines = [
        "### Risk summary",
        "",
        f"- Files touched: {len(files)}",
        f"- Blast radius: +{data.additions} / -{data.deletions} lines",
        "- Top-level dirs: " + (", ".join(f"`{d}`" for d in dirs) if dirs else "none"),
    ]
    adjacent = sorted(dict.fromkeys(_posix(p) for p in data.sensitive_adjacent))
    if adjacent:
        lines.append(f"- Sensitive-adjacent paths ({len(adjacent)}):")
        lines += [f"  - `{p}`" for p in adjacent]
    else:
        lines.append("- Sensitive-adjacent paths: none")
    notes = [_inline(n) for n in data.risk_notes if n.strip()]
    if notes:
        lines.append("- Risk notes:")
        lines += [f"  - {n}" for n in notes]
    lines.append("")
    if files:
        items = [f"- `{f}`" for f in files]
        if len(files) > MAX_INLINE_FILES:
            lines += ["<details>", f"<summary>Files ({len(files)})</summary>", "", *items, "", "</details>", ""]
        else:
            lines += ["**Files**", "", *items, ""]
    return lines


def _ac_table(proofs: list[AcProof]) -> list[str]:
    lines = ["### Acceptance criteria", ""]
    if not proofs:
        return [*lines, "_No acceptance criteria recorded._", ""]
    lines += ["| Criterion | Proof | Status |", "|---|---|---|"]
    for proof in proofs:
        evidence = f"`{_cell(proof.proof)}`" if proof.proof.strip() else "none"
        status = STATUS_TEXT.get(proof.status, proof.status)
        lines.append(f"| {_cell(proof.criterion)} | {evidence} | {status} |")
    return [*lines, ""]


def _verification_table(runs: list[VerificationRun]) -> list[str]:
    lines = ["### Verification", ""]
    if not runs:
        return [*lines, "_No verification commands were run._", ""]
    lines += ["| Command | Result | Duration |", "|---|---|---|"]
    for run in runs:
        result = ("passed" if run.passed else "FAILED") + f" (exit {run.exit_code})"
        if run.notes.strip():
            result += f" - {_cell(run.notes)}"
        duration = f"{run.duration_s:.1f}s" if run.duration_s is not None else "n/a"
        lines.append(f"| `{_cell(run.command)}` | {result} | {duration} |")
    return [*lines, ""]


def _screenshots(shots: list[tuple[str, str]]) -> list[str]:
    if not shots:
        return []
    groups: dict[str, list[tuple[str, str]]] = {"Before": [], "After": [], "": []}
    for caption, url in shots:
        match = _PHASE_PREFIX.match(caption or "")
        if match:
            phase = match.group(1).capitalize()
            label = match.group(2).strip() or phase.lower()
            groups[phase].append((label, url))
        else:
            groups[""].append((caption.strip() or "screenshot", url))
    lines = ["### Screenshots", ""]
    for phase, entries in groups.items():
        if not entries:
            continue
        if phase:
            lines += [f"**{phase}**", ""]
        for label, url in entries:
            lines += [f"![{_alt(label)}]({url})", ""]
    return lines


def _readiness_warning(data: EvidenceData) -> list[str]:
    readiness = data.readiness
    if readiness is None:
        return []
    score = int(getattr(readiness, "score", 0))
    if score >= data.min_readiness:
        return []
    repo = str(getattr(readiness, "repo", "") or "<repo>")
    failing: list[str] = []
    seen: set[str] = set()
    for check in getattr(readiness, "checks", None) or []:
        if getattr(check, "ok", True):
            continue
        name = str(getattr(check, "name", ""))
        detail = _inline(str(getattr(check, "detail", "") or ""))
        failing.append(f"`{name}`" + (f" ({detail})" if detail else ""))
        seen.add(name)
    for name in getattr(readiness, "missing", None) or []:
        if str(name) not in seen:
            failing.append(f"`{name}`")
            seen.add(str(name))
    flaky = [f"`{_inline(str(cmd))}`" for cmd in (getattr(readiness, "flaky", None) or [])]
    lines = [
        f"> **Low repo readiness: {score}/100 (minimum {data.min_readiness}).** "
        "This repo's verification signals are weak; review the change with extra care.",
        "> Failing or missing checks: " + (", ".join(failing) if failing else "none recorded"),
    ]
    if flaky:
        lines.append("> Flaky commands: " + ", ".join(flaky))
    lines.append(f"> Improve it with `factory ready {repo}`.")
    return [*lines, ""]


def _footer(data: EvidenceData) -> str:
    parts = [f"kind {data.kind}" if data.kind else "", f"run {data.run_id}" if data.run_id else ""]
    parts.append(f"head {data.head_sha}" if data.head_sha else "")
    meta = ", ".join(p for p in parts if p)
    return f"<sub>factory evidence{': ' + meta if meta else ''}</sub>"


# ------------------------------------------------------------------------------- body editing
def find_evidence(body: str) -> tuple[int, int] | None:
    """``(start, end)`` of the marked region (markers included), or ``None`` when absent/malformed."""
    start = body.find(START)
    if start < 0:
        return None
    end = body.find(END, start + len(START))
    if end < 0:
        return None
    return start, end + len(END)


def upsert_evidence(body: str, section: str) -> str:
    """Replace only the marked region of ``body`` with ``section``; append (after a blank line) when
    there is none. Text outside the markers is returned byte-identical, so applying the same section
    twice equals applying it once."""
    section = section.strip()
    span = find_evidence(body)
    if span is not None:
        start, end = span
        return body[:start] + section + body[end:]
    if not body.strip():
        return section
    if body.endswith("\n\n"):
        separator = ""
    elif body.endswith("\n"):
        separator = "\n"
    else:
        separator = "\n\n"
    return body + separator + section


def compose_pr_body(summary: str, issue_key: str | None, evidence_section: str, footer: str) -> str:
    """Summary, ``Fixes <KEY>`` (when a key is given), the evidence section, then the disclosure
    footer -- each separated by a blank line. The footer comes last so later ``upsert_evidence``
    calls leave it untouched."""
    parts = [summary.strip()]
    if issue_key and issue_key.strip():
        parts.append(f"Fixes {issue_key.strip()}")
    parts += [evidence_section.strip(), footer.strip()]
    return "\n\n".join(p for p in parts if p) + "\n"


# ------------------------------------------------------------------------------------ helpers
def sensitive_adjacent_paths(
    changed_files: list[str], sensitive_globs: list[str], *, repo_files: list[str] | None = None
) -> list[str]:
    """Changed files that share a directory with a sensitive-glob match (among the changed files, plus
    ``repo_files`` when given) or whose path contains one of ``SENSITIVE_KEYWORDS``. Direct matches
    are included: they are the most sensitive of all. Sorted, de-duplicated, POSIX separators."""
    files = [_posix(f) for f in changed_files if f.strip()]
    patterns = [g for g in sensitive_globs if g.strip()]
    matched_dirs: set[str] = set()
    if patterns:
        spec = pathspec.GitIgnoreSpec.from_lines(patterns)  # gitignore semantics (pathspec >= 0.10)
        candidates = files + [_posix(f) for f in (repo_files or []) if f.strip()]
        matched_dirs = {posixpath.dirname(f) for f in candidates if spec.match_file(f)}
    adjacent = [
        f
        for f in files
        if posixpath.dirname(f) in matched_dirs or any(key in f.lower() for key in SENSITIVE_KEYWORDS)
    ]
    return sorted(dict.fromkeys(adjacent))


def screenshot_label(shot: Screenshot) -> str:
    """Caption for ``EvidenceData.screenshots``: ``"<phase>: <caption or file name>"`` so the
    renderer can group before/after captures."""
    text = shot.caption.strip() or posixpath.basename(_posix(shot.path)) or shot.kind
    return f"{shot.phase}: {text}"


def evidence_asset_path(repo: str, pr_or_branch: str, filename: str) -> str:
    """Destination on the ``factory/evidence`` branch: ``<repo name>/<pr-N | branch>/<filename>``.
    Every segment is sanitised to ``[A-Za-z0-9._-]``; a numeric ``pr_or_branch`` becomes ``pr-N``."""
    name = _segment(repo.split("/")[-1]) or "repo"
    ref = str(pr_or_branch).strip()
    folder = f"pr-{ref}" if ref.isdigit() else (_segment(ref) or "unknown")
    base = _segment(posixpath.basename(_posix(filename))) or "asset"
    return f"{name}/{folder}/{base}"


def evidence_url(repo: str, path: str) -> str:
    """Raw-content URL of an asset on the evidence branch (works in PR bodies as an image source)."""
    return f"https://github.com/{repo}/blob/{EVIDENCE_BRANCH}/{quote(path.strip('/'), safe='/')}?raw=true"


def _posix(path: str) -> str:
    text = path.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _segment(text: str) -> str:
    cleaned = _UNSAFE_PATH_CHARS.sub("-", text.strip())
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    return cleaned.strip("-.")


def _inline(text: str) -> str:
    """Collapse whitespace/newlines so a value stays on one markdown line."""
    return " ".join(text.split())


def _cell(text: str) -> str:
    """Table-cell safe: one line, pipes escaped."""
    return _inline(text).replace("|", "\\|")


def _alt(text: str) -> str:
    return _inline(text).replace("[", "(").replace("]", ")")
