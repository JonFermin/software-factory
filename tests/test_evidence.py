"""evidence.py: section rendering, idempotent body editing, PR body composition, asset helpers."""

from __future__ import annotations

from types import SimpleNamespace

from software_factory.evidence import (
    END,
    START,
    EvidenceData,
    compose_pr_body,
    evidence_asset_path,
    evidence_url,
    find_evidence,
    render_evidence,
    screenshot_label,
    sensitive_adjacent_paths,
    upsert_evidence,
)
from software_factory.workers.schemas import AcProof, Screenshot, VerificationRun

FOOTER = "---\n_This pull request was prepared by an AI coding agent._"


# ------------------------------------------------------------------------------- stand-ins
def _check(name: str, ok: bool, detail: str = "") -> SimpleNamespace:
    return SimpleNamespace(name=name, ok=ok, detail=detail, weight=10)


def _readiness(score: int, *, checks=(), flaky=(), missing=None, repo="JaneDoe/game") -> SimpleNamespace:
    """A duck-typed ``Readiness`` (readiness.py is not imported by evidence.py at runtime)."""
    checks = list(checks)
    return SimpleNamespace(
        repo=repo,
        score=score,
        checks=checks,
        missing=list(missing) if missing is not None else [c.name for c in checks if not c.ok],
        flaky=list(flaky),
        runtime_s=1.0,
        computed_at="",
        head_sha="",
    )


def _data(**overrides) -> EvidenceData:
    base = dict(
        changed_files=["src/app.py", "tests/test_app.py", "README.md"],
        additions=12,
        deletions=4,
        ac_proofs=[
            AcProof(criterion="Users can log in", proof="tests/test_app.py::test_login", status="proven"),
            AcProof(criterion="Rate limit applies", status="unproven"),
            AcProof(criterion="Docs updated", proof="n/a for code", status="not_applicable"),
        ],
        verification=[
            VerificationRun(command="uv run pytest", exit_code=0, passed=True, duration_s=12.34),
            VerificationRun(command="uv run ruff check .", exit_code=1, passed=False, notes="2 errors"),
        ],
        kind="build",
        run_id="run-0001",
        head_sha="0123456789abcdef0123456789abcdef01234567",
    )
    base.update(overrides)
    return EvidenceData(**base)


# -------------------------------------------------------------------------------- rendering
def test_render_has_markers_heading_and_every_section_in_order():
    text = render_evidence(_data())
    assert text.startswith(START + "\n## Factory evidence\n")
    assert text.endswith("\n" + END)
    order = ["### Risk summary", "### Acceptance criteria", "### Verification", "<sub>factory evidence"]
    positions = [text.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "### Screenshots" not in text  # none given


def test_risk_summary_counts_blast_radius_and_dirs():
    text = render_evidence(_data())
    assert "- Files touched: 3" in text
    assert "- Blast radius: +12 / -4 lines" in text
    assert "- Top-level dirs: `(root)`, `src`, `tests`" in text
    assert "- Sensitive-adjacent paths: none" in text
    assert "**Files**\n\n- `src/app.py`\n- `tests/test_app.py`\n- `README.md`\n" in text
    assert "<details>" not in text


def test_sensitive_adjacent_and_risk_notes_are_listed():
    text = render_evidence(
        _data(sensitive_adjacent=["src/auth/login.py", "src\\auth\\session.py", "src/auth/login.py"], risk_notes=["touches  session\nhandling"])
    )
    assert "- Sensitive-adjacent paths (2):\n  - `src/auth/login.py`\n  - `src/auth/session.py`" in text
    assert "- Risk notes:\n  - touches session handling" in text


def test_long_file_list_collapses_into_details():
    files = [f"src/module_{i:02d}.py" for i in range(16)]
    text = render_evidence(_data(changed_files=files))
    assert "- Files touched: 16" in text
    assert "<details>\n<summary>Files (16)</summary>\n\n- `src/module_00.py`" in text
    assert "</details>" in text
    assert "**Files**" not in text
    fifteen = render_evidence(_data(changed_files=files[:15]))
    assert "<details>" not in fifteen and "**Files**" in fifteen


def test_ac_table_uses_word_statuses_without_emoji():
    text = render_evidence(_data())
    assert "| Criterion | Proof | Status |\n|---|---|---|" in text
    assert "| Users can log in | `tests/test_app.py::test_login` | proven |" in text
    assert "| Rate limit applies | none | UNPROVEN |" in text
    assert "| Docs updated | `n/a for code` | n/a |" in text
    assert text.isascii()


def test_empty_ac_and_verification_render_placeholders():
    text = render_evidence(_data(ac_proofs=[], verification=[]))
    assert "_No acceptance criteria recorded._" in text
    assert "_No verification commands were run._" in text


def test_verification_table_rows():
    text = render_evidence(_data())
    assert "| Command | Result | Duration |" in text
    assert "| `uv run pytest` | passed (exit 0) | 12.3s |" in text
    assert "| `uv run ruff check .` | FAILED (exit 1) - 2 errors | n/a |" in text


def test_table_cells_escape_pipes_and_newlines():
    proofs = [AcProof(criterion="a | b\nc", proof="x|y", status="proven")]
    runs = [VerificationRun(command="echo a | grep b", exit_code=0, passed=True, notes="line1\nline2")]
    text = render_evidence(_data(ac_proofs=proofs, verification=runs))
    assert "| a \\| b c | `x\\|y` | proven |" in text
    assert "| `echo a \\| grep b` | passed (exit 0) - line1 line2 | n/a |" in text


def test_screenshots_grouped_by_before_after_with_images():
    shots = [
        ("after: Level select", "https://example.test/after.png"),
        ("before: Level select", "https://example.test/before.png"),
        ("[After] Pause menu", "https://example.test/pause.png"),
        ("Overview gif", "https://example.test/overview.gif"),
        ("Afternoon view", "https://example.test/noon.png"),  # not an "after" capture
    ]
    text = render_evidence(_data(screenshots=shots))
    section = text[text.index("### Screenshots") : text.index("<sub>")]
    assert section.index("**Before**") < section.index("**After**")
    assert "**Before**\n\n![Level select](https://example.test/before.png)" in section
    assert "**After**\n\n![Level select](https://example.test/after.png)\n\n![Pause menu](https://example.test/pause.png)" in section
    assert section.rstrip().endswith(
        "![Overview gif](https://example.test/overview.gif)\n\n![Afternoon view](https://example.test/noon.png)"
    )
    assert "![noon view]" not in section


def test_screenshot_label_from_worker_schema():
    assert screenshot_label(Screenshot(path=".factory/evidence/a.png", caption="Home", phase="before")) == "before: Home"
    assert screenshot_label(Screenshot(path=".factory/evidence/after-01.png")) == "after: after-01.png"


def test_footer_line_carries_kind_run_and_sha():
    text = render_evidence(_data())
    assert "<sub>factory evidence: kind build, run run-0001, head 0123456789abcdef0123456789abcdef01234567</sub>" in text
    bare = render_evidence(_data(kind="", run_id="", head_sha=""))
    assert "<sub>factory evidence</sub>" in bare


# --------------------------------------------------------------------------- readiness block
def test_low_readiness_warning_lists_failing_and_flaky_checks():
    readiness = _readiness(
        40,
        checks=[
            _check("commands_pass", False, "`npm test` exit 1: 3 failed"),
            _check("agent_doc_present", True, "CLAUDE.md"),
            _check("not_flaky", False),
        ],
        missing=["commands_pass", "not_flaky", "dev_server"],
        flaky=["npm test"],
    )
    text = render_evidence(_data(readiness=readiness, min_readiness=60))
    assert "> **Low repo readiness: 40/100 (minimum 60).**" in text
    assert "> Failing or missing checks: `commands_pass` (`npm test` exit 1: 3 failed), `not_flaky`, `dev_server`" in text
    assert "> Flaky commands: `npm test`" in text
    assert "> Improve it with `factory ready JaneDoe/game`." in text
    assert text.index("### Verification") < text.index("> **Low repo readiness") < text.index("<sub>")


def test_readiness_warning_absent_at_or_above_threshold_or_when_unknown():
    assert "Low repo readiness" not in render_evidence(_data(readiness=_readiness(60), min_readiness=60))
    assert "Low repo readiness" not in render_evidence(_data(readiness=_readiness(95), min_readiness=60))
    assert "Low repo readiness" not in render_evidence(_data(readiness=None, min_readiness=100))
    assert "Low repo readiness" in render_evidence(_data(readiness=_readiness(59), min_readiness=60))


# -------------------------------------------------------------------------- body editing
def test_upsert_replaces_only_the_marked_region_and_is_idempotent():
    section = render_evidence(_data())
    body = f"Summary paragraph.\r\n\r\nFixes ABC-1\n\n{START}\nold stuff\n{END}\n\n{FOOTER}\n"
    once = upsert_evidence(body, section)
    twice = upsert_evidence(once, section)
    assert once == twice
    assert once.startswith("Summary paragraph.\r\n\r\nFixes ABC-1\n\n" + START)
    assert once.endswith(END + "\n\n" + FOOTER + "\n")
    assert "old stuff" not in once
    assert once.count(START) == 1 and once.count(END) == 1
    start, end = find_evidence(once)
    assert once[start:end] == section
    # a fresh render with different data changes only the marked region
    other = upsert_evidence(once, render_evidence(_data(additions=99)))
    assert other[:start] == once[:start]
    assert other[other.index(END) + len(END) :] == once[end:]
    assert "+99 / -4 lines" in other


def test_upsert_appends_after_one_blank_line_when_absent():
    section = f"{START}\nx\n{END}"
    assert upsert_evidence("Body text", section) == "Body text\n\n" + section
    assert upsert_evidence("Body text\n", section) == "Body text\n\n" + section
    assert upsert_evidence("Body text\n\n", section) == "Body text\n\n" + section
    assert upsert_evidence("", section) == section
    assert upsert_evidence("   \n", section) == section
    appended = upsert_evidence("Body text", section)
    assert upsert_evidence(appended, section) == appended


def test_upsert_with_malformed_markers_appends_a_fresh_section():
    section = f"{START}\nnew\n{END}"
    body = f"intro\n{START}\nunterminated"
    assert find_evidence(body) is None
    assert upsert_evidence(body, section) == body + "\n\n" + section


# --------------------------------------------------------------------------- compose body
def test_compose_pr_body_includes_fixes_and_footer_in_order():
    section = render_evidence(_data())
    body = compose_pr_body("Adds login.\n", "ABC-123", section, FOOTER)
    assert body == f"Adds login.\n\nFixes ABC-123\n\n{section}\n\n{FOOTER}\n"
    assert body.index("Fixes ABC-123") < body.index(START) < body.index(END) < body.index(FOOTER)


def test_compose_pr_body_without_issue_key_or_footer():
    section = render_evidence(_data())
    body = compose_pr_body("Summary", None, section, "")
    assert "Fixes" not in body
    assert body == f"Summary\n\n{section}\n"
    assert compose_pr_body("Summary", "  ", section, FOOTER).count("Fixes") == 0


def test_composed_body_survives_later_evidence_refresh():
    body = compose_pr_body("Summary", "ABC-1", render_evidence(_data()), FOOTER)
    refreshed = upsert_evidence(body, render_evidence(_data(kind="ci", run_id="run-0002")))
    assert refreshed.startswith("Summary\n\nFixes ABC-1\n\n" + START)
    assert refreshed.endswith(END + "\n\n" + FOOTER + "\n")
    assert "kind ci, run run-0002" in refreshed and "run run-0001" not in refreshed


# ------------------------------------------------------------------------ sensitive paths
def test_sensitive_adjacent_paths_direct_sibling_and_keyword_matches():
    globs = ["**/auth/**", "**/*.pem", "**/migrations/**"]
    changed = [
        "src/auth/login.py",  # direct glob match
        "src/app.py",  # plain
        "certs/README.md",  # sibling of a repo-level .pem (via repo_files)
        "src/payments_api/handler.py",  # keyword: payment
        "db/MIGRATION_notes.md",  # keyword, case-insensitive
        "src\\tokens\\issue.py",  # keyword + backslashes
        "docs/guide.md",
    ]
    result = sensitive_adjacent_paths(changed, globs, repo_files=["certs/server.pem", "src/other.py"])
    assert result == [
        "certs/README.md",
        "db/MIGRATION_notes.md",
        "src/auth/login.py",
        "src/payments_api/handler.py",
        "src/tokens/issue.py",
    ]


def test_sensitive_adjacent_paths_sibling_of_changed_match_and_no_globs():
    result = sensitive_adjacent_paths(["lib/keys/x.pem", "lib/keys/README.md", "lib/other/y.py"], ["**/*.pem"])
    assert result == ["lib/keys/README.md", "lib/keys/x.pem"]
    assert sensitive_adjacent_paths(["src/app.py"], []) == []
    assert sensitive_adjacent_paths(["src/auth.py"], []) == ["src/auth.py"]


# --------------------------------------------------------------------------- asset helpers
def test_evidence_asset_path_sanitises_segments():
    assert evidence_asset_path("JaneDoe/game", "42", "after.png") == "game/pr-42/after.png"
    assert evidence_asset_path("JaneDoe/game", "feat/abc-1", "before shot.png") == "game/feat-abc-1/before-shot.png"
    assert evidence_asset_path("o/r", "7", "../../evil.png") == "r/pr-7/evil.png"
    assert evidence_asset_path("o/r", "7", ".factory/evidence/x.gif") == "r/pr-7/x.gif"


def test_evidence_url_points_at_raw_blob_on_evidence_branch():
    assert (
        evidence_url("JaneDoe/game", "game/pr-42/after.png")
        == "https://github.com/JaneDoe/game/blob/factory/evidence/game/pr-42/after.png?raw=true"
    )
    assert evidence_url("o/r", "/a b/c#1.png") == "https://github.com/o/r/blob/factory/evidence/a%20b/c%231.png?raw=true"
