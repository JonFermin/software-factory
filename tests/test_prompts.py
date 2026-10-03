"""Static worker prompts (``workers/<kind>.md`` + ``workers/_common.md``).

The prompts are plain markdown appended to the Claude Code preset system prompt. These tests pin
what other modules rely on: every kind has one, each prompt names every field and enum value of its
output schema (so prompt and schema cannot drift apart silently), the engine's user-prompt headings
are referenced, and each kind's guardrails are present.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel

from software_factory.actions import Kind
from software_factory.workers import schemas

PROMPT_NAMES = [kind.value for kind in Kind] + ["resolve", "_common"]

# Emphasis by capitals is "shouting"; the prompts explain why instead.
SHOUTING = re.compile(r"\b(NEVER|ALWAYS|IMPORTANT|CRITICAL|MUST|DO NOT|WARNING)\b")

# Headings the engine uses in the user prompt; each kind prompt must point at the ones it consumes.
USER_PROMPT_HEADINGS: dict[str, list[str]] = {
    "ci": ["## Pull request", "## Failing checks", "## Verification commands"],
    "conflict": ["## Pull request", "## Verification commands"],
    "rc": ["## Pull request", "## Review feedback", "## Verification commands"],
    "rv": ["## Pull request", "## Verification commands"],
    "build": ["## Issue", "## Verification commands", "## Repo readiness"],
    "ready": ["## Repo readiness", "## Verification commands"],
    "refine": ["## Issue"],
    "plan": ["## Issue"],
    "resolve": ["## Issue"],
}


def read_prompt(name: str) -> str:
    return (resources.files("software_factory.workers") / f"{name}.md").read_text(encoding="utf-8")


def mentions(text: str, name: str) -> bool:
    """True when ``name`` appears as code: `name`, `name: value`, `name.child` or `name[]`."""
    return re.search(rf"`{re.escape(name)}(?:`|[:.\[])", text) is not None


def literal_values(model: type[BaseModel], field: str) -> tuple[str, ...]:
    values = get_args(model.model_fields[field].annotation)
    assert values, f"{model.__name__}.{field} is not a Literal"
    return values


# ------------------------------------------------------------------------------ existence


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_every_kind_has_a_non_empty_prompt(name: str) -> None:
    text = read_prompt(name)
    assert len(text.strip()) > 500, f"{name}.md is too short to be a real prompt"
    assert text.lstrip().startswith("# "), f"{name}.md should open with a title heading"


def test_output_models_match_prompt_set() -> None:
    # Every schema has a prompt and every non-common prompt has a schema.
    assert set(schemas.OUTPUT_MODELS) == set(PROMPT_NAMES) - {"_common"}


# ----------------------------------------------------------------------- schema alignment


@pytest.mark.parametrize("kind", sorted(schemas.OUTPUT_MODELS))
def test_prompt_names_every_top_level_output_field(kind: str) -> None:
    text = read_prompt(kind)
    missing = [f for f in schemas.OUTPUT_MODELS[kind].model_fields if not mentions(text, f)]
    assert not missing, f"{kind}.md does not explain output field(s): {missing}"


@pytest.mark.parametrize("kind", [k for k in schemas.OUTPUT_MODELS if k != "resolve"])
def test_prompt_names_every_worker_status(kind: str) -> None:
    text = read_prompt(kind)
    missing = [s for s in get_args(schemas.WorkerStatus) if not mentions(text, s)]
    assert not missing, f"{kind}.md does not say when to use status {missing}"


@pytest.mark.parametrize(
    ("kind", "model", "field"),
    [
        ("ci", schemas.CiFailure, "classification"),
        ("conflict", schemas.HunkResolution, "classification"),
        ("rc", schemas.CommentDisposition, "disposition"),
        ("rv", schemas.ReviewFinding, "severity"),
        ("rv", schemas.RvOutput, "verdict"),
        ("build", schemas.AcProof, "status"),
        ("build", schemas.Screenshot, "phase"),
        ("build", schemas.Screenshot, "kind"),
    ],
)
def test_prompt_names_every_enum_value(kind: str, model: type[BaseModel], field: str) -> None:
    text = read_prompt(kind)
    missing = [v for v in literal_values(model, field) if not mentions(text, v)]
    assert not missing, f"{kind}.md does not define {model.__name__}.{field} value(s): {missing}"


@pytest.mark.parametrize(("kind", "headings"), sorted(USER_PROMPT_HEADINGS.items()))
def test_prompt_points_at_engine_headings(kind: str, headings: list[str]) -> None:
    text = read_prompt(kind)
    missing = [h for h in headings if f"`{h}`" not in text]
    assert not missing, f"{kind}.md does not reference user-prompt heading(s): {missing}"


# ------------------------------------------------------------------------- common preamble


def test_common_says_the_engine_pushes() -> None:
    text = read_prompt("_common").lower()
    assert re.search(r"engine[^.\n]*push", text), "_common.md must say pushing is done by the engine"
    assert "draft pull request" in text
    assert "linear" in text and "slack" in text


def test_common_denial_means_adapt_not_retry() -> None:
    text = read_prompt("_common").lower()
    assert "denied" in text or "denial" in text
    assert re.search(r"do not retry", text)
    assert "no human" in text or "nobody is watching" in text or "no one will answer" in text


def test_common_names_scratch_dir_and_repo_instructions() -> None:
    text = read_prompt("_common")
    assert "`.factory/`" in text
    assert "CLAUDE.md" in text and "AGENTS.md" in text


def test_common_demands_truthful_verification() -> None:
    text = read_prompt("_common")
    for field in ("`push_ok`", "`push_reasons`", "`verification`", "`status`"):
        assert field in text
    lowered = text.lower()
    assert "actually ran" in lowered
    assert "exit code" in lowered
    assert re.search(r"never (list|report) a command", lowered)


def test_common_forbids_history_rewrite_and_no_verify() -> None:
    text = read_prompt("_common")
    assert "--no-verify" in text
    assert "HEAD:refs/heads/<branch>" in text
    assert "rebase" in text.lower()


def test_common_explains_disabled_hooks_git_level_push_block_and_unauthenticated_gh() -> None:
    """Workers commit with ``core.hooksPath`` pointed at an empty dir, push URLs rewritten to an
    unusable transport, and (without GH_WORKER_TOKEN) no gh login: the prompt must say so, and tell
    the worker to run the repo's lint/format commands itself since no hook will."""
    text = read_prompt("_common")
    lowered = text.lower()
    assert "hooks are disabled" in lowered and "core.hookspath" in lowered
    assert re.search(r"run those commands yourself", lowered)
    assert "lint" in lowered and "format" in lowered and "CLAUDE.md" in text
    assert "push url" in lowered and "cannot succeed" in lowered
    assert "not logged in" in lowered and "read-only worker token" in lowered


def test_common_names_the_repo_instructions_heading_the_runner_appends(tmp_path: Path) -> None:
    """Workers run with ``setting_sources=[]``: the repo's CLAUDE.md reaches them only through
    ``runner.project_instructions``. The prompt must point at the heading that function really
    emits, and must no longer suggest the repo's settings file is in effect."""
    from software_factory.runner import project_instructions

    (tmp_path / "CLAUDE.md").write_text("# repo\n\nrun `make lint` before committing\n", encoding="utf-8")
    heading = project_instructions(tmp_path).splitlines()[0]
    assert heading == "## Repository instructions (CLAUDE.md)"
    text = read_prompt("_common")
    assert "`## Repository instructions`" in text and heading.startswith("## Repository instructions")
    assert ".claude/settings.json` is not loaded" in text
    assert "regardless of what the repo's own settings allow" not in text


# ---------------------------------------------------------------------------------- ci


def test_ci_verifies_head_first() -> None:
    text = read_prompt("ci")
    assert "headRefOid" in text and "`head_verified`" in text
    assert "git rev-parse HEAD" in text
    # The HEAD check is step 1, before classification.
    assert text.index("headRefOid") < text.index("Classify every failing check")


def test_ci_classifies_before_fixing_and_fixes_only_pr_causes() -> None:
    lowered = read_prompt("ci").lower()
    assert re.search(r"classif\w*[^.\n]*before touching code", lowered)
    assert re.search(r"only `caused_by_pr` gets a code change", lowered)
    assert "base branch" in lowered  # compare with the base to spot inherited failures


def test_ci_inherited_uses_the_engines_base_branch_line_not_gh() -> None:
    """Workers may run gh unauthenticated, so the ``inherited`` classification must rest on the
    ``base branch:`` line the engine renders under ``## Failing checks`` (kinds/ci.py), not on a
    ``gh run list`` the worker cannot run."""
    from software_factory.kinds.ci import render_base_run

    text = read_prompt("ci")
    inherited = next(line for line in text.splitlines() if line.startswith("- `inherited`"))
    assert "`base branch:`" in inherited and "base branch: <conclusion> at <sha> (<url>)" in inherited
    assert "gh run list" not in text
    assert "unauthenticated" in text.lower()
    # the format the prompt names is the one the engine renders
    line = render_base_run({"conclusion": "failure", "headSha": "abc", "url": "https://r/1"}, "main", "CI")
    assert line == "base branch: failure at abc (https://r/1)"


def test_common_describes_the_unplantable_hooks_path_and_the_end_of_session_kill() -> None:
    text = read_prompt("_common")
    assert "a path that cannot exist" in text and "empty directory" not in text
    assert "When the session ends, whether you answered or the timeout cut it off" in text


def test_ci_targeted_verification_not_full_suite() -> None:
    lowered = read_prompt("ci").lower()
    assert "not the full suite" in lowered
    assert "smallest correct fix" in lowered


def test_ci_push_rule_names_sensitive_areas() -> None:
    lowered = read_prompt("ci").lower()
    for word in ("authentication", "security", "billing", "data integrity", "migrations"):
        assert word in lowered, f"ci.md push rule should name {word}"


# ---------------------------------------------------------------------------- conflict


def test_conflict_merges_never_rebases_never_aborts() -> None:
    text = read_prompt("conflict")
    assert "git merge origin/<base>" in text
    assert "never rebase" in text.lower()
    assert "force-push does not exist" in text.lower()
    assert "--abort" in text


def test_conflict_partial_resolution_handoff() -> None:
    text = read_prompt("conflict")
    assert "conflict markers" in text.lower()
    assert "`all_resolved: false`" in text and "`push_ok: false`" in text
    assert re.search(r"classif\w*[^.\n]*before resolving any", text.lower())
    assert "keeps this worktree" in text.lower()


# ---------------------------------------------------------------------------------- rc


def test_rc_skipped_comment_blocks_the_push() -> None:
    lowered = read_prompt("rc").lower()
    assert re.search(r"skipped[^.\n]*block[^.\n]*push", lowered), "rc.md must say a skipped comment blocks the push"
    assert "partial push" in lowered


def test_rc_never_replies_and_verifies_bot_claims() -> None:
    lowered = read_prompt("rc").lower()
    assert "never reply" in lowered
    assert "resolve or unresolve a thread" in lowered
    for bot in ("cursor bugbot", "copilot", "coderabbit"):
        assert bot in lowered
    assert "fallible" in lowered
    assert "smallest correct fix" in lowered


def test_rc_one_disposition_per_comment() -> None:
    lowered = read_prompt("rc").lower()
    assert "exactly one entry" in lowered
    assert "`comment_id`" in lowered and "`author`" in lowered


# ---------------------------------------------------------------------------------- rv


def test_rv_is_read_only_and_never_posted() -> None:
    lowered = read_prompt("rv").lower()
    assert "nothing you produce is posted" in lowered
    assert "do not edit files, do not commit" in lowered
    assert "headRefOid" in read_prompt("rv")


def test_rv_findings_cite_evidence_and_calibrate() -> None:
    lowered = read_prompt("rv").lower()
    assert "`file:line`" in lowered
    assert "`evidence`" in lowered
    assert "calibrate" in lowered
    assert "ready" in lowered or "paste" in lowered  # review_markdown is a ready-to-paste draft


# ------------------------------------------------------------------------------- build


def test_build_maps_acceptance_criteria_and_runs_verification() -> None:
    text = read_prompt("build")
    assert "`evidence.ac_proofs`" in text
    assert "`unproven`" in text
    assert "`tests_added`" in text
    assert "CLAUDE.md" in text and "AGENTS.md" in text
    assert "real exit code" in text.lower()


def test_build_visual_evidence_with_playwright() -> None:
    lowered = read_prompt("build").lower()
    assert "playwright" in lowered
    assert "npx playwright" in lowered and "python -m playwright" in lowered
    assert ".factory/evidence/" in lowered
    assert "gif" in lowered
    assert re.search(r"skip the captures and say so", lowered)


def test_build_leaves_issue_reference_and_footer_to_engine() -> None:
    lowered = read_prompt("build").lower()
    assert "do not write them yourself" in lowered
    assert "`pr_title`" in lowered and "`pr_summary`" in lowered


# ------------------------------------------------------------------------------- ready


def test_ready_no_product_behaviour_changes() -> None:
    lowered = read_prompt("ready").lower()
    assert re.search(r"no product behaviou?r", lowered)
    assert "verification section" in lowered
    assert "flaky" in lowered
    assert ".github/workflows/" in lowered  # warns that workflow edits are sensitive


# ------------------------------------------------------------------------------ refine


def test_refine_preserves_everything_and_asks_instead_of_inventing() -> None:
    lowered = read_prompt("refine").lower()
    assert "never drop them" in lowered
    assert "every link" in lowered
    assert "`open_questions`" in lowered and "`suggested_split`" in lowered
    assert "assumption:" in lowered
    assert "read-only" in lowered or "reading only" in lowered


# -------------------------------------------------------------------------------- plan


def test_plan_one_repo_per_sub_issue_with_dependencies() -> None:
    lowered = read_prompt("plan").lower()
    assert "exactly one repository" in lowered
    assert "`blocked_by`" in lowered and "0-based" in lowered
    assert "independently shippable" in lowered
    assert "nothing more" in lowered  # do not plan beyond the parent


# ----------------------------------------------------------------------------- resolve


def test_resolve_calibrated_confidence_and_multi_repo() -> None:
    text = read_prompt("resolve")
    assert "`null`" in text
    assert "`multi_repo: true`" in text
    assert "0.7" in text  # the engine's default min_confidence appears as a calibration anchor
    assert "no tools" in text.lower()


# ------------------------------------------------------------------------ style guards


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_no_shouting(name: str) -> None:
    hits = SHOUTING.findall(read_prompt(name))
    assert not hits, f"{name}.md shouts: {hits}"
