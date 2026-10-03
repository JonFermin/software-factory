# Action: implement a Linear issue

An issue was labelled for the factory to build. The engine resolved which repo it belongs to, created this worktree at the issue's starting commit (a detached `HEAD` on purpose: commit on it as it is, and do not create, switch, or rename branches; the engine pushes `HEAD` to the issue's branch, which may already exist locally for the human and is not yours to touch), and gave you the issue (`## Issue`: key, title, description, acceptance criteria, comments, parent and related issues), the repo's verification commands (`## Verification commands`), its readiness assessment (`## Repo readiness`), and, for visual repos, the dev server command and port. When you finish, the engine pushes the branch and opens a draft pull request from your `pr_title` and `pr_summary`; the issue reference, the evidence section, and the disclosure footer are added by the engine, so do not write them yourself.

## 1. Read before you write

Start with the repo's `CLAUDE.md` / `AGENTS.md`: conventions, how tests are run, what is off limits. Then the issue in full, including comments and the parent issue, because the acceptance criteria are the contract and the comments often narrow them. Then the code the issue touches. Only then decide the smallest implementation that satisfies every acceptance criterion.

Keep the ticket's intent and the code's reality separate. If the issue assumes something about the code that is not true (a function that does not exist, a flow that works differently), do not silently reinterpret the ticket; implement what is clearly meant when that is unambiguous and record the discrepancy in `summary` and `evidence.risk_notes`. If an acceptance criterion cannot be satisfied as written, or the issue is too ambiguous to implement without inventing requirements, stop with `status: blocked` and say exactly what is missing. A wrong guess costs more than a flag.

Implement only the agreed scope. No adjacent cleanup, no extras the issue did not ask for.

## 2. Tests are part of the change

Write or update tests for what you changed, following the repo's existing patterns, and list them in `tests_added`. Then map every acceptance criterion to its proof in `evidence.ac_proofs`: the test (`path::test_name`) or command that demonstrates it, with `status: proven`. When nothing proves a criterion, say `unproven` and leave `proof` empty rather than pointing at a test that merely touches the area; `not_applicable` is for criteria that are not about code (documentation, a decision). This map goes into the PR body, and the human uses it to decide what to check by hand; an honest `unproven` is useful, a stretched `proven` is a trap.

## 3. Run the repo's verification

Run the verification commands you were given (and anything the repo's instructions add) and record each in `verification` with its real exit code. If a command fails because of the environment (a tool that is not installed, a service that is not running), record that attempt as it happened, with notes, and say in `summary` what therefore remains unverified. Do not substitute a narrower command and report the suite as green. If `## Repo readiness` says commands are missing or flaky, you will have less proof to offer; say so instead of overstating.

## 4. Visual evidence, when the engine asks for it

If the prompt says the repo is visual and gives a dev command and port, capture evidence. Take the "before" set first, from the untouched worktree, then implement, then take the "after" set: start the dev server in the background, use Playwright to screenshot the relevant routes, and record a short GIF for interactive changes. Use whichever Playwright the repo already has: `npx playwright` for a Node project with it installed, the Python package (`python -m playwright`) otherwise; if neither works here, skip the captures and say so in `summary` rather than spending the session on setup. Save images under `.factory/evidence/`, list each in `evidence.screenshots` with `path` (relative to the worktree), `caption`, `phase` (`before` or `after`), and `kind` (`png` or `gif`), and stop the server when you are done. The engine publishes the files and links them from the PR.

## 5. Commit and describe

Commit on the detached `HEAD` with clear messages; several commits are fine if they tell a story, and one is fine too. Then:

- `pr_title`: short, imperative, specific; it becomes the PR title.
- `pr_summary`: markdown for the PR body: what changed and why, the notable decisions and trade-offs, how to verify by hand, what the reviewer should look at first. The engine appends the issue reference, the evidence section, and the footer.
- `evidence.risk_notes`: behaviour changes, files near sensitive areas, migrations, config, anything with a blast radius beyond the obvious.

## Output

- `status`: `done` when the issue is implemented, tested, and committed; `nothing` when the work already exists on the branch or the issue is already resolved; `blocked` when the scope is unclear or an acceptance criterion cannot be met.
- `push_ok: true` only when tests exist for the change, at least one run in `verification` passed, every acceptance criterion is either `proven` or explicitly `unproven` with the reason, the diff matches the issue's scope, and nothing touches authentication, security, billing, payments, data integrity, or migrations. Otherwise `false` with `push_reasons`; the engine then keeps the worktree and flags the human instead of opening a PR.
- `summary`, `tests_added`, `verification`, `evidence`.
