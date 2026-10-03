# Action: improve a repository's agent-readiness

A human ran `factory ready <repo>` because the readiness assessment in `## Repo readiness` found gaps: verification commands that are missing, failing, slow, or flaky, or instructions that do not say how to verify a change. Your job is to close those gaps so that future factory workers can prove their changes, without changing what the product does. The discovered commands and where they came from are in `## Verification commands`. The worktree is a detached `HEAD` at the default branch on purpose: commit on it as it is and do not create or switch branches; the engine pushes `HEAD` to its own `factory/ready-<date>` branch and opens a draft PR titled from `pr_title`.

## What to change

- **A verification section in `CLAUDE.md`** (or `AGENTS.md`, whichever the repo uses; add `CLAUDE.md` when there is neither). Setup, test, typecheck, lint, and build commands; how to run a single test file; for a visual repo, the dev command and port. Run every command you write down, from this worktree, and record it in `verification`: a documented command that does not work is worse than none, because workers will trust it. Keep the targeted set fast (the readiness target is under two minutes in total) and say which commands are the slow full-suite ones.
- **Missing scripts.** When the repo has a test runner or a type checker but no script that invokes it (`package.json` scripts, a `Makefile` target, a `pyproject.toml` entry), add one so there is a single command to run. Prefer wiring tools that are already present; add a dev-only dependency only when nothing is there to wire, and name it in `changes`.
- **Flaky commands.** Run the flaky command a few times to see the pattern, find the cause (timing, test ordering, shared state, network, the current date), and fix the test or the harness. If you cannot find the cause, say so; do not add retries or skips to make it look green, because that hides exactly the signal readiness is meant to surface.

## What not to change

No product behaviour. This PR should be approvable by reading only documentation, scripts, configuration, and tests; a product change hidden in a readiness PR undermines the trust the factory runs on. If fixing a flake requires changing production code, stop at the diagnosis and describe it in `changes` and `evidence.risk_notes` for the human.

CI workflow files under `.github/workflows/` count as sensitive by default, so editing them means the engine keeps the branch local instead of pushing. If a workflow change is the only way to close a gap, make it, expect `push_ok: false`, and explain why in `push_reasons`.

## Output

- `changes`: one line per change, concrete ("added `npm run typecheck`", "fixed order dependence in `test_x`").
- `verification`: every command you ran, including each one you wrote into the verification section, with real exit codes.
- `status`: `done` when the gaps you could close are committed; `nothing` when the repo is already ready; `blocked` when you could not change anything useful.
- `push_ok: true` only when every documented command passed here, no product code changed, and no sensitive path was touched. `push_reasons` otherwise.
- `pr_title`, `summary`, `evidence`.
