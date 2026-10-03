# Action: fix failing CI on my pull request

The PR is mine, its current head has at least one failing check, and your job is to work out why and, when the PR itself is the cause, fix it. The prompt carries the PR with its head and base (`## Pull request`), the failing checks with their logs (`## Failing checks`), and the repo's verification commands (`## Verification commands`).

## 1. Verify the worktree before trusting it

Compare `git rev-parse HEAD` with the PR's `headRefOid` from the prompt and report the result in `head_verified`. If they differ, stop: `status: blocked`, `head_verified: false`, and say what you found. Fixing a different commit than the one the checks ran against produces a commit that may not even apply to the PR, and the engine will not push anything from an unverified head. The worktree is created at the head, but runs have started elsewhere before, which is why this check exists.

The failures you were given also belong to a specific head. If the logs clearly reference an older commit than HEAD, say so; a stale failure is not the current state of the PR.

## 2. Classify every failing check before touching code

Read the logs in the prompt first. `gh run view <run_id> --log-failed` and `gh pr checks` work only when the profile gave you a read-only token; you may well be running `gh` unauthenticated, so do not build a classification on them. For each failing check decide:

- `caused_by_pr`: the failure is in code, tests, or config this PR changed, or in something the PR's changes depend on.
- `flaky`: timing, ordering, network, or resource related, unrelated to the diff; it would likely pass on rerun.
- `inherited`: the same thing fails on the base branch. Each Actions check under `## Failing checks` carries a `base branch:` line: the engine looked up the latest run of that workflow on the PR's base branch, as `base branch: <conclusion> at <sha> (<url>)`. A `failure` there is the evidence for `inherited`; a `success` means the base is green and the failure is the PR's own or flaky; `not checked` or no line means the base state is unknown, so do not claim `inherited` on it alone. Also check whether the PR touches the failing area at all (`git diff origin/<base>...HEAD --stat`) and whether a recent base commit changed the failing code (`git log origin/<base>`). Cite the `base branch:` line in `cause`.
- `infra`: runner, dependency download, quota, or an outage; nothing in the repo is at fault.
- `unknown`: you could not determine the cause with the evidence available. Say what you looked at.

Write the reasoning into `failures[].cause`. Only `caused_by_pr` gets a code change. The others get a classification with evidence so the human can rerun, ignore, or chase the base branch; changing code to make a flaky or inherited failure go quiet hides the real problem and makes the PR carry a fix that is not its own.

## 3. Make the smallest correct fix

Read the repo's `CLAUDE.md` / `AGENTS.md` for how it runs, formats, and generates things. Fixes that are mechanical and safe, even when the diff is large (judge by intent, not line count):

- lint or format autocorrect, run on the files this PR changed, not the whole repo
- generated files, lockfiles, and snapshots regenerated with the repo's own command, never hand-edited
- a test, fixture, or type annotation correction that the failing check proves is wrong
- a sequence or version collision (a renumbering) with no logic change

A bug in code this PR itself introduced or changed is the PR's own mistake, and fixing it is the point of this job: when the PR's intent is unambiguous (its title, description, names, and tests agree on what the code should do, as with a `multiply` that adds), fix the code, prove it with the failing test, and push. That is not "changing production logic to make a test pass"; it is making the PR do what it says.

Fixes that need a human first. Make them when they are clearly right, but set `push_ok: false` and say why:

- changing production logic the PR did not touch, or production logic where the test and the code disagree and nothing in the PR says which one is right (then the question is which behaviour the author wants, and that is theirs to answer)
- adding or upgrading a dependency
- anything that touches authentication, security, billing, payments, data integrity, or migrations
- a fix whose blast radius you cannot describe in one sentence

## 4. Verify with a targeted command, then commit

Run the specific test file, lint target, or build step the check corresponds to, using the repo's commands from `## Verification commands` or its instructions. Not the full suite: it is slow, its unrelated failures muddy the signal, and CI will run it after the push anyway. Record each command in `verification` with its real exit code. If the targeted command cannot run here (missing tool, needs a service), record that attempt with its exit code and notes rather than substituting something that passes.

Commit once with a plain message describing the fix. Anything a reviewer should double-check goes in `evidence.risk_notes`.

## Output

- `head_verified`; `failures` with one entry per failing check in the prompt (`check`, `classification`, `cause`, `fixed`).
- `status`: `done` when every check is classified and the ones the PR owns are fixed; `nothing` when the checks are already green on this head or every failure is stale; `blocked` when the head did not verify or a `caused_by_pr` failure is beyond a safe fix.
- `push_ok: true` only when every fixed failure has an identified cause, the fix is proven by a passing targeted run in `verification`, and nothing sensitive was touched. Otherwise `false`, with `push_reasons`. Flaky, inherited, and infra failures do not need a push; if those are the only failures, `status: done` with no commits and `push_ok: false` ("nothing to push") is the right answer.
- `summary`: two or three sentences: what failed, why, what you did.
