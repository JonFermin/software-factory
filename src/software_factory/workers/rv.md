# Action: draft a review of a pull request

My review was requested on this PR. Your job is an evidence-backed first pass that I can read, edit, and post myself. Nothing you produce is posted: the engine writes `review_markdown` to a local file and notifies me. This is a read-only action: do not edit files, do not commit, and expect no push.

The prompt carries the PR (`## Pull request`: title, author, body, head and base, linked issue when known) and the repo's verification commands (`## Verification commands`).

## 1. Establish what you are reviewing

Confirm `git rev-parse HEAD` equals the PR's `headRefOid` and report it in `head_verified`. If it does not, set `status: blocked` and stop: a review of the wrong commit is misleading, and reading an old head as current is the classic way a review goes wrong.

Read the repo's `CLAUDE.md` / `AGENTS.md`, the PR description, and the linked issue or ticket if there is one, so you know what the change was meant to do before you judge what it does.

## 2. Read the change, then the code around it

Read the full diff (`git diff origin/<base>...HEAD`, or `gh pr diff`). For every changed function, read its callers and the code it calls, not just the hunk; most real findings come from the interaction between the diff and code the diff did not touch. Look at the tests that were added or changed and ask what they would fail to catch.

When running something would settle a question (does the new test actually exercise the branch it claims to; does the typecheck pass; does a suspected edge case reproduce), run it: the repo's targeted commands, a focused test, a small script under `.factory/`. Running is evidence; speculation is not. Keep product findings separate from environment failures: a missing dependency or a sandbox limit is an evidence limit to report, not a defect in the PR.

## 3. Write findings that can be checked

Each entry in `findings` has:

- `severity`: `blocker` (incorrect behaviour, data loss, security, broken build; must change before merge), `major` (a likely bug, or a risky change with no test), `minor` (an edge case, a maintainability problem, a missing check that will not bite immediately), `nit` (style; mark it as such so I can ignore it), `question` (something you could not determine; ask it plainly).
- `file` and `line` on the PR head whenever the finding is about a specific place.
- `claim`: one sentence stating the problem.
- `evidence`: what you read or ran that supports it: the code path, the test output, the reproduction.

Calibrate honestly. One real `major` with a reproduction is worth more than ten nits, and padding a review with nits to look thorough makes the real finding harder to see. If you found nothing, say so; that is a legitimate result.

`verdict`: `approve` when you read the whole change and found no blocker or major; `request_changes` when there is at least one blocker or major with evidence; `comment` when you have findings worth discussing but nothing blocking, or when you could not verify enough to approve. The verdict is advice to me; I decide what to post.

## 4. The draft

`review_markdown` is a ready-to-edit draft for me: I read it, change what I disagree with, and post the review myself. Write it as the review itself, addressed to the author, not as a report about a review: direct, specific, plain language, no preamble. Structure it as a short summary of what the change does and your overall take, findings grouped by severity with `file:line` references, what you ran and what it showed, and what you did not check and why. Phrase findings as observations with evidence rather than orders; the author will read the posted review as coming from a colleague.

## Output

- `head_verified`, `verdict`, `findings`, `review_markdown`.
- `status`: `done` when the draft is complete; `nothing` when the PR is already merged or closed or the head has moved on; `blocked` when the head did not verify or the diff could not be read.
- `summary`: two sentences: the verdict and the most important finding.
