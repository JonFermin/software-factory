# Action: resolve a merge conflict on my pull request

GitHub reports the PR as `CONFLICTING` with its base. Your job is to bring the base branch into the PR branch and resolve what can be resolved safely, leaving a clean handoff for anything that cannot. The prompt carries the PR with head and base names and SHAs (`## Pull request`) and the repo's verification commands (`## Verification commands`).

## 1. Verify the worktree

`git rev-parse HEAD` must equal the PR's `headRefOid`; report it in `head_verified` and stop with `status: blocked` if it does not. A conflict resolved on the wrong commit is worse than no resolution.

## 2. Merge the base in; never rebase

`git fetch origin <base>`, then `git merge origin/<base>`. A merge commit is the only option here: a rebase rewrites the branch's history, which can only be pushed with force, and force-push does not exist in this pipeline (the engine pushes `HEAD:refs/heads/<branch>` only). The same goes for `git reset`, amending existing commits, or re-creating the branch; any of those produces a branch that cannot be pushed.

Do not `git merge --abort` when the conflict turns out to be hard. The kept worktree is how the human picks up your work, and an aborted merge throws away both the safe resolutions and the classification you did.

## 3. Classify every hunk before resolving any

Go through every conflicted file and every hunk in it and decide what kind of conflict it is. Looking at all of them first matters because the right resolution for one hunk often depends on what the other side did elsewhere.

- `trivial`: whitespace, import ordering, two independent additions in the same region, a version string. Resolve directly.
- `mechanical`: generated files, lockfiles, snapshots, compiled assets. Take either side, then regenerate with the repo's own command; never hand-merge generated content.
- `semantic`: both sides changed the same logic and the result must be reasoned about. Resolve only when one reading is clearly correct (base renamed a function this branch calls; both sides made the same fix differently and either works). Describe the reasoning in `resolution`.
- `unsafe`: both sides changed the same function's behaviour in different directions; the conflict spans many files or has a blast radius you cannot describe; or it sits in authentication, security, billing, data integrity, or migrations. Do not resolve.

Taking `ours` or `theirs` wholesale on a semantic hunk silently drops someone's work; if you catch yourself reaching for that, the hunk is probably `unsafe`.

## 4. Resolve what is safe; hand the rest over cleanly

Resolve every trivial and mechanical hunk and every semantic hunk you are confident in, and `git add` those files.

If any hunk is `unsafe`, or semantic and you are not sure: leave those files exactly as git left them, conflict markers and all, unstaged, and do not commit the merge. Set `all_resolved: false`, `push_ok: false`, and `status: blocked`. The engine then keeps this worktree and flags the human, who continues from your partial resolution with your `hunks` notes as the guide. A merge commit with markers inside it, or one that guesses, is the worst outcome because it looks finished.

When everything is resolved: run the targeted verification for the files involved (build, typecheck, the tests touching the conflicted areas; the repo's commands are in the prompt) and record each in `verification`. Commit the merge with git's default merge message or a short plain one.

## Output

- `head_verified`; `hunks` with one entry per hunk, or per file when the whole file resolves the same way (`file`, `classification`, `resolution`); `all_resolved`.
- `status`: `done` when the merge is committed and verified; `blocked` with `all_resolved: false` when you resolved part and left a clean handoff, or when the head did not verify; `nothing` when the branch already merges cleanly.
- `push_ok: true` only when `all_resolved` is true, no hunk was classified `unsafe`, verification passed, and nothing sensitive was involved; otherwise `false`, with `push_reasons` naming the hunks or checks that stopped it. The engine also checks for leftover conflict markers before pushing.
- `summary`: what conflicted, what you did, what is left. Anything the reviewer should re-check after the merge goes in `evidence.risk_notes`.
