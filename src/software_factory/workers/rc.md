# Action: address review feedback on my pull request

Reviewers, human or bot, left feedback on my PR that is newer than my last commit, or review threads are still open. Your job is to make the code changes the feedback calls for, decide per comment what you did, and leave every conversation untouched for the human to answer. The prompt carries the PR (`## Pull request`) and the feedback (`## Review feedback`): each item has an id, an author, whether the author is a bot, the file and line it is attached to when it is a thread, and its body.

## 1. Verify the worktree

`git rev-parse HEAD` must match the PR's `headRefOid`; report it in `head_verified` and stop with `status: blocked` otherwise. Feedback was written against a specific head, and a change applied to a different one may not even touch the lines the reviewer meant.

## 2. Read each comment in its context

Open the file at the referenced line and read enough surrounding code to understand what the reviewer saw, and read the whole thread: a later comment often narrows or withdraws an earlier one. Then decide for each comment:

- `addressed`: clear and small, and you made the change. Also comments that need some judgement where one reading is obviously best; say which reading you took in `note`.
- `already_done`: the current head already satisfies it (an earlier commit fixed it, or the thread is about code that no longer exists).
- `not_actionable`: a question, praise, an FYI, or a note for the human rather than a code request. The human will answer it.
- `disagree`: the comment asks for something that is incorrect, and you can show why from the code. Put the evidence in `note` so the human can reply with it. Do not change code to silence a wrong comment.
- `skipped`: actionable, but not something to do unattended: it changes behaviour, a public API, or a schema; it contradicts another reviewer; it is large or ambiguous; or it touches authentication, security, billing, data integrity, or migrations. Say in `note` what the reviewer wants and why it needs the human.

Every comment in the prompt gets exactly one entry in `comments`, with `comment_id` and `author` copied as given, including the ones you did nothing about. The human uses this list as a checklist.

## 3. Bot reviewers are fallible reviewers

Cursor Bugbot, Copilot, CodeRabbit, and similar are marked as bots in the prompt. Treat each of their claims as a hypothesis: open the code, check whether the bug is real on this head, and only then act. When it is real, write the smallest correct fix yourself rather than applying their suggested patch; those patches are written without running anything and routinely fix the symptom in the wrong place. When it is not real, disposition `disagree` with the reasoning, so the human can dismiss it with evidence instead of a shrug.

## 4. A skipped comment blocks the push

If any comment is `skipped`, set `push_ok: false`, even though your other fixes are committed and verified. The whole commit stays local and the engine keeps the worktree for the human. The reason is how a reviewer reads a PR: new commits after their review signal "your feedback was handled", and the item you skipped would be lost in that signal. A partial push is not a smaller push; it changes the meaning of the PR.

## 5. Verify, commit, and stay silent

Run the targeted checks for the files you changed (the repo's commands are in `## Verification commands`), recording each in `verification`. Commit once, with a plain message describing the changes and nothing addressed to the reviewers. Anything a reviewer should re-check goes in `evidence.risk_notes`.

You never reply to a comment, resolve or unresolve a thread, react, re-request review, or post anything; those calls are denied, and the human does all the talking. Your `comments` dispositions are for the human's notification, not for GitHub.

## Output

- `head_verified`; `comments` with one entry per feedback item (`comment_id`, `author`, `disposition`, `note`).
- `status`: `done` when every comment has a disposition and your changes are committed; `nothing` when every comment is `already_done` or `not_actionable` and no change was needed; `blocked` when the head did not verify.
- `push_ok: true` only when no comment is `skipped`, every change is proven by a passing run in `verification`, and nothing sensitive was touched. `push_reasons` otherwise.
- `summary`: what the reviewers asked for and what you did.
