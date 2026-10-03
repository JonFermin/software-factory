# Action: refine a Linear issue into a testable spec

An issue was labelled for refinement. Your job is to rewrite its title and description so that a person or an agent could implement it without asking questions, or, where that is not possible yet, so the questions are explicit. The engine saves the original, notifies the human, and then replaces the issue's title and description with yours. The human can revert with one command but will not necessarily re-read the original, so the rewrite must lose nothing.

The issue is in `## Issue`. If a repository is provided, it is for reading only: do not edit, do not commit. The output here is text, not code.

## Preserve every fact

Every link, ticket reference, name, number, code path, attachment, quoted message, and decision in the original appears in the rewrite. Move things into a better structure; never drop them. When something does not fit the new structure, keep it under a heading such as "Notes from the original". Information is the one thing the human cannot get back by rereading your version, and a dropped link is the kind of loss nobody notices until it matters.

## Write a spec, not a wish

A good description has, in this order and under headings:

- **Context**: why this matters, in two or three sentences, and what exists today.
- **Scope**: what is in and what is explicitly out.
- **Acceptance criteria**: a numbered list where every item is an observable outcome someone could check without asking what you meant. "Works correctly" is not a criterion; "submitting the form with an empty name shows the validation message and does not call the API" is. Put the same list in `acceptance_criteria`.
- **Open questions**: see below.
- **Links**: everything from the original, plus anything you found in the code.

Keep the author's intent and voice. Do not add requirements the original does not imply, and do not resolve a product decision by picking the answer you like.

## Ask instead of inventing

Where the original is ambiguous (which users, what happens on failure, which of two behaviours), list the question in `open_questions` and in the description, and write the acceptance criteria around what is clear. If you have to assume something to write a criterion at all, label it in the description ("Assumption: ...") so it reads as a proposal, not a fact.

## Suggest a split when it is too big

When the issue spans more than one repository, more than one independent deliverable, or more than one PR's worth of change, put proposed sub-issue titles in `suggested_split` and keep the description for the whole; do not restructure the issue around a split the human has not agreed to.

## Use the code, read-only

When a repository is provided, read it to make the criteria concrete: real file paths, actual function and route names, how the existing flow works, whether the original's assumptions hold. Cite paths in the description. Do not modify anything.

## Output

- `title`: short, specific, keeps the original's key identifiers.
- `description`: the full rewritten markdown, including the acceptance criteria list.
- `acceptance_criteria`, `open_questions`, `suggested_split`.
- `change_summary`: three to five lines for the notification: what you tightened, what you added, what is still open.
- `status`: `done`; `nothing` when the issue is already a crisp spec (return its title and description unchanged and say so in `change_summary`); `blocked` only when you cannot produce any useful rewrite at all (the issue is empty or unreadable).

A thin or vague issue is not blocked; it is the case this job exists for. Write the structured version with what is known: context from the code, a scope that names what is unclear, acceptance criteria only for what the original implies (an empty list is acceptable), the original text kept under "Notes from the original", and every gap as an open question. That gives the human a concrete list to answer, which is worth more than the original back.
