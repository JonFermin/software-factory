# Action: pick the repository an issue belongs to

The engine could not determine from labels, links, sibling issues, or name matches which repository this issue belongs to, and is asking you to decide from the text alone. You have no tools in this session; the issue (`## Issue`) and the candidate repositories with a summary of each (`## Candidate repos`) are all the evidence there is.

Your answer decides where the build worker will implement the issue, so a confident wrong pick is the worst outcome: it produces a PR in the wrong repository that someone has to notice and close. A null or a low confidence just sends the issue to the human.

## Decide

- `repo`: the slug of exactly one candidate, copied as given, or `null` when no single candidate fits.
- `confidence`: your honest probability that this is the right repository. Around 0.9 and above when the issue names the repo, its files, or a domain only one candidate owns; 0.7 to 0.9 when the domain clearly matches one candidate and nothing points elsewhere; below 0.7 when you are choosing between plausible candidates or extrapolating from a title. The engine treats anything below its threshold as unresolved and flags it, so do not round up.
- `multi_repo: true` with `repo: null` when the work clearly spans more than one repository; the planner will split it into per-repo sub-issues.
- `reasoning`: two or three sentences naming the evidence you used, so the human can check it in seconds.

Weigh the issue's project, team, labels, explicit paths, and description over title keywords; a title word that happens to match a repository name is weak evidence when the body describes something else. When the issue describes a domain that no candidate's summary covers, answer `null` rather than the closest-sounding name.
