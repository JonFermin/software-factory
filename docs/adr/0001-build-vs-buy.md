# ADR 0001: Build vs buy for the ticket -> PR path

- **Status:** Accepted
- **Date:** 2026-10-01
- **Scope:** plan build-order step 5 (timeboxed build-vs-buy check for `build`, i.e. Linear issue ->
  draft PR). Upkeep kinds (`ci`, `conflict`, `rc`, `rv`), `refine` and `plan` are out of scope except
  where an external option would also cover them.
- **Method:** ~1 hour of desk research on 2026-10-01 against vendor documentation (URLs cited inline).
  No option was trialled hands-on. Where a document did not answer a question, this ADR says so rather
  than guessing. Third-party pricing round-ups are cited as secondary sources and marked as such.

## Context

The factory (see `docs/architecture.md`) is a deterministic, locally scheduled engine that scans
Linear and GitHub state, dispatches one narrow Claude Agent SDK worker per action, and performs every
outward write itself through `effects.py`. The plan's default path implements the `build` kind in-house.
Before committing to that, the plan asks whether an existing product already does "labelled ticket ->
draft PR" well enough that step 6 should become "dispatch to it by label + include its PRs in upkeep
scans" instead of a `build` worker.

The evaluation criteria are the plan's hard requirements, restated as a checklist:

| # | Requirement | What it means concretely |
|---|---|---|
| R1 | Deterministic dispatch | No LLM decides *whether* or *when* work starts; a label/state change does. The factory must be able to trigger the option from its own scan loop. |
| R2 | Structural outward-write limits | "Never comment" (GitHub or Linear), drafts only, never merge / close / ready-for-review / force-push, enforced by something other than a prompt. |
| R3 | Local repos + local verification on the user's own machine | Work happens in a worktree of the local clone; the repo's discovered verification commands run on that machine, Windows included (Playwright captures for `visual` repos included). |
| R4 | Evidence section in the PR body | Machine-maintained `<!-- factory:evidence -->` block: AC -> test map, risk summary, verification results, screenshots. |
| R5 | Readiness warnings | Low-readiness repos get a warning in the evidence section and a flag suggesting `factory ready <repo>`. |
| R6 | Dedupe / attempt caps | `(target, kind, version)` handled-once semantics and `max_attempts` per issue. |
| R7 | Profiles / portability | Same install usable for a `personal` profile (Linear, OAuth) and an employer profile (any tracker, API key / Bedrock / Vertex, `observe` first) without crossing streams. |
| R8 | Cost on the subscription | Runs should draw on a Claude subscription the user already pays for, not add a second bill, within whatever the terms allow. |

A ninth, cross-cutting constraint surfaced during research and is treated as a first-class input to the
decision: **R9, authentication terms.** Which options Anthropic explicitly allows to run on subscription
OAuth, and which it steers to API keys.

## Options considered

### O1. In-house `build` worker (the plan's default)

A Python `claude-agent-sdk` session in a throwaway worktree, denied all outward writes by `policy.py`
(`can_use_tool` + `PreToolUse` hook), returning a typed `BuildOutput`; the engine pushes
`HEAD:refs/heads/<branch>`, opens a draft PR via `gh`, stamps the evidence section, transitions the
Linear issue. Everything in R1-R7 is by construction, because the factory *is* the dispatcher, the
policy layer and the effects layer.

- **Pros:** meets R1-R7 exactly as specified (that is what the architecture was designed to do); one
  codebase for `build` and upkeep, so evidence/readiness/metrics are shared; portable across trackers and
  providers via `Tracker` and `provider.kind`.
- **Cons:** it is the most code to write and maintain (worker prompt, resolver chain, effects,
  Playwright capture path); it carries the R9 question below on the `personal` profile; model quality is
  whatever the SDK gives us with no vendor-side harness improvements for free.
- **R9 (auth) for O1.** The Claude Code legal page says two things that pull in different directions
  ([legal-and-compliance](https://code.claude.com/docs/en/legal-and-compliance)):
  - "Advertised usage limits for Pro and Max plans assume ordinary, individual usage of Claude Code and
    the Agent SDK." (This sentence contemplates individual Agent SDK use on a subscription.)
  - "Developers building products or services that interact with Claude's capabilities, including those
    using the Agent SDK, should use API key authentication through Claude Console or a supported cloud
    provider. Anthropic does not permit third-party developers to offer Claude.ai login into their own
    applications, or to route requests through Free, Pro, or Max plan credentials on behalf of their
    users." The Agent SDK overview repeats it: "Unless previously approved, Anthropic does not allow
    third party developers to offer claude.ai login or rate limits for their products, including agents
    built on the Claude Agent SDK" ([agent-sdk/overview](https://code.claude.com/docs/en/agent-sdk/overview)).
  - A February 2026 press summary reads the change as banning subscription OAuth "in all contexts
    outside of Claude Code and Claude.ai" ([alternativeto, 2026-02](https://alternativeto.net/news/2026/2/anthropic-officially-bans-using-subscription-authentication-for-third-party-claude-use));
    that is a secondary source and stricter than the primary text, which is framed around third parties
    serving *their users*.
  - **Honest position:** a single-user tool run by the subscription owner on their own machine for
    their own repos is not the case the prohibition describes, but it is also not explicitly carved out
    the way the unmodified Claude Code binary is. This ADR does not resolve that; the plan already
    assigns the owner the prerequisite "confirm current terms allow unattended use on your plan". The
    architecture already hedges: `provider.kind = "api_key" | "bedrock" | "vertex"` are first-class, and
    `employer = true` profiles reject `oauth` outright (`config.py`). If the owner decides OAuth is not
    appropriate for the factory, the switch is one config line and the cost moves to API billing
    (Opus 5.5 is listed at $4 / $20 per million input/output tokens on the API price list as of late
    September 2026, Sonnet 5.5 at $2 / $10; see [claude.com/platform/api](https://claude.com/platform/api)).

### O2. Claude Code GitHub Action (`anthropics/claude-code-action@v1`)

What it is ([docs](https://code.claude.com/docs/en/github-actions),
[usage.md](https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md),
[capabilities-and-limitations.md](https://github.com/anthropics/claude-code-action/blob/main/docs/capabilities-and-limitations.md),
[security.md](https://github.com/anthropics/claude-code-action/blob/main/docs/security.md)): Claude Code
run inside a GitHub Actions job on a GitHub-hosted (Linux) runner; built on the Agent SDK.

- **Triggering.** Two modes are auto-detected. *Interactive* waits for `@claude`; *automation* runs
  whenever the workflow provides a `prompt`, "subject only to the checks on who can trigger runs", on any
  GitHub event including `issues: [opened, labeled, assigned]`, `schedule` and `workflow_dispatch`. Inputs
  `label_trigger` and `assignee_trigger` exist for issue events. Shipped examples include
  `issue-triage.yml` and `ci-failure-auto-fix.yml` (the latter pushes `claude-auto-fix-ci-*` branches, i.e.
  it overlaps our `ci` kind). Trigger checks: the actor must have write access and must be human
  ("rejects a bot actor unless you list it in `allowed_bots`").
- **Writes / PRs.** In its default configuration "Claude does not create pull requests automatically
  ... Claude provides a link to the GitHub PR creation page ... The user must click the link and create
  the PR themselves" (security.md). In automation mode it can open PRs only if given a tool that can,
  e.g. `--allowedTools "Bash(gh pr create --draft:*)"`. It "cannot submit formal GitHub PR reviews",
  "cannot approve pull requests", and "Cannot merge branches, rebase, or perform other git operations
  beyond pushing commits" (capabilities doc). Branches get `branch_prefix` (default `claude/`).
- **Tool restriction.** "For a plain-text prompt, Claude has no shell or GitHub API access until you
  grant the tools the prompt needs, with `--allowedTools` in `claude_args` or a `permissions.allow` rule
  in the `settings` input." "By default, results appear in the workflow run log rather than a comment.
  Claude can post to the issue or pull request when the prompt directs it to and it has a tool that can
  post." So *not granting* comment tools is a structural "never comment" in automation mode; interactive
  mode always posts a tracking comment. The allowlist is Claude Code's permission-rule syntax (prefix
  matching), weaker than our `classify_shell` parser but still enforced outside the prompt.
- **Permissions.** The shared Claude GitHub App asks for Actions/Checks/Contents/Discussions/Issues/
  Pull requests/Repository hooks/Workflows read+write; "GitHub doesn't let you accept a subset". A custom
  GitHub App with only Contents/Issues/Pull requests is the documented way to narrow it. The job token
  is "short-lived ... scoped specifically to the repository".
- **Auth / cost.** `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` ("available on Pro, Max, Team, and
  Enterprise plans ... `claude setup-token`"); "If you authenticate with an OAuth token, runs use your
  Claude subscription instead of API billing." This is the one external option where subscription OAuth
  is *documented as supported*. Second cost line: GitHub Actions minutes, 2,000/month free for private
  repos on GitHub Free (3,000 on Pro), public repos free, then $0.006/min Linux
  ([GitHub billing](https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-github-actions/about-billing-for-github-actions)).
- **Gaps against the checklist.**
  - R1: the issue lives in Linear, not GitHub. The factory would have to fire the workflow itself
    (`gh workflow run` / `repository_dispatch`), which is a new outward-write kind for `effects.py`, or
    rely on a Linear->GitHub mirror that does not exist.
  - R3: the job runs on a GitHub-hosted Linux runner, not the user's machine. A self-hosted runner
    on that machine would in principle satisfy R3, but the action's support for self-hosted Windows
    runners was not verified in this timebox (see "Revisit triggers").
  - R4/R5: no evidence section, no readiness score, no Playwright capture path; the PR body is whatever
    the prompt produces.
  - R6: GitHub's `labeled` event is idempotent per label application, but attempt caps and
    handled-at-version would live in the factory anyway.
  - R7: workflow files must be committed per repo (or a reusable workflow); fine for personal repos,
    policy-dependent at an employer.
  - Each repo needs the `.github/workflows/*.yml` checked in; these files match our default
    `sensitive_globs` (`.github/workflows/**`) for a reason.

### O3. Claude Code routines (cloud) and Desktop local scheduled tasks

**O3a. Routines** ([docs](https://code.claude.com/docs/en/routines)): "a saved Claude Code
configuration: a prompt, one or more repositories, and a set of connectors", run "on Anthropic-managed
cloud infrastructure". Research preview; "available on Pro, Max, Team, and Enterprise plans".

- **Triggers.** Schedule ("The minimum interval is one hour"), API (`POST .../routines/<id>/fire` with a
  bearer token under the `experimental-cc-routine-2026-04-01` beta header; the optional `text` arrives
  "wrapped in a `<routine-fire-payload>` block that labels it as untrusted data"), and GitHub events,
  which are limited to two categories: "Pull request" and "Release". There is **no issue event and no
  Linear trigger**; Linear is reachable only as an MCP connector the prompt polls.
- **Writes / permissions.** "Routines run autonomously as full Claude Code cloud sessions: there is no
  permission-mode picker". "Claude can use every tool from an included connector, including writes,
  without asking for permission during a run." "Anything a routine does through your connected GitHub
  identity or connectors appears as you: commits and pull requests carry your GitHub user, and ...
  Linear tickets ... use your linked accounts". Pushes go to `claude/`-prefixed branches ("always
  accepted"); pushes elsewhere are refused if the branch is protected, has someone else's open PR, or
  carries others' commits. The docs' own example routine "opens a draft pull request", so drafts are
  achievable by prompt, not by policy. The only structural limits are *which connectors are attached*
  and the environment's network allowlist; there is no tool allowlist and no "never comment" switch.
- **Cost.** "Routines draw down subscription usage the same way interactive sessions do"; over the
  limit, "organizations with usage credits turned on can keep running routines on metered overage".
  Cloud sessions: "There is no separate compute charge for the cloud VM"
  ([cloud sessions](https://code.claude.com/docs/en/claude-code-on-the-web)). Hourly caps: 30 API fires
  per routine per hour, 100 per account.
- **Gaps.** R1 partially (the factory could `POST /fire` per issue, which is deterministic, but that is
  another new outward-write kind and the beta header may change). R2 no: writes are prompt-level, and the
  Linear connector would give the worker exactly the comment/edit capability invariant 1 forbids. R3 no
  ("Access to local files: No (fresh clone)", Linux VM). R4/R5 no. R7 no for the employer case (routines
  belong to the claude.ai account; `employer` profiles cannot use the subscription). R9: this is a native
  Anthropic surface, so subscription use is unambiguously allowed.
- Adjacent: cloud **auto-fix** for PRs reacts to CI failures and review comments but "Claude may reply
  to review comment threads on GitHub as part of resolving them ... posted using your GitHub account",
  and "GitHub does not emit a webhook when the base branch advances and creates a merge conflict, so
  auto-fix can't react to conflicts". It overlaps `ci`/`rc`, violates "never comment", and misses
  `conflict`; not a substitute for upkeep either.

**O3b. Desktop local scheduled tasks** ([docs](https://code.claude.com/docs/en/desktop-scheduled-tasks)):
"A local task runs on your machine with direct access to your files and tools, but only fires while the
app is open and your computer is awake." Minimum interval 1 minute; per-task permission mode; optional
isolated worktree; "Claude can edit files, run commands, create commits, and open pull requests". This
satisfies R3 and R8 (R9 clean, it is Claude Code itself) but fails R1 and R2 in the way an earlier
script-based PR routine showed: the LLM would be the dispatcher (reading Linear via a connector and
deciding what to do), and write limits are `settings.json` allow rules plus "always allow" approvals
rather than a deny-by-default policy with a typed hand-off to an effects layer. No evidence/readiness,
no dedupe state, no profiles.

### O4. Linear agent integrations

Linear's agent model ([developers/agents](https://linear.app/developers/agents),
[integrations/agents](https://linear.app/integrations/agents)): an app with `app:assignable` /
`app:mentionable` scopes is activated by **delegation** ("when a user assigns an issue to your agent")
or **@mention**; an `AgentSession` tracks the task; the agent "must emit a `thought` activity within 10
seconds to acknowledge the session has begun" and "Session state is visible to users, and updated
automatically based on the agent's emitted activities". Agents "do not count as billable users"; the
coding work is billed by the agent vendor. The listed agents include Codex, Cursor, GitHub Copilot,
Factory, Devin and Cyrus. **No Anthropic-published Linear agent is listed.** The agent API has no label
trigger; vendors that offer one implement it themselves (Devin: "apply the *devin* label").

| Agent | Where it runs | Trigger | PR behaviour (as documented) | In-issue chatter | Cost (vendor) |
|---|---|---|---|---|---|
| Cursor cloud agents ([docs](https://cursor.com/docs/integrations/linear)) | Cursor cloud | delegate / `@Cursor` / Linear triage rules | "create PRs automatically"; draft status not documented | "show real-time status in Linear"; no documented off switch | Cursor plan + usage-based pricing |
| GitHub Copilot coding agent ([docs](https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-coding-agent)) | "a GitHub Actions-powered environment", 59-minute cap, single repo | assign `Copilot`; from "Azure Boards, JIRA, or Linear" | opens a PR on a branch; draft status not confirmed by the page fetched | commits + session logs on GitHub | "all paid Copilot plans", AI credits + Actions minutes |
| Devin ([Linear page](https://linear.app/integrations/devin)) | Devin cloud | assign / `@devin` / `devin` label | does **not** open a PR from the integration; "click the link in Devin's response to draft a PR in a Devin session" | "Devin comments on the issue" with plan and questions | Core $20/mo + ACUs (~$2.25 each) per third-party round-ups |
| Cyrus ([repo](https://github.com/ceedaragents/cyrus), [Hookdeck guide](https://hookdeck.com/webhooks/platforms/how-to-run-claude-code-as-a-linear-agent-with-cyrus-and-hookdeck-cli)) | **your machine** (worktree per issue, webhook via tunnel) | assign / mention | "opens a pull request" | "streams detailed agent activity updates back to Linear" | Apache-2.0, BYOK (API key; guide covers macOS/Linux, Windows not mentioned) |

- **Pros.** Zero build effort; mature vendor harnesses; Cursor/Copilot run in parallel in the cloud;
  Linear-native UX (assignee dropdown, status in the issue).
- **Cons against the checklist.** R2 is failed *by design*: the Linear agent protocol requires the agent
  to post activities into the issue, and every listed vendor comments or streams status; none documents a
  way to suppress it. Drafts-only and never-merge are prompt/vendor conventions, not something the
  factory controls. R3 fails for all cloud agents; Cyrus is local but Linux/macOS-oriented and we did not
  verify Windows. R4/R5 absent. R8 fails: every option except Cyrus is a second subscription; Cyrus runs
  Claude Code under the hood and therefore inherits the same R9 gray zone as O1, closer to the prohibited
  "third-party product" line because it is someone else's product. R1 is satisfiable (the factory could
  set the delegate deterministically), R6 only via factory state.

### O5. Jira / Atlassian (for an employer profile on another tracker)

Brief, because v1 is Linear-only; Jira stands in here for the trackers an employer profile is likely
to use. Atlassian's "agents in Jira" lets you "Assign work to agents, just like you always did to a
teammate" and "@mention agents in comments"; agents respond "right there, in
context" on the work item (open beta announced 2026-02-25,
[blog](https://www.atlassian.com/blog/rovo/ai-agents-in-jira); third-party coverage reports GA at Team
'26 in May 2026). The coding agent is Rovo Dev / "Jira Coding Agent": "assign a work item to Jira Coding
Agent", choose a repository, optionally "Generate code in draft pull request"; requires "Bitbucket Cloud
or GitHub Cloud is connected to your Jira site" and "You have Rovo credits available"
([Atlassian support](https://support.atlassian.com/rovo/docs/generate-code-from-a-work-item-in-jira/)).
Pricing per Atlassian's credits docs and round-ups: Rovo credits bundled with Standard/Premium/Enterprise
seats (25/70/150 per user per month), Rovo Dev as a separate ~$20/developer/month subscription, overage
billing at $0.01/credit from 2026-12-03 ([usage limits](https://support.atlassian.com/rovo/docs/rovo-usage-limits/)).

Same shape as O4: deterministic assignment is possible, drafts are an option, but it comments in the
work item by design, runs in Atlassian's cloud, has no local verification, no evidence section, and is
paid for by the employer's Atlassian plan rather than anything we control. Whether an employer even
enables it is a policy question. The `Tracker` protocol is the right seam: when an employer profile on
Jira exists, Jira agents are evaluated as a possible *external builder adapter target*, not as a
replacement for the engine, and the rollout ladder (`observe` -> upkeep -> `build`) applies regardless.

## Scorecard

"yes" = met structurally; "partial" = achievable but by prompt, vendor convention, or with factory-side
work; "no" = not achievable with what is documented today.

| Req | O1 in-house | O2 GH Action | O3a routines | O3b Desktop tasks | O4 Linear agents | O5 Jira agents |
|---|---|---|---|---|---|---|
| R1 deterministic dispatch | yes | partial (factory fires workflow; new effect) | partial (factory `POST /fire`; beta) | no (LLM polls) | partial (factory sets delegate) | partial |
| R2 structural write limits | yes | partial (tool allowlist; custom GitHub App needed) | no | no | no (protocol requires in-issue activity) | no |
| R3 local verification | yes | no (Linux runner; self-hosted unverified) | no | yes | no (Cyrus: maybe, unverified) | no |
| R4 evidence section | yes | no | no | no | no | no |
| R5 readiness warnings | yes | no | no | no | no | no |
| R6 dedupe / attempt caps | yes | partial (factory state) | partial | no | partial | partial |
| R7 profiles / portability | yes | partial | no (account-bound) | no | partial | n/a (employer only) |
| R8 subscription cost | yes (if R9 ok) | yes (+ Actions minutes) | yes | yes | no (second bill; Cyrus BYOK) | no |
| R9 auth explicitly allowed | **unclear** | yes (documented) | yes | yes | n/a / unclear (Cyrus) | n/a |

## Decision

1. **The in-house `build` worker stays the primary path for v1.** No external option meets R2, R3, R4
   or R5, and those are the requirements that make the factory trustworthy enough to run unattended:
   deny-by-default writes with a typed hand-off, local verification on the machine where the repos
   live, and a PR body that proves what was checked. Buying would save the worker prompt and the
   `effects.push/open_draft_pr` code but would reintroduce the exact failure modes an earlier
   script-based PR routine showed (prompt-level "never post" rules, LLM-driven dispatch, no
   version/attempt state).

2. **Record the auth gray zone instead of papering over it.** The `personal` profile's `oauth` provider
   is the plan's decision and remains the default, but the legal page's wording is not a clean yes for
   SDK-driven unattended use. The owner confirms (or declines) before `build` goes live; `api_key`
   is the sanctioned fallback and is one config line away. Employer profiles already cannot use OAuth.
   This ADR does not change `config.py`.

3. **The attractive external option is the Claude Code GitHub Action, and only as a follow-up
   adapter, not a replacement.** It is the one external path that (a) runs the same model family,
   (b) documents subscription OAuth as supported, (c) can be made comment-free and draft-only by
   *withholding* tools rather than by asking nicely, and (d) can be fired deterministically. It still
   cannot satisfy R3-R5 on its own. If it is ever adopted, it is as the plan's "dispatch by label +
   include its PRs in upkeep scans" shape:
   - a profile setting selects an external backend for `build` (`build.backend = "external:<name>"`),
     per repo if needed;
   - one new effect per backend performs the dispatch deterministically from the scan loop (GitHub
     Action: `gh workflow run`/`repository_dispatch` with the issue key; routines: `POST /fire`; Linear
     agents: set the delegate). Each is an *additional* allowed outward write that must be added to
     invariant 2 in `docs/architecture.md` at that time, with the same audit record as every effect;
   - the factory records the attempt under `(issue, build, version)` so R6 holds, and the existing
     "no open linked PR" rule stops re-dispatch;
   - the upkeep scan widens `my_open_prs` to PRs opened by configured agent identities or branch
     prefixes (`claude/`, `copilot/`, ...) that link to a factory-dispatched issue, so `ci`, `conflict`
     and `rc` cover them;
   - the factory registers those PRs as factory PRs, runs the repo's verification commands locally in a
     worktree at the PR head, and stamps the evidence section itself (R3-R5 recovered factory-side,
     which is the real value of the adapter: an external agent writes the code, the user's machine
     proves it).
   None of this is scheduled; it is backlog gated by the metrics from build-order step 4 (if the in-house
   `build` kind's merge rate / rework rate is poor after real use, the adapter is the experiment to run).

4. **Routines, Desktop tasks, Linear vendor agents and Jira agents are not adopted** for the build
   path. Routines are worth watching (native, subscription-billed, API-triggerable) and become
   interesting the day they gain issue/Linear triggers and a per-run tool allowlist. Linear vendor agents
   and Jira agents are structurally incompatible with "never comment" and are second bills; they remain
   candidates only as adapter targets at an employer that already pays for them.

## Consequences

- **Positive.** Step 6 proceeds as planned with no redesign; `BuildOutput`, `effects.open_draft_pr`,
  evidence and readiness stay exactly as contracted. The `rv`-style "write locally, never post" rule
  stays absolute. No new secrets, GitHub App installs, or per-repo workflow files are required for v1.
- **Negative.** We own the whole build path, including the parts a vendor would improve for free
  (harness quality, parallel cloud compute). On `oauth` the subscription is the only budget, so
  Opus-everywhere with `max_parallel = 3` will hit the subscription's 5-hour / weekly usage windows
  (e.g. the [Max plan](https://support.claude.com/en/articles/11049741-what-is-the-max-plan));
  `fallback_model` and `max_parallel` remain the knobs the plan already names.
- **Neutral / deferred.** The adapter design above adds up to three new outward-write kinds if ever
  built; they are not in `docs/architecture.md` and must not be implemented without updating invariant
  2 and the policy tests. The R9 question is owned by the person, not the code.
- **Contract impact now:** none. No module signature changes; nothing for other agents to import.

## Revisit triggers

Re-open this ADR when any of the following is true:

1. Anthropic clarifies (either way) whether individual, self-operated Agent SDK use on Pro/Max OAuth is
   permitted; or an `api_key` month costs materially more than the subscription for the same work.
2. Claude Code routines add an issue or Linear trigger, a per-run tool allowlist / permission mode, or a
   way to attach connectors read-only.
3. The Claude Code GitHub Action documents self-hosted Windows runners (or we verify them in a sandbox
   repo), which would satisfy R3 and make the adapter cheap.
4. An Anthropic-published Linear agent appears in Linear's agent directory, or Linear adds label-based
   agent triggers and a "no activity" mode.
5. Step 4 metrics after four or more weeks of real `build` use show merge rate below, or rework rate
   above, what we would accept from a contractor; at that point run the adapter experiment with the
   GitHub Action on one sandbox repo before changing anything else.
6. An employer profile is created: evaluate that employer's actual tracker / GitHub agent
   entitlements against this scorecard before enabling `build` there.

## Sources

Primary vendor documentation fetched 2026-10-01:

- Claude Code GitHub Actions: https://code.claude.com/docs/en/github-actions ;
  https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md ;
  https://github.com/anthropics/claude-code-action/blob/main/docs/capabilities-and-limitations.md ;
  https://github.com/anthropics/claude-code-action/blob/main/docs/security.md ;
  examples directory (issue-triage.yml, ci-failure-auto-fix.yml)
- Claude Code legal and compliance: https://code.claude.com/docs/en/legal-and-compliance
- Agent SDK overview: https://code.claude.com/docs/en/agent-sdk/overview
- Routines: https://code.claude.com/docs/en/routines
- Cloud sessions and auto-fix: https://code.claude.com/docs/en/claude-code-on-the-web
- Desktop scheduled tasks: https://code.claude.com/docs/en/desktop-scheduled-tasks
- Max plan: https://support.claude.com/en/articles/11049741-what-is-the-max-plan
- API pricing: https://claude.com/platform/api
- GitHub Actions billing: https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-github-actions/about-billing-for-github-actions
- Linear agents: https://linear.app/developers/agents ; https://linear.app/integrations/agents ;
  https://linear.app/integrations/devin
- Cursor Linear integration: https://cursor.com/docs/integrations/linear
- GitHub Copilot coding agent: https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-coding-agent
- Cyrus: https://github.com/ceedaragents/cyrus ;
  https://hookdeck.com/webhooks/platforms/how-to-run-claude-code-as-a-linear-agent-with-cyrus-and-hookdeck-cli
- Atlassian: https://www.atlassian.com/blog/rovo/ai-agents-in-jira ;
  https://support.atlassian.com/rovo/docs/generate-code-from-a-work-item-in-jira/ ;
  https://support.atlassian.com/rovo/docs/rovo-usage-limits/

Secondary (commentary / pricing round-ups, used only where marked):

- https://alternativeto.net/news/2026/2/anthropic-officially-bans-using-subscription-authentication-for-third-party-claude-use
- Devin pricing round-ups surfaced by search (e.g. https://aicodingdir.com/tools/devin/)
- Atlassian agents GA timing and Rovo pricing round-ups surfaced by search
