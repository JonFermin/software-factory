# software-factory

A local, scheduled software factory. **Linear is the source of truth**: label an issue and the
factory refines it, breaks it into sub-issues, builds it into a draft pull request, and then keeps
that PR healthy (fixes failing CI, resolves merge conflicts, answers review feedback, drafts the
reviews you were asked for). Everything runs on your own machine from an OS scheduler
(`factory tick` every 20 minutes), with one narrow Claude Agent SDK worker per action and a
deterministic Python engine around it.

The design generalises the "unattended PR routine" pattern: read external state (not notifications),
one worker per action, dedupe by `(target, kind, version)`, push only when earned, flag a human when
not. The implementation contract lives in [`docs/architecture.md`](docs/architecture.md); the
build-vs-buy check for the issue -> PR path is [`docs/adr/0001-build-vs-buy.md`](docs/adr/0001-build-vs-buy.md).

## Core principle: workers propose, the engine acts

Workers (Claude sessions) get **no outward-write capability**. `git push`, every `gh` write, `gh api`
with a non-GET method or the `graphql` endpoint, Linear/Slack HTTP, package publishing and all
`mcp__*` tools are denied by `policy.py` in both the SDK `can_use_tool` callback and a `PreToolUse`
hook. A worker edits, tests and commits inside a throwaway git worktree and returns a typed result
(`push_ok`, reasons, summary, dispositions, sub-issue specs...). The engine (`effects.py`) applies
deterministic policy and performs every outward write itself. Denials return a message, so the model
adapts instead of hanging on a permission prompt.

The tool policy is not the only wall. Below it, each worker:

- loads **no filesystem settings** (`setting_sources=[]`): the repo's `.claude/settings.json` (its
  hooks, `apiKeyHelper`, `env`, plugins, allow rules) is never applied, because a PR head controls
  that file. The repo's `CLAUDE.md` / `AGENTS.md` are still followed: the engine appends them to the
  worker's system prompt.
- has **no `gh` login**: `GH_CONFIG_DIR` points at an empty file (your `hosts.yml` / keyring login is
  unreachable) and `GH_TOKEN` is the optional read-only `GH_WORKER_TOKEN` from the profile `.env`, or
  empty. Without that token workers run `gh` unauthenticated by design; the engine already put the
  PR, CI and review context in the prompt.
- runs git with **every GitHub push URL rewritten** to a transport that does not exist (a scripted
  push fails too) and with **git hooks disabled**: `core.hooksPath` points at a path *under a
  regular file* (`%SystemRoot%\System32\drivers\etc\hosts\no-hooks` on Windows,
  `/dev/null/no-hooks` elsewhere), where no process can ever create a hook. The repo's `pre-commit`
  / `commit-msg` hooks do not run for worker commits; the worker prompt tells workers to run the
  repo's lint / format commands themselves when its `CLAUDE.md` asks for them, and a
  `.pre-commit-config.yaml` becomes a verification command (`pre-commit run --all-files
  --show-diff-on-failure`).
- may not set `GIT_*` / `GH_*` / `SSH_*` / editor / pager variables, change git config keys that run
  commands or redirect remotes and credentials, or write into a `.git` directory or a git/gh/ssh
  config or credential file. A change to the clone's shared git config or hooks that slips through
  anyway is reverted after the session, flagged, and the run's output is discarded.
- has its **whole process tree killed** when the session ends (finished, failed or timed out), so a
  dev server, watcher or background loop it started does not outlive it.

The engine's own git (worktree setup, every fetch -- including the evidence branch's and `factory
ready`'s -- and every push) runs with the same unplantable `core.hooksPath`, so a hook planted in the
clone's shared `.git/hooks` never executes with your credentials; when the host cannot provide that
path, git does not run at all. With hooks off, Git LFS's `pre-push` hook cannot upload LFS objects
either: the engine uploads the pushed change's LFS objects itself before pushing the branch, and
refuses the push (keeping the work for you) when it cannot.

## Safety model

Allowed outward writes (all in `effects.py`, each one audited):

- push branches with the refspec `HEAD:refs/heads/<branch>` only -- never `+`, never `--force*`,
  never `--no-verify` (includes the orphan `factory/evidence` branch for screenshots)
- open **draft** pull requests (with a configurable AI-disclosure footer)
- edit the `<!-- factory:evidence -->` section of PRs the factory itself opened
- Linear: state transitions, sub-issue creation (+ blocked-by relations), and -- for `refine` only --
  issue title/description edits plus removal of the `factory:refine` label

Never: comments, replies, reviews (GitHub or Linear), resolving threads, merge, close,
ready-for-review, force-push, label creation.

Invariants the tests enforce:

1. Workers propose, the engine acts (above).
2. Every outward write goes through `effects.py`.
3. `mode = "observe"` performs **zero** outward writes to GitHub or the tracker: only `rv` (review
   drafts, written locally) runs for real; every other kind is recorded as "would run". `--dry-run`
   does the same for a single tick: scan, filter, print, start no worker, write no state or flags
   (and it needs no provider credentials).
4. State is recorded only on worker outcome `done` / `nothing`; anything else is a failed attempt,
   and `max_attempts` (default 3) per `(target, kind, version)` stops retries (flagged once).
5. Auth failures stop the wave: a precheck failure (gh, tracker, provider credentials) flags once and
   exits 2 without spending attempts; a mid-wave auth failure stops launching new actions. A model
   rate / usage limit (or an overloaded provider) stops the tick the same way: exit 2, no attempt
   spent, flagged once per day; the next tick simply retries.
6. Profiles never cross: one process loads one profile; secrets live in `Profile.secrets`, never in
   `os.environ`; state, worktrees and logs live under that profile's data dir.
7. Refine notifies before it edits: the original title/description is persisted and a toast/Slack DM
   delivered *before* the Linear issue is touched. No notification -> no edit.

Push policy on top of the worker's `push_ok`: at least one passing verification run; no changed file
matching the sensitive globs (auth, security, billing, migrations, workflows, per-repo additions); no
changed file that CODEOWNERS assigns to someone else -- evaluated **per changed file** with GitHub's
last-match-wins rule, so `* @other-team` followed by `/docs/ @you` lets a docs change through while
anything else stays refused (owners count as yours when they are your login or a team in
`policy.my_teams`); `rc` never pushes when a comment was `skipped`; `conflict` never pushes with
unresolved or unsafe hunks or leftover markers, and judges only its own conflict resolutions, not
every base-branch file the merge brought in. A refused push keeps the worktree for you and flags it.

## Action kinds

| Kind | Trigger | Worker returns | Engine effect | Version key |
|---|---|---|---|---|
| `ci` | my open PR has >= 1 failing check on its current head | fix commit + classification (caused-by-PR / flaky / inherited / infra; the prompt carries the base branch's latest run of each failing workflow, fetched by the engine) | push if `push_ok` and policy passes; all-flaky with no commit -> handled | head SHA |
| `conflict` | PR `mergeable == CONFLICTING` | merge of `origin/<base>` + per-hunk classification | push if every hunk resolved safely; else keep worktree + flag | `{head}:{base}` SHAs |
| `rc` | new reviewer/bot feedback since my last commit, or unresolved non-outdated threads | commit + per-comment disposition | push only if **no** comment was skipped | latest feedback timestamp |
| `rv` | my review requested directly, non-draft | review markdown | written to `reviews/<repo>-<pr>.md` + notification; **never posted** | head SHA |
| `refine` | issue has `factory:refine` | rewritten title + description (AC list), open questions, suggested split | save original -> notify -> edit issue -> remove the label (`factory revert-refine KEY` restores); nothing to refine -> the label is removed and you are told | issue `updated_at` |
| `plan` | issue has `factory:plan`, no children | sub-issue specs with repo + blocked-by indices | create sub-issues (with `repo:<name>` label when the parent's team can use it, else a description marker; **without** `factory:build` -- you approve by labelling), move parent to started, notify; nothing to plan -> flagged once (remove the plan label, then label `factory:build`) | issue `updated_at` |
| `build` | `factory:build`, unstarted, blockers done, no open linked PR, no pending `factory:refine` / `factory:plan` label, repo resolvable | commits on the issue's branch, tests run, `push_ok` | push branch, open draft PR with `Fixes KEY` + evidence section + footer, move issue to started, notify; nothing to build -> flagged once | issue id (attempt-capped) |
| `ready` | manual: `factory ready <repo>` | commits improving agent readiness (CLAUDE.md verification section, scripts, deflaking) | push `factory/ready-<date>`, draft PR | `origin/<default>` SHA |

Actions on one target run in a fixed order: `ci -> conflict -> rc -> rv` on a PR, `refine -> plan ->
build` on an issue. A build waits, without spending an attempt and with one flag naming the label,
while a refine or plan label is *pending*: its kind is enabled (a disabled kind's label is ignored),
and for plan only while the issue has no sub-issues and plan has not said "nothing to plan" for
the issue as it is now. Different targets run in parallel up to `max_parallel`.

Repo inference for issues (`resolve.py`): `repo:<name>` label or planner marker -> the repo the
planner chose when it created the sub-issue -> GitHub attachments -> majority vote of sibling issues
-> repo names / paths in the text -> a model pick over candidates. Below `resolve.min_confidence`
(0.7) or multi-repo -> flag and skip.

Every factory-opened PR body carries a machine-maintained evidence section: changed files and
sensitive-adjacent paths (next to a sensitive-glob match or inside another team's CODEOWNERS area;
informational, the push policy decides per file), acceptance-criteria -> test mapping, verification commands run, screenshots
for `visual = true` repos (pushed to `factory/evidence`), and a readiness warning when the repo scores
below `policy.min_readiness`.

## Install

Prerequisites: Python 3.12+ (the repo pins 3.13 in `.python-version`), [uv](https://docs.astral.sh/uv/),
git, the GitHub CLI `gh` (logged in: `gh auth login`), and the Claude Code CLI. On Windows the Agent
SDK needs the native `claude.exe` (the installer from claude.ai); it refuses npm's `claude.cmd` shim.
The CLI must be at least the version the SDK was built against (Windows SDK wheels bundle none, so
the one on `PATH` is used): an older CLI rejects the configured model and every worker dies at
startup. `factory doctor` compares the two; `claude update` fixes it, or `provider.cli_path` points
the SDK at a newer `claude.exe` that is not on `PATH`.

```
git clone https://github.com/JonFermin/software-factory.git
cd software-factory
uv sync
uv run factory --help
```

`uv run factory ...` is the only entry point (`factory` is the console script; `python -m
software_factory` works too). Nothing is installed globally.

## Profile setup

A profile is a directory `<profiles_dir>/<name>/` with `factory.toml` (config) and `.env` (secrets,
never committed). `<profiles_dir>` is `--profiles-dir`, else `$SOFTWARE_FACTORY_PROFILES_DIR`, else
the per-user config dir `<config root>/software-factory/profiles` when it exists (`<config root>` is
`~/Library/Application Support` on macOS, else `$XDG_CONFIG_HOME` or `~/.config` on Linux **and
Windows** (not `%APPDATA%`: packaged apps such as the Claude desktop app virtualize AppData, so scheduled ticks would not see the same files), else `<repo>/profiles`. Real profiles live outside this repository: `<repo>/profiles`
only ships its `README.md` and [`profiles/example/factory.toml`](profiles/example/factory.toml), and
everything else under it is gitignored. The default profile name is `personal`; create it by
copying the example:

```
# Linux and Windows (Git Bash: ~ is %USERPROFILE%); on macOS use
# ~/Library/Application Support/software-factory/profiles/personal
mkdir -p ~/.config/software-factory/profiles/personal
cp profiles/example/factory.toml ~/.config/software-factory/profiles/personal/factory.toml
# edit github.owners and repos.root, add the .env next to it, then:
uv run factory doctor
```

The example manages the clones under `repos.root` (`~/code`) whose GitHub owner is listed in
`github.owners`, and starts with the upkeep kinds. Every key is annotated in [`factory.example.toml`](factory.example.toml);
the full profile story (isolation, employer profiles, rollout) is in
[`profiles/README.md`](profiles/README.md).

`.env` keys (`KEY=value` per line; never exported to the environment):

| Key | Needed when |
|---|---|
| `CLAUDE_CODE_OAUTH_TOKEN` | `provider.kind = "oauth"` (from `claude setup-token`; personal profiles only) |
| `ANTHROPIC_API_KEY` | `provider.kind = "api_key"` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` | `provider.kind = "bedrock"` without ambient AWS credentials |
| `GOOGLE_APPLICATION_CREDENTIALS` | `provider.kind = "vertex"` without Application Default Credentials |
| `LINEAR_API_KEY` | `linear.enabled = true` and an issue kind (`refine`/`plan`/`build`) is enabled |
| `SLACK_BOT_TOKEN`, `SLACK_USER_ID` | `notify.slack = true` |
| `GH_WORKER_TOKEN` | optional: a fine-grained **read-only** GitHub PAT (repository permissions contents, pull requests and actions: read) that workers get as `GH_TOKEN`. Without it workers have no `gh` authentication at all, by design: your own `gh` login is never reachable from a worker, and the engine puts the PR / CI / review context in the prompt. `doctor` warns when it is absent |

Key `factory.toml` settings: `mode` (`observe` | `full`), `enabled_kinds` (the rollout ladder),
`max_parallel`, `max_attempts`, `session_timeout_min`, `[provider]` (kind, model, effort,
`fallback_model`, `cli_path`), `[github] owners`, `[repos] root/include/exclude` plus per-repo
`[repos.overrides.<name>]` (`visual`, `dev_command`, `dev_port`, `verify`, `setup`,
`sensitive_globs`, `min_readiness`, `default_branch`), `[linear]` labels, `[policy]` sensitive globs
and `min_readiness`, `[notify]`, `[schedule]`.

## Prerequisites you do by hand

The factory never creates external resources, so these are yours:

1. **Claude credentials.** For `provider.kind = "oauth"` run `claude setup-token` and paste the token
   into `.env` as `CLAUDE_CODE_OAUTH_TOKEN`. Caveat recorded in the ADR: Anthropic's terms allow
   individual Agent SDK use on a subscription but steer products that act for third parties to API
   keys; a single-user tool on your own machine is not the prohibited case, but it is not explicitly
   carved out either. Confirm the current terms allow unattended use on your plan; switching to
   `api_key` / `bedrock` / `vertex` is one config line (employer profiles reject `oauth` outright).
2. **Linear.** A personal API key (`LINEAR_API_KEY`), Linear's GitHub integration enabled (so branches
   named with the issue's `branchName` auto-link and the factory never opens a second PR), and the
   labels `factory:refine`, `factory:plan`, `factory:build` created in Linear. Optional `repo:<name>`
   labels pin an issue to a repo and let the planner label sub-issues. `factory doctor` reports
   missing labels; it cannot create them (label creation is not an allowed write).
3. **Slack** (for DMs to yourself, i.e. phone push). A Slack app with the `chat:write` and `im:write`
   bot scopes installed to your workspace; put its bot token in `SLACK_BOT_TOKEN` and your member id in
   `SLACK_USER_ID`. Desktop toasts work without any setup (`desktop-notifier`).
4. **Repos.** Local clones under `repos.root` whose `origin` is a GitHub repo owned by one of
   `github.owners` (or listed in `repos.include`). Workers run in linked worktrees of those clones,
   always on a detached HEAD: your checkouts are never modified and no local branch is created,
   reset or deleted in your clone (the engine pushes `HEAD:refs/heads/<branch>` straight from the
   worktree).
5. **Optional: a read-only worker token.** Create a fine-grained GitHub PAT limited to the managed
   repos with read-only contents, pull requests and actions, and put it in `.env` as
   `GH_WORKER_TOKEN` if you want workers to be able to run `gh pr view` / `gh run view` themselves.
   Skip it and workers rely on the context the engine fetched (`doctor` shows a `warn` line, which is
   expected).

Then `uv run factory doctor` until it is green (Slack/Linear lines may stay `warn` until you enable
those features, and `worker gh token` stays `warn` without the optional `GH_WORKER_TOKEN`).

## Commands

Global options: `--profile/-p NAME` (default `personal`), `--profiles-dir PATH`.

| Command | What it does | Exit code |
|---|---|---|
| `tick [--dry-run]` | One tick: lock -> precheck -> scan -> filter -> dispatch -> record. `--dry-run` prints the actions and skips (with reasons) and starts nothing; it needs no provider credentials and raises no flags. | 0 ok, 1 an action failed, 2 auth failure or provider rate/usage limit; "tick already running" exits 0 |
| `run <kind> <target> [--dry-run]` | One action by hand from the *current* external state. PR kinds take `owner/name#123`, `name#123` or a PR URL; issue kinds take the key; `ready` takes a repo. Bypasses the dedupe/attempt/enabled-kind filters (the way to retry an exhausted `build`), still respects `mode`, and still refuses a target with a kept worktree until `release`. | 1 failed, 2 bad target / auth failure / provider rate/usage limit |
| `status` | Mode, enabled kinds, lock state, open flags, kept worktrees, factory-opened PRs, last 10 runs. | |
| `flags [--ack ID] [--all]` | Open flags (things waiting for you); `--ack` closes one. | 1 unknown id |
| `doctor [--repo NAME]` | Profile + employer rules, `.env`, provider credentials, Claude CLI (the binary the SDK will launch, incl. `provider.cli_path`, and that it is not older than the SDK's expected CLI), `gh auth` (and that the account `gh` is logged into matches `github.login`), the optional `GH_WORKER_TOKEN` (warn when absent: workers then run `gh` unauthenticated), managed repos, unique `dev_port`s, Linear reachability + factory labels (+ `repo:<name>` labels, warn only; checked per team when `linear.team_keys` is set), Slack `auth.test`, toast backend, data dir writable, scheduler jobs (and that they use this shell's data dir and may run on battery; a scheduler config that cannot be evaluated is a `fail` line, not a crash). `--repo` adds verification discovery and the static readiness checks (runtime checks show as `n/a`). | 1 if any check fails |
| `readiness [--repo NAME] [--no-run] [--force]` | Readiness score per managed repo. Full runs execute the discovered verification commands twice in a throwaway worktree at `origin/<default>` (flakiness, runtime); `--no-run` is static only. Cached per HEAD; informational, never blocks. | |
| `ready <repo> [--dry-run]` | The manual `ready` action: worker improves readiness on `factory/ready-<date>`, engine pushes and opens a draft PR. | as `run` |
| `report [--since 7d] [--digest]` | Refresh outcomes of every factory-touched PR, then: runs per kind, merged / closed / open, median hours to merge, rework rate, reverted pushes, cost, top flag reasons. `--digest` DMs the compact version to you on Slack. | 1 if the digest could not be sent |
| `revert-refine <KEY>` | Restore the original title/description of a refined issue (status `reverted`; the label is not re-added). | 1 no record |
| `schedule install [--interval N] [--no-digest]` / `uninstall` / `show` | OS scheduler jobs for this profile (below). | 1 scheduler error |
| `cleanup [--all] [--yes]` | Remove leftover worktrees that are not kept for you; `--all` also releases + removes the kept ones after confirmation. Refuses while a tick runs. | |
| `release <target> [--remove]` | Release a kept worktree so the target is dispatched again; `--remove` deletes the directory (otherwise it stays until `cleanup`; being a detached HEAD it does not block the next run). | 1 unknown target |

Output is plain ASCII; stdout is switched to UTF-8 (with replacement) so a cp1252 Windows console
never breaks a scheduled run.

## Scheduling

`uv run factory schedule install` registers two jobs for the profile with the OS scheduler, running
`<uv> run --project <repo> factory --profile <name> ...` (with `SOFTWARE_FACTORY_PROFILES_DIR` set
to the profile's directory whenever it lives outside `<repo>/profiles`, the user config dir included, and `SOFTWARE_FACTORY_DATA_DIR` set when it was
set in the shell that ran `install` -- scheduled jobs inherit no shell environment, and without it
the scheduled ticks would use a different `state.db`, lock and logs than your CLI):

- `tick` every `schedule.interval_minutes` (default 20)
- `report --digest --since 7d` weekly at `schedule.digest_weekday` / `digest_time`

Backends: Windows Task Scheduler (`schtasks`, each job a generated `.cmd` wrapper under the profile's
`scripts/` dir because `/TR` is limited to 261 characters; stdout/stderr go to `logs/scheduler.log`;
tasks run only while you are logged on, which is what lets toasts appear; the tasks are allowed to
start and keep running on battery, which `schtasks` alone would not permit), launchd on macOS
(`~/Library/LaunchAgents/com.software-factory.<profile>.<job>.plist`), cron on Linux (a block
between `# software-factory:<profile> begin/end` markers). `schedule show` prints what the OS
reports (on Windows also the env the wrapper sets); `schedule uninstall` removes both jobs and the
wrappers. Re-run `install` after changing `SOFTWARE_FACTORY_DATA_DIR` or the profile location;
`doctor` warns when the installed jobs disagree with the current shell. Ticks never overlap: a
global file lock makes a second tick exit immediately.

## Rollout ladder

Turn kinds on by config, not code, one rung at a time, and let `factory report` and the audit log
justify the next rung:

1. `mode = "observe"` -- only `rv` runs for real (local review drafts); every other kind is
   recorded as "would run" with zero outward writes. This is where an employer profile starts.
2. `mode = "full"`, `enabled_kinds = ["ci", "conflict"]` -- upkeep pushes on your own open PRs.
3. add `"rc"` -- review-feedback commits (never when a comment was skipped).
4. add `"build"` -- labelled issues become draft PRs.
5. add `"plan"` (and `"refine"`) -- the factory decomposes and rewrites issues.

The example profile (`profiles/example/factory.toml`) starts at rung 2-3 (`ci`, `conflict`, `rc`,
`rv`); give it about a week of upkeep metrics before adding `build`. Setting `enabled_kinds` to include an
issue kind is what makes `LINEAR_API_KEY` mandatory at precheck; until then a missing key only
skips the tracker.

## Runtime data

Everything mutable lives outside the repo, per profile, under
`<data_root>/software-factory/<profile>/` where `data_root` is
`~/Library/Application Support` (macOS) or `$XDG_DATA_HOME` / `~/.local/share` (Linux and Windows; not `%APPDATA%`: packaged apps such as the Claude desktop app virtualize AppData, so scheduled ticks would not see the same files);
`SOFTWARE_FACTORY_DATA_DIR` overrides the base (`schedule install` embeds it in the jobs so
scheduled ticks and the CLI share one dir):

```
state.db              SQLite: handled, attempts, kept worktrees, flags, factory PRs, pushes,
                      resolutions, refinements, runs, PR outcomes, caches
wt/                   throwaway git worktrees (short name: Windows MAX_PATH)
logs/audit.jsonl      every effect, notification, flag and tick boundary
logs/runs.jsonl       one record per action run (kind, target, version, outcome, cost, duration)
logs/transcripts/     one JSONL transcript per worker session (secrets redacted)
logs/scheduler.log    stdout/stderr of scheduled runs
reviews/              rv drafts: <repo-name>-<pr>.md
refinements/          refine records: <KEY>.md (original, new, diff, open questions)
scripts/              generated scheduler wrappers
scratch/              cwd for workers that have no worktree (refine without a repo, plan, resolver)
gh-noauth             empty file: the workers' GH_CONFIG_DIR (so gh never finds your login)
tick.lock             the global tick lock
```

## Portability and employer profiles

The same checkout runs an employer profile next to a personal one without crossing streams:
separate config, secrets, repo set, tracker, state, worktrees and logs; a tick only ever sees one
profile. Employer profiles (`employer = true`) must use `api_key`, `bedrock` or `vertex` (never the
subscription token), require a disclosure footer, default to `mode = "observe"`, and live outside
any git checkout like every real profile (in the user config dir, or a dedicated directory such as
`SOFTWARE_FACTORY_PROFILES_DIR=~/factory-profiles`; `doctor` warns when an employer profile sits
inside a git checkout). Workers only ever receive the one credential their
provider needs (`providers.worker_env`); inherited `ANTHROPIC_*` / `CLAUDE_CODE_*` / tracker / chat
variables are scrubbed from the process before any worker starts.

## Troubleshooting

- **`doctor` says `provider credentials: CLAUDE_CODE_OAUTH_TOKEN missing`** -- create
  `<profiles_dir>/<name>/.env` next to the profile's `factory.toml` with the token from
  `claude setup-token` (or switch `provider.kind`).
- **`profile 'personal' not found: <path> does not exist`** -- no profile at the path the message
  names; copy `profiles/example/factory.toml` there (see "Profile setup"), or point
  `--profiles-dir` / `SOFTWARE_FACTORY_PROFILES_DIR` at the directory that holds your profiles.
- **`gh auth: ... gh auth login`** -- run `gh auth login`; the engine uses your `gh` session for
  every GitHub read and write (workers never do).
- **`worker gh token: GH_WORKER_TOKEN not in the profile .env`** -- expected unless you want workers
  to run `gh` reads themselves: without the token they run `gh` unauthenticated by design and use
  the PR / CI / review context the engine put in the prompt. To enable it, add a fine-grained
  read-only PAT (contents, pull requests, actions: read) as `GH_WORKER_TOKEN`; never put a token
  with write access there.
- **A worker said `gh` is not logged in, or a repo's pre-commit hook did not run** -- by design (see
  "Core principle" above). Workers run with git hooks disabled and without the
  repo's `.claude/settings.json`. A `.pre-commit-config.yaml` is picked up by verification
  discovery (`pre-commit run --all-files --show-diff-on-failure`); checks wired into hooks any other
  way belong in the repo's `CLAUDE.md` (workers run what it asks for) or in a
  `[repos.overrides.<name>] verify` entry.
- **`claude cli: ... script shim`** -- on Windows the SDK refuses `claude.cmd`; install the native
  `claude.exe` so it is found on `PATH` (the doctor shows which binary will be used and its version).
- **`claude cli: ... older than the CLI the SDK expects`** -- the Agent SDK launches that binary and
  it does not know the configured model (workers fail with "does not support this model"). Run
  `claude update`, or set `provider.cli_path` to a newer `claude.exe` (Windows: a native `.exe`,
  never `claude.cmd`); `doctor` reports the binary `cli_path` names, and fails when it is missing.
- **`gh auth: gh is logged in as X but github.login = Y`** -- the profile pins a login that is not
  the account `gh` uses; the scan filters your PRs and your own comments by that login, so fix
  `github.login` (or leave it empty to detect it).
- **`dev ports: dev_port shared across repos.overrides`** -- two visual repos on one port would
  screenshot each other's dev server when built in the same tick; pin unique ports
  (`dev_command = "npm run dev -- --port N --strictPort"`).
- **`scheduler: jobs installed but they do not run from the current data dir / carry
  SOFTWARE_FACTORY_DATA_DIR`** -- the jobs were installed from a shell with a different
  `SOFTWARE_FACTORY_DATA_DIR`, so scheduled ticks and your CLI use different state; re-run
  `factory schedule install` from this shell. **`scheduler power`** (Windows) -- the tasks still
  carry the schtasks default of not starting on battery; re-run `schedule install`.
- **`linear: LINEAR_API_KEY missing`** is only a warning until an issue kind is enabled; with
  `build`/`plan`/`refine` on it is a precheck failure (exit 2, flagged once per day).
- **`tick already running`** -- another tick holds `tick.lock`; it exits 0 on purpose. If a crashed
  process left the lock, it is released automatically (the lock is an OS file lock, not a marker).
- **An action keeps being skipped with `kept worktree awaiting me`** -- a push was refused or a
  conflict left unresolved; the worktree path is in `factory status` / the flag. The kept worktree
  is a **detached HEAD** (the engine never creates local branches): to continue by hand, `cd` into it
  and `git switch -c <branch>-factory` first (the flag names it; the suffix keeps it clear of your own
  local `<branch>`, which may be checked out elsewhere), or your commits are easy to lose at the next
  checkout. For a PR you can instead review the kept commits and `git push origin
  HEAD:refs/heads/<branch>` yourself. Then `factory release <target>`; without `--remove` the directory stays for
  you (until `factory cleanup`) and no longer blocks the next run, with `--remove` it is deleted.
- **`attempts exhausted`** -- `max_attempts` (3) failed attempts at this version. The factory flags it
  once when the cap is hit (with how the last attempt ended), then skips the action silently. What
  re-arms it depends on the kind: PR kinds (`ci`, `conflict`, `rc`, `rv`) re-arm on a new head, base
  or review feedback; `refine` / `plan` re-arm when the issue is edited (new `updated_at`); `build`
  is keyed by the issue id, so editing the issue does **not** re-arm it -- fix the cause, then retry
  with `factory run build KEY`. `factory run <kind> <target>` retries any kind by hand.
- **`provider unavailable` / the tick exits 2 with a rate or usage limit** -- the model provider
  refused the session (rate limit, usage limit, overloaded, billing). The tick stops launching new
  actions, spends no attempt and flags once per day; the next scheduled tick retries the same work
  (`factory run` / `ready` exit 2 the same way). Only a session that produced no result counts: one
  that hit a limit on the way (or in a subagent) but finished with valid output keeps its work.
  Lower `max_parallel`, set `provider.fallback_model`, or wait for the limit window to reset.
- **A push was refused with `changed files owned by another team per CODEOWNERS`** -- CODEOWNERS is
  evaluated per changed file with GitHub's last-match-wins rule; the file's owners are neither your
  login nor a team in `policy.my_teams`. Add the team to `my_teams` if it is yours, or push that
  change yourself.
- **Nothing is pushed in `observe` mode / `--dry-run`** -- by design; look for `would_*` records in
  `logs/audit.jsonl`.
- **Garbled characters in a Windows console** -- the CLI prints ASCII and reconfigures stdout to
  UTF-8; worker output in transcripts is UTF-8. Use `type` / a UTF-8 capable viewer on the logs.
- **A tick ran for hours** -- each worker session has `session_timeout_min` (30); readiness full
  runs have a 120 s runtime budget and a 10 min per-command timeout. `cleanup` removes orphaned
  worktrees once the tick is over.

## Development

```
uv run pytest                   # the whole suite (fakes everywhere: no network, no workers, tmp git repos)
uv run ruff check src tests
uv run factory doctor
```

### Local end-to-end sandbox (real workers, nothing leaves the machine)

```
uv run python scripts/e2e_local.py [--profile NAME] [--cli-path PATH] [--effort medium] [--keep]
```

Runs the real engine — scan, dispatch, handlers, Claude workers, policy, push policy, effects — against a
throwaway sandbox: a local bare repo stands in for GitHub (its "CI" really runs the sandbox tests on each
head), and in-process fakes stand in for the `gh` and Linear APIs. One tick exercises all seven kinds
(`ci`, `rc`, `conflict`, `rv` on four PRs; `build`, `refine`, `plan` on three issues), a second tick must
find nothing to do, then the refinement is reverted; 37 checks cover pushes, CI results, PR body
(Fixes / evidence / footer), notify-before-edit, and that every outward call is on the allowed list. It
spends real model tokens (about $1 at medium effort, ~1 minute). Without `--profile` the workers use
this machine's existing Claude Code login. Use it after changing prompts, policy, runner or effects.

See [`CLAUDE.md`](CLAUDE.md) for the working rules for agents and [`docs/architecture.md`](docs/architecture.md)
for the module contract.
