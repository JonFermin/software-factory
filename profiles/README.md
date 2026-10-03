# Profiles

A profile is everything one `factory` process is allowed to see: config, secrets, repo set, tracker,
state, worktrees and logs. A tick loads exactly one profile (`--profile <name>`, default `personal`)
and nothing is shared between profiles.

```
<profiles_dir>/
  personal/
    factory.toml   # config (start from ./example/factory.toml; ../factory.example.toml annotates every key)
    .env           # secrets (never committed, never exported to os.environ)
```

## Where profiles live

Real profiles live **outside** this repository, so your config and secrets never end up in its git
history. This directory only ships this README and [`example/factory.toml`](example/factory.toml);
everything else under `profiles/` is gitignored.

Resolution order (first hit wins):

1. `--profiles-dir PATH`
2. `$SOFTWARE_FACTORY_PROFILES_DIR`
3. the per-user config dir `<config root>/software-factory/profiles`, when it exists, where
   `<config root>` is `~/Library/Application Support` on macOS and `$XDG_CONFIG_HOME` (default
   `~/.config`) on Linux and Windows (not `%APPDATA%`: packaged apps such as the Claude desktop app
   virtualize AppData, so a scheduled tick would not see a profile written from inside one) (`SOFTWARE_FACTORY_CONFIG_DIR` replaces
   `<config root>`)
4. `<repo>/profiles` (only the example lives there)

## Creating a profile

Copy the example into the user config dir under the profile's name (`personal` is the name
`factory` loads when no `--profile` is given), edit it, and put the secrets next to it:

```
# Linux and Windows (Git Bash: ~ is %USERPROFILE%); on macOS use
# ~/Library/Application Support/software-factory/profiles/personal
mkdir -p ~/.config/software-factory/profiles/personal
cp profiles/example/factory.toml ~/.config/software-factory/profiles/personal/factory.toml
# set github.owners and repos.root, create .env next to it (keys below), then:
uv run factory doctor
```

The profile name is the directory name (`profile.name` may be omitted; when set it must match).
Alternatively keep profiles in a directory of your choice and point `SOFTWARE_FACTORY_PROFILES_DIR`
at it (or pass `--profiles-dir`).

Employer profiles live outside the repo too, ideally in their own directory so work config and
secrets never sit next to personal ones:

```
export SOFTWARE_FACTORY_PROFILES_DIR=~/factory-profiles     # ~/factory-profiles/work/{factory.toml,.env}
factory --profile work doctor
```

`doctor` warns when an `employer = true` profile sits inside a git checkout.

## `.env` keys

| Key | Needed when |
|---|---|
| `CLAUDE_CODE_OAUTH_TOKEN` | `provider.kind = "oauth"` (from `claude setup-token`; subscription token, personal profiles only) |
| `ANTHROPIC_API_KEY` | `provider.kind = "api_key"` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | `provider.kind = "bedrock"` without ambient AWS credentials (an `aws_profile` or instance role works without them) |
| `GOOGLE_APPLICATION_CREDENTIALS` | `provider.kind = "vertex"` without Application Default Credentials |
| `LINEAR_API_KEY` | `linear.enabled = true` **and** an issue kind (`refine` / `plan` / `build`) is in `enabled_kinds`; until then a missing key only skips the tracker (`doctor` warns, `tick` does not fail) |
| `SLACK_BOT_TOKEN`, `SLACK_USER_ID` | `notify.slack = true` (bot needs `chat:write` + `im:write`; the DM goes to `SLACK_USER_ID`) |
| `GH_WORKER_TOKEN` | optional: a fine-grained **read-only** GitHub PAT (contents, pull requests, actions: read) that workers get as `GH_TOKEN` for `gh` reads. Without it workers have no `gh` authentication at all, by design -- the engine fetches the PR/CI/review context into the prompt and your own `gh` login is never reachable from a worker (`doctor` warns when it is absent) |

Format: `KEY=value` per line; `#` comments, blank lines, optional quotes and an optional `export `
prefix are accepted.

## Isolation guarantees

- Secrets are parsed into `Profile.secrets` and are **never** written to `os.environ`. Workers receive
  only the provider credential they need (`providers.worker_env`) and inherited credential variables
  are scrubbed from the process first, so a worker can never read the Linear or Slack token.
- Workers never see your `gh` login: their `GH_CONFIG_DIR` is a plain file (no `hosts.yml`, no
  keyring fallback) and `GH_TOKEN` is the optional read-only `GH_WORKER_TOKEN` or empty. Their git
  runs with every GitHub push URL rewritten to an unusable transport and with hooks disabled
  (`core.hooksPath` -> a path under a regular file -- `%SystemRoot%\System32\drivers\etc\hosts\no-hooks`
  on Windows, `/dev/null/no-hooks` elsewhere -- where no hook can ever be created), every git the
  engine runs in a clone or worktree (fetches, worktree setup, pushes) uses the same path, and shell
  writes into a `.git` directory or a git/gh/ssh config file are denied; changes to the clone's shared
  git config and hooks are reverted. A worker's whole process tree is killed when its session ends.
  Workers load no filesystem settings
  (`setting_sources=[]`): a repo's `.claude/settings.json` is never applied; its `CLAUDE.md` /
  `AGENTS.md` are appended to the system prompt by the engine instead.
- Runtime data lives under the per-profile data dir, never in the repo:
  `~/Library/Application Support/software-factory/<profile>/` on macOS,
  `$XDG_DATA_HOME/software-factory/<profile>/` (default `~/.local/share`) on Linux and Windows
  (`SOFTWARE_FACTORY_DATA_DIR` overrides the base; `factory schedule install` embeds it in the
  scheduled jobs so they use the same dir as the shell that installed them). It holds `state.db`,
  `wt/` (worktrees), `logs/` (`audit.jsonl`, `runs.jsonl`, transcripts), `reviews/`, `refinements/`,
  `scripts/`, `scratch/`, `tick.lock` and `gh-noauth` (the empty file workers get as
  `GH_CONFIG_DIR`). There is no hooks directory to keep empty: see the hooks path above.
- Worktrees under `wt/` are detached HEADs of your clones; the factory never creates, resets or
  deletes a local branch in a clone. A worktree kept for you (push refused, conflict left open) is
  continued with `git switch -c <branch>-factory` inside it (the suffix keeps it clear of your own
  local `<branch>`) -- or, for a PR, its commits are reviewed and pushed with
  `git push origin HEAD:refs/heads/<branch>` -- then released with `factory release <target>`.
- Loading profile A reads only `<profiles_dir>/A/`; profile B's `factory.toml` and `.env` are never opened.
- Employer profiles (`employer = true`) refuse `provider.kind = "oauth"`, require a disclosure footer and
  default to `mode = "observe"` (only `rv` runs for real; everything else is recorded as "would run").

## Rollout ladder

Change config, not code, one rung at a time, and let the audit log + `factory report` justify the next
rung: `mode = "observe"` -> `enabled_kinds = ["ci", "conflict"]` -> add `"rc"` -> add `"build"`
(self-labeled tickets) -> add `"plan"`.
