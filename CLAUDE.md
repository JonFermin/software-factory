# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Purpose

`software-factory` is a local, scheduled engine: it scans Linear + GitHub state, dispatches one
narrow Claude Agent SDK worker per action (`ci`, `conflict`, `rc`, `rv`, `refine`, `plan`, `build`,
`ready`), and performs every outward write itself behind deterministic policy. **Workers propose,
the engine acts.** `README.md` is the user-facing overview; `docs/architecture.md` is the module
contract (public names/signatures, invariants, implemented deviations, kind handler spec) and wins
over anything else when they disagree. `docs/adr/` holds decisions.

## Layout

```
src/software_factory/
  cli.py          typer app `factory` (console script); every command loads exactly one profile
  config.py       profiles: <profiles_dir>/<name>/factory.toml + .env -> Profile (secrets never in os.environ);
                  profiles dir: --profiles-dir > $SOFTWARE_FACTORY_PROFILES_DIR > user_profiles_dir() if present > <repo>/profiles
  paths.py        per-profile data dir (state.db, wt/, logs/, reviews/, refinements/, scripts/, tick.lock)
  actions.py      Action / Kind, version keys, grouping + ordering
  scan.py         GitHub + tracker state -> Action[]; deterministic filters (Skip with a reason)
  dispatch.py     tick: lock -> scrub env -> precheck -> scan -> filter -> dispatch -> record; run_one
  kinds/          one handler per kind (base.py = shared prompt/push/flag layer)
  effects.py      the ONLY place outward writes happen (LiveEffects) + RecordingEffects (observe / dry-run)
  policy.py       worker tool allow/deny (can_use_tool + PreToolUse hook), shell classifier, push policy
  runner.py       Claude Agent SDK wrapper (structured output, timeout, transcript)
  worktree.py     throwaway linked worktrees of the user's clones
  providers.py    oauth | api_key | bedrock | vertex -> worker env; credential checks; env scrubbing
  sources/github.py (gh CLI, injectable runner)   trackers/linear.py (GraphQL, httpx)   resolve.py (repo inference)
  discover.py / readiness.py / codeowners.py / evidence.py / metrics.py / notify.py / scheduler/
  workers/*.md    worker prompts (+ schemas.py structured outputs); prompt headings are pinned by tests
tests/            pytest; conftest.py has the shared fixtures
profiles/         README.md + example/factory.toml only (the rest is gitignored); real profiles, employer ones
                  included, live outside the repo in <config root>/software-factory/profiles (macOS ~/Library/
                  Application Support, else $XDG_CONFIG_HOME or ~/.config -- Windows too: MSIX apps virtualize AppData) or $SOFTWARE_FACTORY_PROFILES_DIR
```

## Commands

```
uv run pytest                        # full suite; single file: uv run pytest tests/test_cli.py -q
uv run ruff check src tests          # lint (line length 110; E501 ignored)
uv run factory doctor                # environment check (read-only)
uv run factory tick --dry-run        # scan + filter + print, no workers, no state writes
uv run python scripts/e2e_local.py   # real workers vs a local sandbox (no outward writes; ~$1 of model tokens)
```

Run the local E2E after changing worker prompts, `policy.py`, `runner.py`, `effects.py` or a kind
handler: unit tests fake the worker, and the bugs that matter most (an SDK option's real meaning, a tool
the CLI needs, a prompt that makes the model refuse its job) only show up with a real one.

Python 3.13 via uv; `src` layout; do not add dependencies without asking (`pyproject.toml` /
`uv.lock`).

## Invariants that must never be broken

1. Workers never perform an outward write; `policy.py` denies push / gh writes / graphql / Linear /
   Slack / publishing / `mcp__*` in both `can_use_tool` and the `PreToolUse` hook, plus `GIT_*` /
   `GH_*` env assignments, command-running git config keys and shell writes into git dirs. Below the
   tool layer workers get no `gh` login (optional read-only `GH_WORKER_TOKEN` only), GitHub push URLs
   rewritten to an unusable transport, hooks disabled (`core.hooksPath` ->
   `providers.unplantable_hooks_path()`, a path under a regular file where nothing can be planted;
   the same for every git the engine runs in a clone or worktree, fetches included), their process
   tree killed when the session ends, and no filesystem settings (`setting_sources=[]`;
   CLAUDE.md/AGENTS.md are appended to the system prompt instead).
2. Every outward write goes through `effects.py`; push refspec `HEAD:refs/heads/<branch>` only
   (never `+`, `--force*`, `--no-verify`); draft PRs only; evidence edits only on factory-opened PRs;
   never comments, reviews, merges, closes, ready-for-review, label creation. The push policy refuses
   sensitive globs and every changed file CODEOWNERS assigns to someone else (per path, last match
   wins). Git LFS objects of a push are uploaded explicitly (hooks are off) or the push is refused.
3. `mode = "observe"` and `--dry-run` perform zero GitHub/tracker writes (`RecordingEffects`).
4. State is recorded only on outcome `done` / `nothing`; `max_attempts` caps retries per
   `(target, kind, version)` and the dispatcher flags the cap once (`--dry-run` writes no state and no
   flags).
5. Auth failures flag once and stop the wave (exit 2, also for `factory run` / `ready`) without
   spending attempts; a provider rate / usage limit (`ProviderUnavailable`, raised only for a worker
   run that produced no output) stops the wave the same way, flagged once per day.
6. Profiles never cross: one profile per process, secrets only in `Profile.secrets`, runtime data
   under that profile's data dir.
7. Refine persists the original and delivers a user-facing notification before editing the issue.

If a change touches one of these, add or extend the test that proves it still holds.

## Testing conventions

- Everything is faked: no network, no real `gh` / Linear / Slack, no OS scheduler, and **never a real
  Claude worker** -- inject a fake `run_worker` (see `FakeWorker` in `tests/test_kinds_issue.py`) or
  `no_worker`. GitHub is a fake runner returning fixture JSON (`tests/fixtures/github/`), Linear uses
  `httpx.MockTransport`, the scheduler takes a fake `Runner`, the CLI monkeypatches its seams
  (`cli.build_deps`, `cli.gh_client`, `cli.tracker_client`, ...).
- `tests/conftest.py` isolates every test (private data dir via `SOFTWARE_FACTORY_DATA_DIR`, no
  inherited secrets, hermetic git identity) and provides `make_profile` / `load` / `profile`,
  `state`, `git_repo_factory` / `bare_repo_factory` (real local git repos, local bare origins).
- Windows is the primary host: `pathlib`, list-args `subprocess` (never `shell=True`), UTF-8 with
  `errors="replace"`, explicit timeouts, paths with spaces in tests.
- Keep output plain ASCII; `typer.echo` only in the CLI.

## Do not run for real without the user's say-so

`factory tick` (without `--dry-run`), `factory run`, `factory ready`, `factory revert-refine`,
`factory schedule install|uninstall`, and anything else that pushes, opens PRs, edits Linear or
registers scheduler jobs. `doctor`, `status`, `flags`, `report`, `readiness --no-run`,
`schedule show` and `tick --dry-run` are read-only (they may write flags/audit records to the
profile's own data dir). Never commit or push from an agent session unless asked.
