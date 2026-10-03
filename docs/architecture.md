# software-factory architecture (implementation contract)

This document is the contract between modules. Public names and signatures listed here are what other
modules import; implementations may add private helpers but must not change these shapes without
updating this file. Already-written type layer (treat as fixed): `actions.py`, `config.py`,
`workers/schemas.py`, `trackers/base.py`, `sources/models.py`.

## Invariants (tests must enforce these)

1. **Workers propose, the engine acts.** Workers (Claude Agent SDK sessions) never perform an outward
   write. `git push`, every `gh` write, `gh api` with a non-GET method or the `graphql` endpoint,
   Linear/Slack HTTP, package publishing, and all `mcp__*` tools are denied by `policy.py` in *both*
   the `can_use_tool` callback and a `PreToolUse` hook; so are `GIT_*` / `GH_*` / `SSH_*` / editor /
   pager env assignments, git config keys that run a command or redirect remotes and credentials,
   and shell writes into a git dir or a git/gh/ssh config or credential file. Denials return a
   message so the model adapts instead of hanging. Below the tool layer: workers load **no**
   filesystem settings (`setting_sources=[]`, so a repo's `.claude/settings.json` hooks /
   `apiKeyHelper` / `env` / plugins never run; its `CLAUDE.md` / `AGENTS.md` are appended to the
   system prompt by the engine), get no `gh` login (`GH_CONFIG_DIR` is a file; `GH_TOKEN` is the
   optional read-only `GH_WORKER_TOKEN` or empty), run git with every GitHub push URL rewritten to an
   unusable transport and with hooks disabled (`core.hooksPath` -> `providers.unplantable_hooks_path()`,
   a path under a regular file where no hook can ever be created), have their whole process tree
   killed when the session ends, and a worker that changes the clone's shared git config or hooks
   anyway has the change reverted and its output discarded.
2. **Every outward write goes through `effects.py`.** Allowed outward writes: push branches
   (incl. orphan `factory/evidence`) with refspec `HEAD:refs/heads/<branch>` only (never `+`, never
   `--force*`, never `--no-verify`), always with git hooks disabled (`-c core.hooksPath=<unplantable
   path>`, like every other git the engine runs in a clone or worktree), Git LFS objects of the pushed
   change uploaded explicitly first (hooks off means git-lfs's `pre-push` never runs) or the push
   refused; open **draft** PRs; edit the `<!-- factory:evidence -->` section of PRs the factory opened; Linear
   state transitions; Linear sub-issue creation (+ blocked-by relations); Linear issue
   title/description edits + `factory:refine` label removal (refine only). Never: comments, replies,
   reviews, resolving threads, merge, close, ready-for-review, force-push, label creation. The push
   policy refuses a change set with a file matching the sensitive globs or a file CODEOWNERS assigns
   to someone else (per changed path, GitHub's last-match-wins rule).
3. **Observe mode performs zero outward writes to GitHub or the tracker.** `mode = "observe"` runs
   only `rv` for real; every other kind is recorded as "would run" in the audit log. Effects are the
   `RecordingEffects` implementation. Notifications to self (toast/Slack DM/audit) still happen.
   `--dry-run` scans, filters and prints: no worker, no state write, no flag.
4. **State is recorded only on worker outcome `done`/`nothing`** (after effects succeed). Anything
   else is a failed attempt; `max_attempts` (default 3) per `(target, kind, version)` stops retries,
   and the dispatcher flags the cap once per `(target, kind, version)`.
5. **Auth failures and provider outages stop the wave.** Precheck failure (gh, tracker, provider
   creds) -> flag once per source and day, exit code 2, no attempts spent. A mid-wave auth failure
   (gh / tracker / model / git credentials on push) stops launching new actions, flags once, exit 2,
   and does not count an attempt. A model provider that cannot serve sessions (rate / usage limit,
   overloaded, billing: `ProviderUnavailable`) is handled the same way under its own once-a-day flag.
6. **Profiles never cross.** A process loads exactly one profile; secrets live in `Profile.secrets`
   and are never put in `os.environ`; state/worktrees/logs live under that profile's data dir.
7. **Refine: notify before edit.** Persist original title/description (state + `refinements/<KEY>.md`)
   and deliver a notification on >=1 user-facing channel *before* editing the Linear issue. No
   notification -> no edit.

## Runtime layout (`paths.py`)

```python
def data_root() -> Path            # macOS ~/Library/Application Support, else $XDG_DATA_HOME or ~/.local/share (Windows too: MSIX apps virtualize AppData)
DATA_DIR_ENV = "SOFTWARE_FACTORY_DATA_DIR"   # replaces data_root() when set
class ProfilePaths:                # all under data_root()/software-factory/<profile>/
    root: Path
    state_db: Path                 # state.db
    worktrees: Path                # wt/   (short names: Windows MAX_PATH)
    logs: Path                     # logs/
    audit_log: Path                # logs/audit.jsonl
    runs_log: Path                 # logs/runs.jsonl
    transcripts: Path              # logs/transcripts/
    reviews: Path                  # reviews/  -> reviews/<repo-name>-<pr>.md
    refinements: Path              # refinements/ -> refinements/<KEY>.md
    lock_file: Path                # tick.lock
    scripts: Path                  # scripts/ (scheduler wrapper scripts)
    @classmethod
    def for_profile(cls, name: str, root: Path | None = None) -> "ProfilePaths"
    def ensure(self) -> "ProfilePaths"   # mkdir -p all dirs
```
Also under the profile root, created on demand: `scratch/` (cwd of workers without a worktree) and
`gh-noauth` (a regular *file* used as the workers' `GH_CONFIG_DIR`). There is no hooks directory:
`core.hooksPath` is `providers.unplantable_hooks_path()`, outside the data dir (see Providers).

## Config (`config.py`, written)

```python
def load_profile(name: str = "personal", profiles_dir: str | Path | None = None) -> Profile
def list_profiles(profiles_dir: str | Path | None = None) -> list[str]
def resolve_profiles_dir(explicit: str | Path | None = None) -> Path
def config_root() -> Path          # macOS ~/Library/Application Support, else $XDG_CONFIG_HOME or ~/.config (Windows too)
def user_profiles_dir() -> Path    # config_root()/software-factory/profiles (may not exist)
```

Profiles dir, first hit wins: `--profiles-dir` (the explicit argument), `SOFTWARE_FACTORY_PROFILES_DIR`,
`user_profiles_dir()` when it is a directory, else `<repo>/profiles` (which tracks only `README.md`
and `example/`; everything else under it is gitignored). `SOFTWARE_FACTORY_CONFIG_DIR` replaces
`config_root()` (tests, portable installs). Real profiles, employer ones included, live outside the
repo. `repos.root` defaults to `~/code`.

See `factory.example.toml` for every key. Key fields:
`mode`, `employer`, `enabled_kinds` (rollout ladder; default upkeep kinds ci/conflict/rc/rv),
`max_parallel`, `max_attempts`, `session_timeout_min`, `provider.*` (incl. `cli_path`: the Claude
CLI binary the SDK launches; Windows SDK wheels bundle none), `github.*`, `repos.*` (`root`,
`include`, `exclude`, `overrides[name] -> RepoOverride`), `linear.*`, `resolve.*`, `policy.*`,
`notify.*`, `schedule.*`. `ScheduleConfig` is validated at load (`interval_minutes` 1..1439,
`digest_weekday` MON..SUN, `digest_time` HH:MM), so a typo is a `ConfigError` from `load_profile`,
not a traceback out of `doctor` / `schedule show`. Secrets keys: `CLAUDE_CODE_OAUTH_TOKEN`,
`ANTHROPIC_API_KEY`, `LINEAR_API_KEY`, `SLACK_BOT_TOKEN`, `SLACK_USER_ID`, optional
`GH_WORKER_TOKEN` (+ `AWS_*` / `GOOGLE_*` when needed).

## Providers (`providers.py`)

```python
SCRUB_ENV_VARS: tuple[str, ...]    # inherited vars that must not leak into workers: ANTHROPIC_API_KEY,
                                   # ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, CLAUDE_CODE_OAUTH_TOKEN,
                                   # CLAUDE_CODE_USE_BEDROCK, CLAUDE_CODE_USE_VERTEX, CLAUDECODE,
                                   # CLAUDE_CODE_ENTRYPOINT, CLAUDE_CODE_OAUTH_SCOPES, LINEAR_API_KEY,
                                   # SLACK_BOT_TOKEN, ... -- deliberately NOT GH_TOKEN / GITHUB_TOKEN:
                                   # the engine's own gh may depend on them (worker_env overrides them)
def scrub_process_env() -> None    # pop SCRUB_ENV_VARS from os.environ (the SDK merges os.environ)
def worker_env(profile: Profile) -> dict[str, str]
    # oauth   -> {"CLAUDE_CODE_OAUTH_TOKEN": secret}
    # api_key -> {"ANTHROPIC_API_KEY": secret}
    # bedrock -> {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE"?, "AWS_REGION"}
    # vertex  -> {"CLAUDE_CODE_USE_VERTEX": "1", "ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION"}
    # always: {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "GH_PROMPT_DISABLED": "1",
    #          "GH_NO_UPDATE_NOTIFIER": "1", "GIT_TERMINAL_PROMPT": "0"}
    #   + gh_isolation_env: GH_CONFIG_DIR=<profile root>/gh-noauth (a file), GH_TOKEN=GH_WORKER_TOKEN or "",
    #     GITHUB_TOKEN=""
    #   + worker_git_config_env: GIT_CONFIG_COUNT/KEY_i/VALUE_i = url.factory-no-push://.pushInsteadOf for
    #     every GitHub URL form, then core.hooksPath=unplantable_hooks_path()
def model_id(profile: Profile, logical: str | None = None) -> str   # provider.model_ids override
def fallback_model_id(profile: Profile) -> str | None
def check_credentials(profile: Profile) -> list[str]   # human-readable problems; [] = ok
class ProviderError(RuntimeError)
GH_WORKER_TOKEN_KEY = "GH_WORKER_TOKEN"
def gh_noauth_config_path(profile) -> Path; def gh_isolation_env(profile) -> dict[str, str]
def unplantable_hooks_path() -> str          # <regular file or char device>/no-hooks; raises (fail closed) when unavailable
def git_config_env(entries) -> dict[str, str]
def git_no_push_entries() -> list[tuple[str, str]]; def git_no_hooks_entries(...) -> list[tuple[str, str]]
def git_no_push_env() -> dict[str, str]; def worker_git_config_env(profile) -> dict[str, str]
```
`GH_CONFIG_DIR` pointing at a *file* means gh finds no `hosts.yml` (so it never falls back to the
keyring) and nothing can create one there. Without `GH_WORKER_TOKEN` workers run `gh`
unauthenticated by design; the engine puts the PR / CI / review context in the prompt.

`unplantable_hooks_path()` is the one `core.hooksPath` of the whole system -- the workers' (through
`worker_git_config_env`) and every git the engine runs in a clone or worktree (worktree setup and
removal, every fetch including the evidence-parent fetch and `ready_action`'s fetch, pushes). It is a
path *under a regular file*: `%SystemRoot%\System32\drivers\etc\hosts\no-hooks` on Windows,
`/dev/null/no-hooks` on POSIX. git finds no hooks there and carries on; no process can ever create
anything under a file, so there is no directory to empty, no window between "checked empty" and "git
ran" in which a worker (or a background process it left behind) could plant a `pre-push`, and
nothing to hide from the worker: knowing the path gives it nothing. When the host offers no such
parent, the path is refused and git does not run (fail closed).

## Repos (`repos.py`)

```python
class RepoInfo(BaseModel):
    slug: str               # owner/name from origin remote
    name: str               # GitHub repo name
    dirname: str            # local directory name (may differ from name)
    path: Path              # local clone
    default_branch: str     # from override, else `git symbolic-ref refs/remotes/origin/HEAD`, else "main"
    override: RepoOverride
class RepoIndex:
    @classmethod
    def discover(cls, profile: Profile) -> "RepoIndex"   # scan repos.root children with .git and a GitHub origin
                                                         # whose owner is in github.owners (or include list); apply exclude
    def all(self) -> list[RepoInfo]
    def get(self, key: str) -> RepoInfo | None           # accepts slug, name, or dirname (case-insensitive)
    def slugs(self) -> list[str]
    def summary(self, slug: str) -> str                  # first section of CLAUDE.md/AGENTS.md/README (<=1500 chars)
def parse_github_remote(url: str) -> str | None          # git@github.com:o/n(.git) | https://github.com/o/n(.git) -> "o/n"
```

## State (`state.py`) -- SQLite, one file per profile

```python
class State:
    def __init__(self, path: Path)              # creates schema; WAL; check_same_thread=False + a lock
    # dedupe / attempts
    def is_handled(self, target: str, kind: str, version: str) -> bool
    def mark_handled(self, target: str, kind: str, version: str, outcome: str, note: str = "") -> None
    def attempts(self, target: str, kind: str, version: str) -> int
    def last_attempt(self, target: str, kind: str, version: str) -> dict | None   # for the exhausted flag
    def record_attempt(self, target: str, kind: str, version: str, outcome: str, note: str = "") -> int
    # kept worktrees (awaiting me)
    def keep_worktree(self, target: str, kind: str, path: str, reason: str) -> None
    def kept_worktree(self, target: str) -> dict | None
    def kept_worktrees(self) -> list[dict]
    def release_worktree(self, target: str) -> None
    # flags
    def add_flag(self, target: str, kind: str, reason: str, detail: str = "", dedupe_key: str | None = None) -> int | None
        # returns None when an open flag with the same dedupe_key exists (flag-once semantics)
    def flags(self, open_only: bool = True) -> list[dict]
    def ack_flag(self, flag_id: int) -> None
    # factory-opened PRs and pushes (policy + metrics)
    def register_pr(self, repo: str, number: int, branch: str, kind: str, issue_key: str | None = None, url: str = "") -> None
    def is_factory_pr(self, repo: str, number: int) -> bool
    def factory_prs(self) -> list[dict]
    def record_push(self, repo: str, number: int | None, branch: str, kind: str, before_sha: str, after_sha: str, commit_shas: list[str]) -> None
    def link_pushes(self, repo: str, branch: str, number: int) -> int
        # attach the (repo, branch) push rows recorded with number NULL (build/ready push before the PR
        # exists) to the PR just opened; returns the number of rows linked; rows with a PR are untouched
    def pushed_shas(self, repo: str, number: int) -> set[str]
    def touched_prs(self) -> list[dict]        # every (repo, number) with >=1 factory push or factory-opened
    # repo resolution cache
    def get_resolution(self, issue_id: str) -> dict | None   # {repo, confidence, source}
    def set_resolution(self, issue_id: str, repo: str | None, confidence: float, source: str) -> None
    # refinements
    def save_refinement(self, issue_id: str, key: str, original_title: str, original_description: str,
                        new_title: str, new_description: str) -> None    # status='pending'
    def set_refinement_status(self, key: str, status: str) -> None        # pending|notified|edited|applied|reverted
    def get_refinement(self, key: str) -> dict | None
    # runs (also appended to runs.jsonl by the dispatcher)
    def record_run(self, run: dict) -> None
    def runs(self, since_iso: str | None = None) -> list[dict]
    # pr outcomes (metrics)
    def upsert_pr_outcome(self, repo: str, number: int, **fields) -> None
    def pr_outcomes(self) -> list[dict]
    # generic JSON cache (discover/readiness/sibling votes/evidence/pending PRs)
    def cache_get(self, namespace: str, key: str) -> Any | None
    def cache_set(self, namespace: str, key: str, value: Any) -> None
    def cache_delete(self, namespace: str, key: str | None = None) -> None
```

## GitHub (`sources/github.py`)

All GitHub access goes through `gh`. The runner is injectable so tests use fixture JSON.

```python
class GhError(RuntimeError); class GhAuthError(GhError)
GhRunner = Callable[[list[str]], str]           # args after "gh"; returns stdout; raises GhError
def default_gh_runner(args, *, cwd=None, input=None) -> str   # subprocess, GH_PROMPT_DISABLED=1, timeout
class GitHub:
    def __init__(self, login: str | None = None, run: GhRunner | None = None)
    def login(self) -> str                                         # cached `gh api user --jq .login`
    def auth_check(self) -> None                                   # raises GhAuthError
    # reads
    def my_open_prs(self, repos: list[str], max_age_days: int) -> list[PullRequest]   # basic fields
    def review_requested(self, max_age_days: int) -> list[PullRequest]                # non-draft, requested of me directly
    def pr_details(self, repo: str, number: int) -> PullRequest     # checks + feedback + threads populated
    def failed_log(self, repo: str, run_id: int, max_bytes: int = 60_000) -> str     # `gh run view --log-failed`, tail-truncated
    def pr_body(self, repo: str, number: int) -> str
    def pr_commits(self, repo: str, number: int) -> list[PrCommit]
    def pr_state(self, repo: str, number: int) -> PullRequest      # state/merged_at/closed_at/created_at
    def find_open_pr_for_branch(self, repo: str, branch: str) -> PullRequest | None
    # writes -- ONLY called from effects.py
    def create_draft_pr(self, repo: str, head: str, base: str, title: str, body: str) -> PullRequest
    def edit_pr_body(self, repo: str, number: int, body: str) -> None
```
Review threads come from a read-only `gh api graphql` query (engine-side reads are fine; only workers
are barred from graphql). Feedback excludes my own comments and `github.ignored_logins`.
`PullRequest.feedback` keeps the comments that sit on a resolved thread (the `rc` version key is
computed over it, so it never moves backwards); `PullRequest.open_feedback` drops them and is the
list that can still call for work.

## Scan (`scan.py`)

```python
class FlagSpec(BaseModel): reason: str; detail: str = ""; dedupe_key: str | None = None; level: str = "warn"
class Skip(BaseModel): action: Action; reason: str; flag: FlagSpec | None = None
class ScanResult(BaseModel): actions: list[Action]; skipped: list[Skip]; errors: list[str]
def scan_github(profile, gh: GitHub, repos: RepoIndex, state: State) -> list[Action]
def scan_tracker(profile, tracker: Tracker, state: State, gh: GitHub | None = None) -> list[Action]
def filter_actions(profile, actions, state, tracker=None) -> tuple[list[Action], list[Skip]]
def skip_for(profile, action, state, tracker=None) -> Skip | None;  def skip_reason(...) -> str | None
def exhausted_flag(profile, action, attempts, last=None) -> FlagSpec
def managed_slug(repos: RepoIndex, slug: str) -> str
def scan(profile, *, gh, tracker, repos, state) -> ScanResult
```
The scan writes nothing (no state, no flag, no notification), so `--dry-run` stays pure; a skip
that the human must hear about carries a `FlagSpec` and the dispatcher raises it. Every PR key the
engine derives (action targets, `gh` calls, state rows) uses the managed clone's spelling of the
slug (`managed_slug`: GitHub slugs are case-insensitive), so tick-derived and `factory run` keys
agree.

Action detection and version keys:

| Kind | Trigger | `version` | `context` keys |
|---|---|---|---|
| ci | my open PR in a managed repo, updated <= authored_max_age_days, >=1 failing check on current head | head SHA | head_ref_name, head_ref_oid, base_ref_name, base_ref_oid, failing: [{name, run_id, job_id, workflow, details_url}] |
| conflict | same PR set, `mergeable == CONFLICTING` | `{head}:{base}` SHAs | head/base names+oids |
| rc | same PR set, `open_feedback` newer than my last commit/factory push, or unresolved non-outdated threads carrying such feedback | ISO timestamp of the latest item over *all* of `pr.feedback` (resolved-thread comments included: the key never regresses to an older, never-recorded value) | head/base, feedback_ids, unresolved_thread_ids |
| rv | review requested of me directly, non-draft, updated <= review_max_age_days | head SHA | head/base, author |
| refine | issue has `linear.label_refine` | issue `updated_at` | issue_id, labels |
| plan | issue has `label_plan`, no children | issue `updated_at` | |
| build | `label_build`, `state_type == "unstarted"`, no open linked PR | issue id (attempt-capped) | branch_name, team_id, unfinished_blockers |

Filters (each produces a `Skip` with a reason): kind not in `enabled_kinds` (only when `mode="full"`;
in observe, non-rv actions are kept and dry-run); handled at same version; attempts >= max_attempts
(the skip carries the "attempts exhausted" `FlagSpec`: how the last attempt ended, what re-arms the
action -- a new head/base/feedback for PR kinds, an issue edit for refine/plan, nothing for build
whose version is the issue id -- and the `factory run <kind> <target>` retry; dedupe
`exhausted:<target>:<kind>:<version>`); kept worktree for the target; build blocked by unfinished
relations.

## Repo inference (`resolve.py`)

```python
class Resolution(BaseModel): repo: str | None; confidence: float; source: str; multi_repo: bool = False; candidates: list[str] = []
class Resolver:
    def __init__(self, profile, tracker: Tracker, repos: RepoIndex, state: State, gh: GitHub | None,
                 picker: Callable[[Issue, list[RepoInfo]], Awaitable[RepoPick]] | None)
    async def resolve(self, issue: Issue) -> Resolution
```
Chain (first confident hit wins; cache result in state): 1 `repo:<name>` label (also the
`<!-- factory:repo owner/name -->` marker the planner writes into descriptions, since label creation
is not allowed); then the planner's own record for a child it created (`set_resolution(...,
source="plan")`), honoured whatever the issue's `updated_at` (labelling the child for build edits
it) unless it names a repo the profile no longer manages; 2 GitHub attachments/PR links on the issue;
3 majority vote of repos linked from sibling issues in the same project (>=
`resolve.sibling_min_votes`); 4 repo names / file paths in title+description matched against
`RepoIndex`; 5 `picker` (model, Opus, structured `RepoPick`) over candidates summarised by
`RepoIndex.summary`. Below `min_confidence` or `multi_repo` -> unresolved.

## Worktrees (`worktree.py`)

```python
class Worktree(BaseModel): path: Path; repo: str; clone: Path; base_sha: str; branch: str | None; detached: bool
class WorktreeError(RuntimeError)
def clone_lock(clone: Path) -> threading.RLock          # process-wide, per clone, re-entrant
class WorktreeManager:
    def __init__(self, root: Path, git: Callable[..., str] | None = None)   # refuses (WorktreeError) when hooks cannot be disabled
    def for_pr(self, repo: RepoInfo, pr: PullRequest) -> Worktree       # fetch head (pull/<n>/head for forks), detached at head_ref_oid, verify HEAD
    def for_new_branch(self, repo: RepoInfo, branch: str) -> Worktree   # detached at origin/<branch> if it exists, else origin/<default>
    def for_default_branch(self, repo: RepoInfo) -> Worktree            # detached at origin/<default> (readiness, refine/plan reading)
    def no_hooks_args(self) -> list[str]                                # ["-c", "core.hooksPath=<unplantable path>"], checked each call
    def head(self, wt: Worktree) -> str
    def new_commits(self, wt: Worktree) -> list[str]                    # base_sha..HEAD
    def changed_files(self, wt: Worktree) -> list[str]                  # base_sha..HEAD, name-only
    def has_conflict_markers(self, wt: Worktree) -> bool
    def remove(self, wt: Worktree) -> None                              # git worktree remove --force + prune
    def cleanup(self, keep_paths: set[str]) -> list[Path]               # remove everything under root not kept
```
Every worktree is a **detached HEAD**; no local branch is ever created, reset or deleted in the
user's clone (`git worktree add -B` would silently reset an existing local branch, and a branch
checked out in a kept worktree would block every later run). For `for_new_branch`, `wt.branch` is
the push target (the engine pushes `HEAD:refs/heads/<branch>`), `wt.base_sha` the start commit; the
default branch itself is refused. Clone-level git operations (fetch, worktree add/remove/prune, the
exclude file) are serialised per clone with `clone_lock` (concurrent fetches fail with `cannot
lock ref`; a ref-lock failure is retried) and every one runs with `-c core.hooksPath=<unplantable
path>` (`worktree add` fires `post-checkout`, `fetch` fires `reference-transaction` from the clone's
shared, worker-writable `.git/hooks`); when that path is unavailable git does not run (`WorktreeError`).
Each worktree adds `.factory/` to the clone's shared `info/exclude` (worker scratch + evidence images).

## Policy (`policy.py`)

```python
class ToolDecision(BaseModel): allow: bool; reason: str = ""
PRE_TOOL_USE_HOOK_TIMEOUT_S = 30.0
class ToolPolicy:
    def __init__(self, worktree: Path, profile: Profile, extra_read_dirs: list[Path] | None = None, *, read_only: bool = False)
    def check(self, tool_name: str, tool_input: dict) -> ToolDecision
    async def can_use_tool(self, tool_name, tool_input, context) -> PermissionResultAllow | PermissionResultDeny
    async def pre_tool_use_hook(self, input_data, tool_use_id, context) -> dict   # hookSpecificOutput permissionDecision deny
    def hooks(self) -> dict[str, list[HookMatcher]]                  # PreToolUse, timeout=PRE_TOOL_USE_HOOK_TIMEOUT_S
def classify_shell(command: str, *, powershell: bool = False) -> ToolDecision
class PushDecision(BaseModel): allowed: bool; reasons: list[str]
def evaluate_push(kind: Kind, output: CodeWorkerOutput, changed_files: list[str], sensitive_globs: list[str],
                  *, conflict_markers: bool = False, foreign_owned: list[str] | None = None) -> PushDecision
def sensitive_globs_for(profile: Profile, repo: RepoInfo, *, my_login: str | None = None) -> list[str]  # policy + override globs
def foreign_owned_changes(profile, repo, changed_files: list[str], my_login: str | None = None) -> list[str]
def may_edit_pr_body(state: State, repo: str, number: int) -> bool       # factory-opened PRs only
def matches_any(path: str, globs: list[str]) -> bool                     # gitwildmatch semantics (pathspec)
```
Tool allowlist: Read, Glob, Grep, Edit, Write, MultiEdit, NotebookEdit, TodoWrite, Task/Agent,
Bash, PowerShell (shell commands classified), WebFetch/WebSearch only if `policy.allow_web_tools`,
plus `policy.extra_allowed_tools`. Unknown tools and `mcp__*` are denied. Edit/Write/NotebookEdit paths
must resolve inside the worktree (no OS temp dir) and never under a `.git` component.

Shell classification is structural: the command is split into simple commands (`; && || | &`,
newlines, `$( )`, backticks, parentheses), env assignments and wrappers (`env`, `sudo`, `xargs`,
`uv run` ...) are stripped, nested shells (`bash -c`, `sh -c`, `pwsh -Command`, `cmd /c`,
`-EncodedCommand`) are unwrapped recursively, and each `argv` is matched. Denials:
`git push|send-pack|http-push`, `git remote add|set-url|remove|rename|...`, `git credential*`,
`git -c` / `--config-env` / `git config` writes of keys that redirect remotes or credentials or run a
command (`url.*`, `remote.*`, `credential*`, `alias.*`, `core.hooksPath|sshCommand|editor|fsmonitor|...`,
`filter.*`, `diff.*.textconv`, `merge.*.driver`, `include*`, `http.*`, pagers other than off, ...);
env assignments / `export` / `setx` of `GIT_*`, `GH_*`, `SSH_*`, `GITHUB_TOKEN`, `EDITOR`, `VISUAL`,
`BROWSER`, `PAGER` (pager vars only to an "off" value); every `gh` subcommand outside a read-only
allowlist (`pr view|list|diff|checks|status`, `run view|list|watch`, `issue view|list`, `repo view`,
`release view|list`, `search *`, `auth status`, `api` with GET only and not `graphql`);
`curl`/`wget`/`Invoke-WebRequest`/`Invoke-RestMethod`/`iwr`/`irm` with a non-GET method or a body;
any mention of `api.linear.app`, `slack.com/api`, `hooks.slack.com`; publishing (`npm|pnpm|yarn
publish`, `uv publish`, `twine upload`, `cargo publish`, `gem push`, `docker push`). **Write
targets**: output redirections (`>`, `>>`, `2>`, `&>`, PowerShell `*>`) and the destinations of
`tee` / `cp` / `mv` / `install` / `ln` / `touch` / `chmod` / `rm` / `mkdir` / `sed -i` / `perl -i` /
`dd of=` / `cd` and their PowerShell cmdlets are denied inside a git dir (a `.git` component,
`<x>.git/...`, `$GIT_DIR` / `$GIT_COMMON_DIR`, a `$(git rev-parse --git-dir|--git-common-dir|--git-path
...)` substitution, or a variable assigned one of those) or on a git/gh/ssh config or credential
file (`.gitconfig`, `.git-credentials`, `.netrc`, `hosts.yml`, `.ssh/`, `.config/git|gh/`). Reads of
`.git` stay allowed. Unparseable input fails closed only when it mentions a dangerous token.

Push policy: worker `push_ok` and `status == "done"`; >=1 passing verification run; no changed file
matches the sensitive globs (`sensitive_globs_for`: `policy.sensitive_globs` + the repo override's;
CODEOWNERS is **not** folded into globs any more); no changed file in `foreign_owned` (computed by
`foreign_owned_changes` with exact per-path last-match-wins over the repo's CODEOWNERS; owners that
are neither my login nor in `policy.my_teams` make the file foreign; with no resolvable login every
owned path counts as foreign -- fail closed); `rc`: no comment dispositioned `skipped`; `conflict`:
`all_resolved`, no `unsafe` hunk and no conflict markers; ci/conflict/rc: `head_verified`.

## Runner (`runner.py`)

```python
class RunSummary(BaseModel): cost_usd: float | None; usage: dict | None; num_turns: int; duration_ms: int;
                             session_id: str; is_error: bool; subtype: str; permission_denials: int; model: str | None
class WorkerRun(BaseModel): output: dict | None; summary: RunSummary | None; error: str | None;
                            auth_failed: bool = False; timed_out: bool = False; rate_limited: bool = False
async def run_worker(*, kind: str, prompt: str, cwd: Path, schema: type[BaseModel], profile: Profile,
                     policy: ToolPolicy | None, system_append: str, timeout_s: float,
                     transcript_path: Path | None = None, tools_enabled: bool = True,
                     client_factory: Callable[[ClaudeAgentOptions], Any] | None = None,
                     interrupt_grace_s: float = 60.0, kill_tree: Callable[[int], None] | None = None) -> WorkerRun
def build_options(*, cwd, schema, profile, policy, system_append, tools_enabled=True, kind="",
                  stderr: Callable[[str], None] | None = None) -> ClaudeAgentOptions
def project_instructions(cwd: Path, *, limit: int = 40_000) -> str   # "## Repository instructions (CLAUDE.md|AGENTS.md)" blocks
def json_schema_for(model: type[BaseModel]) -> dict      # model_json_schema with $refs inlined
def load_prompt(kind: str) -> str                         # workers/_common.md + workers/<kind>.md
```
Options: `cwd=worktree`, `cli_path=provider.cli_path`, `model`/`fallback_model`/`effort` from
provider, `max_turns`, **`setting_sources=[]`** (no filesystem settings: a worktree's
`.claude/settings.json` is repo-controlled -- a fork's PR head controls it -- and its `hooks` /
`apiKeyHelper` / `env` / `enabledPlugins` would run outside the policy; SDK hooks passed in `hooks=`
are unaffected), `system_prompt={"type":"preset","preset":"claude_code","append": system_append +
project_instructions(cwd)}` (the repo's `CLAUDE.md` / `AGENTS.md`, each clipped at 40 000 chars, only
when tools are enabled), `output_format={"type":"json_schema","schema":...}`,
`can_use_tool=policy.can_use_tool`, `hooks=policy.hooks()` (PreToolUse timeout 30 s),
`env=worker_env(profile)`, `permission_mode="default"`, `mcp_servers={}`, `strict_mcp_config=True`,
`allowed_tools=[]` (nothing auto-approved, so can_use_tool sees every ask), `stderr=` a callback (the
CLI's stderr is captured, redacted and written to the transcript; its last 20 lines are appended to
the error when the session ended without a result). `ClaudeAgentOptions.user` is never set (it is
the OS account to run as). Uses `ClaudeSDKClient` with streaming input; after `timeout_s` the
session is interrupted, and cancelled after `interrupt_grace_s`; at the end of *every* session
(finished, failed or timed out) the CLI's whole process tree is killed (`kill_tree`, default
`readiness.kill_tree`, best effort) before disconnect, so a dev server, watcher or background loop
the worker started neither keeps the worktree or its port busy nor runs on after the session.
`ResultMessage.structured_output` validated with the pydantic model; validation failure -> `error`.
`AssistantMessage.error == "authentication_failed"`, `ResultMessage.api_error_status in (401, 403)`
or auth wording in the error text -> `auth_failed=True`. `api_error_status in (429, 529)`,
`AssistantMessage.error in ("rate_limit", "server_error", "billing_error")` or rate/usage-limit
wording -> `rate_limited=True` (cleared when `auth_failed`). Only the session's top-level messages
count (a subagent's rate limit is absorbed by the main session), and a session that ended with valid
structured output was served: the kinds layer never discards such output over a rate-limit signal
(`run_code_worker` and the resolver picker raise `ProviderUnavailable` only for a run without output). Secrets in transcripts and errors are
redacted. `tools_enabled=False` (resolver picker) sets `tools=[]` and appends no repo instructions.

## Discovery / readiness / CODEOWNERS

```python
# discover.py
class Cmd(BaseModel): name: str; command: str; source: str
class Verification(BaseModel): setup: list[Cmd]; verify: list[Cmd]; source: str; dev_command: str | None; dev_port: int | None
def discover(repo_path: Path, override: RepoOverride | None = None) -> Verification   # override -> CLAUDE.md/AGENTS.md -> .github/workflows -> package.json -> Makefile -> pyproject/justfile
    # + `pre-commit run --all-files --show-diff-on-failure` (source .pre-commit-config.yaml) when that file exists and
    #   no discovered command runs pre-commit; an explicit override.verify is kept as given; hook installers never count
def discover_cached(state: State, repo: RepoInfo) -> Verification                      # keyed by HEAD sha
# codeowners.py
def parse_codeowners(text: str) -> list[tuple[str, list[str]]]
def find_codeowners(repo_path: Path) -> Path | None
def load_rules(repo_path: Path) -> list[tuple[str, list[str]]]
def owners_for(path: str, rules) -> list[str]                                         # last matching rule wins
def is_mine(owner, my_teams, my_login) -> bool; def is_foreign(owners, my_teams, my_login) -> bool
def foreign_owned_paths(repo_path: Path, paths, my_teams: list[str], my_login: str) -> list[str]   # what the push policy uses
def foreign_owned_globs(repo_path: Path, my_teams: list[str], my_login: str) -> list[str]          # legacy; not used by the push policy
# readiness.py
class ReadinessCheck(BaseModel): name: str; ok: bool; detail: str = ""; weight: int
class Readiness(BaseModel): repo: str; score: int; checks: list[ReadinessCheck]; missing: list[str]; flaky: list[str]; runtime_s: float; computed_at: str; head_sha: str
def score_repo(repo: RepoInfo, verification: Verification, *, run_commands: bool = True, wt_manager: WorktreeManager | None = None, runner=None) -> Readiness
def readiness_cached(state, repo, ...) -> Readiness
def kill_tree(pid: int) -> None                         # taskkill /T /F on Windows, process group elsewhere
```
Readiness runs commands in a throwaway worktree at `origin/<default>` (never the user's checkout),
setup once, each verify command twice (flakiness), target runtime < 120 s, CLAUDE.md/AGENTS.md with a
verification section, visual repos: dev server responds on its port. Informational only; never blocks.
Without a manager it creates one in a private temp dir (removed afterwards); its git runs with the
same unplantable `core.hooksPath` as every other engine git.

## Evidence (`evidence.py`)

```python
START, END = "<!-- factory:evidence -->", "<!-- /factory:evidence -->"
class EvidenceData(BaseModel): changed_files, sensitive_adjacent, additions, deletions, ac_proofs, verification,
                               screenshots: list[tuple[str, str]] (caption, url), risk_notes, readiness: Readiness | None,
                               min_readiness: int, kind, run_id, head_sha
def render_evidence(data: EvidenceData) -> str          # includes START/END markers
def upsert_evidence(body: str, section: str) -> str     # replace only between markers; append if absent; idempotent
def compose_pr_body(summary: str, issue_key: str | None, evidence_section: str, footer: str) -> str   # "Fixes KEY"
def evidence_asset_path(repo: str, pr_or_branch: str, filename: str) -> str
def evidence_url(repo: str, path: str) -> str           # https://github.com/<repo>/blob/factory/evidence/<path>?raw=true
```

## Effects (`effects.py`)

```python
class Effects(Protocol):
    def push(self, wt: Worktree, branch: str, *, kind: str, pr: int | None) -> str          # returns new remote sha
    def open_draft_pr(self, repo: str, branch: str, base: str, title: str, body: str, *, kind: str, issue_key: str | None) -> PullRequest
    def update_pr_evidence(self, repo: str, number: int, section: str) -> None              # refuses non-factory PRs
    def push_evidence_assets(self, repo: RepoInfo, files: list[tuple[Path, str]]) -> list[str]  # (local, dest path) -> urls; orphan factory/evidence
    def transition_issue(self, issue_id: str, state_type: str) -> None
    def create_sub_issues(self, parent: Issue, specs: list[SubIssueSpec]) -> list[Issue]    # labels repo:<name> if usable, marker otherwise; blocked_by relations
    def update_issue_text(self, issue_id: str, title: str, description: str) -> None
    def remove_label(self, issue_id: str, label: str) -> None
    def write_review(self, repo: str, number: int, markdown: str) -> Path                   # local file, allowed in every mode
class LiveEffects: __init__(profile, paths, state, gh, tracker, audit, git_runner=None)
class RecordingEffects: same ctor; logs "would ..." to the audit log and returns placeholders; never calls gh/tracker writes
class EffectError(RuntimeError); class PushRejected(EffectError); class EffectRefused(EffectError)
def make_effects(profile, *, dry_run: bool, ...) -> Effects
def is_recording(effects) -> bool
def push_refspec(branch: str) -> str                   # "HEAD:refs/heads/<branch>" (validated branch name)
def is_rejected_push(message: str) -> bool             # non-fast-forward / fetch first / stale info
def is_auth_failure(message: str) -> bool              # git credential failures (expired PAT, no prompt, publickey, 401/403)
def link_pushes(state: State, repo: str, branch: str, number: int) -> int   # state.link_pushes
```
Every live effect appends an audit record (`notify.Audit`). Push: `git -c core.hooksPath=<unplantable
path> push origin HEAD:refs/heads/<branch>` from the worktree (never `+`, never `--force*`,
`--no-verify` never used); every other git an effect runs (`ls-remote`, the evidence branch's fetch
and plumbing, the LFS upload) carries the same `-c`: a linked worktree's hooks are the clone's
shared, worker-writable ones. The path is checked before use; when the host cannot provide it the
effect is refused (`EffectError`), never run with hooks enabled. Hooks off also means git-lfs's
`pre-push` never uploads LFS objects: when the pushed change set touches LFS-tracked paths the
objects are uploaded explicitly before the branch push, and a failed (or impossible) upload refuses
the push -- LFS pointers are never pushed without their objects. The pushed commit SHAs are the worktree's first-parent
line `base_sha..HEAD` (a merge of the base contributes the merge commit, not every base commit it
brought in) and are recorded in state; build/ready push before the PR exists (`number` NULL) and
`open_draft_pr` links those rows to the new PR (`link_pushes`) so metrics see the factory's commits.
A rejected push raises `PushRejected` (never retried with force); a git credential failure
(`is_auth_failure`) is an auth stop, not a per-PR failure. Evidence branch: the parent is the remote
tip from `ls-remote` (never `FETCH_HEAD`, which concurrent fetches in the same clone rewrite), the
commit is built with a temporary index (`hash-object --no-filters`), pushed as
`<sha>:refs/heads/factory/evidence` with hooks disabled, retried once on a rejection, all under the
clone's `clone_lock`. `create_sub_issues` decides `repo:<name>` labels per parent team
(`tracker.ensure_label`, a lookup; `labels_exist` is workspace-wide and would count a label another
team owns, which Linear silently drops) and writes the description marker otherwise.

## Notify (`notify.py`)

```python
class Audit:  __init__(path: Path); def write(self, event: str, **fields) -> None    # JSONL, ts + profile
class Delivered(BaseModel): channels: list[str]; errors: list[str]
    @property user_facing -> bool   # toast or slack delivered
class Notifier:
    def __init__(self, profile: Profile, audit: Audit, http: httpx.Client | None = None, toaster=None)
    def notify(self, title: str, body: str, *, target: str = "", kind: str = "", level: str = "info") -> Delivered
    def slack_dm(self, text: str) -> bool        # conversations.open(users=SLACK_USER_ID) + chat.postMessage
    def toast(self, title: str, body: str) -> bool
```

## Kinds (`kinds/<kind>.py`) and dispatch (`dispatch.py`)

```python
class Outcome(BaseModel): status: Literal["done","nothing","kept","flagged","failed","skipped","dry_run"];
                          pushed: bool = False; note: str = ""; pr_url: str | None = None; cost_usd: float | None = None
class Deps: profile, paths, state, gh, tracker, repos, effects, notifier, audit, wt_manager, resolver, run_worker
class KindHandler(Protocol):
    kind: Kind
    async def run(self, action: Action, deps: Deps) -> Outcome
HANDLERS: dict[Kind, KindHandler]
async def run_tick(profile, paths, *, dry_run: bool = False, deps=None, print_fn=print) -> TickReport
async def run_one(profile, paths, action: Action, *, dry_run: bool = False, deps=None, print_fn=print, force: bool = False) -> Outcome
```
Dispatch: global non-blocking file lock (`tick already running` -> exit 0); `providers.scrub_process_env()`;
auth precheck; scan; raise the flags attached to skips (attempts exhausted); group by target;
`asyncio.Semaphore(max_parallel)` across groups; sequential within a group in `ci -> conflict -> rc ->
rv` (PR targets) or `refine -> plan -> build` (issue targets: a refine rewrites the text a plan reads,
a plan splits what a build would do; `actions.order_group`). Before each action the target's kept
worktree is re-checked (an earlier action of the group may just have kept one: the rest of the group
is `skipped`) and a PR action re-fetches the PR (closed or version changed -> `skipped`). After each
action: state per invariant 4, JSONL run record (`runs.jsonl`: ts, profile, kind, target, version,
outcome, pushed, cost_usd, usage, duration_ms, note), one printed line `kind target outcome pushed
note`. Flags go through `Notifier` and `state.add_flag`. `WAVE_STOPPERS = (AuthFailure, GhAuthError,
TrackerAuthError, ProviderUnavailable)` propagate out of handlers: the stop event is set (in-flight
groups finish, nothing new starts), no attempt is recorded, exit 2; auth flags dedupe
`auth:<source>:<date>`, provider flags `provider:<date>`. Any other handler exception -> `failed` +
a "handler crashed" flag.

`run_one` (`factory run`): same lock, scrub and precheck; `force=True` (what the CLI passes) bypasses
the scan filters, but a kept worktree for the target still refuses the run (it would overwrite the
human handoff record) until `factory release`.

## CLI (`cli.py`, typer)

Global options `--profile/-p` (default `personal`), `--profiles-dir`. Commands: `tick [--dry-run]`,
`run <kind> <target> [--dry-run]`, `status`, `flags [--ack ID] [--all]`, `doctor [--repo NAME]`,
`readiness [--repo NAME] [--no-run] [--force]`, `ready <repo> [--dry-run]`, `report [--since 7d]
[--digest]`, `revert-refine <KEY>`, `schedule install|uninstall|show`, `cleanup [--all] [--yes]`,
`release <target> [--remove]`.

`doctor` checks, one line each (`ok` / `warn` / `fail` / `n/a`; exit 1 on any fail; a profile that
does not load is a `profile` fail line): profile + employer rules, `.env` present, provider
credentials, the Claude CLI the SDK will launch (`claude_cli_status`), `gh auth` + the real login vs
`github.login`, `worker gh token` (warn when `GH_WORKER_TOKEN` is absent: workers run gh
unauthenticated; the engine supplies PR/CI/review context), managed repos, unique `dev_port`s,
Linear reachability + factory labels + `repo:<name>` labels (warn only; with `linear.team_keys`
checked per team through `tracker.ensure_label`, team ids learnt from issues carrying a factory
label, `n/a` for a team without one), Slack `auth.test`, toast backend, data dir writable, scheduler
(not installed -> warn; a query the OS cannot answer -> warn; a spec or drift check that raises
`ValueError` / `SchedulerError` -> a `scheduler` fail line, never a traceback that hides the other
checks; installed jobs from another data dir -> warn; Windows tasks still on the no-battery default
-> `scheduler power` warn). `--repo` adds verification
discovery and the static readiness checks.

`release <target>` drops the kept-worktree record so the target is dispatched again; the directory
stays (a detached HEAD: nothing in the clone references it, so it never blocks the next run) until
`--remove` or `factory cleanup`.

## Scheduler (`scheduler/`)

`scheduler/__init__.py`: `install(profile, paths, *, interval_minutes=None, digest=True, runner=None,
os_name=None, uv_path=None) -> ScheduleReport`, `uninstall(profile, paths=None, ...)`, `show(profile,
paths=None)`, `current_os()`, `ScheduleSpec`, `install_drift(spec, show_text) -> list[str]`,
`env_data_dir()`, `SchedulerError`. `windows.py` (schtasks + a `.cmd` wrapper in `paths.scripts`
because `/TR` is limited to 261 chars), `macos.py` (launchd plist in `~/Library/LaunchAgents`,
`StartInterval`), `linux.py` (crontab block between `# software-factory:<profile> begin/end`
markers). Each module exposes pure `render_*` functions (tested) and thin `install`/`uninstall` that
execute them. Two jobs per profile: tick every `interval_minutes`, weekly digest `factory report
--digest --since 7d`. `ScheduleSpec.data_dir` (from `SOFTWARE_FACTORY_DATA_DIR` at install time,
resolved) is emitted by `ScheduleSpec.env()` next to `SOFTWARE_FACTORY_PROFILES_DIR`, so scheduled
jobs use the installing shell's data dir (scheduled jobs inherit no shell environment);
`install_drift` is what doctor uses to warn when the installed jobs disagree with the current shell.
`windows.install` follows every `schtasks /Create` with `windows.render_power_settings(task)`
(`Set-ScheduledTask`: start on battery, do not stop on battery, 6 h execution limit; a failure is a
report note, not an error), and `windows.show` appends the wrapper's `set` env (`wrapper_env`) to the
query output.

## Metrics (`metrics.py`)

```python
def refresh_outcomes(state: State, gh: GitHub) -> int                 # for every touched PR: state, created/merged, rework, reverts
def rework(commits: list[PrCommit], factory_shas: set[str]) -> tuple[int, int]   # (commits, lines) after the factory's first push not pushed by it
def reverted_factory_pushes(commits: list[PrCommit], factory_shas: set[str]) -> list[str]
class Report(BaseModel): per_kind: dict[str, dict]; merged; closed_unmerged; open; median_hours_to_merge;
                         rework_rate; cost_total; cost_per_merged_pr; top_flag_reasons
def build_report(state: State, since_iso: str) -> Report
def render_report(r: Report) -> str
def digest_text(r: Report) -> str
```

## Implemented deviations from the contract (wave 1, all additive)

- `policy.classify_shell(cmd, *, powershell=False)`; `policy.evaluate_push(..., *, conflict_markers=False,
  foreign_owned=None)` and it requires `output.status == "done"`. ToolPolicy also allows NotebookRead,
  LS, BashOutput, KillShell and the CLI's StructuredOutput; denies AskUserQuestion; denies reads of the
  profiles dir / credential files outside the worktree; denies edits under any `.git` component.
  `pre_tool_use_hook` returns `{}` for allowed calls (an explicit allow would skip `can_use_tool`).
- `runner.build_options` raises when tools are enabled without a policy. `load_prompt(kind)` =
  `_common.md + kind.md` and accepts `"resolve"`.
- `worktree.WorktreeManager(root, git=None)` (no hooks directory of its own: `core.hooksPath` is the
  unplantable path); git runner `git(args, *, cwd=None) -> str` raising `WorktreeError`. `.factory/` is excluded via the clone's shared `info/exclude`.
- `paths.ProfilePaths.for_profile(name, root=None)` (`root` replaces `data_root()`); env
  `SOFTWARE_FACTORY_DATA_DIR`; helpers `review_file(repo_name, pr)`, `refinement_file(key)`.
- `providers.worker_env` raises `ProviderError` when `check_credentials` reports problems; the
  dispatcher precheck calls `check_credentials` first (except `--dry-run`, which only warns).
- `notify.Audit(path, profile="")`, `Audit.read()`; `Notifier.notify` always includes `"audit"`;
  `Delivered.user_facing` counts only toast/slack.
- `repos.RepoIndex(repos: list[RepoInfo] | None = None)` for tests.
- `state.State` extras: `handled()`, `last_attempt()`, `pushes(repo=None, number=None)`,
  `link_pushes()`, `cache_delete()`, `close()`; `touched_prs()` rows `{repo, number, factory_opened,
  branch, kind, issue_key, url, pushes}`.
- `sources.github.GitHub.from_profile(profile, run=None)` (honours ignored_logins/bot_reviewers) --
  use it everywhere. Runner signature `run(args, *, cwd=None, input=None, timeout=None)`.
- `scan.scan(..., gh=None | tracker=None)` skips that source; `GhAuthError` / `TrackerAuthError`
  propagate from every scan function; build blockers come back as `Skip("blocked by ...")` from
  `filter_actions`, not from `scan_tracker`. Build actions carry `context["unfinished_blockers"]`.
- `trackers.linear.LinearTracker(profile, http=None, *, api_key=None, sleep=time.sleep)`;
  `create_sub_issue` silently skips labels that do not exist (returned `Issue.labels` = applied);
  `remove_label` is a no-op when absent; `ensure_label(team_id, name)` is a lookup only (a missing
  label raises `TrackerError`; label creation is not an allowed write); module function
  `render_context(issue)`.
- `resolve.Resolution.source` in `label|marker|plan|attachment|siblings|text|picker|none`; a
  label/marker naming an unmanaged repo ends the chain unresolved; caches in
  `state.cache_set("resolve", ...)`.
- `readiness.score_repo(..., workdir=None, dev_probe=None, command_timeout_s=600, runtime_budget_s=120,
  dev_timeout_s=60)`; `readiness_cached(state, repo, verification=None, *, run_commands=True,
  force=False, **kw)`. Static-only scores cap at ~50 (runtime checks count as "not run").
- `evidence`: `sensitive_adjacent_paths(...)`, `screenshot_label(s)`, `EVIDENCE_BRANCH`;
  `render_evidence` has no trailing newline; footer is last in `compose_pr_body`.
- `metrics.refresh_outcomes(state, gh, *, force=False)`, `since_iso_from_spec("7d")`.
- `readiness.ReadinessCheck.not_run` (detail == `NOT_RUN`): the CLI renders such checks as `n/a`
  and `doctor --repo` scores static runs against the static weights only. `readiness.port_in_use`;
  `default_dev_probe` refuses to start when the port is already bound (the probe cannot tell whose
  server answered).
- `cli.claude_cli_status(profile=None)` honours `provider.cli_path` first (must be a file; Windows: a
  native `.exe`/`.com`), then the SDK's own lookup (bundled, `claude` on PATH, the SDK's fixed extra
  locations), and fails when the CLI is older than `claude_agent_sdk._cli_version.__cli_version__`.
  `cli.gh_detected_login()` runs `gh api user` so doctor compares it with `github.login`.

## Kind handlers (wave 2 spec)

Shared module `kinds/base.py`:

```python
class AuthFailure(RuntimeError)          # raised by handlers/dispatcher on gh/tracker/model/git-credential auth failure
class ProviderUnavailable(RuntimeError)  # the model provider refused or could not serve a session (WorkerRun.rate_limited)
class Outcome(BaseModel): status: Literal["done","nothing","kept","flagged","failed","skipped","dry_run"]
                          pushed: bool = False; note: str = ""; pr_url: str | None = None
                          cost_usd: float | None = None; usage: dict | None = None; duration_ms: int | None = None
                          record_handled -> bool  # property: status in ("done","nothing")
@dataclass class Deps: profile, paths, state, gh, tracker | None, repos, effects, notifier, audit,
                       wt_manager, resolver | None, run_worker (callable, defaults to runner.run_worker),
                       now (callable -> datetime, for tests), run_id: str
class KindHandler(Protocol): kind: Kind; async def run(self, action: Action, deps: Deps) -> Outcome
def user_prompt(sections: list[tuple[str, str]]) -> str     # "## Heading\n\nbody" blocks, verbatim headings
def render_pr_section / render_checks_section / render_feedback_section / render_verification_section
    / render_readiness_section / render_candidates_section
async def run_code_worker(action, deps, *, repo, wt, kind, schema, prompt, read_only=False, cwd=None,
                          extra_read_dirs=None) -> WorkerRun
async def finish_code_worker(action, deps, *, repo, wt, output, branch, pr_number, factory_pr, run=None,
                             kind=None, conflict_markers=None, screenshots=None, readiness=None,
                             base_ref=None) -> Outcome
    # shared tail for ci/conflict/rc/build/ready: commits? -> evaluate_push -> (full mode) effects.push
    # -> evidence refresh for factory PRs -> remove worktree; else keep worktree + flag with reasons
class GitDirGuard  # capture(deps, wt) -> snapshot | None; restore(deps) -> list of changes (reverted)
def worker_changes(deps, wt, *, base_ref=None, base_oid="") -> WorkerChanges   # files, additions, deletions, merges_of_base
def push_policy_inputs(deps, repo, changed) -> tuple[list[str], list[str]]     # (sensitive globs, foreign-owned changed files)
def evidence_globs(deps, repo, sensitive_globs) -> list[str]   # sensitive globs + codeowners.foreign_owned_globs: evidence only
def my_login(deps) -> str                    # gh api user, falling back to github.login on a non-auth gh error
def load_evidence / save_evidence / merge_evidence / evidence_from_dict        # state cache "evidence", key "<repo>#<n>"
def keep_worktree_record(deps, target, kind_value, wt, reason, detail) -> str  # never overwrites a live kept record
def detached_head_hint(branch, *, pr=False) -> str
    # "detached HEAD: `git switch -c <branch>-factory` to continue by hand" (+ for PR kinds:
    # ", or review and `git push origin HEAD:refs/heads/<branch>` yourself"); never `<branch>` itself:
    # the clone usually has that local branch already, or has it checked out in its main worktree
def flag(deps, action, reason, detail, *, level="warn", dedupe_key=None) -> None
    # state.add_flag + notifier.notify (notify only when add_flag returns an id: flag-once)
```

`run_code_worker` snapshots the worktree's shared git dir (`GitDirGuard`: the clone's `config`, the
per-worktree `config.worktree`, every file in the hooks dir) before the session and verifies it
after: a worker that planted a hook or rewrote the config gets the change reverted, an error-level
flag (dedupe `git-dir:<target>:<kind>:<version>`) and no usable output (the handler treats the run as
failed; nothing of it is pushed). An `auth_failed` run raises `AuthFailure` and a `rate_limited` run
*without output* `ProviderUnavailable`, both after removing the worktree (no attempt, no leak); the
resolver picker does the same. A run that finished with output is returned even when it is also
`rate_limited` (audited `worker.rate_limited_with_output`): a finished session's work is never
discarded over a rate-limit blip. Whether it finished is judged before the git-dir guard discards a
tampering run's output, so tampering stays a failed attempt, not a provider outage.

`finish_code_worker`: no commits -> `nothing` when the worker says so; `done` without a commit is a
worker bug -> `failed` + a "no commits" flag (the PR would otherwise stay red / conflicting with
nobody told); `blocked` -> `failed` + flag. Commits -> `evaluate_push` over the worker's *own* change
set (`worker_changes`: without merges the `base_sha..HEAD` diff; with a merge of the base
(`origin/<base_ref>` / `base_oid`), only the files whose merge result differs from every parent,
so a conflict resolution is not refused for base-branch files the PR never touched; a merge of
anything else counts whole) with `push_policy_inputs` (globs + CODEOWNERS-foreign files for the real
login). Refused -> `state.keep_worktree` (via `keep_worktree_record`) + flag with the reasons, the
path and the detached-HEAD hint, `kept`; a rejected push likewise; a git credential failure ->
`AuthFailure`; any other push error keeps the worktree, `failed`. Pushed -> on factory-opened PRs
the evidence section is refreshed from the stored evidence merged with the new push (`merge_evidence`
keeps the build's AC -> test map, screenshots and readiness warning; files / risk notes are unioned,
verification runs accumulate) and the result is stored again; worktree removed. The evidence's
`sensitive_adjacent` list (here and in build's `publish_branch`) is computed over `evidence_globs`:
the sensitive globs plus every CODEOWNERS pattern owned by someone else, so a change inside another
team's area is pointed out to the reviewer; those pattern-level globs never reach `evaluate_push`,
which applies CODEOWNERS per changed path.

`policy.ToolPolicy` gains `read_only: bool = False` (deny Edit/Write/MultiEdit/NotebookEdit with a
message; shell still classified). Used by rv, refine, plan.

Per kind. Headings in the user prompt must match what `workers/*.md` reference verbatim:
`## Action`, `## Pull request`, `## Failing checks`, `## Review feedback`, `## Issue`,
`## Verification commands`, `## Repo readiness`, `## Candidate repos`.

- **PR kinds common pre-flight**: `repo = deps.repos.get(action.repo)` (missing -> skipped);
  `pr = gh.pr_details(repo, n)`; closed/merged -> `skipped`; version no longer matches the current
  state (ci: head sha; conflict: head:base; rc: latest feedback ts over all feedback; rv: head sha) ->
  `skipped("stale")`; fork head on ci/conflict/rc -> flag + `skipped` (cannot push to forks).
  `wt = wt_manager.for_pr(repo, pr)`. Verification via `discover.discover_cached`. Kept-worktree
  flags carry the PR head branch in the detached-HEAD hint (`pr=True`: switch to `<branch>-factory`,
  or review and push `HEAD:refs/heads/<branch>` by hand).
- **ci**: failing checks + `gh.failed_log(run_id)` per failing Actions run (dedupe run ids; non-Actions
  checks listed with details_url only), and per failing workflow the latest run on the PR's base
  branch, fetched read-only with the engine's own gh (`gh run list -R <repo> --branch <base>
  --workflow <name> --limit 1 --json conclusion,headSha,url,createdAt`; through the client's
  `latest_workflow_run` when it has one, else its runner) and rendered under `## Failing checks` as
  `base branch: <conclusion> at <sha> (<url>)` -- the evidence for an `inherited` classification,
  which workers cannot fetch themselves when they run gh unauthenticated. A failed lookup is a
  `base branch: not checked (...)` note; `GhAuthError` propagates (worktree removed). Output `CiOutput`. If no commits and every failure classified
  flaky/inherited/infra -> `done` (handled at this head), notify with the classifications (dedupe by
  target+version). Else `finish_code_worker`.
- **conflict**: prompt includes base branch name; worker merges `origin/<base>`. `evaluate_push(...,
  conflict_markers=wt_manager.has_conflict_markers(wt))` over the merge's own resolutions
  (`base_ref`). `all_resolved == False` -> keep worktree + flag (outcome `kept`, not recorded
  handled; the kept-worktree filter blocks re-dispatch).
- **rc**: feedback items from `context["feedback_ids"]` (fall back to `pr.open_feedback` for a manual
  run); no item at all (e.g. only resolved-thread feedback on a manual run) -> `nothing` ("no open
  feedback", recorded) before any worktree or session. `RcOutput`. Any `skipped` disposition -> keep + flag. `status == "nothing"` -> `nothing`
  (recorded), notify the dispositions.
- **rv**: read-only policy. `RvOutput`. `effects.write_review(repo, n, markdown)` (allowed in every
  mode) -> notify "review draft ready: <path>" -> `done`. Worktree always removed.
- **build** (issue): `issue = tracker.get_issue(action.issue_id)`; re-check label + `unstarted` +
  no unfinished blockers (else `skipped`); while a refine or plan label is *pending* the build waits
  (`skipped`, no attempt: those kinds run first) and flags once which label holds it (info, dedupe
  `build-waits:<KEY>`). A label is pending only when its kind is enabled (a disabled kind never
  removes it), and the plan label only while the issue has no children and plan has not recorded
  `nothing` for the issue's current version (`state.handled(KEY, "plan", updated_at)`). `resolution = await
  resolver.resolve(issue)`; unresolved -> flag once (dedupe `resolve:<KEY>:<updated_at>`) with the
  best guess, `skipped` (no attempt spent). Branch = `issue.branch_name` or `factory/<key-lower>`;
  refuse if an open PR already exists for it (`gh.find_open_pr_for_branch`) -> `skipped`. A pending
  PR record (`state` cache `pr_pending`: the branch was pushed but `gh pr create` failed) is resumed
  first: the PR is opened from the record without a new worker session (the record is dropped when
  the branch is gone from origin). `wt = for_new_branch` (detached; the `## Action` section says the
  worktree is a detached HEAD at `origin/<branch>` or `origin/<default>` and that `HEAD` is pushed
  as-is). Prompt: `## Action`, `## Issue` (`render_context`), `## Verification commands`,
  `## Repo readiness` (`readiness_cached(..., run_commands=False)` unless a full cached result
  exists), visual block (dev command + port + routes) when `override.visual`. `BuildOutput`. No
  commits: `nothing` is recorded and flagged once (info, dedupe `build-nothing:<target>:<version>`:
  the version is the issue id, so only `factory run build KEY` retries); `done` without commits /
  `blocked` -> `failed` + flag. Push policy -> push branch -> screenshots:
  `effects.push_evidence_assets` -> evidence section -> `compose_pr_body(pr_summary, KEY, section,
  profile.disclosure_footer)` -> `effects.open_draft_pr` (registers the PR, links the pushes; a
  failure stores the `pr_pending` record and is an attempt) -> evidence data saved for the upkeep
  kinds -> `effects.transition_issue(issue.id, "started")` if not already started -> notify "draft PR
  opened" -> `done`. Readiness below `min_readiness` -> warning in evidence + the notify body
  suggests `factory ready <repo>`.
- **ready** (manual, target = repo): version = origin default HEAD sha; branch
  `factory/ready-<YYYYMMDD>`; `ReadyOutput`; prompt includes the full readiness report and the same
  detached-HEAD lines as build; push + draft PR (no issue key) -> `done`; shares build's
  `publish_branch` and pending-PR resume.
- **refine** (issue): if `state.get_refinement(KEY)` exists with status in pending|notified|edited ->
  resume the remaining effects without re-running the worker, unless the issue's title/description
  changed since the proposal was saved or the label is gone (the stale proposal is `reverted` +
  flagged and the worker runs afresh). Else optional read-only worktree of the resolved repo
  (`for_default_branch`; resolution failure is fine -> cwd = `paths.root/"scratch"`). `RefineOutput`.
  `nothing` -> the refine label is removed (allowed write; otherwise every later edit of the issue
  would cost another session), an info flag says so (dedupe `refine-nothing:<KEY>:<version>`), and
  `nothing` is recorded. Sequence (invariant 7): `state.save_refinement` (pending) + write
  `refinements/<KEY>.md` (original, new, unified diff, open questions, split) ->
  `notifier.notify(...)` with diff summary + open questions + `factory revert-refine <KEY>`; if
  `not delivered.user_facing` -> flag + `failed` (no edit) -> status `notified` ->
  `effects.update_issue_text` -> `edited` -> `effects.remove_label(label_refine)` -> `applied` -> `done`.
  `revert_refine(deps, key)` restores the original via effects, status `reverted`, notifies.
- **plan** (issue): skip if the issue has children (`nothing`). cwd = `paths.root/"scratch"` with
  `extra_read_dirs=[repos.root]`, read-only. Prompt: `## Issue` + `## Candidate repos` (slug + summary).
  `PlanOutput`; `nothing` -> recorded and flagged once per issue (info, dedupe `plan-nothing:<target>`:
  the plan label stays on and does nothing; the flag says to remove the plan label, then label
  `factory:build` to build it as one PR -- in that order, because labelling edits the issue and a plan
  label left on would re-plan the new version first). Validate every `spec.repo` is managed and `blocked_by`
  indices are in range and acyclic (else flag + `failed`). `effects.create_sub_issues(parent, specs)`
  (never adds `factory:build`) -> `state.set_resolution(child.id, repo, 1.0, "plan")` + resolver
  cache (the resolver honours it after later edits of the child) ->
  `effects.transition_issue(parent, "started")` -> notify "plan ready for review" with the list -> `done`.

Dispatcher (`dispatch.py`): see "Kinds and dispatch" above. `TickReport` has `lines`, `outcomes`,
`skipped`, `errors`, `auth_error`, `provider_error`, `exit_code` (0 ok, 1 some action failed, 2 auth
failure or provider unavailable). Observe mode: non-rv kinds produce `dry_run` outcomes (audit "would
run"), never call the worker. `--dry-run`: scan + filter + print, no workers, no state writes, no
flags: provider credentials are not required (a problem is printed as a warning), an auth failure is
printed and exits 2 but is not flagged, skip flags are not raised. Precheck order: lock ->
`scrub_process_env` -> `check_credentials` (skipped for `--dry-run`) -> `gh.auth_check` -> tracker
`ping` (only when an issue kind is enabled and Linear is enabled) -> scan -> skip flags. After
dispatch (unless an auth failure stopped it), `metrics.refresh_outcomes` (errors logged, never fatal).
