"""Tool allow/deny policy for workers + engine-side push policy.

Workers propose, the engine acts (architecture invariant 1). Every outward write a worker could
attempt through its tools is denied here, in *both* the ``can_use_tool`` callback and a ``PreToolUse``
hook, with a message that tells the model what to do instead so it adapts rather than hangs.

Shell classification is structural: the command is tokenised into simple commands (splitting on
``; && || | & newlines``, ``$( )``, backticks, parentheses), env assignments and wrappers (``env``,
``sudo``, ``xargs``, ``uv run`` ...) are stripped, nested shells (``bash -c``, ``pwsh -Command``,
``cmd /c``, ``-EncodedCommand``) are unwrapped recursively, and each resulting ``argv`` is matched
against the rules. Unparseable input fails closed only when it mentions a dangerous token.

Write targets are checked too: output redirections (``>``, ``>>``, ``2>``, ``&>``, PowerShell
``*>``) and the destinations of ``tee`` / ``cp`` / ``mv`` / ``install`` / ``ln`` / ``touch`` /
``chmod`` / ``rm`` / ``sed -i`` / ``dd of=`` / ``cd`` and their PowerShell cmdlets are denied when
the path is inside a git directory (a ``.git`` component, ``<x>.git/...``, ``$GIT_DIR``, a
``$(git rev-parse --git-dir|--git-common-dir|--git-path ...)`` substitution, or a variable assigned
one of those) or is a git/gh/ssh configuration or credential file: a hook or config planted there
would run with the engine's credentials. Reads of ``.git`` stay allowed.
"""

from __future__ import annotations

import base64
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pathspec
from claude_agent_sdk import HookMatcher, PermissionResultAllow, PermissionResultDeny
from pydantic import BaseModel, Field

from . import codeowners
from .actions import Kind
from .config import Profile, resolve_profiles_dir
from .workers.schemas import CodeWorkerOutput

# Seconds (``HookMatcher.timeout`` is in seconds; the SDK default is 60). The hook is an in-process
# classification that finishes in microseconds; the bound only matters when something is wedged.
PRE_TOOL_USE_HOOK_TIMEOUT_S = 30.0

if TYPE_CHECKING:  # owned by other modules; only needed for type hints
    from .repos import RepoInfo


class ToolDecision(BaseModel):
    allow: bool
    reason: str = ""


def _allow() -> ToolDecision:
    return ToolDecision(allow=True)


def _deny(reason: str) -> ToolDecision:
    return ToolDecision(allow=False, reason=reason)


# ------------------------------------------------------------------------------ deny messages
MSG_PUSH = (
    "git push is not available to workers. Pushing is done by the engine after you finish; "
    "commit locally and report push_ok (with push_reasons) in your structured output."
)
MSG_GH_WRITE = (
    "gh write commands are not available to workers (no comments, reviews, merges, PR/issue creation "
    "or edits). Read-only gh (pr view/list/diff/checks/status, run view/list/watch, issue view/list, "
    "repo view, release view/list, search, auth status, api GET) is allowed. Describe what should "
    "happen in your structured output; the engine performs every outward write."
)
MSG_GH_API_WRITE = (
    "gh api is allowed for GET only (no -X/--method other than GET, no -f/-F/--field/--raw-field/"
    "--input, no graphql). Report what needs changing in your structured output instead."
)
MSG_HTTP_WRITE = (
    "HTTP requests with a non-GET method or a body are not available to workers. Fetch read-only "
    "data with GET if needed; any write to GitHub, Linear or Slack is performed by the engine."
)
MSG_TRACKER_HOST = (
    "Linear/Slack APIs are not reachable from workers. The engine already placed the issue context "
    "in your prompt; put anything that must be sent back (comments, state changes, open questions) "
    "in your structured output."
)
MSG_PUBLISH = "Publishing packages/images is never done by workers. Commit locally and report in your output."
MSG_REMOTE = (
    "Changing git remotes, credentials, aliases or hooks is not allowed. Work against the existing "
    "origin; the engine pushes for you."
)
MSG_AUTH_TOKEN = "Printing or changing auth credentials is not allowed. Use `gh auth status` to check auth."
MSG_ENCODED = "Encoded shell commands cannot be inspected and are denied; run the command in plain text."
MSG_ENV = (
    "Setting environment variables that change how git, gh or ssh behave (GIT_*, GH_*, SSH_*, "
    "EDITOR/PAGER/BROWSER, GITHUB_TOKEN) is not allowed; they can redirect credentials or run "
    "commands. Run the plain command without the assignment; the engine already configured git and gh."
)
MSG_ASK_USER = (
    "There is no human in this session. Decide yourself using the prompt, and record any open "
    "questions in your structured output."
)
MSG_READ_ONLY = (
    "This worker is read-only: Edit, Write, MultiEdit and NotebookEdit are not available. Read the "
    "code and report your findings in the structured output instead of changing files."
)


def _git_dir_write_message(what: str) -> str:
    return (
        f"Writing to or working inside {what} is not allowed: hooks and config there run commands "
        "with the engine's credentials, and the clone is shared with the user's own checkout. "
        "Reading .git is fine. Change tracked files with the Edit tool and repository state with "
        "plain git commands; anything that would need a hook, config or credential change goes in "
        "your structured output."
    )


_DANGEROUS_TOKEN_RE = re.compile(
    r"(push|\bgh\b|curl|wget|\biwr\b|\birm\b|invoke-webrequest|invoke-restmethod|publish|upload|"
    r"send-pack|twine|linear|slack|remote|credential|encodedcommand|\.git\b|hookspath|"
    r"--git-(dir|path|common-dir))",
    re.IGNORECASE,
)
_FORBIDDEN_HOSTS = ("api.linear.app", "slack.com/api", "hooks.slack.com")
_PS_HTTP_METHOD_CALL_RE = re.compile(
    r"\.(upload(string|data|file|values)|(post|put|patch|delete|send)async)\s*\(", re.IGNORECASE
)
# PowerShell / .NET ways of assigning an environment variable that the tokenizer does not see as
# ``VAR=value``: ``$env:NAME = x``, ``${env:NAME}=x``, ``Set-Item env:NAME x``, ``Set-Item -Path
# env:NAME``, ``[Environment]::SetEnvironmentVariable("NAME", ...)``, ``setx NAME x``.
_PS_ENV_ASSIGN_RE = re.compile(
    r"(?:\$\{?env:(?P<a>[A-Za-z_][A-Za-z0-9_]*)\}?\s*=(?!=)"
    r"|set-item\s+(?:-path\s+)?['\"]?env:(?P<b>[A-Za-z_][A-Za-z0-9_]*)"
    r"|setenvironmentvariable\s*\(\s*['\"](?P<c>[A-Za-z_][A-Za-z0-9_]*)['\"]"
    r"|\bsetx(?:\.exe)?\s+(?P<d>[A-Za-z_][A-Za-z0-9_]*)\b)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------------- tokenizer
class _ParseError(Exception):
    pass


# Sentinels the tokenizer leaves in ``argv``. ``\x00`` cannot come from a quoted word.
_REDIR_OUT = "\x00>"  # an output redirection; its target is the next word
_REDIR_IN = "\x00<"  # an input redirection / heredoc delimiter; its target is the next word
_GITDIR_MARK = "\x00gitdir\x00"  # part of a word produced by ``$(git rev-parse --git-dir ...)``
_GIT_DIR_QUERY_OPTS = ("--git-dir", "--git-common-dir", "--git-path", "--absolute-git-dir")


def _is_git_dir_query(argv: list[str]) -> bool:
    """``git [opts] rev-parse --git-dir|--git-common-dir|--git-path|--absolute-git-dir ...``."""
    if "rev-parse" not in argv or not any(_basename(t) == "git" for t in argv):
        return False
    return any(t.startswith(_GIT_DIR_QUERY_OPTS) for t in argv)


def _find_matching_paren(text: str, start: int, *, powershell: bool = False) -> int:
    """``start`` points just past ``$(`` (or a PowerShell ``(``); return the index of the matching ``)``."""
    depth = 1
    i = start
    quote: str | None = None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\" and quote == '"' and not powershell:
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise _ParseError("unbalanced $( )")


def _split_commands(text: str, *, powershell: bool = False) -> list[list[str]]:
    """Split a shell command line into simple commands (lists of words).

    Handles POSIX quoting (and PowerShell quoting when ``powershell`` is true, where the backtick is
    an escape character rather than command substitution). Command substitutions and subshells are
    parsed recursively and their simple commands appended to the result.
    """
    commands: list[list[str]] = []
    argv: list[str] = []
    word: list[str] = []
    has_word = False
    i = 0
    n = len(text)

    def end_word() -> None:
        nonlocal word, has_word
        if has_word:
            argv.append("".join(word))
        word = []
        has_word = False

    def end_command() -> None:
        nonlocal argv
        end_word()
        if argv:
            commands.append(argv)
        argv = []

    def substitute(inner: str) -> None:
        nonlocal has_word
        inner_commands = _split_commands(inner, powershell=powershell)
        commands.extend(inner_commands)
        if any(_is_git_dir_query(c) for c in inner_commands):
            word.append(_GITDIR_MARK)  # the surrounding word remembers it names a git dir
        has_word = True  # the substitution leaves a (placeholder) word behind

    def redirect() -> None:
        """``i`` is at ``>``: emit ``_REDIR_OUT`` (its target is the next word). A glued fd (``2>``,
        PowerShell ``*>``) is part of the operator, not a word; ``>&2`` / ``>&-`` dup an fd and have
        no file target; bash ``>&file`` is ``&>file``."""
        nonlocal i, word, has_word
        current = "".join(word) if has_word else ""
        if has_word and (current.isdigit() or (powershell and current == "*")):
            word, has_word = [], False
        else:
            end_word()
        i += 1
        if i < n and text[i] in ">|":
            i += 1
        if i < n and text[i] == "&":
            j = i + 1
            while j < n and (text[j].isdigit() or text[j] == "-"):
                j += 1
            if j > i + 1:
                i = j
                return
            i += 1
        argv.append(_REDIR_OUT)

    while i < n:
        ch = text[i]
        if ch == "'":
            j = text.find("'", i + 1)
            if j < 0:
                raise _ParseError("unbalanced single quote")
            word.append(text[i + 1 : j])
            has_word = True
            i = j + 1
            continue
        if ch == '"':
            i += 1
            has_word = True
            while True:
                if i >= n:
                    raise _ParseError("unbalanced double quote")
                c = text[i]
                if c == '"':
                    i += 1
                    break
                if c == "\\" and not powershell and i + 1 < n and text[i + 1] in '"\\$`\n':
                    if text[i + 1] != "\n":
                        word.append(text[i + 1])
                    i += 2
                    continue
                if c == "`" and powershell and i + 1 < n:
                    word.append(text[i + 1])
                    i += 2
                    continue
                if c == "$" and text.startswith("$(", i):
                    j = _find_matching_paren(text, i + 2, powershell=powershell)
                    substitute(text[i + 2 : j])
                    i = j + 1
                    continue
                if c == "`" and not powershell:
                    j = text.find("`", i + 1)
                    if j < 0:
                        raise _ParseError("unbalanced backtick")
                    substitute(text[i + 1 : j])
                    i = j + 1
                    continue
                word.append(c)
                i += 1
            continue
        if ch == "\\" and not powershell:
            if i + 1 < n:
                nxt = text[i + 1]
                if nxt.isalnum() or nxt in ":.":
                    word.append(ch)  # keep Windows path separators (``C:\x\git.exe``, ``\.gitconfig``) inspectable
                    word.append(nxt)
                    has_word = True
                elif nxt != "\n":
                    word.append(nxt)
                    has_word = True
                i += 2
            else:
                i += 1
            continue
        if ch == "`":
            if powershell:  # escape character (also line continuation)
                if i + 1 < n and text[i + 1] != "\n":
                    word.append(text[i + 1])
                    has_word = True
                i += 2
                continue
            j = text.find("`", i + 1)
            if j < 0:
                raise _ParseError("unbalanced backtick")
            substitute(text[i + 1 : j])
            i = j + 1
            continue
        if text.startswith("$(", i):
            j = _find_matching_paren(text, i + 2, powershell=powershell)
            substitute(text[i + 2 : j])
            i = j + 1
            continue
        if ch == "#" and not has_word:
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if ch in " \t\r":
            end_word()
            i += 1
            continue
        if text.startswith("{}", i):  # literal ``{}`` (xargs -I {}), not a command group
            word.append("{}")
            has_word = True
            i += 2
            continue
        if text.startswith("${", i):  # ``${VAR}`` / ``${env:VAR}`` is part of the word, not a group
            j = text.find("}", i + 2)
            if j < 0:
                raise _ParseError("unbalanced ${")
            word.append(text[i : j + 1])
            has_word = True
            i = j + 1
            continue
        if ch == "&" and not powershell and text.startswith("&>", i):
            end_word()
            i += 1
            redirect()
            continue
        if ch == "(" and powershell:  # ``(git rev-parse --git-dir)``: subexpression inside a word
            j = _find_matching_paren(text, i + 1, powershell=True)
            substitute(text[i + 1 : j])
            i = j + 1
            continue
        if ch in ";\n&|(){}":
            end_command()
            i += 1
            continue
        if ch == ">":
            redirect()
            continue
        if ch == "<" and not powershell:
            end_word()
            argv.append(_REDIR_IN)
            i += 1
            continue
        word.append(ch)
        has_word = True
        i += 1
    end_command()
    return commands


# ---------------------------------------------------------------------------- argv helpers
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Variables that make git / gh / ssh run a command or read credentials from somewhere else
# (GIT_CONFIG_* and GIT_CONFIG_GLOBAL reach every git setting the ``-c`` rules deny, GIT_EDITOR /
# GIT_SSH_COMMAND / GH_PAGER run a shell command, GH_CONFIG_DIR / GH_TOKEN swap credentials, GIT_DIR
# points git at another checkout). A ``VAR=value cmd`` prefix is live syntax in the worker's Bash.
_DANGEROUS_ENV_PREFIXES = ("GIT_", "GH_", "SSH_")
_DANGEROUS_ENV_EXACT = {"GITHUB_TOKEN", "EDITOR", "VISUAL", "BROWSER", "PAGER"}
# Harmless members of the dangerous prefixes a worker may legitimately set.
_HARMLESS_ENV = {
    "GIT_TERMINAL_PROMPT", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_DATE", "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL", "GIT_COMMITTER_DATE", "GIT_OPTIONAL_LOCKS", "GIT_LFS_SKIP_SMUDGE",
    "GIT_MERGE_AUTOEDIT", "GIT_ADVICE", "GIT_FLUSH", "GIT_CURL_VERBOSE", "GIT_TRACE", "GIT_TRACE2",
    "GIT_TRACE_PERFORMANCE", "GIT_TRACE_SETUP", "GIT_TRACE_PACKET", "GIT_TRACE_CURL",
    "GH_REPO", "GH_PROMPT_DISABLED", "GH_NO_UPDATE_NOTIFIER", "GH_FORCE_TTY", "GH_DEBUG", "GH_COLOR",
}
# Pager variables are fine when they disable paging.
_PAGER_ENV = {"PAGER", "GIT_PAGER", "GH_PAGER"}
_SAFE_PAGER_VALUES = {"", "cat", "true", "false", ":"}


class _Denied(Exception):
    """Raised inside the argv helpers to abort classification with a denial."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _env_key_is_dangerous(key: str, value: str | None) -> bool:
    """``value=None`` means unknown (fail closed for pager-type keys)."""
    name = key.upper()
    if name in _PAGER_ENV:
        return value is None or value.strip() not in _SAFE_PAGER_VALUES
    if name in _HARMLESS_ENV:
        return False
    return name in _DANGEROUS_ENV_EXACT or name.startswith(_DANGEROUS_ENV_PREFIXES)


def _check_env_assignment(token: str) -> None:
    """``VAR=value``: deny dangerous keys; classify values that look like commands (``X='git push'``
    followed by ``eval "$X"``) so the assignment itself is already refused."""
    key, _, value = token.partition("=")
    if _env_key_is_dangerous(key, value):
        raise _Denied(MSG_ENV)
    if _DANGEROUS_TOKEN_RE.search(value):
        decision = _classify(value)
        if not decision.allow:
            raise _Denied(decision.reason)


# ------------------------------------------------------------------------- write targets
# A write whose target is a git directory (hooks run with the engine's credentials on push /
# worktree add / commit; config can redirect remotes and credentials) or a git/gh/ssh config or
# credential file read by the engine's own git and gh.
_GIT_ENV_REF_RE = re.compile(r"(?:\$\{?|\$env:|%)(GIT_DIR|GIT_COMMON_DIR)\b", re.IGNORECASE)
_VAR_REF_RE = re.compile(r"(?:\$\{?|\$env:|%)([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_ASSIGN_RE = re.compile(r"^\$?\{?(?:env:)?([A-Za-z_][A-Za-z0-9_]*)\}?=(.*)$", re.DOTALL)
_PATH_SPLIT_RE = re.compile(r"[\\/]+")
_PROTECTED_BASENAMES = {".gitconfig", ".git-credentials", ".netrc", "_netrc", "hosts.yml"}
# Commands whose *last* positional (or ``-t DIR`` / ``-Destination``) is the write target.
_WRITERS_DEST_LAST = {"cp", "mv", "install", "ln", "rsync", "copy", "move", "copy-item", "cpi", "move-item", "mi"}
_WRITERS_TARGET_DIR_OPT = {"cp", "mv", "install", "ln"}
# Commands whose every positional path is created, overwritten, deleted or entered.
_WRITERS_ALL_PATHS = {
    "tee", "touch", "chmod", "chown", "chgrp", "mkdir", "rmdir", "rm", "unlink", "truncate", "shred",
    "cd", "pushd", "chdir", "set-location", "sl",
    "out-file", "set-content", "sc", "add-content", "ac", "clear-content", "clc", "new-item", "ni",
    "remove-item", "ri", "del", "erase", "rd", "set-itemproperty", "sp",
}
_WRITERS_IN_PLACE = {"sed", "perl"}
_PS_CMDLETS = {
    "out-file", "set-content", "sc", "add-content", "ac", "clear-content", "clc", "new-item", "ni",
    "remove-item", "ri", "copy-item", "cpi", "move-item", "mi", "set-location", "sl", "set-itemproperty", "sp",
}
_PS_PATH_PARAMS = ("path", "literalpath", "filepath", "pspath", "destination")


def _protected_write_target(raw: str, tainted: frozenset[str] = frozenset()) -> str | None:
    """Why ``raw`` (a path a command would write to or enter) is off limits; ``None`` when it is not."""
    if _GITDIR_MARK in raw:
        return "the git directory (a `git rev-parse --git-dir/--git-common-dir/--git-path` substitution)"
    if _GIT_ENV_REF_RE.search(raw):
        return "the git directory ($GIT_DIR / $GIT_COMMON_DIR)"
    for match in _VAR_REF_RE.finditer(raw):
        if match.group(1).upper() in tainted:
            return f"the git directory (via ${match.group(1)})"
    parts = [p for p in _PATH_SPLIT_RE.split(raw.strip().strip("'\"")) if p not in ("", ".")]
    low = [p.lower() for p in parts]
    for idx, part in enumerate(low):
        if part == ".git":
            return "the git directory (.git)"
        if part.endswith(".git") and idx + 1 < len(low):
            return f"a git directory ({parts[idx]}/)"
        if part == ".ssh":
            return "ssh configuration (.ssh/)"
        if part == ".config" and idx + 1 < len(low) and low[idx + 1] in ("git", "gh"):
            return f"git/gh configuration (.config/{low[idx + 1]}/)"
    if low and low[-1] in _PROTECTED_BASENAMES:
        return f"git/gh credentials or configuration ({parts[-1]})"
    return None


def _tainted_names(commands: list[list[str]]) -> frozenset[str]:
    """Variables assigned a git-dir path anywhere on the line (``H=.git/hooks``, ``H=$(git rev-parse
    --git-path hooks)``, PowerShell ``$h = (git rev-parse --git-dir)`` / ``$h = git rev-parse
    --git-dir``), so that ``> $H/pre-commit`` is caught as well."""
    names: set[str] = set()
    for argv in commands:
        for idx, tok in enumerate(argv):
            match = _ASSIGN_RE.match(tok)
            if match:
                if _protected_write_target(match.group(2), frozenset(names)) is not None:
                    names.add(match.group(1).upper())
            elif tok == "=" and idx > 0 and argv[idx - 1].startswith("$"):
                var = _ASSIGN_RE.match(argv[idx - 1] + "=")
                rest = argv[idx + 1 :]
                if var and (_is_git_dir_query(rest) or _protected_write_target(" ".join(rest), frozenset(names))):
                    names.add(var.group(1).upper())
    return frozenset(names)


def _strip_redirections(argv: list[str], tainted: frozenset[str]) -> list[str]:
    """Drop redirection sentinels and their targets; deny an output redirection into a git dir."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in (_REDIR_OUT, _REDIR_IN):
            target = argv[i + 1] if i + 1 < len(argv) and argv[i + 1] not in (_REDIR_OUT, _REDIR_IN) else None
            if tok == _REDIR_OUT and target is not None:
                what = _protected_write_target(target, tainted)
                if what is not None:
                    raise _Denied(_git_dir_write_message(what))
            i += 2 if target is not None else 1
            continue
        out.append(tok)
        i += 1
    return out


def _ps_param(tok: str) -> tuple[str, str | None] | None:
    """PowerShell ``-Name`` / ``-Name:value`` -> (name lowercased, inline value)."""
    if not tok.startswith("-") or tok.startswith("--") or len(tok) < 2:
        return None
    name, has_value, value = tok[1:].partition(":")
    return name.lower(), (value if has_value else None)


def _write_targets(argv: list[str]) -> list[str]:
    """Paths ``argv`` would create, overwrite, move onto, delete or enter (``[]`` for other commands)."""
    name = _basename(argv[0])
    args = argv[1:]
    if name in _WRITERS_IN_PLACE:
        in_place = any(
            t.startswith("--in-place") or (t.startswith("-") and not t.startswith("--") and "i" in t[1:]) for t in args
        )
        return [t for t in args if not t.startswith("-")] if in_place else []
    if name == "dd":
        return [t[3:] for t in args if t.lower().startswith("of=")]
    if name not in _WRITERS_DEST_LAST and name not in _WRITERS_ALL_PATHS:
        return []
    positional: list[str] = []
    named: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--":
            positional.extend(args[i + 1 :])
            break
        if tok.startswith("-") and len(tok) > 1:
            opt, has_eq, value = tok.partition("=")
            if name in _WRITERS_TARGET_DIR_OPT and opt in ("-t", "--target-directory"):
                if not has_eq:
                    value = args[i + 1] if i + 1 < len(args) else ""
                    i += 1
                named.append(value)
            elif name in _PS_CMDLETS and (param := _ps_param(tok)) is not None:
                pname, inline = param
                if len(pname) >= 2 and any(p.startswith(pname) for p in _PS_PATH_PARAMS):
                    if inline is None:
                        inline = args[i + 1] if i + 1 < len(args) else ""
                        i += 1
                    named.append(inline)
            i += 1
            continue
        positional.append(tok)
        i += 1
    if name == "install" and any(t in ("-d", "--directory") for t in args):
        return positional + named  # ``install -d DIR...``: every positional is created
    if name in _WRITERS_DEST_LAST:
        return named or (positional[-1:] if len(positional) >= 2 else [])
    return positional + named


def _check_write_targets(argv: list[str], tainted: frozenset[str]) -> None:
    for target in _write_targets(argv):
        what = _protected_write_target(target, tainted)
        if what is not None:
            raise _Denied(_git_dir_write_message(what))


_ENV_SETTERS = {"export", "declare", "typeset", "setenv", "set", "setx"}


def _check_env_setter(argv: list[str]) -> None:
    """``export VAR=value`` / ``declare -x VAR=value`` / ``setx VAR value`` (cmd.exe / PowerShell)."""
    name = _basename(argv[0])
    if name not in _ENV_SETTERS:
        return
    rest = [t for t in argv[1:] if not t.startswith("-")]
    if name == "setx":
        if rest and _env_key_is_dangerous(rest[0], None):
            raise _Denied(MSG_ENV)
        return
    for tok in rest:
        if _ENV_ASSIGN_RE.match(tok):
            _check_env_assignment(tok)
        elif name == "setenv" and rest and tok == rest[0] and _env_key_is_dangerous(tok, None):
            raise _Denied(MSG_ENV)
_WRAPPER_VALUE_OPTS: dict[str, set[str]] = {
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "sudo": {"-u", "--user", "-g", "--group", "-C", "-D", "-h", "--host", "-p", "--prompt", "-r", "-t", "-U"},
    "doas": {"-u", "-C"},
    "xargs": {"-I", "-i", "-n", "-P", "-L", "-s", "-d", "-E", "-a", "--arg-file", "--max-args", "--max-procs"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "uv": {"-p", "--python", "-w", "--with", "--with-editable", "--with-requirements", "--group",
           "--extra", "--env-file", "--directory", "--project", "--package", "--index", "--find-links",
           "--only-group", "--no-group", "--exclude-newer", "--cache-dir", "--config-file"},
    "npx": {"-p", "--package", "-c", "--call", "--shell"},
    "pnpm": {"-C", "--dir", "--filter", "-F"},
    "npm": {"-w", "--workspace", "--package", "-p", "-c", "--call", "--prefix", "-C"},
    "yarn": {"--cwd"},
    "bun": {"--cwd"},
    "poetry": {"-C", "--directory"},
    "pdm": {"-p", "--project"},
    "hatch": {"-e", "--env"},
    "conda": {"-n", "--name", "-p", "--prefix", "--cwd"},
    "cmd": set(),
    "watch": {"-n", "--interval", "-d", "--differences"},
    "stdbuf": {"-i", "-o", "-e"},
}
# wrapper -> words that must follow for it to be a wrapper (``uv run``, ``npm exec``...)
_WRAPPER_SUBWORDS: dict[str, set[str]] = {
    "uv": {"run"},
    "uvx": set(),
    "npx": set(),
    "pnpm": {"exec", "dlx"},
    "npm": {"exec", "x"},
    "yarn": {"exec", "dlx"},
    "bunx": set(),
    "bun": {"x"},
    "poetry": {"run"},
    "pipenv": {"run"},
    "pdm": {"run"},
    "hatch": {"run"},
    "rye": {"run"},
    "conda": {"run"},
}
# Shell keywords / operators that may precede a command on the same simple-command line.
_SHELL_NOISE_WORDS = {"&", "&&", "||", "!", ".", "source", "if", "then", "elif", "else", "do", "while",
                      "until", "fi", "done", "esac", "in"}
_SIMPLE_WRAPPERS = {
    "env", "command", "nohup", "time", "xargs", "sudo", "doas", "exec", "builtin", "nice", "ionice",
    "timeout", "stdbuf", "caffeinate", "watch", "chronic", "unbuffer", "script", "strace", "ltrace",
}


def _basename(token: str) -> str:
    """``/usr/bin/git`` | ``C:\\Program Files\\Git\\cmd\\git.exe`` | ``git`` -> ``git`` (lower)."""
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".com", ".ps1", ".sh"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _unwrap(argv: list[str]) -> list[str]:
    """Strip env assignments and known wrappers until a real command is exposed.

    Every stripped ``VAR=value`` is inspected first (``_check_env_assignment``); a dangerous one
    raises ``_Denied`` instead of being silently dropped.
    """
    argv = list(argv)
    for _ in range(12):  # bounded: wrappers can be nested, but not forever
        while argv and (_ENV_ASSIGN_RE.match(argv[0]) or argv[0] in _SHELL_NOISE_WORDS):
            tok = argv.pop(0)
            if _ENV_ASSIGN_RE.match(tok):
                _check_env_assignment(tok)
        if not argv:
            return argv
        name = _basename(argv[0])
        if name in _SIMPLE_WRAPPERS:
            argv = _strip_wrapper_opts(argv[1:], _WRAPPER_VALUE_OPTS.get(name, set()))
            if name == "timeout" and argv and not argv[0].startswith("-"):
                argv = argv[1:]  # DURATION
            if name == "env":  # env may carry more assignments after its options
                continue
            continue
        if name in _WRAPPER_SUBWORDS:
            rest = argv[1:]
            needed = _WRAPPER_SUBWORDS[name]
            if needed:
                # options may precede the subword (``uv -q run``)
                rest = _strip_wrapper_opts(rest, _WRAPPER_VALUE_OPTS.get(name, set()))
                if not rest or rest[0] not in needed:
                    return argv
                rest = rest[1:]
            argv = _strip_wrapper_opts(rest, _WRAPPER_VALUE_OPTS.get(name, set()))
            continue
        return argv
    return argv


def _strip_wrapper_opts(args: list[str], value_opts: set[str]) -> list[str]:
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--":
            return args[i + 1 :]
        if tok.startswith("-") and len(tok) > 1:
            if tok in value_opts and "=" not in tok:
                i += 2
            else:
                i += 1
            continue
        if _ENV_ASSIGN_RE.match(tok):
            _check_env_assignment(tok)
            i += 1
            continue
        return args[i:]
    return []


def _git_subcommand(args: list[str]) -> tuple[str | None, list[str], list[str]]:
    """Return (subcommand, global options, subcommand args) for ``git <global opts> <sub> ...``."""
    value_opts = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--super-prefix",
                  "--config-env", "--attr-source", "--list-cmds"}
    globals_: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("-") and len(tok) > 1:
            globals_.append(tok)
            if tok in value_opts and i + 1 < len(args):
                globals_.append(args[i + 1])
                i += 2
            else:
                i += 1
            continue
        return tok.lower(), globals_, args[i + 1 :]
    return None, globals_, []


# Config keys that redirect credentials/remotes or make git run a command during an ordinary
# subcommand (editor on commit, sequence editor on rebase, fsmonitor on status, external diff /
# textconv on diff, merge drivers on merge, clean/smudge filters on checkout, gpg on commit -S,
# submodule.<n>.update=!cmd, difftool/mergetool commands, browsers). Case-insensitive.
_GIT_SENSITIVE_CONFIG_RE = re.compile(
    r"^(alias\.|url\.|credential|remote\.|"
    r"core\.(sshcommand|hookspath|gitproxy|askpass|editor|fsmonitor|alternaterefscommand)|"
    r"sequence\.editor|diff\.external|diff\.[^.]+\.(textconv|command)|difftool\.|mergetool\.|"
    r"merge\.[^.]+\.driver|filter\.|gpg\.|interactive\.difffilter|submodule\.|trailer\.[^.]+\.command|"
    r"sendemail\.|browser\.|web\.browser|help\.browser|instaweb\.|man\.|"
    r"http\.|https\.|ssh\.|include\.|includeif\.|protocol\.|uploadpack\.|receive\.|push\.pushoption)",
    re.IGNORECASE,
)
# Pagers run via the shell when stdout is a TTY; only paging-off values are allowed.
_GIT_PAGER_CONFIG_RE = re.compile(r"^(core\.pager|pager\.[^.]+)$", re.IGNORECASE)


def _git_setting_is_sensitive(key: str, value: str | None) -> bool:
    """``value=None`` means unknown (fail closed for pager keys)."""
    key = key.strip()
    if _GIT_PAGER_CONFIG_RE.match(key):
        return value is None or value.strip() not in _SAFE_PAGER_VALUES
    return _GIT_SENSITIVE_CONFIG_RE.match(key) is not None


def _classify_git(args: list[str]) -> ToolDecision:
    sub, globals_, rest = _git_subcommand(args)
    i = 0
    while i < len(globals_):
        g = globals_[i]
        setting: str | None = None
        if g == "-c" and i + 1 < len(globals_):
            setting = globals_[i + 1]
            i += 1
        elif g.startswith("-c") and not g.startswith("--"):
            setting = g[2:]  # glued: ``-ccore.editor=x``
        elif g.startswith("--config-env"):
            return _deny(MSG_REMOTE)
        if setting is not None:
            key, has_value, value = setting.partition("=")
            if _git_setting_is_sensitive(key, value if has_value else "true"):
                return _deny(MSG_REMOTE)
        i += 1
    if sub is None:
        return _allow()
    if sub in ("push", "send-pack", "http-push", "svn"):
        return _deny(MSG_PUSH)
    if sub == "remote":
        verb = next((t.lower() for t in rest if not t.startswith("-")), "")
        if verb in ("add", "set-url", "remove", "rm", "rename", "set-branches", "set-head"):
            return _deny(MSG_REMOTE)
        return _allow()
    if sub == "config":
        return _classify_git_config(rest)
    if sub == "credential" or sub.startswith("credential-"):
        return _deny(MSG_AUTH_TOKEN)
    return _allow()


_GIT_CONFIG_READ_FLAGS = {"--get", "--get-all", "--get-regexp", "--get-urlmatch", "--list", "-l",
                          "--show-origin", "--show-scope", "--name-only", "--get-color", "--get-colorbool"}
_GIT_CONFIG_WRITE_FLAGS = {"--unset", "--unset-all", "--add", "--replace-all", "--rename-section",
                           "--remove-section", "--edit", "-e"}


def _classify_git_config(rest: list[str]) -> ToolDecision:
    """``git config``: reads are fine; writes to remote/credential/alias/hook keys are denied."""
    flags = {t for t in rest if t.startswith("-")}
    positional = [t for t in rest if not t.startswith("-")]
    if positional and positional[0].lower() in ("get", "list"):
        return _allow()
    if flags & _GIT_CONFIG_READ_FLAGS:
        return _allow()
    is_write = bool(flags & _GIT_CONFIG_WRITE_FLAGS) or len(positional) >= 2
    if positional and positional[0].lower() in ("set", "unset", "edit", "rename-section", "remove-section"):
        is_write = True
    if not is_write:
        return _allow()
    keys = [t for t in positional if t.lower() not in ("set", "unset", "edit", "rename-section", "remove-section")]
    value = keys[1] if len(keys) >= 2 else None
    if any(_git_setting_is_sensitive(t, value) for t in keys[:1]) or any(
        _GIT_SENSITIVE_CONFIG_RE.match(t) for t in keys[1:]
    ):
        return _deny(MSG_REMOTE)
    return _allow()


_GH_READ_ONLY: dict[str, set[str]] = {
    "pr": {"view", "list", "diff", "checks", "status"},
    "run": {"view", "list", "watch"},
    "issue": {"view", "list"},
    "repo": {"view"},
    "release": {"view", "list"},
    "workflow": {"view", "list"},
    "label": {"list"},
    "auth": {"status"},
    "search": {"*"},
}
_GH_API_VALUE_OPTS = {"-X", "--method", "-H", "--header", "--hostname", "--input", "-q", "--jq", "-t",
                      "--template", "--cache", "-p", "--preview", "-f", "--raw-field", "-F", "--field"}
_GH_API_WRITE_OPTS = {"-f", "--raw-field", "-F", "--field", "--input"}


def _classify_gh(args: list[str]) -> ToolDecision:
    positional = [a for a in args if not a.startswith("-")]
    if not positional:
        return _allow() if args and args[0] in ("--version", "-h", "--help") else _deny(MSG_GH_WRITE)
    if any(a in ("-h", "--help") for a in args):
        return _allow()
    group = positional[0].lower()
    if group in ("help", "version"):
        return _allow()
    if group == "api":
        return _classify_gh_api(args[args.index(positional[0]) + 1 :])
    if group == "auth" and (len(positional) < 2 or positional[1].lower() != "status"):
        return _deny(MSG_AUTH_TOKEN)
    allowed = _GH_READ_ONLY.get(group)
    if allowed is None:
        return _deny(MSG_GH_WRITE)
    if "*" in allowed:
        return _allow()
    if len(positional) < 2 or positional[1].lower() not in allowed:
        return _deny(MSG_GH_WRITE)
    return _allow()


def _classify_gh_api(args: list[str]) -> ToolDecision:
    endpoint: str | None = None
    method = "GET"
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--":
            i += 1
            continue
        if tok.startswith("-"):
            if len(tok) > 2 and not tok.startswith("--"):  # glued short form: ``-XPOST`` / ``-fk=v``
                name, value, has_eq = tok[:2], tok[2:], True
            else:
                name, has_eq, value = tok.partition("=")
            if name in _GH_API_WRITE_OPTS:
                return _deny(MSG_GH_API_WRITE)
            if name in ("-X", "--method"):
                if not has_eq:
                    value = args[i + 1] if i + 1 < len(args) else ""
                    i += 1
                method = value.strip().upper()
            elif name in _GH_API_VALUE_OPTS and not has_eq:
                i += 1
            i += 1
            continue
        if endpoint is None:
            endpoint = tok
        i += 1
    if method not in ("GET", "HEAD"):
        return _deny(MSG_GH_API_WRITE)
    if endpoint is None:
        return _deny(MSG_GH_API_WRITE)
    if endpoint.strip("/").lower().split("?", 1)[0] == "graphql":
        return _deny(MSG_GH_API_WRITE)
    return _allow()


# curl short options that take a value (the rest of the cluster or the next token).
_CURL_VALUE_SHORT = set("AbcCdDeEFhHKmoPQrtTuUwxXyYz")
_CURL_BODY_SHORT = set("dFT")
_CURL_BODY_LONG = {
    "--data", "--data-raw", "--data-binary", "--data-urlencode", "--data-ascii", "--json", "--form",
    "--form-string", "--upload-file", "--post-data", "--post-file", "--body-data", "--body-file",
}
_HTTP_SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


def _classify_http(args: list[str]) -> ToolDecision:
    """curl / wget flags and PowerShell cmdlet parameters (``curl``/``wget`` alias iwr on Windows)."""
    i = 0
    while i < len(args):
        tok = args[i]
        low = tok.lower()
        name, has_eq, value = tok.partition("=")
        if name in _CURL_BODY_LONG:
            return _deny(MSG_HTTP_WRITE)
        if name in ("--request", "--method"):
            if not has_eq:
                value = args[i + 1] if i + 1 < len(args) else ""
                i += 1
            if value.strip().upper() not in _HTTP_SAFE_METHODS:
                return _deny(MSG_HTTP_WRITE)
        elif tok.startswith("-") and not tok.startswith("--") and len(tok) > 1:
            cluster = tok[1:]
            for pos, ch in enumerate(cluster):
                if ch in _CURL_BODY_SHORT:
                    return _deny(MSG_HTTP_WRITE)
                if ch == "X":
                    value = cluster[pos + 1 :]
                    if not value:
                        value = args[i + 1] if i + 1 < len(args) else ""
                        i += 1
                    if value.strip().upper() not in _HTTP_SAFE_METHODS:
                        return _deny(MSG_HTTP_WRITE)
                    break
                if ch in _CURL_VALUE_SHORT:
                    if pos == len(cluster) - 1:
                        i += 1
                    break
        # PowerShell cmdlet parameters (prefix matching, case-insensitive): -Method Post, -Body ...
        if low.startswith("-") and not low.startswith("--"):
            pname = low[1:].split(":", 1)[0]
            if len(pname) >= 2 and "method".startswith(pname) or len(pname) >= 2 and "custommethod".startswith(pname):
                value = low.split(":", 1)[1] if ":" in low else (args[i + 1] if i + 1 < len(args) else "")
                if value.strip().strip("'\"").upper() not in _HTTP_SAFE_METHODS:
                    return _deny(MSG_HTTP_WRITE)
            if pname and ("body".startswith(pname) or (len(pname) >= 2 and "infile".startswith(pname)) or pname == "form"):
                return _deny(MSG_HTTP_WRITE)
        i += 1
    return _allow()


_PUBLISH_RULES: dict[str, set[str]] = {
    "npm": {"publish", "unpublish", "deprecate", "dist-tag", "owner", "access", "token", "login", "adduser"},
    "pnpm": {"publish", "unpublish", "deprecate", "dist-tag", "owner", "login"},
    "yarn": {"publish", "npm"},
    "uv": {"publish"},
    "twine": {"upload", "register"},
    "cargo": {"publish", "yank", "owner", "login"},
    "gem": {"push", "yank", "owner", "signin"},
    "docker": {"push", "login"},
    "podman": {"push", "login"},
    "nerdctl": {"push"},
    "nuget": {"push", "delete"},
    "dotnet": {"nuget"},
    "flutter": {"pub"},
    "dart": {"pub"},
    "helm": {"push", "registry"},
    "poetry": {"publish"},
    "hatch": {"publish"},
    "pdm": {"publish"},
    "flit": {"publish"},
    "mvn": {"deploy"},
    "gradle": {"publish", "publishtomavencentral", "publishtomavenlocal", "uploadarchives"},
    "lerna": {"publish"},
    "changeset": {"publish"},
    "np": set(),
    "oras": {"push"},
    "crane": {"push", "copy"},
    "skopeo": {"copy", "sync"},
    "vsce": {"publish"},
    "ovsx": {"publish"},
    "expo": {"publish"},
    "eas": {"submit", "update", "build"},
    "firebase": {"deploy"},
    "vercel": {"deploy"},
    "netlify": {"deploy"},
    "wrangler": {"publish", "deploy"},
    "fly": {"deploy"},
    "flyctl": {"deploy"},
    "heroku": set(),
    "gcloud": {"app", "run", "functions"},
    "aws": set(),
    "az": set(),
}

_NESTED_SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "fish", "ash", "busybox"}
_INLINE_INTERPRETERS = {"python", "python3", "py", "node", "ruby", "perl", "deno", "bun", "php", "osascript"}
_INLINE_DANGER_RE = re.compile(
    r"(git\W{0,3}push|send-pack|\bgh\W{0,3}(pr|api|issue|release|repo)\b|api\.linear\.app|slack\.com/api|"
    r"hooks\.slack\.com|publish|requests\.(post|put|patch|delete)|urlopen|fetch\(|http\.client|"
    r"httpx|aiohttp|axios|XMLHttpRequest|Net::HTTP|LWP::|curl|wget|subprocess|os\.system|child_process|"
    r"Invoke-RestMethod|Invoke-WebRequest|\.git[/\\]|hooksPath|--git-(dir|path|common-dir)|\.gitconfig)",
    re.IGNORECASE,
)


def _decode_encoded_command(value: str) -> str | None:
    try:
        raw = base64.b64decode(value, validate=False)
    except Exception:
        return None
    for enc in ("utf-16-le", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _classify_powershell_host(args: list[str]) -> ToolDecision:
    """``pwsh``/``powershell`` host invocation: unwrap -Command / -EncodedCommand / -File."""
    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith(("-", "/")) and len(tok) > 1:
            pname = tok[1:].lower().split(":", 1)[0]
            inline = tok.split(":", 1)[1] if ":" in tok else None
            if pname and ("command".startswith(pname) or pname in ("cwa", "commandwithargs")):
                script = inline if inline is not None else " ".join(args[i + 1 :])
                if script.strip() == "-":
                    return _deny(MSG_ENCODED)
                return _classify(script, powershell=True)
            if pname and ("encodedcommand".startswith(pname) or pname in ("e", "ec", "enc")):
                value = inline if inline is not None else (args[i + 1] if i + 1 < len(args) else "")
                decoded = _decode_encoded_command(value)
                if decoded is None:
                    return _deny(MSG_ENCODED)
                return _classify(decoded, powershell=True)
            if pname and "file".startswith(pname) and len(pname) >= 1:
                return _allow()
            if pname in ("ex", "ep", "executionpolicy", "w", "windowstyle", "v", "version", "wd",
                         "workingdirectory", "inputformat", "outputformat", "if", "of", "configurationname",
                         "settingsfile", "psconsolefile", "custompipename"):
                i += 2
                continue
            i += 1
            continue
        # First positional is treated as the command text (pwsh semantics).
        return _classify(" ".join(args[i:]), powershell=True)
    return _allow()


def _classify_cmd_host(args: list[str]) -> ToolDecision:
    for i, tok in enumerate(args):
        if tok.lower() in ("/c", "/k", "/r"):
            return _classify(" ".join(args[i + 1 :]))
    return _allow()


def _classify_posix_host(args: list[str]) -> ToolDecision:
    """``bash -c '...'`` / ``sh -lc '...'``; ``bash script.sh`` is allowed (not inspectable)."""
    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("-") and len(tok) > 1 and not tok.startswith("--"):
            if "c" in tok[1:]:
                script = args[i + 1] if i + 1 < len(args) else ""
                return _classify(script)
            if tok in ("-o", "-O", "+o", "+O") and i + 1 < len(args):
                i += 2
                continue
            i += 1
            continue
        if tok.startswith("--"):
            i += 1
            continue
        return _allow()
    return _allow()


# Binaries whose arguments we classify. When the first word is none of these, a later token with
# one of these basenames (``some-wrapper --flag git push``) re-anchors classification there.
_KNOWN_BINARIES = (
    {"git", "gh", "curl", "wget", "iwr", "irm", "invoke-webrequest", "invoke-restmethod", "pwsh",
     "powershell", "cmd", "eval", "iex", "invoke-expression", "invoke-command", "icm", "ssh", "scp",
     "rsync", "sftp", "http", "https", "httpie", "xh", "curlie"}
    | _NESTED_SHELLS | _INLINE_INTERPRETERS | set(_PUBLISH_RULES)
)


def _classify_argv(argv: list[str], tainted: frozenset[str] = frozenset()) -> ToolDecision:
    try:
        argv = _strip_redirections(argv, tainted)
        argv = _unwrap(argv)
        if argv:
            _check_env_setter(argv)
            _check_write_targets(argv, tainted)
    except _Denied as exc:
        return _deny(exc.reason)
    if not argv:
        return _allow()
    name = _basename(argv[0])
    if name not in _KNOWN_BINARIES:
        for idx in range(1, len(argv)):
            if _basename(argv[idx]) in _KNOWN_BINARIES and not argv[idx].startswith("-"):
                return _classify_argv(argv[idx:], tainted)
        return _allow()
    rest = argv[1:]
    if name == "git":
        return _classify_git(rest)
    if name == "gh":
        return _classify_gh(rest)
    if name in ("curl", "wget", "iwr", "irm", "invoke-webrequest", "invoke-restmethod", "http", "https",
                "httpie", "xh", "curlie"):
        return _classify_http(rest)
    if name in ("pwsh", "powershell"):
        return _classify_powershell_host(rest)
    if name == "cmd":
        return _classify_cmd_host(rest)
    if name in _NESTED_SHELLS:
        if name == "busybox":
            return _classify_argv(rest, tainted) if rest else _allow()
        return _classify_posix_host(rest)
    if name in ("eval", "iex", "invoke-expression", "invoke-command", "icm"):
        script = " ".join(t for t in rest if not t.startswith("-"))
        return _classify(script, powershell=name != "eval")
    if name in _INLINE_INTERPRETERS:
        for j, tok in enumerate(rest):
            if tok in ("-c", "-e", "--eval", "-E", "-r", "--exec") and j + 1 < len(rest):
                snippet = rest[j + 1]
                if _INLINE_DANGER_RE.search(snippet):
                    return _deny(
                        "Inline scripts that may perform network writes or spawn git/gh are denied; "
                        "run plain commands instead so they can be checked."
                    )
        return _allow()
    if name in ("ssh", "scp", "sftp", "rsync") and any("github.com" in t or "gitlab.com" in t for t in rest):
        return _deny(MSG_PUSH)
    if name in _PUBLISH_RULES:
        return _classify_publish(name, [t.lower() for t in rest if not t.startswith("-")])
    return _allow()


# Tools whose first positional selects a group and whose *second* positional is the real verb
# (``yarn npm publish``, ``dotnet nuget push``, ``helm registry login``, ``gcloud run deploy``).
_CLOUD_WRITE_PAIRS: dict[str, set[tuple[str, ...]]] = {
    "aws": {("s3", "cp"), ("s3", "sync"), ("s3", "mv"), ("s3", "rm"), ("s3api", "put-object"),
            ("ecr", "put-image"), ("lambda", "update-function-code"), ("lambda", "publish-version"),
            ("codeartifact", "publish-package-version"), ("deploy", "push"), ("deploy", "create-deployment")},
    "az": {("webapp", "deploy"), ("webapp", "up"), ("acr", "build"), ("acr", "import"),
           ("functionapp", "deployment"), ("containerapp", "up"), ("storage", "blob")},
    "heroku": {("container", "push"), ("container", "release"), ("git", "remote"), ("releases", "rollback")},
}
_PUBLISH_TWO_LEVEL = {"yarn": {"npm"}, "dotnet": {"nuget"}, "flutter": {"pub"}, "dart": {"pub"},
                      "helm": {"registry"}, "gcloud": {"app", "run", "functions"}}
_PUBLISH_SECOND_VERBS = {"publish", "push", "deploy", "login", "unpublish"}


def _classify_publish(name: str, positional: list[str]) -> ToolDecision:
    if name == "np":
        return _deny(MSG_PUBLISH)
    if name in _CLOUD_WRITE_PAIRS:
        if tuple(positional[:2]) in _CLOUD_WRITE_PAIRS[name] or (positional[:1] == ["container:push"]):
            return _deny(MSG_PUBLISH)
        return _allow()
    if not positional:
        return _allow()
    first = positional[0]
    if first in _PUBLISH_TWO_LEVEL.get(name, set()):
        if len(positional) > 1 and positional[1] in _PUBLISH_SECOND_VERBS:
            return _deny(MSG_PUBLISH)
        return _allow()
    if first in _PUBLISH_RULES[name]:
        return _deny(MSG_PUBLISH)
    return _allow()


def _classify(command: str, *, powershell: bool = False) -> ToolDecision:
    try:
        commands = _split_commands(command, powershell=powershell)
    except _ParseError as exc:
        if _DANGEROUS_TOKEN_RE.search(command):
            return _deny(f"Could not parse command ({exc}); commands mentioning push/gh/curl/... "
                         "are denied unless they parse cleanly. Simplify the quoting and retry.")
        return _allow()
    tainted = _tainted_names(commands)
    for argv in commands:
        decision = _classify_argv(argv, tainted)
        if not decision.allow:
            return decision
    return _allow()


def classify_shell(command: str, *, powershell: bool = False) -> ToolDecision:
    """Classify a Bash (or PowerShell, with ``powershell=True``) command line.

    Fails closed on unparseable input only when it contains a dangerous token.
    """
    if not command or not command.strip():
        return _allow()
    low = command.lower()
    for host in _FORBIDDEN_HOSTS:
        if host in low:
            return _deny(MSG_TRACKER_HOST)
    if _PS_HTTP_METHOD_CALL_RE.search(command):
        return _deny(MSG_HTTP_WRITE)
    for match in _PS_ENV_ASSIGN_RE.finditer(command):
        key = next(v for v in match.groups() if v)
        if _env_key_is_dangerous(key, None):
            return _deny(MSG_ENV)
    return _classify(command, powershell=powershell)


# ------------------------------------------------------------------------------- tool policy
READ_TOOLS = ("Read", "Glob", "Grep", "NotebookRead", "LS")
EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
SHELL_TOOLS = ("Bash", "PowerShell")
# StructuredOutput is how the CLI delivers output_format results; denying it means no worker can
# ever return its result.
PLAIN_TOOLS = ("TodoWrite", "Task", "Agent", "BashOutput", "KillShell", "StructuredOutput")
WEB_TOOLS = ("WebFetch", "WebSearch")
_PATH_KEYS = ("file_path", "notebook_path", "path")
# Files whose contents are credentials: readable inside the worktree only.
_SECRET_FILE_NAMES = {
    ".env", ".netrc", "_netrc", ".git-credentials", ".credentials.json", "credentials", ".pypirc",
    ".npmrc", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", "hosts.yml", ".pgpass", "kubeconfig",
}


def _norm(path: Path) -> str:
    return os.path.normcase(str(path))


def _is_within(path: Path, root: Path) -> bool:
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def _resolve(raw: str, base: Path) -> Path:
    p = Path(os.path.expandvars(raw)).expanduser()
    if not p.is_absolute():
        p = base / p
    try:
        return p.resolve()
    except OSError:
        return p.absolute()


class ToolPolicy:
    """Per-worker tool policy: knows the worktree and the profile's policy settings.

    ``read_only`` (rv, refine, plan) denies every edit tool with a message; shell commands are still
    classified exactly as for a code worker.
    """

    def __init__(
        self,
        worktree: Path,
        profile: Profile,
        extra_read_dirs: list[Path] | None = None,
        *,
        read_only: bool = False,
    ):
        self.worktree = Path(worktree).resolve()
        self.profile = profile
        self.read_only = read_only
        self.extra_read_dirs = [Path(p).resolve() for p in (extra_read_dirs or [])]
        self.temp_dir = Path(tempfile.gettempdir()).resolve()
        pdir = profile.profile_dir
        self.profiles_dir = (Path(pdir).resolve().parent if pdir else resolve_profiles_dir())
        self.allowed_tools: tuple[str, ...] = (
            READ_TOOLS + EDIT_TOOLS + SHELL_TOOLS + PLAIN_TOOLS
            + (WEB_TOOLS if profile.policy.allow_web_tools else ())
            + tuple(profile.policy.extra_allowed_tools)
        )

    # ----------------------------------------------------------------------------- decisions
    def check(self, tool_name: str, tool_input: dict[str, Any] | None) -> ToolDecision:
        tool_input = tool_input or {}
        if tool_name.lower().startswith("mcp__"):
            return _deny(self._unknown_tool_message(tool_name))
        if tool_name == "AskUserQuestion":
            return _deny(MSG_ASK_USER)
        if tool_name not in self.allowed_tools:
            return _deny(self._unknown_tool_message(tool_name))
        if tool_name in SHELL_TOOLS:
            command = tool_input.get("command")
            if not isinstance(command, str):
                return _deny(f"{tool_name} needs a string `command`.")
            return classify_shell(command, powershell=tool_name == "PowerShell")
        if tool_name in EDIT_TOOLS:
            if self.read_only:
                return _deny(MSG_READ_ONLY)
            return self._check_edit_path(tool_name, tool_input)
        if tool_name in READ_TOOLS:
            return self._check_read_path(tool_input)
        if tool_name in WEB_TOOLS:
            url = str(tool_input.get("url") or tool_input.get("query") or "").lower()
            if any(h in url for h in _FORBIDDEN_HOSTS):
                return _deny(MSG_TRACKER_HOST)
        return _allow()

    def _unknown_tool_message(self, tool_name: str) -> str:
        return (
            f"Tool {tool_name!r} is not available to workers. Allowed tools: "
            f"{', '.join(self.allowed_tools)}. Anything that must reach GitHub, Linear or Slack "
            "goes in your structured output; the engine performs it."
        )

    def _check_edit_path(self, tool_name: str, tool_input: dict[str, Any]) -> ToolDecision:
        raw = tool_input.get("notebook_path") if tool_name == "NotebookEdit" else tool_input.get("file_path")
        if not isinstance(raw, str) or not raw.strip():
            return _deny(f"{tool_name} needs a file path inside the worktree {self.worktree}.")
        path = _resolve(raw, self.worktree)
        if not _is_within(path, self.worktree):
            return _deny(
                f"Writes are confined to the worktree {self.worktree}; {raw} is outside it. Edit files "
                "under the worktree only and put scratch files under .factory/ inside it."
            )
        if ".git" in path.parts:
            return _deny("Files under .git (hooks, config, refs) are not editable. Use git commands instead.")
        return _allow()

    def _check_read_path(self, tool_input: dict[str, Any]) -> ToolDecision:
        raw = next((tool_input[k] for k in _PATH_KEYS if isinstance(tool_input.get(k), str)), None)
        if raw is None:
            return _allow()  # Glob/Grep default to cwd
        path = _resolve(raw, self.worktree)
        if _is_within(path, self.worktree) or any(_is_within(path, d) for d in self.extra_read_dirs):
            return _allow()
        if _is_within(path, self.profiles_dir) or _is_within(self.profiles_dir, path):
            return _deny("The factory profiles directory holds secrets and is not readable by workers.")
        name = path.name.lower()
        if name in _SECRET_FILE_NAMES or name.startswith(".env.") or ".ssh" in path.parts:
            return _deny(
                f"{raw} looks like a credentials file outside the worktree and is not readable. "
                "Everything you need for this task is in the worktree and the prompt."
            )
        return _allow()

    # ----------------------------------------------------------------------- SDK adapters
    async def can_use_tool(
        self, tool_name: str, tool_input: dict[str, Any], context: Any
    ) -> PermissionResultAllow | PermissionResultDeny:
        decision = self.check(tool_name, tool_input)
        if decision.allow:
            return PermissionResultAllow()
        return PermissionResultDeny(message=decision.reason, interrupt=False)

    async def pre_tool_use_hook(self, input_data: Any, tool_use_id: str | None, context: Any) -> dict:
        data = input_data or {}
        decision = self.check(str(data.get("tool_name", "")), data.get("tool_input") or {})
        if decision.allow:
            return {}  # no opinion: fall through to permission rules -> can_use_tool
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": decision.reason,
            }
        }

    def hooks(self) -> dict[str, list[HookMatcher]]:
        matcher = HookMatcher(matcher=None, hooks=[self.pre_tool_use_hook], timeout=PRE_TOOL_USE_HOOK_TIMEOUT_S)
        return {"PreToolUse": [matcher]}


# ------------------------------------------------------------------------------- push policy
class PushDecision(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)


def matches_any(path: str, globs: list[str]) -> bool:
    """gitwildmatch semantics (``pathspec``); backslashes are treated as separators."""
    if not globs:
        return False
    spec = pathspec.GitIgnoreSpec.from_lines(globs)
    return bool(spec.match_file(path.replace("\\", "/").lstrip("/")))


def evaluate_push(
    kind: Kind | str,
    output: CodeWorkerOutput,
    changed_files: list[str],
    sensitive_globs: list[str],
    *,
    conflict_markers: bool = False,
    foreign_owned: list[str] | None = None,
) -> PushDecision:
    """Engine-side push policy applied on top of the worker's ``push_ok``.

    ``foreign_owned`` is the subset of ``changed_files`` CODEOWNERS assigns to someone other than
    me (``foreign_owned_changes``); any entry refuses the push.
    """
    kind_value = kind.value if isinstance(kind, Kind) else str(kind)
    reasons: list[str] = []
    if output.status != "done":
        reasons.append(f"worker status is {output.status!r}, not 'done'")
    if not output.push_ok:
        why = "; ".join(output.push_reasons) or "no reason given"
        reasons.append(f"worker did not set push_ok ({why})")
    if not any(v.passed for v in output.verification):
        reasons.append("no passing verification run reported")
    hits = sorted({f for f in changed_files if matches_any(f, sensitive_globs)})
    if hits:
        reasons.append(f"changed files match sensitive globs: {', '.join(hits)}")
    if foreign_owned:
        reasons.append(f"changed files owned by another team per CODEOWNERS: {', '.join(foreign_owned)}")
    if kind_value in ("ci", "conflict", "rc") and not getattr(output, "head_verified", False):
        reasons.append("worker did not verify the worktree HEAD against the PR head")
    if kind_value == "rc":
        skipped = [c.comment_id for c in getattr(output, "comments", []) if c.disposition == "skipped"]
        if skipped:
            reasons.append(f"comments skipped: {', '.join(skipped)}")
    if kind_value == "conflict":
        if not getattr(output, "all_resolved", False):
            reasons.append("not all conflicts resolved")
        unsafe = [h.file for h in getattr(output, "hunks", []) if h.classification == "unsafe"]
        if unsafe:
            reasons.append(f"unsafe hunks: {', '.join(unsafe)}")
        if conflict_markers:
            reasons.append("conflict markers remain in the worktree")
    return PushDecision(allowed=not reasons, reasons=reasons)


def sensitive_globs_for(profile: Profile, repo: RepoInfo, *, my_login: str | None = None) -> list[str]:
    """policy.sensitive_globs + the repo override's globs, deduplicated.

    CODEOWNERS is deliberately not folded in here: at glob level ``* @team`` followed by
    ``/docs/ @me`` kept ``*`` sensitive and refused every push although GitHub assigns ``docs/`` to
    me. Ownership is applied per changed path by ``foreign_owned_changes`` and passed to
    ``evaluate_push(foreign_owned=...)``. ``my_login`` is accepted for compatibility (it used to
    pick the login for the CODEOWNERS check) and no longer affects the globs.
    """
    globs: list[str] = list(profile.policy.sensitive_globs)
    override = getattr(repo, "override", None)
    if override is not None:
        globs.extend(getattr(override, "sensitive_globs", []) or [])
    return list(dict.fromkeys(globs))


def foreign_owned_changes(
    profile: Profile, repo: RepoInfo, changed_files: list[str], my_login: str | None = None
) -> list[str]:
    """The changed files CODEOWNERS assigns to someone other than me (exact per-path last-match-wins).

    ``my_login`` is the resolved GitHub login (``GitHub.login()``), falling back to the profile's
    ``github.login``. With no login at all every owned path counts as foreign (fail closed: the
    profile's documented ``login = ""`` means "detect with gh", not "everything is mine").
    ``[]`` when the repo has no CODEOWNERS file.
    """
    login = my_login or profile.github.login or ""
    try:
        return codeowners.foreign_owned_paths(Path(repo.path), list(changed_files), list(profile.policy.my_teams), login)
    except FileNotFoundError:  # the file vanished between lookup and read
        return []


def may_edit_pr_body(state: Any, repo: str, number: int) -> bool:
    """Only PR bodies of PRs the factory opened may be edited (evidence section)."""
    return bool(state.is_factory_pr(repo, number))
