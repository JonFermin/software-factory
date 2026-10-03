"""Policy tests: the safety core. Every denied case here is an outward write a worker must never do."""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

import pytest
from claude_agent_sdk import HookMatcher, PermissionResultAllow, PermissionResultDeny

from software_factory.actions import Kind
from software_factory.config import Profile
from software_factory.policy import (
    PRE_TOOL_USE_HOOK_TIMEOUT_S,
    PushDecision,
    ToolDecision,
    ToolPolicy,
    classify_shell,
    evaluate_push,
    foreign_owned_changes,
    matches_any,
    may_edit_pr_body,
    sensitive_globs_for,
)
from software_factory.workers.schemas import (
    BuildOutput,
    CiOutput,
    CommentDisposition,
    ConflictOutput,
    HunkResolution,
    RcOutput,
    VerificationRun,
)

ENC_GIT_PUSH = base64.b64encode("git push".encode("utf-16-le")).decode()

DENIED_SHELL = [
    # --- git push in every disguise
    "git push",
    "git push --force",
    "git push origin +HEAD:main",
    "FOO=1 git push",
    'FOO="a b" BAR=1 git push',
    "cd x && git push",
    'git commit -m "x"; git push',
    'git commit -m "x" || git push',
    "git commit -m x\ngit push",
    "git commit | git push",
    "echo hi > out.txt; git push",
    "(git push)",
    "{ git push; }",
    "if true; then git push; fi",
    "git push 2>/dev/null",
    "git push || true",
    "echo $(git push)",
    "echo `git push`",
    'echo "result: $(git push)"',
    'bash -c "git push"',
    'bash -lc "git push"',
    "sh -c 'gh pr merge 1'",
    'pwsh -Command "git push"',
    'pwsh -NoProfile -Command "git push"',
    'pwsh -c "& {git push}"',
    "powershell -c git push",
    "powershell -enc " + ENC_GIT_PUSH,
    "powershell -EncodedCommand notbase64!!",
    'cmd /c "git push"',
    "cmd.exe /c git push",
    'eval "git push"',
    'iex "git push"',
    "git -C ../repo push",
    "git -c k=v push",
    "git -c alias.p=push p",
    "git -c alias.x='!git push' x",
    "/usr/bin/git push",
    "git.exe push",
    '"C:\\Program Files\\Git\\cmd\\git.exe" push',
    "C:\\Users\\me\\git.exe push",
    '& "C:\\git.exe" push',
    "env FOO=1 git push",
    "command git push",
    "exec git push",
    "nohup git push &",
    "time git push",
    "timeout 30 git push",
    "nice -n 5 git push",
    "xargs -I {} git push {}",
    "sudo -u me git push",
    "uv run git push",
    "uv run --python 3.12 git push",
    "npx gh pr merge 1",
    "some-unknown-wrapper --flag git push",
    "git send-pack",
    "git http-push",
    "GIT_DIR=x git push",
    "python -c \"import subprocess; subprocess.run(['git','push'])\"",
    "node -e \"require('child_process').execSync('git push')\"",
    # --- git remote / config / credentials
    "git remote set-url origin git@github.com:x/y.git",
    "git remote add up https://x",
    "git remote remove origin",
    "git config url.x.insteadOf y",
    "git config credential.helper store",
    "git config --add remote.origin.pushurl x",
    "git -c core.sshCommand=x fetch",
    "git credential fill",
    "ssh git@github.com",
    # --- git config keys that run a command during an ordinary subcommand (finding: classifier gap)
    "git -c core.editor='git push #' commit",
    "git -c sequence.editor=x rebase -i HEAD~1",
    "git -c core.fsmonitor=x status",
    "git -c diff.external=x diff",
    "git -c diff.evil.textconv=x diff",
    "git -c core.pager='git push' log -p",
    "git -c core.alternateRefsCommand=x fetch",
    "git -c filter.lfs.clean=x checkout .",
    "git -c merge.evil.driver=x merge main",
    "git -c gpg.program=x commit -S",
    "git -c submodule.x.update='!git push' submodule update",
    "git -c difftool.x.cmd='git push' -c diff.tool=x difftool -y",
    "git -ccore.editor=x commit",
    "git config core.editor x",
    "git config set core.fsmonitor x",
    "git config core.pager 'git push'",
    "git config --add diff.external x",
    # --- git / gh / ssh behaviour through environment variables (finding: env bypass)
    "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.z GIT_CONFIG_VALUE_0='!git push origin HEAD:refs/heads/evil' git z",
    "GIT_CONFIG_GLOBAL=.factory/evil.cfg git ls-remote origin",
    "GIT_CONFIG_SYSTEM=.factory/evil.cfg git status",
    "GIT_EDITOR='git push' git commit",
    "GIT_SEQUENCE_EDITOR=x git rebase -i HEAD~1",
    "GIT_SSH_COMMAND='x' git fetch",
    "GIT_SSH=x git fetch",
    "GIT_PROXY_COMMAND=x git fetch",
    "GIT_EXTERNAL_DIFF=x git diff",
    "GIT_ASKPASS=x git fetch",
    "GIT_DIR=../other/.git git commit -m x",
    "GIT_WORK_TREE=../other git add -A",
    "GIT_EXEC_PATH=.factory git status",
    "GIT_PAGER='git push' git log",
    "PAGER='git push' git log",
    "EDITOR='git push' git commit",
    "VISUAL=x git commit",
    "GH_PAGER='git push' gh pr view 1",
    "GH_EDITOR=x gh pr view 1",
    "GH_BROWSER=x gh pr view 1 --web",
    "GH_CONFIG_DIR=.factory/gh gh pr view 1",
    "GH_TOKEN=ghp_x gh pr view 1",
    "GITHUB_TOKEN=ghp_x gh pr view 1",
    "GH_HOST=evil.example gh pr view 1",
    "SSH_ASKPASS=x git fetch",
    "env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.z GIT_CONFIG_VALUE_0='!git push' git z",
    "env -i GIT_SSH_COMMAND=x git fetch",
    "export GIT_SSH_COMMAND=x; git fetch",
    "export GIT_CONFIG_GLOBAL=.factory/evil.cfg && git status",
    "declare -x GIT_EDITOR=x; git commit",
    "setx GIT_SSH_COMMAND x",
    "$env:GIT_SSH_COMMAND = 'x'; git fetch",
    "$env:GH_CONFIG_DIR='.factory'; gh pr view 1",
    "${env:GIT_CONFIG_GLOBAL}='x'; git status",
    "Set-Item env:GIT_EDITOR 'x'; git commit",
    "Set-Item -Path env:GH_TOKEN -Value x",
    "[Environment]::SetEnvironmentVariable('GIT_SSH_COMMAND', 'x')",
    "X='git push'; eval \"$X\"",
    "CMD=\"gh pr merge 1\" bash -c \"$CMD\"",
    # --- gh writes (deny by default)
    "gh",
    "gh pr",
    "gh pr comment 1 -b hi",
    "gh pr merge",
    "gh pr ready",
    "gh pr review --approve",
    "gh pr create",
    "gh pr edit 1",
    "gh pr close 1",
    "gh issue comment",
    "gh issue create",
    "gh release create v1",
    "gh repo clone x",
    "gh auth token",
    "gh auth login",
    # --- gh api non-GET / body / graphql
    "gh api",
    "gh api -X POST repos/o/r/issues",
    "gh api -XPOST repos/o/r",
    "gh api --method PATCH repos/o/r",
    "gh api --method=PATCH repos/o/r",
    "gh api -X DELETE repos/o/r/issues/1",
    "gh api graphql -f query=...",
    "gh api /graphql",
    "gh api repos/o/r/pulls/1/reviews -f event=APPROVE",
    "gh api -F a=b repos/o/r",
    "gh api --field a=b repos/o/r",
    "gh api --raw-field a=b repos/o/r",
    "gh api --input body.json repos/o/r",
    "gh api -fa=b repos/o/r",
    # --- curl / wget writes
    "curl -X POST https://x",
    "curl -XPOST https://x",
    "curl --request=PUT https://x",
    "curl -X PATCH https://api.github.com/repos/o/r",
    "curl -X DELETE https://x",
    "curl -d a=b https://x",
    "curl -sSLd a=b https://x",
    "curl --data-raw x https://x",
    "curl --data-binary @f https://x",
    "curl --json '{\"a\":1}' https://x",
    "curl -F f=@x https://x",
    "curl -T file https://x",
    "curl --upload-file f https://x",
    "wget --post-data=a https://x",
    "wget --post-file=f https://x",
    "wget --method=PUT https://x",
    "wget --body-data=x https://x",
    # --- PowerShell cmdlets
    "Invoke-RestMethod -Method Post -Uri https://x",
    "invoke-restmethod -method post -uri https://x",
    "Invoke-RestMethod -Uri https://x -Method:Patch",
    "Invoke-WebRequest -InFile f https://x",
    "iwr -Me Post https://x",
    "irm -Body x https://x",
    "(New-Object System.Net.WebClient).UploadString('https://x', 'data')",
    # --- tracker / chat hosts
    "curl https://api.linear.app/graphql",
    "curl -X POST https://hooks.slack.com/services/x",
    "curl https://slack.com/api/chat.postMessage",
    "irm https://API.LINEAR.APP/graphql",
    # --- publishing
    "npm publish",
    "pnpm publish",
    "yarn npm publish",
    "npm deprecate x",
    "npm unpublish",
    "uv publish",
    "twine upload dist/*",
    "cargo publish",
    "gem push x.gem",
    "docker push img",
    "dotnet nuget push x.nupkg",
    # --- fail closed on unparseable input mentioning a dangerous token
    'git push "unbalanced',
    "gh pr merge 'unbalanced",
    "echo 'x > .git/hooks/pre-commit",
    # --- writes into a git dir (a planted hook runs with the engine's credentials)
    "echo 'git push' > .git/hooks/pre-commit",
    "echo x >> .git/hooks/post-checkout",
    "echo x >.git/config",
    "echo x > ./.git/hooks/pre-commit",
    "echo x > ../other/.git/hooks/pre-commit",
    "echo x 2> .git/hooks/x",
    "echo x &> .git/hooks/x",
    "echo x &>> .git/hooks/x",
    "printf '%s' x >| .git/hooks/x",
    "echo x 1>.git/hooks/x",
    "echo x >& .git/hooks/x",
    "echo x > .git\\hooks\\pre-commit",
    "echo x > '.git/hooks/pre-commit'",
    'echo x > ".git/hooks/pre-commit"',
    "echo x > .git",
    "echo x > /srv/repo.git/hooks/pre-commit",
    "echo x > /srv/repo.git/config",
    'echo x > "$(git rev-parse --git-dir)/hooks/pre-commit"',
    "echo x > $(git rev-parse --git-dir)/hooks/pre-commit",
    "echo x > $(git rev-parse --git-path hooks)/pre-commit",
    "echo x > $(git rev-parse --git-common-dir)/hooks/pre-commit",
    "echo x > $(git rev-parse --absolute-git-dir)/config",
    "echo x > `git rev-parse --git-dir`/hooks/pre-commit",
    "echo x > $(git -C sub rev-parse --git-dir)/hooks/pre-commit",
    'echo x > "$GIT_DIR/hooks/pre-commit"',
    "echo x > ${GIT_DIR}/hooks/pre-commit",
    "echo x > $GIT_COMMON_DIR/hooks/pre-commit",
    "H=$(git rev-parse --git-path hooks); echo x > $H/pre-commit",
    "H=.git/hooks && echo x > $H/pre-commit",
    "export H=$(git rev-parse --git-dir); echo x > ${H}/hooks/pre-commit",
    "cat <<EOF > .git/hooks/pre-commit\n#!/bin/sh\necho hi\nEOF",
    "tee .git/hooks/pre-commit < evil.sh",
    "echo x | tee -a .git/hooks/pre-commit",
    "echo x | sudo tee .git/hooks/pre-commit",
    "cp evil.sh .git/hooks/pre-commit",
    "cp -r hooks/ .git/hooks",
    "cp -t .git/hooks evil.sh",
    "cp --target-directory=.git/hooks evil.sh",
    "mv evil .git/hooks/pre-commit",
    "install -m755 evil .git/hooks/pre-commit",
    "install -d .git/hooks",
    "ln -s /tmp/evil .git/hooks/pre-commit",
    'ln -sf ../evil "$(git rev-parse --git-dir)"/hooks/pre-commit',
    "rsync -a hooks/ .git/hooks/",
    "touch .git/hooks/pre-commit",
    "chmod +x .git/hooks/pre-commit",
    "mkdir -p .git/hooks",
    "rm -rf .git",
    "rm .git/index",
    "truncate -s0 .git/config",
    "dd if=evil of=.git/hooks/pre-commit",
    "sed -i 's/a/b/' .git/config",
    "sed -i.bak -e 's/a/b/' .git/config",
    "perl -pi -e 's/a/b/' .git/config",
    "cd .git/hooks && echo x > pre-commit",
    "cd $(git rev-parse --git-dir); echo x > hooks/pre-commit",
    "pushd .git/hooks",
    "xargs -I {} cp {} .git/hooks/ < list",
    'bash -c "echo x > .git/hooks/pre-commit"',
    'sh -c "cp evil .git/hooks/pre-commit"',
    'eval "echo x > .git/hooks/pre-commit"',
    "X='echo x > .git/hooks/pre-commit'; eval \"$X\"",
    'pwsh -Command "Set-Content .git/hooks/pre-commit x"',
    "python -c \"open('.git/hooks/pre-commit','w').write('git push')\"",
    "node -e \"require('fs').writeFileSync('.git/hooks/pre-commit', 'x')\"",
    # --- git / gh / ssh configuration and credential files (read by the engine's git and gh)
    "echo '[core] hooksPath = /evil' >> ~/.gitconfig",
    "echo x > $HOME/.gitconfig",
    "echo x > ~/.config/git/config",
    "echo x >> ~/.git-credentials",
    "echo x > ~/.netrc",
    "cp evil ~/.config/gh/hosts.yml",
    "echo x >> ~/.ssh/config",
    "echo x >> $HOME/.ssh/authorized_keys",
    "echo x > %USERPROFILE%\\.gitconfig",
]

ALLOWED_SHELL = [
    "uv run pytest -q",
    "uv run --python 3.12 pytest",
    "uv run ruff check .",
    "uv sync",
    "npm test",
    "npm run build",
    "npm run test -- --watch=false",
    "npm ci",
    "npm install",
    "npm run publish-docs",
    "npx vitest run",
    "npm exec vitest",
    "pnpm exec vitest",
    "pnpm test",
    "yarn test",
    "yarn npm info x",
    "cargo test",
    "docker build .",
    "dotnet test",
    "pytest -q",
    'pytest -k "test_push"',
    "make test",
    "just test",
    'git commit -am "x"',
    'git commit -m "feat: push notifications"',
    "git status",
    "git diff",
    "git diff --name-only main..HEAD",
    "git log",
    "git log --grep push",
    'git log -S"push" --oneline',
    "git fetch origin",
    "git fetch origin pull/1/head",
    "git pull",
    "git rebase main",
    "git merge main",
    "git add -A",
    "git stash",
    "git checkout -b x",
    "git switch main",
    "git cherry-pick abc",
    "git revert HEAD",
    "git reset --hard HEAD~1",
    "git show HEAD",
    "git branch -a",
    "git rev-parse HEAD",
    "git ls-files",
    "git worktree list",
    "git remote -v",
    "git remote show origin",
    "git remote get-url origin",
    'git config user.name "x"',
    "git config user.email me@example.com",
    "git config --get remote.origin.url",
    "git config remote.origin.url",
    "git config --list",
    "git config core.autocrlf false",
    "git -C ../other status",
    "git -c core.pager=cat log",
    "git -c core.pager= log",
    "git -c core.autocrlf=false status",
    "git -c diff.renames=true diff",
    "git -c merge.conflictstyle=diff3 merge main",
    "git config core.pager cat",
    "git config --unset user.name",
    "git config --get core.editor",
    "git push-check",
    # harmless environment variables
    "GIT_TERMINAL_PROMPT=0 git fetch origin",
    "GIT_PAGER=cat git log",
    "PAGER=cat git log -p",
    "GIT_AUTHOR_NAME=x GIT_AUTHOR_EMAIL=x@y git commit -m x",
    "GIT_TRACE=1 git status",
    "GH_REPO=o/r gh pr list",
    "GH_PAGER=cat gh pr view 1",
    "GH_NO_UPDATE_NOTIFIER=1 gh pr view 1",
    "export PATH=$HOME/.local/bin:$PATH; npm test",
    "export NODE_ENV=test; npm test",
    'MSG="feat: push notifications" git commit -m "$MSG"',
    "$env:FOO='bar'; npm test",
    "echo $env:GIT_SSH_COMMAND",
    "Set-Item env:NODE_ENV test; npm test",
    "gh pr view 1 --json title",
    "gh pr list --repo o/r",
    "gh pr diff 1",
    "gh pr checks 1",
    "gh pr status",
    "gh run view 123 --log-failed",
    "gh run list",
    "gh run watch 1",
    "gh issue view 1",
    "gh issue list",
    "gh repo view o/r",
    "gh release list",
    "gh release view v1",
    "gh search prs merge conflict",
    'gh search code "git push"',
    "gh auth status",
    "gh --version",
    "gh pr merge --help",
    "gh api repos/o/r/pulls/1/comments",
    "gh api repos/o/r/pulls/1/reviews",
    "gh api -X GET repos/o/r",
    "gh api --method GET repos/o/r",
    "gh api -X HEAD repos/o/r",
    "gh api --jq .title repos/o/r/pulls/1",
    'gh api -H "Accept: x" repos/o/r',
    "gh api --paginate repos/o/r/issues",
    "gh api repos/o/r/pulls/1 -q .title",
    "curl -sSL https://example.com",
    "curl -I https://x",
    "curl -X GET https://x",
    "curl -X HEAD https://x",
    "curl -sS -o out.json https://x",
    'curl -H "Accept: json" https://x',
    "curl -u user:pass https://x",
    "curl --max-time 5 https://x",
    "curl -m 5 https://x",
    'curl -sSL -X GET -H "a: b" https://x',
    "curl -fsSL https://x | sh",
    "wget https://x",
    "wget -O out https://x",
    "iwr https://x",
    "irm https://x",
    "Invoke-WebRequest -Uri https://x",
    "Invoke-RestMethod -Method Get -Uri https://x",
    'python -c "print(1)"',
    'node -e "console.log(1)"',
    'echo "git push"',
    "echo done",
    "ls -la",
    "cat README.md",
    "grep -r push src/",
    'rg "git push" docs/',
    "bash script.sh",
    'bash -c "npm test"',
    'pwsh -Command "npm test"',
    "PowerShell -NoProfile -File build.ps1",
    'cmd /c "npm test"',
    "cd src && npm test",
    "npm test && echo ok",
    "npm test; echo done",
    "npm test || echo fail",
    "npm test | tail",
    "FOO=1 npm test",
    "env FOO=1 npm test",
    "time npm test",
    "timeout 60 npm test",
    "echo `date`",
    "echo $(git rev-parse HEAD)",
    "VAR=$(git log -1 --format=%H)",
    "# comment\nnpm test",
    "npm test # git push",
    'echo "a=b" > .env.local',
    "gem install x",
    "twine check dist/*",
    "docker images",
    "aws s3 ls",
    "ssh -T git@example.com",
    "",
    "   ",
    # unparseable but harmless -> allowed
    "echo it's",
    'echo "unbalanced',
    # --- reads of .git and writes to look-alike paths stay allowed
    "cat .git/HEAD",
    "cat .git/config",
    "ls -la .git/hooks",
    "cat < .git/HEAD",
    "wc -l < .git/HEAD",
    "cat .git/hooks/pre-commit",
    "grep -r push .git/",
    "find .git -name '*.lock'",
    "cp .git/config /tmp/backup-config",
    "cp .git/HEAD .factory/head.txt",
    "echo $(git rev-parse --git-dir)",
    "cat $(git rev-parse --git-dir)/HEAD",
    "cat \"$(git rev-parse --git-path HEAD)\"",
    'ls "$GIT_DIR"',
    "git rev-parse --git-dir",
    "git rev-parse --git-path hooks",
    "echo x > src/hooks/useThing.ts",
    "echo x > .github/workflows/ci.yml",
    "echo x > .gitignore",
    "echo x > .gitattributes",
    "echo x > .gitmodules",
    "echo x > docs/git/notes.md",
    "echo x > .git-blame-ignore-revs",
    "echo x > out.txt 2>&1",
    "npm test > log.txt",
    "npm test 2>&1 | tee .factory/test.log",
    "git log > .factory/log.txt",
    "git status >/dev/null 2>&1",
    "echo x >&2",
    "echo x 1>&2",
    "exec 3>&1",
    "cp a.txt b.txt",
    "mv src/a.py src/b.py",
    "ln -s ../shared src/shared",
    "install -m644 a.conf etc/a.conf",
    "touch src/__init__.py",
    "chmod +x scripts/run.sh",
    "mkdir -p build/out",
    "rm -rf node_modules",
    "sed -i 's/a/b/' src/x.py",
    "sed -n '1p' .git/config",
    "dd if=/dev/zero of=.factory/blob bs=1k count=1",
    "cd src && npm test",
    "cd .github/workflows && ls",
    "tee out.txt",
    "echo x > .env.local",
    "echo x > gitconfig.example",
    "cp .gitconfig.example ~/dotfiles-backup/x",
]


DENIED_POWERSHELL = [
    "Set-Content .git/hooks/pre-commit 'git push'",
    "Set-Content -Path .git\\hooks\\pre-commit -Value x",
    "Set-Content -LiteralPath .git/hooks/pre-commit -Value x",
    "Set-Content -Path:.git/hooks/pre-commit x",
    "sc .git/hooks/pre-commit x",
    "Add-Content .git/config x",
    "ac .git/config x",
    "'x' | Out-File .git/hooks/pre-commit",
    "'x' | Out-File -FilePath .git/hooks/pre-commit",
    "'x' | Out-File -Path .git/hooks/pre-commit -Encoding ascii",
    "'x' > .git/hooks/pre-commit",
    "'x' >> .git/hooks/pre-commit",
    "'x' 2> .git/hooks/pre-commit",
    "'x' *> .git/hooks/pre-commit",
    "'x' > \"$(git rev-parse --git-dir)/hooks/pre-commit\"",
    "'x' > (git rev-parse --git-dir)/hooks/pre-commit",
    "'x' > $env:GIT_DIR/hooks/pre-commit",
    "$h = git rev-parse --git-path hooks; 'x' > $h/pre-commit",
    "$h = (git rev-parse --git-dir); Set-Content $h/hooks/pre-commit x",
    "Copy-Item evil.ps1 .git/hooks/pre-commit",
    "Copy-Item evil.ps1 -Destination .git/hooks/",
    "cpi evil.ps1 .git/hooks/pre-commit",
    "Move-Item evil .git/hooks/pre-commit",
    "New-Item .git/hooks/pre-commit -ItemType File",
    "New-Item -Path .git/hooks -Name pre-commit",
    "ni .git/hooks/pre-commit",
    "Remove-Item -Recurse -Force .git",
    "Set-Location .git/hooks; Set-Content pre-commit x",
    "cd .git; 'x' > config",
    "sl .git",
    "Set-Content \"$(git rev-parse --git-dir)/hooks/pre-commit\" x",
    "'x' > $env:USERPROFILE\\.gitconfig",
    "Add-Content $HOME/.gitconfig '[core] hooksPath=/evil'",
    "'x' | Out-File ~/.config/gh/hosts.yml",
    "Set-Content ~/.ssh/config x",
    "cmd /c \"echo x > .git\\hooks\\pre-commit\"",
]

ALLOWED_POWERSHELL = [
    "Get-Content .git/HEAD",
    "cat .git/config",
    "gc .git/hooks/pre-commit",
    "Get-ChildItem .git/hooks",
    "Set-Content src/x.ts 'y'",
    "Set-Content -Path src\\x.ts -Value y",
    "'x' > out.txt",
    "'x' >> .factory/log.txt",
    "npm test > log.txt 2>&1",
    "npm test *> $null",
    "'x' 2>&1",
    "Copy-Item .git/config backup-config",
    "Copy-Item a.txt b.txt",
    "Move-Item src/a.ts src/b.ts",
    "New-Item -ItemType Directory build",
    "Remove-Item -Recurse -Force node_modules",
    "Set-Location src; npm test",
    "cd .github; dir",
    "Write-Host (git rev-parse --git-dir)",
    "$d = git rev-parse --git-dir; Write-Host $d",
    "Out-File -FilePath .factory/out.txt -InputObject x",
    "'x' > .gitignore",
    "foreach ($f in 1..3) { Write-Host $f }",
    "if ($x -eq 1) { npm test }",
]


@pytest.mark.parametrize("command", DENIED_SHELL)
def test_classify_shell_denies(command: str) -> None:
    decision = classify_shell(command)
    assert not decision.allow, command
    assert decision.reason, "denials must carry a reason the model can act on"


@pytest.mark.parametrize("command", ALLOWED_SHELL)
def test_classify_shell_allows(command: str) -> None:
    decision = classify_shell(command)
    assert decision.allow, f"{command!r} -> {decision.reason}"


@pytest.mark.parametrize("command", DENIED_POWERSHELL)
def test_classify_powershell_denies_git_dir_writes(command: str) -> None:
    decision = classify_shell(command, powershell=True)
    assert not decision.allow, command
    assert decision.reason, "denials must carry a reason the model can act on"


@pytest.mark.parametrize("command", ALLOWED_POWERSHELL)
def test_classify_powershell_allows_reads_and_worktree_writes(command: str) -> None:
    decision = classify_shell(command, powershell=True)
    assert decision.allow, f"{command!r} -> {decision.reason}"


def test_git_dir_write_denial_message_points_at_the_edit_tool() -> None:
    """Regression: ``echo 'git push' > .git/hooks/pre-commit`` followed by ``git commit`` ran a push
    the classifier never saw. Writes whose target is inside a git dir (or a git/gh/ssh config
    file) are denied with a message that names the alternative; reads of .git stay allowed."""
    reason = classify_shell("echo 'git push' > .git/hooks/pre-commit").reason
    assert "git" in reason.lower() and "Edit" in reason
    assert ".git" in classify_shell("cp evil .git/hooks/pre-commit").reason
    assert "rev-parse" in classify_shell("echo x > $(git rev-parse --git-dir)/hooks/x").reason
    assert "GIT_DIR" in classify_shell('echo x > "$GIT_DIR/hooks/x"').reason
    assert ".gitconfig" in classify_shell("echo x >> ~/.gitconfig").reason
    assert classify_shell("cat .git/hooks/pre-commit").allow
    assert classify_shell("cat $(git rev-parse --git-dir)/HEAD").allow


def test_redirections_are_stripped_before_the_command_is_classified() -> None:
    """The redirect target is checked and removed, so the remaining argv classifies as before."""
    assert not classify_shell("git push > out.txt").allow  # still a push
    assert not classify_shell("gh pr merge 1 2>&1 | tee log").allow
    assert classify_shell("git status > .factory/status.txt").allow
    assert classify_shell("> out.txt").allow  # bare truncation of a worktree file
    assert not classify_shell("> .git/hooks/pre-commit").allow
    assert not classify_shell("> .git/index").allow


def test_deny_messages_tell_the_model_what_to_do_instead() -> None:
    push = classify_shell("git push").reason
    assert "engine" in push and "push_ok" in push
    gh = classify_shell("gh pr comment 1 -b hi").reason
    assert "structured output" in gh
    assert "gh pr view" in gh or "read-only" in gh.lower()
    api = classify_shell("gh api -X POST repos/o/r").reason
    assert "GET" in api
    linear = classify_shell("curl https://api.linear.app/graphql").reason
    assert "structured output" in linear


def test_powershell_mode_treats_backtick_as_escape_and_still_catches_push() -> None:
    assert classify_shell('Write-Host "a`"b"; git push', powershell=True).allow is False
    assert classify_shell('Write-Host "hello `"world`""', powershell=True).allow is True
    assert classify_shell("& git push", powershell=True).allow is False
    assert classify_shell('& "C:\\Program Files\\Git\\cmd\\git.exe" push', powershell=True).allow is False
    assert classify_shell("git status; npm test", powershell=True).allow is True


def test_encoded_command_is_decoded_then_classified() -> None:
    safe = base64.b64encode("npm test".encode("utf-16-le")).decode()
    assert classify_shell(f"pwsh -EncodedCommand {safe}").allow is True
    assert classify_shell(f"pwsh -e {ENC_GIT_PUSH}").allow is False
    assert classify_shell(f"pwsh -ec {ENC_GIT_PUSH}").allow is False


def test_fail_closed_only_when_dangerous_token_present() -> None:
    assert classify_shell("git push 'oops").allow is False
    assert "parse" in classify_shell("git push 'oops").reason.lower()
    assert classify_shell("echo 'oops").allow is True


def test_env_denials_mirror_the_dash_c_denials() -> None:
    """Regression: a leading ``VAR=value`` used to be dropped unseen, so every git setting the ``-c``
    rules deny was reachable through GIT_CONFIG_* / GIT_CONFIG_GLOBAL / GIT_EDITOR / GIT_SSH_COMMAND."""
    pairs = [
        ("git -c alias.z='!git push' z", "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.z GIT_CONFIG_VALUE_0='!git push' git z"),
        ("git -c credential.helper=x fetch", "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0=x git fetch"),
        ("git -c core.sshCommand=x fetch", "GIT_SSH_COMMAND=x git fetch"),
        ("git -c core.editor=x commit", "GIT_EDITOR=x git commit"),
    ]
    for dash_c, env_form in pairs:
        assert not classify_shell(dash_c).allow, dash_c
        decision = classify_shell(env_form)
        assert not decision.allow, env_form
        assert "GIT_" in decision.reason or "engine" in decision.reason
    # the plain trigger stays allowed: it is the setting that is denied, not the subcommand
    assert classify_shell("git z").allow
    assert classify_shell("git commit -m x").allow


def test_env_denials_apply_in_powershell_mode_too() -> None:
    assert not classify_shell("$env:GIT_SSH_COMMAND = 'x'; git fetch", powershell=True).allow
    assert not classify_shell("$env:GH_CONFIG_DIR='x'; gh pr view 1", powershell=True).allow
    assert not classify_shell("Set-Item env:GIT_EDITOR 'x'; git commit", powershell=True).allow
    assert not classify_shell("[Environment]::SetEnvironmentVariable('GH_TOKEN', 'x')", powershell=True).allow
    assert classify_shell("$env:NODE_ENV = 'test'; npm test", powershell=True).allow
    assert classify_shell("Write-Host $env:GIT_SSH_COMMAND", powershell=True).allow


# --------------------------------------------------------------------------------- ToolPolicy
@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    wt = tmp_path / "wt"
    (wt / "src").mkdir(parents=True)
    (wt / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (wt / ".env").write_text("SECRET=1\n", encoding="utf-8")
    return wt


@pytest.fixture
def profile(tmp_path: Path) -> Profile:
    pdir = tmp_path / "profiles" / "personal"
    pdir.mkdir(parents=True)
    (pdir / ".env").write_text("LINEAR_API_KEY=lin_api_secret\n", encoding="utf-8")
    p = Profile(name="personal")
    p.profile_dir = pdir
    p.secrets = {"LINEAR_API_KEY": "lin_api_secret"}
    return p


@pytest.fixture
def policy(worktree: Path, profile: Profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToolPolicy:
    # pytest's tmp_path lives under the OS temp dir, which the policy allows for writes. Pin the
    # policy's notion of "temp dir" to a sibling so paths "outside the worktree" really are outside.
    ostmp = tmp_path / "ostmp"
    ostmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(ostmp))
    return ToolPolicy(worktree, profile)


def test_edit_tools_confined_to_worktree(policy: ToolPolicy, worktree: Path, tmp_path: Path) -> None:
    assert policy.check("Write", {"file_path": str(worktree / "src" / "new.py"), "content": ""}).allow
    assert policy.check("Edit", {"file_path": "src/a.py", "old_string": "x", "new_string": "y"}).allow
    assert policy.check("MultiEdit", {"file_path": str(worktree / "README.md")}).allow
    assert policy.check("NotebookEdit", {"notebook_path": str(worktree / "nb.ipynb")}).allow
    # The OS temp dir is NOT writable (found by the real-SDK smoke test): scratch goes in .factory/.
    tmp_file = Path(tempfile.gettempdir()) / "factory-scratch.txt"
    assert not policy.check("Write", {"file_path": str(tmp_file)}).allow
    assert policy.check("Write", {"file_path": str(worktree / ".factory" / "scratch.txt")}).allow

    outside = tmp_path / "elsewhere" / "x.py"
    d = policy.check("Write", {"file_path": str(outside), "content": ""})
    assert not d.allow and "worktree" in d.reason
    # traversal out of the worktree
    assert not policy.check("Edit", {"file_path": str(worktree / ".." / "escape.py")}).allow
    assert not policy.check("Write", {"file_path": "../escape.py"}).allow
    assert not policy.check("NotebookEdit", {"notebook_path": str(outside)}).allow
    # missing path -> deny
    assert not policy.check("Write", {"content": "x"}).allow
    # .git internals (hooks could push on commit)
    assert not policy.check("Write", {"file_path": str(worktree / ".git" / "hooks" / "pre-commit")}).allow


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive paths are a Windows property")
def test_edit_path_check_is_case_insensitive_on_windows(policy: ToolPolicy, worktree: Path) -> None:
    upper = str(worktree / "src" / "a.py").upper()
    assert policy.check("Edit", {"file_path": upper}).allow


def test_read_tools_anywhere_except_secrets(policy: ToolPolicy, worktree: Path, tmp_path: Path, profile: Profile) -> None:
    assert policy.check("Read", {"file_path": str(worktree / "src" / "a.py")}).allow
    assert policy.check("Read", {"file_path": str(worktree / ".env")}).allow  # inside worktree is fine
    assert policy.check("Glob", {"pattern": "**/*.py"}).allow
    assert policy.check("Grep", {"pattern": "push", "path": str(tmp_path / "other")}).allow
    assert policy.check("Read", {"file_path": str(tmp_path / "other" / "notes.md")}).allow
    # a recursive search rooted above the profiles dir would sweep up its .env files
    assert not policy.check("Grep", {"pattern": "KEY", "path": str(tmp_path)}).allow

    other_env = tmp_path / "other" / ".env"
    assert not policy.check("Read", {"file_path": str(other_env)}).allow
    assert not policy.check("Read", {"file_path": str(tmp_path / "other" / ".env.production")}).allow
    assert profile.profile_dir is not None
    assert not policy.check("Read", {"file_path": str(profile.profile_dir / ".env")}).allow
    assert not policy.check("Read", {"file_path": str(profile.profile_dir / "factory.toml")}).allow
    assert not policy.check("Grep", {"pattern": "KEY", "path": str(profile.profile_dir.parent)}).allow
    assert not policy.check("Glob", {"pattern": "*", "path": str(profile.profile_dir)}).allow
    assert not policy.check("Read", {"file_path": str(Path.home() / ".config" / "gh" / "hosts.yml")}).allow
    assert not policy.check("Read", {"file_path": str(Path.home() / ".ssh" / "id_ed25519")}).allow


def test_extra_read_dirs_allow_secret_names_inside_them(worktree: Path, profile: Profile, tmp_path: Path) -> None:
    extra = tmp_path / "shared"
    extra.mkdir()
    policy = ToolPolicy(worktree, profile, extra_read_dirs=[extra])
    assert policy.check("Read", {"file_path": str(extra / ".env")}).allow


def test_shell_tools_use_classifier(policy: ToolPolicy) -> None:
    assert not policy.check("Bash", {"command": "git push"}).allow
    assert policy.check("Bash", {"command": "uv run pytest -q"}).allow
    assert not policy.check("PowerShell", {"command": "git push"}).allow
    assert not policy.check("PowerShell", {"command": "Invoke-RestMethod -Method Post -Uri https://x"}).allow
    assert policy.check("PowerShell", {"command": "Get-ChildItem"}).allow
    assert not policy.check("Bash", {}).allow
    assert not policy.check("Bash", {"command": ["git", "push"]}).allow


def test_unknown_and_mcp_tools_denied_with_allowed_list(policy: ToolPolicy) -> None:
    d = policy.check("mcp__linear__create_issue", {"title": "x"})
    assert not d.allow
    assert "Bash" in d.reason and "Read" in d.reason and "mcp__linear__create_issue" in d.reason
    assert not policy.check("MCP__Slack__post", {}).allow
    assert not policy.check("SomethingNew", {}).allow
    assert not policy.check("WebFetch", {"url": "https://example.com"}).allow  # off by default
    assert not policy.check("AskUserQuestion", {"question": "?"}).allow
    for name in ("TodoWrite", "Task", "Agent", "Glob", "Grep"):
        assert policy.check(name, {}).allow, name


def test_web_tools_follow_profile_flag(worktree: Path, profile: Profile) -> None:
    profile.policy.allow_web_tools = True
    policy = ToolPolicy(worktree, profile)
    assert policy.check("WebFetch", {"url": "https://docs.example.com"}).allow
    assert policy.check("WebSearch", {"query": "pytest fixtures"}).allow
    assert not policy.check("WebFetch", {"url": "https://api.linear.app/graphql"}).allow
    assert not policy.check("WebFetch", {"url": "https://hooks.slack.com/x"}).allow


def test_extra_allowed_tools(worktree: Path, profile: Profile) -> None:
    profile.policy.extra_allowed_tools = ["Skill"]
    policy = ToolPolicy(worktree, profile)
    assert policy.check("Skill", {"skill": "x"}).allow
    assert not policy.check("mcp__x__y", {}).allow  # never, even if listed


async def test_can_use_tool_returns_sdk_results(policy: ToolPolicy, worktree: Path) -> None:
    allow = await policy.can_use_tool("Bash", {"command": "npm test"}, None)
    assert isinstance(allow, PermissionResultAllow)
    deny = await policy.can_use_tool("Bash", {"command": "git push"}, None)
    assert isinstance(deny, PermissionResultDeny)
    assert deny.message and deny.interrupt is False
    deny_write = await policy.can_use_tool("Write", {"file_path": str(worktree.parent / "x")}, None)
    assert isinstance(deny_write, PermissionResultDeny)


async def test_pre_tool_use_hook_output_shape(policy: ToolPolicy) -> None:
    hook_input = {
        "hook_event_name": "PreToolUse",
        "session_id": "s",
        "transcript_path": "t",
        "cwd": "c",
        "tool_name": "Bash",
        "tool_input": {"command": "gh pr merge 1"},
        "tool_use_id": "toolu_1",
    }
    out = await policy.pre_tool_use_hook(hook_input, "toolu_1", {"signal": None})
    assert out == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": out["hookSpecificOutput"]["permissionDecisionReason"],
        }
    }
    assert "engine" in out["hookSpecificOutput"]["permissionDecisionReason"]
    # allowed calls return no opinion so can_use_tool still sees them (an allow here would skip it)
    ok = await policy.pre_tool_use_hook({**hook_input, "tool_input": {"command": "npm test"}}, "toolu_2", {})
    assert ok == {}
    garbage = await policy.pre_tool_use_hook(None, None, {})  # garbage in -> fail closed, no crash
    assert garbage["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hooks_registers_pre_tool_use_for_all_tools(policy: ToolPolicy) -> None:
    hooks = policy.hooks()
    assert list(hooks) == ["PreToolUse"]
    matcher = hooks["PreToolUse"][0]
    assert isinstance(matcher, HookMatcher)
    assert matcher.matcher is None
    assert matcher.hooks == [policy.pre_tool_use_hook]
    # An explicit timeout (HookMatcher.timeout is in seconds; the SDK default is 60) so a wedged
    # hook cannot hold a worker session open for the whole SDK default.
    assert matcher.timeout == PRE_TOOL_USE_HOOK_TIMEOUT_S == 30.0


# ------------------------------------------------------------------------------- push policy
def _ci(**overrides) -> CiOutput:
    base = dict(
        status="done", summary="fixed", push_ok=True, head_verified=True,
        verification=[VerificationRun(command="pytest", exit_code=0, passed=True)],
    )
    base.update(overrides)
    return CiOutput(**base)


def test_evaluate_push_happy_path() -> None:
    d = evaluate_push(Kind.CI, _ci(), ["src/a.py"], ["**/auth/**"])
    assert d == PushDecision(allowed=True, reasons=[])


def test_evaluate_push_requires_push_ok_verification_and_head() -> None:
    d = evaluate_push("ci", _ci(push_ok=False, push_reasons=["tests flaky"]), [], [])
    assert not d.allowed and any("push_ok" in r and "tests flaky" in r for r in d.reasons)
    d = evaluate_push("ci", _ci(verification=[VerificationRun(command="pytest", exit_code=1, passed=False)]), [], [])
    assert not d.allowed and any("verification" in r for r in d.reasons)
    d = evaluate_push("ci", _ci(verification=[]), [], [])
    assert not d.allowed
    d = evaluate_push("ci", _ci(head_verified=False), [], [])
    assert not d.allowed and any("HEAD" in r for r in d.reasons)
    d = evaluate_push("ci", _ci(status="blocked"), [], [])
    assert not d.allowed and any("blocked" in r for r in d.reasons)


def test_evaluate_push_sensitive_globs_and_codeowners_style_paths() -> None:
    globs = ["**/auth/**", ".github/workflows/**", "**/*.pem", "**/migrations/**"]
    d = evaluate_push(Kind.CI, _ci(), ["src/auth/login.py", "README.md"], globs)
    assert not d.allowed and "src/auth/login.py" in d.reasons[0]
    d = evaluate_push(Kind.CI, _ci(), [".github/workflows/ci.yml"], globs)
    assert not d.allowed
    d = evaluate_push(Kind.CI, _ci(), ["certs\\server.pem"], globs)  # backslashes normalised
    assert not d.allowed
    d = evaluate_push(Kind.CI, _ci(), ["src/author.py", "docs/auth.md"], globs)
    assert d.allowed


def test_evaluate_push_rc_rejects_skipped_comments() -> None:
    def rc(disp: str) -> RcOutput:
        return RcOutput(
            status="done", summary="s", push_ok=True, head_verified=True,
            verification=[VerificationRun(command="npm test", exit_code=0, passed=True)],
            comments=[CommentDisposition(comment_id="c1", author="bob", disposition=disp)],
        )

    assert evaluate_push(Kind.RC, rc("addressed"), ["a"], []).allowed
    assert evaluate_push(Kind.RC, rc("not_actionable"), ["a"], []).allowed
    d = evaluate_push(Kind.RC, rc("skipped"), ["a"], [])
    assert not d.allowed and "c1" in d.reasons[0]


def test_evaluate_push_conflict_rules() -> None:
    def conflict(**kw) -> ConflictOutput:
        base = dict(
            status="done", summary="s", push_ok=True, head_verified=True, all_resolved=True,
            verification=[VerificationRun(command="npm test", exit_code=0, passed=True)],
            hunks=[HunkResolution(file="a.py", classification="trivial", resolution="took ours")],
        )
        base.update(kw)
        return ConflictOutput(**base)

    assert evaluate_push(Kind.CONFLICT, conflict(), ["a.py"], []).allowed
    assert not evaluate_push(Kind.CONFLICT, conflict(all_resolved=False), ["a.py"], []).allowed
    unsafe = conflict(hunks=[HunkResolution(file="b.py", classification="unsafe", resolution="?")])
    d = evaluate_push(Kind.CONFLICT, unsafe, ["b.py"], [])
    assert not d.allowed and "b.py" in d.reasons[0]
    d = evaluate_push(Kind.CONFLICT, conflict(), ["a.py"], [], conflict_markers=True)
    assert not d.allowed and "markers" in d.reasons[0]


def _build() -> BuildOutput:
    return BuildOutput(
        status="done", summary="s", push_ok=True, pr_title="t", pr_summary="b",
        verification=[VerificationRun(command="pytest", exit_code=0, passed=True)],
    )


def test_evaluate_push_build_does_not_need_head_verified() -> None:
    assert evaluate_push(Kind.BUILD, _build(), ["src/x.py"], ["**/auth/**"]).allowed


def test_matches_any_gitwildmatch() -> None:
    assert matches_any("src/auth/x.py", ["**/auth/**"])
    assert matches_any("auth/x.py", ["**/auth/**"])
    assert matches_any("a/b/c.key", ["**/*.key"])
    assert matches_any("CODEOWNERS", ["**/CODEOWNERS"])
    assert matches_any(".github/CODEOWNERS", ["**/CODEOWNERS"])
    assert matches_any("db/migrations/0001.sql", ["**/migrations/**"])
    assert not matches_any("src/x.py", ["**/auth/**", "*.md"])
    assert not matches_any("src/x.py", [])
    assert matches_any("/src/x.py", ["src/*.py"])
    assert matches_any("src\\x.py", ["src/*.py"])


# ------------------------------------------------------------------- sensitive globs / pr body
class _Override:
    def __init__(self, globs: list[str]):
        self.sensitive_globs = globs


class _Repo:
    def __init__(self, path: Path, globs: list[str] | None = None):
        self.slug = "o/r"
        self.name = "r"
        self.path = path
        self.override = _Override(globs or [])


CODEOWNERS_FIXTURE = Path(__file__).parent / "fixtures" / "repos" / "codeowners"


def _codeowners_repo(tmp_path: Path, text: str, globs: list[str] | None = None) -> _Repo:
    (tmp_path / ".github").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".github" / "CODEOWNERS").write_text(text, encoding="utf-8")
    return _Repo(tmp_path, globs)


def test_sensitive_globs_for_merges_policy_and_override_only(tmp_path: Path, profile: Profile) -> None:
    """Regression: CODEOWNERS used to be folded into the glob list at *pattern* level, so
    ``* @acme/core-team`` followed by ``/docs/ @me-login`` kept ``*`` sensitive and refused a push
    that only touched ``docs/guide.md`` although GitHub assigns it to me. CODEOWNERS is now applied
    per changed path (``foreign_owned_changes``); the glob list carries policy + override only."""
    repo = _codeowners_repo(tmp_path, "* @acme/core-team\n/docs/ @me-login\n", ["**/billing-extra/**", "**/auth/**"])
    profile.policy.my_teams = []
    profile.github.login = "me-login"
    globs = sensitive_globs_for(repo=repo, profile=profile)
    assert "*" not in globs and "/docs/" not in globs
    assert globs[: len(profile.policy.sensitive_globs)] == profile.policy.sensitive_globs
    assert "**/billing-extra/**" in globs and globs.count("**/auth/**") == 1  # deduped
    # the keyword stays accepted so existing callers keep working; it no longer changes the globs
    assert sensitive_globs_for(profile, repo, my_login="someone-else") == globs
    assert sensitive_globs_for(profile, _Repo(tmp_path / "no-codeowners")) == list(profile.policy.sensitive_globs)


def test_foreign_owned_changes_and_evaluate_push_use_per_path_codeowners(tmp_path: Path, profile: Profile) -> None:
    repo = _codeowners_repo(tmp_path, "* @acme/core-team\n/docs/ @me-login\n")
    profile.policy.my_teams = []
    profile.github.login = "me-login"
    globs = sensitive_globs_for(profile, repo)

    foreign = foreign_owned_changes(profile, repo, ["docs/guide.md", "README.md"])
    assert foreign == ["README.md"]
    refused = evaluate_push(Kind.CI, _ci(), ["docs/guide.md", "README.md"], globs, foreign_owned=foreign)
    assert not refused.allowed
    assert refused.reasons == ["changed files owned by another team per CODEOWNERS: README.md"]

    assert foreign_owned_changes(profile, repo, ["docs/guide.md"]) == []
    assert evaluate_push(Kind.CI, _ci(), ["docs/guide.md"], globs, foreign_owned=[]).allowed  # docs/ is mine
    assert evaluate_push(Kind.CI, _ci(), ["docs/guide.md"], globs).allowed  # keyword optional
    # my team owning the catch-all makes README.md mine too
    profile.policy.my_teams = ["@acme/core-team"]
    assert foreign_owned_changes(profile, repo, ["docs/guide.md", "README.md"]) == []


def test_foreign_owned_changes_login_resolution_fails_closed(tmp_path: Path, profile: Profile) -> None:
    """The profile's ``login = ""`` (detect via gh) must not make a path mine by accident: with no
    login at all every owned path is foreign; a resolved ``my_login`` wins over an empty profile
    login; a configured profile login is used when the caller passes none."""
    repo = _codeowners_repo(tmp_path, "* @acme/core-team\n/docs/ @me-login\n/generated/\n")
    profile.policy.my_teams = []
    files = ["docs/guide.md", "README.md", "generated/out.js"]
    profile.github.login = ""
    assert foreign_owned_changes(profile, repo, files) == ["docs/guide.md", "README.md"]  # unowned is never foreign
    assert foreign_owned_changes(profile, repo, files, my_login="me-login") == ["README.md"]
    profile.github.login = "me-login"
    assert foreign_owned_changes(profile, repo, files) == ["README.md"]
    assert foreign_owned_changes(profile, repo, files, my_login="someone-else") == ["docs/guide.md", "README.md"]
    assert foreign_owned_changes(profile, repo, ["docs\\sub\\x.md", "/docs/rooted.md"]) == []  # path forms


def test_foreign_owned_changes_against_the_codeowners_fixture(profile: Profile) -> None:
    repo = _Repo(CODEOWNERS_FIXTURE)
    profile.policy.my_teams = []
    profile.github.login = "me-login"
    files = ["docs/guide.md", "docs/img/logo.png", "README.md", "src/auth/login.py", "src/billing/invoice.py",
             "infra/main.tf", "generated/out.js"]  # fmt: skip
    assert foreign_owned_changes(profile, repo, files) == ["README.md", "src/billing/invoice.py", "infra/main.tf"]
    reasons = evaluate_push(Kind.BUILD, _build(), files, [], foreign_owned=foreign_owned_changes(profile, repo, files)).reasons
    assert reasons == ["changed files owned by another team per CODEOWNERS: README.md, src/billing/invoice.py, infra/main.tf"]


def test_foreign_owned_changes_without_codeowners_is_empty(tmp_path: Path, profile: Profile) -> None:
    profile.github.login = ""
    assert foreign_owned_changes(profile, _Repo(tmp_path), ["src/auth/login.py"]) == []
    assert evaluate_push(Kind.CI, _ci(), ["src/x.py"], [], foreign_owned=None).allowed


def test_may_edit_pr_body_only_factory_prs() -> None:
    class FakeState:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def is_factory_pr(self, repo: str, number: int) -> bool:
            self.calls.append((repo, number))
            return (repo, number) == ("o/r", 7)

    state = FakeState()
    assert may_edit_pr_body(state, "o/r", 7) is True
    assert may_edit_pr_body(state, "o/r", 8) is False
    assert may_edit_pr_body(state, "o/other", 7) is False
    assert state.calls[0] == ("o/r", 7)


def test_tool_decision_model() -> None:
    assert ToolDecision(allow=True).reason == ""
    assert ToolDecision(allow=False, reason="no").model_dump() == {"allow": False, "reason": "no"}


# -------------------------------------------------------------------------------- read-only
def test_read_only_policy_denies_edit_tools_with_a_message(worktree: Path, profile: Profile) -> None:
    policy = ToolPolicy(worktree, profile, read_only=True)
    assert policy.read_only is True and ToolPolicy(worktree, profile).read_only is False
    inside = str(worktree / "src" / "a.py")
    for tool, tool_input in (
        ("Edit", {"file_path": inside, "old_string": "x", "new_string": "y"}),
        ("Write", {"file_path": inside, "content": ""}),
        ("MultiEdit", {"file_path": inside, "edits": []}),
        ("NotebookEdit", {"notebook_path": str(worktree / "nb.ipynb")}),
    ):
        decision = policy.check(tool, tool_input)
        assert not decision.allow, tool
        assert "read-only" in decision.reason and "structured output" in decision.reason
    # reads and plain tools are unaffected
    assert policy.check("Read", {"file_path": inside}).allow
    assert policy.check("Glob", {"pattern": "**/*.py"}).allow
    assert policy.check("TodoWrite", {}).allow


def test_read_only_policy_still_classifies_shell(worktree: Path, profile: Profile) -> None:
    policy = ToolPolicy(worktree, profile, read_only=True)
    assert policy.check("Bash", {"command": "uv run pytest -q"}).allow
    assert policy.check("Bash", {"command": "git diff origin/main...HEAD"}).allow
    assert not policy.check("Bash", {"command": "git push"}).allow
    assert not policy.check("PowerShell", {"command": "gh pr comment 1 -b hi"}).allow
    assert not policy.check("mcp__x__y", {}).allow


async def test_read_only_policy_denies_in_both_sdk_adapters(worktree: Path, profile: Profile) -> None:
    policy = ToolPolicy(worktree, profile, read_only=True)
    deny = await policy.can_use_tool("Write", {"file_path": str(worktree / "x.py"), "content": ""}, None)
    assert isinstance(deny, PermissionResultDeny) and "read-only" in deny.message
    hook = await policy.pre_tool_use_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": str(worktree / "src" / "a.py")}}, "toolu_1", {}
    )
    assert hook["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "read-only" in hook["hookSpecificOutput"]["permissionDecisionReason"]
    assert await policy.pre_tool_use_hook({"tool_name": "Bash", "tool_input": {"command": "npm test"}}, "t2", {}) == {}


def test_structured_output_tool_is_allowed(policy: ToolPolicy) -> None:
    # Regression (real-SDK smoke test): the CLI delivers output_format results through the
    # StructuredOutput tool; denying it made every worker finish without output.
    assert policy.check("StructuredOutput", {"status": "done"}).allow
