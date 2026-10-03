# Fixture project

Short description. `npm run dev` appears here in prose and must be ignored.

## Setup

```bash
npm ci
```

## Commands

- `npm run dev` — Vite dev server on port 5177
- `npm run typecheck` — `tsc --noEmit`
- `npm run db:migrate` — apply migrations (not verification)
- `src/core/tuning.ts` — single source of truth

## Verification

Run these before pushing:

```bash
$ npm run lint   # eslint
npm test
cd apps/api && npm run smoke
```

1. `uv run pytest` for the python side
2. The second item has `no runner here` in code

### Notes

```
# a comment line only
```

## Architecture

```bash
npm run build   # not under a verification heading, ignored
```
