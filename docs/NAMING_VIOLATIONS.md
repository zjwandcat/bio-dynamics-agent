# Naming Conventions Audit (Task G.1)

Audit date: 2026-07-07
Scope: `bio-dynamics-agent/` repository (v4 RC).

This document records naming-convention exceptions found during the G.1
structural cleanup. **No files were renamed** — renaming existing modules is
too risky for the RC window (would invalidate many import paths). Violations
are recorded here for a future hardening pass.

## Conventions

| Layer            | Expected convention   | Status |
|------------------|-----------------------|--------|
| Python files     | `snake_case.py`       | OK     |
| Python packages  | `snake_case/`         | OK     |
| Frontend routes  | `kebab-case` (Next.js)| OK     |
| Env vars         | `UPPER_CASE`          | OK     |
| Components       | `PascalCase.tsx`      | See exceptions below |

## Findings

### Python — no violations
All modules under `backend/app/` and its subpackages use `snake_case.py`
(`benchmark_runner.py`, `bio_db_client.py`, `ode_renderer_v2.py`,
`reaction_ir_v2/`, `validation_v2/`, `sbml_grounder/`, etc.). Version suffixes
such as `_v2` / `_v3` / `_v4` are kept and remain snake_case-compliant.

### Env vars — no violations
Every variable in `backend/app/config.py` and `backend/.env.example` is
`UPPER_CASE` (e.g. `OPENAI_API_KEY`, `V4_SCIENTIFIC_LAYER_ENABLED`,
`CHROMA_PERSIST_DIR`).

### Frontend routes — no violations
`app/benchmarks/`, `app/workspace/`, `app/report/[id]/` are lowercase single
words; no multi-word route requires kebab-case today.

### Component files — intentional exceptions (shadcn/ui primitives)
`frontend/components/ui/` ships lowercase / kebab-case `.tsx` files that
violate the `PascalCase.tsx` component convention:

- `avatar.tsx`
- `badge.tsx`
- `button.tsx`
- `card.tsx`
- `input.tsx`
- `scroll-area.tsx`
- `separator.tsx`
- `tabs.tsx`
- `textarea.tsx`

These are auto-generated **shadcn/ui** primitives whose filenames are dictated
by the `shadcn` CLI and the `components.json` registry alias (`@/components/ui`).
Renaming them would break the registry contract and every import across the app.
They are therefore treated as an **intentional, framework-mandated exception**,
not actionable violations.

### Non-component `.ts` utilities — OK (not a violation)
Lowercase `.ts` files such as `frontend/lib/api.ts`, `frontend/lib/store.ts`,
`frontend/lib/utils.ts`, `frontend/components/hypothesis/types.ts`, and
`frontend/components/simulation/shared.ts` are utility modules, not components;
lowercase is the correct convention for them.

## Recommendation
No renames required for RC. The shadcn/ui primitives should remain lowercase to
preserve the registry contract. If a future hardening pass adopts a strict
`PascalCase.tsx` rule, it must exclude `components/ui/` (or regenerate the
registry alias accordingly).
