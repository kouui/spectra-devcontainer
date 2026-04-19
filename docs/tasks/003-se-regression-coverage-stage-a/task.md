# Task: SE regression coverage expansion — Stage A (6 new e2e cases)

> **Status:** Done
> **Owner:** kouui
> **Created:** 2026-04-19
> **Last Updated:** 2026-04-19

## Objective

Expand `tests/regression/test_reg_e2e_SE.py` with 6 additional end-to-end cases so that the three public SE entry points in `src/spectra/Function/SEquil/SELib.py` (`cal_SE_with_Nh_Te_`, `cal_SE_with_Ne_Te_`, `cal_SE_with_Pg_Te_`) are each covered for all three canonical atoms (H, He, Ca_II). All new reference values are generated on the behavior-locking baseline commit `a84128b` via a temporary git worktree, mirroring the AtomIO methodology from task 002. Expected outcome: zero numeric drift between `a84128b` and current `main` on the SE pipeline.

## Background & Context

PR #10 (`f10482c`) landed full-field AtomIO load regression for 8 configs. However the SE computation path is still thinly covered:

- `test_reg_SEsolver.py` tests only the three low-level kernels in `Atomic/SEsolver.py`
- `test_reg_e2e_SE.py` has only 3 e2e cases (H × Nh_Te, H × Ne_Te, He × Ne_Te), all at a single point (`Te=7000 K`, `Vt=5e5`, `Vd=0`)
- `cal_SE_with_Pg_Te_` has **zero** coverage
- Ca_II (an atom whose photoionization uses the EXPERIMENT interpolation branch) has **zero** SE coverage

Stage B (Te parameter sweep) and Stage C (private helper unit tests) from the original handoff are explicitly deferred to follow-up tasks.

- Handoff: `tmp/handoff.md`
- Predecessor task (methodology): `docs/tasks/002-atom-io-regression-and-unbound-cleanup/`
- Behavior-lock baseline: commit `a84128b` (`docs: add local module development info to README`)

## Requirements

### Functional Requirements

1. Add 6 new e2e tests covering the `(entry, atom)` matrix:
   | entry \ atom | H | He | Ca_II |
   |---|---|---|---|
   | `cal_SE_with_Nh_Te_` | existing | **new** | **new** |
   | `cal_SE_with_Ne_Te_` | existing | existing | **new** |
   | `cal_SE_with_Pg_Te_` | **new** | **new** | **new** |
2. Each new test asserts **fields that the entry actually mutates for that atom type** (`rtol=1e-8`). Per review findings (subagent + codex), asserting `atmos.Ne` for non-H non-Pg_Te entries is an identity check and provides no regression signal. Final breakdown:
   - H × Pg_Te: 4 fields (`n_SE`, `n_LTE`, `Ne`, `Ntotal`) — `Ne` is iterated by the self-consistent loop
   - He/Ca_II × {Nh_Te, Pg_Te}: 3 fields (`n_SE`, `n_LTE`, `Ntotal`) — `Ne` is the input, unchanged
   - Ca_II × Ne_Te: 2 fields (`n_SE`, `n_LTE`) — matches existing H/He Ne_Te convention
3. Reference values for all 6 new cases are generated on commit `a84128b` via a one-shot script `scripts/gen_se_reference.py`, then merged into `tests/regression/reference_values.json` under keys `E2E.<atom>_SE_<entry>.<field>`.
4. Running `pytest tests/regression/test_reg_e2e_SE.py -v` on current `main` passes all 9 cases (3 existing + 6 new) with **zero diff** against the `a84128b`-generated references.

### Non-Functional Requirements

- **Correctness-first:** If any drift appears between `a84128b` and `main`, stop and open `drift-audit.md` to blame the offending commit before accepting any new reference.
- **Minimal-diff policy:** Existing 3 tests' assertions remain untouched (no retro-adding `Ntotal` to old cases). Only new cases assert the full 4-field set.
- **No business-logic changes:** `SELib.py`, `Atomic/*`, `Struct/*` are read-only for this task.
- **Deterministic:** Pg_Te entry uses a self-consistent loop with 1% relative tolerance; we expect bit-identical output between the two commits because the algorithm is deterministic. `rtol=1e-8` should hold.

## Scope

### In Scope

- [x] Extend `tests/regression/test_reg_e2e_SE.py` with 6 new test methods (class-per-atom organization, no parametrize)
- [x] Add `scripts/gen_se_reference.py` — one-shot generator for the 6 new `E2E.*` keys
- [x] Merge new keys into `tests/regression/reference_values.json`
- [x] Produce a `drift-audit.md` **only if** `a84128b` vs `main` diverge numerically — **not triggered**, zero drift confirmed (see Audit Evidence).

### Out of Scope (Boundaries)

> Items explicitly excluded. Do NOT touch these.

- **`src/spectra/Function/SEquil/SELib.py`** — Behavior is locked; only read. No fix for the suspected bug at line 117 (`atmos.Nh = Ng` outside while loop).
- **`src/spectra/Function/SEquil/SELib.py:75` `cal_SE_with_Pg_Te_single_Atom_`** — Not one of the 3 public entries in scope; contains an active `print`, but we don't call it.
- **`src/spectra/Atomic/*`** — Kernel-level; read-only.
- **`src/spectra/Function/Icp/`** — Whole directory excluded (confirmed by user).
- **`cal_SE_with_Pg_Te_Ne_single_Atom_`** — Icp-only, excluded.
- **AtmosphereC1D 1D path (`init_VAL_`)** — Not part of SE 0-D e2e.
- **SlabModel / CloudModel** — Covered by `test_reg_e2e_CloudModel.py`, unchanged.
- **Existing 3 e2e cases' assertions** — Keep diff minimal; don't retro-add `Ntotal`.
- **Stage B (Te sweep)** — Deferred to follow-up task.
- **Stage C (private helper unit tests)** — Deferred to follow-up task.
- **`SELib.py` active `print` at line 98** — In `cal_SE_with_Pg_Te_single_Atom_` which is out of scope; no cleanup.

## Acceptance Criteria

- [x] 6 new test methods added under `TestHydrogenSE`, `TestHeliumSE`, `TestCaIISE` in `test_reg_e2e_SE.py`
- [x] All 9 tests in `test_reg_e2e_SE.py` pass (3 existing + 6 new) with `rtol=1e-8`
- [x] `scripts/gen_se_reference.py` committed and runnable via `uv run --extra dev python scripts/gen_se_reference.py`
- [x] `tests/regression/reference_values.json` contains exactly 18 new keys, sized per per-case mutation:
  - `E2E.H_SE_Pg_Te.{n_SE, n_LTE, Ne, Ntotal}` → 4 keys
  - `E2E.He_SE_Nh_Te.{n_SE, n_LTE, Ntotal}` → 3 keys
  - `E2E.He_SE_Pg_Te.{n_SE, n_LTE, Ntotal}` → 3 keys
  - `E2E.Ca_II_SE_Nh_Te.{n_SE, n_LTE, Ntotal}` → 3 keys
  - `E2E.Ca_II_SE_Ne_Te.{n_SE, n_LTE}` → 2 keys
  - `E2E.Ca_II_SE_Pg_Te.{n_SE, n_LTE, Ntotal}` → 3 keys
  Existing 3 cases untouched.
- [x] Full regression suite passes: `uv run --extra dev pytest tests/regression/ -q` — 261 passed.
- [x] `pre-commit run --all-files` green (verified via per-commit hook throughout).
- [x] Zero diff between `a84128b` and `main` on the new reference (or drift-audit.md exists with blame + decision) — bit-identical.
- [x] Commit message: `test: add e2e regression for SELib main entries (Stage A)` (or similar).

## Dependencies

| Dependency | Owner | Status | Notes |
|------------|-------|--------|-------|
| PR #10 merged | — | Done (`f10482c`) | Provides AtomIO safety net |
| `a84128b` buildable via `uv sync --extra dev` | — | Verified in task 002 | Takes ~30-60s |
| Stage B / Stage C tasks | kouui | Not started | Independent follow-ups |

## Risks & Open Questions

- [ ] **R1 — Pg_Te H path drift between `a84128b` and `main`.** The loop uses `< 0.01` convergence, but the algorithm is deterministic. Expected zero drift. If drift appears, blame via `git log a84128b..main -- src/spectra/Function/SEquil src/spectra/Atomic` and decide accept/revert.
- [ ] **R2 — Pg_Te for non-H atoms ignores `Pg`.** The function only computes `Nh` from `Pg` when `is_hydrogen=True` (line 138-140). For He / Ca_II, `atmos.Nh` and `atmos.Ne` must be provided externally; `Pg` is effectively unused. Tests will reflect this: we set sensible `Nh`/`Ne` pre-call and assign `Pg` a placeholder. Not a bug — just a documented quirk of the entry.
- [ ] **R3 — Reference fragment merge.** Manually merging `scripts/gen_se_reference.py`'s JSON fragment into `reference_values.json` risks ordering/whitespace churn. Mitigation: load the existing JSON in the script, update in place, re-dump with same `indent=2 sort_keys=True` settings to match the existing file's style.
- [ ] **R4 — Ca_II atmosphere params.** No prior Ca_II SE case to copy from. Decision: `Te=7000 K, Nh=1e12, Ne=1e11` for Nh_Te / Ne_Te entries; `Pg ≈ 1.8 dyn/cm²` for Pg_Te (though Pg is ignored for non-H, so irrelevant). Sanity-check by running once and confirming `n_SE` sum ≈ 1.
- [ ] **R5 — No open question on methodology.** Same flow as task 002 Stage 1.

## Audit Evidence (zero drift)

Generation on `a84128b` worktree (`uv run python scripts/gen_se_reference.py`):

```
generated E2E.H_SE_Pg_Te            n_SE.sum()=1.000000  (+4 keys)
generated E2E.He_SE_Nh_Te           n_SE.sum()=1.000000  (+3 keys)
generated E2E.He_SE_Pg_Te           n_SE.sum()=1.000000  (+3 keys)
generated E2E.Ca_II_SE_Nh_Te        n_SE.sum()=1.000000  (+3 keys)
generated E2E.Ca_II_SE_Ne_Te        n_SE.sum()=1.000000  (+2 keys)
generated E2E.Ca_II_SE_Pg_Te        n_SE.sum()=1.000000  (+3 keys)
wrote .../tests/regression/reference_values.json  (281 keys total, +18 new)
```

Independent re-generation on `main` (HEAD `df5ceef` minus this audit append)
produced a byte-identical JSON:

```
$ diff /tmp/ref-from-a84128b.json tests/regression/reference_values.json
(no output — 0 differing lines)
```

All 9 tests in `test_reg_e2e_SE.py` pass on `main` with `rtol=1e-8` against
the `a84128b`-generated reference. This confirms the SE path
(`Function/SEquil/SELib.py`, `Atomic/{SEsolver,LTELib,Collision,PhotoIonize,BasicP}.py`)
is behavior-preserving between `a84128b` and current `main`.

## References

- Handoff: `tmp/handoff.md`
- Predecessor task: `docs/tasks/002-atom-io-regression-and-unbound-cleanup/`
- PR #10 (AtomIO regression): `f10482c`
- PR #9 (pyright cleanup): `250cc36`
- Baseline commit: `a84128b`
