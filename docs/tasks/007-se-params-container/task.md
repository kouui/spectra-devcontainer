# Task: Move SE radiation/continuum switches off `Atmosphere` into a dedicated `SE_Params_Container`

> **Status:** Draft
> **Owner:** kouui
> **Created:** 2026-05-15
> **Last Updated:** 2026-05-15

## Objective

Move three SE-solver behavior switches — `Tr`, `use_Tr`, `doppler_shift_continuum` — off `Atmosphere0D` / `AtmosphereC1D` and into a new `SE_Params_Container`. These fields describe how the SE solver picks its radiation source and treats the continuum mesh; they are not physical properties of the atmosphere. The new struct collapses `Tr` + `use_Tr` into a single nullable `Tr: T_FLOAT | None` (None ⇒ interpolated solar background; not-None ⇒ `planck(Tr)`, including the coronal-equilibrium `Tr=0.0` edge case).

## Background & Context

GitHub issue [#5](https://github.com/kouui/spectra-devcontainer/issues/5) ("`use_Tr` should not be a parameter in slab") proposes that `use_Tr` (and by extension `doppler_shift_continuum`) belong in their own parameter struct rather than on the atmosphere. Today both `Atmosphere0D` (`src/spectra/Struct/Atmosphere.py:42-47`) and `AtmosphereC1D` (`src/spectra/Struct/Atmosphere.py:69-72`) duplicate these three fields, conflating "what the atmosphere is" with "how the SE solver reads the radiation field". The duplication propagates downstream: `cal_SE_` reads `atmos.Tr`, `atmos.use_Tr`, `atmos.doppler_shift_continuum` at `SELib.py:336-340` and threads them into `_B_Jbar_` and the photoionization rate path.

Design decisions taken during discussion with the user (recorded in `plan.md` §5):

- **Unify `Tr` + `use_Tr` into `Tr: T_FLOAT | None`.** None means "fall back to `radiation.solar`"; not-None means "use `planck(Tr)`". This preserves the `Tr=0., use_Tr=True` coronal-equilibrium case verbatim (now spelled `Tr=0.`).
- **Container, not loose kwargs.** A struct scales when more SE switches arrive and reads cleanly across the four-deep wrapper chain (`cal_SE_with_*_` → `cal_SE_`).
- **Keep `doppler_shift_continuum` in the same struct now** even though it's an unimplemented stub (`SELib.py:340` raises `NotImplementedError`) — pulling it across at the same time avoids a second migration when it's wired up.
- **Notebooks are out of scope for this PR.** They will be updated in a follow-up PR (issue/branch TBD) that targets this feature branch — not `main`. PR for this task does not merge to `main` until the notebook PR has merged into it.

A scope audit was performed before writing this task (`grep "atmos\\.Tr\\|atmos\\.use_Tr\\|atmos\\.doppler_shift_continuum"`), and an independent codex audit refined the numbers (`tmp/codex_outputs/007_scope_review.md`):

- **`src/`**: 3 reads in `Function/SEquil/SELib.py:336-340`; 1 docstring mention in `Struct/Radiation.py:34`; 1 inline comment in `Struct/Container/SEquil.py:58`; 2 stale changelog-header comments in `Function/SEquil/SELib.py:10-11` that mention `use_Tr` / `doppler_shift_continuum` and must be cleaned to keep the acceptance grep clean.
- **`tests/`**: 0 reads of the fields, but 13 `cal_SE_*` call sites that all rely on the public signature (`tests/regression/test_reg_e2e_SE.py` ×9, `test_reg_e2e_CloudModel.py` ×2, `tests/unittest/test.SE.H_I.py` ×1, `tests/examples/*.py` ×4).
- **`scripts/`**: 1 indirect call site (`scripts/gen_se_reference.py:113`).
- **`notebooks/`**: 15 notebooks reference these fields (~142 JSON occurrences, ~111 `Atmosphere0D(..., use_Tr=...)` construction sites). Biggest hitters: `H_spectra_otsu_ver20240424.ipynb` (33 constructions), `Hydrogen_atom.ipynb` + `StatisticalEquilibrium/Hydrogen_atom.ipynb` (15 + 16), `StatisticalEquilibrium_CloudModel.demo.ipynb` (13). All deferred to follow-up PR.
- **No indirect coupling**: codex audit confirmed zero `getattr` / `setattr` / `dataclasses.fields` / `asdict` / `astuple` / `__dict__` access on the removed names, and no `Tr` / `use_Tr` / `doppler_shift_continuum` keys in `tests/regression/reference_values.json`.

The companion file `src/spectra/Function/Icp/SELib.py` (a parallel SE path that previously held the same `use_Tr` plumbing) has been deleted in the working tree; that deletion is committed separately and is not part of this refactor's scope.

References:

- Issue: [#5](https://github.com/kouui/spectra-devcontainer/issues/5)
- Closest precedent task (same audit-and-rename pattern): `docs/tasks/005-cont-intensity-rename/`

## Requirements

### Functional Requirements

1. **New struct** in `src/spectra/Struct/Container/SEquil.py`:
   ```python
   @_dataclass(**STRUCT_KWGS_UNFROZEN)
   class SE_Params_Container:
       Tr: T_FLOAT | None = None
       doppler_shift_continuum: T_BOOL = False
   ```
   Exported through `src/spectra/Struct/Container/__init__.py`.

2. **Remove from atmosphere** — drop `Tr`, `use_Tr`, `doppler_shift_continuum` from `Atmosphere0D` (`src/spectra/Struct/Atmosphere.py:42-47`) and `AtmosphereC1D` (lines 69-72). Field ordering of remaining fields preserved.

3. **`cal_SE_` signature** — add `se_params: SE_Params_Container` as a new required keyword argument. Replace the three `atmos.…` reads with `se_params.…` reads. Unpack to primitive `(Tr: T_FLOAT, use_Tr: T_BOOL)` at the boundary before calling `_B_Jbar_` (whose inner signature stays primitive — numba-friendly):
   ```python
   use_Tr: T_BOOL = se_params.Tr is not None
   Tr: T_FLOAT = se_params.Tr if use_Tr else 0.0
   ```

4. **Wrapper signatures** — propagate `se_params` through `cal_SE_with_Pg_Te_single_Atom_`, `cal_SE_with_Pg_Te_`, `cal_SE_with_Nh_Te_`, `cal_SE_with_Ne_Te_`. All four wrappers accept `se_params` and pass it through.

5. **Test/script call-site update** — add `se_params=SE_Params_Container()` (the default = old `use_Tr=False`) to every `cal_SE_*` call in `tests/regression/`, `tests/unittest/`, `tests/examples/`, `scripts/gen_se_reference.py`. Regression must remain green with zero numerical drift.

6. **Update inline doc references** — `Struct/Container/SEquil.py:58` (the `PI_intensity` field comment that says "planck(Tr) when atmos.use_Tr") and `Struct/Radiation.py:34` (docstring mentioning `atmos.use_Tr=True`).

### Non-Functional Requirements

- **Behavior-preserving for tests/scripts.** All defaults (`Tr=None`, `doppler_shift_continuum=False`) reproduce the old `use_Tr=False` path bit-for-bit. `pytest tests/regression/` must pass 261/261 with zero diff against `reference_values.json`.
- **No deprecation shim.** Consistent with project policy (`CLAUDE.md` and task 005 §5 decision 4): hard rename, no fallback. External callers must migrate.
- **Numba-friendly boundary.** `_B_Jbar_` keeps primitive `Tr, use_Tr` args; no struct passing into jitted scope.
- **No new files beyond what's strictly needed.** The new dataclass lives in the existing `Container/SEquil.py`.

## Scope

### In Scope

- [ ] `src/spectra/Struct/Container/SEquil.py` — add `SE_Params_Container`; update `PI_intensity` field comment.
- [ ] `src/spectra/Struct/Atmosphere.py` — remove 3 fields from `Atmosphere0D`, 3 fields from `AtmosphereC1D`. Trim/adjust the changelog header.
- [ ] `src/spectra/Struct/Radiation.py:34` — fix docstring mention of `atmos.use_Tr`.
- [ ] `src/spectra/Function/SEquil/SELib.py` — add `se_params` to 5 public signatures; unpack at `cal_SE_` body; wire through wrappers. Also clean up the stale `use_Tr` / `doppler_shift_continuum` mentions in the file-header changelog at lines 10-11.

> **Verified — no edit needed**: `src/spectra/Struct/Container/__init__.py` is `from .SEquil import *` with no `__all__`, so `SE_Params_Container` is re-exported automatically (codex audit, 2026-05-15).
- [ ] `tests/regression/test_reg_e2e_SE.py` — 9 `cal_SE_*` call sites.
- [ ] `tests/regression/test_reg_e2e_CloudModel.py` — 2 call sites.
- [ ] `tests/unittest/test.SE.H_I.py` — 1 call site.
- [ ] `tests/examples/example.SE.py`, `example.CM.py`, `example.He.py` — 4 call sites total.
- [ ] `scripts/gen_se_reference.py:113` — 1 dispatch site.

### Out of Scope (Boundaries)

> Items explicitly excluded from this task.

- **`notebooks/`** — 15 notebooks (~111 construction sites, ~142 JSON occurrences) reference the migrated fields. Top files by edit volume: `H_spectra_otsu_ver20240424.ipynb` (33), `StatisticalEquilibrium/Hydrogen_atom.ipynb` (16), `Hydrogen_atom.ipynb` (15), `StatisticalEquilibrium_CloudModel.demo.ipynb` (13). Handled in a follow-up PR targeting this feature branch; this PR does not merge until that PR has merged in.
- **`src/spectra/Function/Icp/`** — already deleted in working tree by a separate commit; not part of this refactor.
- **`tests/regression/reference_values.json`** — must not change. Refactor is behavior-preserving; any diff is a regression.
- **`src/spectra/Atomic/PhotoIonize.py`** — already exposes `PI_intensity` vocabulary; not touched.
- **Numba-jitted inner functions (`_B_Jbar_`, `_bf_R_rate_`)** — keep primitive arg signatures. Only the surrounding glue changes.
- **Predecessor task docs (`docs/tasks/00[1-6]-*/`)** — historic audit trail; not edited.

## Acceptance Criteria

- [ ] `SE_Params_Container` exists and is importable from `spectra.Struct.Container`.
- [ ] `Atmosphere0D` and `AtmosphereC1D` no longer declare `Tr`, `use_Tr`, or `doppler_shift_continuum`.
- [ ] `grep -rn "atmos\.Tr\|atmos\.use_Tr\|atmos\.doppler_shift_continuum" src/ tests/ scripts/` returns zero matches.
- [ ] `grep -rn "use_Tr\|doppler_shift_continuum" src/ tests/ scripts/` matches only inside `SE_Params_Container` definition, the unpacking line in `cal_SE_`, and the `_B_Jbar_` primitive parameter — no `atmos.` reads.
- [ ] All 5 SEquil wrapper signatures include `se_params: SE_Params_Container`.
- [ ] `uv run pytest tests/regression/ -q` is 261/261 green with zero diff against `reference_values.json`.
- [ ] `uv run pytest tests/unittest/ -q` is green.
- [ ] Pre-commit hooks pass on the feature branch (`ruff format`, `ruff check --fix`, `pyright`, `pytest-regression`, `protect-main-branch`).
- [ ] PR opened as **draft** against `main` for professor review.
- [ ] Follow-up notebook PR exists and is targeted at this feature branch (not `main`).

## Dependencies

| Dependency | Owner | Status | Notes |
|------------|-------|--------|-------|
| Issue #5 reviewed and design agreed with user | kouui | ✅ Done | This conversation, 2026-05-15 |
| Icp/ deletion committed separately | kouui | 🟡 Pending | Already in working tree; commit separately so it isn't entangled with this task |
| Professor review of draft PR | (professor) | ⏳ Pending | Draft PR is the deliverable that unblocks the notebook follow-up |
| Notebook follow-up PR | kouui | ⏳ Pending | Targets this feature branch; merges before this PR merges to main |

## Risks & Open Questions

- [ ] **Notebook breakage during review window.** Between this PR landing on the feature branch and the notebook PR merging in, every notebook in the repo fails at the `Atmosphere0D(..., use_Tr=...)` constructor. The professor reviews the source diff only. Accepted explicitly; rationale captured in `plan.md` §5 Decision 3.
- [ ] **Implicit `None` vs `0.0` semantics for `Tr`.** Reviewers might read `Tr: T_FLOAT | None` and miss that `Tr=0.0` is still a *valid* coronal-equilibrium request (different from `Tr=None`). Mitigated by an inline comment on the field; final wording in `Container/SEquil.py` finalized during implementation.
- [ ] **`Atmosphere0D` is `STRUCT_KWGS_UNFROZEN`.** Removing fields from a mutable dataclass is structurally safe; any code mutating `atmos.use_Tr = X` would now fail at attribute write. Audit found no such mutation in `src/`, `tests/`, `scripts/`; notebooks are excluded by scope.

## References

- GitHub issue: [#5 `use_Tr` should not be a parameter in slab](https://github.com/kouui/spectra-devcontainer/issues/5)
- Closest precedent (same touch surface, same project conventions): `docs/tasks/005-cont-intensity-rename/`
- Design discussion: conversation on 2026-05-15 (decisions 1–6 summarized in `plan.md` §5)
