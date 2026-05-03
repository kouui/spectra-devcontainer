# Task: Stop caching `PI_intensity` in `Radiation` struct + rename `backRad → solar`

> **Status:** Done
> **Owner:** kouui
> **Created:** 2026-05-03
> **Last Updated:** 2026-05-03

## Objective

Decouple `Radiation` from atmosphere/wavelength-mesh state by removing the cached `PI_intensity` field. The continuum PI intensity becomes a **derived quantity computed inside `cal_SE_`** (or supplied by the caller as an override), and is exposed via `SE_Container` for downstream consumers. The atlas field is also renamed from `backRad → solar` to reflect what it actually holds (the QS solar atlas) and to align with the issue #6 proposal. As a side effect, `_bf_R_rate_` becomes a pure b-f rate calculator (no `Tr`/`use_Tr`/`backRad` arguments), and dead branches (`use_Tr` planck reseed at init, `doppler_shift_continuum` inside `_bf_R_rate_`) are removed.

## Background & Context

- **Issue:** [#6](https://github.com/kouui/spectra-devcontainer/issues/6) — `stop caching PI_Intensity in Radiation struct`
- **Current state (`src/spectra/Struct/Radiation.py`)**:
  - `Radiation` carries both `backRad: (2, n_atlas)` (atlas spectrum) and `PI_intensity: (nCont, _N_CONT_MESH)` (atlas pre-interpolated onto continuum mesh).
  - `init_Radiation_(atmos, wMesh)` requires `atmos` + `wMesh` to compute `PI_intensity` — this couples a pure radiation-data struct to atmosphere and wavelength-mesh state.
  - `init_Radiation_` has a `use_Tr` branch that pre-computes `planck(Tr)` into `PI_intensity`, but `_bf_R_rate_` recomputes `planck(Tr)` itself when `use_Tr=True`, so the cached value in that branch is **dead**.
  - `_bf_R_rate_` accepts `Tr`, `use_Tr`, `backRad`, `doppler_shift_continuum` and dispatches PI intensity source internally; the `doppler_shift_continuum` branch is unreachable because `cal_SE_` raises `NotImplementedError` upstream when that flag is set.
- **Existing primitive:** `Atomic/PhotoIonize.py:interpolate_PI_intensity_` already does a 1D-flattened single `numpy.interp` call (issue's "flatten and interpolate once" item is **already done**).
- **Notebook usage of `radiation.PI_intensity`** falls into three patterns:
  1. Bulk scaling: `radiation.PI_intensity[:,:] *= X` — replaceable by `radiation.solar[1,:] *= X` (changes the source spectrum, semantically cleaner).
  2. Bulk zeroing: `radiation.PI_intensity[:,:] = 0` — replaceable by `radiation.solar[1,:] = 0`.
  3. Index-selective amplification: `solar_spec2.PI_intensity.reshape(-1)[ii] *= ampl` — needs the new `cont_intensity=` parameter on `cal_SE_` (compute manually, modify, pass override).

## Requirements

### Functional Requirements

1. **`Radiation` struct simplified to a single field `solar: T_ARRAY` (2d, `(2, n_wavelength)`)**. `PI_intensity` field removed.
2. **`init_Radiation_(path: Path | None = None)`** — drop `atmos`/`wMesh` parameters; accept an optional override path, default to `data/intensity/atlas/QS/atlas_QS.20221118.npy`.
3. **`SE_Container` gains two fields** for symmetry with bound-bound diagnostics:
   - `cont_intensity: T_ARRAY` — 2d `(nCont, _N_CONT_MESH)`, `[erg/cm^2/Sr/cm/s]`
   - `cont_wave_mesh_shifted: T_ARRAY` — 2d `(nCont, _N_CONT_MESH)`, `[cm]` — currently aliases `wMesh.Cont_mesh`; placeholder for future Doppler-shifted continuum mesh, parallel to `wave_mesh_shifted_1d` for lines.
4. **`cal_SE_` signature change** (both `Function/SEquil/SELib.py` and `Function/Icp/SELib.py`):
   - New optional kwarg `cont_intensity: T_ARRAY | None = None`.
   - When `None`: dispatch inline — `use_Tr=True` → `_LTELib.planck_cm_(cont_wave_mesh_shifted, atmos.Tr)`; else → `_PhotoIonize.interpolate_PI_intensity_(radiation.solar, cont_wave_mesh_shifted)`.
   - When provided: use as-is (no override-only validation).
   - Store final `cont_intensity` and `cont_wave_mesh_shifted` into the returned `SE_Container`.
5. **`_bf_R_rate_` becomes pure**: signature `(Cont, Cont_mesh, Te, nj_by_ni_Cont, alpha_interp, cont_intensity)`. Drop `Tr`, `use_Tr`, `backRad`, `doppler_shift_continuum`.
6. **`cal_SE_with_*` wrappers** (in both SEquil and Icp) reuse `cont_intensity` across SE iterations via the response-capture pattern:
   ```python
   cont_intensity = None
   for ...:
       SE_con, _ = cal_SE_(..., cont_intensity=cont_intensity)
       cont_intensity = SE_con.cont_intensity
   ```
7. **Variable rename inside SE modules**: local `PI_intensity` / `PI_I` / `PI_I0` → `cont_intensity*`. The line-side `_B_Jbar_` continues to reference `radiation.solar` under the local name `solar` (different semantics; not renamed to `cont_intensity`).
8. **`doppler_shift_continuum=True` continues to raise `NotImplementedError`** at `cal_SE_` entry. Only the **dead duplicate branch inside `_bf_R_rate_`** is removed.
9. **All callers updated**:
   - `tests/regression/test_reg_e2e_SE.py` (9 call sites)
   - `tests/regression/test_reg_e2e_CloudModel.py`
   - `tests/unittest/test.SE.H_I.py`
   - `tests/examples/example.SE.py`, `tests/examples/example.He.py`, `tests/examples/example.CM.py`
   - All notebooks under `notebooks/StatisticalEquilibrium/` and `notebooks/Hydrogen_atom.ipynb` that reference `radiation.PI_intensity` or `radiation.backRad` (notebooks are the **last step**).
10. **`tests/regression/test_reg_PhotoIonize.py`** continues to work — `interpolate_PI_intensity_` itself is unchanged; the test loads the atlas directly, no `Radiation` involvement.

### Non-Functional Requirements

- **No numerical drift**: `pytest tests/regression/test_reg_e2e_SE.py -v` must pass with zero diff against existing `reference_values.json`. The rename + dispatch refactor is a behavior-preserving rearrangement; `cont_intensity` produced internally must be bit-identical to what `init_Radiation_` previously cached.
- **`numba` compatibility**: `_bf_R_rate_` is `@nb_njit`; `cont_intensity` must always be a typed `ndarray` at the call site (`None` allowed only at the public `cal_SE_` boundary).
- **No new public helpers** in `Struct/Radiation.py` (per design discussion). `cal_SE_` inlines the dispatch; wrappers reuse via SE_Container response capture.
- **Minimal-diff for unrelated areas**: `Atomic/PhotoIonize.py`, `_B_Jbar_` line-side logic, and other modules are read-only for this task.

## Scope

### In Scope

- [ ] `src/spectra/Struct/Radiation.py` — struct & init refactor, rename
- [ ] `src/spectra/Struct/Container/SEquil.py` — add 2 fields to `SE_Container`
- [ ] `src/spectra/Function/SEquil/SELib.py` — `cal_SE_`, `_bf_R_rate_`, all 4 `cal_SE_with_*` wrappers
- [ ] `src/spectra/Function/Icp/SELib.py` — same scope as SEquil
- [ ] `tests/regression/test_reg_e2e_SE.py` — update `init_Radiation_()` calls, rename `backRad` → `solar`
- [ ] `tests/regression/test_reg_e2e_CloudModel.py` — same
- [ ] `tests/unittest/test.SE.H_I.py`, `tests/examples/*.py` — same
- [ ] Notebooks (last step):
  - [ ] `notebooks/StatisticalEquilibrium/H_spectra.ipynb`
  - [ ] `notebooks/StatisticalEquilibrium/HHeCa.ipynb`
  - [ ] `notebooks/StatisticalEquilibrium/CaII_flush.ipynb`
  - [ ] `notebooks/StatisticalEquilibrium/spectra_multi_atom.ipynb`
  - [ ] `notebooks/StatisticalEquilibrium/He_plasma.ipynb`, `He_plasma.NeTe.ipynb`
  - [ ] `notebooks/StatisticalEquilibrium/Hydrogen_atom.ipynb`, `Hydrogen_atom.ipynb` (root)
  - [ ] `notebooks/StatisticalEquilibrium/HeH.ipynb` *(added during Phase 3 grep — not in original scope)*
  - [ ] `notebooks/StatisticalEquilibrium_CloudModel.demo.ipynb` *(added during Phase 3 grep — not in original scope)*
  - [ ] `notebooks/StatisticalEquilibrium/HHeCa_0.ipynb` (older, but still touches `PI_intensity`)
  - [ ] `notebooks/StatisticalEquilibrium/H_spectra_otsu_ver20240424.ipynb` (historic; update minimally — name rename only, no behavior changes)

### Out of Scope (Boundaries)

> Items explicitly excluded from this task. Do NOT touch these areas.

- **`src/spectra/Atomic/PhotoIonize.py`:** `interpolate_PI_intensity_` and `bound_free_radiative_transition_coefficient_` are stable primitives; the issue's "flatten + interpolate once" is already implemented inside `interpolate_PI_intensity_`.
- **`_B_Jbar_` body and line-side wavelength-mesh logic:** out of scope. Only the variable name change `backRad → solar` propagates inward (parameter type and call shape unchanged).
- **`Atmosphere0D` / `AtmosphereC1D`:** untouched.
- **`SE_Container.wave_mesh_shifted_1d`, `absorb_prof_1d`, `Line_mesh_idxs`:** untouched.
- **Reference values JSON (`tests/regression/reference_values.json`):** must not change. If any test diff appears, stop and audit before touching the JSON.
- **Doppler-shifted continuum mesh implementation:** still raises `NotImplementedError`; this task only adds the `cont_wave_mesh_shifted` field as a future-ready alias.

## Acceptance Criteria

- [ ] `Radiation` has only one field (`solar`); `PI_intensity` field removed.
- [ ] `init_Radiation_()` callable with no arguments; optional `path` accepted.
- [ ] `SE_Container.cont_intensity` and `SE_Container.cont_wave_mesh_shifted` populated by every `cal_SE_` call.
- [ ] `_bf_R_rate_` signature is `(Cont, Cont_mesh, Te, nj_by_ni_Cont, alpha_interp, cont_intensity)` — 6 args, no Tr/backRad/use_Tr/doppler_shift.
- [ ] `cal_SE_with_*` wrappers reuse `cont_intensity` via SE_Container capture (no inlined dispatch duplication).
- [ ] `pytest tests/regression/ -v` passes with **zero numeric drift** (verify against `main` baseline before merging).
- [ ] All non-notebook callers of `init_Radiation_(atmos, wMesh)` updated to `init_Radiation_()`.
- [ ] All `radiation.backRad` references in source / tests / examples updated to `radiation.solar`.
- [ ] All `radiation.PI_intensity` references replaced per the three notebook patterns documented above.
- [ ] No new files in `Struct/Radiation.py` (no `make_PI_intensity_` helper).

## Dependencies

| Dependency | Owner | Status | Notes |
|------------|-------|--------|-------|
| Issue #6   | kouui | Open   | Tracking issue, comment confirms `solar` rename |

## Risks & Open Questions

- [ ] **`numba` JIT signature breakage** — `_bf_R_rate_` argument count changes; first call after refactor must trigger a fresh JIT compile. Verify by running e2e SE tests; if signature mismatch errors appear, clear `numba` cache (`__pycache__`) and rerun.
- [ ] **Wrapper response-capture correctness** — first `cal_SE_` call inside each wrapper must successfully populate `SE_Container.cont_intensity`; subsequent iterations must pass it through. Bug risk: forgetting to assign `cont_intensity = SE_con.cont_intensity` at end of loop body in any wrapper.
- [ ] **Notebook `reshape(-1)[ii] *= ampl` migration** — semantics depend on user intent (which specific continuum bins to amplify). Migrating to `cal_SE_(..., cont_intensity=...)` requires careful manual translation; flag and double-check before saving each affected notebook.
- [ ] **Older notebooks (`H_spectra_otsu_ver20240424`, `HHeCa_0`)** — large historic files with many `PI_intensity *= Jfactor` calls. Bulk-rename `backRad → solar` and `PI_intensity → solar[1,:]` patterns; do NOT validate cell-by-cell behavior unless the user runs them.

## References

- [Issue #6](https://github.com/kouui/spectra-devcontainer/issues/6)
- Relevant source: `src/spectra/Struct/Radiation.py`, `src/spectra/Function/SEquil/SELib.py:279-560`, `src/spectra/Function/Icp/SELib.py:330-630`, `src/spectra/Struct/Container/SEquil.py`
- Predecessor task (regression methodology): `docs/tasks/003-se-regression-coverage-stage-a/`
