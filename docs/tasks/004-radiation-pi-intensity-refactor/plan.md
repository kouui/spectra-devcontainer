# Plan: Stop caching `PI_intensity` in `Radiation` struct + rename `backRad → solar`

> **Task:** [task.md](./task.md)
> **Owner:** kouui
> **Created:** 2026-05-03
> **Target Completion:** 2026-05-04

---

## 0. Context

> **Objective:** Decouple `Radiation` from atmosphere/wMesh state by removing `PI_intensity`; expose `cont_intensity` via `SE_Container`; rename `backRad → solar`; make `_bf_R_rate_` a pure rate calculator.
> **Full spec:** [task.md](./task.md)
>
> This plan describes *how* to achieve the above objective.
> For requirements and acceptance criteria, refer to the task.

---

## 1. Overall Architecture

### System Overview — Before vs After

**Before:**
```
init_Radiation_(atmos, wMesh)
  ├─ load atlas → backRad
  └─ if use_Tr: planck(Tr) → PI_intensity (DEAD: overwritten in _bf_R_rate_)
     else:      interp(backRad, Cont_mesh) → PI_intensity (cached)

cal_SE_(atom, atmos, wMesh, radiation, Nh_SE, ...)
  └─ _bf_R_rate_(Cont, Cont_mesh, Te, nj_, alpha, PI_intensity, backRad, Tr, use_Tr, doppler)
        ├─ if use_Tr:                  PI_I0 = planck(Cont_mesh, Tr)
        ├─ elif doppler_shift:         PI_I0 = interp(backRad, Cont_mesh)  # DEAD: gated upstream
        └─ else:                       PI_I0 = PI_intensity                # ONLY live path
```

**After:**
```
init_Radiation_(path=None)
  └─ load atlas → solar          # only field

cal_SE_(atom, atmos, wMesh, radiation, Nh_SE, ..., cont_intensity=None)
  ├─ if doppler_shift_continuum: raise NotImplementedError
  ├─ cont_wave_mesh_shifted = wMesh.Cont_mesh                # alias; future shift hook
  ├─ if cont_intensity is None:
  │     if atmos.use_Tr:  cont_intensity = planck(cont_wave_mesh_shifted, atmos.Tr)
  │     else:             cont_intensity = interp(radiation.solar, cont_wave_mesh_shifted)
  ├─ _bf_R_rate_(Cont, cont_wave_mesh_shifted, Te, nj_, alpha, cont_intensity)   # pure
  └─ SE_Container(... , cont_intensity=cont_intensity, cont_wave_mesh_shifted=cont_wave_mesh_shifted)

cal_SE_with_*(...)  # wrappers
  cont_intensity = None
  for ...:
      SE_con, _ = cal_SE_(..., cont_intensity=cont_intensity)
      cont_intensity = SE_con.cont_intensity   # capture & reuse
```

### Key Components

| Component                                              | Responsibility                                                                          | New / Modified |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------|----------------|
| `Struct/Radiation.py:Radiation`                        | Holds solar atlas only                                                                  | Modified       |
| `Struct/Radiation.py:init_Radiation_`                  | Load atlas; optional path override                                                      | Modified       |
| `Struct/Container/SEquil.py:SE_Container`              | Carries `cont_intensity` + `cont_wave_mesh_shifted`                                     | Modified       |
| `Function/SEquil/SELib.py:cal_SE_`                     | Inlines `use_Tr` dispatch; accepts `cont_intensity` override                            | Modified       |
| `Function/SEquil/SELib.py:_bf_R_rate_`                 | Pure b-f rate calculator; no Tr/backRad/use_Tr/doppler args                             | Modified       |
| `Function/SEquil/SELib.py:cal_SE_with_*` (4 wrappers)  | Reuse `cont_intensity` via SE_Container capture                                         | Modified       |
| `Function/Icp/SELib.py` (mirror of SEquil)             | Same as SEquil                                                                          | Modified       |
| Tests / examples                                       | Update `init_Radiation_()` calls and `backRad → solar` references                       | Modified       |
| Notebooks                                              | Same as tests; plus `PI_intensity` → `solar[1,:]` or override pattern                   | Modified       |

### Data Flow (single SE solve, `use_Tr=False`)

1. `init_Radiation_()` loads `solar` atlas from disk (no atmos/wMesh dependency).
2. Caller invokes `cal_SE_(atom, atmos, wMesh, radiation, Nh_SE)` (or a wrapper).
3. Inside `cal_SE_`: `cont_wave_mesh_shifted = wMesh.Cont_mesh` (alias).
4. `cont_intensity = interpolate_PI_intensity_(radiation.solar, cont_wave_mesh_shifted)`.
5. `_bf_R_rate_(..., cont_intensity)` returns `Rik, Rki_stim, Rki_spon`.
6. `SE_Container` populated with `cont_intensity`, `cont_wave_mesh_shifted` and the line-side fields.

### Data Flow (wrapper iteration, e.g. `cal_SE_with_Pg_Te_`)

1. `cont_intensity = None`.
2. Iteration k: `cal_SE_(..., cont_intensity=cont_intensity)` → returns `SE_con` with newly-computed (k=0) or pass-through (k>0) `cont_intensity`.
3. `cont_intensity = SE_con.cont_intensity` → cached for next iteration.
4. Loop converges; final `SE_con` returned.

---

## 2. Implementation Phases

### Phase 1: Source code refactor (struct + SE)

> **Goal:** All source modules under `src/spectra/` reflect the new architecture; legacy field/parameter names removed.
> **Estimated Effort:** 0.5 day

- [ ] Step 1.1 — `Struct/Radiation.py`:
  - Drop import of `_Atmosphere`, `_LTELib`, `_PhotoIonize`, `_WavelengthMesh` (no longer needed at module level)
  - Replace `backRad`/`PI_intensity` fields with single `solar` field
  - Replace `init_Radiation_(atmos, wMesh)` with `init_Radiation_(path: Path | None = None)`
  - Add a changelog entry (matching existing version-comment block style)
- [ ] Step 1.2 — `Struct/Container/SEquil.py`:
  - Add `cont_intensity: T_ARRAY` field with `(nCont, _N_CONT_MESH)` shape comment + units `[erg/cm^2/Sr/cm/s]`
  - Add `cont_wave_mesh_shifted: T_ARRAY` field with `(nCont, _N_CONT_MESH)` shape + units `[cm]`
  - Update changelog block
- [ ] Step 1.3 — `Function/SEquil/SELib.py:_bf_R_rate_`:
  - Drop parameters `PI_I`, `backRad`, `Tr`, `use_Tr`, `doppler_shift_continuum`
  - Rename remaining `PI_I0` references to `cont_intensity` (consume directly from arg)
  - Delete the dead `if use_Tr / elif doppler_shift / else` block; body becomes a single `cont_intensity` reference
  - Delete the historical commented-out fallback block (lines previously at 530-533)
- [ ] Step 1.4 — `Function/SEquil/SELib.py:cal_SE_`:
  - Add new kwarg `cont_intensity: T_ARRAY | None = None` (placed after `rate_only`)
  - Replace `backRad = radiation.backRad` line with `solar = radiation.solar`
  - After the `doppler_shift_continuum` guard, compute `cont_wave_mesh_shifted = Cont_mesh` (alias)
  - When `cont_intensity is None`: branch on `use_Tr` and call `_LTELib.planck_cm_` or `_PhotoIonize.interpolate_PI_intensity_(solar, cont_wave_mesh_shifted)`
  - Update `_bf_R_rate_` call: `(Cont, cont_wave_mesh_shifted, Te, nj_by_ni_Cont, alpha_interp, cont_intensity)`
  - Update `_B_Jbar_` call: rename `backRad` argument to local `solar` variable (parameter name in `_B_Jbar_` itself stays `backRad` — out of scope; we just pass `solar` to it)
  - Add `cont_intensity=cont_intensity` and `cont_wave_mesh_shifted=cont_wave_mesh_shifted` to the `SE_Container(...)` constructor
- [ ] Step 1.5 — `Function/SEquil/SELib.py` wrappers (`cal_SE_with_Pg_Te_single_Atom_`, `cal_SE_with_Pg_Te_`, `cal_SE_with_Nh_Te_`, `cal_SE_with_Ne_Te_`):
  - Add `cont_intensity = None` initialization before the iteration loop (or before the single non-loop call, where applicable — `cal_SE_with_Ne_Te_` may be a single call)
  - Inside loop: `SE_con, tran_rate_con = cal_SE_(..., cont_intensity=cont_intensity)` then `cont_intensity = SE_con.cont_intensity`
- [ ] Step 1.6 — `Function/Icp/SELib.py`: **deviation from original plan** — after confirming with `grep -rn "from.*Icp"` that there are no consumers in `src/`, `tests/`, or active notebook code (only commented-out imports), the entire `Function/Icp/SELib.py` was stubbed by inserting a top-level `raise ImportError(...)` ahead of the original code. The original implementation is preserved below the raise as a quick reference; full restoration is via git history. Rationale: avoids the cost of mirroring 5+ wrappers and a JIT-decorated `_bf_R_rate_` for code with no live consumers.
- [ ] Step 1.7 — Run regression to confirm source-only changes are bit-identical:
  ```
  uv run pytest tests/regression/test_reg_e2e_SE.py tests/regression/test_reg_e2e_CloudModel.py -v 2>&1 | tail -40
  ```
  Expected: tests fail because callers still pass old args. Proceed to Phase 2 to fix call sites.

**Phase 1 Exit Criteria:**
- [ ] All source files compile without syntax errors (`uv run python -c "import spectra"` succeeds)
- [ ] No `radiation.backRad` or `radiation.PI_intensity` reference remains under `src/spectra/`
- [ ] `_bf_R_rate_` signature matches spec; numba decorator still applies

---

### Phase 2: Update non-notebook callers (tests + examples)

> **Goal:** All Python callers under `tests/` outside notebooks compile and pass with new API.
> **Estimated Effort:** 0.25 day
> **Depends on:** Phase 1

- [ ] Step 2.1 — `tests/regression/test_reg_e2e_SE.py`: replace all `Radiation.init_Radiation_(atmos, wMesh)` with `Radiation.init_Radiation_()` (9 sites)
- [ ] Step 2.2 — `tests/regression/test_reg_e2e_CloudModel.py`: same single-site update
- [ ] Step 2.3 — `tests/unittest/test.SE.H_I.py`: same
- [ ] Step 2.4 — `tests/examples/example.SE.py`, `tests/examples/example.He.py`, `tests/examples/example.CM.py`: same
- [ ] Step 2.5 — `tests/regression/test_reg_PhotoIonize.py`: variable name `backRad` is local to the test function (loads npy directly); leave as-is to keep diff minimal **unless** consistency is preferred — confirm before changing
- [ ] Step 2.5b — **`tests/unittest/test.SE.H_I.py` reference values regenerated.** This file was already broken on `main` before the refactor: hardcoded reference arrays had 9 elements but `data/atom/H/H.Level` declares 11 levels (atom data drift, predates this task). Phase 2 review caught it; verified pre-existing via `git stash` + rerun. Resolution: regenerate the reference values from the current main code on the new no-arg `init_Radiation_()`. Reference regen is acceptable here only because the failure was a data-shape mismatch, not a numerical drift — the regression tests would have caught any silent value drift.
- [ ] Step 2.6 — Run full regression:
  ```
  uv run pytest tests/regression/ -v 2>&1 | tail -50
  ```
  Expected: all pass with zero numeric drift.

**Phase 2 Exit Criteria:**
- [ ] `pytest tests/regression/` is fully green
- [ ] No reference value in `tests/regression/reference_values.json` was modified

---

### Phase 3: Update notebooks (last step)

> **Goal:** All active notebooks compile and reflect the new API; minimal touch to historic notebooks.
> **Estimated Effort:** 0.5 day
> **Depends on:** Phase 2

- [ ] Step 3.1 — Active notebooks (verify by re-running impacted cells if reasonably fast):
  - `H_spectra.ipynb`, `HHeCa.ipynb`, `CaII_flush.ipynb`, `spectra_multi_atom.ipynb`, `He_plasma.ipynb`, `He_plasma.NeTe.ipynb`, `Hydrogen_atom.ipynb`
  - For each:
    - `Radiation.init_Radiation_(atmos, wMesh)` → `Radiation.init_Radiation_()`
    - `radiation.backRad` → `radiation.solar`
    - `radiation.PI_intensity[:,:] *= X` → `radiation.solar[1,:] *= X` *(keep the comment if any)*
    - `radiation.PI_intensity[:,:] = 0` → `radiation.solar[1,:] = 0`
    - `solar_spec2.PI_intensity.reshape(-1)[ii] *= ampl` → manual override (compute via `interpolate_PI_intensity_`, modify, pass via `cal_SE_(..., cont_intensity=...)`). Flag each occurrence in the commit message.
- [ ] Step 3.2 — Historic notebooks (rename only, no behavior verification):
  - `HHeCa_0.ipynb`, `H_spectra_otsu_ver20240424.ipynb`
  - Mechanical replace: `backRad` → `solar`, `PI_intensity[:,:]` → `solar[1,:]` (only when followed by `*=` or `= 0`)
- [ ] Step 3.3 — Search verification: `grep -rn "backRad\|PI_intensity" notebooks/` returns only comments/strings, no executable code references.

**Phase 3 Exit Criteria:**
- [ ] All active notebooks have updated `init_Radiation_` calls
- [ ] No `radiation.backRad` or `radiation.PI_intensity` Python expressions remain in notebooks
- [ ] Active-notebook spot-checks: at least one notebook per atom (H, He, Ca) re-run end to end

---

## 3. Boundaries — Do NOT Touch

> These files, modules, and APIs are explicitly out of scope. Any changes to them require a separate task and review.

| Area                                    | Path / Identifier                                               | Reason                                                                    |
|-----------------------------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------|
| PI primitive                            | `src/spectra/Atomic/PhotoIonize.py:interpolate_PI_intensity_`   | Already does flatten + single `numpy.interp`; stable                      |
| PI rate primitive                       | `src/spectra/Atomic/PhotoIonize.py:bound_free_radiative_*`      | Pure physics function; not a refactor target                              |
| `_B_Jbar_` body                         | `src/spectra/Function/SEquil/SELib.py:_B_Jbar_`                 | Line-side rates; only the local var `backRad` → `solar` rename propagates |
| Atmosphere structs                      | `src/spectra/Struct/Atmosphere.py`                              | Untouched; we only read `atmos.use_Tr`, `atmos.Tr`, `atmos.doppler_shift_continuum` (existing) |
| Wavelength_Mesh struct                  | `src/spectra/Struct/WavelengthMesh.py`                          | We read `wMesh.Cont_mesh` only                                            |
| Reference JSON                          | `tests/regression/reference_values.json`                        | Must not change; refactor must be bit-identical                           |
| Doppler-shifted continuum implementation| any new code that actually shifts `Cont_mesh`                   | Future work; this task only adds the field as a forward-ready alias       |
| Other regression tests                  | `test_reg_LTELib.py`, `test_reg_BasicP.py`, etc.                | Unrelated kernels                                                         |

**Rule of thumb:** If your change requires modifying anything listed above, stop and revisit the plan.

---

## 4. Test Coverage

### Testing Strategy

| Level       | Scope                                                                     | Tool / Framework |
|-------------|---------------------------------------------------------------------------|------------------|
| Regression  | `cal_SE_with_*` end-to-end values match `reference_values.json` exactly   | `pytest`         |
| Smoke       | Active notebooks execute without exception                                | manual           |

### Required Test Cases

#### Existing tests (must remain green)

- [ ] `tests/regression/test_reg_e2e_SE.py` — 9 cases (3 atoms × 3 entries, partially covered)
- [ ] `tests/regression/test_reg_e2e_CloudModel.py`
- [ ] `tests/regression/test_reg_PhotoIonize.py` — verifies primitive interpolation
- [ ] `tests/regression/test_reg_SEsolver.py` — kernel-level

#### New tests

- None required. The refactor preserves behavior; the existing e2e SE suite (task 003) provides bit-identical regression coverage.

#### Edge Cases & Error Handling

- [ ] `init_Radiation_()` with no args loads default atlas (verified via existing tests)
- [ ] `init_Radiation_(path=...)` with explicit path loads the user-specified file (covered by manual smoke; no regression case)
- [ ] `cal_SE_(..., cont_intensity=<override>)` honors override (covered indirectly through wrapper response-capture pattern)
- [ ] `cal_SE_(..., cont_intensity=None)` when `use_Tr=True` produces same Rik values as before (existing H/He cases with Tr setup, if any — verify manually)
- [ ] `doppler_shift_continuum=True` still raises `NotImplementedError` at `cal_SE_` entry

### Coverage Target

- **Numerical regression**: zero drift on existing 9 e2e cases + photoionize primitive
- **API surface**: every changed signature exercised by at least one test

---

## 5. Key Decisions

### Decision 1: Drop `make_PI_intensity_` helper

- **Context:** Earlier proposal added a public helper `Struct/Radiation.py:make_PI_intensity_(radiation, wMesh, atmos)` to centralize the `use_Tr` dispatch.
- **Options Considered:**
  1. Public helper in `Struct/Radiation.py` — pros: single source of truth; cons: introduces atmos dependency on Radiation module
  2. Private helper `_compute_cont_intensity_` in each `SELib.py` — pros: file-local; cons: duplicated across SEquil and Icp
  3. Inline the `use_Tr` if/else inside `cal_SE_` only; wrappers reuse via SE_Container response capture
- **Decision:** Option 3.
- **Rationale:** Wrappers don't need to know the dispatch logic at all — they read it back from `SE_Container.cont_intensity` after the first call. `cal_SE_` is the single dispatch location. No new abstraction.
- **Consequences:** Wrappers must remember to capture `SE_con.cont_intensity` at end of each iteration. The dispatch costs `nCont * 41` ≈ 600 interpolation points exactly once per wrapper invocation (at iteration 0).

### Decision 2: Rename `backRad → solar`

- **Context:** Issue #6 comment proposes the rename; current name `backRad` (background radiation) is generic.
- **Options Considered:**
  1. Keep `backRad` to minimize notebook diff
  2. Rename to `solar`
- **Decision:** Option 2.
- **Rationale:** Atlas file is literally `atlas_QS.20221118.npy` (Quiet Sun); `solar` reflects content. Notebook diff is mechanical search/replace.
- **Consequences:** `_B_Jbar_` parameter name remains `backRad` (out of scope); the local variable in `cal_SE_` that passes it becomes `solar`. Slightly inconsistent at the param boundary but acceptable since `_B_Jbar_` body is read-only.

### Decision 3: `cont_wave_mesh_shifted` as alias to `wMesh.Cont_mesh`

- **Context:** Continuum Doppler shift is unimplemented. Storing a separate "shifted" copy in SE_Container is forward-looking.
- **Options Considered:**
  1. Don't add the field; only add `cont_intensity`
  2. Add the field, store an alias for now
  3. Add the field, copy the array
- **Decision:** Option 2 (alias).
- **Rationale:** Naming parity with bound-bound (`wave_mesh_shifted_1d`) clarifies intent. Alias avoids memory waste; future implementation of continuum shift will naturally produce a new array. Caller mutating the shared array is a user-error class we accept.
- **Consequences:** Users who mutate `SE_Container.cont_wave_mesh_shifted` also mutate `wMesh.Cont_mesh`. Documented in field docstring.
- **Mid-flight refinement (post Phase 1 review):** Phase 1 review flagged that inserting the new `cont_*` fields between existing dataclass fields (`Jbar` and `Ntotal`) breaks any positional-arg construction. Resolution: append the new fields at the end of the dataclass instead of placing them adjacent to the bound-bound siblings. Naming parity is preserved through the field name only, not through dataclass position. All in-tree constructors use kwargs, so no behavioral impact.

### Decision 4: `_bf_R_rate_` becomes pure rate calculator

- **Context:** Currently `_bf_R_rate_` accepts `Tr`, `use_Tr`, `backRad`, `doppler_shift_continuum` to dispatch its PI intensity source.
- **Options Considered:**
  1. Keep dispatch inside `_bf_R_rate_`
  2. Hoist dispatch up to `cal_SE_`
- **Decision:** Option 2.
- **Rationale:** `Tr` and `use_Tr` are only used to compute `PI_I0` — the rate loop itself doesn't depend on them. Single Responsibility: `_bf_R_rate_` should compute rates given inputs, not dispatch sources.
- **Consequences:** `_bf_R_rate_` argument count drops from 10 to 6. JIT recompile required after refactor. Future doppler-shift logic lives upstream in `cal_SE_` next to the `cont_wave_mesh_shifted` computation.

### Decision 6: Phase 3 redo — manual subagent migration after script bug

- **Context:** Original Phase 3 used a single Python script (`tmp/fix_notebooks.py`) doing regex-based mechanical replacement across 12 notebooks (12+ patterns). User-flagged a bug afterward: paired-pattern cells like `solar_spec_H.backRad[1,:] *= Ifact; solar_spec_H.PI_intensity[:,:] *= Ifact` were both rewritten to `solar_spec_H.solar[1, :] *= Ifact`, producing **double scaling** because in the new API `solar` drives both line Jbar and continuum PI rates (the old API kept them as decoupled arrays).
- **Resolution:** Reverted all 12 notebooks via `git checkout HEAD -- notebooks/`; wrote a shared migration brief (`tmp/notebook_migration_brief.md`) explaining old vs new semantics and per-pattern translation rules; delegated one subagent per notebook to read the cell context, understand author intent (paired vs standalone vs zero), and apply minimal correct fixes via `NotebookEdit` (or fallback to direct JSON write where `NotebookEdit` cell-id requirements weren't met).
- **Output post-processing:** Subagents that fell back to JSON writes either (a) flattened `source` from array-of-lines to a single string or (b) inadvertently cleared `outputs`/`execution_count`/cell `metadata`. Ran `tmp/restore_outputs.py` to restore those fields from HEAD while keeping the new `source` text — final diff is now content-only.
- **Consequences:** Notebook diff is minimal and correct. Two pre-existing bugs were also fixed: a double-statement-glued typo in `spectra_multi_atom.ipynb:145` and an EUV-amplification index-mismatch (`PI_intensity.reshape(-1)[ii]` with atlas-sized `ii`).

### Decision 5: Wrapper response-capture for reuse

- **Context:** Wrappers iterate `cal_SE_` many times (e.g. self-consistent `Ne` solver). Recomputing `cont_intensity` per iteration wastes the interpolation.
- **Options Considered:**
  1. Wrapper inlines the use_Tr dispatch and pre-computes `cont_intensity` once
  2. Wrapper duplicates the dispatch lines (same as cal_SE_ body)
  3. Wrapper initializes `cont_intensity = None`, captures from `SE_con.cont_intensity` after each call
- **Decision:** Option 3.
- **Rationale:** Wrapper is decoupled from dispatch logic; if dispatch evolves (e.g. doppler shift), only `cal_SE_` changes. Cost: trivial array assignment per iteration.
- **Consequences:** Slight reliance on SE_Container as a return-channel for state. Acceptable given SE_Container already holds many derived fields.

---

## 6. Precautions

### Technical Risks

| Risk                                                       | Likelihood | Impact | Mitigation                                                                                       |
|------------------------------------------------------------|-----------|--------|--------------------------------------------------------------------------------------------------|
| `numba` JIT signature mismatch on `_bf_R_rate_`            | Medium    | Low    | Run regression after Phase 1.6; clear `__pycache__` if fresh-compile errors appear               |
| Wrapper forgets to capture `SE_con.cont_intensity`         | Medium    | Medium | Code review + regression covers each wrapper                                                     |
| Notebook `reshape(-1)[ii] *= ampl` translation error       | Medium    | Low    | Manual review per occurrence; preserve original cell as a comment if uncertain                   |
| Regression numeric drift                                   | Low       | High   | Refactor is rearrangement, not algorithmic change; verify zero diff before any reference update  |
| Field-rename touches a piece of code not in `grep`         | Low       | Medium | Run `grep -rn "backRad\|PI_intensity" --include="*.py" --include="*.ipynb"` after each phase     |

### Rollback Plan

1. Source code is on a feature branch; `git checkout main -- src/spectra/Struct/Radiation.py src/spectra/Function/SEquil/SELib.py src/spectra/Function/Icp/SELib.py src/spectra/Struct/Container/SEquil.py` reverts source.
2. If regression drift appears mid-refactor, `git diff main -- tests/regression/reference_values.json` should be empty; if not, abort and audit.
3. If notebooks are partially updated, revert via `git checkout main -- notebooks/` and redo Phase 3 cleanly.

### Migration Notes

- **Backward compatibility:** None required (single-developer project; `Radiation` struct is internal API).
- **Feature flag:** None; refactor is in-place.
- **Migration script:** Not needed; users update import sites mechanically per the patterns in task.md scope list.

### Performance Considerations

- Interpolation cost per `cal_SE_` invocation: `nCont * _N_CONT_MESH ≈ 600` interp points; trivial vs SE solver cost (~ms).
- Wrapper iteration: first call computes; subsequent calls read alias from `SE_con`. Net cost = same as before refactor (one interpolation per wrapper invocation, amortized).
- Memory: `cont_wave_mesh_shifted` is an alias (zero extra memory); `cont_intensity` is the same array previously stored in `Radiation`.

### Security Considerations

- N/A; no external input, no auth boundary.

---

## Changelog

| Date       | Author | Change                                                                                |
|------------|--------|---------------------------------------------------------------------------------------|
| 2026-05-03 | kouui  | Initial draft after discussion (issue #6, sessions converged on response-capture pattern) |
| 2026-05-03 | kouui  | Phase 1 review: append new SE_Container `cont_*` fields at end of dataclass (positional-arg compat) |
| 2026-05-03 | kouui  | Mid-flight: `Function/Icp/SELib.py` stubbed instead of refactored (no consumers).      |
| 2026-05-03 | kouui  | Phase 2 review: regenerate `test.SE.H_I.py` references for pre-existing atom-data drift |
| 2026-05-03 | kouui  | Phase 3 redo: replace mechanical-script with per-notebook subagent migration after paired-pattern double-scaling bug; added 2 notebooks (`HeH.ipynb`, `CloudModel.demo.ipynb`) discovered via grep |
| 2026-05-03 | kouui  | Final review: removed `issue#6` references from source code (per project convention)   |
