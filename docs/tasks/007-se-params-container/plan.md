# Plan: Move SE radiation/continuum switches into `SE_Params_Container`

> **Task:** [task.md](./task.md)
> **Owner:** kouui
> **Created:** 2026-05-15
> **Target Completion:** 2026-05-15

---

## 0. Context

> **Objective:** Move `Tr`, `use_Tr`, `doppler_shift_continuum` off `Atmosphere0D`/`AtmosphereC1D` into a new `SE_Params_Container`. Collapse `Tr` + `use_Tr` into `Tr: T_FLOAT | None`. Thread the container through all `cal_SE_*` wrappers. Update tests/scripts; defer notebooks to a follow-up PR.
> **Full spec:** [task.md](./task.md)

---

## 1. Overall Architecture

### What changes

```
Atmosphere0D / AtmosphereC1D                       SE_Params_Container (new)
  - Tr: float                                        + Tr: float | None  = None
  - use_Tr: bool                              -->    + doppler_shift_continuum: bool = False
  - doppler_shift_continuum: bool

cal_SE_(atom, atmos, wMesh, radiation, Nh_SE,      cal_SE_(atom, atmos, wMesh, radiation, Nh_SE,
        rate_only, PI_intensity)                           se_params, rate_only, PI_intensity)
  - reads atmos.Tr / atmos.use_Tr / atmos.dsc  -->    - reads se_params.Tr / se_params.doppler_shift_continuum
  - passes Tr, use_Tr to _B_Jbar_                     - unpacks: use_Tr = Tr is not None
                                                                  Tr_val = se_params.Tr if use_Tr else 0.0
                                                       passes (Tr_val, use_Tr) to _B_Jbar_ (unchanged)
```

`_B_Jbar_` and `_bf_R_rate_` keep their primitive signatures — the rewrite is entirely above the JIT boundary.

### Key components

| Component | File | New / Modified |
|-----------|------|----------------|
| `SE_Params_Container` dataclass | `src/spectra/Struct/Container/SEquil.py` | New |
| `Atmosphere0D` / `AtmosphereC1D` fields | `src/spectra/Struct/Atmosphere.py:42-47, 69-72` | Modified (3 fields removed each) |
| `Radiation` docstring | `src/spectra/Struct/Radiation.py:34` | Modified (text) |
| `SE_Container` field comment | `src/spectra/Struct/Container/SEquil.py:58` | Modified (text) |
| `cal_SE_*` wrappers (×5) | `src/spectra/Function/SEquil/SELib.py` | Modified (signatures + body) |
| Regression / unit / example tests | `tests/regression/*.py`, `tests/unittest/*.py`, `tests/examples/*.py` | Modified (call sites) |
| SE-reference generator | `scripts/gen_se_reference.py:113` | Modified (dispatch site) |

### Data flow

1. Caller builds `Atmosphere0D` (3 fields lighter than before) + `SE_Params_Container` (defaults match old `use_Tr=False`).
2. Caller invokes one of `cal_SE_with_Nh_Te_`, `cal_SE_with_Ne_Te_`, `cal_SE_with_Pg_Te_`, `cal_SE_with_Pg_Te_single_Atom_`, or `cal_SE_` directly, passing the new `se_params` arg.
3. Inside `cal_SE_`: unpack `se_params` once at the top into primitives `(Tr, use_Tr)` and a guard for `doppler_shift_continuum`.
4. Same downstream code path: `_B_Jbar_(..., Tr, use_Tr)`, photoionization rate computation, SE solve.

---

## 2. Implementation Phases

### Phase 1: Source-side refactor (single atomic commit)

> **Goal:** Introduce `SE_Params_Container`, remove fields from `Atmosphere`, thread through `SELib.py`. Source-side internally consistent.
> **Estimated Effort:** 0.5 day

- [ ] Step 1.1 — `src/spectra/Struct/Container/SEquil.py`:
  - Add `SE_Params_Container` dataclass at module top-level (after `SE_Container`, before `TranRates_Container` or at end — choose by file flow).
  - Bump file-header changelog to a new entry, e.g.:
    ```
    # 0.2.1
    #    2026-05-15   u.k.   added SE_Params_Container (host for Tr / doppler_shift_continuum, moved off Atmosphere)
    ```
  - Update the inline comment on `SE_Container.PI_intensity` at line 58 from `... planck(Tr) when atmos.use_Tr, ...` to `... planck(Tr) when se_params.Tr is not None, ...`.
  - Field comment for the new container:
    ```python
    Tr: T_FLOAT | None = None
    """None => use radiation.solar interpolation; not-None => planck(Tr)
    (Tr=0.0 is a valid coronal-equilibrium request; not the same as None)."""
    doppler_shift_continuum: T_BOOL = False
    ```
    (or single-line inline comments matching the existing style — choose based on what `SE_Container` does.)

- [ ] Step 1.2 — _(was: `Container/__init__.py` re-export check; codex audit verified `__init__.py:2` is `from .SEquil import *` with no `__all__`, so no edit needed. Skipped.)_

- [ ] Step 1.3 — `src/spectra/Struct/Atmosphere.py`:
  - `Atmosphere0D`: remove `Tr` (line 42), `use_Tr` (line 43), `doppler_shift_continuum` (line 47). Field ordering of remaining fields preserved; resulting fields are `Nh, Ne, Te, Vd, Vt, Ti, ndim, is_uniform, Pg, _coord_type`.
  - `AtmosphereC1D`: same removal at lines 69, 70, 72.
  - Update file-header changelog with a new entry (`0.1.3   2026-05-15   u.k.   move Tr / use_Tr / doppler_shift_continuum to SE_Params_Container`).
  - Verify `init_VAL_` constructor at line 986 does not pass any of the removed fields — quick read; today it only passes the survivors.

- [ ] Step 1.4 — `src/spectra/Struct/Radiation.py:34`: update docstring text replacing `atmos.use_Tr=True` with `se_params.Tr is not None` (or equivalent phrasing).

- [ ] Step 1.5 — `src/spectra/Function/SEquil/SELib.py`:
  - Import: `from ...Struct.Container import SE_Params_Container` (or via the existing `_Container` alias if already imported).
  - **File-header changelog comments at lines 10-11**: remove or rephrase the two lines that mention `use_Tr` / `doppler_shift_continuum` (`#   - func _bf_R_rate_ : move if of use_Tr outside of doppler_shift_continuum ...`). They are historical changelog text; cleaning them is required so the `grep "use_Tr\|doppler_shift_continuum" src/` acceptance gate is clean. Add a new changelog entry for this refactor (e.g. `# 0.1.5   2026-05-15   u.k.   se_params: SE_Params_Container threaded through cal_SE_* wrappers`).
  - **`cal_SE_` (the core, line 285):**
    - Add `se_params: _Container.SE_Params_Container` as a new required keyword arg (positioned after `Nh_SE`, before `rate_only` / `PI_intensity`).
    - Replace lines 336-340:
      ```python
      Tr = atmos.Tr
      use_Tr = atmos.use_Tr
      if atmos.doppler_shift_continuum:
          raise NotImplementedError(...)
      ```
      with:
      ```python
      use_Tr: T_BOOL = se_params.Tr is not None
      Tr: T_FLOAT = se_params.Tr if use_Tr else 0.0  # value is unused when use_Tr=False
      if se_params.doppler_shift_continuum:
          raise NotImplementedError("Doppler shift of continuum wavelength mesh not yet implemented.")
      ```
      (Pyright note: `se_params.Tr if use_Tr else 0.0` narrows correctly because `use_Tr` is `se_params.Tr is not None`.)
  - **Wrappers (`cal_SE_with_Pg_Te_single_Atom_` at line 75, `cal_SE_with_Pg_Te_` at line 124, `cal_SE_with_Nh_Te_` at line 177, `cal_SE_with_Ne_Te_` at line 224):**
    - Add `se_params: _Container.SE_Params_Container` to each signature, in the same position (after `Nh_SE`).
    - Forward to inner `cal_SE_` calls (lines 100, 147, 197, 263) as `se_params=se_params`.

- [ ] Step 1.6 — Source-side grep gate:
  ```bash
  grep -rn "atmos\.Tr\b\|atmos\.use_Tr\|atmos\.doppler_shift_continuum" src/
  # expected: empty
  grep -rn "atmos\.Tr\b\|atmos\.use_Tr\|atmos\.doppler_shift_continuum" tests/ scripts/
  # expected: empty (tests don't read these; Step 2 below handles the cal_SE_ kwargs)
  grep -rn "\.use_Tr\|\.doppler_shift_continuum" src/
  # expected: only inside Container/SEquil.py (SE_Params_Container definition) and SELib.py (se_params.… reads / unpacking)
  grep -rn "use_Tr\|doppler_shift_continuum" src/
  # expected: only the new SE_Params_Container definition + cal_SE_ unpacking + _B_Jbar_ primitive param.
  # The stale changelog-comment matches at SELib.py:10-11 must be gone (see Step 1.5).
  ```

**Phase 1 Exit Criteria:**
- [ ] All source files compile; `python -c "from spectra.Struct.Container import SE_Params_Container"` succeeds.
- [ ] Grep gates above are clean.

---

### Phase 2: Callers (tests, scripts) updated

> **Goal:** All `cal_SE_*` callers in `tests/` and `scripts/` pass the new `se_params` arg. Regression remains green.
> **Estimated Effort:** 0.25 day
> **Depends on:** Phase 1

- [ ] Step 2.1 — `tests/regression/test_reg_e2e_SE.py`: add `se_params=SE_Params_Container()` to all 9 `cal_SE_with_*_` calls. Import `SE_Params_Container` at the top alongside the existing imports.

- [ ] Step 2.2 — `tests/regression/test_reg_e2e_CloudModel.py`: 2 call sites.

- [ ] Step 2.3 — `tests/unittest/test.SE.H_I.py`: 1 call site.

- [ ] Step 2.4 — `tests/examples/example.SE.py`: 2 call sites. `tests/examples/example.CM.py`: 1 call site. `tests/examples/example.He.py`: 1 call site.

- [ ] Step 2.5 — `scripts/gen_se_reference.py`:
  - Add `SE_Params_Container` import.
  - At line 113 (`SE_con, _ = entry(atom, atmos, wMesh, radiation, None)`), pass `se_params=SE_Params_Container()` — defaults reproduce old behavior.
  - (Optional follow-thought: if `cases` definition could carry `se_params_kwargs`, hold off — single-developer project, no current per-case Tr customization in tests; YAGNI.)

- [ ] Step 2.6 — Regression gate:
  ```bash
  uv run pytest tests/regression/ -q
  # expected: 261/261 pass, zero diff against reference_values.json
  uv run pytest tests/unittest/ -q
  # expected: green
  uv run python tests/examples/example.SE.py
  uv run python tests/examples/example.CM.py
  uv run python tests/examples/example.He.py
  # expected: all complete without exception (they don't assert anything; smoke check)
  ```

**Phase 2 Exit Criteria:**
- [ ] All non-notebook callers updated.
- [ ] `pytest tests/regression/` is 261/261 green, zero numeric drift.
- [ ] `pytest tests/unittest/` is green.
- [ ] Pre-commit hooks pass (`ruff format`, `ruff check --fix`, `pyright`, `pytest-regression`, `protect-main-branch`).

---

### Phase 3: Draft PR & professor review handoff

> **Goal:** Open draft PR against `main` for professor's architectural review. Hand off to follow-up notebook PR.
> **Estimated Effort:** 0.1 day
> **Depends on:** Phase 2

- [ ] Step 3.1 — Push feature branch to remote: `git push -u origin refactor/se-params-container`.
- [ ] Step 3.2 — Open PR as **draft** with title `refactor(SE): move Tr / use_Tr / doppler_shift_continuum into SE_Params_Container`. Body links to issue #5 and explicitly states "notebooks broken on this branch — fix lands via a follow-up PR targeting this branch".
- [ ] Step 3.3 — In the PR body, include the grep-gate commands and their expected-clean output for the reviewer.
- [ ] Step 3.4 — Notify professor; await architectural feedback.

**Phase 3 Exit Criteria:**
- [ ] Draft PR open against `main`.
- [ ] Notebook follow-up PR queued (targeting this feature branch, not main).
- [ ] This PR remains draft until the notebook PR merges in.

---

## 3. Boundaries — Do NOT Touch

> These files, modules, and APIs are explicitly out of scope.

| Area | Path / Identifier | Reason |
|------|-------------------|--------|
| Notebooks | `notebooks/**/*.ipynb` | Handled in a follow-up PR targeting this feature branch. Updating them here would noise up the architectural review |
| Reference values | `tests/regression/reference_values.json` | Refactor is behavior-preserving; any diff is a regression |
| Numba-jitted inner functions | `_B_Jbar_`, `_bf_R_rate_` in `Function/SEquil/SELib.py` | Keep primitive `Tr, use_Tr` signature — JIT-friendly, and the boundary is unpacked above them |
| Photoionize primitive | `src/spectra/Atomic/PhotoIonize.py:interpolate_PI_intensity_` | No reference to the migrated fields; unrelated |
| Predecessor task docs | `docs/tasks/00[1-6]-*/` | Historic audit trail; do not edit |
| Removed Icp module | `src/spectra/Function/Icp/` | Deletion is committed separately; not part of this refactor |

**Rule of thumb:** If a change requires modifying anything listed above, stop and revisit the plan.

---

## 4. Test Coverage

### Testing strategy

| Level | Scope | Tool |
|-------|-------|------|
| Regression | 261 e2e tests, zero numeric drift expected | `pytest tests/regression/` |
| Unit | `test.SE.H_I.py` and any other unit suite | `pytest tests/unittest/` |
| Smoke | Run each example script end-to-end | `uv run python tests/examples/example.*.py` |
| Static | Pyright + ruff via pre-commit | `pre-commit` |

### Required new tests

None. The refactor is behavior-preserving; existing regression coverage (predecessor task #003 established the 261-test floor) covers the affected path. Adding a `SE_Params_Container` constructor test would be busywork — the struct has no logic.

### Coverage target

- 261/261 regression tests passing.
- Zero diff against `reference_values.json`.

---

## 5. Key Decisions

### Decision 1: Collapse `Tr` + `use_Tr` into `Tr: T_FLOAT | None`

- **Context:** The old pair `(Tr: float, use_Tr: bool)` encodes a tagged union of "use Planck at temperature Tr" vs "use solar interpolation". Two fields can drift apart; one nullable field can't.
- **Options considered:**
  1. Keep both fields (`Tr: float, use_Tr: bool`) in the new container, mirroring the old shape exactly.
  2. Replace with `Tr: float | None` — None ⇒ solar, not-None ⇒ Planck.
  3. Add an enum (`RadiationSource.SOLAR / PLANCK`) + always-present `Tr: float`.
- **Decision:** Option 2.
- **Rationale:** User preference (this conversation). Encodes the invariant directly: there is no `Tr` value to interpret when `use_Tr=False`, so absence is the right model. Preserves the coronal-equilibrium edge case (`Tr=0.0`, non-None) without ambiguity.
- **Consequences:** Slightly more annotation noise downstream (`Tr if use_Tr else 0.0` unpacking) but only at the single boundary in `cal_SE_`. Pyright narrows correctly given the `use_Tr = se_params.Tr is not None` derivation.

### Decision 2: Container, not loose function kwargs

- **Context:** Issue #5 proposed a struct. Could also have used 2 explicit kwargs on `cal_SE_` and its 4 wrappers.
- **Options considered:**
  1. Pass `Tr`, `use_Tr`, `doppler_shift_continuum` as 3 plain kwargs on every wrapper.
  2. Pass a single `SE_Params_Container` struct.
- **Decision:** Option 2.
- **Rationale:** Wrapper chain is 4-deep; 3 kwargs per signature × 5 sites = 15 redundant lines. A struct concentrates the choice at one construction site, and future SE switches (PRD flag, etc.) extend the struct without churning signatures. User-stated preference.
- **Consequences:** One extra dataclass; callers construct it explicitly (no implicit default at the call boundary).

### Decision 3: Notebooks deferred to follow-up PR

- **Context:** Notebook construction sites reference `Atmosphere0D(..., use_Tr=...)` in ~20 places. Including them in this PR triples the diff size.
- **Options considered:**
  1. One PR with everything (source + tests + notebooks).
  2. Source-and-tests PR first; notebook PR follows, targets this feature branch.
  3. Source-only PR; tests AND notebooks follow.
- **Decision:** Option 2.
- **Rationale:** User wants to show the draft to the professor for architectural feedback before sinking time into notebook edits. Tests/scripts must be in this PR (otherwise regression isn't reviewable). Notebooks are inert documentation — they can lag without affecting the review.
- **Consequences:** This PR's branch state has all notebooks broken at import time. The follow-up notebook PR targets `refactor/se-params-container` (not `main`); this PR doesn't merge until the notebook PR has merged into it.

### Decision 4: No deprecation alias

- **Context:** Could keep the 3 fields on `Atmosphere` as deprecated-and-ignored, or accept both kwarg forms on `cal_SE_*`.
- **Options considered:**
  1. Hard remove. No fallback.
  2. Leave fields on `Atmosphere` but make `cal_SE_*` ignore them (warns on use).
  3. Accept both `se_params` and the old per-field kwargs on `cal_SE_*` with a `DeprecationWarning`.
- **Decision:** Option 1.
- **Rationale:** Project conventions (`CLAUDE.md`: "don't add backwards-compatibility shims when you can just change the code"). Single-developer codebase; out-of-tree exposure is zero. Identical precedent in task #005 (decision 4).
- **Consequences:** Notebooks on the feature branch are broken until the follow-up PR. Accepted (decision 3 above).

### Decision 5: Inner functions keep primitive args

- **Context:** Could pass `se_params` deep into `_B_Jbar_` and `_bf_R_rate_`.
- **Options considered:**
  1. Pass primitives `(Tr, use_Tr)` to `_B_Jbar_` (unchanged signature).
  2. Pass the whole `SE_Params_Container` into `_B_Jbar_`.
- **Decision:** Option 1.
- **Rationale:** `_B_Jbar_` is on the JIT boundary; passing a dataclass into a numba region requires reflected types or `@jitclass`, neither of which the project uses. Unpacking once in `cal_SE_` localizes the struct dependency at the public boundary and leaves the compute kernel untouched.
- **Consequences:** Trivial — one unpacking line in `cal_SE_`.

### Decision 6: Single struct hosts both `Tr` and `doppler_shift_continuum`

- **Context:** `doppler_shift_continuum` is currently a stub (`NotImplementedError`). Could defer it.
- **Options considered:**
  1. Both fields in `SE_Params_Container` now.
  2. Only move `Tr` now; leave `doppler_shift_continuum` on `Atmosphere` until it's actually implemented.
- **Decision:** Option 1.
- **Rationale:** User preference. Pulling both at once means a single migration; pulling them separately means a second round of changes to the same wrapper chain when the doppler-shift code arrives. The empty-default field costs nothing.
- **Consequences:** `SE_Params_Container` has two fields from the start; the second one is dormant until the doppler-shift implementation lands.

---

## 6. Precautions

### Technical risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pyright narrowing fails on `Tr: T_FLOAT | None` unpack | Low | Med | The `use_Tr = se_params.Tr is not None` pattern is the canonical narrow; if pyright complains, use an explicit `if … is None: … else: …` block. |
| Numba JIT cache stale for `_B_Jbar_` | Very low | Low | Inner signature unchanged. Numba indexes by position, not surrounding code. Fresh compile expected and accepted. |
| `init_VAL_` construction breaks | Very low | Low | Audit verified it doesn't pass the removed fields. Re-grep at Step 1.3. |
| Regression tests pass by accident (default = old behavior, so no exercise of new code path) | Med | Low | Defaults reproduce old behavior intentionally — that *is* the test. No notebook exercises non-default `Tr`; the existing regression matrix doesn't either. Acceptable: this PR is a refactor, not a feature add. |
| Reviewer (professor) confuses `Tr=None` vs `Tr=0.0` | Med | Low | Inline field comment in `SE_Params_Container` calls out the distinction. Plan §5 Decision 1 captures rationale. |

### Rollback plan

If a post-merge issue surfaces:

1. `git revert <merge-commit>` of this PR's eventual merge.
2. Notebook PR will need re-revert as well (it'd be on top); same `git revert` mechanism.
3. Re-open issue #5 with the failure mode for re-evaluation.

No data migration, no `reference_values.json` change, no external-consumer coordination required.

### Migration notes

- **Backward compatibility:** None preserved. Single-developer project; out-of-tree exposure ≈ 0.
- **Feature flag:** N/A.
- **Migration script:** N/A — call-site edits are mechanical and listed exhaustively in Phase 2.

### Performance considerations

- Refactor changes parameter passing only; instruction count in the SE solve loop is unchanged.
- One additional pointer-deref (`se_params.Tr`) per `cal_SE_` invocation — negligible vs the SE solver cost.

### Security considerations

- N/A.

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-05-15 | kouui | Initial draft after grep audit and design discussion (issue #5) |
| 2026-05-15 | kouui | Apply codex scope-review fixes (`tmp/codex_outputs/007_scope_review.md`): drop dead `Container/__init__.py` step, add SELib.py:10-11 changelog-comment cleanup, correct notebook scope (15 notebooks / ~111 sites, not ~4 / ~20) |
| 2026-05-15 | kouui | Make `se_params` optional (`SE_Params_Container \| None = None`) across all 5 `cal_SE_*` signatures; `cal_SE_` defaults a fresh container when `None`. This matches the existing `Nh_SE: T_ARRAY \| None` / `PI_intensity: T_ARRAY \| None` idiom and keeps the common-case (solar background) call site invocation-light. Drop the 18 now-redundant `Container.SE_Params_Container()` calls from tests/scripts + their `Container` imports. Decision 4 (no deprecation alias) still holds — this is optional-arg idiom, not a backward-compat shim. |
