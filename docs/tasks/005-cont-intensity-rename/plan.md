# Plan: Rename `cont_intensity` → `PI_intensity` across the SE pipeline

> **Task:** [task.md](./task.md)
> **Owner:** kouui
> **Created:** 2026-05-09
> **Target Completion:** 2026-05-09

---

## 0. Context

> **Objective:** Rename `cont_intensity` to `PI_intensity` across `SE_Container` field, `cal_SE_` kwarg, `_bf_R_rate_` parameter, 3 SEquil wrappers, and 3 notebooks. Behavior preserved bit-for-bit.
> **Full spec:** [task.md](./task.md)

---

## 1. Overall Architecture

### What changes

This is a pure identifier substitution. No structural, signature-shape, or semantic changes.

```
SE_Container.cont_intensity                         →  SE_Container.PI_intensity
cal_SE_(..., cont_intensity=...)                    →  cal_SE_(..., PI_intensity=...)
_bf_R_rate_(... , cont_intensity)                   →  _bf_R_rate_(... , PI_intensity)
wrapper local: cont_intensity = SE_con.cont_intensity
                                                    →  wrapper local: PI_intensity = SE_con.PI_intensity
notebook local: cont_intensity = LTELib.planck_cm_(...)
                                                    →  notebook local: PI_intensity = LTELib.planck_cm_(...)
```

### Affected components

| Component | File | Lines (approx.) | Role |
|-----------|------|-----------------|------|
| `SE_Container` field | `src/spectra/Struct/Container/SEquil.py` | 12, 58 | dataclass field declaration + changelog |
| `cal_SE_` kwarg + body | `src/spectra/Function/SEquil/SELib.py` | 292, 295, 346, 348, 350, 380, 430 | public boundary + dispatch + construction |
| `_bf_R_rate_` parameter + body | `src/spectra/Function/SEquil/SELib.py` | 526, 544 | numba-decorated rate calculator |
| 3 SEquil wrappers | `src/spectra/Function/SEquil/SELib.py` | 97, 100, 101 / 144, 147, 148 / 194, 197, 198 | capture-and-reuse local variable |
| `Hydrogen_atom.ipynb` (root) | `notebooks/` | 865, 868, 874 | comment + assignment + pass-through |
| `Hydrogen_atom.ipynb` (StatEq) | `notebooks/StatisticalEquilibrium/` | 870, 876 | assignment + pass-through |
| `He_plasma.ipynb` | `notebooks/StatisticalEquilibrium/` | 340 | comment |

### Why one commit (not multi-phase)

Identifier rename in a closed dependency loop — `cal_SE_` writes the field, wrappers read it, `_bf_R_rate_` consumes the kwarg, the dataclass holds the field. Splitting into multiple commits would leave intermediate states with `AttributeError` (writer renamed, reader not) that the `pytest-regression` pre-commit hook would catch and reject. Single commit avoids that.

Notebooks are independent of source-side rename at runtime (notebooks pass positional args to `_bf_R_rate_`; the parameter name doesn't appear in the call), but their local variables / comments still refer to `cont_intensity` and would create stale-vocabulary noise in a follow-up commit. Bundling notebooks into the same commit keeps `grep cont_intensity` clean atomically.

---

## 2. Implementation Phases

### Phase 1: Single-shot rename across source + notebooks

> **Goal:** Atomically rename across all 5 files; verify with grep + regression; commit.
> **Estimated Effort:** 0.25 day

- [ ] Step 1.1 — `src/spectra/Struct/Container/SEquil.py`:
  - line 58: rename field `cont_intensity` → `PI_intensity`. Update inline comment (`## continuum intensity used to drive bound-free rates...`) to read `## PI intensity (continuum-mesh-resolved) used to drive bound-free rates`.
  - line 12: changelog block — replace `added cont_intensity, cont_wave_mesh_shifted` with `added PI_intensity (continuum-mesh-resolved bound-free intensity), cont_wave_mesh_shifted`. Bump nothing; this is a clarifying rename, not a new version.

- [ ] Step 1.2 — `src/spectra/Function/SEquil/SELib.py` (single-pass rename, 16 lines):
  - line 292: `cal_SE_` kwarg: `cont_intensity: T_ARRAY | None = None` → `PI_intensity: T_ARRAY | None = None`.
  - line 295: TODO comment — update `cont_intensity` mention.
  - lines 346, 348, 350: dispatch block local variable rename.
  - line 380: positional pass-through to `_bf_R_rate_`.
  - line 430: `SE_Container(..., cont_intensity=cont_intensity)` → `SE_Container(..., PI_intensity=PI_intensity)`.
  - line 526: `_bf_R_rate_` parameter: `cont_intensity: T_ARRAY` → `PI_intensity: T_ARRAY`.
  - line 544: parameter read inside loop body: `J=cont_intensity[kL, ::-1]` → `J=PI_intensity[kL, ::-1]`.
  - lines 97, 100, 101: `cal_SE_with_Pg_Te_single_Atom_` wrapper — rename the local variable (`cont_intensity = None`), the kwarg in `cal_SE_(..., cont_intensity=cont_intensity)`, and the capture (`cont_intensity = SE_con.cont_intensity`).
  - lines 144, 147, 148: `cal_SE_with_Pg_Te_` wrapper — same 3-line pattern.
  - lines 194, 197, 198: `cal_SE_with_Nh_Te_` wrapper — same 3-line pattern.

- [ ] Step 1.3 — Notebooks (no semantics change, just identifier substitution):
  - `notebooks/Hydrogen_atom.ipynb` (root): cell at line ~865-874 — rename comment + `cont_intensity = LTELib.planck_cm_(...)` + positional argument in `SELib._bf_R_rate_(...)` call.
  - `notebooks/StatisticalEquilibrium/Hydrogen_atom.ipynb`: cell at line ~870-876 — same 2 changes as above.
  - `notebooks/StatisticalEquilibrium/He_plasma.ipynb`: line ~340 — comment update only.
  - Edit via direct JSON write (notebooks have no cell IDs that NotebookEdit needs); preserve `outputs` / `execution_count` / `metadata` from HEAD by editing only `source` arrays.

- [ ] Step 1.4 — Verify grep is clean:
  ```bash
  grep -rn "cont_intensity" src/ tests/ scripts/ notebooks/
  # expected: empty output
  grep -rn "cont_intensity" docs/tasks/
  # expected: only docs/tasks/004-radiation-pi-intensity-refactor/ matches (historic, preserved)
  ```

- [ ] Step 1.5 — Regression:
  ```bash
  uv run pytest tests/regression/ -q
  # expected: 261/261 pass, zero diff against reference_values.json
  ```

- [ ] Step 1.6 — JSON validity check on the 3 notebooks:
  ```bash
  python3 -c "import json; [json.load(open(p)) for p in ['notebooks/Hydrogen_atom.ipynb','notebooks/StatisticalEquilibrium/Hydrogen_atom.ipynb','notebooks/StatisticalEquilibrium/He_plasma.ipynb']]"
  ```

**Phase 1 Exit Criteria:**
- [ ] `grep -rn "cont_intensity" src/ tests/ scripts/ notebooks/` returns nothing.
- [ ] `pytest tests/regression/` is 261/261 green, zero numeric drift.
- [ ] All 3 modified notebooks parse as valid JSON.
- [ ] Branch is on a feature branch (not `main`); pre-commit hooks pass when staging.

---

## 3. Boundaries — Do NOT Touch

> These files, modules, and APIs are explicitly out of scope.

| Area | Path / Identifier | Reason |
|------|-------------------|--------|
| Predecessor task docs | `docs/tasks/004-radiation-pi-intensity-refactor/{task,plan}.md` | Historic audit trail — must reflect original-decision name to stay truthful |
| Reference values | `tests/regression/reference_values.json` | Rename is behavior-preserving; any diff is a regression |
| SE reference generator | `scripts/gen_se_reference.py` | No `cont_intensity` reference today; do not preemptively add `PI_intensity` to whitelist |
| Photoionize primitive | `src/spectra/Atomic/PhotoIonize.py:interpolate_PI_intensity_` | Function name already uses `PI_intensity` — unchanged |
| Icp stub | `src/spectra/Function/Icp/SELib.py` | Gated by `raise ImportError`; reference code below the raise is preserved verbatim |
| Other regression tests / kernel tests / AtomLoad | `tests/regression/test_reg_*` | Unaffected by the rename |

**Rule of thumb:** If a change requires modifying anything listed above, stop and revisit the plan.

---

## 4. Test Coverage

### Testing Strategy

| Level | Scope | Tool |
|-------|-------|------|
| Regression | Existing 261 e2e tests must remain green with zero numeric drift | `pytest tests/regression/` |
| Static | `pyright` type check via pre-commit hook | `pyright` |
| Format | `ruff format` + `ruff check --fix` via pre-commit | `ruff` |
| Manual smoke | Spot-check that `python tests/examples/example.SE.py` still runs | manual |

### Required Test Cases

No new tests are added. Predecessor task #004 already established the e2e regression coverage; this rename has zero behavioral surface to test independently.

### Coverage Target

- 261/261 regression tests passing.
- Zero diff against `reference_values.json`.

---

## 5. Key Decisions

### Decision 1: Rename to `PI_intensity` instead of keeping `cont_intensity`

- **Context:** During PR #17 design, `cont_intensity` was chosen to emphasize "intensity already mapped to the continuum mesh". After the merge, the user reviewed and prefers `PI_intensity` for vocabulary consistency with `atom.PI`, `PhotoIonize.interpolate_PI_intensity_`, and the project's general naming conventions.
- **Options Considered:**
  1. Keep `cont_intensity` (no work).
  2. Rename to `PI_intensity` — physics-role-emphasized.
  3. Rename to `cont_PI_intensity` — both axes.
- **Decision:** Option 2.
- **Rationale:** The struct it lives on (`SE_Container`) already disambiguates the lifetime ("an SE call result, not a Radiation field"). The `PI_` prefix is the project's established prefix for photoionization-related quantities. Option 3 is verbose without adding clarity; the storage layout is documented in the field comment.
- **Consequences:** External callers using `cal_SE_(..., cont_intensity=)` will break. Project is single-developer; out-of-tree exposure is negligible.

### Decision 2: Single commit, not multi-phase

- **Context:** PR #17 used multi-phase commits (source + callers, then notebooks). This rename touches the same fan-out (struct + cal_SE_ + wrappers + _bf_R_rate_ + notebooks).
- **Options Considered:**
  1. One commit covering everything.
  2. Two commits: source-side rename, then notebooks.
- **Decision:** Option 1.
- **Rationale:** Source-side rename is internally coupled (writer/reader pair); splitting causes intermediate `AttributeError`. Notebooks are technically independent (positional args), but bundling avoids a `cont_intensity` grep noise window between commits. The whole change is small (33 substitutions) and atomic.

### Decision 3: Don't bump `SE_Container` version

- **Context:** Project convention has changelog blocks at file top. The `SE_Container` block has version `0.2.0` from PR #17.
- **Options Considered:**
  1. Bump to `0.2.1` (clarifying rename).
  2. Update `0.2.0` entry text to reflect the new name.
- **Decision:** Option 2.
- **Rationale:** No external API contract to track; the rename is a clarification, not a behavioral change. Bumping creates change-log noise without informational value.

### Decision 4: No deprecation alias for the old kwarg

- **Context:** Could add `cont_intensity` as a deprecated alias keyword to `cal_SE_` for one release.
- **Options Considered:**
  1. Hard rename, no shim.
  2. Accept both keywords with a `DeprecationWarning` for the old one.
- **Decision:** Option 1.
- **Rationale:** Project guideline: "don't add backwards-compatibility shims when you can just change the code"; "trust internal code and framework guarantees". No external consumers; the cost of a shim is permanent dead code.

---

## 6. Precautions

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Numba JIT mismatch on `_bf_R_rate_` | Very low | Low | Numba indexes by position, not name. Fresh compile expected. |
| Stray `cont_intensity` reference missed by grep | Low | Medium | Codex secondary audit confirmed no indirect references; final grep gate before commit. |
| Notebook JSON corruption from manual edit | Low | Low | Direct JSON write preserves outputs/metadata; final `json.load` validation gate. |
| Pre-commit pytest fails because rename leaves an inconsistent state | Low | High | Single commit ensures atomic application; intermediate state never staged. |

### Rollback Plan

If post-merge issue surfaces:

1. `git revert <commit>` on a feature branch.
2. Open a new task to re-evaluate the rename if user changes mind.

No data migration, no JSON schema change, no external-consumer coordination required.

### Migration Notes

- **Backward compatibility:** None preserved. Project is single-developer; out-of-tree exposure is minimal.
- **Feature flag:** N/A.
- **Migration script:** N/A.

### Performance Considerations

- Identifier rename does not change instruction count or memory layout.
- `numpy.interp` and `planck_cm_` calls are unchanged.

### Security Considerations

- N/A.

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-05-09 | kouui | Initial draft after grep + codex scope audit |
