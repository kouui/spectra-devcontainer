# Task: Rename `cont_intensity` → `PI_intensity` across the SE pipeline

> **Status:** Draft
> **Owner:** kouui
> **Created:** 2026-05-09
> **Last Updated:** 2026-05-09

## Objective

Rename the continuum-mesh-resolved bound-free intensity from `cont_intensity` to `PI_intensity` (short for *photoionization intensity*) across the SE pipeline. The new name aligns with the existing project vocabulary (`atom.PI`, `PhotoIonize.interpolate_PI_intensity_`) and emphasizes the physics role (driver of bound-free rates) over the storage detail (continuum mesh). Behavior is preserved bit-for-bit; this is a pure rename.

## Background & Context

PR #17 (issue #6 refactor, commit `dbf242d`) introduced `SE_Container.cont_intensity` along with a parameter-and-variable chain that propagates through `cal_SE_`, `_bf_R_rate_`, three SEquil wrappers, and two `Hydrogen_atom.ipynb` cells. The name `cont_intensity` was chosen during design discussion to describe "intensity already mapped to the continuum wavelength mesh" — but the field's actual job is to drive photoionization (b-f) rates, and the project already has a `Radiation.PI_intensity` history (deleted in #17), `atom.PI.alpha_interp`, and `interpolate_PI_intensity_`.

User reviewed the post-merge codebase and prefers `PI_intensity` for vocabulary consistency. The earlier deletion of `Radiation.PI_intensity` does not collide: the new field lives on `SE_Container` (an SE call result), not on `Radiation` (a radiation-field input), so the names occupy different structs and different lifetimes.

A scope audit was performed twice (grep + codex sub-audit) before writing this task:

- **`src/`**: 27 literal occurrences across 16 lines in 2 files (`Struct/Container/SEquil.py`, `Function/SEquil/SELib.py`).
- **`tests/`**: 0.
- **`scripts/`**: 0 (no consumer of the field; `getattr(SE_con, field)` whitelist in `gen_se_reference.py` does not include it).
- **`notebooks/`**: 6 occurrences in 3 files (`Hydrogen_atom.ipynb`, `StatisticalEquilibrium/Hydrogen_atom.ipynb`, `StatisticalEquilibrium/He_plasma.ipynb`).
- **`docs/tasks/004-radiation-pi-intensity-refactor/`**: 73 occurrences (historic task documents — preserved as audit trail, not changed).
- No indirect references found: no `getattr`/`setattr` string usage, no `__dataclass_fields__`/`dataclasses.fields` reflection, no `asdict`/JSON/pickle key dependence, no `reference_values.json` key, no `_atom_serde` involvement.

References:

- Predecessor PR / commit: PR #17, commit `dbf242d` (`refactor(radiation): drop cached PI_intensity, rename backRad -> solar`).
- Codex scope audit output: `tmp/codex_outputs/cont_intensity_scope.md`.

## Requirements

### Functional Requirements

1. `SE_Container.cont_intensity` field renamed to `SE_Container.PI_intensity`. Type and shape unchanged: `T_ARRAY`, 2-D `(nCont, _N_CONT_MESH)`, `[erg/cm^2/Sr/cm/s]`. Position in the dataclass unchanged.
2. `cal_SE_(..., cont_intensity: T_ARRAY | None = None)` keyword renamed to `PI_intensity`. Default value, type, and override semantics unchanged.
3. `_bf_R_rate_(Cont, Cont_mesh, Te, nj_by_ni_Cont, alpha_interp, cont_intensity)` parameter renamed to `PI_intensity`. Argument order and types unchanged. Numba JIT decoration unchanged.
4. The 3 SEquil wrappers (`cal_SE_with_Pg_Te_single_Atom_`, `cal_SE_with_Pg_Te_`, `cal_SE_with_Nh_Te_`) update their local capture variable from `cont_intensity` to `PI_intensity`. Capture-and-reuse pattern unchanged. Single-call wrappers (`cal_SE_with_Ne_Te_`) unaffected.
5. The 3 affected notebooks update their local variable / call-site references:
   - `notebooks/Hydrogen_atom.ipynb` (root): comment + assignment + positional pass-through.
   - `notebooks/StatisticalEquilibrium/Hydrogen_atom.ipynb`: assignment + positional pass-through.
   - `notebooks/StatisticalEquilibrium/He_plasma.ipynb`: comment.
6. Inline `cal_SE_` dispatch (`if cont_intensity is None: ...`) and the surrounding local variables update consistently.
7. `SE_Container(..., cont_intensity=cont_intensity)` construction site in `cal_SE_` updates to keyword `PI_intensity=PI_intensity`.

### Non-Functional Requirements

- **Zero numerical drift**: `pytest tests/regression/` must pass 261/261 against the existing `reference_values.json`. Rename is behavior-preserving; if any test diff appears, stop and audit before touching JSON.
- **No new files, no API surface growth**: only identifier substitutions.
- **Numba**: `_bf_R_rate_` is `@nb_njit`. Numba indexes parameters by position, not name; the rename is transparent to the JIT cache. A fresh compile on first invocation is expected and accepted.
- **Public-API caveat**: `cal_SE_(..., cont_intensity=)` is a public keyword; external callers (out-of-tree notebooks) using the old kwarg will break. The project has no compatibility shim policy and the user accepts the break (single-developer project; predecessor PR #17 was merged 2 days ago, so external exposure is minimal).

## Scope

### In Scope

- [ ] `src/spectra/Struct/Container/SEquil.py` — field declaration + changelog block.
- [ ] `src/spectra/Function/SEquil/SELib.py` — 16 lines (wrappers, `cal_SE_` kwarg + body + construction, `_bf_R_rate_` parameter + body, TODO comment).
- [ ] `notebooks/Hydrogen_atom.ipynb` — comment + 2 code references.
- [ ] `notebooks/StatisticalEquilibrium/Hydrogen_atom.ipynb` — 2 code references.
- [ ] `notebooks/StatisticalEquilibrium/He_plasma.ipynb` — 1 comment reference.

### Out of Scope (Boundaries)

> Items explicitly excluded from this task. Do NOT touch these areas.

- **`docs/tasks/004-radiation-pi-intensity-refactor/`:** 73 occurrences in task.md / plan.md. Historic audit documents — must reflect the original-decision name to remain truthful.
- **`tests/regression/reference_values.json`:** must not change. Rename is behavior-preserving; any change is a regression.
- **`scripts/gen_se_reference.py`:** no `cont_intensity` reference today. Do not preemptively add the new field to the whitelist.
- **`src/spectra/Atomic/PhotoIonize.py:interpolate_PI_intensity_`:** function name already uses `PI_intensity` — unchanged.
- **`src/spectra/Function/Icp/SELib.py`:** stub file gated by `raise ImportError`. The reference code below the raise is preserved verbatim from before #17 and is not exercised; no rename needed.
- **Other regression tests, kernel tests, atom load tests:** unchanged.

## Acceptance Criteria

- [ ] `grep -rn "cont_intensity" src/ tests/ scripts/ notebooks/` returns zero matches (only `docs/tasks/004-...` retains the old name).
- [ ] `SE_Container.PI_intensity` attribute exists; `SE_Container.cont_intensity` does not.
- [ ] `cal_SE_(..., PI_intensity=...)` accepts the new keyword; the old keyword is gone.
- [ ] `_bf_R_rate_` signature is `(Cont, Cont_mesh, Te, nj_by_ni_Cont, alpha_interp, PI_intensity)`.
- [ ] All 3 SEquil wrappers reuse via `PI_intensity = SE_con.PI_intensity` capture line.
- [ ] `pytest tests/regression/` is green (261/261), zero diff against `reference_values.json`.
- [ ] All 3 modified notebooks parse as valid JSON.
- [ ] Pre-commit hooks pass: `protect-main-branch` (run on a feature branch), `ruff format`, `ruff check --fix`, `pyright`, `pytest-regression`.

## Dependencies

| Dependency | Owner | Status | Notes |
|------------|-------|--------|-------|
| PR #17 merged into main | kouui | ✅ Done | commit `dbf242d`; this task starts from that baseline |
| Codex scope audit | codex | ✅ Done | `tmp/codex_outputs/cont_intensity_scope.md` |

## Risks & Open Questions

- [ ] **Numba JIT cache**: theoretical risk that a stale JIT cache for `_bf_R_rate_` keyed on the old signature confuses dispatch. Numba does not key on parameter names; risk rejected. If observed, clear `.numba_cache` and rerun.
- [ ] **`asdict(SE_con)` external consumer**: out-of-tree code might depend on the dict key `cont_intensity`. Repository scan finds zero such consumers; risk accepted.
- [ ] **Notebook semantic drift**: notebooks store user-facing variable names; readers may mistake `PI_intensity` (now an `SE_Container` field) for the deleted `Radiation.PI_intensity` (issue #6). Mitigation: keep the field comment in `Container/SEquil.py` explicit about which struct hosts it.

## References

- PR #17 (predecessor): https://github.com/kouui/spectra-devcontainer/pull/17
- Codex scope audit: `tmp/codex_outputs/cont_intensity_scope.md`
- Predecessor task documents: `docs/tasks/004-radiation-pi-intensity-refactor/{task.md,plan.md}`
