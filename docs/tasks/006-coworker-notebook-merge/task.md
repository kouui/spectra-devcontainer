# Task: Merge coworker notebook updates into upstream notebooks (with PR #17/#19 API preservation)

> **Status:** Draft
> **Owner:** kouui
> **Created:** 2026-05-09
> **Last Updated:** 2026-05-09

## Objective

Coworker shared two updated notebooks (`notebooks_local/hydrogen.ipynb`, `notebooks_local/StatisticalEquilibrium/H_spectra.ipynb`) that contain substantive new content the upstream notebooks lack. Merge those updates into the upstream notebooks at the matching paths, while ensuring PR #17 (`backRad → solar`, drop `Radiation.PI_intensity`, `init_Radiation_(path)` signature) is correctly applied to the coworker source. Output cells in the upstream notebooks must match coworker's runs.

## Background & Context

Two upstream-vs-coworker notebook pairs:

| Coworker | Upstream | source diff lines | Output cells used by | API gap |
|---|---|---|---|---|
| `notebooks_local/hydrogen.ipynb` | `notebooks/hydrogen.ipynb` | 37 | coworker | none — notebook does not use `Radiation`/`SE_Container` API |
| `notebooks_local/StatisticalEquilibrium/H_spectra.ipynb` | `notebooks/StatisticalEquilibrium/H_spectra.ipynb` | 324 | coworker | PR #17: `backRad → solar`, drop standalone `PI_intensity` access, `init_Radiation_(...)` signature |

Source-cell diff (extracted via `jq` and persisted at `tmp/nb_diff/`):

### `hydrogen.ipynb` — 37 lines, all cosmetic
- Image attachment ref name (`image-2.png` vs `image.png`)
- One commented-out `set_xscale('log')` line removed
- Plot title text (`'b-b Oscillator strength'` vs `'Oscillator strength'`)
- `nmax/kmax` 20 vs 10 (resolution upgrade)
- Plot setting layout (1-line vs 3-line)
- Latex string raw-string vs plain
- Comment separator length
- No PR #17/#19/#20 API references on either side.

### `H_spectra.ipynb` — 324 lines, three categories

**Category A — PR #17 API renames (apply upstream new API to coworker source):**

| Coworker (old) | New API |
|---|---|
| `Radiation.init_Radiation_(atmos, wMesh)` (line 78) | `Radiation.init_Radiation_()` |
| `Radiation.init_Radiation_(slab, wMesh)` (line 598) | `Radiation.init_Radiation_()` |
| `solar_spec.backRad[0,:]` (line 81) | `solar_spec.solar[0,:]` |
| `solar_spec.backRad[1,:].copy()` (line 83) | `solar_spec.solar[1,:].copy()` |
| paired `solar_spec.backRad[1,:] *= 0.5; solar_spec.PI_intensity[:,:] *= 0.5` (lines 86–87) | single `solar_spec.solar[1,:] *= 0.5` |
| paired `ssp.backRad[1,:] *= Jfactor; ssp.PI_intensity[:,:] *= Jfactor` (lines 100–101, 330–331, 391–392) | single `ssp.solar[1,:] *= Jfactor` (× 3 sites) |
| standalone `solar_spec.PI_intensity[0,:]` diagnostic line (line 108) | delete (no replacement) |
| `ssp.backRad[0,:]` (line 600) | `ssp.solar[0,:]` |
| string-literal print of `ssp.backRad[1,:]` × 4 (lines 1113–1116) | rename to `ssp.solar[1,:]` (both code expression and the literal text inside `f"..."`) |

**Category B — Coworker's substantive additions (already in coworker, kept verbatim):**
- Cell #32: tau profile diagnostic plot, with comment "tau profile can be negative in over population of higher levels" — directly aligns with issue #18 / PR #20.
- Cell #33: `Peak_intensity(l)` helper using `np.abs(l.tau_1D).argmax()` to pick line peak intensity — same conceptual fix as PR #20 in the notebook layer.
- Cell #34: rewrite consuming `Peak_intensity(l)` instead of `l.Ipeak[ii]` (the latter is broken — `CloudModel_Container` has no `Ipeak` field; upstream's reference is stale).
- `Ne = 1.E13` (vs upstream `1.E11`).
- `depth = 500 * 1.E5  # km *1e5 -> cm` comment.
- ylabel `'population, n_SE'` (vs `'population'`).
- `wrange = [3500, 11000]` (vs `[3500, 20000]`).
- raw-string `r"..."` for several latex labels.

**Category C — Upstream-only refinements (NOT ported back; coworker version kept):**
- **First** Cmat title site (coworker line 187 of `tmp/nb_diff/Hspectra.cw.txt`, terse `'Cmat (theoretical)'`) stays terse; upstream's elaborated form `'Cmat (theoretical): '+r"$\overline{J}=$"+f"{Jfactor:3.1F}"` is **not** ported. The other two sites (coworker lines 909 and 953) already use the elaborated form in coworker — they match upstream and need no action.
- Upstream's *additional* `#help_(atom)` placeholder (cell #30 in upstream's count, around line 625 of `tmp/nb_diff/Hspectra.up.txt`) is dropped. Coworker has its **own** `#help_(atom)` early in the notebook (line 36 of the dump) — that one stays; only the upstream-only second occurrence is excluded from the merge.
- Minor whitespace / blank-line reflow throughout.

User explicitly approved keeping coworker for Category C.

### `init_theoretical_hydrogen_atom_` correctness

Upstream notebook calls `atom, wMesh, path_dict = Atom.init_theoretical_hydrogen_atom_(...)` (3-tuple), but the function (`src/spectra/Struct/Atom.py:262`) currently returns 2-tuple after `dadc110 fix: Atom.init_theoretical_hydrogen_atom_ returns tuple of two value instead of three`. Upstream notebook is **stale** on this signature; coworker's `atom, wMesh = ...` is correct. Starting from coworker as base avoids re-introducing the stale call.

References:
- Predecessor PRs: #17 (`refactor(radiation): drop cached PI_intensity, rename backRad -> solar`), #19 (`cont_intensity → PI_intensity` in SE_Container — not visible in either notebook), #20 (`tau_max abs(tau).max()`).
- Source-cell dumps (excluding outputs): `tmp/nb_diff/{hydrogen,Hspectra}.{cw,up}.txt`.

## Requirements

### Functional Requirements

1. `notebooks/hydrogen.ipynb` becomes byte-for-byte identical to `notebooks_local/hydrogen.ipynb`.
2. `notebooks/StatisticalEquilibrium/H_spectra.ipynb` consists of:
   - Source cells: coworker's source, with the 14 Category-A edits applied (2 `init_Radiation_` arg drops, 3 standalone `backRad → solar` accesses, 4 paired→single collapses, 1 standalone diagnostic line deletion, 4 print-statement renames inside f-strings).
   - Output cells: from coworker's notebook, untouched (rename is behavior-preserving, so outputs remain valid).
   - Cell metadata (`execution_count`, `metadata`, `id`): from coworker's notebook.
   - Notebook-level `metadata.kernelspec.display_name` becomes `spectra (3.13.5)` (was `spectra (3.13.11)` in upstream); `language_info.version` becomes `3.13.5`. Accepted: `kernelspec.name == "python3"` is unchanged, so the kernel still resolves locally; only the cosmetic display string downgrades.
   - Source-cell **comments** that mention `PI_intensity` are retained (coworker's lines 65–66 in `tmp/nb_diff/Hspectra.cw.txt`: `# it looks better to separate solar spectrum and PI_intensity ...` and `# wl incomplete information of PI_intensity ...`). After PR #17/#19, `PI_intensity` is the valid `SE_Container` field name; these comments became more accurate, not stale.
3. Upstream `notebooks/StatisticalEquilibrium/H_spectra.ipynb` `Ipeak` references (currently broken — `CloudModel_Container` has no `Ipeak` field) are gone, replaced by coworker's `Peak_intensity()` helper approach.
4. Cell counts: `hydrogen.ipynb` 69 cells (matches coworker); `H_spectra.ipynb` 68 cells (matches coworker; upstream had 67 because of the dropped `#help_(atom)` cell — net +0 after coworker's restructure).
5. Both notebooks parse as valid JSON.

### Non-Functional Requirements

- **Zero regression test impact**: notebook changes do not affect `tests/regression/`. Verify after by running the suite — expect 262/262 still green.
- **No source-code changes**: only notebook files modified. `src/`, `tests/`, `scripts/`, `data/` unchanged.
- **No new dependencies**: no new imports introduced.
- **Pre-commit hooks**: must pass (`ruff format`, `ruff check`, `pyright`, `pytest-regression`). Notebooks are not linted by ruff/pyright, but format check on any incidental .py files (none expected) must pass.

## Scope

### In Scope

- [ ] Replace `notebooks/hydrogen.ipynb` content with `notebooks_local/hydrogen.ipynb` content (whole-file replace).
- [ ] Use a workflow of: (a) backup upstream `H_spectra.ipynb` to `H_spectra.backup.ipynb`, (b) copy coworker `H_spectra.ipynb` over, (c) apply 14 Category-A source edits, (d) delete backup once verified.
- [ ] After both notebooks land, delete `notebooks_local/` (it was a temporary handoff folder, not tracked content). **Not** part of this task — leave for the user to confirm.

### Out of Scope (Boundaries)

> Items explicitly excluded from this task. Do NOT touch these areas.

- **`src/`, `tests/`, `scripts/`, `data/`:** no source code changes. Fix is notebook-only.
- **Other notebooks** in `notebooks/` and `notebooks/StatisticalEquilibrium/`: no migrations, no API rename sweeps. Already migrated by PR #17 commit `2a190a1`.
- **`docs/tasks/004-...`, `docs/tasks/005-...`:** historic audit; do not retroactively rewrite.
- **Output regeneration**: do not re-run the notebooks. Output cells are inherited from coworker's run; they reflect identical physics (PR #17/#19/#20 are pure renames / abs() fix with no numeric impact for non-inverted populations).
- **Category C content**: do not port upstream's Cmat title elaboration / `#help_(atom)` placeholder back into coworker.
- **`init_theoretical_hydrogen_atom_` 3-tuple call form**: never re-introduce; current function returns 2-tuple.

## Acceptance Criteria

All grep gates against `notebooks/StatisticalEquilibrium/H_spectra.ipynb` are run on **source cells only**, extracted via:

```bash
src() { jq -r '.cells[] | select(.cell_type=="code") | .source | join("")' "$1"; }
```

Cached output cells inherited from coworker may legitimately contain `backRad` / `PI_intensity` text (e.g., in `help_(...)` ndarray dumps and in `print()` literal strings rendered before the rename) — those don't fail the gate.

- [ ] `diff notebooks_local/hydrogen.ipynb notebooks/hydrogen.ipynb` is empty.
- [ ] `src notebooks/StatisticalEquilibrium/H_spectra.ipynb | grep -n "backRad"` returns zero matches.
- [ ] `src notebooks/StatisticalEquilibrium/H_spectra.ipynb | grep -nE "(solar_spec|ssp)\.PI_intensity"` returns zero matches (executable accesses gone). Comments mentioning `PI_intensity` are intentionally retained — see Functional Requirement #2 — so a bare `PI_intensity` grep would false-positive on those.
- [ ] `src notebooks/StatisticalEquilibrium/H_spectra.ipynb | grep -nE "init_Radiation_\((atmos|slab|wMesh)"` returns zero matches.
- [ ] `src notebooks/StatisticalEquilibrium/H_spectra.ipynb | grep -n "atom, wMesh, path_dict"` returns zero matches.
- [ ] `src notebooks/StatisticalEquilibrium/H_spectra.ipynb | grep -c "def Peak_intensity"` is ≥ 1 (Category B preserved).
- [ ] `src notebooks/StatisticalEquilibrium/H_spectra.ipynb | grep -c "tau profile can be negative"` is ≥ 1 (Category B preserved).
- [ ] Both notebooks load via `python3 -c "import json; json.load(open(p))"`.
- [ ] `notebooks/StatisticalEquilibrium/H_spectra.backup.ipynb` is gone after step (d).
- [ ] `pytest tests/regression/` is 262/262 green.

## Dependencies

| Dependency | Owner | Status | Notes |
|------------|-------|--------|-------|
| PR #17 merged | kouui | ✅ Done | `solar` field, `init_Radiation_(path)` signature |
| PR #19 merged | kouui | ✅ Done | `PI_intensity` field on `SE_Container` (irrelevant to these two notebooks — they don't reference SE_Container's PI_intensity) |
| PR #20 merged | kouui | ✅ Done | `tau_max` abs() fix; coworker's notebook independently arrives at the same insight |
| Coworker notebooks delivered | coworker | ✅ Done | At `notebooks_local/` |

## Risks & Open Questions

- [ ] **Output-cell cosmetic staleness on Category-A renamed sites**: coworker's cached outputs in renamed cells include `backRad`-flavored text (e.g., `f"... ssp.backRad[1,:] ..."` print results, `help_(solar_spec)` ndarray field labels). After the rename, the source reads `solar` while the cached output text still reads `backRad`. This is a display-only artifact; outputs were generated under physics identical to the new API (PR #17 is a pure rename, no numeric impact). Accepted per user spec ("output cells inherit from coworker"); not used as a verification oracle.
- [ ] **Cell ID / metadata drift**: coworker may have different `metadata.id` keys than upstream. Acceptable; we replace whole cells from coworker. The kernelspec / language_info should still parse.
- [ ] **Standalone `solar_spec.PI_intensity[0,:]` diagnostic line in cell #7**: gets removed entirely. The cell retains `help_(solar_spec); help_(atmos)` after deletion. Output cell of cell #7 may show one fewer ndarray dump than coworker's cached output — acceptable; not regenerating.
- [ ] **`init_Radiation_` no longer needs `wMesh`**: function uses `path` only. The `wMesh` variable in the affected cells remains used downstream, so no dead-variable issue.

## References

- PR #17: https://github.com/kouui/spectra-devcontainer/pull/17
- PR #19: https://github.com/kouui/spectra-devcontainer/pull/19
- PR #20 (`fix(CloudModel)`): https://github.com/kouui/spectra-devcontainer/pull/20
- Source-cell dumps for diff inspection: `tmp/nb_diff/{hydrogen,Hspectra}.{cw,up}.txt`
- Coworker handoff folder: `notebooks_local/`
