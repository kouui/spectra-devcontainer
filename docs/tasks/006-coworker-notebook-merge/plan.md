# Plan: Merge coworker notebook updates into upstream notebooks

> **Task:** [task.md](./task.md)
> **Owner:** kouui
> **Created:** 2026-05-09
> **Target Completion:** 2026-05-09

---

## 0. Context

> **Objective:** Replace upstream `notebooks/hydrogen.ipynb` and `notebooks/StatisticalEquilibrium/H_spectra.ipynb` with coworker's versions, applying the 14 Category-A API edits to the latter so it matches PR #17 (`backRad → solar`, drop `Radiation.PI_intensity`, `init_Radiation_()` signature).
> **Full spec:** [task.md](./task.md)

---

## 1. Overall Architecture

### Strategy: copy-and-patch

```
notebooks_local/hydrogen.ipynb                         ──cp──▶  notebooks/hydrogen.ipynb
                                                                  (whole-file replace; no edits — no API references)

notebooks_local/StatisticalEquilibrium/H_spectra.ipynb ──┐
                                                         │
notebooks/StatisticalEquilibrium/H_spectra.ipynb ──mv──▶ H_spectra.backup.ipynb (safety net)
                                                         │
                                                         ▼
                                                       cp coworker → upstream path
                                                         │
                                                         ▼
                                                       apply 14 Category-A edits
                                                         │
                                                         ▼
                                                       verify (json, grep, regression)
                                                         │
                                                         ▼
                                                       rm H_spectra.backup.ipynb
```

This avoids 3-way merge entirely. Coworker's source is the base (already contains Category B additions and the correct 2-tuple `init_theoretical_hydrogen_atom_` call). We layer Category A renames on top. Category C is intentionally left as coworker (per user direction).

### Why not 3-way merge

- Coworker's source is *strictly more recent* than the fork base in source-cell terms (cells 32–34 are net-new, `Peak_intensity` helper is new, `Ne=1.E13` and `wrange=[3500,11000]` are deliberate physics/visualization choices the coworker iterated to).
- Upstream's value-add is just (i) PR #17 API renames and (ii) a Cmat title elaboration. (i) is a mechanical 14-edit pass; (ii) is Category C and explicitly dropped.
- 3-way merge requires the fork base, which isn't available (coworker's local working tree predates a known commit). Copy-and-patch sidesteps this.

### Data flow per file

| File | Source pipeline | Output cells |
|------|-----------------|--------------|
| `hydrogen.ipynb` | coworker → cp → upstream | from coworker (untouched) |
| `H_spectra.ipynb` | coworker → cp → 14 Category-A edits → upstream | from coworker (untouched) |

### Why output cells stay untouched

- PR #17 rename `backRad → solar`: physically equivalent; numeric outputs identical.
- PR #19: not referenced by either notebook.
- PR #20 `tau_max abs()`: H I has no population inversion → no numeric difference.
- Therefore coworker's cached outputs match the post-patch source semantics. Re-running is unnecessary; user explicitly requested coworker outputs.

---

## 2. Implementation Phases

### Phase 1: `hydrogen.ipynb` whole-file replace

> **Goal:** Replace upstream `hydrogen.ipynb` with coworker's version verbatim.
> **Estimated Effort:** < 5 min

- [ ] Step 1.1 — `cp notebooks_local/hydrogen.ipynb notebooks/hydrogen.ipynb`
- [ ] Step 1.2 — `diff -q notebooks_local/hydrogen.ipynb notebooks/hydrogen.ipynb` (must be empty / "files are identical")
- [ ] Step 1.3 — JSON validity: `python3 -c "import json; json.load(open('notebooks/hydrogen.ipynb'))"`
- [ ] Step 1.4 — `git diff --stat notebooks/hydrogen.ipynb` to confirm the file shows non-trivial diff against HEAD (sanity that we replaced, not no-op'd).

**Phase 1 Exit Criteria:**
- [ ] `diff notebooks_local/hydrogen.ipynb notebooks/hydrogen.ipynb` empty.
- [ ] JSON parses.

---

### Phase 2: `H_spectra.ipynb` copy-and-patch

> **Goal:** Replace upstream `H_spectra.ipynb` with coworker's, then apply 14 Category-A source edits.
> **Estimated Effort:** ~30 min
> **Depends on:** none

- [ ] Step 2.1 — Backup:
  ```bash
  mv notebooks/StatisticalEquilibrium/H_spectra.ipynb notebooks/StatisticalEquilibrium/H_spectra.backup.ipynb
  ```

- [ ] Step 2.2 — Copy coworker over:
  ```bash
  cp notebooks_local/StatisticalEquilibrium/H_spectra.ipynb notebooks/StatisticalEquilibrium/H_spectra.ipynb
  ```

- [ ] Step 2.3 — Apply 14 Category-A edits in source cells. All edits are in `notebooks/StatisticalEquilibrium/H_spectra.ipynb`. Since notebooks are JSON with cell-level granularity, prefer per-cell `Edit` operations (use `jq` or direct file edits via the `Edit` tool, with sufficient context to disambiguate). The 14 edits, grouped by cell:

  Cell containing `init_Radiation_(atmos, wMesh)` (likely cell #4):
  - 2.3.1: `Radiation.init_Radiation_(atmos, wMesh)` → `Radiation.init_Radiation_()` (1 site)
  - 2.3.2: `solar_spec.backRad[0,:]` → `solar_spec.solar[0,:]` (1 site)
  - 2.3.3: `solar_spec.backRad[1,:].copy()` → `solar_spec.solar[1,:].copy()` (1 site)
  - 2.3.4: paired `solar_spec.backRad[1,:] *= 0.5\nsolar_spec.PI_intensity[:,:] *= 0.5` → single `solar_spec.solar[1,:] *= 0.5` (1 site, 2 lines → 1 line)

  Cell containing `def solspec(...)` helper:
  - 2.3.5: paired `ssp.backRad[1,:] *= Jfactor\nssp.PI_intensity[:,:] *= Jfactor` → single `ssp.solar[1,:] *= Jfactor` (1 site)

  Cell containing standalone `help_(solar_spec); help_(atmos); solar_spec.PI_intensity[0,:]`:
  - 2.3.6: delete the `solar_spec.PI_intensity[0,:]` line entirely (the surrounding `help_(solar_spec)` and `help_(atmos)` calls stay).

  Two later cells with the same paired pattern (likely the recombining/ionizing sweep cells around lines 330 and 391 in the dump):
  - 2.3.7: paired `ssp.backRad[1,:] *= Jfactor\nssp.PI_intensity[:,:] *= Jfactor` → single (site #2)
  - 2.3.8: paired `ssp.backRad[1,:] *= Jfactor\nssp.PI_intensity[:,:] *= Jfactor` → single (site #3)

  Slab-recompute cell (line 596–600 in the dump):
  - 2.3.9: `Radiation.init_Radiation_(slab, wMesh)` → `Radiation.init_Radiation_()` (1 site)
  - 2.3.10: `ssp.backRad[0,:]` → `ssp.solar[0,:]` (1 site)

  Diagnostic-print cell (lines 1113–1116 in the dump):
  - 2.3.11: `print(f"\nBackground radiation intensity (ssp.backRad[1,:]):")` → `... (ssp.solar[1,:]):"` (rename inside the literal string too)
  - 2.3.12: `print(f"ssp.backRad[1,:] min: {ssp.backRad[1,:].min()}")` → `... ssp.solar[1,:] ... ssp.solar[1,:].min()` (2 occurrences in this line; rename both)
  - 2.3.13: same for `max`.
  - 2.3.14: same for `mean`.

  > Edits 2.3.11–2.3.14 each rename `backRad` → `solar` in **two places per line**: the f-string literal and the actual attribute access. Preserve everything else (formatting, units, indentation, backslash escaping for newlines in JSON).

- [ ] Step 2.4 — Verify with grep (source cells only, to avoid false positives on coworker's cached outputs):
  ```bash
  src() { jq -r '.cells[] | select(.cell_type == "code") | .source | join("")' "$1"; }
  src notebooks/StatisticalEquilibrium/H_spectra.ipynb \
    | grep -nE "backRad|init_Radiation_\((atmos|slab|wMesh)|atom, wMesh, path_dict|(solar_spec|ssp)\.PI_intensity"
  # Expected: empty.
  ```
  Notes on the regex:
  - `(solar_spec|ssp)\.PI_intensity` catches **executable** `PI_intensity` accesses without false-positiving on coworker's prescient comments at lines 65–66 of `tmp/nb_diff/Hspectra.cw.txt` that read `# ... PI_intensity ...`.
  - `init_Radiation_\((atmos|slab|wMesh)` catches any leftover positional arg variant.

- [ ] Step 2.5 — Verify Category B preservation (run on source cells, same `src` helper):
  ```bash
  src notebooks/StatisticalEquilibrium/H_spectra.ipynb | grep -c "def Peak_intensity"
  # Expected: ≥ 1.
  src notebooks/StatisticalEquilibrium/H_spectra.ipynb | grep -c "tau profile can be negative"
  # Expected: ≥ 1.
  ```

- [ ] Step 2.6 — JSON validity:
  ```bash
  python3 -c "import json; nb = json.load(open('notebooks/StatisticalEquilibrium/H_spectra.ipynb')); print('cells:', len(nb['cells']))"
  # Expected: cells: 68.
  ```

- [ ] Step 2.7 — Regression sanity:
  ```bash
  uv run pytest tests/regression/ -q
  # Expected: 262 passed.
  ```

- [ ] Step 2.8 — Remove backup:
  ```bash
  rm notebooks/StatisticalEquilibrium/H_spectra.backup.ipynb
  ```

**Phase 2 Exit Criteria:**
- [ ] All 14 Category-A edits applied; grep checks pass.
- [ ] Category B markers (`Peak_intensity`, "tau profile can be negative") still present.
- [ ] JSON parses, cell count is 68.
- [ ] `tests/regression/` 262/262 green.
- [ ] `H_spectra.backup.ipynb` removed.

---

## 3. Boundaries — Do NOT Touch

> These files, modules, and APIs are explicitly out of scope.

| Area | Path / Identifier | Reason |
|------|-------------------|--------|
| Source code | `src/spectra/` | Notebook-only task; no API or behavior changes. |
| Tests | `tests/` | No test modifications; verifying regression suite stays 262/262. |
| Other notebooks | `notebooks/StatisticalEquilibrium/{Hydrogen_atom,He_plasma,CaII_flush,HHeCa,...}.ipynb`, `notebooks/Hydrogen_atom.ipynb`, etc. | Already migrated by PR #17 commit `2a190a1`. Out of scope. |
| Historic task docs | `docs/tasks/{001..005}-*` | Audit trail. Do not retro-edit. |
| Coworker handoff | `notebooks_local/` | Source of truth for this merge. Do not modify; user decides post-task whether to delete. |
| Output regeneration | All notebook output cells | Inherit from coworker; do not re-run. |
| Category C content | Cmat title elaboration, `#help_(atom)` placeholder | Coworker's terser version kept (user direction). |
| `init_theoretical_hydrogen_atom_` 3-tuple call | Anywhere in any notebook | Function returns 2-tuple per `Atom.py:262`; coworker's 2-tuple call form is correct. |

**Rule of thumb:** If a change requires modifying anything listed above, stop and revisit the plan.

---

## 4. Test Coverage

### Testing Strategy

| Level | Scope | Tool |
|-------|-------|------|
| Regression | `pytest tests/regression/` must stay 262/262 — notebook changes can't affect this, but it's the safety net | `pytest` |
| Static (JSON) | Both notebooks must parse | `json.load` |
| Grep gate | No `backRad`, no stale `init_Radiation_(atmos|slab)`, no `atom, wMesh, path_dict` in source cells | `jq` + `grep` |
| Pre-commit | Hooks pass on commit | git pre-commit |

### Required Test Cases

- [ ] `pytest tests/regression/` → 262/262 (pre-existing).
- [ ] `json.load` on both target notebooks succeeds.
- [ ] Grep gates on `H_spectra.ipynb` source cells.

### Coverage Target

- All 14 Category-A edits land; all 3 Category-B markers preserved; all 3 Category-C choices = coworker.

---

## 5. Key Decisions

### Decision 1: Copy-and-patch over 3-way merge

- **Context:** Need to combine coworker's substantive additions with upstream's PR-#17 API renames.
- **Options Considered:**
  1. Copy coworker over upstream, patch the 14 API sites — simple, deterministic.
  2. Take upstream as base, port coworker's added cells in cell-by-cell — accurate but requires cherry-picking 5+ cells across mid-notebook insertions; brittle.
  3. Use `nbdime` 3-way merge — needs known fork base; we don't have it.
- **Decision:** Option 1.
- **Rationale:** Coworker's notebook is strictly more recent on source-cell axis (Category B is net-add); upstream's value-add is mechanical (Category A is rename-only). Option 1 minimizes manual judgment surface to 14 well-defined site edits.
- **Consequences:** Upstream's Category C (Cmat title elaboration, `#help_(atom)`) is dropped. User explicitly approved.

### Decision 2: Backup-then-replace workflow

- **Context:** User requested an explicit backup step before overwriting.
- **Options Considered:**
  1. Direct overwrite (rely on `git` for rollback).
  2. `mv` to `.backup.ipynb`, then `cp`, then `rm` after verify.
- **Decision:** Option 2.
- **Rationale:** Safety net during patching — if a mid-step error corrupts the working copy, the backup is one `mv` away. `git` is also available, but the backup is local and removes any chance of confusion if working tree state is mixed.
- **Consequences:** One extra file in working tree during the operation; cleaned up at the end.

### Decision 3: Inherit coworker's output cells (no re-run)

- **Context:** User wants the final notebook outputs to match coworker's runs.
- **Options Considered:**
  1. Re-run all cells (regenerates outputs from migrated source).
  2. Keep coworker's cached outputs verbatim.
- **Decision:** Option 2.
- **Rationale:** PR #17 (rename) and PR #19 (rename) are numerically no-ops; PR #20 (`abs(tau).max()`) is a no-op on H I because the H I reference has no population inversion in the cached cell results. Coworker's outputs are valid representations of identical physics. (Coworker's notebook does add a Category-B `Peak_intensity` helper that uses `np.abs(l.tau_1D).argmax()` — same semantic intent as PR #20 in the source layer; the cached outputs of those new cells are coherent with the post-merge source.)
- **Consequences:** Cell outputs may show stale `backRad` / `solar_spec.PI_intensity` text in `help_(...)` dumps and cached `print()` output. Cosmetic drift only; outputs are not used as a verification oracle. The grep gates in §4 explicitly filter source cells to avoid false-positiving on this drift.

### Decision 4: Category C scope is one specific cell, not a broad sweep

- **Context:** Subagent + codex review flagged that "keep coworker for Category C" was described too broadly in the initial draft. Coworker's notebook actually contains the `'Cmat (theoretical)'` title in **three** cells: line 187 (terse, the only divergence from upstream), and lines 909 + 953 (already elaborated `+r"$\overline{J}=$"+f"{Jfactor:3.1F}"`, matching upstream). Coworker also has its **own** `#help_(atom)` early in the notebook (line 36 of the dump); upstream's *additional* `#help_(atom)` lives near cell #30 (line 625 area of the upstream dump).
- **Options Considered:**
  1. Coarse rule: "keep coworker style for everything Category-C-like" — risks deleting coworker's elaborated Cmat titles or coworker's own `#help_(atom)`.
  2. Anchor rule: name the exact cells. Only the line-187 Cmat title stays terse; only upstream's *second* `#help_(atom)` (which never lands because we copy from coworker) is excluded.
- **Decision:** Option 2.
- **Rationale:** The implementation (copy-and-patch) automatically realizes Option 2: coworker's source already has the correct mix (terse first Cmat, elaborated others, single `#help_(atom)`), so no edits are needed for Category C. Calling this out explicitly avoids a follow-up reviewer / future maintainer assuming the cells are inconsistent.
- **Consequences:** No edits in step 2.3 are added for Category C; this decision is documentation-only.

### Decision 5: Delete `solar_spec.PI_intensity[0,:]` standalone diagnostic line entirely

- **Context:** Cell #7 has three diagnostic lines: `help_(solar_spec); help_(atmos); solar_spec.PI_intensity[0,:]`. After PR #17, `Radiation.PI_intensity` is gone — the third line raises `AttributeError` if executed.
- **Options Considered:**
  1. Delete the line.
  2. Replace with an SE-result-side equivalent (would require running SE, which the cell doesn't do yet).
  3. Leave it as-is and accept a runtime error.
- **Decision:** Option 1.
- **Rationale:** Matches what upstream's PR #17 migration `2a190a1` did to the equivalent cell elsewhere (deleted the access). The other two `help_(...)` calls retain the diagnostic intent.
- **Consequences:** Cached output of cell #7 from coworker shows one fewer ndarray dump than fresh execution; tolerated.

---

## 6. Precautions

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stray `backRad` reference missed by grep gate | Low | Medium | Phase-2 grep gate runs against jq-extracted code-cell sources only; precise regex catches all occurrences. |
| Notebook JSON corruption during multi-line `Edit` paired-collapse | Low | High | Use `Edit` tool with full-line context; verify JSON parses immediately after each cell edit. |
| Cell ID collision with already-tracked git history | Very low | Low | Coworker's cells have their own IDs; git treats this as a content replacement. No semantic issue. |
| Pre-commit `pytest-regression` fails due to incidental tree state | Very low | Medium | Notebooks aren't loaded by tests; failure would point at unrelated source state. Triage before commit. |
| Cell-numbering drift between this plan's references (cells "around 32/33/34") and the actual coworker file | Medium | Low | Plan references coworker's *line numbers* in `tmp/nb_diff/Hspectra.cw.txt` (deterministic) and the surrounding code text (unique anchors), not cell indices. Edit tool matches by surrounding context. |
| Kernelspec metadata downgrade `3.13.11 → 3.13.5` after copying coworker's notebook | Certain | Low | Documented and accepted: `kernelspec.name` is `python3` on both sides — the kernel still resolves to whatever the local environment provides; only the cosmetic `display_name` and `language_info.version` strings change. No runtime impact. Re-saving the notebook locally after open will refresh these fields against the active interpreter. |

### Rollback Plan

If post-merge issue:

1. `git checkout HEAD -- notebooks/hydrogen.ipynb notebooks/StatisticalEquilibrium/H_spectra.ipynb` to revert.
2. If issue is mid-task (before commit): `mv notebooks/StatisticalEquilibrium/H_spectra.backup.ipynb notebooks/StatisticalEquilibrium/H_spectra.ipynb` to restore from backup.
3. Re-evaluate which Category-A edit corrupted the file.

### Migration Notes

- **Backward compatibility:** N/A — notebooks are interactive scratchpads.
- **Feature flag:** N/A.
- **Migration script:** N/A; manual edit pass.

### Performance Considerations

- N/A — notebook content; no runtime impact.

### Security Considerations

- N/A.

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-05-09 | kouui | Initial draft after diff inspection of `tmp/nb_diff/{hydrogen,Hspectra}.{cw,up}.txt` and signature verification of `init_theoretical_hydrogen_atom_` |
| 2026-05-09 | kouui | Revisions per subagent + codex review: AC grep gates filter source cells via `jq` to avoid coworker-output false-positives; `PI_intensity` AC carved between executable accesses (forbidden) and prescient comments (kept); fixed Category-A count typo (4→2 init_Radiation drops); pinned Category C to specific anchor cells (Decision 4); explicit kernelspec downgrade risk; tightened output-staleness rationale |
