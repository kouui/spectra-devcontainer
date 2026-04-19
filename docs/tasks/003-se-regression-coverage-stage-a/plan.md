# Plan: SE regression coverage expansion — Stage A

> **Task:** `docs/tasks/003-se-regression-coverage-stage-a/task.md`
> **Owner:** kouui
> **Created:** 2026-04-19
> **Target Completion:** 2026-04-20 (≤1 工作日)

---

## 0. Context

> **Objective:** 为 `SELib.py` 三个 public entry (`cal_SE_with_Nh_Te_`, `cal_SE_with_Ne_Te_`, `cal_SE_with_Pg_Te_`) × 三原子 (H, He, Ca_II) 的 9-cell 矩阵补齐 6 个缺失的 e2e case,所有新 reference 在 `a84128b` worktree 上生成并回落到 main,期望零 drift。
> **Full spec:** `docs/tasks/003-se-regression-coverage-stage-a/task.md`
>
> This plan describes *how* to achieve the above objective. Requirements and acceptance criteria live in the task doc.

---

## 1. Overall Architecture

### System Overview

```
 scripts/gen_se_reference.py (one-shot)
           │
           │  run on a84128b worktree
           ▼
 JSON fragment with 18 E2E.* keys
           │
           │  merge into main
           ▼
 tests/regression/reference_values.json (updated)
           │
           │  read by `ref` session fixture (existing)
           ▼
 tests/regression/test_reg_e2e_SE.py
           │  6 new test methods across 3 classes
           ▼
 ┌───────────────────────────┐
 │ Stage A safety net (SE)   │
 └───────────────────────────┘
```

### Key Components

| Component | Responsibility | New / Modified |
|-----------|---------------|----------------|
| `scripts/gen_se_reference.py` | One-shot: run 6 new cases, extract the asserted fields per case (2-4), update `reference_values.json` in place | New |
| `tests/regression/test_reg_e2e_SE.py` | Add 6 test methods across `TestHydrogenSE`, `TestHeliumSE`, `TestCaIISE` | Modified |
| `tests/regression/reference_values.json` | Add 18 new keys under `E2E.*` namespace | Modified |
| (worktree only) `/tmp/spectra-se-old/` | Throwaway checkout of `a84128b` for generation | Transient |

### Data Flow

**Generation (on a84128b worktree)**:
1. `git worktree add /tmp/spectra-se-old a84128b`
2. `cp scripts/gen_se_reference.py /tmp/spectra-se-old/scripts/`
3. Copy a staging copy of `reference_values.json` (so script has existing keys to preserve)
4. `cd /tmp/spectra-se-old && uv sync --extra dev`
5. `uv run --extra dev python scripts/gen_se_reference.py`
6. Script reads `reference_values.json`, adds 18 new keys, writes back (in-place, preserving the existing ad-hoc key order via `sort_keys=False`).
7. `cp /tmp/spectra-se-old/tests/regression/reference_values.json ./tests/regression/`
8. `git worktree remove --force /tmp/spectra-se-old`

**Verification (on main)**:
1. `uv run --extra dev pytest tests/regression/test_reg_e2e_SE.py -v`
2. Expected: all 9 tests pass.
3. If any fail: `git diff a84128b..HEAD -- src/spectra/Function/SEquil src/spectra/Atomic` + drift-audit.

---

## 2. Implementation Phases

### Phase 1: Test scaffolding (on main, tests will temporarily fail)

> **Goal:** Add 6 new test methods referencing keys that don't yet exist, so running pytest surfaces a clear `KeyError` rather than a soft skip. Prepares the regression surface for P3's reference merge.
> **Estimated Effort:** 0.25 day

- [x] Step 1.1 — Extend `TestHydrogenSE` in `tests/regression/test_reg_e2e_SE.py`:
  - New method `test_SE_with_Pg_Te(self, ref)`. Atmosphere: `Pg=1.8, Te=7.0e3, Vt=5.0e5, Vd=0`. Assert `n_SE`, `n_LTE`, `Ne`, `Ntotal`.
- [x] Step 1.2 — Extend `TestHeliumSE`:
  - New method `test_SE_with_Nh_Te(self, ref)`. Atmosphere: `Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vt=5.0e5, Vd=0`.
  - New method `test_SE_with_Pg_Te(self, ref)`. Atmosphere: `Pg=1.8, Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vt=5.0e5, Vd=0`. (Pg is placeholder for non-H per R2.)
- [x] Step 1.3 — Add new class `TestCaIISE`:
  - `_load_Ca_II()` helper (parallel to `TestHydrogenSE._load_H`).
  - `test_SE_with_Nh_Te(self, ref)`. Atmosphere: `Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vt=5.0e5, Vd=0`.
  - `test_SE_with_Ne_Te(self, ref)`. Atmosphere: `Nh=1.0e11, Ne=5.0e10, Te=7.0e3, Vt=5.0e5, Vd=0`.
  - `test_SE_with_Pg_Te(self, ref)`. Atmosphere: `Pg=1.8, Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vt=5.0e5, Vd=0`.
- [x] Step 1.4 — Run `uv run --extra dev pytest tests/regression/test_reg_e2e_SE.py -v --no-header 2>&1 | tail -n 40`. Expected: 3 pass + 6 fail with `KeyError: 'E2E.xxx'`. This validates wiring before generation.

**Phase 1 Exit Criteria:**
- [x] 6 new methods compile & import; existing 3 still pass.
- [x] Expected failures have identifiable `KeyError` messages (not structural errors).

---

### Phase 2: Generation script (on main)

> **Goal:** Author `scripts/gen_se_reference.py` that is idempotent, deterministic, and preserves the existing JSON format.
> **Estimated Effort:** 0.25 day
> **Depends on:** Phase 1

- [x] Step 2.1 — Create `scripts/gen_se_reference.py`:
  - Import `Atom`, `Atmosphere`, `Radiation`, `SELib`, `CFG` (same as AtomIO script).
  - Declare a `CASES` tuple of 6 records: `(atom_name, conf_rel, is_hydrogen, entry, atmos_kwargs)`.
  - For each case: load atom → build atmos → build radiation → call entry → extract 4 fields.
  - Load existing `reference_values.json` → update keys → write back with `json.dump(..., indent=2, sort_keys=False)` to preserve the file's existing ad-hoc key order.
  - Print a summary line per case: `"generated E2E.<atom>_SE_<entry>: n_SE(sum)=..., Ne=..., Ntotal=..."` for sanity eyeballing.
- [x] Step 2.2 — Dry-run on main: `uv run --extra dev python scripts/gen_se_reference.py`. Observe no crash, JSON file re-sorts but new keys appear.
- [x] Step 2.3 — `git diff tests/regression/reference_values.json` to confirm only the 18 new keys appear (no spurious rewrites from sort reshuffling).
- [x] Step 2.4 — Revert the dry-run changes: `git checkout -- tests/regression/reference_values.json`. (Keep the script committed-to-be; the real values come from a84128b in Phase 3.)
- [x] Step 2.5 — Run `pytest tests/regression/test_reg_e2e_SE.py -v`. Still expecting 6 KeyErrors.

**Phase 2 Exit Criteria:**
- [x] Script runs on main without error.
- [x] Dry-run diff shows only the 18 target keys (no whitespace / sort reshuffling).
- [x] `reference_values.json` restored to pre-dry-run state.

---

### Phase 3: a84128b worktree generation

> **Goal:** Produce the authoritative reference values on the behavior-locked baseline.
> **Estimated Effort:** 0.25 day
> **Depends on:** Phase 2

- [x] Step 3.1 — `git worktree add /tmp/spectra-se-old a84128b`
- [x] Step 3.2 — `cp scripts/gen_se_reference.py /tmp/spectra-se-old/scripts/`
  - (Also cp `tests/regression/reference_values.json` from main into the worktree so the script has the current key set to merge into — the worktree's own `reference_values.json` would be older.)
- [x] Step 3.3 — `cd /tmp/spectra-se-old && uv sync --extra dev` (expected: 30-60s)
- [x] Step 3.4 — `uv run --extra dev python scripts/gen_se_reference.py > /tmp/gen_se_summary.log 2>&1; tail -n 20 /tmp/gen_se_summary.log`
- [x] Step 3.5 — Eyeball the summary: confirm `n_SE(sum)` close to 1 (LTE sanity) for each case.
- [x] Step 3.6 — Copy the updated `reference_values.json` back to main: `cp /tmp/spectra-se-old/tests/regression/reference_values.json tests/regression/`
- [x] Step 3.7 — `git worktree remove --force /tmp/spectra-se-old`
- [x] Step 3.8 — `git diff tests/regression/reference_values.json` — confirm only 18 new keys added, no other churn.

**Phase 3 Exit Criteria:**
- [x] a84128b script ran cleanly.
- [x] 18 new keys landed in main's `reference_values.json`.
- [x] No other diff in the file.

---

### Phase 4: Verify zero drift on main

> **Goal:** Prove current `main`'s SE pipeline matches `a84128b`'s output bit-for-rtol.
> **Estimated Effort:** 0.125 day
> **Depends on:** Phase 3

- [x] Step 4.1 — `uv run --extra dev pytest tests/regression/test_reg_e2e_SE.py -v 2>&1 | tail -n 30`
  - **Expected:** 9 pass / 0 fail.
- [x] Step 4.2 — If any fail with tolerance issue: open `docs/tasks/003-se-regression-coverage-stage-a/drift-audit.md`, run `git log --oneline a84128b..HEAD -- src/spectra/Function/SEquil src/spectra/Atomic src/spectra/Struct/Atmosphere.py`, blame the diff, decide accept/revert per commit.
- [x] Step 4.3 — On pass: run full regression suite `uv run --extra dev pytest tests/regression/ -q 2>&1 | tail -n 20`. Expected: no regressions in the 247 pre-existing cases.

**Phase 4 Exit Criteria:**
- [x] `test_reg_e2e_SE.py` green (9/9).
- [x] Full regression green (no impact on other suites). 261 passed.
- [x] Drift-audit.md produced only if drift observed; otherwise not created. Zero drift confirmed.

---

### Phase 5: Polish & commit

> **Goal:** Clean up, update docs, commit.
> **Estimated Effort:** 0.125 day
> **Depends on:** Phase 4

- [x] Step 5.1 — `uv run --extra dev pre-commit run --all-files 2>&1 | tail -n 20`. Fix lint/format if any.
- [x] Step 5.2 — Update `task.md` status → In Progress → Done as appropriate.
- [x] Step 5.3 — Tick checkboxes in `plan.md` as steps complete (incremental).
- [x] Step 5.4 — Commit:
  - Touched files: `tests/regression/test_reg_e2e_SE.py`, `tests/regression/reference_values.json`, `scripts/gen_se_reference.py`, `docs/tasks/003-se-regression-coverage-stage-a/{task.md,plan.md}`
  - Message: `test: add e2e regression for SELib main entries (Stage A)` + body summarizing 6 new cases + a84128b audit.
- [x] Step 5.5 — Ask user whether to open a PR (default no).

**Phase 5 Exit Criteria:**
- [x] Single commit on feature branch (e.g., `feat/se-regression-stage-a`) — delivered as 5 commits (docs + P2 script + P2 review-fix + P3 landing + P3 review-fix) per review cycle requirements.
- [x] task.md status = Done.
- [x] pre-commit green.

---

## 3. Boundaries — Do NOT Touch

> These files, modules, and APIs are explicitly out of scope. Any changes to them require a separate task and review.

| Area | Path / Identifier | Reason |
|------|-------------------|--------|
| SE kernel | `src/spectra/Function/SEquil/SELib.py` | Behavior locked; only read. Even the suspected bug at line 117 (`atmos.Nh = Ng` outside while) is NOT fixed — if a84128b matches main, it is current behavior, not a regression |
| Active print | `src/spectra/Function/SEquil/SELib.py:98` | In `cal_SE_with_Pg_Te_single_Atom_`, which is out-of-scope. Our targets (`cal_SE_with_Pg_Te_` at line 122) have only commented-out prints. No cleanup this task |
| SE atomic primitives | `src/spectra/Atomic/SEsolver.py`, `LTELib.py`, `Collision.py`, `PhotoIonize.py`, `BasicP.py`, `Hydrogen.py` | Read-only dependencies |
| Icp module | `src/spectra/Function/Icp/` | Excluded per user (PR #10 context) |
| Icp-only entry | `cal_SE_with_Pg_Te_Ne_single_Atom_` | Excluded |
| 1-D atmosphere | `AtmosphereC1D`, `init_VAL_` | Out of 0-D e2e scope |
| CloudModel | `test_reg_e2e_CloudModel.py`, `SlabModel`, `CloudModel` | Covered by separate e2e |
| Existing SE e2e assertions | `test_reg_e2e_SE.py` lines 15-45 (the 3 existing methods) | No retro-add `Ntotal`; keep diff minimal |
| AtomIO serde | `tests/regression/_atom_serde.py` | Not used for SE; only AtomIO load assertions |
| Stage B / C | Te sweep, private helper unit tests | Deferred to follow-up task |
| Pyright suppressions | `pyproject.toml` | Already cleaned in PR #10 |
| Notebook SE usage | `notebooks/*.ipynb` | Not run under pytest |

**Rule of thumb:** If your change requires modifying anything listed above, stop and revisit the plan.

---

## 4. Test Coverage

### Testing Strategy

| Level | Scope | Tool / Framework |
|-------|-------|------------------|
| Regression (E2E) | 3 entries × 3 atoms = 9 SE pipeline cases | pytest + JSON reference |
| Smoke | Full regression suite runs clean after patch | pytest |
| Drift audit | a84128b vs main bit-rtol parity on new cases | one-shot script on worktree |

### Required Test Cases

#### New Regression Cases (6 total)

Each case asserts only the fields the entry actually mutates for that atom
type (per Decision 4, post-P1-review). All assertions use `rtol=1e-8`.

- [x] `TestHydrogenSE.test_SE_with_Pg_Te` — H via `cal_SE_with_Pg_Te_`, `Pg=1.8, Te=7000, Vt=5e5, Vd=0`. Covers the self-consistent Ne2Nh loop (hydrogen branch). Asserts 4 fields: `n_SE`, `n_LTE`, `atmos.Ne`, `Ntotal`.
- [x] `TestHeliumSE.test_SE_with_Nh_Te` — He via `cal_SE_with_Nh_Te_`, `Nh=1e12, Ne=1e11`. Covers non-H branch of Nh_Te entry (no iteration). Asserts 3 fields: `n_SE`, `n_LTE`, `Ntotal`.
- [x] `TestHeliumSE.test_SE_with_Pg_Te` — He via `cal_SE_with_Pg_Te_`, `Nh=1e12, Ne=1e11` (Pg unused for non-H). Asserts 3 fields.
- [x] `TestCaIISE.test_SE_with_Nh_Te` — Ca_II via `cal_SE_with_Nh_Te_`, `Nh=1e12, Ne=1e11`. Covers EXPERIMENT PI interpolation branch. Asserts 3 fields.
- [x] `TestCaIISE.test_SE_with_Ne_Te` — Ca_II via `cal_SE_with_Ne_Te_`, `Nh=1e11, Ne=5e10`. Matches existing H/He Ne_Te convention. Asserts 2 fields: `n_SE`, `n_LTE`.
- [x] `TestCaIISE.test_SE_with_Pg_Te` — Ca_II via `cal_SE_with_Pg_Te_`, `Nh=1e12, Ne=1e11`. Asserts 3 fields.

Total new reference keys: **18** (4 + 3 + 3 + 3 + 2 + 3).

#### Edge Cases & Error Handling

- [x] `n_SE.sum()` approximately 1 (LTE sanity) — enforced by a hard gate in the generation script (P2-review fix, commit 8dce1c7); all 6 cases produced 1.000000.
- [x] Pg_Te non-H path: `Pg` is effectively ignored; `atmos.Nh`/`Ne` provided pre-call. Documented as R2 in task.md.
- [x] No mutation leak between tests: each test creates fresh `Atmosphere0D`; atom/wMesh are re-loaded per class method. Verified during P1 review.

### Coverage Target

- This task: expand SE e2e from 3 → 9 cases; covers all 3 public entries × 3 canonical atoms.
- Not a coverage-percentage push; drift-prevention net.

---

## 5. Key Decisions

### Decision 1: Stage A only, defer Stage B/C

- **Context:** Handoff proposed A+B+C (up to 27+ cases). Stage B needs Te parameter sweep; Stage C needs private helper unit tests.
- **Options Considered:**
  1. All stages in one task — large scope, risk of overrun.
  2. Stage A only — minimal scope, highest ROI.
- **Decision:** Stage A only.
- **Rationale:** Per user confirmation. A delivers the most coverage per hour (all 3 entries × 3 atoms). B and C are refinements.
- **Consequences:** B and C spin off as `docs/tasks/00X-*-stage-b/` and `-stage-c/` later.

### Decision 2: class-per-atom file organization, no parametrize in Stage A

- **Context:** Choice between class-per-atom vs `pytest.parametrize` matrix.
- **Options Considered:**
  1. Keep class-per-atom, append methods — readable, small diff.
  2. Rewrite as `@pytest.mark.parametrize` matrix — compact but larger diff.
- **Decision:** Option 1.
- **Rationale:** 9 tests still fit in ~150-line file; parametrize shines when Stage B's ×3 Te sweep (27 cases) arrives.
- **Consequences:** Stage B will refactor to parametrize (or split into 3 files if file exceeds ~300 lines).

### Decision 3: New reference keys go into `reference_values.json`, not a new file

- **Context:** 18 new float/array values, ~tens of KB total.
- **Options Considered:**
  1. Merge into `reference_values.json`.
  2. New `se_reference_values.json` + new `ref_se` fixture.
- **Decision:** Option 1.
- **Rationale:** Consistent with existing `E2E.*` keys (already in `reference_values.json`). Size negligible.
- **Consequences:** `ref` fixture keeps serving SE + CloudModel + unit test refs. No fixture churn.

### Decision 4: Assert only fields actually mutated by the SE entry (revised post-review)

- **Context:** Initial draft committed to 4 fields for all new cases. P1 bug review (subagent + codex) flagged that `atmos.Ne` is an identity assertion for non-H Nh_Te and Pg_Te entries (the function does not mutate it for non-hydrogen), so the assertion carries no regression signal.
- **Options Considered:**
  1. Keep 4 fields uniformly — cosmetic symmetry but weak identity checks bloat reference JSON and mislead readers.
  2. Swap `atmos.Ne` for `atmos.Nh` on non-H Ne_Te (where `atmos.Nh` IS mutated to `2*atmos.Ne`) — still requires crafting non-coincidental inputs.
  3. Assert only fields the entry actually mutates for that atom type.
- **Decision:** Option 3.
- **Rationale:** A regression test's value is its ability to catch behavioral change. Identity assertions don't. Keep surface minimal and honest.
- **Consequences:** Per-case assertion depth varies (2, 3, or 4 fields). 18 new reference keys instead of 24. Existing 3 cases unchanged (minimal diff per the original intent).
- **Per-case breakdown:**
  - H × Pg_Te: 4 fields (n_SE, n_LTE, Ne, Ntotal) — Ne is iterated
  - He × {Nh_Te, Pg_Te}, Ca_II × {Nh_Te, Pg_Te}: 3 fields (n_SE, n_LTE, Ntotal) — Ne is input
  - Ca_II × Ne_Te: 2 fields (n_SE, n_LTE) — matches existing H/He Ne_Te convention

### Decision 5: `rtol=1e-8` — match existing style

- **Context:** Handoff (D2) suggested trying 1e-8 first, relax only if Pg_Te fails.
- **Options Considered:**
  1. `rtol=1e-8` everywhere — matches existing style.
  2. `rtol=1e-6` for Pg_Te (because of 1% convergence).
- **Decision:** Option 1 first.
- **Rationale:** Algorithm is deterministic per commit; 1e-8 should hold on a84128b vs a84128b. rtol is about test-vs-reference, not about algorithm accuracy.
- **Consequences:** If Pg_Te drifts at 1e-8 between a84128b and main, that itself is a useful signal — investigate before relaxing.

### Decision 6: No `capsys` — target function has no active prints

- **Context:** Handoff assumed `cal_SE_with_Pg_Te_` prints, recommended `capsys`.
- **Verification:** `grep -n "^[^#]*print(" src/spectra/Function/SEquil/SELib.py` shows only line 98 (in `cal_SE_with_Pg_Te_single_Atom_`, out of scope). Line 143's print in `cal_SE_with_Pg_Te_` is commented.
- **Decision:** No `capsys` needed.
- **Rationale:** Keeps test code minimal. If `cal_SE_` or a deeper helper prints, revisit defensively in P1.
- **Consequences:** If P1 run shows unexpected print noise, add `capsys.disabled()` or `capsys.readouterr()` locally.

---

## 6. Precautions

> Things that can go wrong and how to guard against them.

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Numeric drift between a84128b and main on SE path | Low | High (blocks task) | `git log --oneline a84128b..HEAD -- src/spectra/Function/SEquil src/spectra/Atomic src/spectra/Struct/Atmosphere.py src/spectra/Struct/Radiation.py`; blame per commit; open drift-audit.md |
| `reference_values.json` whitespace churn from re-sort | Med | Low (review noise) | Generation script uses `indent=2 sort_keys=False` to preserve the file's historic ad-hoc key order; verify via `git diff` that only 18 keys change (resolved during P2). |
| `uv sync` fails on a84128b worktree | Low | Med (blocks P3) | Task 002 verified it works; if fails, fall back to generating on main and note it as "a84128b parity not audited" in task.md — but this weakens the guarantee |
| Pg_Te H path produces different values than expected due to SELib.py:117 bug-like structure | Med | Low | Test reflects current behavior; a84128b audit catches regression; bug fix (if any) is a separate task |
| Ca_II EXPERIMENT PI interpolation produces NaN | Low | High | Script summary prints `n_SE(sum)`; NaN shows as `nan` → investigate before landing |
| Pytest auto-captures stdout — hides `print` noise | Low | Low | Run with `-s` during P1 dry-run to verify (then remove -s for final run) |

### Rollback Plan

If something goes wrong post-commit:

1. `git log --oneline` find this task's commit hash.
2. `git revert <hash>` — restores `test_reg_e2e_SE.py`, `reference_values.json`, `scripts/gen_se_reference.py` to pre-task state.
3. No side effects on production code (task touches only tests and scripts).

### Migration Notes

- No data migration. Tests and reference JSON are the only deliverables.
- Backward compatibility: old test keys unchanged; new keys are additive. No existing test breaks.

### Performance Considerations

- Test runtime: 6 new e2e cases × ~1-3s each ≈ 10-20s added to `test_reg_e2e_SE.py`. Full regression still < 2 min.
- Generation script: runs 6 cases once, ~30s total on a84128b worktree.

### Security Considerations

- [x] No secrets. JSON file is plain numerics.
- [x] Script paths use `CFG._ROOT_DIR` (pathlib-safe).
- [x] No shell injection surface.

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-04-19 | kouui | Initial draft |
| 2026-04-19 | kouui | Post-P1-review Decision 4 revision: per-case assertion depth (18 total reference keys, not 24); `sort_keys=False` adopted after discovering existing JSON was not sort-ordered. |
| 2026-04-19 | kouui | Stage A delivered. Task status: Done. |
