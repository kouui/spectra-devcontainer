# Changelog

Live changelog for ongoing work. Add new entries at the top, using the
same `## YYYY-MM-DD` / `### <file> — <author>` layout as the archives.

When this file approaches 300 lines (enforced by
`scripts/check_changelogs_size.py` in pre-commit), rotate it into
`changelogs/archives/changelog_<YYYYMMDD>.md` (filename uses the most
recent date inside the file) and start a fresh entry below.

## 2026-08-08

### `Experimental/MALI/{GlobalMesh.py, ProfileTable.py, Structs.py, Loop.py}`, `tests/unittest/test.MALI_{GlobalMesh,Structs,ProfileTable,Loop2lv,LoopML}.py` — YW.Huang

toy-scale MALI prototype (Rybicki & Hummer 1992), isolated in `Experimental/` because the 0-D `Function/SEquil` architecture does not carry to 1-D: the SE loop there recomputes loop-invariant thermodynamics per call, and its structs have no depth axis. The package splits the computation into three tiers. Build-once: `GlobalMesh` anchors each line's dimensionless template with one scalar ruler, `wl = w0·(1 + q·ξ_ref/c)` — never the local Doppler width — then merges, sorts and dedups all lines into a single depth-independent axis with per-line `(Nblue, span)` windows; that axis-locality separation is the precondition for multi-D RT. Per-`(Te, Ne)`, outside the iteration: `Structs.precompute_` evaluates everything populations never touch — LTE ratios (`SELib._ni_nj_LTE_`), collision rates (`Collision.Cij_to_Cji_`), Planck values, and the profile table, where depth enters exclusively through `x = (wl − w0)/dopWidth(k)` (the 0-D code multiplies the template by the width; the 1-D table divides by it) and `wphi` records the numerical profile norm so rate integrals dividing by it cancel the quadrature error identically; passive b-f rates come from the production `SELib._bf_R_rate_` under a prescribed Planck field. Per-iteration: jitted sweeps accumulate `Jbar` and the diagonal `Lambda_star` from `formal_improved_RH_(..., with_psi=True)` in one pass, and `update_populations_` assembles preconditioned line rates (`Aji·(1−Λ*)`, `Jbar − Λ*·S` in both directions) plus unpreconditioned passive continuum rates, feeding the **unchanged** production `set_matrixR_`/`set_matrixC_`/`solve_SE_` one depth at a time. The per-iteration kernels compile unconditionally (`nb_njit(**NB_NJIT_KWGS)` regardless of `CFG._IS_JIT`); interpreted-by-default callees are bound by compiling their raw `.pyfunc` (`voigt_`, `bb_extinction_`, `bb_emissivity_`), gated callees by wrapping the production function itself. Everything is verified on fabricated toys against exact oracles rather than real atoms: `eps = 1` converges in one iteration; the sqrt(eps) surface law; plain Lambda-iteration and a deliberately half-scaled operator reach the same fixed point (the operator cancels at `S_new = S_old`, so it sets the rate, never the answer); collision domination lands on Boltzmann, and with the passive continuum on Saha-Boltzmann, at 1e-6; a level connected through exactly one transition carries zero net flux, so a 3-level atom with a high spectator level reproduces the pure two-level balance at 1e-9 — the multilevel machinery reducing to the separately-verified two-level case with full-strength rates. Two deviations recorded: toys carry fabricated `Cij_coe` directly because `SELib._get_Cij_` rejects continuum-free models, and the sweeps integrate tau inline because `Tau.z_to_dtau_` is an incomplete stub (allocates, never returns). 39 new tests; full suite 367 passed / 1 skipped; ruff findings unchanged from main.

### `notebooks/demo/mali_3level/{build_notebook.py, mali_3level.ipynb}` — YW.Huang

new generated demo walking the pipeline end to end: the shared axis with its three line windows; the profile-table heat map (same columns every depth, hotter rows broader) beside the `wphi` degradation of a deliberately coarse window; the sqrt(eps) law over four decades with convergence histories of MALI, half-operator MALI and plain Lambda-iteration collapsing onto one fixed point; 3-level departure coefficients with the collision-dominated run flattening onto `b_i = 1`; the single-channel reduction; and the passive-continuum toy detail-balancing onto the Saha ionization fraction.

## 2026-07-26

### `RadiativeTransfer/Feautrier.py`, `Enums.py`, `Types.py`, `ImportAll.py`, `tests/unittest/test.Feautrier.py`, `tests/numba/test_Feautrier_jit.py`, `tests/regression/test_reg_Feautrier.py` — YW.Huang

finish the Feautrier port: the reference implementation carries three capabilities the second-order formal solver never picked up. `formal_improved_RH_` now takes `order` / `with_psi` and returns `Feautrier_Result(j, I_emergent, Psi)`. The order switch is a new `E_FEAUTRIER_ORDER` IntEnum (values are the formal accuracy order, so name and value cannot drift) rather than a string, because string dispatch does not lower under numba — the precedent is `E_ABSORPTION_PROFILE_TYPE` inside the jitted `_B_Jbar_`; the named tuple lives at module scope for the same reason `_B_Jbar_Result` does. Coefficients moved from loop scalars into `abc`/`A1`/`C1`/`Stmp` arrays because the Hermite corrections (Auer 1976) rewrite them after the second-order values exist and the `Psi` recursion needs them a second time; the floating-point operation order was preserved, so `order=SECOND` is bit-identical to the previous implementation across 96 parameter combinations and the stored regression references pass unchanged (two call sites take `.j`). `Psi = diag(T^-1) = dj/dS` is extracted by a backward `G` recursion mirroring `F`, in O(ND) — the local operator ALI needs, without ever forming an ND×ND matrix, which is why the deferred T-matrix work shrinks to multilevel MALI. Unlike the reference, which terminates on HERMITE + `Psi`, the approximate operator is returned and documented: the ALI fixed point is independent of `Lambda_star` (the terms cancel once `S_new = S_old`), so an imperfect operator changes the convergence rate, not the answer — whereas zero-filling would silently degenerate ALI to plain Lambda-iteration. Two measured findings are recorded in the docstring rather than papered over: HERMITE converges at **3rd** order, not 4th (2.953 measured; the interior rows gain two orders but the two boundary rows gain one, and the boundary caps the global rate — the reference's "fourth order" describes the interior scheme), and the `Psi <= 1` bound breaks at *either* boundary once that interval exceeds sqrt(6). Also new: `direct_feautrier_` (ported from `Experimental/`, hard-coded Gauss constants replaced by `GaussLeg.gauss_quad_coe_` behind `n_angle`, left interpreted as a verification-only reference) and `I_emergent`, inverting the upper boundary condition. 328 passed / 1 skipped; ruff findings unchanged from main.

### `notebooks/demo/feautrier_method/{build_notebook.py, feautrier_method.ipynb}` — YW.Huang

new generated demo: closed-form verification (constant source, pure attenuation, Eddington-Barbier), a boundary-condition gallery, measured convergence order, and the two-level atom solved three ways (Lambda-iteration vs ALI vs `direct_feautrier_`) with a convergence-history panel and a perturbed-`Lambda_star` demonstration of the fixed-point argument above. The two-level reference curve that `notebooks/RadiativeTransfer/FeautrierMethod.ipynb` labels "exact" is the Eddington (two-stream) approximation — it reproduces the sqrt(eps) surface law exactly, which is what makes it convincing, but its tau-dependence differs from the exact Hopf solution by a few percent mid-atmosphere. It is relabelled here, kept for context only, and never used to verify the solver.

## 2026-07-04

### `Util/AtomUtils/AtomIO.py`, `Function/SEquil/SELib.py`, `Function/SlabModel/CloudModel.py`, `Experimental/ExSpectrum.py`, `tests/unittest/test.doppler_split.py`, `tests/regression/atom_reference_values.json` — YW.Huang

lines between degenerate levels (f0 ≤ 0, e.g. the unresolved fine-structure pairs in Ca_I-II-III) carried `w0 = 0.0`, and `bb_extinction_` / `bb_emissivity_` divide by `c_/w0` — ZeroDivisionError in `SE_to_slab_0D_`. AtomIO now writes the sentinel `w0 = inf` instead (the physical limit of c/f0; `c_/w0 → 0.0`), and every w0-arithmetic consumer skips f0 ≤ 0 lines explicitly: `SELib._B_Jbar_` gets an early-skip (wavelength-like slices inf, radiation-like slices 0) replacing the fall-through arithmetic and late guard; `CloudModel._SE_to_slab_0D_bb_` merges its two per-line loops into one with a single skip block (background interpolation moves per-line); `ExSpectrum` gets the same skip. `test.doppler_split` filters inactive slices with `isfinite` instead of `> 0`; `atom_reference_values.json` regenerated (9 leaves: 3 Ca sentinel lines × {`Line.w0`, `Line.w0_AA`, `Line_Coe.w0`}, 0.0 → Infinity). Full suite 289 passed; H/He e2e goldens unchanged — active-line paths are computationally identical. Follow-up: the ExSpectrum skip left `line_mesh_cm` uninitialized for a skipped line while the belonging-line loop still iterated the full RL list and read those slices — the RL line indices are now prefiltered once and shared by both loops, so an RL-listed inactive line is consistently left out of the spectrum (`Spec.RL_lineindex` keeps the unfiltered copy, field semantics unchanged).

### `data/atom/Ca_I-II-III/{Ca.Level, Ca.Aji, Ca.Alpha, Ca.CE.electron, Ca.CI.electron, Ca.RadiativeLine, CaI+II_45.RH.configuration-table.txt}`, `tests/regression/atom_reference_values.json` — YW.Huang

correct the Ca II 3d 2D 5/2 level label typo `2d → 3d` (in `Ca.Level` also the principal quantum number column 2 → 3) and propagate the corrected configuration through every table that references the level by name (Aji, Alpha, CE, CI, RadiativeLine, RH configuration table). Energies and rate coefficients are untouched — only the label was wrong; `atom_reference_values.json` regenerated for the renamed ctj keys.

## 2026-07-01

### `notebooks/demo/rydberg_se_sensitivity/{build_notebook.py, rydberg_se_sensitivity.ipynb}` — YW.Huang

drop the imposed radiation temperature from the SE solve: `cal_SE_with_Ne_Te_(...)` is now called with the default `se_params=None` (solar radiation field) instead of `SE_Params_Container(Tr=Te)` (Planck driver at Te) — for a statistical-equilibrium study the radiation temperature should not be imposed. The physics changes substantially, so the narrative was reworked to match: the analytic-Saha expectation and the "high-Ne limit reduces to exact LTE" claim are removed, a third Rydberg channel is documented (the shifted transition wavelengths re-sample the non-Planckian solar spectrum), the Saha-validation section is deleted, and the takeaway now reports a radiation-driven sensitivity — excited-level populations swing by order-unity fractions of their small populations; only the collision-dominated ground state at high Ne stays Saha-order (~−1% at 7000 K). Notebook regenerated and executed end-to-end.

## 2026-06-28

### `pyproject.toml`, `.pre-commit-config.yaml`, `tests/unittest/{test.RomanElement.py, test.SE.H_I.py, test.basic_functions.py}` — YW.Huang

make `tests/unittest/` discoverable and enforced. Two layers let the directory rot undetected: pytest's default `python_files` glob (`test_*.py`) never matched the project's dot-named files, so a directory-level run collected nothing there; and the pre-commit pytest hook only ran `tests/regression/`. Fix: `python_files = ["test_*.py", "test.*.py"]` (importlib import-mode was already set for the dot-in-name imports) and extend the hook to `tests/unittest/`. Green-up of the now-collected suite: `test.RomanElement.py` asserts the He abundance via the `ELEMENT_ABUN` lookup + `10**(A-12)` transform instead of a stale hardcoded 10.93; delete `test.SE.H_I.py` (stale duplicate of `test_reg_e2e_SE.py::TestHydrogenSE::test_SE_with_Nh_Te`, which has maintained goldens) and the empty `test.basic_functions.py`.

### `Struct/Atom.py` — YW.Huang

`init_Atom_` returns a single `Atom`, dropping the `path_dict` from its return value. Callers that still need the data-file paths call `AtomIO.read_conf_(conf_path)` directly, which yields the identical dict. All non-notebook callers updated (scripts, `tests/examples`, `tests/regression`, `tests/unittest`); `atom_reference_values.json` verified byte-identical after regeneration.

### `notebooks/demo/hydrogen_doppler/{build_notebook.py, hydrogen_doppler.ipynb}` — YW.Huang

adapt the one `init_Atom_` call site to the single-return API.

### `notebooks/demo/rydberg_se_sensitivity/{build_notebook.py, rydberg_se_sensitivity.ipynb}` — YW.Huang

new generated demo: sensitivity of the hydrogen SE populations to the Rydberg constant (R∞ = 109737.316 vs measured R_H = 109678.758 cm⁻¹) over Te ∈ {7000, 10000, 50000} K × Ne ∈ {1e8 … 1e13} cm⁻³. Monkeypatches `Constants.E_Rydberg_H_` so the theoretical hydrogen atom rebuilds its level energies and rate coefficients from the one constant — a single-knob experiment (guarded by `assert Hydrogen.CST is CST`; valid because JIT is off and every lookup is a qualified `CST.E_Rydberg_H_`). Summary tables, Te×Ne heatmaps, per-level bar chart. Initially solved with a Planck-at-Te driver; superseded by the 2026-07-01 entry above.

### `Atomic/emisivity.py`, `Function/SlabModel/CloudModel.py` — YW.Huang

kill the spurious vertical spike at the Balmer ionization limit: `bf_emissivity_` computed `eps = h·nu − chi` by reconstructing `nu = c/wl`, a float round-trip that lands 1 ULP negative at the mesh edge for some series, so the `eps < 0` guard zeroed the edge emissivity. Rearranged to the algebraically identical `eps = chi·(w0 − wl)/wl`, which is exactly 0 at the edge column (`wl == w0` bitwise) independent of the level energies; the single caller now passes the Cont `w0`. Follow-up safety net: relax the hard `eps < 0` cut to a small tolerance (~1e-9·chi — far above 1 ULP, far below kTe) so meshes not built from this exact `w0` stay finite at the edge while genuine sub-threshold points still return 0.

## 2026-06-27

### `Constants.py`, `Atomic/{Hydrogen.py, ContinuumOpacity.py}`, `Experimental/Cmat_Hydrogen.py`, `Function/Hydrogen/DegenerateN.py`, `Struct/Atom.py`, `Util/AtomUtils/MakeTheoreticalHydrogenLevels.py` — YW.Huang

standardize the hydrogen Rydberg energy on the measured ionization: `E_Rydberg_H_ = R_H_ · c_ · h_` with `R_H_ = 109678.758 cm⁻¹` (NIST/RH, ~13.5984 eV), routed through all hydrogenic level / transition / photoionization energy calculations, replacing the infinite-mass `E_Rydberg_` (R∞). A single empirical constant reproduces the NIST level energies, the ionization edge, and the Balmer/Lyman line positions; the level builders reference it instead of redefining a Rydberg locally. `E_Rydberg_` is retained for the van Regemorter collision constant, with its docstring corrected to state it is the infinite-nuclear-mass value.

### `data/atom/H/{H, H6, H8, H13, H21, H34}.Level`, `scripts/gen_hydrogen_levels.py` — YW.Huang

regenerate the theoretical hydrogen level energies from the measured constant; add the generator script so the tables stay reproducible.

### `data/atomic_raw/H_NIST/level_energy_eV.csv` — YW.Huang

add the NIST term-averaged H I level-energy reference table (the empirical basis for the measured-R_H choice).

### `scripts/gen_reference.py`, `tests/regression/{conftest.py, README.md, reference_values.json, atom_reference_values.json, test_reg_Hydrogen.py}` — YW.Huang

complete the regression reference generators and regenerate the goldens for the measured constant. `gen_reference.py` rebuilds `reference_values.json` by running the regression suite in record mode (`REGEN_REFS=1`): each `assert_close` records its computed value under its ref key, then a verify pass confirms — the tests themselves are the single source of truth, superseding the hand-maintained `gen_se_reference.py` (removed, it covered only 6 SE cases). Recording is keyed on the exact object `ref[key]` returns rather than a global "fresh" flag, so locally computed invariants (e.g. sum-to-one checks) are still asserted during the record pass. `test_reg_Hydrogen.py` derives the PI threshold wavelength from `E_Rydberg_H_` instead of a hardcoded literal.

## 2026-06-06

### `Atomic/{emisivity.py, extinction.py}` — YW.Huang

new modules with wavelength-base coefficients: `bb_emissivity_` / `bf_emissivity_` and `bb_extinction_` / `bf_extinction_` (renamed to the public trailing-underscore convention shortly after introduction). `bf_emissivity_` guards `eps < 0`.

### `Function/SlabModel/CloudModel.py`, `Struct/Container/CloudModel.py` — YW.Huang

add the bound-free continuum to the 0D slab model. `SE_to_slab_0D_` splits into `_SE_to_slab_0D_bb_` (line) + `_SE_to_slab_0D_bf_` (continuum) and the public function returns both as a tuple. `CloudModel_Container` becomes `CloudModel_BB_Container` (physical emissivity/absorption fields with zero-copy `line_emissivity` / `line_absorption` aliases); new wavelength-resolved `CloudModel_BF_Container`. The b-f path evaluates j/alpha on the SE continuum mesh via `atom.PI.alpha_interp`, forms S = j/alpha, tau, and the emergent intensity (no Doppler shift); the b-b RT outputs are bit-for-bit unchanged. The b-b opacity, emissivity, and source function are rerouted through the new `Atomic` functions instead of inline formulas. Consumers migrated: `Util/AtomUtils/AtomInfo.py`, `Experimental/ExLibTmp.py` annotations, and the regression / example / doppler-split tests call `_SE_to_slab_0D_bb_` directly (reference values unchanged).

### `notebooks/demo/bf_continuum/{build_notebook.py, bf_continuum.ipynb}`, `notebooks/demo/hydrogen_doppler/{build_notebook.py, hydrogen_doppler.ipynb}` — YW.Huang

new generated demo for the bound-free continuum slab model; the hydrogen_doppler demo migrated to unpack the new `SE_to_slab_0D_` `(bb, bf)` tuple return.

## 2026-05-30

### `RadiativeTransfer/CloudModel.py`, `Function/SlabModel/CloudModel.py`, `Util/AtomUtils/AtomInfo.py`, `Experimental/ExLibTmp.py` — YW.Huang

optional background intensity for the slab model: new `emergent_intensity_(Src, tau, I0=0.0)` computes the exact uniform-slab solution `I = Src·(1−exp(−tau)) + I0·exp(−tau)` and is reused across call sites. `SE_to_slab_0D_` gains an optional trailing `I0` argument — a `(2, n_wavelength)` background table (like `Radiation.solar`) or None — interpolated once onto the observer-frame wavelength mesh and applied per line; default None reproduces the previous output. The two duplicated `extract_lprof` copies (AtomInfo, ExLibTmp) now call the shared helper, preserving their once-vs-per-line background semantics.

### `Function/SEquil/SELib.py`, `Struct/Container/SEquil.py` — YW.Huang

export the continuum wavelength mesh via `SE_Container`.

## 2026-05-29

### `Struct/Atom.py`, `Struct/WavelengthMesh.py`, `Function/SEquil/SELib.py`, `Experimental/ExSpectrum.py` — YW.Huang

embed the `Wavelength_Mesh` into the private field `Atom._wave_mesh`, matching the `_ctj_table` / `_idx_table` convention: `init_Atom_` returns `(atom, path_dict)`, `init_theoretical_hydrogen_atom_` returns `atom`; the 5 public `cal_SE_*` functions and `ExSpectrum.init_spectrum_` drop their `wMesh` parameter and read `atom._wave_mesh` internally. Dead `Line_mesh_share` / `Line_mesh_share_idxs` fields removed (never read by any production path). Call sites in scripts and tests updated; `atom_reference_values.json` regenerated (904 keys). Follow-ups the same day: the string forward-reference annotation on `_wave_mesh` becomes a plain annotation (no circular import exists), and `Atom` gains `nBoundLevel` (bound levels only, excludes the continuum level).

### `notebooks/{Hydrogen_atom, RadiativeTransfer/Transfer1D, StatisticalEquilibrium/*}.ipynb` (12 notebooks) — YW.Huang

migrate to the `Atom._wave_mesh` API (drop the wMesh threading through cal_SE_* / init_spectrum_ calls). `RadiativeTransfer/Transfer1D.ipynb` additionally passes `Vd_obs=0.0` in its `init_spectrum_` call.
