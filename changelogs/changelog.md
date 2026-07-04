# Changelog

Live changelog for ongoing work. Add new entries at the top, using the
same `## YYYY-MM-DD` / `### <file> — <author>` layout as the archives.

When this file approaches 300 lines (enforced by
`scripts/check_changelogs_size.py` in pre-commit), rotate it into
`changelogs/archives/changelog_<YYYYMMDD>.md` (filename uses the most
recent date inside the file) and start a fresh entry below.

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
