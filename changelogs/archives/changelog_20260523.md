# Changelog archive

Covers 2026-05-23 (newest) down to 2026-05-20 (oldest).

## 2026-05-23

### `notebooks/Hydrogen_atom.ipynb`, `notebooks/StatisticalEquilibrium_CloudModel.demo.ipynb`, `notebooks/StatisticalEquilibrium/H_spectra.ipynb`, `notebooks/StatisticalEquilibrium/H_spectra_otsu_ver20240424.ipynb`, `notebooks/StatisticalEquilibrium/HHeCa.ipynb`, `notebooks/StatisticalEquilibrium/HHeCa_0.ipynb`, `notebooks/StatisticalEquilibrium/HeH.ipynb`, `notebooks/StatisticalEquilibrium/He_plasma.NeTe.ipynb`, `notebooks/StatisticalEquilibrium/He_plasma.ipynb`, `notebooks/StatisticalEquilibrium/Hydrogen_atom.ipynb`, `notebooks/StatisticalEquilibrium/CaII_flush.ipynb`, `notebooks/StatisticalEquilibrium/spectra_multi_atom.ipynb`, `notebooks/RadiativeTransfer/H_population.ipynb`, `notebooks/RadiativeTransfer/TransferBasic.20210609.ipynb`, `notebooks/RadiativeTransfer/TransferBasic.ichimoto.20220105.ipynb`, `notebooks/RadiativeTransfer/TransferBasic.ipynb` — YW.Huang

refactor 06: migrate the 16 legacy notebooks from removed `Atmosphere.Vd` to the new `Vd_sun` + `Vd_obs` API. 274 `\bVd\b` source-tokens collapse to 0. Sign policy is intent-preservation (legacy `+Vd` = atom outflow / toward observer): **0D** non-zero sites split as `Vd_sun = +v, Vd_obs = -v`; **C1D LTE** sites rename to `Vd_sun = +v` only (Vd_obs stays at the `init_VAL_` zeros). Acceptance gates: scan (`tmp/scan_legacy_Vd.py`) → 0 residual tokens; metadata-drift (`tmp/refactor_06_metadata_diff.py`) → only `cells[*].source` differs vs baseline; ast.parse (`tmp/refactor_06_ast_gate.py`) → no new SyntaxError (4 pre-existing failures unchanged). Outputs frozen per §2.3. Design doc: `docs/tasks/009-doppler-velocity-split/refactor_06.md` (v4, codex-APPROVED). Per-notebook changes:

- `notebooks/Hydrogen_atom.ipynb`: 30 → 0 tokens across 13 cells. R1 + R3 bulk drops; **R4 cell 67** (`Vd = 5E5` → `Vd_sun = +5E5, Vd_obs = -5E5`); **§2.5 cell 51** drops `Vd = np.ones(nNe)*0E5` local + `_Vd=Vd` kwarg from `SELib.B_Jbar(...)`, inserts `raise RuntimeError(...)` after the local-setup block and before `SELib.ni_nj_LTE`.
- `notebooks/StatisticalEquilibrium/Hydrogen_atom.ipynb`: 33 → 0 tokens across 14 cells. Same shape as above; **R4 at cell 66** (`Vd = 5E5` split); **§2.5 at cell 51**.
- `notebooks/StatisticalEquilibrium_CloudModel.demo.ipynb`: 26 → 0 tokens across 11 cells. R1 + R3; **R4 cell 35** (`Vd = 5E5` → `Vd_sun = +5E5, Vd_obs = -5E5`). Cell 8 `HelpUtil.help_(atmos)` output kept frozen with stale `|- Vd`.
- `notebooks/StatisticalEquilibrium/H_spectra.ipynb`: 34 → 0 tokens across 12 cells. R1 + R3 only. Cell 16 `HelpUtil.help_(a)` output kept frozen with stale `|- Vd`.
- `notebooks/StatisticalEquilibrium/H_spectra_otsu_ver20240424.ipynb`: 76 → 0 tokens across 23 cells. R1 + R3 only (all `Vd` values were literal zero).
- `notebooks/StatisticalEquilibrium/HHeCa.ipynb`: 15 → 0 tokens across 4 cells. R1 + R3; one commented-out alternative-construction line stripped to satisfy the zero-token gate.
- `notebooks/StatisticalEquilibrium/HHeCa_0.ipynb`: 12 → 0 tokens across 4 cells. R1 + R3 + 1 commented-out cleanup.
- `notebooks/StatisticalEquilibrium/HeH.ipynb`: 4 → 0 tokens across 2 cells. R1 + R3 only.
- `notebooks/StatisticalEquilibrium/He_plasma.ipynb`: 5 → 0 tokens in cell 7. R3 + 1 commented-out alternative-construction cleanup.
- `notebooks/StatisticalEquilibrium/He_plasma.NeTe.ipynb`: 5 → 0 tokens in cell 7. R3 + 1 commented-out cleanup.
- `notebooks/StatisticalEquilibrium/CaII_flush.ipynb`: 13 → 0 tokens across 5 cells. R1 + R3; **R4 cells 14 & 16** (`Vd = -3e5` → `Vd_sun = -3e5, Vd_obs = +3e5`).
- `notebooks/StatisticalEquilibrium/spectra_multi_atom.ipynb`: 9 → 0 tokens across 3 cells. R1 + R3 only.
- `notebooks/RadiativeTransfer/H_population.ipynb`: 6 → 0 tokens in cell 3. **§2.4 special-case**: drop the unused `a.Vd = a.Vd[ii] ; a.Vt = a.Vd[ii]` slicing trailing segment from both atmosphere blocks (per-depth arrays never consumed by cells 4-6); insert `raise RuntimeError(...)` before the still-removed `Atmosphere.init_MALTM_()` call. Notebook remains unrunnable on independent grounds (separate restoration task).
- `notebooks/RadiativeTransfer/TransferBasic.ipynb`: 2 → 0 tokens in cell 16. **R5a** drops redundant `atm.Vd[:] = 0.` (`init_VAL_` pre-zeros); **R5b** renames `atm.Vd[0:5] = 2.E5` → `atm.Vd_sun[0:5] = 2.E5`; `Vd_obs` untouched.
- `notebooks/RadiativeTransfer/TransferBasic.20210609.ipynb`: 2 → 0 tokens in cell 18. R5a + R5b, same shape as above.
- `notebooks/RadiativeTransfer/TransferBasic.ichimoto.20220105.ipynb`: 2 → 0 tokens in cell 21. R5a + R5b, same shape as above.

The maintained demo at `notebooks/demo/hydrogen_doppler/` was already on the new API and is excluded.

### `notebooks/demo/hydrogen_doppler/{build_notebook.py, hydrogen_doppler.ipynb}` — YW.Huang

normalise math-notation in markdown cells (refactor 05): every $w_0$ / $w_0^{\rm obs}$ / $wl_{1D}$ symbol becomes $\lambda_0$ / $\lambda_0^{\rm obs}$ / $\lambda_{1D}$ to match standard physics convention and stop visually colliding with $\omega$ (angular frequency). For the dimensionless mesh, use the conventional Voigt-profile argument $x_m$ (the code's `wm`, with $x = (\lambda - \lambda_0)/\Delta\lambda_D$); the cm mesh `wm_cm_1d` becomes $\lambda_m$ (no `^{\rm cm}` superscript needed) — dimensionally clean: $\lambda_m = x_m\,\Delta\lambda_D + \lambda_0$ and $\lambda_{1D} = \lambda_m + \lambda_0\,V_{\rm obs}/c$. Inserts a code↔math legend cell right after the title cell so the reader has a single mapping from Python identifiers (`w0`, `w0_cm`, `wm`, `wm_cm_1d`, `wl_1D`) to the math notation, with explicit units column. Also: Section A right-panel x-label corrected to "sun-frame **unshifted** wavelength [Angstrom]" (both panels plot against the unshifted `wm_cm_1d` axis, so labels must agree). Code-cell sources otherwise untouched (verified by post-execution byte-diff against the pre-refactor `.ipynb` — only the right-panel xlabel string drifts); Section A printed Hα shifts unchanged at `±6.569e-01 Å`.

### `Function/SEquil/SELib.py` — YW.Huang

revert `_B_Jbar_` from profile-shift to mesh-shift mechanic (refactor 04). The absorption profile now stays anchored at `w0` on the dense atom-rest-frame mesh; the solar spectrum is sampled at `wm_cm_shifted = wm_cm - w0*Vd_sun/c` instead. Fixes the off-mesh peak-truncation bug that collapsed Jbar toward zero at large `|Vd_sun|` (≥ qwing Doppler widths). Return type promoted from positional 6-tuple to a `collections.namedtuple` (`_B_Jbar_Result`, 7 named fields) for safer downstream unpacking; `cal_SE_` rewired to access via attribute. `use_Tr=True` branch stores broadcast `planck_cm_(w0, Tr)` as `solar_intensity_shifted`. Inactive-line (f0<=0) slices of the new arrays remain zero (see `docs/tasks/009-doppler-velocity-split/refactor_04.md` §2.6).

### `Struct/Container/SEquil.py` — YW.Huang

drop `absorb_prof_shifted_1d` from `SE_Container`; add `SE_BB_Container` (debug / post-analysis) with `wm_cm_shifted_1d` and `solar_intensity_shifted_1d`, exposed via `SE_Container.se_bb_con`. Cleaned stale comments on `SE_BB_Container.wm_cm_shifted_1d` (it stores wavelength labels, not a profile) and on `SE_Container.Line_mesh_idxs` (no longer references the removed field).

### `Experimental/ExLTELib.py` — YW.Huang

update sign-convention comment to call out that this LTE module independently keeps the profile-shift mechanic (sampling at `v + vv`). SE reverted to mesh-shift in refactor 04 but the LTE path is unaffected.

### `tests/unittest/test.doppler_split.py` — YW.Huang

retire `test_se_absorb_prof_shifted_1d_tracks_Vd_sun` (referenced the removed field). Add `test_se_wm_cm_shifted_1d_tracks_Vd_sun` (locks `wm_cm_shifted_1d = wm_cm_1d - w0*Vd_sun/c` per line at rtol=1e-10), `test_se_solar_intensity_shifted_1d_matches_interp` (locks the `numpy.interp` storage contract), `test_se_jbar_robust_at_large_Vd_sun` (Vd_sun=2e8 cm/s, ≈170 Hα Doppler widths — off-mesh under the old profile-shift; the test sweeps all active lines and locks `Jbar(Vd_sun=2e8) / Jbar(Vd_sun=0) > 1e-4` so the bug-signature uniform-collapse-to-machine-zero is caught while permitting the real physical variation around solar UV / Hα-core absorption features — see in-test comment for the empirical floor rationale), and `test_se_wm_cm_shifted_within_backRad_at_large_Vd_sun` (locks that the shifted mesh stays inside the solar spectrum coverage so `numpy.interp` does not silently endpoint-clamp).

### `notebooks/demo/hydrogen_doppler/build_notebook.py` — YW.Huang

rewrite Section A markdown and code for the mesh-shift mechanic. Section A now plots `solar_intensity_shifted_1d` (the local solar spectrum the atom samples) against `wm_cm_1d` (left panel) and the invariant `absorb_prof_1d` (right panel). Updated derivation text and verification narrative. Sections B and C unchanged (they were already independent of the shifted field).

### `notebooks/demo/hydrogen_doppler/hydrogen_doppler.ipynb` — YW.Huang

regenerated from updated builder and re-executed end-to-end. Section A printed shifts now report `expected dwl = -w0*Vd_sun/c` and verify the mesh-shift formula numerically at ±30 km/s on Hα (≈ ±0.657 Å).


### `Struct/Atmosphere.py` — YW.Huang

`Atmosphere0D.Vd_obs` and `Vd_sun` now default to `0.0` (and are reordered to follow `Vt` per the dataclass defaults-last rule). `AtmosphereC1D` array fields stay required (per-depth shape; both construction sites already build `zeros(nDep)`). See `docs/tasks/009-doppler-velocity-split/refactor_03.md`.

### `tests/{regression/test_reg_e2e_{SE,CloudModel}.py, examples/example.{CM,He,SE}.py, unittest/test.SE.H_I.py}` — YW.Huang

drop redundant literal `Vd_obs=0.0, Vd_sun=0.0` kwargs at 16 `Atmosphere0D` construction sites now that the dataclass defaults to zero. Helper-threaded sites (`_hydrogen_setup`, `make_atmos`) keep the kwargs (parametric forwarding).

### `scripts/gen_se_reference.py` — YW.Huang

drop `"Vd_obs": 0.0, "Vd_sun": 0.0` from the 6 `atmos_kwargs` dict literals in the `CASES` table; the constructor fills in defaults.

### `tests/unittest/test.doppler_split.py` — YW.Huang

add `test_atmos0d_default_doppler_zero`: construct `Atmosphere0D` without `Vd_obs` / `Vd_sun` kwargs; assert both default to `0.0` and that `Vt` / `Te` round-trip correctly (guards against silent field-reorder typos).

### `Function/SlabModel/CloudModel.py` — YW.Huang

flip `Vd_obs` sign convention from TOWARDS to AWAY from observer (astronomy radial-velocity convention). Observer-frame wavelength axis now built as `wl_1D = wm_cm_1d + w0·Vd_obs/c` (was `−`); `+Vd_obs` produces a red shift. Docstring and inline comment updated.

### `Experimental/ExSpectrum.py` — YW.Huang

flip `Vd_obs` sign in `init_spectrum_`: wavelength mesh shift is now `+= w0·Vd_obs/c` (was `−=`). Sign-convention comment updated.

### `Struct/Atmosphere.py` — YW.Huang

update `Atmosphere0D.Vd_obs` and `AtmosphereC1D.Vd_obs` comments to describe the new convention (+Vd_obs = AWAY from observer = red shift; line center at w0 + w0·Vd_obs/c). `Vd_sun` convention unchanged.

### `Struct/Container/SEquil.py` — YW.Huang

update example formula in `wm_cm_1d` docstring to `wl_obs = wm_cm_1d + w0·Vd_obs/c` (was `−`).

### `tests/unittest/test.doppler_split.py` — YW.Huang

- flip `test_slab_wl_1D_matches_Vd_obs_formula` expected formula to `+ w0·Vd_obs/c`
- flip `test_cloud_tau_peak_tracks_Vd_obs_only` expected_peak to `w0 + w0·Vd_obs/c`; add semantic peak-side assertion (expected_peak > w0 and measured peak > w0 for +Vd_obs)
- add `test_cloud_tau_peak_negative_Vd_obs_blue_shift`: Vd_obs = -3 km/s expects BLUE-side peak (< w0)

### `notebooks/demo/hydrogen_doppler/build_notebook.py` — YW.Huang

rebuild Section 2 physics derivation (Doppler formula direction + line-center expression), Section B + Section C `expected_peak` formulas, summary-of-signs table for the new convention. Section A (Vd_sun-only) unchanged.

### `notebooks/demo/hydrogen_doppler/hydrogen_doppler.ipynb` — YW.Huang

regenerated from updated builder and re-executed end-to-end. Cached numeric outputs flipped: `Vd_obs = +30 km/s` peak now at 6565.29 Å (red); `−30 km/s` at 6563.97 Å (blue).

### `docs/tasks/009-doppler-velocity-split/{task,plan,refactor_01}.md` — YW.Huang

sweep prose and formulas for the new convention. `refactor_02.md` is the design source-of-truth for this flip and keeps its before/after examples intact by design.

## 2026-05-20

### `Struct/Atmosphere.py` — YW.Huang

split `Vd` into `Vd_obs` (atom-line-of-sight velocity vs observer) and `Vd_sun` (atom velocity in the sun's rest frame). Sign convention: +Vd_obs TOWARDS observer (blue-shifts observer wavelengths); +Vd_sun OUTWARDS from sun (sun-frame absorption center at w0 − w0·Vd_sun/c). `init_VAL_` initialises both to zeros.

### `Struct/Container/SEquil.py` — YW.Huang

- dropped `wave_mesh_shifted_1d` and `cont_wave_mesh_shifted` (mesh-shift mechanic replaced by profile-shift)
- added `absorb_prof_shifted_1d` (Vd_sun-shifted profile; diagnostic/debug)
- added `wm_cm_1d` (sun-frame atom-rest-frame wavelength labels in cm; Te/Vt-dependent SE result, consumed by cloud model)

### `Function/SEquil/SELib.py` — YW.Huang

- `_B_Jbar_`: profile-shift mechanic — evaluate σ(wm) and σ(wm + dv_sun) on the fixed sun-frame mesh; Jbar integrates the sun-shifted profile against backRad while the wavelength mesh and backRad stay fixed
- export `wm_cm_all` (sun-frame cm wavelength labels) via SE_Container
- `cal_SE_`: read `Vd_sun` from atmos; pass dual profiles + `wm_cm_1d` to SE_Container

### `Function/SlabModel/CloudModel.py` — YW.Huang

- new signature: `SE_to_slab_0D_(atom, atmos, SE_con, depth)` (drops the `wMesh` argument)
- read unshifted `SE_con.absorb_prof_1d` and `SE_con.wm_cm_1d`; build observer-frame `wl_1D = wm_cm_1d − w0·Vd_obs/c`
- no more per-line `doppler_width_` recompute (already baked into `wm_cm_1d`)

### `Experimental/ExFAL.py` — YW.Huang

map FAL per-depth `vel` to `Vd_sun`; `Vd_obs` initialised to zeros.

### `Experimental/ExLTELib.py` — YW.Huang

read `atmos.Vd_sun[:]` (per-depth sun-frame velocity); profile sampled at `(v + vv)` to match the +Vd_sun outward convention.

### `Experimental/ExSpectrum.py` — YW.Huang

rename `Vd` arg to `Vd_obs`; wavelength shift `-= w0·Vd_obs/c` (was `+=`) to match the +Vd_obs-toward-observer convention.
