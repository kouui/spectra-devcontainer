# Changelog

Live changelog for ongoing work. Add new entries at the top, using the
same `## YYYY-MM-DD` / `### <file> — <author>` layout as the archives.

When this file approaches 300 lines (enforced by
`scripts/check_changelogs_size.py` in pre-commit), rotate it into
`changelogs/archives/changelog_<YYYYMMDD>.md` (filename uses the most
recent date inside the file) and start a fresh entry below.

## 2026-05-23

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
