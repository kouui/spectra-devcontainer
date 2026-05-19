# Changelog archive

Covers 2026-05-15 (newest) down to 2021-06-15 (oldest).

## 2026-05-15

### `Function/SEquil/SELib.py` — YW.Huang

se_params: SE_Params_Container threaded through cal_SE_* wrappers

### `Struct/Atmosphere.py` — YW.Huang

- removed Tr / use_Tr / doppler_shift_continuum (moved to SE_Params_Container)

### `Struct/Container/SEquil.py` — YW.Huang

added SE_Params_Container (Tr / doppler_shift_continuum moved off Atmosphere; Tr is None ⇔ use radiation.solar)


## 2026-05-09

### `Function/SlabModel/CloudModel.py` — YW.Huang

take abs(tau).max() so tau_max is meaningful under population inversion (negative tau)

### `Struct/Container/CloudModel.py` — YW.Huang

clarify tau_max docstring: max |tau| (handles population inversion)


## 2026-05-03

### `Struct/Container/SEquil.py` — YW.Huang

added PI_intensity (continuum-mesh-resolved bound-free intensity), cont_wave_mesh_shifted (parallel to bound-bound wave_mesh_shifted_1d / absorb_prof_1d)

### `Struct/Radiation.py` — YW.Huang

drop cached PI_intensity; rename backRad -> solar; init no longer takes atmos/wMesh


## 2025-07-07

### `Function/SlabModel/CloudModel.py` — j.natsume

add line_emissivity and line_absorption

### `Struct/Container/CloudModel.py` — j.natsume

add line_emissivity, line_absorption


## 2024-04-06

### `Experimental/Cmat_Hydrogen.py` — k.ichimoto

from H_spectra.


## 2024-02-09

### `Function/SEquil/SELib.py` — YW.Huang

Ntotal, Nh, Ne, Te in SE_con

### `Struct/Container/SEquil.py` — YW.Huang

added Ntotal,Nh,Ne,Te to SE_Container


## 2024-02-03

### `Function/SEquil/SELib.py` — YW.Huang

add rate_only argument


## 2024-02-02

### `Util/AtomicDataUtils/MakeTheoreticalHydrogenLevels.py` — YW.Huang


## 2023-07-06

### `Function/SEquil/SELib.py` — YW.Huang

in _B_Jbar_, when f0 = 0, skip calculation and assign 0 to the result


## 2023-07-04

### `Atomic/Collision.py` — YW.Huang

in interp_omega_, when table[:] are 0s, return 0

### `Function/SlabModel/CloudModel.py` — YW.Huang

when Aji equals to 0, set Src to 0 to avoid the zero division warning

### `Util/AtomUtils/AtomIO.py` — YW.Huang

when f0 equals 0, w0 Aji Bji Bij are set to 0


## 2023-04-29

### `Function/SEquil/SELib.py` — YW.Huang

- is_single_element keyword in cal_SE_with_Ne_Te_


## 2022-09-04

### `Function/SEquil/SELib.py` — YW.Huang

- added cal_SE_with_Pg_Te_single_Atom_


## 2022-08-01

### `Struct/Radiation.py` — YW.Huang

modified atlas(added 10000-11000A absorption lines into backRad)


## 2022-07-29

### `Elements.py` — YW.Huang

WEIGHTED_TOTAL_MASS TOTAL_ABUN

### `Function/SEquil/SELib.py` — YW.Huang

- added cal_SE_with_Pg_Te_

### `Struct/Atmosphere.py` — YW.Huang

- added Pg to Atmosphere0D


## 2022-07-24

### `Util/AtomUtils/AtomInfo.py` — k.ichimoto, YW.Huang

each_prof, extract_lprof, level_info


## 2022-07-20

### add Src, tau_1D — k.ichimoto

Files:
- `Function/SlabModel/CloudModel.py`
- `Struct/Container/CloudModel.py`


## 2022-06-26

### `Function/Hydrogen/DegenerateN.py` — YW.Huang

compute_Hydrogenic_PI_cross_section_

### `Util/AtomUtils/AtomIO.py` — YW.Huang

data & Hydrogenic mixed photoionization cross section


## 2022-01-07

### `Atomic/ContinuumOpacity.py` — YW.Huang

- removed warnings and added JIT for `HI_rayleigh_cross_sec_`

### `Experimental/ExLTELib.py` — YW.Huang

- added vectorization for `population_to_H_` when JIT applied

### `Struct/Atmosphere.py` — YW.Huang

- in `init_VAL_`, added zero array of `column_mass` to initialize struct

### `Struct/Radiation.py` — YW.Huang

modified atlas(backRad)


## 2021-08-10

### `Experimental/ExFAL.py` — YW.Huang

### `Experimental/ExScatter.py` — YW.Huang

### `Experimental/ExSpectrum.py` — YW.Huang


## 2021-07-13

### `Util/AtomicDataUtils/MakeElectronImpactIoniz.py` — YW.Huang

### `Util/AtomicDataUtils/MakePhotoioniz.py` — YW.Huang


## 2021-07-06

### `Experimental/ExFeautrier.py` — YW.Huang

### `RadiativeTransfer/Feautrier.py` — YW.Huang

### `RadiativeTransfer/Tau.py` — YW.Huang


## 2021-07-04

### `Function/SEquil/SELib.py` — YW.Huang

- tran_rate_con now has Rmat, Cmat

### `Struct/Container/SEquil.py` — YW.Huang

added Rmat, Cmat to TranRates_Container

### `Visual/Animation.py` — YW.Huang


## 2021-06-18

### `Visual/Grotrian.py` — YW.Huang

spectra-re


## 2021-06-16

### `Util/AtomUtils/AtomIO.py` — YW.Huang

- `atom.Line` now is a ndarray instead of a recarray


## 2021-06-15

### `Atomic/Hydrogen.py` — YW.Huang

- `Gaunt_factor_Gingerich_cm_()`, `w_um = w * 1E4` -> `w_um = w * 1E5` [cm] -> [um] is *1E4, why *1E5?

### `Function/SEquil/SELib.py` — YW.Huang

- func _bf_R_rate_ : move radiation-source switch outside the continuum doppler-shift branch to enable the coronal-equilibrium "no radiation" mode (Tr at 0 K with Planck-source selected).
- func _bf_R_rate_ : add local variable `PI_I0` to prevent the update of `PI_I` udring simulation
