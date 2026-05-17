# Changelog archive

Covers 2021-06-07 (newest) down to 2006-05-23 (oldest).

## 2021-06-07

### `Atomic/ContinuumOpacity.py` — YW.Huang

- modified HI_rayleigh_cross_sec_
- in _avH2p_, cubic v.s. linear : 3x difference

### `Util/AtomUtils/AtomIO.py` — YW.Huang

- `Abun : T_FLOAT = _ElementUtil.sym_to_abun_( element )``

### `Util/ElementUtil.py` — YW.Huang

- sym_to_abun_ return  10.**(x-12.0)


## 2021-05-29

### `Experimental/ExLTELib.py` — k.ichimoto, YW.Huang

- fixed log_saha_, population_to_H_ --> now we are able to reproduce ichimoto's LTE transfer result

### `Experimental/ExLine.py` — k.ichimoto, YW.Huang

- class Line

### `Experimental/ExTau.py` — k.ichimoto, YW.Huang

### `Util/RomanUtil.py` — YW.Huang

- created this module for ionization stage


## 2021-05-18

### spectra-re — YW.Huang

Files:
- `Atomic/BasicP.py`
- `Atomic/Collision.py`
- `Atomic/LTELib.py`
- `Atomic/PhotoIonize.py`
- `Atomic/SEsolver.py`
- `Configurations.py`
- `Elements.py`
- `Enums.py`
- `Experimental/ExLTELib.py`
- `Experimental/ExLine.py`
- `Experimental/ExTau.py`
- `Function/Hydrogen/DegenerateN.py`
- `Function/SEquil/SELib.py`
- `Function/SlabModel/CloudModel.py`
- `ImportAll.py`
- `Math/BasicM.py`
- `Math/GaussLeg.py`
- `Math/Integrate.py`
- `Math/Special.py`
- `RadiativeTransfer/Profile.py`
- `RadiativeTransfer/Thin.py`
- `Removal.ImportExternalModule.py`
- `Struct/Atmosphere.py`
- `Struct/Atom.py`
- `Struct/Container/CloudModel.py`
- `Struct/Container/SEquil.py`
- `Struct/Radiation.py`
- `Struct/WavelengthMesh.py`
- `Types.py`
- `Util/AtomUtils/AtomIO.py`
- `Util/AtomUtils/AtomInfo.py`
- `Util/ElementUtil.py`
- `Util/HelpUtil.py`
- `Util/MeshUtil.py`
- `Util/RomanUtil.py`
- `Visual/Animation.py`
- `Visual/Plotting.py`
- `Warnings.py`

### `Atomic/ContinuumOpacity.py` — YW.Huang

spectra-re

- migrated from opacity.py

### `Atomic/Hydrogen.py` — YW.Huang

spectra-re

- `Gaunt_factor_Gingerich_cm_()`, `w_um = w * 1E5` -> `w_um = w * 1E4`

### `Constants.py` — YW.Huang

spectra-re

- added R_


## 2021-05-08

### `Atomic/PhotoIonize.py` — YW.Huang

- func interpolate_PI_alpha : flip boundary value `_fill_value`

### `Function/SEquil/SELib.py` — YW.Huang

- func B_Jbar : _meshInfo assigment
- func B_Jbar : _Gamma = numpy.atleast_1d( _Gamma )


## 2020-11-16

### `Function/SEquil/SELib.py` — YW.Huang

- Te and Tr identification in bf_R_rate() and bf_R_rate_loop()
- 0.5*intensity --> 1.0*intensity in the integration in B_Jbar_CRD()

- by defalut the line absorption profile is Gaussian shape so no damping effect in line wings ```B_Jbar() _meshInfo[3] = 2 ```
- in LevelN.collisional_broadening(), ground hydrogen population is set(fixed) to 1E10


## 2020-11-10

### `Atomic/Hydrogen.py` — YW.Huang

- `PI_cross_section_cm()` and `PI_cross_section()`, if `x<1.0` then cross section `alpha=0.`


## 2020-05-08

### `Constants.py` — k.ichimoto

alp_


## 2020-02-13

### `Atomic/LTELib.py` — k.ichimoto

func `Ufunc_`: from IDL ufunc_gray.pro


## 2019-11-26

### `Atomic/LTELib.py` — k.ichimoto

func `Ufunc_`: 'Ba' from Gary 2009 (use poly_ufunc.pro)


## 2019-09-15

### `Atomic/ContinuumOpacity.py` — k.ichimoto

func `H2p_cross_sec_`

func `HI_bf_LTE_cross_sec_`: from IDL ahic.pro

func `HI_ff_cross_sec_`: from IDL ahic.pro

func `H_LTE_continuum_opacity_`: from IDL avray.pro

func `Hminus_cross_sec_`: from IDL ahic.pro

func `hydrogenic_bf_cross_sec_n_`: from IDL ahic.pro


## 2019-09-11

### `Atomic/ContinuumOpacity.py` — k.ichimoto

func `_avH2p_`: from IDL avh2p.pro


## 2019-09-10

### `Atomic/ContinuumOpacity.py` — k.ichimoto

func `HI_rayleigh_cross_sec_`: from IDL avray.pro


## 2015-07-05

### `Atomic/LTELib.py` — k.ichimoto

func `Ufunc_`: 'h_ii'


## 2006-05-23

### `Atomic/LTELib.py` — k.ichimoto

func `Ufunc_`
