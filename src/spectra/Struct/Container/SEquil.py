# -------------------------------------------------------------------------------
# definition of struct for storing Statistical Equilibrium result
# -------------------------------------------------------------------------------
# VERSION
# 0.1.1
#    2021/07/04   u.k.   added Rmat, Cmat to TranRates_Container
# 0.1.0
#    2021/05/18   u.k.   spectra-re
# 0.1.1
#    2024/02/09   u.k.   added Ntotal,Nh,Ne,Te to SE_Container
# 0.2.0
#    2026/05/03   u.k.   added PI_intensity (continuum-mesh-resolved bound-free
#                        intensity), cont_wave_mesh_shifted
#                        (parallel to bound-bound wave_mesh_shifted_1d / absorb_prof_1d)
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

from ...ImportAll import *


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class SE_Container:
    """Statistical Equilibrium result container for
    - single spatial point
    """

    n_SE: T_ARRAY  # 1d (nLevel,), [/cm^3]
    n_LTE: T_ARRAY  # 1d (nLevel,), [/cm^3]

    nj_by_ni: T_ARRAY  # 1d (nLine+nCont,), [-]

    ## wavelength mesh of line transition
    wave_mesh_shifted_1d: T_ARRAY  # 1d (sum_of_line_wavelength_mesh,), [cm]
    ## absorption profile of line transition
    absorb_prof_1d: T_ARRAY  # 1d (sum_of_line_wavelength_mesh,), [/cm]
    ## index array of line transition
    Line_mesh_idxs: T_ARRAY  # 1d (sum_of_line_wavelength_mesh,), [-]

    Jbar: T_ARRAY  # 1d (nLine,), [erg/cm^2/Sr/s]

    ## atom density, if hydrogen, N_total==N_h
    Ntotal: T_FLOAT  # 0d, [/cm^3]
    ## hydrogen density
    Nh: T_FLOAT  # 0d, [/cm^3]
    ## electron density
    Ne: T_FLOAT  # 0d, [/cm^3]
    ## electron temperature
    Te: T_FLOAT  # 0d, [K]

    ## continuum wavelength mesh actually used in this SE call.
    ## currently aliases `wMesh.Cont_mesh` (no doppler shift implemented yet);
    ## the field exists as a forward-ready hook so future doppler-shifted continuum
    ## arrays land here, parallel to bound-bound `wave_mesh_shifted_1d`.
    ## Mutating it also mutates wMesh.Cont_mesh — copy if you need to modify.
    cont_wave_mesh_shifted: T_ARRAY  # 2d (nCont, _N_CONT_MESH), [cm]
    ## PI (photoionization) intensity used to drive bound-free rates.
    ## continuum-mesh-resolved: planck(Tr) when atmos.use_Tr,
    ## else interp(radiation.solar, cont_wave_mesh_shifted).
    PI_intensity: T_ARRAY  # 2d (nCont, _N_CONT_MESH), [erg/cm^2/Sr/cm/s]


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class TranRates_Container:
    """Statistical Equilibrium (Transition Rates) result container for
    - single spatial point
    """

    Rji_spon: T_ARRAY  # 1d (nLine+nCont), [/s]
    Rji_stim: T_ARRAY  # 1d (nLine+nCont), [/s]
    Rij: T_ARRAY  # 1d (nLine+nCont), [/s]

    Cji_Ne: T_ARRAY  # 1d (nLine+nCont), [/s]
    Cij_Ne: T_ARRAY  # 1d (nLine+nCont), [/s]

    Rmat: T_ARRAY  # 2d (nLevel, nLevel), [/s]
    Cmat: T_ARRAY  # 2d (nLevel, nLevel), [/s]
