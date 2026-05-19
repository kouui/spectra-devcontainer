# -------------------------------------------------------------------------------
# definition of struct for storing Statistical Equilibrium result
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

    ## Unshifted base absorption profile of line transitions: sigma(wm) / dopWidth_cm
    ## evaluated on `wMesh.Line_mesh` (atom rest frame). This is the canonical
    ## profile used by downstream forward-model consumers (e.g. slab/cloud), which
    ## apply their own velocity shift via the output wavelength axis. Sliced per
    ## line via Line_mesh_idxs.
    absorb_prof_1d: T_ARRAY  # 1d (sum_of_line_wavelength_mesh,), [/cm]
    ## SE-internal Vd_sun-shifted profile: sigma(wm + dv_sun) / dopWidth_cm. This is
    ## the profile SE actually integrated against the sun-frame background
    ## radiation to compute Jbar. Exposed for diagnostics / debug (lets callers
    ## inspect or plot the exact profile shape SE used); NOT for cloud-model use.
    absorb_prof_shifted_1d: T_ARRAY  # 1d (sum_of_line_wavelength_mesh,), [/cm]
    ## Sun-frame, atom-rest-frame wavelength labels in cm: wm * dopWidth_cm + w0,
    ## sliced per line via Line_mesh_idxs. These are the wavelength positions
    ## absorb_prof_1d is sampled at; downstream forward models pair them with
    ## their own Vd_obs shift (e.g. wl_obs = wm_cm_1d - w0*Vd_obs/c in the cloud
    ## model). Depends on Te / Vt (via dopWidth_cm), so it belongs alongside the
    ## SE result rather than the transition-only wMesh struct.
    wm_cm_1d: T_ARRAY  # 1d (sum_of_line_wavelength_mesh,), [cm]
    ## index array partitioning absorb_prof_1d / absorb_prof_shifted_1d /
    ## wm_cm_1d into per-line segments. Mirrors wMesh.Line_mesh_idxs.
    Line_mesh_idxs: T_ARRAY  # 2d (nLine, 2), [-]

    Jbar: T_ARRAY  # 1d (nLine,), [erg/cm^2/Sr/s]

    ## atom density, if hydrogen, N_total==N_h
    Ntotal: T_FLOAT  # 0d, [/cm^3]
    ## hydrogen density
    Nh: T_FLOAT  # 0d, [/cm^3]
    ## electron density
    Ne: T_FLOAT  # 0d, [/cm^3]
    ## electron temperature
    Te: T_FLOAT  # 0d, [K]

    ## PI (photoionization) intensity used to drive bound-free rates. Returned
    ## here (not just consumed internally) so callers can inspect/analyse the
    ## per-call radiation field. planck(Tr) when se_params.Tr is not None, else
    ## interp(radiation.solar, wMesh.Cont_mesh). 2d (nCont, _N_CONT_MESH).
    PI_intensity: T_ARRAY  # 2d (nCont, _N_CONT_MESH), [erg/cm^2/Sr/cm/s]


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class SE_Params_Container:
    """Statistical Equilibrium (parameter) container for
    - single spatial point
    """

    ## Radiation temperature for the photoionization driver.
    ## None ⇒ use `radiation.solar`; not-None ⇒ use `planck(Tr)`.
    ## `Tr=0.0` is a distinct, valid (non-None) request — coronal-equilibrium
    ## "shut down radiation".
    Tr: T_FLOAT | None = None
    ## Currently unimplemented; `cal_SE_` raises `NotImplementedError` when True.
    doppler_shift_continuum: T_BOOL = False


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
