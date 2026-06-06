# -------------------------------------------------------------------------------
# definition of struct for storing cloud model result
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

from ...ImportAll import *


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class CloudModel_BB_Container:
    """Cloud Model bound-bound (line) result container for
    - multiple line transitions
    - single spatial point

    The `emissivity` / `absorption` fields are wavelength-**integrated** line
    coefficients (one scalar per line): under CRD (`psi == phi`) the wavelength
    dependence factors entirely into the normalized profile, leaving a single
    integrated coefficient. The spectral coefficients follow by multiplying with
    the profile. `line_emissivity` / `line_absorption` are zero-copy aliases
    (the SAME ndarray object, assigned in the constructor) kept for backward
    compatibility and for visibility under `help_`. Do NOT reassign them
    independently of `emissivity` / `absorption`, or they will silently desync.
    """

    w0: T_ARRAY  # 1d, (nLine,) wavelength, [cm]
    tau_max: T_ARRAY  # 1d, (nLine,), max |optical depth| per line (abs handles population inversion), [-]
    Ibar: T_ARRAY  # 1d, (nLine,), intensity profile integrated over wavelength, [erg/cm^2/Sr/s]
    Src: T_ARRAY  # 1d, (nLine,), source function, [erg/cm^2/Sr/cm/s]
    tau_1D: T_ARRAY  # 1d, (sum_of_line_wavelength_mesh,), optical thickness profile, [-]
    prof_1D: T_ARRAY  # 1d, (sum_of_line_wavelength_mesh,), out intensity profile, [erg/cm^2/Sr/cm/s]
    wl_1D: T_ARRAY  # 1d, (sum_of_line_wavelength_mesh,), doppler shifted wavelength mesh , [cm]
    Line_mesh_idxs: T_ARRAY  # 2d, (nLine,2), [-]
    emissivity: T_ARRAY  # 1d, (nLine,), wavelength-integrated line emissivity, [erg/cm^3/s/Sr]
    line_emissivity: T_ARRAY  # 1d, (nLine,), zero-copy alias of `emissivity` (backward compat)
    absorption: T_ARRAY  # 1d, (nLine,), wavelength-integrated opacity (= emissivity/Src), dimensionless
    line_absorption: T_ARRAY  # 1d, (nLine,), zero-copy alias of `absorption` (backward compat)


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class CloudModel_BF_Container:
    """Cloud Model bound-free (continuum) result container for
    - multiple continuum transitions
    - single spatial point

    Fully wavelength-resolved: unlike the line case, the b-f cross section varies
    continuously across each continuum, so `emissivity` / `absorption` / `Src` /
    `tau` / `prof` are 2d `(nCont, nWavelength)`. No Doppler shift is applied to
    the continuum, so `wl` equals the SE continuum mesh.
    """

    w0: T_ARRAY  # 1d, (nCont,) continuum edge wavelength, [cm]
    Src: T_ARRAY  # 2d, (nCont, nWavelength), source function, [erg/cm^2/Sr/cm/s]
    tau: T_ARRAY  # 2d, (nCont, nWavelength), optical thickness, [-]
    prof: T_ARRAY  # 2d, (nCont, nWavelength), out intensity profile, [erg/cm^2/Sr/cm/s]
    wl: T_ARRAY  # 2d, (nCont, nWavelength), wavelength mesh (no doppler), [cm]
    emissivity: T_ARRAY  # 2d, (nCont, nWavelength), spectral emissivity, [erg/cm^3/s/Sr/cm]
    absorption: T_ARRAY  # 2d, (nCont, nWavelength), extinction (= emissivity/Src), [cm^-1]
