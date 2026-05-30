# -------------------------------------------------------------------------------
# definition of functions to perform statistical equilibrium
# -------------------------------------------------------------------------------

import numpy as _numpy

from ...ImportAll import *
from ...Math import Integrate as _Integrate
from ...RadiativeTransfer import CloudModel as _RTCloud
from ...Struct import Atmosphere as _Atmosphere
from ...Struct import Atom as _Atom
from ...Struct import Container as _Container


def SE_to_slab_0D_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    SE_con: _Container.SE_Container,
    depth: T_FLOAT,
    I0: T_ARRAY | None = None,
) -> _Container.CloudModel_Container:
    """calculate the optical depth and source function for a slab model.

    The output `wl_1D` is the observer-frame wavelength mesh, derived from the
    sun-frame `SE_con.wm_cm_1d` (atom-rest-frame line centers in cm) by
    adding the observer-frame Doppler shift `w0*Vd_obs/c`. The cloud
    model reads only what it needs from `SE_con` (no `wMesh` dependency); SE
    already computed both `dopWidth_cm` (baked into `wm_cm_1d`) and the
    unshifted `absorb_prof_1d` (sampled at those same wavelengths).

    `I0` is the optional background intensity entering the slab from behind,
    given as a 2d table `(2, n_wavelength)` like `Radiation.solar` (row 0
    wavelength [cm], row 1 intensity), or `None` for zero background. It is
    interpolated once onto the observer-frame `wl_1D` so it aligns with `tau`.
    """

    nLine = atom.nLine
    Line = atom.Line

    N_ele = atmos.Nh * atom.Abun

    Line_mesh_idxs = SE_con.Line_mesh_idxs
    wm_cm_1d = SE_con.wm_cm_1d
    absorb_prof_1d = SE_con.absorb_prof_1d

    Vd_obs = atmos.Vd_obs

    ## 1. obtain the upper/lower level population for line transitions
    nj: T_ARRAY = SE_con.n_SE[Line["idxJ"][:]]
    ni: T_ARRAY = SE_con.n_SE[Line["idxI"][:]]

    ## 2. compute extinction coefficient alpha
    hv: T_ARRAY = CST.h_ * Line["f0"][:]
    Bij: T_ARRAY = Line["BIJ"][:]
    Bji: T_ARRAY = Line["BJI"][:]
    alp0: T_ARRAY = hv / (4.0 * CST.pi_) * (Bij * ni - Bji * nj) * N_ele

    ## 3. compute line source function
    Aji: T_ARRAY = Line["AJI"][:]
    # Src   : T_ARRAY = ( Aji * nj ) / ( Bij * ni - Bji * nj )
    Src: T_ARRAY = _numpy.zeros_like(Aji)
    line_emissivity: T_ARRAY = Aji * nj
    line_absorption: T_ARRAY = Bij * ni - Bji * nj
    for k in range(nLine):
        if Aji[k] <= 0.0:
            Src[k] = 0.0
            line_emissivity[k] = 0
        else:
            Src[k] = (Aji[k] * nj[k]) / (Bij[k] * ni[k] - Bji[k] * nj[k])

    ## 4. compute optical depth given the thichness of the slab
    ## 5. compute the line profile
    arr_w0 = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    arr_tau_max = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    arr_Ibar = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    arr_prof_1D = _numpy.empty_like(absorb_prof_1d)
    arr_tau_1D = _numpy.empty_like(absorb_prof_1d)
    arr_wl_1D = _numpy.empty_like(absorb_prof_1d)

    # fill the full observer-frame wavelength mesh first, so the background
    # intensity can be interpolated onto it in a single pass below.
    for k in range(nLine):
        i1 = Line_mesh_idxs[k, 0]
        i2 = Line_mesh_idxs[k, 1]
        # observer-frame wavelength mesh: sun-frame atom-rest-frame mesh
        # shifted by +w0*Vd_obs/c. Astronomy radial-velocity convention:
        # +Vd_obs = atom velocity AWAY from observer (source receding) →
        # observer sees line center red-shifted to w0 + w0*Vd_obs/c.
        arr_wl_1D[i1:i2] = wm_cm_1d[i1:i2] + (Line["w0"][k] * Vd_obs / CST.c_)

    # one interpolation of the background onto the full observer-frame mesh;
    # sliced per line below so it stays aligned with each line's tau.
    if I0 is None:
        bg_1D = _numpy.zeros_like(arr_wl_1D)
    else:
        if I0.shape[0] != 2:
            raise ValueError(f"background intensity I0 must have shape (2, n_wavelength), but got {I0.shape}")
        bg_1D = _numpy.interp(arr_wl_1D, I0[0, :], I0[1, :])

    for k in range(nLine):
        i1 = Line_mesh_idxs[k, 0]
        i2 = Line_mesh_idxs[k, 1]

        w0 = Line["w0"][k]
        wl = arr_wl_1D[i1:i2]

        tau = depth * alp0[k] * absorb_prof_1d[i1:i2]

        prof = _RTCloud.emergent_intensity_(Src[k], tau[:], bg_1D[i1:i2])

        # l.tau0[i] = np.max(tau)
        # l.prof[i][:] = S[i] * (1. - np.exp(-tau[:]))
        Ibar = _Integrate.trapze_(prof[:], wl[:])

        # store value
        arr_w0[k] = w0
        # abs() handles population inversion: alp0 < 0 when Bji*nj > Bij*ni,
        # which makes tau negative; .max() on a negative array returns the
        # least-negative value, masking the strongest |tau|.
        arr_tau_max[k] = _numpy.abs(tau[:]).max()
        arr_Ibar[k] = Ibar
        arr_prof_1D[i1:i2] = prof[:]
        arr_tau_1D[i1:i2] = tau[:]

    cloud_con = _Container.CloudModel_Container(
        w0=arr_w0,
        tau_max=arr_tau_max,
        Ibar=arr_Ibar,
        Src=Src,
        tau_1D=arr_tau_1D,
        prof_1D=arr_prof_1D,
        wl_1D=arr_wl_1D,
        Line_mesh_idxs=Line_mesh_idxs.copy(),
        line_emissivity=line_emissivity,
        line_absorption=line_absorption,
    )

    return cloud_con
