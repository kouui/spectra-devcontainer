# -------------------------------------------------------------------------------
# definition of functions to perform statistical equilibrium
# -------------------------------------------------------------------------------

import numpy as _numpy

from ...Atomic import BasicP as _BasicP
from ...ImportAll import *
from ...Math import Integrate as _Integrate
from ...Struct import Atmosphere as _Atmosphere

# from ..SEquil import SELib as _SELib
from ...Struct import Atom as _Atom
from ...Struct import Container as _Container
from ...Struct import WavelengthMesh as _WavelengthMesh


def SE_to_slab_0D_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    wMesh: _WavelengthMesh.Wavelength_Mesh,
    SE_con: _Container.SE_Container,
    depth: T_FLOAT,
) -> _Container.CloudModel_Container:
    """calculate the optical depth and source function for a slab model.

    The output `wl_1D` is the observer-frame wavelength mesh, built from
    `wMesh.Line_mesh` (atom rest frame, doppler-width units) and shifted by
    `atmos.Vd_obs`. The absorption profile in `SE_con` already encodes the
    `Vd_sun` shift (applied inside SE), so the two velocities compose
    naturally without further mixing here.
    """

    nLine = atom.nLine
    Line = atom.Line

    N_ele = atmos.Nh * atom.Abun

    Line_mesh_idxs = wMesh.Line_mesh_idxs
    Line_mesh = wMesh.Line_mesh
    absorb_prof_1d = SE_con.absorb_prof_1d

    Te = atmos.Te
    Vt = atmos.Vt
    Vd_obs = atmos.Vd_obs
    Mass = atom.Mass

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
    for k in range(nLine):
        i1 = Line_mesh_idxs[k, 0]
        i2 = Line_mesh_idxs[k, 1]

        w0 = Line["w0"][k]
        dopWidth_cm = _BasicP.doppler_width_(w0, Te, Vt, Mass)
        # observer-frame wavelength mesh: atom-frame mesh shifted by Vd_obs.
        # +Vd_obs = atom velocity TOWARDS observer (source approaching) → observer
        # sees line center blue-shifted to w0 - w0*Vd_obs/c.
        wl = Line_mesh[i1:i2] * dopWidth_cm + w0 - (w0 * Vd_obs / CST.c_)

        tau = depth * alp0[k] * absorb_prof_1d[i1:i2]

        # TODO: this formula should be converted into an individual function,
        # with background intensity as optional input argument.
        prof = Src[k] * (1.0 - _numpy.exp(-tau[:]))

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
        arr_wl_1D[i1:i2] = wl[:]

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
