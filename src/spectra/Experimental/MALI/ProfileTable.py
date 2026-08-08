# -------------------------------------------------------------------------------
# per-depth absorption profile table on the frozen global wavelength axis
#
# this is RH's "stage D": the axis never moves; the local physics (Doppler
# width, damping) enters exclusively through the coordinate transform
#     x = (wl - w0) / dopWidth_cm(k)
# i.e. the DIVISION by the local width, where the 0-D code MULTIPLIES the
# dimensionless template by it. every value is exact (the profile is analytic);
# only the quadrature of an integral over the axis can be coarse, and wphi
# records exactly how coarse (the numerical profile norm, analytically 1).
# -------------------------------------------------------------------------------

import numpy as _numpy

from ...ImportAll import *
from ...RadiativeTransfer import Profile as _Profile

# per-iteration-tier kernels are compiled unconditionally; a jitted caller needs
# a jitted callee. production ships voigt_ as numba-vectorize under CFG._IS_JIT
# (njit-callable as is) but as numpy.vectorize otherwise -- compile its raw
# python function (.pyfunc) in that case.
if CFG._IS_JIT:
    _voigt_ = _Profile.voigt_
else:
    _voigt_ = nb_njit(**NB_NJIT_KWGS)(_Profile.voigt_.pyfunc)  # type: ignore[attr-defined]


def line_profile_table_(
    wl_win: T_ARRAY,
    weight_win: T_ARRAY,
    w0: T_FLOAT,
    dopWidth_cm: T_ARRAY,
    adamp: T_ARRAY,
    proftype: T_INT,
) -> T_TUPLE[T_ARRAY, T_ARRAY]:
    """Evaluate one line's normalized absorption profile on its fixed window.

    Input:
        wl_win: (nw,), the line's window of the global axis, [cm]
        weight_win: (nw,), quadrature weights of the window, [cm]
        w0: (,), line center, [cm]
        dopWidth_cm: (ND,), local Doppler width per depth, [cm]
        adamp: (ND,), Voigt damping parameter per depth (ignored for GAUSSIAN)
        proftype: (,), E_ABSORPTION_PROFILE_TYPE value

    Output:
        phi: (nw, ND), wavelength-normalized profile, [cm^-1].
            every value is exact; rows are depths of the SAME columns.
        wphi: (ND,), numerical norm sum_j weight*phi on this window.
            analytically 1; deviates when the window undersamples the line.
            rate integrals must divide by it so that the normalization error
            cancels identically (RH's wphi renormalization).
    """
    nw = wl_win.shape[0]
    ND = dopWidth_cm.shape[0]
    phi = _numpy.empty((nw, ND), dtype=DT_NB_FLOAT)
    wphi = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
    for k in range(ND):
        for iw in range(nw):
            x = (wl_win[iw] - w0) / dopWidth_cm[k]
            if proftype == E_ABSORPTION_PROFILE_TYPE.VOIGT:
                h = _voigt_(adamp[k], x)
            else:
                h = _numpy.exp(-x * x) / CST.sqrtPi_
            phi[iw, k] = h / dopWidth_cm[k]
            wphi[k] += weight_win[iw] * phi[iw, k]
    return phi, wphi


# -------------------------------------------------------------------------------
# numba optimization : per-iteration kernels compile unconditionally
# -------------------------------------------------------------------------------

line_profile_table_ = nb_njit(**NB_NJIT_KWGS)(line_profile_table_)
