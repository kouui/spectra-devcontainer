# -------------------------------------------------------------------------------
# function/struct for computing optical depth
# -------------------------------------------------------------------------------

import numpy as _numpy

from ..ImportAll import *


def make_tau_(ND: T_INT, e_max: T_FLOAT, relax: T_FLOAT = 0.0):
    """
    make a 1D mesh of log scale from 0 to 1Ee_max with ND grid points.

    Input:
        ND: (,), number of grid points
        e_max: (,), with minvalue of 1Ee_max
        relax: (,), 0 -> 1  logspace -> linspace

    Output:
        Tau: (ND,), optical depth,
    """
    if relax < 0.0:
        relax = 0
    if relax > 1.0:
        relax = 1.0

    e_min = -4.0
    tau = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    mid = e_max - _numpy.log10(2.0)
    tau[: ND // 2 + 1] = (
        _numpy.logspace(e_min, mid, ND // 2 + 1) * (1 - relax)
        + _numpy.linspace(10.0**e_min, 10.0**mid, ND // 2 + 1) * relax
    )

    tau[0] = 0
    tau[ND // 2 + 1 :] = (-tau[: ND // 2 + 1][::-1] + 2 * tau[ND // 2])[1:]

    return tau


def z_to_dtau_(z: T_ARRAY, alpha: T_ARRAY):
    r"""[summary]

    Parameters
    ----------
    z : T_ARRAY, [cm]
        1D depth with z[0] as the surface and z[i] < z[i+1]
    alpha : T_ARRAY, [cm^{-1}]
        extinction coefficient

    Returns
    --------
    dtau : T_ARRAY, [-]
        difference of optical depth
    """
    nZ = z.shape[0]
    _dtau = _numpy.zeros(nZ, dtype=DT_NB_FLOAT)


def z_to_tau_(Z: T_ARRAY, xi: T_ARRAY) -> T_ARRAY:
    r"""given height `Z` and opacity `xi`, calculate optical depth tau

    Trapezoidal integration of `xi` from the surface inwards. Note the
    ordering is the reverse of `z_to_dtau_`.

    Parameters
    ----------
    Z : T_ARRAY
        height with direction : [interior, surface]
        [:math:`cm`]
    xi : T_ARRAY
        opacity
        [:math:`cm^{-1}`]

    Returns
    -------
    T_ARRAY
        optical depth, tau[-1] at the surface
        [-]

    Notes
    -----
    The surface point is not zero: the layer above the topmost grid point is
    assumed to be as thick as half the last interval,
    ``tau[-1] = xi[-1] * (Z[-1] - Z[-2]) / 2``.
    """

    nZ = Z.shape[0]
    tau = _numpy.zeros(nZ, dtype=DT_NB_FLOAT)

    tau[-1] = xi[-1] * (Z[-1] - Z[-2]) * 0.5
    for i in range(nZ - 2, -1, -1):
        tau[i] = tau[i + 1] + (Z[i + 1] - Z[i]) * (xi[i] + xi[i + 1]) * 0.5

    return tau
