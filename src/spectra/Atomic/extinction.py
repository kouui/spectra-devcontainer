# -------------------------------------------------------------------------------
# wavelength-base extinction coefficients for bound-bound and bound-free transitions
# -------------------------------------------------------------------------------


import numpy as _numpy

from ..ImportAll import *


@_numpy.vectorize
def bb_extinction_(
    wl: T_FLOAT,
    Bji: T_FLOAT,
    Bij: T_FLOAT,
    Nj: T_FLOAT,
    Ni: T_FLOAT,
    psi: T_FLOAT = 1.0,
    phi: T_FLOAT = 1.0,
) -> T_FLOAT:
    r"""bound-bound (line) extinction coefficient.

    .. math::
        \alpha = \frac{h\nu}{4\pi}\,(N_i B_{ij}\phi - N_j B_{ji}\psi)

    ``Bji`` / ``Bij`` are already wavelength-base Einstein coefficients. With the
    default unit profiles (``psi = phi = 1``) this returns the wavelength-
    **integrated** opacity (dimensionless); passing the normalized profiles
    (``[cm^{-1}]``) yields the spectral extinction ``[cm^{-1}]``. May be negative
    under population inversion (``Nj B_{ji} > Ni B_{ij}``).

    Parameters
    ----------
    wl : T_FLOAT
        wavelength, [cm]
    Bji : T_FLOAT
        Einstein B coefficient for stimulated emission (wavelength base)
    Bij : T_FLOAT
        Einstein B coefficient for absorption (wavelength base)
    Nj : T_FLOAT
        number density of the upper level, [1/cm^3]
    Ni : T_FLOAT
        number density of the lower level, [1/cm^3]
    psi : T_FLOAT
        emission line profile (default 1.0 → integrated), [cm^-1] or [-]
    phi : T_FLOAT
        absorption line profile (default 1.0 → integrated), [cm^-1] or [-]

    Returns
    -------
    T_FLOAT
        line extinction coefficient, [cm^-1] (or dimensionless if integrated)
    """
    nu = CST.c_ / wl
    return CST.h_ * nu / (4.0 * CST.pi_) * (Bij * Ni * phi - Bji * Nj * psi)


@_numpy.vectorize
def bf_extinction_(wl: T_FLOAT, alpha: T_FLOAT, Te: T_FLOAT, Ni: T_FLOAT) -> T_FLOAT:
    r"""bound-free (continuum) extinction coefficient, corrected for stimulated emission.

    .. math::
        \alpha_\nu^{bf} = \sigma_\nu^{bf} N_i (1 - e^{-h\nu/k_B T_e})

    Parameters
    ----------
    wl : T_FLOAT
        wavelength, [cm]
    alpha : T_FLOAT
        photoionization cross section :math:`\sigma_\nu^{bf}`, from
        ``atom.PI.alpha_interp``, [cm^2]
    Te : T_FLOAT
        electron temperature, [K]
    Ni : T_FLOAT
        number density of the lower (bound) level, [1/cm^3]

    Returns
    -------
    T_FLOAT
        b-f extinction coefficient, [cm^-1]
    """
    nu = CST.c_ / wl
    return alpha * Ni * (1.0 - _numpy.exp(-CST.h_ * nu / (CST.k_ * Te)))
