# -------------------------------------------------------------------------------
# wavelength-base emission coefficients for bound-bound and bound-free transitions
# -------------------------------------------------------------------------------


import numpy as _numpy

from ..ImportAll import *


@_numpy.vectorize
def bb_emissivity_(wl: T_FLOAT, Aji: T_FLOAT, Nj: T_FLOAT) -> T_FLOAT:
    r"""bound-bound (line) emission coefficient.

    .. math:: j = \frac{h\nu}{4\pi} A_{ji} N_j

    Returned as the wavelength-**integrated** line emissivity (no profile factor);
    the spectral emissivity :math:`j_\lambda` follows by multiplying with the
    normalized emission profile :math:`\psi_\lambda\ [\mathrm{cm^{-1}}]`. With the
    profile, units become ``erg/cm^3/s/Sr/cm``.

    Parameters
    ----------
    wl : T_FLOAT
        wavelength, [cm]
    Aji : T_FLOAT
        Einstein A coefficient, [1/s]
    Nj : T_FLOAT
        number density of the upper level, [1/cm^3]

    Returns
    -------
    T_FLOAT
        wavelength-integrated line emissivity, [erg/cm^3/s/Sr]
    """
    nu = CST.c_ / wl
    return CST.h_ * nu / (4.0 * CST.pi_) * Aji * Nj


@_numpy.vectorize
def bf_emissivity_(
    wl: T_FLOAT,
    alpha: T_FLOAT,
    Te: T_FLOAT,
    Ne: T_FLOAT,
    Ni1: T_FLOAT,
    gi: T_FLOAT,
    gk: T_FLOAT,
    chi: T_FLOAT,
) -> T_FLOAT:
    r"""bound-free (continuum) spectral emission coefficient :math:`j_\lambda`.

    Recombination emissivity. For readability and consistency with the formula we
    work in frequency, build :math:`j_\nu`, then convert to wavelength base via
    :math:`j_\lambda = j_\nu\, c/\lambda^2`.

    .. math::
        j_\nu = \frac{h\nu}{4\pi}\,\sigma_\nu^{fb}\, f(\varepsilon)v\, N_{I+1} N_e\,
                \frac{d\varepsilon}{d\nu}

    with the Milne relation, the Maxwellian recombination flux, and the
    photoelectron energy

    .. math::
        \sigma_\nu^{fb} = \frac{g_i}{2 g_k}\frac{(h\nu)^2}{m_e c^2\,\varepsilon}
                          \sigma_\nu^{bf}, \quad
        f(\varepsilon)v = 8\pi m_e\,\varepsilon\,(2\pi m_e k_B T_e)^{-3/2}
                          e^{-\varepsilon/k_B T_e}, \quad
        \varepsilon = h\nu - \chi, \quad
        \frac{d\varepsilon}{d\nu} = h .

    The product :math:`\sigma_\nu^{fb} f(\varepsilon)v` is evaluated with
    :math:`\varepsilon` (and :math:`m_e`) cancelled analytically, so the result is
    finite at the continuum edge (:math:`\varepsilon \to 0`) without shifting the
    wavelength mesh.

    Parameters
    ----------
    wl : T_FLOAT
        wavelength, [cm]
    alpha : T_FLOAT
        photoionization cross section :math:`\sigma_\nu^{bf}`, from
        ``atom.PI.alpha_interp``, [cm^2]
    Te : T_FLOAT
        electron temperature, [K]
    Ne : T_FLOAT
        electron number density, [1/cm^3]
    Ni1 : T_FLOAT
        number density of the ground state of the ion one stage higher than the
        ion that produces the b-f continuum, [1/cm^3]
    gi : T_FLOAT
        statistical weight of the lower (bound) level, [-]
    gk : T_FLOAT
        statistical weight of the upper (ion) level, [-]
    chi : T_FLOAT
        ionization energy from the lower (bound) level, :math:`\chi = h f_0`
        (continuum edge), [erg]

    Returns
    -------
    T_FLOAT
        spectral b-f emissivity :math:`j_\lambda`, [erg/cm^3/s/Sr/cm]
    """
    nu = CST.c_ / wl
    eps = CST.h_ * nu - chi
    # below threshold (wl > edge, eps < 0) there is no recombination continuum;
    # return 0 rather than the analytically-continued exp(-eps/kTe) which would
    # blow up (and give 0*inf=nan when alpha==0). Production meshes are
    # edge-first and stay at/below the edge, so this only guards misuse.
    if eps < 0.0:
        return 0.0
    # sigma_fb * f(eps)v with the divergent factors cancelled analytically:
    #   - eps: 1/eps (sigma_fb) * eps (Maxwell flux) -> 1  (this is what removes
    #     the 0/0 at the edge; the cancelled form stays finite there)
    #   - one net m_e: 1/m_e (sigma_fb) * the standalone m_e in the flux prefactor
    #     (8*pi*m_e) -> 1; the m_e inside (2*pi*m_e*k*Te)^-1.5 below SURVIVES.
    sfb_fv = (
        (gi / (2.0 * gk))
        * (CST.h_ * nu) ** 2
        / CST.c_**2
        * alpha
        * 8.0
        * CST.pi_
        * (2.0 * CST.pi_ * CST.me_ * CST.k_ * Te) ** (-1.5)
        * _numpy.exp(-eps / (CST.k_ * Te))
    )
    # trailing CST.h_ is d(epsilon)/d(nu); do NOT drop it.
    jnu = CST.h_ * nu / (4.0 * CST.pi_) * sfb_fv * Ni1 * Ne * CST.h_
    return jnu * CST.c_ / wl**2
