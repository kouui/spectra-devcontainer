# -------------------------------------------------------------------------------
# background continuum opacity for the MALI line windows
#
# ported from RH (hydrogen.c, thomson.c, rayleigh.c, chemequil.c) so that a comparison run
# against RH sees the same background physics. everything is CGS and
# wavelength-base; every source assembled by background_chi_ is a THERMAL
# absorber, so the caller pairs
#     eta_bg = chi_bg * B_lambda(Te)
# which is exact for the stimulated-emission-corrected chi returned by each
# function (Kirchhoff). Thomson scattering is deliberately NOT part of that
# sum: its emissivity is sigma*ne*J, and pairing it with B turns it into a
# thermal emitter -- at a 1e5 K transition-region top that is a phantom EUV
# source inside the Lyman line windows, strong enough to displace the whole
# FALC hydrogen solution (measured: 4x at the temperature minimum, 6.6x for
# n = 4,5 in the transition region, both vanishing once it is removed).
# thomson_ and rayleigh_ feed the coherent-scattering slot of the sweep
# (chi += sigma, eta += sigma * J of the previous iteration) instead.
#
# build-once tier: evaluated per (window column, depth) at setup, so plain
# interpreted numpy is fine.
# -------------------------------------------------------------------------------

import numpy as _numpy

from ....ImportAll import *

# --- constants derived once, mirroring the RH expressions in CGS ------------

# Thomson cross section, [cm^2]
_SIGMA_THOMSON: T_FLOAT = (8.0 * CST.pi_ / 3.0) * (CST.e_**2 / (CST.me_ * CST.c_**2)) ** 2

# Einstein A -> absorption oscillator strength: f = A * (gj/gi) * w0^2 / C, [cm^2 s^-1]
_C_A_TO_F: T_FLOAT = 8.0 * CST.pi_**2 * CST.e_**2 / (CST.me_ * CST.c_)

# hydrogenic b-f cross section scale (Mihalas 1978 p.99), [cm^2]
_SIGMA0_H_BF: T_FLOAT = (
    32.0 / (3.0 * _numpy.sqrt(3.0)) * CST.e_**2 / (CST.me_ * CST.c_) * CST.h_ / (2.0 * CST.E_Rydberg_)
)

# H free-free coefficient (Mihalas 1978 p.101): chi = C_FF/sqrt(T)/nu^3 * ne*np*stim*gff
_C_H_FF: T_FLOAT = (
    4.0
    / 3.0
    * _numpy.sqrt(2.0 * CST.pi_ / (3.0 * CST.k_))
    * (CST.e_**2 / _numpy.sqrt(CST.me_)) ** 3
    / (CST.h_ * CST.c_)
)

# H-minus ionization energy 0.754 eV, [erg]
_E_ION_HMIN: T_FLOAT = 0.754 * CST.eV2erg_

# Geltman (1962) H-minus b-f cross section, wavelength [nm], alpha [1e-17 cm^2]
_HMBF_LAMBDA = _numpy.array(
    [
        0.0,
        50.0,
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
        350.0,
        400.0,
        450.0,
        500.0,
        550.0,
        600.0,
        650.0,
        700.0,
        750.0,
        800.0,
        850.0,
        900.0,
        950.0,
        1000.0,
        1050.0,
        1100.0,
        1150.0,
        1200.0,
        1250.0,
        1300.0,
        1350.0,
        1400.0,
        1450.0,
        1500.0,
        1550.0,
        1600.0,
        1641.9,
    ]
)
_HMBF_ALPHA = _numpy.array(
    [
        0.0,
        0.15,
        0.33,
        0.57,
        0.85,
        1.17,
        1.52,
        1.89,
        2.23,
        2.55,
        2.84,
        3.11,
        3.35,
        3.56,
        3.71,
        3.83,
        3.92,
        3.95,
        3.93,
        3.85,
        3.73,
        3.58,
        3.38,
        3.14,
        2.85,
        2.54,
        2.20,
        1.83,
        1.46,
        1.06,
        0.71,
        0.40,
        0.17,
        0.0,
    ]
)

# Stilley & Callaway (1970) H-minus f-f: kappa [1e-29 m^5/J -> converted below],
# rows = wavelength [nm], cols = theta = 5040/T
_HMFF_LAMBDA = _numpy.array(
    [
        0.0,
        303.8,
        455.6,
        506.3,
        569.5,
        650.9,
        759.4,
        911.3,
        1013.0,
        1139.0,
        1302.0,
        1519.0,
        1823.0,
        2278.0,
        3038.0,
        4556.0,
        9113.0,
    ]
)
_HMFF_THETA = _numpy.array(
    [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
    ]
)
_HMFF_KAPPA = _numpy.array(
    [
        [
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
            0.00e00,
        ],
        [
            3.44e-02,
            4.18e-02,
            4.91e-02,
            5.65e-02,
            6.39e-02,
            7.13e-02,
            7.87e-02,
            8.62e-02,
            9.36e-02,
            1.01e-01,
            1.08e-01,
            1.16e-01,
            1.23e-01,
            1.30e-01,
            1.38e-01,
            1.45e-01,
        ],
        [
            7.80e-02,
            9.41e-02,
            1.10e-01,
            1.25e-01,
            1.40e-01,
            1.56e-01,
            1.71e-01,
            1.86e-01,
            2.01e-01,
            2.16e-01,
            2.31e-01,
            2.45e-01,
            2.60e-01,
            2.75e-01,
            2.89e-01,
            3.03e-01,
        ],
        [
            9.59e-02,
            1.16e-01,
            1.35e-01,
            1.53e-01,
            1.72e-01,
            1.90e-01,
            2.08e-01,
            2.25e-01,
            2.43e-01,
            2.61e-01,
            2.78e-01,
            2.96e-01,
            3.13e-01,
            3.30e-01,
            3.47e-01,
            3.64e-01,
        ],
        [
            1.21e-01,
            1.45e-01,
            1.69e-01,
            1.92e-01,
            2.14e-01,
            2.36e-01,
            2.58e-01,
            2.80e-01,
            3.01e-01,
            3.22e-01,
            3.43e-01,
            3.64e-01,
            3.85e-01,
            4.06e-01,
            4.26e-01,
            4.46e-01,
        ],
        [
            1.56e-01,
            1.88e-01,
            2.18e-01,
            2.47e-01,
            2.76e-01,
            3.03e-01,
            3.31e-01,
            3.57e-01,
            3.84e-01,
            4.10e-01,
            4.36e-01,
            4.62e-01,
            4.87e-01,
            5.12e-01,
            5.37e-01,
            5.62e-01,
        ],
        [
            2.10e-01,
            2.53e-01,
            2.93e-01,
            3.32e-01,
            3.69e-01,
            4.06e-01,
            4.41e-01,
            4.75e-01,
            5.09e-01,
            5.43e-01,
            5.76e-01,
            6.08e-01,
            6.40e-01,
            6.72e-01,
            7.03e-01,
            7.34e-01,
        ],
        [
            2.98e-01,
            3.59e-01,
            4.16e-01,
            4.70e-01,
            5.22e-01,
            5.73e-01,
            6.21e-01,
            6.68e-01,
            7.15e-01,
            7.60e-01,
            8.04e-01,
            8.47e-01,
            8.90e-01,
            9.32e-01,
            9.73e-01,
            1.01e00,
        ],
        [
            3.65e-01,
            4.39e-01,
            5.09e-01,
            5.75e-01,
            6.39e-01,
            7.00e-01,
            7.58e-01,
            8.15e-01,
            8.71e-01,
            9.25e-01,
            9.77e-01,
            1.03e00,
            1.08e00,
            1.13e00,
            1.18e00,
            1.23e00,
        ],
        [
            4.58e-01,
            5.50e-01,
            6.37e-01,
            7.21e-01,
            8.00e-01,
            8.76e-01,
            9.49e-01,
            1.02e00,
            1.09e00,
            1.15e00,
            1.22e00,
            1.28e00,
            1.34e00,
            1.40e00,
            1.46e00,
            1.52e00,
        ],
        [
            5.92e-01,
            7.11e-01,
            8.24e-01,
            9.31e-01,
            1.03e00,
            1.13e00,
            1.23e00,
            1.32e00,
            1.40e00,
            1.49e00,
            1.57e00,
            1.65e00,
            1.73e00,
            1.80e00,
            1.88e00,
            1.95e00,
        ],
        [
            7.98e-01,
            9.58e-01,
            1.11e00,
            1.25e00,
            1.39e00,
            1.52e00,
            1.65e00,
            1.77e00,
            1.89e00,
            2.00e00,
            2.11e00,
            2.21e00,
            2.32e00,
            2.42e00,
            2.51e00,
            2.61e00,
        ],
        [
            1.14e00,
            1.36e00,
            1.58e00,
            1.78e00,
            1.98e00,
            2.17e00,
            2.34e00,
            2.52e00,
            2.68e00,
            2.84e00,
            3.00e00,
            3.15e00,
            3.29e00,
            3.43e00,
            3.57e00,
            3.70e00,
        ],
        [
            1.77e00,
            2.11e00,
            2.44e00,
            2.75e00,
            3.05e00,
            3.34e00,
            3.62e00,
            3.89e00,
            4.14e00,
            4.39e00,
            4.63e00,
            4.86e00,
            5.08e00,
            5.30e00,
            5.51e00,
            5.71e00,
        ],
        [
            3.10e00,
            3.71e00,
            4.29e00,
            4.84e00,
            5.37e00,
            5.87e00,
            6.36e00,
            6.83e00,
            7.28e00,
            7.72e00,
            8.14e00,
            8.55e00,
            8.95e00,
            9.33e00,
            9.71e00,
            1.01e01,
        ],
        [
            6.92e00,
            8.27e00,
            9.56e00,
            1.08e01,
            1.19e01,
            1.31e01,
            1.42e01,
            1.52e01,
            1.62e01,
            1.72e01,
            1.82e01,
            1.91e01,
            2.00e01,
            2.09e01,
            2.17e01,
            2.25e01,
        ],
        [
            2.75e01,
            3.29e01,
            3.80e01,
            4.28e01,
            4.75e01,
            5.19e01,
            5.62e01,
            6.04e01,
            6.45e01,
            6.84e01,
            7.23e01,
            7.60e01,
            7.97e01,
            8.32e01,
            8.67e01,
            9.01e01,
        ],
    ]
)


def gaunt_bf_(wl_cm: T_FLOAT, n_eff: T_FLOAT, charge: T_INT) -> T_FLOAT:
    """Seaton (1960) bound-free Gaunt factor."""
    x = CST.h_ * CST.c_ / wl_cm / (CST.E_Rydberg_ * charge * charge)
    x3 = x ** (1.0 / 3.0)
    nsqx = 1.0 / (n_eff * n_eff * x)
    return 1.0 + 0.1728 * x3 * (1.0 - 2.0 * nsqx) - 0.0496 * x3 * x3 * (1.0 - (1.0 - nsqx) * (2.0 / 3.0) * nsqx)


def gaunt_ff_(wl_cm: T_FLOAT, charge: T_INT, Te: T_VEC_FA) -> T_VEC_FA:
    """Seaton (1960) free-free Gaunt factor, clipped at 1 (RH convention)."""
    x = CST.h_ * CST.c_ / wl_cm / (CST.E_Rydberg_ * charge * charge)
    x3 = x ** (1.0 / 3.0)
    y = 2.0 * wl_cm * CST.k_ * _numpy.asarray(Te) / (CST.h_ * CST.c_)
    g = 1.0 + 0.1728 * x3 * (1.0 + y) - 0.0496 * x3 * x3 * (1.0 + (1.0 + y) * y / 3.0)
    return _numpy.maximum(g, 1.0)


def phi_hmin_(Te: T_VEC_FA) -> T_VEC_FA:
    """Saha-like H-minus formation coefficient: nHmin = ne * nH_neutral * PhiHmin, [cm^3]."""
    CI = CST.h_ * CST.h_ / (2.0 * CST.pi_ * CST.me_ * CST.k_)
    Te = _numpy.asarray(Te)
    return 0.25 * (CI / Te) ** 1.5 * _numpy.exp(_E_ION_HMIN / (CST.k_ * Te))


def hminus_bf_(wl_cm: T_FLOAT, Te: T_ARRAY, Ne: T_ARRAY, nH_neutral: T_ARRAY) -> T_ARRAY:
    """H-minus bound-free extinction (stimulated-emission corrected), [cm^-1].

    Geltman (1962) cross sections, linearly interpolated (RH splines the same
    34-point table; the difference is at the percent level).
    """
    wl_nm = wl_cm * 1.0e7
    if wl_nm <= _HMBF_LAMBDA[0] or wl_nm >= _HMBF_LAMBDA[-1]:
        return _numpy.zeros_like(Te)
    alpha = _numpy.interp(wl_nm, _HMBF_LAMBDA, _HMBF_ALPHA) * 1.0e-17  # [cm^2]
    stim = _numpy.exp(-CST.h_ * CST.c_ / (wl_cm * CST.k_ * Te))
    nHmin = Ne * nH_neutral * phi_hmin_(Te)
    return nHmin * (1.0 - stim) * alpha


def hminus_ff_(wl_cm: T_FLOAT, Te: T_ARRAY, Ne: T_ARRAY, nH_neutral: T_ARRAY) -> T_ARRAY:
    """H-minus free-free extinction, [cm^-1].

    Stilley & Callaway (1970) table, bilinear in (wavelength, theta = 5040/T);
    kappa tabulated in 1e-29 m^5/J = 1e-26 cm^4/dyn: chi = nH * pe * kappa.
    """
    wl_nm = wl_cm * 1.0e7
    if wl_nm >= _HMFF_LAMBDA[-1]:  # John (1988) long-wavelength branch not ported
        raise ValueError(f"hminus_ff_: wavelength {wl_nm:.1f} nm beyond table")
    theta = _numpy.clip(5040.0 / Te, _HMFF_THETA[0], _HMFF_THETA[-1])
    il = int(_numpy.searchsorted(_HMFF_LAMBDA, wl_nm) - 1)
    fl = (wl_nm - _HMFF_LAMBDA[il]) / (_HMFF_LAMBDA[il + 1] - _HMFF_LAMBDA[il])
    kl = (1.0 - fl) * _HMFF_KAPPA[il, :] + fl * _HMFF_KAPPA[il + 1, :]
    kappa = _numpy.interp(theta, _HMFF_THETA, kl) * 1.0e-26  # [cm^4 dyn^-1]
    pe = Ne * CST.k_ * Te  # [dyn cm^-2]
    return nH_neutral * pe * kappa


def hydrogen_ff_(wl_cm: T_FLOAT, Te: T_ARRAY, Ne: T_ARRAY, Np: T_ARRAY) -> T_ARRAY:
    """H free-free extinction (stimulated-emission corrected), [cm^-1]."""
    nu3 = (wl_cm / CST.c_) ** 3
    stim = 1.0 - _numpy.exp(-CST.h_ * CST.c_ / (wl_cm * CST.k_ * Te))
    return _C_H_FF / _numpy.sqrt(Te) * nu3 * Ne * Np * stim * gaunt_ff_(wl_cm, 1, Te)


def hydrogen_bf_(
    wl_cm: T_FLOAT,
    Te: T_ARRAY,
    n_level: T_ARRAY,
    erg_level: T_FLOAT,
    erg_cont: T_FLOAT,
) -> T_ARRAY:
    """Hydrogenic bound-free extinction of one level (stim. corrected), [cm^-1].

    LTE-consistent form chi = sigma * (1 - exp(-h nu/kT)) * n_i, so the thermal
    eta = chi * B pairing holds; the NLTE emission correction RH applies for
    its non-active hydrogen is dropped (this backgrounds OTHER species' lines).

    Input:
        n_level: (ND,), population of the lower level, [cm^-3]
        erg_level, erg_cont: level and continuum-edge energies, [erg]
    """
    chi_ion = erg_cont - erg_level
    wl_edge = CST.h_ * CST.c_ / chi_ion
    if wl_cm > wl_edge:
        return _numpy.zeros_like(Te)
    n_eff = _numpy.sqrt(CST.E_Rydberg_ / chi_ion)
    sigma = _SIGMA0_H_BF * n_eff * gaunt_bf_(wl_cm, float(n_eff), 1) * (wl_cm / wl_edge) ** 3
    stim = 1.0 - _numpy.exp(-CST.h_ * CST.c_ / (wl_cm * CST.k_ * Te))
    return sigma * stim * n_level


def thomson_(Ne: T_ARRAY) -> T_ARRAY:
    """Thomson scattering extinction, [cm^-1] (see module note on its emissivity)."""
    return _SIGMA_THOMSON * Ne


def rayleigh_(
    wl_cm: T_ARRAY,
    w0: T_ARRAY,
    Aji: T_ARRAY,
    gi: T_ARRAY,
    gj: T_ARRAY,
    wl_red: T_ARRAY,
) -> T_ARRAY:
    """Rayleigh scattering cross section per ground-state atom, [cm^2]
    (RH's rayleigh.c, Mihalas 1978 p.106):

        sigma(wl) = sigma_T * sum_lines f_ij * (1 / ((wl/w0)^2 - 1))^2

    summed over the lines from the ground level whose red window edge lies
    at or blueward of wl: inside a line's own window the line itself carries
    the opacity, so the far-wing Rayleigh tail starts only at that edge. RH
    draws the edge at w0*(1 + qwing*vmicro_char/c); a mesh anchored with
    xi_ref = vmicro_char has exactly that red edge, so the caller passes the
    mesh's own window edges. the edge column itself COUNTS: RH tests
    lambda > lambda_red, but its last window point is generated on a
    different rounding path and lands a few ulp above lambda_red, so RH
    scatters there (measured on FALC: J at the Ly-alpha edge column agrees
    with RH to 3% inclusive, 11% exclusive). no depth dependence: the
    caller multiplies by the ground level population -- RH uses the
    atmosphere file's hydrogen populations, set once before iterating and
    never updated, so a comparison run passes those, not the NLTE solution.

    Input:
        wl_cm: (Nspect,), [cm]
        w0, Aji, gi, gj, wl_red: (nGroundLine,), line center, Einstein A,
            statistical weights, red window edge, [cm], [s^-1], -, -, [cm]

    Output:
        sigma: (Nspect,), [cm^2]
    """
    fomega = _numpy.zeros_like(wl_cm)
    for kL in range(w0.shape[0]):
        f = Aji[kL] * (gj[kL] / gi[kL]) * w0[kL] ** 2 / _C_A_TO_F
        mask = wl_cm >= wl_red[kL]
        fomega[mask] += f / ((wl_cm[mask] / w0[kL]) ** 2 - 1.0) ** 2
    return _SIGMA_THOMSON * fomega


def background_chi_(
    wl_cm: T_FLOAT,
    Te: T_ARRAY,
    Ne: T_ARRAY,
    nH_level: T_ARRAY,
    Np: T_ARRAY,
    erg_level: T_ARRAY,
    erg_cont: T_FLOAT,
) -> T_ARRAY:
    """Total background extinction at one wavelength, [cm^-1].

    H-minus b-f/f-f + H f-f + H b-f (every level whose edge lies redward of
    wl_cm). eta_bg = chi * planck_cm_ is the caller's job, which is why
    Thomson scattering is excluded (see the module note).

    Input:
        nH_level: (ND, nLevelBound), bound-level H populations, [cm^-3]
        Np: (ND,), proton density; erg_level: (nLevelBound,); erg_cont: [erg]
    """
    nH_neutral = nH_level.sum(axis=1)
    chi = hminus_bf_(wl_cm, Te, Ne, nH_neutral)
    chi += hminus_ff_(wl_cm, Te, Ne, nH_neutral)
    chi += hydrogen_ff_(wl_cm, Te, Ne, Np)
    for i in range(nH_level.shape[1]):
        chi += hydrogen_bf_(wl_cm, Te, nH_level[:, i], float(erg_level[i]), erg_cont)
    return chi
