# -------------------------------------------------------------------------------
# Voigt damping parameter of a line, ported from RH (broad.c)
#
#     adamp = (Grad + Gamma_vdW + Gamma_Stark,quad + Gamma_Stark,lin) * w0 / (4 pi vbroad)
#
# each Gamma is a full collision rate [s^-1]. RH evaluates them in SI; the
# formulas below keep RH's SI constants (constant.h) and convert the cgs
# inputs at the boundary, so a term-by-term comparison with RH's damping
# output needs no unit bookkeeping beyond that. outputs are per depth and
# depend only on the atmosphere (and, for van der Waals, on the hydrogen
# ground population the caller decides to use): build-once tier, plain numpy.
#
# which term applies to which line -- the caller's job, following RH's
# Damping() dispatch:
#
#     term             radiator      perturber                 RH switch
#     van der Waals    any atom      neutral H(1s) (+ He)      cvdWaals[0] > 0 or cvdWaals[2] > 0
#     quadratic Stark  any atom      electrons (+ ions, 28 u)  cStark != 0
#     linear Stark     hydrogen only electrons                 atom ID is "H "
#
# linear Stark is hydrogen's alone because only its n-levels are degenerate in
# l, so a static field shifts them at first order; every other atom responds
# at second order (quadratic Stark). the van der Waals density is that of the
# PERTURBER, so a Ca II or Mg II line is broadened in proportion to n_H(1s)
# as well.
#
# van der Waals flavours in RH (the atom file's vdWaals field): UNSOLD (ported
# here), BARKLEM (Anstee-Barklem-O'Mara cross sections interpolated from RH's
# Barklem_*.dat tables), ABO_EXPL (the same with sigma, alpha given in the
# file) and PARAMTR (Ridder-Rensbergen fit). RH's ABO tables cover NEUTRAL
# radiators only: on an ion, and on a neutral line whose l pair is not s-p,
# p-d or d-f or whose n* falls outside the table, RH itself falls back to
# UNSOLD at read time, keeping cvdWaals[0], [2] and zeroing [1], [3]. so a
# BARKLEM line of Ca II or Mg II (e.g. RH's CaII_medium) is UNSOLD here as in
# RH; a BARKLEM line of a neutral atom (Ca I, Na I, Fe I), ABO_EXPL and
# PARAMTR are not ported, and a caller must reject them rather than
# substitute UNSOLD. references for the unported flavours, as cited in broad.c:
# BARKLEM / ABO_EXPL -- Anstee & O'Mara 1995, MNRAS 276, 859-866;
# Barklem & O'Mara 1998, MNRAS 300, 863-871. PARAMTR -- DeRidder &
# van Rensbergen 1976, A&A Suppl. 23, 147-165. the references of the ported
# terms are in the function docstrings.
#
# only hydrogen (Z = 1) has been checked against RH; for an ion, E_ion and Z
# follow RH: vdW keys on the upper level's continuum and stage, Stark on the
# lower level's.
# -------------------------------------------------------------------------------

import numpy as _numpy

from ....ImportAll import *

# --- RH constant.h (SI) -------------------------------------------------------
# TODO: rewrite the formulas in CGS and take the constants from spectra.Constants
# (adding the hydrogen polarizability there). kept as RH's SI values for now so
# that adamp matches RH's damping output to rounding; the production constants
# are newer CODATA values (h differs from RH's by ~1e-6 relative), so after the
# migration the RH parity test must loosen from 1e-12 to the ~1e-6 level.
_HPLANCK: T_FLOAT = 6.6260755e-34  # [J s]
_KBOLTZMANN: T_FLOAT = 1.380658e-23  # [J/K]
_AMU: T_FLOAT = 1.6605402e-27  # [kg]
_M_ELECTRON: T_FLOAT = 9.1093897e-31  # [kg]
_Q_ELECTRON: T_FLOAT = 1.60217733e-19  # [C]
_EPSILON_0: T_FLOAT = 8.854187817e-12  # [F/m]
_RBOHR: T_FLOAT = 5.29177349e-11  # [m]
_E_RYDBERG: T_FLOAT = 2.1798741e-18  # [J]
_ABARH: T_FLOAT = 7.42e-41  # polarizability of hydrogen [C m^2 V^-1]
_PI: T_FLOAT = 3.14159265358979
_CM3_TO_M3: T_FLOAT = 1.0e6  # number density cm^-3 -> m^-3
_ERG_TO_J: T_FLOAT = 1.0e-7

# RH atomweights.h; the average ionic perturber weight of RH's Stark()
_WEIGHT_H: T_FLOAT = 1.008
_WEIGHT_HE: T_FLOAT = 4.003
_AVERAGE_ATOMIC_WEIGHT: T_FLOAT = 28.0


def van_der_waals_unsold_(
    Te: T_ARRAY,
    nH_ground: T_ARRAY,
    weight: T_FLOAT,
    E_i: T_FLOAT,
    E_j: T_FLOAT,
    E_ion: T_FLOAT,
    cvdw_H: T_FLOAT,
    cvdw_He: T_FLOAT,
    He_abund: T_FLOAT,
    Z: T_INT = 1,
) -> T_ARRAY:
    """van der Waals broadening, Lindholm theory with Unsold's C6 (RH UNSOLD):

        Gamma = 8.08 * (cvdw_H*vrel_H^0.6 + cvdw_He*A_He*vrel_He^0.6) * C6^0.4 * T^0.3 * n_H(1s)

    the perturber density is the HYDROGEN GROUND population -- which one is
    the caller's decision: RH builds its profiles with the atmosphere file's
    n(1s) and never updates them, but writes its damping output with the
    final populations.

    Input:
        Te, nH_ground: (ND,), [K], [cm^-3]
        weight: atomic weight of the radiator [amu]
        E_i, E_j: lower/upper level energy [erg]
        E_ion: energy of the continuum above the UPPER level [erg]
        cvdw_H, cvdw_He: RH cvdWaals[0], cvdWaals[2] (enhancement factors)
        He_abund: helium abundance relative to hydrogen (number)
        Z: charge seen by the radiating electron (upper level's stage + 1)

    Output:
        Gamma: (ND,), [s^-1]

    Notes
    ------
    port of RH broad.c VanderWaals(), case UNSOLD; references as cited there [1]_ [2]_

    References
    ------------
    .. [1] H. Traving, "Über die Theorie der Druckverbreiterung von
           Spektrallinien", 1960, p. 91-97
    .. [2] D. Mihalas, "Stellar Atmospheres", 2nd ed., 1978,
           p. 282ff and Table 9-1
    """
    # RH has no guard here; with E_ion at or below a level the C6 difference
    # divides by zero or turns negative and **0.4 of a python float goes complex
    if E_ion <= E_j or E_ion <= E_i:
        raise ValueError(f"van der Waals needs bound levels: E_i={E_i}, E_j={E_j}, E_ion={E_ion}")
    four_pi_eps0 = 4.0 * _PI * _EPSILON_0
    vrel35_H = (8.0 * _KBOLTZMANN / (_PI * _AMU * weight) * (1.0 + weight / _WEIGHT_H)) ** 0.3
    vrel35_He = (8.0 * _KBOLTZMANN / (_PI * _AMU * weight) * (1.0 + weight / _WEIGHT_HE)) ** 0.3
    E_R = _E_RYDBERG / _ERG_TO_J  # same unit as the level energies
    deltaR = (E_R / (E_ion - E_j)) ** 2 - (E_R / (E_ion - E_i)) ** 2
    C625 = (
        2.5
        * (_Q_ELECTRON**2 / four_pi_eps0)
        * (_ABARH / four_pi_eps0)
        * 2.0
        * _PI
        * (Z * _RBOHR) ** 2
        / _HPLANCK
        * deltaR
    ) ** 0.4
    cross = 8.08 * (cvdw_H * vrel35_H + cvdw_He * He_abund * vrel35_He) * C625
    return cross * Te**0.3 * (nH_ground * _CM3_TO_M3)


def stark_quadratic_(
    Te: T_ARRAY,
    Ne: T_ARRAY,
    weight: T_FLOAT,
    E_i: T_FLOAT,
    E_j: T_FLOAT,
    E_ion: T_FLOAT,
    cStark: T_FLOAT,
    Z: T_INT = 1,
) -> T_ARRAY:
    """Quadratic Stark broadening by electrons and singly charged ions (RH Stark):

        cStark > 0:  Gamma = 11.37 * (cStark*C4)^(2/3) * vrel^(1/3) * ne,  C4 from Traving
        cStark < 0:  Gamma = |cStark| * ne   (ne in m^-3, RH's convention)

    ions are assumed as many as electrons with mean weight 28 (Gray).

    Input:
        Te, Ne: (ND,), [K], [cm^-3]
        weight, E_i, E_j: see van_der_waals_unsold_
        E_ion, Z: continuum above / stage + 1 of the LOWER level -- RH's
            Stark() keys on line->i while its vdW keys on line->j; the two
            differ only for a line across ionization stages
        cStark: RH line cStark

    Output:
        Gamma: (ND,), [s^-1]

    Notes
    ------
    port of RH broad.c Stark(); references as cited there: C4 from [1]_,
    the 11.37 formula from [2]_, nion = ne from [3]_

    References
    ------------
    .. [1] H. Traving, "Über die Theorie der Druckverbreiterung von
           Spektrallinien", 1960, p. 93
    .. [2] D. Mihalas, "Stellar Atmospheres", 2nd ed., 1978,
           p. 282ff and Table 9-1
    .. [3] D. F. Gray, "The Observation and Analysis of Stellar
           Photospheres", 2nd ed., 1992, p. 216, eq. 11.33
    """
    ne_si = Ne * _CM3_TO_M3
    if cStark < 0.0:
        return abs(cStark) * ne_si
    if (E_ion - E_i) <= 0.0 or (E_ion - E_j) <= 0.0:
        return _numpy.zeros_like(Te)
    C = 8.0 * _KBOLTZMANN / (_PI * _AMU * weight)
    # RH writes 1/6 and 2/3 as the literals 0.16666667 and 0.66666667; kept for parity
    Cm = (1.0 + weight / (_M_ELECTRON / _AMU)) ** 0.16666667 + (1.0 + weight / _AVERAGE_ATOMIC_WEIGHT) ** 0.16666667
    E_R = _E_RYDBERG / (1.0 + _M_ELECTRON / (weight * _AMU)) / _ERG_TO_J
    neff_l = Z * _numpy.sqrt(E_R / (E_ion - E_i))
    neff_u = Z * _numpy.sqrt(E_R / (E_ion - E_j))
    C4 = (
        (_Q_ELECTRON**2 / (4.0 * _PI * _EPSILON_0))
        * _RBOHR
        * (2.0 * _PI * _RBOHR**2 / _HPLANCK)
        / (18.0 * Z**4)
        * ((neff_u * (5.0 * neff_u**2 + 1.0)) ** 2 - (neff_l * (5.0 * neff_l**2 + 1.0)) ** 2)
    )
    cStark23 = 11.37 * (cStark * C4) ** 0.66666667
    vrel = (C * Te) ** 0.16666667 * Cm
    return cStark23 * vrel * ne_si


def stark_linear_hydrogen_(Ne: T_ARRAY, n_lower: T_INT, n_upper: T_INT) -> T_ARRAY:
    """Linear Stark broadening of hydrogen lines by electrons (RH StarkLinear,
    Sutton 1978):

        Gamma = a1 * 0.6 * (n_u^2 - n_l^2) * (ne [m^-3])^(2/3) * (1 cm/1 m)^2,
        a1 = 0.642 for n_u - n_l = 1, else 1

    n_lower/n_upper are PRINCIPAL quantum numbers (RH parses them from the
    level labels).

    Input:
        Ne: (ND,), [cm^-3]

    Output:
        Gamma: (ND,), [s^-1]

    Notes
    ------
    port of RH broad.c StarkLinear(), which cites [1]_. production
    Atomic/Hydrogen.py cites the same paper with 0.255 in place of 0.6.

    References
    ------------
    .. [1] K. Sutton, "Approximate line shapes for hydrogen",
           J. Quant. Spectrosc. Radiat. Transfer, Vol. 20, pp. 333-343, 1978
    """
    a1 = 0.642 if (n_upper - n_lower) == 1 else 1.0
    C = a1 * 0.6 * (n_upper**2 - n_lower**2) * 1.0e-4
    return C * (Ne * _CM3_TO_M3) ** 0.66666667


def damping_parameter_(Gamma: T_ARRAY, w0_cm: T_FLOAT, vbroad_cm: T_ARRAY) -> T_ARRAY:
    """Voigt a = Gamma * w0 / (4 pi vbroad) = Gamma / (4 pi Delta nu_D).

    Input:
        Gamma: (ND,), total damping rate (radiative + collisional) [s^-1]
        w0_cm: line center [cm]; vbroad_cm: (ND,), sqrt(2kT/m + vturb^2) [cm/s]

    Notes
    ------
    port of the last loop of RH broad.c Damping(); RH cites no reference
    """
    return Gamma * w0_cm / (4.0 * _PI * vbroad_cm)


def broadening_velocity_(Te: T_ARRAY, vturb_cm: T_ARRAY, weight: T_FLOAT) -> T_ARRAY:
    """RH's vbroad = sqrt(2kT/(m_amu*weight) + vturb^2), returned in [cm/s].

    Notes
    ------
    the atom's vbroad in RH; RH cites no reference
    """
    v_si = _numpy.sqrt(2.0 * _KBOLTZMANN / (_AMU * weight) * Te + (vturb_cm * 1.0e-2) ** 2)
    return v_si * 1.0e2
