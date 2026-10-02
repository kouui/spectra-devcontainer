"""Unit tests for spectra.Experimental.MALI.v2.Broadening (port of RH broad.c).

Oracles:
- RH's own damping output (FALC, H_6_CRD, DAMPING_OUTPUT) at three depths,
  written here as literals: Ly-alpha and the 5-4 line, every broadening term
  on, inputs exactly as RH saw them (vbroad from T/vturb, van der Waals with
  RH's final n(1s))
- closed forms: Sutton's linear Stark coefficient, cStark < 0 branch,
  Gamma -> a conversion
- scaling laws of each term, which a wrong exponent breaks
"""

import numpy as np
import pytest

from spectra.Experimental.MALI.v2 import Broadening as BR

# H_6_CRD level energies [cm^-1] -> erg with RH's h and c
_E = [
    e * 100.0 * 6.6260755e-34 * 2.99792458e8 * 1e7
    for e in (0.0, 82258.211, 97491.219, 102822.766, 105290.508, 109677.617)
]
_HE_ABUND = 10.0 ** (10.99 - 12.0)

# (T, ne, vturb, n(1s) final) at FALC depths 0, 40, 81 and RH's vbroad [m/s],
# adamp of Ly-alpha (Grad 4.70e8) and of 5->4 (Grad 1.15e7)
_RH = [
    (
        100000.0,
        12518910000.0,
        1068096.0,
        192062.8846890786,
        41997.47014796066,
        0.00010970479372389544,
        0.00023714789574918844,
    ),
    (
        6910.0,
        103189700000.0,
        492604.5,
        1613447688013.6172,
        11758.422796891417,
        0.00040773258524188523,
        0.0025111716522053,
    ),
    (
        9400.0,
        3831726000000000.0,
        180678.69999999998,
        1.2878285394710109e17,
        12583.191603891684,
        0.028184098240086115,
        6.142206708715177,
    ),
]


def _adamp(T, ne, vturb, nH1, i, j, n_i, n_j, Grad):
    T, ne, vturb, nH1 = (np.array([x]) for x in (T, ne, vturb, nH1))
    vb = BR.broadening_velocity_(T, vturb, 1.008)
    g = Grad
    g = g + BR.van_der_waals_unsold_(T, nH1, 1.008, _E[i], _E[j], _E[5], 1.0, 1.0, _HE_ABUND)
    g = g + BR.stark_quadratic_(T, ne, 1.008, _E[i], _E[j], _E[5], 1.0)
    g = g + BR.stark_linear_hydrogen_(ne, n_i, n_j)
    w0_cm = 6.6260755e-34 * 2.99792458e8 / ((_E[j] - _E[i]) * 1e-7) * 100.0
    return vb[0], BR.damping_parameter_(g, w0_cm, vb)[0]


@pytest.mark.parametrize("row", _RH)
def test_matches_rh_damping_output(row):
    T, ne, vturb, nH1, vbroad_rh, a_lya_rh, a_54_rh = row
    vb, a_lya = _adamp(T, ne, vturb, nH1, 0, 1, 1, 2, 4.70e8)
    _, a_54 = _adamp(T, ne, vturb, nH1, 3, 4, 4, 5, 1.15e7)
    assert abs(vb * 1e-2 / vbroad_rh - 1.0) < 1e-12
    assert abs(a_lya / a_lya_rh - 1.0) < 1e-12
    assert abs(a_54 / a_54_rh - 1.0) < 1e-12


def test_linear_stark_sutton_coefficient():
    ne = np.array([1.0e12])  # cm^-3 -> (1e18 m^-3)^(2/3) = 1e12
    # n_u - n_l = 1: a1 = 0.642
    assert np.isclose(BR.stark_linear_hydrogen_(ne, 1, 2)[0], 0.642 * 0.6 * 3 * 1e-4 * 1e12, rtol=1e-6)
    # n_u - n_l = 2: a1 = 1
    assert np.isclose(BR.stark_linear_hydrogen_(ne, 2, 4)[0], 1.0 * 0.6 * 12 * 1e-4 * 1e12, rtol=1e-6)


def test_quadratic_stark_negative_cstark_is_direct():
    ne = np.array([1.0e10, 2.0e12])
    g = BR.stark_quadratic_(np.array([5e3, 6e3]), ne, 1.008, _E[1], _E[2], _E[5], -3.0e-12)
    assert np.allclose(g, 3.0e-12 * ne * 1e6, rtol=1e-14)


def test_quadratic_stark_zero_above_the_continuum():
    g = BR.stark_quadratic_(np.array([5e3]), np.array([1e12]), 1.008, _E[1], _E[5] * 1.01, _E[5], 1.0)
    assert g[0] == 0.0
    g = BR.stark_quadratic_(np.array([5e3]), np.array([1e12]), 1.008, _E[5] * 1.01, _E[5] * 1.02, _E[5], 1.0)
    assert g[0] == 0.0


def test_van_der_waals_charge_scaling():
    # deltaR is independent of Z, so only (Z a0)^2 inside C6^0.4 carries it
    args = (np.array([6e3]), np.array([1e15]), 1.008, _E[1], _E[2], _E[5], 1.0, 1.0, _HE_ABUND)
    g1 = BR.van_der_waals_unsold_(*args, Z=1)
    g2 = BR.van_der_waals_unsold_(*args, Z=2)
    assert np.isclose(g2[0] / g1[0], 2.0**0.8, rtol=1e-12)


def test_scaling_laws():
    T = np.array([4000.0, 16000.0])
    nH = np.array([1.0e15, 1.0e15])
    ne = np.array([1.0e12, 1.0e12])
    v = BR.van_der_waals_unsold_(T, nH, 1.008, _E[1], _E[2], _E[5], 1.0, 1.0, _HE_ABUND)
    assert np.isclose(v[1] / v[0], 4.0**0.3, rtol=1e-12)
    v2 = BR.van_der_waals_unsold_(T[:1], 3.0 * nH[:1], 1.008, _E[1], _E[2], _E[5], 1.0, 1.0, _HE_ABUND)
    assert np.isclose(v2[0] / v[0], 3.0, rtol=1e-12)
    s = BR.stark_quadratic_(T, ne, 1.008, _E[1], _E[2], _E[5], 1.0)
    assert np.isclose(s[1] / s[0], 4.0**0.16666667, rtol=1e-12)
    s2 = BR.stark_quadratic_(T[:1], 5.0 * ne[:1], 1.008, _E[1], _E[2], _E[5], 1.0)
    assert np.isclose(s2[0] / s[0], 5.0, rtol=1e-12)
    lin = BR.stark_linear_hydrogen_(np.array([1e12, 8e12]), 2, 3)
    assert np.isclose(lin[1] / lin[0], 8.0**0.66666667, rtol=1e-12)


def test_damping_parameter_is_gamma_over_4pi_doppler_frequency():
    gamma = np.array([1.0e9])
    w0_cm, v = 6563.0e-8, np.array([1.2e6])
    dnu_D = v / w0_cm
    assert np.isclose(BR.damping_parameter_(gamma, w0_cm, v)[0], gamma[0] / (4.0 * np.pi * dnu_D[0]), rtol=1e-14)


def test_van_der_waals_rejects_unbound_levels():
    with pytest.raises(ValueError, match="bound levels"):
        BR.van_der_waals_unsold_(np.array([6e3]), np.array([1e15]), 1.008, _E[1], _E[5], _E[5], 1.0, 1.0, _HE_ABUND)


def test_van_der_waals_enhancement_factors():
    T, nH = np.array([6e3]), np.array([1e15])
    w = 1.008

    def g(cH, cHe):
        return BR.van_der_waals_unsold_(T, nH, w, _E[1], _E[2], _E[5], cH, cHe, _HE_ABUND)[0]

    assert np.isclose(g(2.0, 0.0) / g(1.0, 0.0), 2.0, rtol=1e-12)
    # helium perturbers: abundance x relative-velocity factor with the He reduced mass
    he_over_h = _HE_ABUND * ((1.0 + w / 4.003) / (1.0 + w / 1.008)) ** 0.3
    assert np.isclose(g(0.0, 1.0) / g(1.0, 0.0), he_over_h, rtol=1e-12)


def test_quadratic_stark_cstark_scaling():
    args = (np.array([6e3]), np.array([1e12]), 1.008, _E[1], _E[2], _E[5])
    s1 = BR.stark_quadratic_(*args, 1.0)
    s3 = BR.stark_quadratic_(*args, 3.0)
    assert np.isclose(s3[0] / s1[0], 3.0**0.66666667, rtol=1e-12)
