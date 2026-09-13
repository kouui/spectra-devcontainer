"""Unit tests for spectra.Experimental.MALI.v2.Background: the RH-ported
background opacities (H-minus, H f-f, H b-f, Thomson) as thermal absorbers
plus scattering kept separate."""

import numpy as np

from spectra import Constants as CST
from spectra.Experimental.MALI.v2 import Background as BG


def test_thomson():
    Ne = np.array([1.0e10, 2.0e12])
    chi = BG.thomson_(Ne)
    assert np.allclose(chi / Ne, 6.652e-25, rtol=1e-3)


def test_hydrogen_bf_edge():
    Te = np.array([6000.0])
    n1 = np.array([1.0])
    erg1 = 0.0
    erg_c = CST.E_Rydberg_H_
    wl_edge = CST.h_ * CST.c_ / erg_c
    chi = BG.hydrogen_bf_(wl_edge * 0.9999, Te, n1, erg1, erg_c)
    stim = 1.0 - np.exp(-CST.h_ * CST.c_ / (wl_edge * CST.k_ * Te[0]))
    sigma = chi[0] / stim
    # Seaton's approximate Gaunt factor puts the edge ~2.4% below the exact one
    assert abs(sigma / 6.30e-18 - 1.0) < 0.03
    assert BG.hydrogen_bf_(wl_edge * 1.01, Te, n1, erg1, erg_c)[0] == 0.0


def test_hminus_bf_table_peak():
    Te = np.array([6000.0])
    Ne = np.array([1.0])
    nH = np.array([1.0])
    chi = BG.hminus_bf_(850.0e-7, Te, Ne, nH)
    stim = np.exp(-CST.h_ * CST.c_ / (850.0e-7 * CST.k_ * Te[0]))
    alpha = chi[0] / ((1.0 - stim) * np.asarray(BG.phi_hmin_(Te))[0])
    assert abs(alpha / 3.95e-17 - 1.0) < 0.01
    assert BG.hminus_bf_(1700.0e-7, Te, Ne, nH)[0] == 0.0  # beyond the table


def test_background_assembly_positive():
    Te = np.array([5000.0, 8000.0, 20000.0])
    Ne = np.array([1.0e13, 1.0e11, 1.0e10])
    nH = np.tile(np.array([1.0e15, 1.0e8, 1.0e7, 1.0e6, 1.0e6]), (3, 1))
    Np = np.array([1.0e11, 1.0e11, 1.0e10])
    erg = np.array([0.0, 0.75, 0.889, 0.9375, 0.96]) * CST.E_Rydberg_H_
    chi = BG.background_chi_(5000.0e-8, Te, Ne, nH, Np, erg, CST.E_Rydberg_H_)
    assert np.all(chi > 0.0)
    assert chi[0] > 10.0 * BG.thomson_(Ne)[0]
    # Thomson is a scatterer, never part of the thermal-absorber sum
    Te_hot, Ne_hot = np.array([1.0e5]), np.array([1.0e10])
    nH_hot = np.array([[1.0e4, 1.0, 1.0, 1.0, 1.0]])
    chi_hot = BG.background_chi_(1.0e-5, Te_hot, Ne_hot, nH_hot, Ne_hot, erg, CST.E_Rydberg_H_)
    assert chi_hot[0] < 0.01 * BG.thomson_(Ne_hot)[0]


# hydrogen Lyman alpha..delta: line centre [cm], Einstein A [s^-1], g_upper
_LYMAN_W0 = np.array([121.567, 102.572, 97.254, 94.974]) * 1.0e-7
_LYMAN_AJI = np.array([4.6986e8, 5.5751e7, 1.2785e7, 4.1250e6])
_LYMAN_GJ = np.array([8.0, 18.0, 32.0, 50.0])
_LYMAN_GI = np.full(4, 2.0)
_LYMAN_RED = _LYMAN_W0 * (1.0 + 10.0 * 3.0e5 / CST.c_)  # window edge at 10 x 3 km/s


def test_rayleigh_oscillator_strength_convention():
    # sigma_T * f(Ly-alpha) * (1/((wl/w0)^2 - 1))^2 with f = 0.4162: the A -> f
    # conversion must reproduce the tabulated oscillator strength
    wl = np.array([200.0e-7])
    sigma = BG.rayleigh_(wl, _LYMAN_W0[:1], _LYMAN_AJI[:1], _LYMAN_GI[:1], _LYMAN_GJ[:1], _LYMAN_RED[:1])
    f = sigma[0] / BG._SIGMA_THOMSON * ((wl[0] / _LYMAN_W0[0]) ** 2 - 1.0) ** 2
    assert abs(f / 0.4162 - 1.0) < 1e-3


def test_rayleigh_hydrogen_154nm():
    # value from RH's rayleigh.c on the same four lines (FALC run)
    sigma = BG.rayleigh_(np.array([153.8e-7]), _LYMAN_W0, _LYMAN_AJI, _LYMAN_GI, _LYMAN_GJ, _LYMAN_RED)
    assert abs(sigma[0] / 8.14e-25 - 1.0) < 1e-2


def test_rayleigh_starts_at_each_line_red_edge():
    wl = np.array([121.0e-7, _LYMAN_RED[0], 110.0e-7, 1.0e-7])
    sigma = BG.rayleigh_(wl, _LYMAN_W0, _LYMAN_AJI, _LYMAN_GI, _LYMAN_GJ, _LYMAN_RED)
    # 121.0 nm: inside Ly-alpha's window, blueward of Ly-beta's tail -> only
    # Ly-beta..delta; exactly AT Ly-alpha's edge -> Ly-alpha counts (RH does
    # at its last window point); 110 nm: between the edges -> Ly-beta..delta
    # only; 1 nm: below every edge -> 0
    higher = BG.rayleigh_(wl, _LYMAN_W0[1:], _LYMAN_AJI[1:], _LYMAN_GI[1:], _LYMAN_GJ[1:], _LYMAN_RED[1:])
    assert sigma[0] == higher[0]
    assert sigma[1] > 1e3 * sigma[0]
    assert sigma[2] == higher[2]
    assert sigma[3] == 0.0
    assert np.all(sigma >= 0.0)
