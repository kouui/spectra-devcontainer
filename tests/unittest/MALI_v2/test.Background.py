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
