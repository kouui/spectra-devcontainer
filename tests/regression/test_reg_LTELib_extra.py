"""Regression tests for uncovered functions in spectra.Atomic.LTELib"""

import numpy as np

from spectra.Atomic import LTELib
from spectra.ImportAll import *

from .conftest import assert_close


class TestLTERatioLine:
    def test_ratio_line(self, ref):
        g = np.array([2, 8, 18, 32, 50])
        idxI = np.array([0, 0, 1])
        idxJ = np.array([1, 2, 2])
        w0 = np.array([1216e-8, 1026e-8, 6563e-8])
        assert_close(LTELib.LTE_ratio_Line_(g, idxI, idxJ, w0, 7000.0), ref["LTELib.LTE_ratio_Line"])


class TestLTERatioCont:
    def test_ratio_cont(self, ref):
        g = np.array([2, 8, 1])
        idxI = np.array([0, 1])
        idxJ = np.array([2, 2])
        w0 = np.array([912e-8, 3647e-8])
        assert_close(LTELib.LTE_ratio_Cont_(g, idxI, idxJ, w0, 7000.0, 1e11), ref["LTELib.LTE_ratio_Cont"])


class TestEinsteinBsHz:
    def test_einsteinBs_hz(self, ref):
        f0 = CST.c_ / 1216e-8
        Bji, Bij = LTELib.einsteinA_to_einsteinBs_hz_(6.27e8, f0, 2, 8)
        assert_close(Bji, ref["LTELib.einsteinBs_hz_Bji"])
        assert_close(Bij, ref["LTELib.einsteinBs_hz_Bij"])

    def test_einstein_relation(self):
        # B_ji = A_ji / (2 h nu^3 / c^2) and g_i B_ij = g_j B_ji, written out here
        Aji, f0, gi, gj = 6.27e8, CST.c_ / 1216e-8, 2, 8
        Bji, Bij = LTELib.einsteinA_to_einsteinBs_hz_(Aji, f0, gi, gj)
        assert_close(Bji, Aji * CST.c_**2 / (2.0 * CST.h_ * f0**3))
        assert_close(gi * Bij, gj * Bji)

    def test_consistent_with_wavelength_convention(self):
        # B_ji J_nu = B_ji(lambda) J_lambda with J_lambda = J_nu c / lambda^2
        Aji, w0, gi, gj = 6.27e8, 1216e-8, 2, 8
        Bji_hz, _ = LTELib.einsteinA_to_einsteinBs_hz_(Aji, CST.c_ / w0, gi, gj)
        Bji_cm, _ = LTELib.einsteinA_to_einsteinBs_cm_(Aji, w0, gi, gj)
        assert_close(Bji_hz, Bji_cm * CST.c_ / w0**2)


class TestAjiToBjiCm:
    def test_scalar(self, ref):
        assert_close(LTELib.Aji_to_Bji_cm_(6.27e8, 1216e-8), ref["LTELib.Aji_to_Bji_cm"])


class TestBjiToBij:
    def test_scalar(self, ref):
        Bji = LTELib.Aji_to_Bji_cm_(6.27e8, 1216e-8)
        assert_close(LTELib.Bji_to_Bij_(Bji, 2, 8), ref["LTELib.Bji_to_Bij"])


class TestUfunc:
    def test_H_I(self, ref):
        assert_close(LTELib.Ufunc_("H_I", 5040.0), ref["LTELib.Ufunc_H_I_5040"])
        assert_close(LTELib.Ufunc_("H_I", 7000.0), ref["LTELib.Ufunc_H_I_7000"])

    def test_He_I(self, ref):
        assert_close(LTELib.Ufunc_("He_I", 5040.0), ref["LTELib.Ufunc_He_I_5040"])

    def test_Ca_II(self, ref):
        assert_close(LTELib.Ufunc_("Ca_II", 5040.0), ref["LTELib.Ufunc_Ca_II_5040"])
        assert_close(LTELib.Ufunc_("Ca_II", 7000.0), ref["LTELib.Ufunc_Ca_II_7000"])

    def test_Fe_I(self, ref):
        assert_close(LTELib.Ufunc_("Fe_I", 5040.0), ref["LTELib.Ufunc_Fe_I_5040"])
        assert_close(LTELib.Ufunc_("Fe_I", 7000.0), ref["LTELib.Ufunc_Fe_I_7000"])
