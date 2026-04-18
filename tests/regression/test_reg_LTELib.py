"""Regression tests for spectra.Atomic.LTELib"""

import numpy as np

from spectra.Atomic import LTELib
from spectra.ImportAll import *

from .conftest import assert_close


class TestBoltzmann:
    def test_scalar(self, ref):
        assert_close(LTELib.boltzmann_distribution_(2, 8, 1.63e-11, 7000.0), ref["LTELib.boltzmann_scalar"])

    def test_array(self, ref):
        gi = np.array([2, 2, 2], dtype=np.int64)
        gj = np.array([8, 18, 32], dtype=np.int64)
        Eji = np.array([1.63e-11, 1.94e-11, 2.04e-11])
        assert_close(LTELib.boltzmann_distribution_(gi, gj, Eji, 7000.0), ref["LTELib.boltzmann_array"])


class TestSaha:
    def test_scalar(self, ref):
        assert_close(LTELib.saha_distribution_(2, 1, 2.18e-11, 1e11, 7000.0), ref["LTELib.saha_scalar"])


class TestPlanck:
    def test_planck_cm_T5000(self, ref):
        assert_close(LTELib.planck_cm_(5000e-8, 5000.0), ref["LTELib.planck_cm_T5000"])

    def test_planck_cm_T7000(self, ref):
        assert_close(LTELib.planck_cm_(5000e-8, 7000.0), ref["LTELib.planck_cm_T7000"])

    def test_planck_cm_T10000(self, ref):
        assert_close(LTELib.planck_cm_(5000e-8, 10000.0), ref["LTELib.planck_cm_T10000"])

    def test_planck_hz(self, ref):
        assert_close(LTELib.planck_hz_(CST.c_ / 5000e-8, 7000.0), ref["LTELib.planck_hz"])


class TestEinsteinBs:
    def test_einsteinBs_cm(self, ref):
        Bji, Bij = LTELib.einsteinA_to_einsteinBs_cm_(6.27e8, 1216e-8, 2, 8)
        assert_close(Bji, ref["LTELib.einsteinBs_cm_Bji"])
        assert_close(Bij, ref["LTELib.einsteinBs_cm_Bij"])


class TestLTERatio:
    def test_LTE_ratio(self, ref):
        erg = np.array([0.0, 1.63e-11, 1.94e-11, 2.04e-11, 2.09e-11])
        g = np.array([2, 8, 18, 32, 50])
        stage = np.array([1, 1, 1, 1, 1])
        assert_close(LTELib.LTE_ratio_(erg, g, stage, 7000.0, 1e11), ref["LTELib.LTE_ratio"])
