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
        assert_close(LTELib.LTE_ratio_Line_(g, idxI, idxJ, w0, 7000.0),
                     ref["LTELib.LTE_ratio_Line"])


class TestLTERatioCont:
    def test_ratio_cont(self, ref):
        g = np.array([2, 8, 1])
        idxI = np.array([0, 1])
        idxJ = np.array([2, 2])
        w0 = np.array([912e-8, 3647e-8])
        assert_close(LTELib.LTE_ratio_Cont_(g, idxI, idxJ, w0, 7000.0, 1e11),
                     ref["LTELib.LTE_ratio_Cont"])


class TestEinsteinBsHz:
    def test_einsteinBs_hz(self, ref):
        f0 = CST.c_ / 1216e-8
        Bji, Bij = LTELib.einsteinA_to_einsteinBs_hz_(6.27e8, f0, 2, 8)
        assert_close(Bji, ref["LTELib.einsteinBs_hz_Bji"])
        assert_close(Bij, ref["LTELib.einsteinBs_hz_Bij"])


class TestAjiToBjiCm:
    def test_scalar(self, ref):
        assert_close(LTELib.Aji_to_Bji_cm_(6.27e8, 1216e-8),
                     ref["LTELib.Aji_to_Bji_cm"])


class TestBjiToBij:
    def test_scalar(self, ref):
        Bji = LTELib.Aji_to_Bji_cm_(6.27e8, 1216e-8)
        assert_close(LTELib.Bji_to_Bij_(Bji, 2, 8),
                     ref["LTELib.Bji_to_Bij"])
