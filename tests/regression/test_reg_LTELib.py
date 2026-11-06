"""Regression tests for spectra.Atomic.LTELib"""

import numpy as np
import pytest

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

    def test_LTE_ratio_ignores_allocator_content(self, ref):
        # numpy.empty may return non-zero memory; the result must not depend on it
        from unittest.mock import patch

        erg = np.array([0.0, 1.63e-11, 1.94e-11, 2.04e-11, 2.09e-11])
        g = np.array([2, 8, 18, 32, 50])
        stage = np.array([1, 1, 1, 1, 1])
        # NaN is the strictest poison: any read of an unset entry propagates to every output.
        # LTE_ratio_ is plain python; if it is ever compiled, this patch stops having an effect.
        with patch.object(
            LTELib._numpy, "empty", side_effect=lambda shape, dtype=float, **k: np.full(shape, np.nan, dtype=dtype, **k)
        ):
            nRatio = LTELib.LTE_ratio_(erg, g, stage, 7000.0, 1e11)
        assert_close(nRatio, ref["LTELib.LTE_ratio"])

    def test_LTE_ratio_saha_step(self):
        # a stage step of 1 uses the Saha equation (written out here, independent of
        # saha_distribution_), with (g_lower, g_upper) in that order
        erg = np.array([0.0, 2.18e-11])
        g = np.array([2, 1])
        stage = np.array([0, 1])
        Te, Ne = 7000.0, 1e11
        nRatio = LTELib.LTE_ratio_(erg, g, stage, Te, Ne)
        expected = CST.saha_ * Te**1.5 / Ne * (1 / 2) * np.exp(-2.18e-11 / (CST.k_ * Te))
        assert_close(nRatio[1] / nRatio[0], expected)
        assert_close(nRatio.sum(), 1.0)

    def test_LTE_ratio_long_chain_does_not_overflow(self):
        # 27 stages with Saha factors of ~7e15 each: the unscaled product (~1e412)
        # overflows, so the chain must be rescaled while it is built
        n = 27
        erg = np.arange(n, dtype=float) * 1e-9
        g = np.ones(n, dtype=np.int64)
        stage = np.arange(1, n + 1)
        Te, Ne = 1e7, 1e10
        nRatio = LTELib.LTE_ratio_(erg, g, stage, Te, Ne)
        assert np.isfinite(nRatio).all()
        # reference in log space
        log_f = np.log(CST.saha_ * Te**1.5 / Ne) - 1e-9 / (CST.k_ * Te)
        log_n = np.arange(n) * log_f
        ref_n = np.exp(log_n - log_n.max())
        ref_n /= ref_n.sum()
        np.testing.assert_allclose(nRatio, ref_n, rtol=1e-10, atol=1e-300)

    def test_LTE_ratio_rejects_stage_jump(self):
        erg = np.array([0.0, 1.0e-11, 3.0e-11])
        g = np.array([2, 1, 1])
        stage = np.array([0, 0, 2])
        with pytest.raises(ValueError, match="stages differing by 1"):
            LTELib.LTE_ratio_(erg, g, stage, 7000.0, 1e11)
