"""Regression tests for uncovered functions in spectra.Atomic.ContinuumOpacity"""

from spectra.Atomic import ContinuumOpacity

from .conftest import assert_close


class TestHpFfCrossSec:
    def test_scalar(self, ref):
        assert_close(ContinuumOpacity.Hp_ff_cross_sec_(7000.0, 5000e-8, 1e11), ref["COpac.Hp_ff_Te7000_w5000AA_Ne1e11"])


class TestH2pCrossSec:
    def test_scalar(self, ref):
        assert_close(ContinuumOpacity.H2p_cross_sec_(7000.0, 5000e-8, 1e10), ref["COpac.H2p_Te7000_w5000AA_Np1e10"])


class TestHLTEContinuumOpacity:
    def test_Te7000(self, ref):
        assert_close(
            ContinuumOpacity.H_LTE_continuum_opacity_(7000.0, 1e11, 1e12, 5000e-8),
            ref["COpac.H_LTE_opacity_Te7000_Ne1e11_Nh1e12_w5000AA"],
        )

    def test_Te5000(self, ref):
        assert_close(
            ContinuumOpacity.H_LTE_continuum_opacity_(5000.0, 1e10, 1e11, 8000e-8),
            ref["COpac.H_LTE_opacity_Te5000_Ne1e10_Nh1e11_w8000AA"],
        )
