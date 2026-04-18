"""Regression tests for spectra.Atomic.ContinuumOpacity"""

import pytest

from spectra.Atomic import ContinuumOpacity

from .conftest import assert_close


class TestThomson:
    def test_thomson(self, ref):
        assert_close(ContinuumOpacity.thomson_scattering_(1e11), ref["COpac.thomson_1e11"])


class TestHydrogenicBfCrossSec:
    def test_ni1_w500AA(self, ref):
        assert_close(ContinuumOpacity.hydrogenic_bf_cross_sec_n_(1, 500e-8, 1), ref["COpac.hyd_bf_ni1_w500AA"])

    def test_ni2_w3000AA(self, ref):
        assert_close(ContinuumOpacity.hydrogenic_bf_cross_sec_n_(2, 3000e-8, 1), ref["COpac.hyd_bf_ni2_w3000AA"])


class TestHIBfLTECrossSec:
    @pytest.mark.parametrize("Te", [5000, 7000, 10000])
    def test_HI_bf_LTE(self, ref, Te):
        assert_close(
            ContinuumOpacity.HI_bf_LTE_cross_sec_(float(Te), 5000e-8),
            ref[f"COpac.HI_bf_LTE_Te{Te}_w5000AA"],
        )


class TestHIRayleigh:
    @pytest.mark.parametrize("w_AA", [2000, 5000, 8000])
    def test_rayleigh(self, ref, w_AA):
        assert_close(
            ContinuumOpacity.HI_rayleigh_cross_sec_(w_AA * 1e-8),
            ref[f"COpac.HI_rayleigh_w{w_AA}AA"],
        )


class TestHminus:
    def test_Hminus(self, ref):
        assert_close(
            ContinuumOpacity.Hminus_cross_sec_(7000.0, 5000e-8, 1e11),
            ref["COpac.Hminus_Te7000_w5000AA_Ne1e11"],
        )


class TestHIFf:
    def test_HI_ff(self, ref):
        assert_close(ContinuumOpacity.HI_ff_cross_sec_(7000.0, 5000e-8), ref["COpac.HI_ff_Te7000_w5000AA"])


class TestGauntFactorFf:
    def test_gff(self, ref):
        assert_close(ContinuumOpacity.gaunt_factor_ff_(7000.0, 5000e-8), ref["COpac.gff_Te7000_w5000AA"])
