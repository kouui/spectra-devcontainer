"""Regression tests for uncovered functions in spectra.Atomic.Collision"""

import numpy as np

from spectra.Atomic import Collision

from .conftest import assert_close


class TestInterpOmega:
    def test_at_table_point(self, ref):
        Te_table = np.array([3000.0, 5000.0, 7000.0, 10000.0, 15000.0])
        table = np.array([0.5, 0.8, 1.2, 1.5, 1.8])
        assert_close(Collision.interp_omega_(table, 5000.0, Te_table, 2.0, 3.0), ref["Collision.interp_omega_Te5000"])

    def test_interpolated(self, ref):
        Te_table = np.array([3000.0, 5000.0, 7000.0, 10000.0, 15000.0])
        table = np.array([0.5, 0.8, 1.2, 1.5, 1.8])
        assert_close(Collision.interp_omega_(table, 8000.0, Te_table, 2.0, 3.0), ref["Collision.interp_omega_Te8000"])

    def test_zero_table(self):
        Te_table = np.array([3000.0, 5000.0, 7000.0])
        table = np.array([0.0, 0.0, 0.0])
        assert Collision.interp_omega_(table, 5000.0, Te_table, 2.0, 3.0) == 0.0


class TestCERateCoeGeneral:
    def test_scalar(self, ref):
        assert_close(Collision.CE_rate_coe_(1.0, 7000.0, 2, 1.63e-11), ref["Collision.CE_rate_coe_scalar"])

    def test_scalar2(self, ref):
        assert_close(Collision.CE_rate_coe_(0.5, 10000.0, 8, 2.0e-11), ref["Collision.CE_rate_coe_scalar2"])


class TestCIRateCoeGeneral:
    def test_scalar(self, ref):
        assert_close(Collision.CI_rate_coe_(1e-7, 7000.0, 2.18e-11), ref["Collision.CI_rate_coe_scalar"])

    def test_scalar2(self, ref):
        assert_close(Collision.CI_rate_coe_(5e-8, 10000.0, 1.5e-11), ref["Collision.CI_rate_coe_scalar2"])
