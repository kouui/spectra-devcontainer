"""Regression tests for spectra.Math (Special, Integrate, GaussLeg)"""

import numpy as np
import pytest

from spectra.Math import GaussLeg, Integrate, Special

from .conftest import assert_close


class TestExponentialIntegrals:
    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_E0(self, ref, x):
        assert_close(Special.E0_(x), ref[f"Special.E0_x{x}"])

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_E1(self, ref, x):
        assert_close(Special.E1_(x), ref[f"Special.E1_x{x}"])

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_E2(self, ref, x):
        assert_close(Special.E2_(x), ref[f"Special.E2_x{x}"])

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_E3(self, ref, x):
        assert_close(Special.E3_(x), ref[f"Special.E3_x{x}"])


class TestIntegrate:
    def test_trapze_sin(self, ref):
        x = np.linspace(0, np.pi, 101)
        y = np.sin(x)
        assert_close(Integrate.trapze_(y, x), ref["Integrate.trapze_sin_0_pi"])

    def test_simpson_sin(self, ref):
        x = np.linspace(0, np.pi, 101)
        y = np.sin(x)
        assert_close(Integrate.simpson_(y, x=x), ref["Integrate.simpson_sin_0_pi"])


class TestGaussLegendre:
    @pytest.mark.parametrize("n", [3, 4, 5, 8])
    def test_abscissas_and_weights(self, ref, n):
        x, w = GaussLeg.gauss_quad_coe_(0.0, 1.0, n)
        assert_close(x, ref[f"GaussLeg.x_n{n}"])
        assert_close(w, ref[f"GaussLeg.w_n{n}"])

    @pytest.mark.parametrize("n", [3, 4, 5, 8])
    def test_weights_sum_to_interval(self, n):
        _, w = GaussLeg.gauss_quad_coe_(0.0, 1.0, n)
        assert_close(np.sum(w), 1.0, rtol=1e-12)
