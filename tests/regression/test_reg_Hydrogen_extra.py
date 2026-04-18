"""Regression tests for uncovered functions in spectra.Atomic.Hydrogen"""

import pytest

from spectra.Atomic import Hydrogen

from .conftest import assert_close


class TestGauntFactorCoe:
    @pytest.mark.parametrize("ni", [1, 2, 3])
    @pytest.mark.parametrize("i", [0, 1, 2])
    def test_coe(self, ref, ni, i):
        assert_close(Hydrogen.gaunt_factor_coe_(i, ni),
                     ref[f"Hydrogen.gaunt_coe_i{i}_ni{ni}"])


class TestPICrossSectionX:
    def test_above_threshold(self, ref):
        assert_close(Hydrogen.PI_cross_section_(1, 2.0, 1),
                     ref["Hydrogen.PI_x_ni1_x2_Z1"])

    def test_ni2(self, ref):
        assert_close(Hydrogen.PI_cross_section_(2, 1.5, 1),
                     ref["Hydrogen.PI_x_ni2_x1p5_Z1"])

    def test_below_threshold(self, ref):
        assert_close(Hydrogen.PI_cross_section_(1, 0.5, 1),
                     ref["Hydrogen.PI_x_ni1_x0p5_Z1"])


class TestCollisionalBroadeningResVan:
    def test_ni1_nj2(self, ref):
        assert_close(
            Hydrogen.collisional_broadening_Res_and_Van_(1, 2, 1e12, 7000.0),
            ref["Hydrogen.ResVan_ni1_nj2_nH1e12_Te7000"])

    def test_ni2_nj3(self, ref):
        assert_close(
            Hydrogen.collisional_broadening_Res_and_Van_(2, 3, 1e12, 7000.0),
            ref["Hydrogen.ResVan_ni2_nj3_nH1e12_Te7000"])

    def test_ni1_nj3(self, ref):
        assert_close(
            Hydrogen.collisional_broadening_Res_and_Van_(1, 3, 1e12, 7000.0),
            ref["Hydrogen.ResVan_ni1_nj3_nH1e12_Te7000"])
