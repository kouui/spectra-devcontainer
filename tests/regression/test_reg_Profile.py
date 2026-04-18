"""Regression tests for spectra.RadiativeTransfer.Profile"""

import numpy as np
import pytest

from spectra.RadiativeTransfer import Profile

from .conftest import assert_close


class TestVoigt:
    @pytest.mark.parametrize("a", [0.0, 0.001, 0.01, 0.1, 0.5, 1.0])
    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_voigt(self, ref, a, x):
        assert_close(Profile.voigt_(a, x), ref[f"Profile.voigt_a{a}_x{x}"])


class TestGaussian:
    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0, 5.0])
    def test_gaussian(self, ref, x):
        assert_close(Profile.gaussian_(x), ref[f"Profile.gaussian_x{x}"])


class TestHf:
    def test_hf(self, ref):
        if "Profile.hf_a01_h" not in ref:
            pytest.skip("hf_ reference not available")
        x_arr = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        h_arr, f_arr = Profile.hf_(0.1, x_arr)
        assert_close(h_arr, ref["Profile.hf_a01_h"])
        assert_close(f_arr, ref["Profile.hf_a01_f"])
