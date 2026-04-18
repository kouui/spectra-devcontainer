"""Regression tests for spectra.Atomic.Collision"""

import numpy as np

from spectra.Atomic import Collision

from .conftest import assert_close


class TestCijToCji:
    def test_scalar(self, ref):
        assert_close(Collision.Cij_to_Cji_(1.5e-8, 0.01), ref["Collision.Cij_to_Cji_scalar"])

    def test_array(self, ref):
        cij = np.array([1.5e-8, 2.0e-8, 3.0e-8])
        ratio = np.array([0.01, 0.02, 0.03])
        assert_close(Collision.Cij_to_Cji_(cij, ratio), ref["Collision.Cij_to_Cji_array"])
