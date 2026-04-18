"""Regression tests for uncovered functions in spectra.Atomic.BasicP"""

import numpy as np

from spectra.Atomic import BasicP

from .conftest import assert_close


class TestUpdateLevelGamma:
    def test_update(self, ref):
        Aji = np.array([6.27e8, 1.67e8, 5.58e7])
        idxJ = np.array([1, 2, 2])
        gamma = np.zeros(3)
        BasicP.update_level_gamma_(Aji, idxJ, gamma)
        assert_close(gamma, ref["BasicP.update_level_gamma"])


class TestUpdateLineGamma:
    def test_update(self, ref):
        Aji = np.array([6.27e8, 1.67e8, 5.58e7])
        idxJ = np.array([1, 2, 2])
        gamma_level = np.zeros(3)
        BasicP.update_level_gamma_(Aji, idxJ, gamma_level)

        idxI = np.array([0, 0, 1])
        gamma_line = np.zeros(3)
        BasicP.update_line_gamma_(idxI, idxJ, gamma_level, gamma_line)
        assert_close(gamma_line, ref["BasicP.update_line_gamma"])
