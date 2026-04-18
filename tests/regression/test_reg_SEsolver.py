"""Regression tests for spectra.Atomic.SEsolver"""

import numpy as np

from spectra.Atomic import SEsolver

from .conftest import assert_close


class TestSolveSE:
    def test_4level(self, ref):
        nLevel = 4
        Rmat = np.zeros((nLevel, nLevel))
        Cmat = np.zeros((nLevel, nLevel))
        Rmat[0, 1] = 1e8
        Rmat[1, 0] = 1e6
        Rmat[0, 2] = 5e7
        Rmat[2, 0] = 1e5
        Rmat[1, 2] = 3e7
        Rmat[2, 1] = 2e6
        Rmat[0, 3] = 1e7
        Rmat[3, 0] = 1e4
        Cmat[0, 1] = 5e5
        Cmat[1, 0] = 1e4
        Cmat[0, 2] = 3e5
        Cmat[2, 0] = 5e3
        Cmat[1, 2] = 2e5
        Cmat[2, 1] = 1e4
        Cmat[0, 3] = 1e5
        Cmat[3, 0] = 1e3
        n_SE = SEsolver.solve_SE_(Rmat, Cmat)
        assert_close(n_SE, ref["SEsolver.solve_SE_4level"])

    def test_populations_sum_to_one(self):
        nLevel = 4
        Rmat = np.zeros((nLevel, nLevel))
        Cmat = np.zeros((nLevel, nLevel))
        Rmat[0, 1] = 1e8
        Rmat[1, 0] = 1e6
        Rmat[0, 2] = 5e7
        Rmat[2, 0] = 1e5
        Rmat[1, 2] = 3e7
        Rmat[2, 1] = 2e6
        Rmat[0, 3] = 1e7
        Rmat[3, 0] = 1e4
        Cmat[0, 1] = 5e5
        Cmat[1, 0] = 1e4
        n_SE = SEsolver.solve_SE_(Rmat, Cmat)
        assert_close(np.sum(n_SE), 1.0, rtol=1e-12)

    def test_populations_non_negative(self):
        nLevel = 3
        Rmat = np.zeros((nLevel, nLevel))
        Cmat = np.zeros((nLevel, nLevel))
        Rmat[0, 1] = 1e8
        Rmat[1, 0] = 1e6
        Rmat[0, 2] = 5e7
        Rmat[2, 0] = 1e5
        n_SE = SEsolver.solve_SE_(Rmat, Cmat)
        assert np.all(n_SE >= 0)


class TestSetMatrixC:
    def test_3level(self, ref):
        Cmat = np.zeros((3, 3))
        Cji = np.array([1e-8, 2e-8])
        Cij = np.array([3e-8, 4e-8])
        idxI = np.array([0, 1])
        idxJ = np.array([1, 2])
        SEsolver.set_matrixC_(Cmat, Cji, Cij, idxI, idxJ, 1e11)
        assert_close(Cmat, ref["SEsolver.set_matrixC_3level"])


class TestSetMatrixR:
    def test_3level(self, ref):
        Rmat = np.zeros((3, 3))
        Rji_spon = np.array([1e8, 2e8])
        Rji_stim = np.array([5e6, 1e7])
        Rij = np.array([3e6, 4e6])
        idxI = np.array([0, 1])
        idxJ = np.array([1, 2])
        SEsolver.set_matrixR_(Rmat, Rji_spon, Rji_stim, Rij, idxI, idxJ)
        assert_close(Rmat, ref["SEsolver.set_matrixR_3level"])
