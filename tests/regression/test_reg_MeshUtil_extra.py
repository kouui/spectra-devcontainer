"""Regression tests for uncovered functions in spectra.Util.MeshUtil"""

import numpy as np

from spectra.Util import MeshUtil

from .conftest import assert_close


class TestMakeContinuumMesh:
    def test_10(self, ref):
        assert_close(MeshUtil.make_continuum_mesh_(10), ref["MeshUtil.continuum_mesh_10"])

    def test_5(self, ref):
        assert_close(MeshUtil.make_continuum_mesh_(5), ref["MeshUtil.continuum_mesh_5"])


class TestHalfToFull:
    def test_symmetric(self, ref):
        half = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        assert_close(MeshUtil.half_to_full_(half, False), ref["MeshUtil.half_to_full_sym"])

    def test_antisymmetric(self, ref):
        half = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        assert_close(MeshUtil.half_to_full_(half, True), ref["MeshUtil.half_to_full_anti"])


class TestMakeHalfLineMesh:
    def test_half_mesh(self, ref):
        q = np.zeros(16)
        MeshUtil.make_half_line_mesh_(31, 2.5, 10.0, q)
        assert_close(q, ref["MeshUtil.half_line_mesh_31"])


class TestArrayFrom1D:
    def test_extract(self, ref):
        arr_1D = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        mesh_idxs = np.array([[0, 3], [3, 6]])
        assert_close(MeshUtil.array_from_1D_(arr_1D, mesh_idxs, 0), ref["MeshUtil.array_from_1D_k0"])
        assert_close(MeshUtil.array_from_1D_(arr_1D, mesh_idxs, 1), ref["MeshUtil.array_from_1D_k1"])
