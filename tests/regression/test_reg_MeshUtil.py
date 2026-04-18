"""Regression tests for spectra.Util.MeshUtil"""

import numpy as np

from spectra.Util import MeshUtil

from .conftest import assert_close


class TestFullLineMesh:
    def test_mesh_31(self, ref):
        mesh = MeshUtil.make_full_line_mesh_(31, qcore=2.5, qwing=10.0)
        assert_close(mesh, ref["MeshUtil.full_line_mesh_31"])

    def test_mesh_11(self, ref):
        mesh = MeshUtil.make_full_line_mesh_(11, qcore=2.0, qwing=5.0)
        assert_close(mesh, ref["MeshUtil.full_line_mesh_11"])

    def test_symmetry(self):
        mesh = MeshUtil.make_full_line_mesh_(31, qcore=2.5, qwing=10.0)
        assert_close(mesh, -mesh[::-1])

    def test_odd_size(self):
        mesh = MeshUtil.make_full_line_mesh_(31)
        assert mesh.shape[0] == 31
