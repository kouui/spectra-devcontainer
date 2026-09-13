"""v1 is the frozen oracle for what v2 must reproduce. The source packages
never import each other; this test may, because it is the one place where the
two are compared.

Line-only toys put exactly the same points on the axis in both versions, so
the build-once profile tables must be bit-identical."""

import types

import numpy as np

from spectra.Experimental.MALI.v1 import GlobalMesh as GM1
from spectra.Experimental.MALI.v1 import Structs as ST1
from spectra.Experimental.MALI.v2 import GlobalMesh as GM2
from spectra.Experimental.MALI.v2 import Structs as ST2
from spectra.Util import MeshUtil

XI_REF = 2.5e5
v1 = types.SimpleNamespace(GlobalMesh=GM1, Structs=ST1)
v2 = types.SimpleNamespace(GlobalMesh=GM2, Structs=ST2)


def _build(pkg, atom, atmos, nLambda=41):
    q = MeshUtil.make_full_line_mesh_(nLambda, 2.5, 10.0)
    mesh = pkg.GlobalMesh.merge_meshes_([pkg.GlobalMesh.anchor_line_mesh_(q, w0, XI_REF) for w0 in atom.Line["w0"]])
    return mesh, pkg.Structs.precompute_(atom, atmos, mesh)


def test_line_only_profile_tables_identical():
    atom1, atom2 = v1.Structs.make_toy_atom_3lv_(), v2.Structs.make_toy_atom_3lv_()
    atmos1 = v1.Structs.make_toy_atmos_(9, 1.0e8, Te_top=6.0e3, Te_bottom=1.2e4)
    atmos2 = v2.Structs.make_toy_atmos_(9, 1.0e8, Te_top=6.0e3, Te_bottom=1.2e4)
    mesh1, pre1 = _build(v1, atom1, atmos1)
    mesh2, pre2 = _build(v2, atom2, atmos2)
    assert np.array_equal(mesh1.wl, mesh2.wl)
    assert np.array_equal(mesh1.Nblue, mesh2.Nblue)
    assert np.array_equal(mesh1.span, mesh2.span)
    assert np.array_equal(pre1.win_off, mesh2.win_off)
    for name in ("n_LTE", "nj_by_ni", "Cij_coe", "Cji_coe", "dopWidth_cm", "adamp", "phi", "wphi", "weight"):
        assert np.array_equal(getattr(pre1, name), getattr(pre2, name)), name
