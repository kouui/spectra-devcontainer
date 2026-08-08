"""Unit tests for spectra.Experimental.MALI.Structs (toys + loop-invariant tier).

The precompute tier depends only on (Te, Ne, Vt) per depth -- never on
populations -- so it must be bit-reproducible and consistent with the LTE
primitives it wraps.
"""

import numpy as np
import pytest

from spectra import Constants as CST
from spectra.Atomic import LTELib
from spectra.Experimental.MALI import GlobalMesh, Structs
from spectra.Util import MeshUtil

XI_REF = 2.5e5


def _mesh_for(atom, nLambda=41):
    q = MeshUtil.make_full_line_mesh_(nLambda, 2.5, 10.0)
    return GlobalMesh.merge_meshes_([GlobalMesh.anchor_line_mesh_(q, w0, XI_REF) for w0 in atom.Line["w0"]])


class TestToyAtom:
    def test_two_level_basic(self):
        atom = Structs.make_toy_atom_2lv_(w0_cm=5000.0e-8, Aji=1.0e8)
        assert atom.nLevel == 2
        assert atom.nLine == 1
        assert atom.Line["w0"][0] == pytest.approx(5000.0e-8, rel=1e-15)
        assert atom.Line["f0"][0] == pytest.approx(CST.c_ / 5000.0e-8, rel=1e-12)
        assert atom.Level["isGround"][0]
        assert not atom.Level["isGround"][1]

    def test_three_level_rydberg_consistent(self):
        atom = Structs.make_toy_atom_3lv_()
        w = atom.Line["w0"]
        # lines are (0,1), (1,2), (0,2): energies must close the triangle
        assert 1.0 / w[2] == pytest.approx(1.0 / w[0] + 1.0 / w[1], rel=1e-12)

    def test_einstein_relations(self):
        atom = Structs.make_toy_atom_3lv_()
        assert np.allclose(atom.Line["BJI"] / atom.Line["BIJ"], atom.Line["gi"] / atom.Line["gj"], rtol=1e-13)
        Bji_ref, _ = LTELib.einsteinA_to_einsteinBs_cm_(float(atom.Line["AJI"][0]), float(atom.Line["w0"][0]), 1, 3)
        assert atom.Line["BJI"][0] == pytest.approx(Bji_ref, rel=1e-13)


class TestPrecompute:
    def _setup(self, ND=9, Te_bottom=None):
        atom = Structs.make_toy_atom_3lv_()
        atmos = Structs.make_toy_atmos_(ND, 1.0e8, Te_top=6.0e3, Te_bottom=Te_bottom)
        mesh = _mesh_for(atom)
        return atom, atmos, mesh, Structs.precompute_(atom, atmos, mesh)

    def test_lte_populations_normalized(self):
        _, _, _, pre = self._setup()
        assert np.allclose(pre.n_LTE.sum(axis=1), 1.0, rtol=1e-13)

    def test_nj_by_ni_is_boltzmann(self):
        atom, atmos, _, pre = self._setup()
        Te = atmos.Te[0]
        for kL in range(atom.nLine):
            gi, gj = atom.Line["gi"][kL], atom.Line["gj"][kL]
            Eji = CST.h_ * atom.Line["f0"][kL]
            expected = (gj / gi) * np.exp(-Eji / (CST.k_ * Te))
            assert pre.nj_by_ni[0, kL] == pytest.approx(expected, rel=1e-12)

    def test_detailed_balance_cji(self):
        atom, _, _, pre = self._setup()
        # Cji = Cij / (nj/ni)_LTE
        assert np.allclose(pre.Cji_coe[0, :] * pre.nj_by_ni[0, :], atom.Cij_coe, rtol=1e-13)

    def test_bit_reproducible(self):
        atom, atmos, mesh, pre = self._setup()
        pre2 = Structs.precompute_(atom, atmos, mesh)
        assert np.array_equal(pre.n_LTE, pre2.n_LTE)
        assert np.array_equal(pre.phi, pre2.phi)
        assert np.array_equal(pre.wphi, pre2.wphi)

    def test_hot_depth_has_wider_line(self):
        _, _, _, pre = self._setup(Te_bottom=2.4e4)
        # same columns at every depth; the hot (deep) row is broader, so its
        # center value is lower and its far-wing value higher
        assert pre.dopWidth_cm[-1, 0] > pre.dopWidth_cm[0, 0]
        i0, i1 = pre.win_off[0]
        center = (i0 + i1) // 2
        assert pre.phi[center, -1] < pre.phi[center, 0]
        assert pre.phi[i0, -1] > pre.phi[i0, 0]

    def test_planck_at_line_center(self):
        atom, atmos, _, pre = self._setup()
        ref = LTELib.planck_cm_(float(atom.Line["w0"][1]), float(atmos.Te[3]))
        assert pre.planck_w0[3, 1] == pytest.approx(ref, rel=1e-13)
