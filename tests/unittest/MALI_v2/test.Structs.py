"""Unit tests for spectra.Experimental.MALI.v2.Structs (toys + loop-invariant tier).

The precompute tier depends only on (Te, Ne, Vt) per depth -- never on
populations -- so it must be bit-reproducible and consistent with the LTE
primitives it wraps. On the unified axis it also tabulates the continuum
cross sections on their windows and the per-column factors the sweep needs.
"""

import numpy as np
import pytest

from spectra import Constants as CST
from spectra.Atomic import LTELib
from spectra.Experimental.MALI.v2 import GlobalMesh, Structs
from spectra.Util import MeshUtil

XI_REF = 2.5e5


def _mesh_for(atom, nLambda=41):
    q = MeshUtil.make_full_line_mesh_(nLambda, 2.5, 10.0)
    meshes = [GlobalMesh.anchor_line_mesh_(q, w0, XI_REF) for w0 in atom.Line["w0"]]
    meshes += [atom.Cont_mesh[kC, ::-1].copy() for kC in range(atom.nCont)]
    return GlobalMesh.merge_meshes_(meshes)


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
        assert 1.0 / w[2] == pytest.approx(1.0 / w[0] + 1.0 / w[1], rel=1e-12)

    def test_einstein_relations(self):
        atom = Structs.make_toy_atom_3lv_()
        assert np.allclose(atom.Line["BJI"] / atom.Line["BIJ"], atom.Line["gi"] / atom.Line["gj"], rtol=1e-13)
        Bji_ref, _ = LTELib.einsteinA_to_einsteinBs_cm_(float(atom.Line["AJI"][0]), float(atom.Line["w0"][0]), 1, 3)
        assert atom.Line["BJI"][0] == pytest.approx(Bji_ref, rel=1e-13)

    def test_two_level_cont_layout(self):
        atom = Structs.make_toy_atom_2lv_cont_()
        assert (atom.nLevel, atom.nLine, atom.nCont) == (3, 1, 2)
        assert atom.Level["isGround"][2]
        # thresholds from the level energies; descending production meshes
        for kC in range(2):
            chi_ion = atom.Level["erg"][2] - atom.Level["erg"][atom.Cont["idxI"][kC]]
            assert atom.Cont_mesh[kC, 0] == pytest.approx(CST.h_ * CST.c_ / chi_ion, rel=1e-12)
            assert np.all(np.diff(atom.Cont_mesh[kC]) < 0)


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
        assert np.allclose(pre.Cji_coe[0, :] * pre.nj_by_ni[0, :], atom.Cij_coe, rtol=1e-13)

    def test_bit_reproducible(self):
        atom, atmos, mesh, pre = self._setup()
        pre2 = Structs.precompute_(atom, atmos, mesh)
        assert np.array_equal(pre.n_LTE, pre2.n_LTE)
        assert np.array_equal(pre.phi, pre2.phi)
        assert np.array_equal(pre.wphi, pre2.wphi)

    def test_hot_depth_has_wider_line(self):
        _, _, mesh, pre = self._setup(Te_bottom=2.4e4)
        assert pre.dopWidth_cm[-1, 0] > pre.dopWidth_cm[0, 0]
        i0, i1 = mesh.win_off[0]
        center = (i0 + i1) // 2
        assert pre.phi[center, -1] < pre.phi[center, 0]
        assert pre.phi[i0, -1] > pre.phi[i0, 0]

    def test_line_only_has_empty_continuum_tables(self):
        _, atmos, mesh, pre = self._setup()
        assert pre.alpha_win.shape == (0,)
        assert pre.row_cont0 == pre.phi.shape[0] == mesh.span.sum()
        assert pre.bg_chi.shape == (mesh.wl.shape[0], atmos.ND)
        assert not pre.bg_chi.any()
        assert not pre.bg_eta.any()
        assert not pre.bg_sca.any()

    def test_per_column_factors(self):
        _, atmos, mesh, pre = self._setup()
        iw, k = 5, 3
        wl = float(mesh.wl[iw])
        assert pre.hn_bottom[iw] == pytest.approx(LTELib.planck_cm_(wl, float(atmos.Te[-1])), rel=1e-13)
        assert pre.exp_hnu_kT[iw, k] == pytest.approx(np.exp(-CST.h_ * CST.c_ / (wl * CST.k_ * atmos.Te[k])), rel=1e-13)
        assert pre.twohc2_wl5[iw] == pytest.approx(2.0 * CST.h_ * CST.c_**2 / wl**5, rel=1e-13)

    def test_window_count_mismatch_rejected(self):
        atom = Structs.make_toy_atom_2lv_cont_()
        atmos = Structs.make_toy_atmos_(5, 1.0e8)
        q = MeshUtil.make_full_line_mesh_(21, 2.5, 10.0)
        mesh_lines_only = GlobalMesh.merge_meshes_([GlobalMesh.anchor_line_mesh_(q, atom.Line["w0"][0], XI_REF)])
        with pytest.raises(ValueError, match="windows"):
            Structs.precompute_(atom, atmos, mesh_lines_only)


class TestPrecomputeContinua:
    def _setup(self, ND=7):
        atom = Structs.make_toy_atom_2lv_cont_()
        atmos = Structs.make_toy_atmos_(ND, 1.0e8)
        mesh = _mesh_for(atom, nLambda=21)
        return atom, atmos, mesh, Structs.precompute_(atom, atmos, mesh)

    def test_alpha_exact_at_own_mesh_points(self):
        atom, _, mesh, pre = self._setup()
        for kC in range(atom.nCont):
            t = atom.nLine + kC
            wl_win = mesh.wl[mesh.Nblue[t] : mesh.Nblue[t] + mesh.span[t]]
            a_win = pre.alpha_win[mesh.win_off[t, 0] - pre.row_cont0 : mesh.win_off[t, 1] - pre.row_cont0]
            for wl_own, a_own in zip(atom.Cont_mesh[kC], atom.alpha[kC], strict=True):
                j = int(np.argmin(np.abs(wl_win - wl_own)))
                assert a_win[j] == pytest.approx(a_own, rel=1e-12)

    def test_alpha_loglog_on_foreign_points(self):
        # the (1,k) continuum window contains the line's points: alpha there
        # must follow the hydrogenic wl^3 shape of the toy exactly (log-log
        # interpolation of a power law is exact)
        atom, _, mesh, pre = self._setup()
        t = atom.nLine + 1
        wl_win = mesh.wl[mesh.Nblue[t] : mesh.Nblue[t] + mesh.span[t]]
        a_win = pre.alpha_win[mesh.win_off[t, 0] - pre.row_cont0 : mesh.win_off[t, 1] - pre.row_cont0]
        w_edge = atom.Cont_mesh[1, 0]
        ref = atom.alpha[1, 0] * (wl_win / w_edge) ** 3
        assert np.allclose(a_win, ref, rtol=1e-10)
        assert a_win.shape[0] == mesh.span[t]
        assert np.all(a_win > 0.0)

    def test_row_layout(self):
        atom, _, mesh, pre = self._setup()
        assert pre.row_cont0 == mesh.win_off[atom.nLine, 0]
        assert pre.phi.shape[0] == pre.row_cont0
        assert pre.alpha_win.shape[0] == mesh.win_off[-1, 1] - pre.row_cont0

    def test_background_tables(self):
        atom, atmos, mesh, _ = self._setup()
        chi0, sig0 = 1.0e-12, 3.0e-13

        def bg_fn(wl_cm):
            return np.full(atmos.ND, chi0 * (wl_cm / 5000.0e-8)), np.full(atmos.ND, sig0)

        pre = Structs.precompute_(atom, atmos, mesh, bg_fn=bg_fn)
        iw, k = 10, 2
        wl = float(mesh.wl[iw])
        assert pre.bg_chi[iw, k] == pytest.approx(chi0 * wl / 5000.0e-8, rel=1e-13)
        assert pre.bg_eta[iw, k] == pytest.approx(
            pre.bg_chi[iw, k] * LTELib.planck_cm_(wl, float(atmos.Te[k])), rel=1e-13
        )
        assert np.all(pre.bg_sca == sig0)

    def test_background_contract_enforced(self):
        atom, atmos, mesh, _ = self._setup()
        with pytest.raises(ValueError, match="arrays"):
            Structs.precompute_(atom, atmos, mesh, bg_fn=lambda wl_cm: (1.0e-12, np.zeros(atmos.ND)))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="arrays"):
            Structs.precompute_(atom, atmos, mesh, bg_fn=lambda wl_cm: (np.zeros(atmos.ND + 1), np.zeros(atmos.ND)))
        with pytest.raises(ValueError, match="finite"):
            Structs.precompute_(atom, atmos, mesh, bg_fn=lambda wl_cm: (np.full(atmos.ND, np.nan), np.zeros(atmos.ND)))
