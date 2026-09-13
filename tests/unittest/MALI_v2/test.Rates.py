"""Unit tests for the window integrals of the solved field (v2.Loop.line_rates_,
v2.Loop.bf_rates_).

Oracles:
- line Jbar/Lstar on line-only toys against v1's sweep (same axis, same
  profile tables; only the lower boundary differs -- per-column B_lambda here,
  line-centre B(w0) in v1 -- which bounds the agreement at the ~1e-5 level of
  dB/B across a window, not at round-off)
- b-f rates with J = B against v1's Planck-driven passive rates (different
  quadrature nodes: the continuum's own mesh there, its axis window here)
- detailed balance: LTE populations and J = B give n_i R_ik = n_k (R_ki,stim
  + R_ki,spon) per continuum
"""

import numpy as np
import pytest

from spectra.Atomic import LTELib, PhotoIonize
from spectra.Experimental.MALI.v1 import GlobalMesh as GM1
from spectra.Experimental.MALI.v1 import Loop as LP1
from spectra.Experimental.MALI.v1 import Structs as ST1
from spectra.Experimental.MALI.v2 import GlobalMesh, Loop, Structs
from spectra.Math import GaussLeg
from spectra.Util import MeshUtil

XI_REF = 2.5e5


def _meshes(atom, nLambda=41):
    q = MeshUtil.make_full_line_mesh_(nLambda, 2.5, 10.0)
    meshes = [GlobalMesh.anchor_line_mesh_(q, w0, XI_REF) for w0 in atom.Line["w0"]]
    meshes += [atom.Cont_mesh[kC, ::-1].copy() for kC in range(atom.nCont)]
    return meshes


def _sweep(atom, atmos, mesh, pre, n, n_angle=4):
    mus, wmus = GaussLeg.gauss_quad_coe_(0.0, 1.0, n_angle)
    n_abs = n * atmos.Nt[:, None]
    chi_int, S_line = Loop.line_coefficients_(
        n_abs, atom.Line["w0"].copy(), atom.Line["AJI"].copy(), atom.Line["BJI"].copy(), atom.Line["BIJ"].copy(),
        atom.Line["idxI"].copy(), atom.Line["idxJ"].copy(),
    )  # fmt: skip
    n_low, n_dag = Loop.continuum_coefficients_(
        n_abs, np.ascontiguousarray(pre.nj_by_ni[:, atom.nLine :]), atom.Cont["idxI"].copy(), atom.Cont["idxJ"].copy()
    )
    J, Psi, chi_tot = Loop.unified_sweep_(
        atmos.Z, mesh.wl, mesh.col_ptr, mesh.col_tran, mesh.col_row, atom.nLine,
        atom.Line["w0"].copy(), pre.phi, chi_int, S_line, pre.alpha_win, pre.row_cont0, n_low, n_dag,
        pre.exp_hnu_kT, pre.twohc2_wl5, pre.bg_chi, pre.bg_eta, pre.bg_sca, np.zeros_like(pre.bg_sca),
        pre.hn_bottom, mus, wmus,
    )  # fmt: skip
    return J, Psi, chi_tot, chi_int, S_line, (mus, wmus)


def _line_rates(atom, mesh, pre, J, Psi, chi_tot, chi_int):
    return Loop.line_rates_(
        J, Psi, chi_tot, mesh.wl, mesh.Nblue, mesh.span, mesh.win_off, pre.phi, pre.weight, pre.wphi,
        atom.Line["w0"].copy(), chi_int,
    )  # fmt: skip


class TestLineRatesAgainstV1:
    @pytest.mark.parametrize("Te_bottom", [None, 1.2e4])
    def test_three_level_jbar_lstar(self, Te_bottom):
        atom = Structs.make_toy_atom_3lv_()
        atmos = Structs.make_toy_atmos_(21, 1.0e9, Te_top=6.0e3, Te_bottom=Te_bottom, Ne=1.0e10, Nt=1.0e10)
        mesh = GlobalMesh.merge_meshes_(_meshes(atom))
        pre = Structs.precompute_(atom, atmos, mesh)
        n = pre.n_LTE * np.array([1.0, 2.0, 0.3])
        n /= n.sum(axis=1)[:, None]
        J, Psi, chi_tot, chi_int, S_line, (mus, wmus) = _sweep(atom, atmos, mesh, pre, n)
        Jbar, Lstar = _line_rates(atom, mesh, pre, J, Psi, chi_tot, chi_int)

        atom1 = ST1.make_toy_atom_3lv_()
        atmos1 = ST1.make_toy_atmos_(21, 1.0e9, Te_top=6.0e3, Te_bottom=Te_bottom, Ne=1.0e10, Nt=1.0e10)
        q = MeshUtil.make_full_line_mesh_(41, 2.5, 10.0)
        mesh1 = GM1.merge_meshes_([GM1.anchor_line_mesh_(q, w0, XI_REF) for w0 in atom1.Line["w0"]])
        pre1 = ST1.precompute_(atom1, atmos1, mesh1)
        Jbar1, Lstar1, S1 = LP1.multilevel_sweep_(
            atmos1.Z, n, atmos1.Nt, mesh1.wl, mesh1.Nblue, mesh1.span, pre1.win_off, pre1.phi, pre1.weight,
            pre1.wphi, atom1.Line["w0"].copy(), atom1.Line["AJI"].copy(), atom1.Line["BJI"].copy(),
            atom1.Line["BIJ"].copy(), atom1.Line["idxI"].copy(), atom1.Line["idxJ"].copy(), pre1.planck_w0,
            mus, wmus, pre1.bg_chi, pre1.bg_eta,
        )  # fmt: skip
        assert np.array_equal(S_line, S1)
        # Lstar never sees the boundary: d Jbar / d S is boundary-independent
        assert np.allclose(Lstar, Lstar1, rtol=1e-12, atol=0.0)
        assert np.allclose(Jbar, Jbar1, rtol=1e-4, atol=0.0)

    def test_lstar_in_unit_interval(self):
        atom = Structs.make_toy_atom_3lv_()
        atmos = Structs.make_toy_atmos_(21, 1.0e9)
        mesh = GlobalMesh.merge_meshes_(_meshes(atom))
        pre = Structs.precompute_(atom, atmos, mesh)
        J, Psi, chi_tot, chi_int, _, _ = _sweep(atom, atmos, mesh, pre, pre.n_LTE)
        _, Lstar = _line_rates(atom, mesh, pre, J, Psi, chi_tot, chi_int)
        assert np.all(Lstar >= 0.0)
        assert np.all(Lstar <= 1.0 + 1e-12)


class TestBoundFreeRates:
    def _setup(self):
        atom = Structs.make_toy_atom_2lv_cont_()
        atmos = Structs.make_toy_atmos_(7, 1.0e9, Te_top=6.0e3, Te_bottom=1.1e4, Ne=1.0e12, Nt=1.0e12)
        mesh = GlobalMesh.merge_meshes_(_meshes(atom, nLambda=21))
        pre = Structs.precompute_(atom, atmos, mesh)
        return atom, atmos, mesh, pre

    def _planck_J(self, mesh, atmos):
        return np.array([[LTELib.planck_cm_(float(wl), float(Te)) for Te in atmos.Te] for wl in mesh.wl])

    def test_kernel_matches_production_integrator(self):
        atom, atmos, mesh, pre = self._setup()
        J, *_ = _sweep(atom, atmos, mesh, pre, pre.n_LTE)
        Rik, Rki_stim, Rki_spon = Loop.bf_rates_(J, mesh, pre, atom, atmos)
        for kC in range(atom.nCont):
            t = atom.nLine + kC
            sl = slice(mesh.Nblue[t], mesh.Nblue[t] + mesh.span[t])
            alpha = pre.alpha_win[mesh.win_off[t, 0] - pre.row_cont0 : mesh.win_off[t, 1] - pre.row_cont0]
            for k in range(atmos.ND):
                ref = PhotoIonize.bound_free_radiative_transition_coefficient_(
                    wave=mesh.wl[sl], J=J[sl, k], alpha=alpha, Te=atmos.Te[k], nk_by_ni_LTE=pre.nj_by_ni[k, t]
                )
                assert Rik[k, kC] == pytest.approx(ref[0], rel=1e-13)
                assert Rki_stim[k, kC] == pytest.approx(ref[1], rel=1e-13)
                assert Rki_spon[k, kC] == pytest.approx(ref[2], rel=1e-13)

    def test_planck_field_matches_v1_passive_rates(self):
        atom, atmos, mesh, pre = self._setup()
        Rik, Rki_stim, Rki_spon = Loop.bf_rates_(self._planck_J(mesh, atmos), mesh, pre, atom, atmos)
        atom1 = ST1.make_toy_atom_2lv_cont_()
        atmos1 = ST1.make_toy_atmos_(7, 1.0e9, Te_top=6.0e3, Te_bottom=1.1e4, Ne=1.0e12, Nt=1.0e12)
        q = MeshUtil.make_full_line_mesh_(21, 2.5, 10.0)
        mesh1 = GM1.merge_meshes_([GM1.anchor_line_mesh_(q, atom1.Line["w0"][0], XI_REF)])
        pre1 = ST1.precompute_(atom1, atmos1, mesh1)  # Planck-driven passive rates
        # the (1,k) window carries the line's 21 extra nodes: quadrature differs
        assert np.allclose(Rik, pre1.Rik, rtol=2e-3)
        assert np.allclose(Rki_stim, pre1.Rki_stim, rtol=2e-3)
        assert np.allclose(Rki_spon, pre1.Rki_spon, rtol=2e-3)

    def test_detailed_balance_in_lte(self):
        atom, atmos, mesh, pre = self._setup()
        Rik, Rki_stim, Rki_spon = Loop.bf_rates_(self._planck_J(mesh, atmos), mesh, pre, atom, atmos)
        for kC in range(atom.nCont):
            ni = pre.n_LTE[:, atom.Cont["idxI"][kC]]
            nk = pre.n_LTE[:, atom.Cont["idxJ"][kC]]
            assert np.allclose(ni * Rik[:, kC], nk * (Rki_stim[:, kC] + Rki_spon[:, kC]), rtol=1e-10)

    def test_rates_from_solved_field_are_bounded_by_planck(self):
        # ISOTHERMAL thin slab: J < B above the bottom, so R_ik < the Planck rate
        # (with a hotter lower boundary J at the top would exceed the local B)
        atom = Structs.make_toy_atom_2lv_cont_()
        atmos = Structs.make_toy_atmos_(7, 1.0e9, Te_top=8.0e3, Ne=1.0e12, Nt=1.0e12)
        mesh = GlobalMesh.merge_meshes_(_meshes(atom, nLambda=21))
        pre = Structs.precompute_(atom, atmos, mesh)
        J, *_ = _sweep(atom, atmos, mesh, pre, pre.n_LTE)
        Rik, _, Rki_spon = Loop.bf_rates_(J, mesh, pre, atom, atmos)
        Rik_B, _, Rki_spon_B = Loop.bf_rates_(self._planck_J(mesh, atmos), mesh, pre, atom, atmos)
        assert np.all(Rik[:-1] < Rik_B[:-1])
        assert np.allclose(Rki_spon, Rki_spon_B, rtol=1e-12)  # spontaneous: no J
