"""Unit tests for the window integrals of the solved field (Kernels.line_rates_,
Loop.bf_rates_).

Oracles:
- the line operator Lstar is a fraction: 0 <= Lstar <= 1
- b-f rates against the production PhotoIonize integrator on the same nodes
- detailed balance: LTE populations and J = B give n_i R_ik = n_k (R_ki,stim
  + R_ki,spon) per continuum
- an isothermal slab sees J < B, so R_ik stays below the Planck rate
"""

import numpy as np
import pytest

from spectra.Atomic import LTELib, PhotoIonize
from spectra.Function.MALI import Loop, Setup
from spectra.Math import GaussLeg
from spectra.RadiativeTransfer.MALI import GlobalMesh, Kernels
from spectra.Util import MeshUtil

from . import _toy as Toy

XI_REF = 2.5e5


def _meshes(atom, nLambda=41):
    q = MeshUtil.make_full_line_mesh_(nLambda, 2.5, 10.0)
    meshes = [GlobalMesh.anchor_line_mesh_(q, w0, XI_REF) for w0 in atom.Line["w0"]]
    meshes += [atom.Cont_mesh[kC, ::-1].copy() for kC in range(atom.nCont)]
    return meshes


def _sweep(atom, atmos, mesh, pre, n, n_angle=4):
    mus, wmus = GaussLeg.gauss_quad_coe_(0.0, 1.0, n_angle)
    n_abs = n * atmos.Nt[:, None]
    chi_int, S_line = Kernels.line_coefficients_(
        n_abs, atom.Line["w0"].copy(), atom.Line["AJI"].copy(), atom.Line["BJI"].copy(), atom.Line["BIJ"].copy(),
        atom.Line["idxI"].copy(), atom.Line["idxJ"].copy(),
    )  # fmt: skip
    n_low, n_dag = Kernels.continuum_coefficients_(
        n_abs, np.ascontiguousarray(pre.nj_by_ni[:, atom.nLine :]), atom.Cont["idxI"].copy(), atom.Cont["idxJ"].copy()
    )
    J, Psi, chi_tot = Kernels.unified_sweep_(
        atmos.Z, mesh.wl, mesh.col_ptr, mesh.col_tran, mesh.col_row, atom.nLine,
        atom.Line["w0"].copy(), pre.phi, chi_int, S_line, pre.alpha_win, pre.row_cont0, n_low, n_dag,
        pre.exp_hnu_kT, pre.twohc2_wl5, pre.bg_chi, pre.bg_eta, pre.bg_sca, np.zeros_like(pre.bg_sca),
        pre.hn_bottom, mus, wmus,
    )  # fmt: skip
    return J, Psi, chi_tot, chi_int, S_line, (mus, wmus)


def _line_rates(atom, mesh, pre, J, Psi, chi_tot, chi_int):
    return Kernels.line_rates_(
        J, Psi, chi_tot, mesh.wl, mesh.Nblue, mesh.span, mesh.win_off, pre.phi, pre.weight, pre.wphi,
        atom.Line["w0"].copy(), chi_int,
    )  # fmt: skip


class TestLineRates:
    def test_lstar_in_unit_interval(self):
        atom = Toy.make_toy_atom_3lv_()
        atmos = Toy.make_toy_atmos_(21, 1.0e9)
        mesh = GlobalMesh.merge_meshes_(_meshes(atom))
        pre = Setup.precompute_(atom, atmos, mesh)
        J, Psi, chi_tot, chi_int, _, _ = _sweep(atom, atmos, mesh, pre, pre.n_LTE)
        _, Lstar = _line_rates(atom, mesh, pre, J, Psi, chi_tot, chi_int)
        assert np.all(Lstar >= 0.0)
        assert np.all(Lstar <= 1.0 + 1e-12)


class TestBoundFreeRates:
    def _setup(self):
        atom = Toy.make_toy_atom_2lv_cont_()
        atmos = Toy.make_toy_atmos_(7, 1.0e9, Te_top=6.0e3, Te_bottom=1.1e4, Ne=1.0e12, Nt=1.0e12)
        mesh = GlobalMesh.merge_meshes_(_meshes(atom, nLambda=21))
        pre = Setup.precompute_(atom, atmos, mesh)
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
        atom = Toy.make_toy_atom_2lv_cont_()
        atmos = Toy.make_toy_atmos_(7, 1.0e9, Te_top=8.0e3, Ne=1.0e12, Nt=1.0e12)
        mesh = GlobalMesh.merge_meshes_(_meshes(atom, nLambda=21))
        pre = Setup.precompute_(atom, atmos, mesh)
        J, *_ = _sweep(atom, atmos, mesh, pre, pre.n_LTE)
        Rik, _, Rki_spon = Loop.bf_rates_(J, mesh, pre, atom, atmos)
        Rik_B, _, Rki_spon_B = Loop.bf_rates_(self._planck_J(mesh, atmos), mesh, pre, atom, atmos)
        assert np.all(Rik[:-1] < Rik_B[:-1])
        assert np.allclose(Rki_spon, Rki_spon_B, rtol=1e-12)  # spontaneous: no J
