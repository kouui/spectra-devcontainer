"""Unit tests for the multilevel MALI loop on the unified axis.

Oracles:
- single-channel back-reaction: a level connected by exactly one transition
  carries zero net flux, so the 0-1 excitation ratio equals the two-level one
- collision domination -> Boltzmann; with ACTIVE continua (b-f rates from the
  solved field) -> Saha-Boltzmann, i.e. the b-f channel detail-balances
  against its own J
- the converged populations are independent of the operator: full, halved,
  and absent Lstar agree; iteration counts order the other way -- on the
  line-only toy AND on the overlap toy (two lines sharing the upper level,
  windows overlapping, continua covering both), where the self-term operator
  is weakest
- not converged at itmax is reported, never silently returned
- two-level (eps, B) form and the rate-matrix update are the same physics
"""

import numpy as np

from spectra import Constants as CST
from spectra.Atomic import LTELib
from spectra.Experimental.MALI.v2 import GlobalMesh, Loop, Structs
from spectra.Math import GaussLeg
from spectra.Util import MeshUtil


def _build(atom, ND=41, Nt=1.0e10, Ne=1.0e10, Te=6.0e3, nLambda=41):
    Zmax = 1.0e9
    Z = np.concatenate([[0.0], np.logspace(np.log10(Zmax * 1e-8), np.log10(Zmax), ND - 1)])
    atmos = Structs.make_toy_atmos_(ND, Zmax, Te_top=Te, Ne=Ne, Nt=Nt)
    atmos.Z = Z
    q = MeshUtil.make_full_line_mesh_(nLambda, 2.5, 10.0)
    meshes = [GlobalMesh.anchor_line_mesh_(q, w0, 2.5e5) for w0 in atom.Line["w0"]]
    meshes += [atom.Cont_mesh[kC, ::-1].copy() for kC in range(atom.nCont)]
    mesh = GlobalMesh.merge_meshes_(meshes)
    pre = Structs.precompute_(atom, atmos, mesh)
    return atmos, mesh, pre


class TestMultilevelMALI:
    def test_single_channel_level_is_two_level(self):
        E1 = CST.h_ * CST.c_ / 5000.0e-8
        E2 = E1 + CST.h_ * CST.c_ / 1500.0e-8
        atom3 = Structs.make_toy_atom_(
            np.array([1.0, 3.0, 5.0]),
            np.array([0.0, E1, E2]),
            [(0, 1), (1, 2)],
            np.array([1.0e8, 3.0e7]),
            np.array([1.0e-8, 1.0e-8]),
        )
        r3 = Loop.mali_multilevel_(atom3, *_build(atom3), tol=1e-12)
        atom2 = Structs.make_toy_atom_2lv_()
        r2 = Loop.mali_multilevel_(atom2, *_build(atom2), tol=1e-12)
        assert r3.converged
        assert r2.converged
        ratio3 = r3.n[:, 1] / r3.n[:, 0]
        ratio2 = r2.n[:, 1] / r2.n[:, 0]
        assert np.allclose(ratio3, ratio2, rtol=1e-8)
        assert r3.n[:, 2].max() < 1e-8

    def test_collision_dominated_reaches_boltzmann(self):
        atom = Structs.make_toy_atom_3lv_()
        atmos, mesh, pre = _build(atom, Ne=1.0e20)  # C*Ne = 1e12 >> Aji = 1e8
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10)
        assert np.allclose(r.n, pre.n_LTE, rtol=1e-4)
        assert r.niter < 10

    def test_active_continuum_reaches_saha_boltzmann(self):
        atom = Structs.make_toy_atom_2lv_cont_()
        atmos, mesh, pre = _build(atom, Ne=1.0e20)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10)
        assert r.converged
        assert np.allclose(r.n, pre.n_LTE, rtol=1e-4)

    def test_converged_populations_independent_of_lstar(self):
        atom = Structs.make_toy_atom_3lv_()
        atmos, mesh, pre = _build(atom, Ne=1.0e14)  # eps ~ 1e-2
        r_full = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-11)
        r_half = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-11, lstar_scale=0.5)
        r_none = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-11, use_lstar=False)
        assert np.abs(r_half.n - r_full.n).max() < 1e-9
        assert np.abs(r_none.n - r_full.n).max() < 1e-9
        assert r_full.niter < r_half.niter < r_none.niter

    def test_overlap_toy_fixed_point_independent_of_lstar(self):
        atom = Structs.make_toy_atom_overlap_()
        atmos, mesh, pre = _build(atom, Ne=1.0e14, ND=31)
        # the two line windows really overlap on the axis
        lo = max(mesh.Nblue[0], mesh.Nblue[1])
        hi = min(mesh.Nblue[0] + mesh.span[0], mesh.Nblue[1] + mesh.span[1])
        assert hi - lo > 10
        r_full = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-11)
        r_half = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-11, lstar_scale=0.5)
        r_none = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-11, use_lstar=False, itmax=20000)
        assert r_full.converged
        assert r_half.converged
        assert r_none.converged
        assert np.abs(r_half.n - r_full.n).max() < 1e-9
        assert np.abs(r_none.n - r_full.n).max() < 1e-9
        assert r_full.niter < r_half.niter < r_none.niter

    def test_overlap_toy_collision_domination(self):
        atom = Structs.make_toy_atom_overlap_()
        atmos, mesh, pre = _build(atom, Ne=1.0e20, ND=21)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10)
        assert r.converged
        assert np.allclose(r.n, pre.n_LTE, rtol=1e-4)

    def test_not_converged_is_reported(self):
        atom = Structs.make_toy_atom_3lv_()
        atmos, mesh, pre = _build(atom, Ne=1.0e14)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-11, use_lstar=False, itmax=3)
        assert not r.converged
        assert r.niter == 3

    def test_lte_start_is_fixed_point_when_thermalized(self):
        atom = Structs.make_toy_atom_3lv_()
        atmos, mesh, pre = _build(atom, Ne=1.0e20, Nt=1.0e12)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10)
        assert r.niter <= 3

    def test_two_level_parametrization_equivalence(self):
        atom = Structs.make_toy_atom_2lv_()
        atmos, mesh, pre = _build(atom)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-11)
        Cji_Ne = pre.Cji_coe[:, 0] * atmos.Ne
        eps = Cji_Ne / (Cji_Ne + atom.Line["AJI"][0])
        B = np.array([LTELib.planck_cm_(float(atom.Line["w0"][0]), float(T)) for T in atmos.Te])
        S_pred = (1.0 - eps) * r.Jbar[0, :] + eps * B
        assert np.allclose(r.S_line[0, :], S_pred, rtol=2e-2)

    def test_returned_radiation_belongs_to_returned_populations(self):
        # even when stopped early (itmax), J/Jbar/Lstar/S_line must be the
        # field OF the returned n, recomputed here independently
        atom = Structs.make_toy_atom_2lv_cont_()
        atmos, mesh, pre = _build(atom, Ne=1.0e14, ND=21)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10, itmax=4)
        assert not r.converged
        mus, wmus = GaussLeg.gauss_quad_coe_(0.0, 1.0, 4)
        n_abs = r.n * atmos.Nt[:, None]
        chi_int, S_line = Loop.line_coefficients_(
            n_abs, atom.Line["w0"].copy(), atom.Line["AJI"].copy(), atom.Line["BJI"].copy(),
            atom.Line["BIJ"].copy(), atom.Line["idxI"].copy(), atom.Line["idxJ"].copy(),
        )  # fmt: skip
        n_low, n_dag = Loop.continuum_coefficients_(
            n_abs,
            np.ascontiguousarray(pre.nj_by_ni[:, atom.nLine :]),
            atom.Cont["idxI"].copy(),
            atom.Cont["idxJ"].copy(),
        )
        J, Psi, chi_tot = Loop.unified_sweep_(
            atmos.Z, mesh.wl, mesh.col_ptr, mesh.col_tran, mesh.col_row, atom.nLine, atom.Line["w0"].copy(),
            pre.phi, chi_int, S_line, pre.alpha_win, pre.row_cont0, n_low, n_dag, pre.exp_hnu_kT,
            pre.twohc2_wl5, pre.bg_chi, pre.bg_eta, pre.bg_sca, r.J, pre.hn_bottom, mus, wmus,
        )  # fmt: skip
        Jbar, Lstar = Loop.line_rates_(
            J, Psi, chi_tot, mesh.wl, mesh.Nblue, mesh.span, mesh.win_off, pre.phi, pre.weight, pre.wphi,
            atom.Line["w0"].copy(), chi_int,
        )  # fmt: skip
        assert np.array_equal(r.J, J)
        assert np.array_equal(r.S_line, S_line)
        assert np.array_equal(r.Jbar, Jbar)
        assert np.array_equal(r.Lstar, Lstar)
