"""Unit tests for the two-level MALI driver on the unified sweep.

Oracles (analytic, in (eps, B) form):
- eps = 1: S = B after one iteration
- sqrt(eps) surface law for a semi-infinite isothermal slab
- Lambda-iteration and a deliberately scaled operator reach the same fixed
  point, only more slowly
- wphi renormalization: a coarse window still thermalizes Jbar -> B at depth
"""

import numpy as np
import pytest

from spectra.Atomic import LTELib
from spectra.Experimental.MALI.v2 import GlobalMesh, Loop, Structs
from spectra.Util import MeshUtil


def _setup(ND=61, nLambda=41, tau_target=1.0e4, Te=6.0e3):
    atom = Structs.make_toy_atom_2lv_()
    Zmax = 1.0e9
    Z = np.concatenate([[0.0], np.logspace(np.log10(Zmax * 1e-8), np.log10(Zmax), ND - 1)])
    atmos = Structs.make_toy_atmos_(ND, Zmax, Te_top=Te)
    atmos.Z = Z
    q = MeshUtil.make_full_line_mesh_(nLambda, 2.5, 10.0)
    mesh = GlobalMesh.merge_meshes_([GlobalMesh.anchor_line_mesh_(q, atom.Line["w0"][0], 2.5e5)])
    pre = Structs.precompute_(atom, atmos, mesh)

    phi_win = pre.phi[mesh.win_off[0, 0] : mesh.win_off[0, 1], :]
    w_win = pre.weight[mesh.win_off[0, 0] : mesh.win_off[0, 1]]
    wphi = pre.wphi[:, 0]
    B = np.array([LTELib.planck_cm_(float(atom.Line["w0"][0]), float(T)) for T in atmos.Te])
    phi_c = phi_win[phi_win.shape[0] // 2, 0]
    chi0 = np.full(ND, tau_target / (phi_c * Zmax))
    return Z, chi0, B, mesh.wl, phi_win, w_win, wphi


class TestTwoLevelMALI:
    def test_eps_one_single_iteration(self):
        Z, chi0, B, wl, phi_win, w_win, wphi = _setup()
        r = Loop.mali_two_level_(Z, chi0, np.ones(Z.size), B, wl, phi_win, w_win, wphi)
        assert r.niter == 1
        assert np.allclose(r.S, B, rtol=1e-13)

    @pytest.mark.parametrize("eps_v", [1e-1, 1e-2])
    def test_sqrt_eps_surface_law(self, eps_v):
        Z, chi0, B, wl, phi_win, w_win, wphi = _setup(tau_target=100.0 / eps_v)
        eps = np.full(Z.size, eps_v)
        r = Loop.mali_two_level_(Z, chi0, eps, B, wl, phi_win, w_win, wphi, tol=1e-9)
        assert r.S[0] / B[0] == pytest.approx(np.sqrt(eps_v), rel=2e-2)

    def test_lambda_iteration_same_fixed_point_but_slow(self):
        Z, chi0, B, wl, phi_win, w_win, wphi = _setup(tau_target=1.0e4)
        eps = np.full(Z.size, 1e-2)
        r_mali = Loop.mali_two_level_(Z, chi0, eps, B, wl, phi_win, w_win, wphi, tol=1e-9)
        r_lam = Loop.mali_two_level_(Z, chi0, eps, B, wl, phi_win, w_win, wphi, tol=1e-9, use_lstar=False, itmax=5000)
        assert np.allclose(r_lam.S, r_mali.S, rtol=1e-5)
        assert r_mali.niter < 0.1 * r_lam.niter

    def test_scaled_lstar_same_solution(self):
        Z, chi0, B, wl, phi_win, w_win, wphi = _setup()
        eps = np.full(Z.size, 1e-2)
        r1 = Loop.mali_two_level_(Z, chi0, eps, B, wl, phi_win, w_win, wphi, tol=1e-10)
        r05 = Loop.mali_two_level_(Z, chi0, eps, B, wl, phi_win, w_win, wphi, tol=1e-10, lstar_scale=0.5)
        assert np.allclose(r05.S, r1.S, rtol=1e-6)
        assert r05.niter > r1.niter

    def test_wphi_renormalization_thermal_enclosure(self):
        Z, chi0, B, wl, phi_win, w_win, wphi = _setup(ND=41, nLambda=11, tau_target=1.0e6)
        assert np.abs(wphi - 1.0).max() > 1e-3
        r = Loop.mali_two_level_(Z, chi0, np.ones(Z.size), B, wl, phi_win, w_win, wphi)
        mid = Z.size // 2
        assert r.Jbar[mid] / B[mid] == pytest.approx(1.0, abs=1e-3)

    def test_lstar_within_unit_interval(self):
        Z, chi0, B, wl, phi_win, w_win, wphi = _setup()
        eps = np.full(Z.size, 1e-2)
        r = Loop.mali_two_level_(Z, chi0, eps, B, wl, phi_win, w_win, wphi)
        assert np.all(r.Lstar > 0.0)
        assert np.all(r.Lstar <= 1.0 + 1e-12)
