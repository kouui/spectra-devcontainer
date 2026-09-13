"""Unit tests for coherent scattering in the v2 MALI loop.

Oracles:
- a background-only slab (no transitions) with constant sigma: iterating
  the unified sweep with J_dag <- J is the Lambda-iteration of the discrete
  problem S = eps*B + (1-eps)*J, eps = chi_abs/(chi_abs + sigma), which a
  direct loop over the production Feautrier solver reproduces
- the converged multilevel solution does not depend on the starting J_dag
  (zero, Planck, warm) nor on the operator (full / absent Lstar); the
  operator only changes the iteration count
"""

import numpy as np
import pytest

from spectra.Atomic import LTELib
from spectra.Enums import E_FEAUTRIER_ORDER
from spectra.Experimental.MALI.v2 import GlobalMesh, Loop, Structs
from spectra.Math import GaussLeg
from spectra.RadiativeTransfer import Feautrier
from spectra.Util import MeshUtil

XI_REF = 2.5e5


def _build_cont2(ND=21, sigma_scale=3.0, bg_level=1.0e-14):
    atom = Structs.make_toy_atom_2lv_cont_()
    atmos = Structs.make_toy_atmos_(ND, 1.0e9, Te_top=6.0e3, Te_bottom=9.0e3, Ne=1.0e12, Nt=1.0e12)
    q = MeshUtil.make_full_line_mesh_(21, 2.5, 10.0)
    meshes = [GlobalMesh.anchor_line_mesh_(q, w0, XI_REF) for w0 in atom.Line["w0"]]
    meshes += [atom.Cont_mesh[kC, ::-1].copy() for kC in range(atom.nCont)]
    mesh = GlobalMesh.merge_meshes_(meshes)

    def bg_fn(wl_cm):
        chi = bg_level * (wl_cm / 5000.0e-8) ** 2 * np.linspace(1.0, 3.0, ND)
        return chi, sigma_scale * chi

    pre = Structs.precompute_(atom, atmos, mesh, bg_fn=bg_fn)
    return atom, atmos, mesh, pre


def _planck_table(wl, Te):
    return np.array([LTELib.planck_cm_(wl[:], Te[k]) for k in range(Te.shape[0])]).T


class TestScatteringSlab:
    def test_constant_sigma_slab_matches_direct_lambda_iteration(self):
        ND, Nspect = 15, 3
        Z = np.linspace(0.0, 1.0e9, ND)
        Te = np.linspace(5.0e3, 8.0e3, ND)
        wl = np.array([4000.0e-8, 5000.0e-8, 6000.0e-8])
        chi_abs = np.outer([1.0e-9, 2.0e-9, 4.0e-9], np.linspace(1.0, 4.0, ND))
        sigma = np.outer([9.0e-9, 4.0e-9, 1.0e-9], np.ones(ND))
        B = _planck_table(wl, Te)
        mus, wmus = GaussLeg.gauss_quad_coe_(0.0, 1.0, 3)
        # an axis with no transitions at all: every CSR row is empty
        empty_l = np.empty((0, ND))
        empty_i = np.empty(0, dtype=np.int64)
        col_ptr = np.zeros(Nspect + 1, dtype=np.int64)

        def sweep(J_dag):
            return Loop.unified_sweep_(
                Z, wl, col_ptr, empty_i, empty_i, 0, np.empty(0), empty_l, empty_l, empty_l, np.empty(0), 0,
                empty_l, empty_l, np.ones((Nspect, ND)), np.zeros(Nspect), chi_abs, chi_abs * B, sigma, J_dag,
                B[:, -1].copy(), mus, wmus,
            )  # fmt: skip

        J = np.zeros_like(B)
        J_new, Psi, chi_tot = sweep(J)
        done = False
        for _ in range(400):
            J_new, Psi, chi_tot = sweep(J)
            done = np.abs(J_new - J).max() < 1e-12 * np.abs(J_new).max()
            J = J_new
            if done:
                break
        assert done
        assert np.allclose(chi_tot, chi_abs + sigma, rtol=1e-14, atol=0.0)
        assert np.all(Psi > 0.0)
        assert np.all(Psi <= 1.0 + 1e-12)

        # the direct Lambda-iteration written out per column
        eps = chi_abs / (chi_abs + sigma)
        J_ref = np.zeros_like(B)
        for _ in range(400):
            J_prev = J_ref.copy()
            for iw in range(Nspect):
                S = eps[iw] * B[iw] + (1.0 - eps[iw]) * J_prev[iw]
                tau = np.concatenate([[0.0], np.cumsum(0.5 * (chi_tot[iw, 1:] + chi_tot[iw, :-1]) * np.diff(Z))])
                J_ref[iw] = 0.0
                for mu, wmu in zip(mus, wmus, strict=True):
                    res = Feautrier.formal_improved_RH_(
                        tau, S, mu, 0.0, 0.0, 0.0, B[iw, -1], E_FEAUTRIER_ORDER.SECOND, True
                    )
                    J_ref[iw] += wmu * res.j
            if np.abs(J_ref - J_prev).max() < 1e-12 * np.abs(J_ref).max():
                break
        assert np.allclose(J, J_ref, rtol=1e-9, atol=0.0)
        # the re-emitted sigma*J raises the field above the sink-only sweep
        assert np.all(J[:, 0] > sweep(np.zeros_like(B))[0][:, 0])


class TestScatteringLoop:
    def test_converged_solution_independent_of_j_dag_start(self):
        atom, atmos, mesh, pre = _build_cont2()
        assert pre.bg_sca.max() > 0.0
        kw = {"tol": 1e-12, "tol_J": 1e-10, "itmax": 5000}
        r_B = Loop.mali_multilevel_(atom, atmos, mesh, pre, **kw)
        r_0 = Loop.mali_multilevel_(atom, atmos, mesh, pre, J_init=np.zeros_like(r_B.J), **kw)
        r_w = Loop.mali_multilevel_(atom, atmos, mesh, pre, J_init=r_B.J, n_init=r_B.n, **kw)
        assert r_B.converged
        assert r_0.converged
        assert r_w.converged
        assert np.allclose(r_0.n, r_B.n, rtol=1e-8, atol=1e-30)
        assert np.allclose(r_w.n, r_B.n, rtol=1e-8, atol=1e-30)
        assert np.allclose(r_0.J, r_B.J, rtol=1e-8, atol=0.0)
        assert np.allclose(r_w.J, r_B.J, rtol=1e-8, atol=0.0)
        assert r_w.niter < r_B.niter

    def test_operator_changes_count_not_fixed_point(self):
        atom, atmos, mesh, pre = _build_cont2()
        kw = {"tol": 1e-12, "tol_J": 1e-10, "itmax": 5000}
        r_mali = Loop.mali_multilevel_(atom, atmos, mesh, pre, **kw)
        r_lam = Loop.mali_multilevel_(atom, atmos, mesh, pre, use_lstar=False, **kw)
        assert r_mali.converged
        assert r_lam.converged
        # the absolute dn stop leaves the crawling Lambda-iteration a
        # remaining error of step/(1 - rho), far above its last step
        assert np.allclose(r_mali.n, r_lam.n, rtol=1e-6, atol=1e-30)
        assert np.allclose(r_mali.J, r_lam.J, rtol=1e-6, atol=0.0)
        assert r_mali.niter < r_lam.niter
        assert np.all(r_mali.Lstar > 0.0)
        assert np.all(r_mali.Lstar <= 1.0)

    def test_returned_field_is_the_fixed_point(self):
        # one more sweep on the returned (n, J) must reproduce J to the
        # convergence tolerance: the scattering source has stopped moving
        atom, atmos, mesh, pre = _build_cont2()
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-12, tol_J=1e-10, itmax=5000)
        r2 = Loop.mali_multilevel_(atom, atmos, mesh, pre, n_init=r.n, J_init=r.J, tol=1e-12, tol_J=1e-10, itmax=1)
        assert np.allclose(r2.J, r.J, rtol=1e-9, atol=0.0)
        assert np.allclose(r2.n, r.n, rtol=1e-9, atol=1e-30)

    def test_scattering_changes_the_solution(self):
        # guards against a silently disconnected slot: a background thick
        # enough (tau ~ 1 over the slab) for sigma to matter
        atom, atmos, mesh, pre = _build_cont2(bg_level=1.0e-9)
        pre0 = _build_cont2(sigma_scale=0.0, bg_level=1.0e-9)[3]
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10)
        r0 = Loop.mali_multilevel_(atom, atmos, mesh, pre0, tol=1e-10)
        assert r.converged
        assert not np.allclose(r.J, r0.J, rtol=1e-2)
        assert not np.allclose(r.n, r0.n, rtol=1e-3)


def _sweep_at(atom, atmos, mesh, pre, n, J_dag, n_angle=3):
    mus, wmus = GaussLeg.gauss_quad_coe_(0.0, 1.0, n_angle)
    n_abs = n * atmos.Nt[:, None]
    chi_int, S_line = Loop.line_coefficients_(
        n_abs, atom.Line["w0"].copy(), atom.Line["AJI"].copy(), atom.Line["BJI"].copy(),
        atom.Line["BIJ"].copy(), atom.Line["idxI"].copy(), atom.Line["idxJ"].copy(),
    )  # fmt: skip
    n_low, n_dag = Loop.continuum_coefficients_(
        n_abs, np.ascontiguousarray(pre.nj_by_ni[:, atom.nLine :]),
        atom.Cont["idxI"].copy(), atom.Cont["idxJ"].copy(),
    )  # fmt: skip
    J, Psi, chi_tot = Loop.unified_sweep_(
        atmos.Z, mesh.wl, mesh.col_ptr, mesh.col_tran, mesh.col_row, atom.nLine, atom.Line["w0"].copy(),
        pre.phi, chi_int, S_line, pre.alpha_win, pre.row_cont0, n_low, n_dag, pre.exp_hnu_kT,
        pre.twohc2_wl5, pre.bg_chi, pre.bg_eta, pre.bg_sca, J_dag, pre.hn_bottom, mus, wmus,
    )  # fmt: skip
    return J, Psi, chi_tot, chi_int, (mus, wmus)


class TestScatteringOperator:
    def test_one_iteration_applies_the_diagonal_correction(self):
        # single-step oracle: the driver's J after one update is the sweep's
        # J corrected by (J - w J_dag)/(1 - w), w = Psi sigma/chi, and its
        # Lstar is built from Psi/(1 - w); a dropped, mis-signed or
        # sigma*J_dag-leaking correction changes these exactly
        atom, atmos, mesh, pre = _build_cont2(sigma_scale=30.0, bg_level=1.0e-8)
        n0 = pre.n_LTE * np.array([1.0, 3.0, 0.5])
        n0 /= n0.sum(axis=1)[:, None]
        J_dag0 = _planck_table(mesh.wl, atmos.Te) * 0.3
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, n_init=n0, J_init=J_dag0, itmax=1, n_angle=3)

        def corrected(n, J_dag):
            J, Psi, chi_tot, chi_int, _ = _sweep_at(atom, atmos, mesh, pre, n, J_dag)
            w = np.clip(Psi * pre.bg_sca / chi_tot, 0.0, Loop._W_SCA_MAX)
            Lstar = Loop.line_rates_(
                J, Psi / (1.0 - w), chi_tot, mesh.wl, mesh.Nblue, mesh.span, mesh.win_off,
                pre.phi, pre.weight, pre.wphi, atom.Line["w0"].copy(), chi_int,
            )[1]  # fmt: skip
            return (J - w * J_dag) / (1.0 - w), Lstar, w

        J1, _, w = corrected(n0, J_dag0)
        assert w.max() > 0.5  # the operator actually bites here
        # r.J is the diagnostic sweep on the updated n with J_dag = J1
        J2, Lstar2, _ = corrected(r.n, J1)
        assert np.array_equal(r.J, J2)
        assert np.array_equal(r.Lstar, Lstar2)
        assert np.all(r.Lstar > 0.0)
        assert np.all(r.Lstar <= 1.0)

    def test_thick_pure_scattering_column_stays_finite(self):
        # sigma/chi -> 1 and Psi -> 1: the uncapped weight would divide by
        # zero; capped, the run converges to a finite, positive field
        atom, atmos, mesh, pre = _build_cont2(sigma_scale=1.0e6, bg_level=1.0e-6)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, n_angle=3)
        assert r.converged
        assert np.all(np.isfinite(r.J))
        assert np.all(r.J > 0.0)


class TestScatteringFixedPoint:
    @pytest.mark.parametrize(("sigma_scale", "bg_level"), [(30.0, 1.0e-8), (3.0e3, 1.0e-7)])
    def test_converged_field_solves_the_linear_scattering_problem(self, sigma_scale, bg_level):
        # with the populations frozen at the returned n, one column's
        # scattering problem J = Lambda[(eta_th + sigma J)/chi] is linear:
        # build Lambda by unit-source sweeps and solve it directly. the
        # driver's J (operator-accelerated) must be that solution -- this is
        # what a lagged Lambda-iteration fails at when sigma/chi -> 1 and
        # the column is thick
        atom, atmos, mesh, pre = _build_cont2(sigma_scale=sigma_scale, bg_level=bg_level)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, n_angle=3, tol_J=1e-9)
        assert r.converged
        mus, wmus = GaussLeg.gauss_quad_coe_(0.0, 1.0, 3)
        n_abs = r.n * atmos.Nt[:, None]
        chi_int, S_line = Loop.line_coefficients_(
            n_abs, atom.Line["w0"].copy(), atom.Line["AJI"].copy(), atom.Line["BJI"].copy(),
            atom.Line["BIJ"].copy(), atom.Line["idxI"].copy(), atom.Line["idxJ"].copy(),
        )  # fmt: skip
        n_low, n_dag = Loop.continuum_coefficients_(
            n_abs, np.ascontiguousarray(pre.nj_by_ni[:, atom.nLine :]),
            atom.Cont["idxI"].copy(), atom.Cont["idxJ"].copy(),
        )  # fmt: skip
        # J_dag = 0: the sweep returns Lambda[eta_th/chi] (with boundary) and chi
        J0, _, chi_tot = Loop.unified_sweep_(
            atmos.Z, mesh.wl, mesh.col_ptr, mesh.col_tran, mesh.col_row, atom.nLine, atom.Line["w0"].copy(),
            pre.phi, chi_int, S_line, pre.alpha_win, pre.row_cont0, n_low, n_dag, pre.exp_hnu_kT,
            pre.twohc2_wl5, pre.bg_chi, pre.bg_eta, pre.bg_sca, np.zeros_like(pre.bg_sca), pre.hn_bottom, mus, wmus,
        )  # fmt: skip
        ND = atmos.ND
        for iw in range(0, mesh.wl.shape[0], 7):
            chi = chi_tot[iw]
            tau = np.concatenate([[0.0], np.cumsum(0.5 * (chi[1:] + chi[:-1]) * np.diff(atmos.Z))])
            L = np.empty((ND, ND))
            for k in range(ND):
                e = np.zeros(ND)
                e[k] = 1.0
                L[:, k] = sum(
                    wmu
                    * Feautrier.formal_improved_RH_(tau, e, mu, 0.0, 0.0, 0.0, 0.0, E_FEAUTRIER_ORDER.SECOND, True).j
                    for mu, wmu in zip(mus, wmus, strict=True)
                )
            J_exact = np.linalg.solve(np.eye(ND) - L * (pre.bg_sca[iw] / chi)[None, :], J0[iw])
            assert np.allclose(r.J[iw], J_exact, rtol=1e-6, atol=0.0), iw
