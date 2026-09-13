"""Unit tests for the unified formal sweep (v2.Loop.unified_sweep_).

Oracle: an independent pure-Python per-column implementation on the
2-level + continuum toy with a thermal background -- membership decided from
the raw meshes' wavelength ranges, opacity/emissivity written out from the
formulas, the production Feautrier solver called per angle. The kernel must
reproduce J, Psi and chi_tot column by column, with every axis wavelength
solved exactly once.
"""

import numpy as np
import pytest

from spectra import Constants as CST
from spectra.Enums import E_FEAUTRIER_ORDER
from spectra.Experimental.MALI.v2 import GlobalMesh, Loop, Structs
from spectra.Math import GaussLeg
from spectra.RadiativeTransfer import Feautrier
from spectra.Util import MeshUtil

XI_REF = 2.5e5


def _setup(ND=9, with_bg=True):
    atom = Structs.make_toy_atom_2lv_cont_()
    atmos = Structs.make_toy_atmos_(ND, 1.0e9, Te_top=6.0e3, Te_bottom=9.0e3, Ne=1.0e12, Nt=1.0e12)
    q = MeshUtil.make_full_line_mesh_(21, 2.5, 10.0)
    meshes = [GlobalMesh.anchor_line_mesh_(q, atom.Line["w0"][0], XI_REF)]
    meshes += [atom.Cont_mesh[kC, ::-1].copy() for kC in range(atom.nCont)]
    mesh = GlobalMesh.merge_meshes_(meshes)

    def bg_fn(wl_cm):
        return 1.0e-14 * (wl_cm / 5000.0e-8) ** 2 * np.linspace(1.0, 3.0, ND), np.zeros(ND)

    pre = Structs.precompute_(atom, atmos, mesh, bg_fn=bg_fn if with_bg else None)
    # a non-LTE population state so nothing cancels by accident
    n = pre.n_LTE * np.array([1.0, 3.0, 0.5])
    n /= n.sum(axis=1)[:, None]
    return atom, atmos, mesh, meshes, pre, n


def _run_kernel(atom, atmos, mesh, pre, n, n_angle=3):
    mus, wmus = GaussLeg.gauss_quad_coe_(0.0, 1.0, n_angle)
    n_abs = n * atmos.Nt[:, None]
    chi_int, S_line = Loop.line_coefficients_(
        n_abs, atom.Line["w0"].copy(), atom.Line["AJI"].copy(), atom.Line["BJI"].copy(), atom.Line["BIJ"].copy(),
        atom.Line["idxI"].copy(), atom.Line["idxJ"].copy(),
    )  # fmt: skip
    n_low, n_dag = Loop.continuum_coefficients_(
        n_abs, np.ascontiguousarray(pre.nj_by_ni[:, atom.nLine :]), atom.Cont["idxI"].copy(), atom.Cont["idxJ"].copy()
    )
    return Loop.unified_sweep_(
        atmos.Z, mesh.wl, mesh.col_ptr, mesh.col_tran, mesh.col_row, atom.nLine,
        atom.Line["w0"].copy(), pre.phi, chi_int, S_line, pre.alpha_win, pre.row_cont0, n_low, n_dag,
        pre.exp_hnu_kT, pre.twohc2_wl5, pre.bg_chi, pre.bg_eta, pre.hn_bottom, mus, wmus,
    ), (mus, wmus)  # fmt: skip


def _reference(atom, atmos, mesh, meshes, pre, n, mus, wmus):
    """Everything spelled out; membership from the raw wavelength ranges."""
    ND = atmos.ND
    n_abs = n * atmos.Nt[:, None]
    hnu4pi = lambda wl: CST.h_ * (CST.c_ / wl) / (4.0 * CST.pi_)  # noqa: E731
    J = np.zeros((mesh.wl.shape[0], ND))
    Psi = np.zeros_like(J)
    chi_tot = np.zeros_like(J)
    eps = 1.0e-3 * min(np.min(np.diff(m)) for m in meshes)
    for iw, wl in enumerate(mesh.wl):
        chi = pre.bg_chi[iw].copy()
        eta = pre.bg_eta[iw].copy()
        for t, m in enumerate(meshes):
            if not (m[0] - eps <= wl <= m[-1] + eps):
                continue
            if t < atom.nLine:
                row = mesh.win_off[t, 0] + iw - mesh.Nblue[t]
                ni, nj = n_abs[:, atom.Line["idxI"][t]], n_abs[:, atom.Line["idxJ"][t]]
                # production convention: h*nu at line center, rescaled by w0/wl
                f = hnu4pi(atom.Line["w0"][t]) * (atom.Line["w0"][t] / wl)
                chi_l = f * (ni * atom.Line["BIJ"][t] - nj * atom.Line["BJI"][t]) * pre.phi[row]
                eta_l = f * nj * atom.Line["AJI"][t] * pre.phi[row]
                chi += chi_l
                eta += eta_l
            else:
                kC = t - atom.nLine
                row = mesh.win_off[t, 0] + iw - mesh.Nblue[t]
                a = pre.alpha_win[row - pre.row_cont0]
                n_dag = n_abs[:, atom.Cont["idxJ"][kC]] / pre.nj_by_ni[:, atom.nLine + kC]
                stim = n_dag * np.exp(-CST.h_ * CST.c_ / (wl * CST.k_ * atmos.Te))
                chi += a * (n_abs[:, atom.Cont["idxI"][kC]] - stim)
                eta += a * 2.0 * CST.h_ * CST.c_**2 / wl**5 * stim
        S = eta / chi
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (chi[1:] + chi[:-1]) * np.diff(atmos.Z))])
        for mu, wmu in zip(mus, wmus, strict=True):
            res = Feautrier.formal_improved_RH_(
                tau, S, mu, 0.0, 0.0, 0.0, pre.hn_bottom[iw], E_FEAUTRIER_ORDER.SECOND, True
            )
            J[iw] += wmu * res.j
            Psi[iw] += wmu * res.Psi
        chi_tot[iw] = chi
    return J, Psi, chi_tot


class TestUnifiedSweep:
    def test_matches_reference_with_background(self):
        atom, atmos, mesh, meshes, pre, n = _setup()
        (J, Psi, chi_tot), (mus, wmus) = _run_kernel(atom, atmos, mesh, pre, n)
        J_ref, Psi_ref, chi_ref = _reference(atom, atmos, mesh, meshes, pre, n, mus, wmus)
        assert J.shape == (mesh.wl.shape[0], atmos.ND)
        assert np.allclose(chi_tot, chi_ref, rtol=1e-13, atol=0.0)
        assert np.allclose(J, J_ref, rtol=1e-12, atol=0.0)
        assert np.allclose(Psi, Psi_ref, rtol=1e-12, atol=0.0)

    def test_matches_reference_without_background(self):
        atom, atmos, mesh, meshes, pre, n = _setup(with_bg=False)
        (J, Psi, chi_tot), (mus, wmus) = _run_kernel(atom, atmos, mesh, pre, n)
        J_ref, Psi_ref, chi_ref = _reference(atom, atmos, mesh, meshes, pre, n, mus, wmus)
        assert np.allclose(chi_tot, chi_ref, rtol=1e-13, atol=0.0)
        assert np.allclose(J, J_ref, rtol=1e-12, atol=0.0)
        assert np.allclose(Psi, Psi_ref, rtol=1e-12, atol=0.0)

    def test_every_column_has_opacity_and_bounded_operator(self):
        atom, atmos, mesh, _, pre, n = _setup()
        (J, Psi, chi_tot), _ = _run_kernel(atom, atmos, mesh, pre, n)
        assert np.all(chi_tot > 0.0)
        assert np.all(J >= 0.0)
        assert np.all(Psi >= 0.0)
        assert np.all(Psi <= 1.0 + 1e-12)

    def test_thick_isothermal_column_thermalizes(self):
        # deep, isothermal, dense slab: J -> B = S at the bottom of every column
        atom = Structs.make_toy_atom_2lv_cont_()
        atmos = Structs.make_toy_atmos_(21, 1.0e14, Te_top=7.0e3, Ne=1.0e14, Nt=1.0e14)
        q = MeshUtil.make_full_line_mesh_(21, 2.5, 10.0)
        meshes = [GlobalMesh.anchor_line_mesh_(q, atom.Line["w0"][0], XI_REF)]
        meshes += [atom.Cont_mesh[kC, ::-1].copy() for kC in range(atom.nCont)]
        mesh = GlobalMesh.merge_meshes_(meshes)
        pre = Structs.precompute_(atom, atmos, mesh)
        (J, _, chi_tot), _ = _run_kernel(atom, atmos, mesh, pre, pre.n_LTE)
        tau_tot = (0.5 * (chi_tot[:, 1:] + chi_tot[:, :-1]) * np.diff(atmos.Z)[None, :]).sum(axis=1)
        thick = tau_tot > 10.0
        assert thick.any()
        assert J[thick, -1] / pre.hn_bottom[thick] == pytest.approx(1.0, rel=0.05)
