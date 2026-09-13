"""Unit tests for the active-continuum solver (ContinuumRT).

Oracles:
- collision domination: with active continua the SE must still land on
  Saha-Boltzmann (the b-f channel detail-balances against the solved J)
- boundedness: in an isothermal slab with a Planck lower boundary the solved
  continuum J lies in [B/2, B] at the bottom (thin vs. thick limits), reaches
  B where the column is optically thick, and never exceeds B anywhere
- consistency: at the thermalized depths the active rates reproduce the
  passive Planck-prescribed rates of the precompute tier
"""

import numpy as np

from spectra.Atomic import LTELib
from spectra.Experimental.MALI.v1 import ContinuumRT, GlobalMesh, Loop, Structs
from spectra.Math import GaussLeg
from spectra.Util import MeshUtil


def _setup(Ne=1.0e20, ND=21, length=1.0e9):
    atom = Structs.make_toy_atom_2lv_cont_()
    atmos = Structs.make_toy_atmos_(ND, length, Ne=Ne)
    q = MeshUtil.make_full_line_mesh_(21, 2.5, 10.0)
    mesh = GlobalMesh.merge_meshes_([GlobalMesh.anchor_line_mesh_(q, atom.Line["w0"][0], 2.5e5)])
    pre = Structs.precompute_(atom, atmos, mesh)
    cont_rt = ContinuumRT.build_cont_rt_(atom, atmos)
    return atom, atmos, mesh, pre, cont_rt


def test_collision_domination_saha_boltzmann():
    atom, atmos, mesh, pre, cont_rt = _setup()
    r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10, cont_rt=cont_rt)
    b = r.n / pre.n_LTE
    assert np.abs(b[5:] - 1.0).max() < 1.0e-6


def test_continuum_J_bounded_by_planck():
    # long column: part of the axis optically thick, the rest stays thin
    atom, atmos, _mesh, pre, cont_rt = _setup(length=1.0e11)
    mus, wmus = GaussLeg.gauss_quad_coe_(0.0, 1.0, 4)
    n_abs = pre.n_LTE * atmos.Nt[:, None]
    J = ContinuumRT.continuum_sweep_(
        atmos.Z, n_abs,
        np.ascontiguousarray(atom.Cont["idxI"][:]),
        np.ascontiguousarray(atom.Cont["idxJ"][:]),
        np.ascontiguousarray(pre.nj_by_ni[:, atom.nLine :]),
        cont_rt.alpha_ax, cont_rt.exp_hnu_kT, cont_rt.twohc2_wl5,
        cont_rt.bg_chi, cont_rt.bg_eta, cont_rt.hn_bottom, mus, wmus,
    )  # fmt: skip
    B = np.array([LTELib.planck_cm_(float(wl), atmos.Te[0]) for wl in cont_rt.wl_ax])
    # per-wavelength total vertical optical depth from the LTE opacity
    idxI, idxJ = atom.Cont["idxI"][:], atom.Cont["idxJ"][:]
    n_dag = n_abs[:, idxJ] / pre.nj_by_ni[:, atom.nLine :]  # (ND, nCont)
    stim = n_dag[None, :, :] * cont_rt.exp_hnu_kT[:, :, None]  # (nAx, ND, nCont)
    chi = np.einsum("cw,wkc->wk", cont_rt.alpha_ax, n_abs[None, :, idxI] - stim)
    dz = np.diff(atmos.Z)
    tau_tot = (0.5 * (chi[:, 1:] + chi[:, :-1]) * dz[None, :]).sum(axis=1)

    assert np.all(B[:, None] * (1.0 + 1.0e-10) >= J)
    # bottom: Planck boundary gives J >= B/2 even in the thin limit
    assert np.all(J[:, -1] >= 0.5 * B * (1.0 - 1.0e-10))
    thick = tau_tot > 10.0
    assert thick.any()
    assert np.abs(J[thick, -1] / B[thick] - 1.0).max() < 0.05


def test_active_rates_match_passive_at_depth():
    # every axis wavelength optically thick so J -> B over the full rate range
    atom, atmos, _mesh, pre, cont_rt = _setup(length=1.0e13)
    mus, wmus = GaussLeg.gauss_quad_coe_(0.0, 1.0, 4)
    Rik, Rki_stim, Rki_spon = ContinuumRT.bf_rates_(
        cont_rt, atom, atmos, np.ascontiguousarray(pre.nj_by_ni[:, atom.nLine :]),
        pre.n_LTE, mus, wmus,
    )  # fmt: skip
    k = atmos.ND - 1
    # spontaneous recombination never depends on J: quadrature-level agreement
    assert np.abs(Rki_spon[k] / pre.Rki_spon[k] - 1.0).max() < 0.02
    # J -> B at depth, so the J-driven rates approach the Planck-driven ones
    assert np.abs(Rik[k] / pre.Rik[k] - 1.0).max() < 0.05
    assert np.abs(Rki_stim[k] / pre.Rki_stim[k] - 1.0).max() < 0.05


if __name__ == "__main__":
    test_collision_domination_saha_boltzmann()
    test_continuum_J_bounded_by_planck()
    test_active_rates_match_passive_at_depth()
    print("all ContinuumRT tests passed")
