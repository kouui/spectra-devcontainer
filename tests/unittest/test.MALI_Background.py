"""Unit tests for the MALI background continuum opacities (RH ports).

Oracles:
- Thomson: the textbook cross section times ne
- H-minus b-f: reproduces the Geltman table peak (~3.95e-17 cm^2 near 850 nm)
  through the assembled extinction at known conditions
- H b-f: the Lyman-edge cross section is the classic 6.30e-18 cm^2 (Gaunt
  factor ~0.8 on sigma0 = 7.91e-18)
- outside validity windows (beyond the H- b-f table, blueward of an edge)
  contributions vanish
- the multilevel sweep with EMPTY background arrays is bit-for-bit the
  line-only path, and with a tiny background it converges to the same physics
"""

import numpy as np

from spectra import Constants as CST
from spectra.Experimental.MALI import Background as BG
from spectra.Experimental.MALI import GlobalMesh, Loop, Structs
from spectra.Util import MeshUtil


def test_thomson():
    Ne = np.array([1.0e10, 2.0e12])
    chi = BG.thomson_(Ne)
    assert np.allclose(chi / Ne, 6.652e-25, rtol=1e-3)


def test_hydrogen_bf_edge():
    Te = np.array([6000.0])
    n1 = np.array([1.0])
    erg1 = 0.0
    erg_c = CST.E_Rydberg_H_
    wl_edge = CST.h_ * CST.c_ / erg_c
    chi = BG.hydrogen_bf_(wl_edge * 0.9999, Te, n1, erg1, erg_c)
    stim = 1.0 - np.exp(-CST.h_ * CST.c_ / (wl_edge * CST.k_ * Te[0]))
    sigma = chi[0] / stim
    # Seaton's approximate Gaunt factor puts the edge ~2.4% below the exact one
    assert abs(sigma / 6.30e-18 - 1.0) < 0.03
    # blueward of nothing: a wavelength beyond the edge cannot ionize
    assert BG.hydrogen_bf_(wl_edge * 1.01, Te, n1, erg1, erg_c)[0] == 0.0


def test_hminus_bf_table_peak():
    Te = np.array([6000.0])
    Ne = np.array([1.0])
    nH = np.array([1.0])
    chi = BG.hminus_bf_(850.0e-7, Te, Ne, nH)
    stim = np.exp(-CST.h_ * CST.c_ / (850.0e-7 * CST.k_ * Te[0]))
    alpha = chi[0] / ((1.0 - stim) * np.asarray(BG.phi_hmin_(Te))[0])
    assert abs(alpha / 3.95e-17 - 1.0) < 0.01
    assert BG.hminus_bf_(1700.0e-7, Te, Ne, nH)[0] == 0.0  # beyond the table


def test_background_assembly_positive():
    Te = np.array([5000.0, 8000.0, 20000.0])
    Ne = np.array([1.0e13, 1.0e11, 1.0e10])
    nH = np.tile(np.array([1.0e15, 1.0e8, 1.0e7, 1.0e6, 1.0e6]), (3, 1))
    Np = np.array([1.0e11, 1.0e11, 1.0e10])
    erg = np.array([0.0, 0.75, 0.889, 0.9375, 0.96]) * CST.E_Rydberg_H_
    chi = BG.background_chi_(5000.0e-8, Te, Ne, nH, Np, erg, CST.E_Rydberg_H_)
    assert np.all(chi > 0.0)
    # H-minus dominates over Thomson at photospheric conditions
    assert chi[0] > 10.0 * BG.thomson_(Ne)[0]
    # Thomson is a scatterer, never part of the thermal-absorber sum: at a
    # 1e5 K, ne = 1e10 column the sum must be far below sigma_T * ne
    Te_hot, Ne_hot = np.array([1.0e5]), np.array([1.0e10])
    nH_hot = np.array([[1.0e4, 1.0, 1.0, 1.0, 1.0]])
    chi_hot = BG.background_chi_(1.0e-5, Te_hot, Ne_hot, nH_hot, Ne_hot, erg, CST.E_Rydberg_H_)
    assert chi_hot[0] < 0.01 * BG.thomson_(Ne_hot)[0]


def test_sweep_empty_background_is_line_only():
    atom = Structs.make_toy_atom_3lv_()
    atmos = Structs.make_toy_atmos_(21, 1.0e9)
    q = MeshUtil.make_full_line_mesh_(21, 2.5, 10.0)
    meshes = [GlobalMesh.anchor_line_mesh_(q, w0, 2.5e5) for w0 in atom.Line["w0"]]
    mesh = GlobalMesh.merge_meshes_(meshes)
    pre = Structs.precompute_(atom, atmos, mesh)
    assert pre.bg_chi.shape == (0, atmos.ND)
    r0 = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10)

    # a vanishingly small background must not change the converged physics
    pre_bg = Structs.precompute_(atom, atmos, mesh, bg_chi_fn=lambda wl_cm: np.full(atmos.ND, 1.0e-30))
    assert pre_bg.bg_chi.shape[0] == pre.phi.shape[0]
    r1 = Loop.mali_multilevel_(atom, atmos, mesh, pre_bg, tol=1e-10)
    assert np.abs(r1.n - r0.n).max() < 1.0e-8


def test_cij_dep_broadcast_matches_scalar():
    atom = Structs.make_toy_atom_2lv_()
    atmos = Structs.make_toy_atmos_(15, 1.0e9)
    q = MeshUtil.make_full_line_mesh_(21, 2.5, 10.0)
    mesh = GlobalMesh.merge_meshes_([GlobalMesh.anchor_line_mesh_(q, atom.Line["w0"][0], 2.5e5)])
    pre0 = Structs.precompute_(atom, atmos, mesh)
    Cij_dep = np.tile(atom.Cij_coe, (atmos.ND, 1))
    pre1 = Structs.precompute_(atom, atmos, mesh, Cij_dep=Cij_dep)
    assert np.array_equal(pre0.Cij_coe, pre1.Cij_coe)
    assert np.array_equal(pre0.Cji_coe, pre1.Cji_coe)


if __name__ == "__main__":
    test_thomson()
    test_hydrogen_bf_edge()
    test_hminus_bf_table_peak()
    test_background_assembly_positive()
    test_sweep_empty_background_is_line_only()
    test_cij_dep_broadcast_matches_scalar()
    print("all background tests passed")
