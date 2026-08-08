"""Unit tests for the multilevel MALI loop (preconditioned rate matrix).

Oracles:
- single-channel back-reaction: a level connected by exactly one transition
  carries zero net flux in steady state, so the 0-1 excitation ratio must equal
  the pure two-level result (level 2 pushed high so its population cannot
  perturb the opacity)
- collision domination -> Boltzmann LTE; with a passive continuum and Planck
  prescribed radiation -> Saha-Boltzmann LTE (b-f rates detail-balance)
- the converged populations are independent of the operator: full, halved, and
  absent Lstar agree; iteration counts order the other way
- the two-level (eps, B) update and the multilevel rate-matrix update are the
  same physics: S_line from mali_multilevel_ matches (1-eps)*Jbar + eps*B
"""

import numpy as np

from spectra import Constants as CST
from spectra.Experimental.MALI import GlobalMesh, Loop, Structs
from spectra.Util import MeshUtil


def _build(atom, ND=41, Nt=1.0e10, Ne=1.0e10, Te=6.0e3):
    Zmax = 1.0e9
    Z = np.concatenate([[0.0], np.logspace(np.log10(Zmax * 1e-8), np.log10(Zmax), ND - 1)])
    atmos = Structs.make_toy_atmos_(ND, Zmax, Te_top=Te, Ne=Ne, Nt=Nt)
    atmos.Z = Z
    q = MeshUtil.make_full_line_mesh_(41, 2.5, 10.0)
    mesh = GlobalMesh.merge_meshes_([GlobalMesh.anchor_line_mesh_(q, w0, 2.5e5) for w0 in atom.Line["w0"]])
    pre = Structs.precompute_(atom, atmos, mesh)
    return atmos, mesh, pre


class TestMultilevelMALI:
    def test_single_channel_level_is_two_level(self):
        # level 2 reachable only through line (1,2): in steady state its single
        # channel carries zero net flux, so the 0-1 balance IS the two-level
        # problem. E2 is high so n2 (~5e-9) cannot shift the line opacity.
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

    def test_passive_continuum_reaches_saha_boltzmann(self):
        # unpreconditioned b-f rates (prescribed Planck radiation) next to
        # preconditioned line rates in the same Gamma; detailed balance must
        # carry the system to Saha-Boltzmann under collision domination
        atom = Structs.make_toy_atom_2lv_cont_()
        atmos, mesh, pre = _build(atom, Ne=1.0e20)
        assert atom.nCont == 2
        assert np.any(pre.Rik > 0.0)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10)
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

    def test_lte_start_is_fixed_point_when_thermalized(self):
        # optically very thick + collision-dominated: LTE start barely moves
        atom = Structs.make_toy_atom_3lv_()
        atmos, mesh, pre = _build(atom, Ne=1.0e20, Nt=1.0e12)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-10)
        assert r.niter <= 3

    def test_two_level_parametrization_equivalence(self):
        # the multilevel rate-matrix update and the (eps, B) two-level update
        # are the same physics: at convergence S_line must satisfy
        # S = (1-eps')*Jbar + eps'*B with eps' = Cji*Ne/(Cji*Ne + Aji), up to
        # the stimulated-emission correction, which the tolerance absorbs
        # (h*nu/k*Te ~ 4.8 here, so the correction is ~exp(-4.8) ~ 1e-2)
        atom = Structs.make_toy_atom_2lv_()
        atmos, mesh, pre = _build(atom)
        r = Loop.mali_multilevel_(atom, atmos, mesh, pre, tol=1e-11)
        Cji_Ne = pre.Cji_coe[:, 0] * atmos.Ne
        eps = Cji_Ne / (Cji_Ne + atom.Line["AJI"][0])
        S_pred = (1.0 - eps) * r.Jbar[0, :] + eps * pre.planck_w0[:, 0]
        assert np.allclose(r.S_line[0, :], S_pred, rtol=2e-2)
