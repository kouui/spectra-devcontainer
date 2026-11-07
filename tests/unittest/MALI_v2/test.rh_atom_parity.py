"""Parity of the package RH reader and builder with the frozen FALC helper.

`_falc_setup.parse_h6_atom_` and `build_atom_` are what the frozen v2 case
uses; `Struct.Atom.init_Atom_from_RH_` must hand the later package setup the
same numbers. Temporary: it goes with this directory.

Comparisons are exact (`assert_array_equal`) wherever both sides run the same
arithmetic. The one exception is the line wavelength: the frozen toy atom takes
``h*c/dE`` and `Atom` takes ``c/(dE/h)``, which differ in the last bits.
"""

import numpy as np
import pytest

from spectra import Constants as CST
from spectra.Struct import Atom as Atom_
from spectra.Util import MeshUtil

from . import _falc_setup as F


@pytest.fixture(scope="module")
def frozen():
    levels, lines, conts, Ttab, CE, CI = F.parse_h6_atom_()
    return levels, lines, conts, Ttab, CE, CI, F.build_atom_(levels, lines, conts)


@pytest.fixture(scope="module")
def atom():
    return Atom_.init_Atom_from_RH_(F.RH_ATOM)


def test_levels(frozen, atom):
    levels, *_, toy = frozen
    np.testing.assert_array_equal(atom.RH.level_E_cm, [e for e, _ in levels])
    np.testing.assert_array_equal(atom.Level["g"], [g for _, g in levels])
    np.testing.assert_array_equal(atom.Level["erg"], toy.Level["erg"])
    np.testing.assert_array_equal(atom.Level["isGround"], toy.Level["isGround"])
    # RH's principal quantum numbers, as the frozen damping assembly lists them (proton excluded)
    np.testing.assert_array_equal(atom.Level["n"][:5], F.N_PRINCIPAL)


def test_lines(frozen, atom):
    _, lines, *_, toy = frozen
    assert atom.nLine == len(lines) == toy.nLine
    np.testing.assert_array_equal(atom.Line["idxI"], [ln["i"] for ln in lines])
    np.testing.assert_array_equal(atom.Line["idxJ"], [ln["j"] for ln in lines])
    np.testing.assert_array_equal(atom.Line["AJI"], toy.Line["AJI"])
    np.testing.assert_array_equal(atom.Line["f0"], toy.Line["f0"])
    np.testing.assert_allclose(atom.Line["w0"], toy.Line["w0"], rtol=1e-14, atol=0.0)

    rh = atom.RH
    np.testing.assert_array_equal(rh.line_Grad, [ln["Grad"] for ln in lines])
    np.testing.assert_array_equal(rh.line_cStark, [ln["cStark"] for ln in lines])
    np.testing.assert_array_equal(rh.line_cvdW, [ln["cvdW"] for ln in lines])
    assert rh.line_vdW_type == tuple(ln["vdWaals"] for ln in lines)
    np.testing.assert_array_equal(rh.line_Nlambda, [ln["Nlambda"] for ln in lines])
    assert rh.line_symm == tuple(ln["symm"] for ln in lines)
    np.testing.assert_array_equal(rh.line_qcore, [ln["qcore"] for ln in lines])
    np.testing.assert_array_equal(rh.line_qwing, [ln["qwing"] for ln in lines])


def test_line_mesh_points_follow_the_frozen_rule(frozen, atom):
    # the point counts `F.build_mesh_` uses: SYMM counts the half profile, an even count gets one more
    _, lines, *_ = frozen
    coe = atom._wave_mesh.Line_Coe
    for k, ln in enumerate(lines):
        n_full = ln["Nlambda"] if ln["symm"] == "ASYMM" else 2 * ln["Nlambda"] - 1
        if n_full % 2 == 0:
            n_full += 1
        assert coe["nLambda"][k] == n_full
        lo, hi = atom._wave_mesh.Line_mesh_idxs[k]
        np.testing.assert_array_equal(
            atom._wave_mesh.Line_mesh[lo:hi], MeshUtil.make_full_line_mesh_(n_full, ln["qcore"], ln["qwing"])
        )


def test_continua(frozen, atom):
    _, _, conts, *_, toy = frozen
    assert atom.nCont == len(conts) == toy.nCont
    np.testing.assert_array_equal(atom.Cont["idxI"], [c["i"] for c in conts])
    np.testing.assert_array_equal(atom.Cont["idxJ"], [c["j"] for c in conts])
    np.testing.assert_array_equal(atom.Cont["f0"], toy.Cont["f0"])
    np.testing.assert_array_equal(atom.Cont["gi"], toy.Cont["gi"])
    np.testing.assert_array_equal(atom.Cont["gj"], toy.Cont["gj"])
    np.testing.assert_array_equal(atom.Cont["w0"], CST.c_ / toy.Cont["f0"])

    rh = atom.RH
    np.testing.assert_array_equal(rh.cont_alpha0_m2 * 1.0e4, [c["alpha0_cm2"] for c in conts])
    np.testing.assert_array_equal(atom.PI.Coe["alpha0"], [c["alpha0_cm2"] for c in conts])
    np.testing.assert_array_equal(rh.cont_lamb_min_nm, [c["lamb_min_nm"] for c in conts])


def test_collisions(frozen, atom):
    *_, Ttab, CE, CI, _toy = frozen
    rh = atom.RH
    np.testing.assert_array_equal(rh.Te_table, Ttab)
    # CE: upper level first; CI: lower level first. Same unit conversion as `cij_table_`.
    np.testing.assert_array_equal(rh.CE_idx, [[a, b] for a, b, _ in CE])
    np.testing.assert_array_equal(rh.CE_coe_cgs, np.array([v for _, _, v in CE]) * 1.0e6)
    np.testing.assert_array_equal(rh.CI_idx, [[a, b] for a, b, _ in CI])
    np.testing.assert_array_equal(rh.CI_coe_cgs, np.array([v for _, _, v in CI]) * 1.0e6)
