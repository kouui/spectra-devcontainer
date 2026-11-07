"""Parity of the package RH reader and builder with the frozen FALC helper.

`MALI_v2/_falc_setup.parse_h6_atom_` and `build_atom_` are what the frozen v2
case uses; `Struct.Atom.init_Atom_from_RH_` must hand the later package setup
the same numbers (after the unit conversions it applies once).

Temporary: delete this file together with `tests/unittest/MALI_v2/_falc_setup.py`
(the only thing it imports from there). It lives outside `MALI_v2/` so that it
also runs under NUMBA_DISABLE_JIT=1, where that directory is skipped.

Comparisons are exact (`assert_array_equal`) wherever both sides run the same
arithmetic. The exceptions are the wavelengths: the frozen toy atom takes
``h*c/dE`` and `Atom` takes ``c/(dE/h)``, which differ in the last bits.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spectra import Constants as CST
from spectra.Enums import E_VDW_TYPE
from spectra.Struct import Atom as Atom_
from spectra.Util import MeshUtil


def _load_falc_setup():
    # tests/unittest has no __init__.py (module names contain dots), so the helper
    # is loaded by path
    path = Path(__file__).resolve().parent / "MALI_v2" / "_falc_setup.py"
    spec = importlib.util.spec_from_file_location("_falc_setup_for_rh_parity", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load_falc_setup()


@pytest.fixture(scope="module")
def frozen():
    levels, lines, conts, Ttab, CE, CI = F.parse_h6_atom_()
    return levels, lines, conts, Ttab, CE, CI, F.build_atom_(levels, lines, conts)


@pytest.fixture(scope="module")
def atom():
    return Atom_.init_Atom_from_RH_(F.RH_ATOM)


def test_levels(frozen, atom):
    levels, *_, toy = frozen
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


def test_broadening(frozen, atom):
    _, lines, *_ = frozen
    coe = atom.RL.Coe
    np.testing.assert_array_equal(coe["Grad"], [ln["Grad"] for ln in lines])
    np.testing.assert_array_equal(coe["cStark"], [ln["cStark"] for ln in lines])
    np.testing.assert_array_equal(coe["cvdW"], [ln["cvdW"] for ln in lines])
    assert all(ln["vdWaals"] == "UNSOLD" for ln in lines)
    assert np.all(coe["vdW_type"] == int(E_VDW_TYPE.UNSOLD))


def test_line_mesh_follows_the_frozen_rule(frozen, atom):
    # the point counts `F.build_mesh_` uses: SYMM counts the half profile, an even count gets one more
    _, lines, *_ = frozen
    coe = atom._wave_mesh.Line_Coe
    for k, ln in enumerate(lines):
        n_full = ln["Nlambda"] if ln["symm"] == "ASYMM" else 2 * ln["Nlambda"] - 1
        if n_full % 2 == 0:
            n_full += 1
        assert coe["nLambda"][k] == n_full
        assert coe["qcore"][k] == ln["qcore"]
        assert coe["qwing"][k] == ln["qwing"]
        lo, hi = atom._wave_mesh.Line_mesh_idxs[k]
        np.testing.assert_array_equal(
            atom._wave_mesh.Line_mesh[lo:hi], MeshUtil.make_full_line_mesh_(n_full, ln["qcore"], ln["qwing"])
        )


def test_continua_and_their_mesh(frozen, atom):
    _, _, conts, *_, toy = frozen
    assert atom.nCont == len(conts) == toy.nCont
    np.testing.assert_array_equal(atom.Cont["idxI"], [c["i"] for c in conts])
    np.testing.assert_array_equal(atom.Cont["idxJ"], [c["j"] for c in conts])
    np.testing.assert_array_equal(atom.Cont["f0"], toy.Cont["f0"])
    np.testing.assert_array_equal(atom.Cont["gi"], toy.Cont["gi"])
    np.testing.assert_array_equal(atom.Cont["gj"], toy.Cont["gj"])
    np.testing.assert_array_equal(atom.Cont["w0"], CST.c_ / toy.Cont["f0"])

    np.testing.assert_array_equal(atom.PI.Coe["alpha0"], [c["alpha0_cm2"] for c in conts])
    np.testing.assert_array_equal(atom._wave_mesh.Cont_Coe["lamb_min"], [c["lamb_min_nm"] * 1.0e-7 for c in conts])
    # the 41-point mesh remapped onto [lamb_min, edge]; w0 differs from the frozen edge in the last bits
    assert atom._wave_mesh.Cont_mesh.shape == toy.Cont_mesh.shape
    np.testing.assert_allclose(atom._wave_mesh.Cont_mesh, toy.Cont_mesh, rtol=1e-14, atol=0.0)


def test_collisions(frozen, atom):
    *_, Ttab, CE, CI, _toy = frozen
    np.testing.assert_array_equal(atom.CE.Te_table, Ttab)
    np.testing.assert_array_equal(atom.CI.Te_table, Ttab)
    # the same unit conversion as `cij_table_`; the file lists CE as (upper, lower), CI as (lower, upper)
    line_row = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(atom.Line["idxI"], atom.Line["idxJ"], strict=True))}
    cont_row = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(atom.Cont["idxI"], atom.Cont["idxJ"], strict=True))}
    assert len(CE) == atom.nLine
    assert len(CI) == atom.nCont
    for upper, lower, vals in CE:
        np.testing.assert_array_equal(atom.CE.Omega_table[line_row[(lower, upper)]], vals * 1.0e6)
    for lower, upper, vals in CI:
        np.testing.assert_array_equal(atom.CI.Omega_table[cont_row[(lower, upper)]], vals * 1.0e6)
