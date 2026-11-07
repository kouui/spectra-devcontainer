"""Bitwise parity of the relocated MALI package with the frozen Experimental v2.

The relocation changed imports and where numba decoration sits, not
arithmetic, so on the same input every build-once array and every iterate must
be identical. ``array_equal`` is used on purpose: the FALC pin stops about 3e-3
from the fixed point and is sensitive to 1e-16 changes of the iteration path,
so a tolerance here would hide exactly the drift this test exists to catch.
The claim holds for one environment (numba and BLAS build) only.
The test is skipped under NUMBA_DISABLE_JIT=1: the claim is about compiled code, and the
interpreted run costs about 14 minutes.

Cases: FALC with the 6-level hydrogen atom, and the toy cases of the unit
tests (multilevel, active continuum, doublet under continua, scattering).
"""

import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from numba.core import config as _nb_config
import numpy as np
import pytest

from spectra.Experimental.MALI.v2 import GlobalMesh as GlobalMesh_v2
from spectra.Experimental.MALI.v2 import Loop as Loop_v2
from spectra.Experimental.MALI.v2 import Structs as Structs_v2
from spectra.Function.MALI import Loop as Loop_new
from spectra.Function.MALI import Setup as Setup_new
from spectra.RadiativeTransfer.MALI import GlobalMesh as GlobalMesh_new
from spectra.Util import MeshUtil

from . import _falc_setup as F_new
from . import _toy as Toy

pytestmark = pytest.mark.skipif(
    _nb_config.DISABLE_JIT, reason="bitwise parity is a claim about compiled code; NUMBA_DISABLE_JIT=1 takes ~14 min"
)

XI_REF = 2.5e5
N_ANGLE_FALC = 5
FALC_KWGS = {"tol": 1e-7, "itmax": 3000}
TOY_KWGS = {"tol": 1e-10, "tol_J": 1e-8, "itmax": 2000}


def _load_v2_falc_setup_():
    # the frozen helper lives in a sibling test package; load it by path so that this
    # file does not depend on how pytest names the packages
    path = Path(__file__).resolve().parents[1] / "MALI_v2" / "_falc_setup.py"
    spec = importlib.util.spec_from_file_location("_mali_v2_falc_setup", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F_v2 = _load_v2_falc_setup_()

V2 = SimpleNamespace(
    atom_builders={
        "3lv": Structs_v2.make_toy_atom_3lv_,
        "2lv_cont": Structs_v2.make_toy_atom_2lv_cont_,
        "overlap": Structs_v2.make_toy_atom_overlap_,
    },
    make_toy_atmos_=Structs_v2.make_toy_atmos_,
    GlobalMesh=GlobalMesh_v2,
    precompute_=Structs_v2.precompute_,
    solve_=Loop_v2.mali_multilevel_,
)
NEW = SimpleNamespace(
    atom_builders={
        "3lv": Toy.make_toy_atom_3lv_,
        "2lv_cont": Toy.make_toy_atom_2lv_cont_,
        "overlap": Toy.make_toy_atom_overlap_,
    },
    make_toy_atmos_=Toy.make_toy_atmos_,
    GlobalMesh=GlobalMesh_new,
    precompute_=Setup_new.precompute_,
    solve_=Loop_new.mali_multilevel_,
)


def _assert_same_arrays_(a, b, what):
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        assert np.array_equal(np.asarray(va), np.asarray(vb)), f"{what}.{f.name} differs"


def _assert_same_result_(ra, rb):
    assert ra.niter == rb.niter
    assert ra.converged == rb.converged
    for name in ("n", "J", "S_line", "Jbar", "Lstar", "dn_history", "dJ_history"):
        assert np.array_equal(getattr(ra, name), getattr(rb, name)), f"result.{name} differs"


# --- toy cases --------------------------------------------------------------


def _toy_case_(pkg, name, ND=41, Ne=1.0e14, Nt=1.0e10, scattering=False):
    atom = pkg.atom_builders[name]()
    Zmax = 1.0e9
    atmos = pkg.make_toy_atmos_(ND, Zmax, Te_top=6.0e3, Te_bottom=9.0e3, Ne=Ne, Nt=Nt)
    atmos.Z = np.concatenate([[0.0], np.logspace(np.log10(Zmax * 1e-8), np.log10(Zmax), ND - 1)])
    q = MeshUtil.make_full_line_mesh_(21, 2.5, 10.0)
    meshes = [pkg.GlobalMesh.anchor_line_mesh_(q, w0, XI_REF) for w0 in atom.Line["w0"]]
    meshes += [atom.Cont_mesh[kC, ::-1].copy() for kC in range(atom.nCont)]
    mesh = pkg.GlobalMesh.merge_meshes_(meshes)

    def bg_scattering(wl_cm):
        chi = 1.0e-14 * (wl_cm / 5000.0e-8) ** 2 * np.linspace(1.0, 3.0, ND)
        return chi, 3.0 * chi

    bg_fn = bg_scattering if scattering else None
    pre = pkg.precompute_(atom, atmos, mesh, bg_fn=bg_fn)
    return mesh, pre, pkg.solve_(atom, atmos, mesh, pre, **TOY_KWGS)


TOY_CASES = {
    "multilevel": {"name": "3lv"},
    "continuum": {"name": "2lv_cont", "Ne": 1.0e12, "Nt": 1.0e12},
    "overlap": {"name": "overlap", "Ne": 1.0e12, "Nt": 1.0e12},
    "scattering": {"name": "2lv_cont", "Ne": 1.0e12, "Nt": 1.0e12, "scattering": True},
}


@pytest.mark.parametrize("case", TOY_CASES)
def test_toy_case_is_bitwise_equal(case):
    kwgs = TOY_CASES[case]
    mesh_a, pre_a, res_a = _toy_case_(V2, **kwgs)
    mesh_b, pre_b, res_b = _toy_case_(NEW, **kwgs)
    if case == "scattering":
        assert pre_b.bg_sca.max() > 0.0
    _assert_same_arrays_(mesh_a, mesh_b, "mesh")
    _assert_same_arrays_(pre_a, pre_b, "precompute")
    _assert_same_result_(res_a, res_b)
    assert res_a.converged


# --- FALC ---------------------------------------------------------------------


def _falc_case_(F, precompute_, solve_):
    falc = F.load_falc_()
    levels, lines_in, conts_in, Ttab, CE, CI = F.parse_h6_atom_()
    atom = F.build_atom_(levels, lines_in, conts_in)
    atmos = F.build_atmos_(falc)
    mesh = F.build_mesh_(atom, lines_in)
    pre = precompute_(
        atom,
        atmos,
        mesh,
        Cij_dep=F.cij_table_(atom, levels, Ttab, CE, CI, atmos.Te),
        adamp_dep=F.adamp_all_(levels, lines_in, falc),
        bg_fn=F.make_bg_fn_(atmos, falc, atom, mesh),
    )
    return mesh, pre, solve_(atom, atmos, mesh, pre, n_angle=N_ANGLE_FALC, **FALC_KWGS)


@pytest.fixture(scope="module")
def falc_v2():
    return _falc_case_(F_v2, Structs_v2.precompute_, Loop_v2.mali_multilevel_)


@pytest.fixture(scope="module")
def falc_new():
    return _falc_case_(F_new, Setup_new.precompute_, Loop_new.mali_multilevel_)


def test_falc_mesh_and_precompute_are_bitwise_equal(falc_v2, falc_new):
    _assert_same_arrays_(falc_v2[0], falc_new[0], "mesh")
    _assert_same_arrays_(falc_v2[1], falc_new[1], "precompute")


def test_falc_solution_is_bitwise_equal(falc_v2, falc_new):
    _assert_same_result_(falc_v2[2], falc_new[2])
    assert falc_new[2].converged
