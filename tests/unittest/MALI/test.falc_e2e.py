"""End-to-end pin of MALI on FALC with the 6-level RH hydrogen atom.

Setup (see ``_falc_setup.py`` and ``tests/data/rh_falc/README.md``): the FALC
atmosphere, the H_6_CRD atom (10 lines and 5 bound-free continua over 6 levels
including the proton, on one unified axis, Voigt profiles with every RH
damping term -- radiative, van der Waals, quadratic and linear Stark), a
thermal background with Thomson and Rayleigh scattering, RH-convention
collision rates, 5 angles per hemisphere.

This is the safety net for moving MALI out of ``Experimental``: any change to
the numerical path of the solver, the build-once inputs or the physics kernels
it calls shows up here. Two tests, so that the failure says which kind of
change happened:

``test_falc_all_damping_pin`` is the refactor tripwire. It solves to the
loose criterion (``tol=1e-7``) and pins, in
``tests/data/rh_falc/mali_v2_falc_all_reference.npz``:

- ``n``, the converged populations (normalized to 1 per depth) at rtol 1e-8.
  Populations rather than b = n / n_LTE: ``n`` is the solver's own output, and
  ``n_LTE`` is pinned separately, so b is covered without dividing a tolerance
  through it. 1e-8 rather than tighter because LAPACK/numpy builds differ in
  the last bits of the per-depth linear solves.
- ``n_LTE`` at rtol 1e-8, the Saha-Boltzmann reference of the build-once step.
- the wavelength axis itself (``wl`` at rtol 1e-12), not only its size: a
  shifted or re-spaced mesh would change J without necessarily moving the
  populations at 1e-8.
- ``converged`` is True, and the iteration count is exact. Convergence is
  declared on a threshold crossing, and one iteration more or less moves the
  populations by about the tolerance (1e-7), far above the rtol, so a count
  that differs would fail the population pin anyway; the exact count makes
  that failure readable.

This pin is path-dependent by construction: the loose criterion stops the
iteration about 3e-3 (relative, on the smallest populations) short of the
fixed point, so any change of the iteration path moves it.

``test_falc_deep_fixed_point`` is path-independent. It solves the same setup
to a criterion tight enough that the remaining distance to the fixed point is
a few 1e-7 and compares with ``n_fixed`` at rtol 1e-6. If only the pin fails,
the iteration path changed but the physics did not: regenerate. If both fail,
the converged answer changed.

Why ``tol=3e-12``: the iteration stalls on round-off noise at max|dn| of about
4e-13, so ``tol=1e-12`` is within 2-3x of the noise floor and could fail to
converge on another BLAS build, while ``tol=1e-11`` leaves a truncation error
of 1.1e-6, above the rtol. At 3e-12 the solve takes about 540 iterations and
ends 2.3e-7 from the stored fixed point; starting from populations perturbed
by 1e-3 or 1e-2 ends 2.0e-7 away, so the path dependence left is below 1e-7.
rtol 1e-6 is about 4x that distance. ``n_fixed`` itself comes from a solve at
``tol=1e-12`` during regeneration, 1.1e-7 from the noise-limited limit.

Regeneration after a deliberate change of the solver or its inputs::

    REGEN_REFS=1 uv run --extra dev python -m pytest tests/unittest/MALI/test.falc_e2e.py

With ``REGEN_REFS=1`` the pin test validates the fresh results (converged,
populations finite and positive), then writes the reference file (including the
``n_fixed`` solve) and is skipped; the deep test is skipped. This is the same
switch ``tests/regression/conftest.py`` uses.

The reference file is shared with the frozen copy of this test in
``tests/unittest/MALI_v2``, so a regeneration must be followed by running both.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from spectra.Function.MALI import Loop, Setup

from . import _falc_setup as F

REF_PATH = Path(__file__).resolve().parents[2] / "data" / "rh_falc" / "mali_v2_falc_all_reference.npz"
RTOL = 1.0e-8
RTOL_MESH = 1.0e-12
RTOL_FIXED = 1.0e-6
N_ANGLE = 5
IS_REGEN = os.environ.get("REGEN_REFS") == "1"

# loose criterion of the pin; max|dn| and max|dJ|/max|J| are separate criteria
PIN_KWGS = {"tol": 1e-7, "itmax": 3000}
# test solve of the fixed-point check; see the module docstring for the choice
DEEP_KWGS = {"tol": 3e-12, "tol_J": 3e-10, "itmax": 3000}
# stored fixed point, deeper than the test solve so that the test's own
# truncation error is measured against a better answer
FIXED_KWGS = {"tol": 1e-12, "tol_J": 1e-10, "itmax": 3000}


@pytest.fixture(scope="module")
def falc_problem():
    """The build-once inputs, shared by every solve in this module."""
    falc = F.load_falc_()
    levels, lines_in, conts_in, Ttab, CE, CI = F.parse_h6_atom_()
    atom = F.build_atom_(levels, lines_in, conts_in)
    atmos = F.build_atmos_(falc)
    mesh = F.build_mesh_(atom, lines_in)
    pre = Setup.precompute_(
        atom,
        atmos,
        mesh,
        Cij_dep=F.cij_table_(atom, levels, Ttab, CE, CI, atmos.Te),
        adamp_dep=F.adamp_all_(levels, lines_in, falc),
        bg_fn=F.make_bg_fn_(atmos, falc, atom, mesh),
    )
    return atom, atmos, mesh, pre


def _solve_(problem, **kwgs):
    atom, atmos, mesh, pre = problem
    return Loop.mali_multilevel_(atom, atmos, mesh, pre, n_angle=N_ANGLE, **kwgs)


@pytest.fixture(scope="module")
def falc_result(falc_problem):
    return _solve_(falc_problem, **PIN_KWGS)


@pytest.fixture(scope="module")
def falc_deep(falc_problem):
    return _solve_(falc_problem, **DEEP_KWGS)


def _check_invariants_(res, n_LTE):
    """Hold in both modes: a regenerated reference must never be an
    unconverged or non-physical state."""
    assert res.converged is True
    assert np.all(np.isfinite(res.n))
    assert np.all(res.n > 0.0)
    assert np.all(np.isfinite(n_LTE))
    assert np.all(n_LTE > 0.0)


def test_falc_all_damping_pin(falc_problem, falc_result):
    _, _, mesh, pre = falc_problem
    res = falc_result
    _check_invariants_(res, pre.n_LTE)
    if IS_REGEN:
        fixed = _solve_(falc_problem, **FIXED_KWGS)
        _check_invariants_(fixed, pre.n_LTE)
        np.savez_compressed(
            REF_PATH,
            n=res.n,
            n_LTE=pre.n_LTE,
            niter=res.niter,
            wl=mesh.wl,
            n_fixed=fixed.n,
        )
        pytest.skip(f"wrote {REF_PATH}")
    with np.load(REF_PATH) as ref:
        np.testing.assert_allclose(mesh.wl, ref["wl"], rtol=RTOL_MESH, atol=0.0)
        np.testing.assert_allclose(pre.n_LTE, ref["n_LTE"], rtol=RTOL, atol=0.0)
        assert res.niter == int(ref["niter"]), (
            f"iteration path changed ({res.niter} vs {int(ref['niter'])} iterations); "
            "if the deep-converged test still passes, regenerate with REGEN_REFS=1"
        )
        np.testing.assert_allclose(res.n, ref["n"], rtol=RTOL, atol=0.0)


def test_falc_deep_fixed_point(falc_problem, request):
    if IS_REGEN:
        # requested by name below so that the 500-iteration solve is not run
        # just to be skipped
        pytest.skip("regeneration run")
    falc_deep = request.getfixturevalue("falc_deep")
    _check_invariants_(falc_deep, falc_problem[3].n_LTE)
    with np.load(REF_PATH) as ref:
        np.testing.assert_allclose(falc_deep.n, ref["n_fixed"], rtol=RTOL_FIXED, atol=0.0)
