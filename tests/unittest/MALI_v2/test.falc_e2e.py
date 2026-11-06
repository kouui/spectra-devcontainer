"""End-to-end pin of the frozen MALI v2 on FALC with the 6-level RH hydrogen atom.

Setup (see ``_falc_setup.py`` and ``tests/data/rh_falc/README.md``): the FALC
atmosphere, the H_6_CRD atom (5 lines + continua on one unified axis, Voigt
profiles with every RH damping term -- radiative, van der Waals, quadratic and
linear Stark), a thermal background with Thomson and Rayleigh scattering,
RH-convention collision rates, 5 angles per hemisphere.

This is the safety net for moving MALI out of ``Experimental``: any change to
the numerical path of the solver, the build-once inputs or the physics kernels
it calls shows up here.

What is pinned, in ``tests/data/rh_falc/mali_v2_falc_all_reference.npz``:

- ``n``, the converged populations (normalized to 1 per depth) at rtol 1e-8.
  Populations rather than b = n / n_LTE: ``n`` is the solver's own output, and
  ``n_LTE`` is pinned separately, so b is covered without dividing a tolerance
  through it. 1e-8 rather than tighter because LAPACK/numpy builds differ in
  the last bits of the per-depth linear solves.
- ``n_LTE`` at rtol 1e-8, the Saha-Boltzmann reference of the build-once step.
- ``converged`` is True, and the iteration count is exact. Convergence is
  declared on a threshold crossing, and one iteration more or less moves the
  populations by about the tolerance (1e-7), far above the rtol, so a count
  that differs would fail the population pin anyway; the exact count makes
  that failure readable.
- the number of wavelength points of the unified axis (a changed mesh would
  change J without necessarily moving the populations at 1e-8).

Regeneration after a deliberate change of the solver or its inputs::

    REGEN_REFS=1 uv run --extra dev python -m pytest tests/unittest/MALI_v2/test.falc_e2e.py

With ``REGEN_REFS=1`` the test writes the reference file from the fresh result
and is skipped (the same switch ``tests/regression/conftest.py`` uses).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from spectra.Experimental.MALI.v2 import Loop, Structs

from . import _falc_setup as F

REF_PATH = Path(__file__).resolve().parents[2] / "data" / "rh_falc" / "mali_v2_falc_all_reference.npz"
RTOL = 1.0e-8


@pytest.fixture(scope="module")
def falc_result():
    falc = F.load_falc_()
    levels, lines_in, conts_in, Ttab, CE, CI = F.parse_h6_atom_()
    atom = F.build_atom_(levels, lines_in, conts_in)
    atmos = F.build_atmos_(falc)
    mesh = F.build_mesh_(atom, lines_in)
    pre = Structs.precompute_(
        atom,
        atmos,
        mesh,
        Cij_dep=F.cij_table_(atom, levels, Ttab, CE, CI, atmos.Te),
        adamp_dep=F.adamp_all_(levels, lines_in, falc),
        bg_fn=F.make_bg_fn_(atmos, falc, atom, mesh),
    )
    res = Loop.mali_multilevel_(atom, atmos, mesh, pre, n_angle=5, tol=1e-7, itmax=3000)
    return res, pre, mesh


def test_falc_all_damping_pin(falc_result):
    res, pre, mesh = falc_result
    if os.environ.get("REGEN_REFS") == "1":
        np.savez_compressed(
            REF_PATH, n=res.n, n_LTE=pre.n_LTE, niter=res.niter, converged=res.converged, n_wl=mesh.wl.size
        )
        pytest.skip(f"wrote {REF_PATH}")
    with np.load(REF_PATH) as ref:
        assert res.converged is True
        assert res.niter == int(ref["niter"])
        assert mesh.wl.size == int(ref["n_wl"])
        np.testing.assert_allclose(pre.n_LTE, ref["n_LTE"], rtol=RTOL, atol=0.0)
        np.testing.assert_allclose(res.n, ref["n"], rtol=RTOL, atol=0.0)
