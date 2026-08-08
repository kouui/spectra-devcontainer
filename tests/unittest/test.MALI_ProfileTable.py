"""Unit tests for spectra.Experimental.MALI.ProfileTable (RH stage D).

The axis is frozen; every profile value is exact (analytic function of the
coordinate transform). What CAN degrade is the quadrature of integrals over
the axis, and wphi measures exactly that degradation.
"""

import numpy as np
import pytest

from spectra import Constants as CST
from spectra.Enums import E_ABSORPTION_PROFILE_TYPE
from spectra.Experimental.MALI import GlobalMesh, ProfileTable
from spectra.RadiativeTransfer import Profile
from spectra.Util import MeshUtil

W0 = 5000.0e-8
XI_REF = 2.5e5
GAUSS = int(E_ABSORPTION_PROFILE_TYPE.GAUSSIAN)
VOIGT = int(E_ABSORPTION_PROFILE_TYPE.VOIGT)


def _window(nLambda=41, qcore=2.5, qwing=10.0):
    q = MeshUtil.make_full_line_mesh_(nLambda, qcore, qwing)
    wl = GlobalMesh.anchor_line_mesh_(q, W0, XI_REF)
    return wl, GlobalMesh.trapezoidal_weight_(wl)


class TestGaussian:
    def test_values_are_exact(self):
        wl, w = _window()
        dop = np.array([0.5e-8, 1.2e-8])
        adamp = np.zeros(2)
        phi, _ = ProfileTable.line_profile_table_(wl, w, W0, dop, adamp, GAUSS)
        for k in (0, 1):
            x = (wl - W0) / dop[k]
            ref = np.exp(-x * x) / CST.sqrtPi_ / dop[k]
            assert np.allclose(phi[:, k], ref, rtol=1e-14)

    def test_wphi_converges_to_unity_on_dense_mesh(self):
        # ruler chosen so the window resolves and covers the line: dop in
        # ruler units = dop_cm / (W0*xi/c)
        wl, w = _window(nLambda=201, qcore=8.0, qwing=16.0)
        dop = np.array([2.0 * W0 * XI_REF / CST.c_])  # 2 ruler units wide
        _, wphi = ProfileTable.line_profile_table_(wl, w, W0, dop, np.zeros(1), GAUSS)
        assert wphi[0] == pytest.approx(1.0, abs=1e-6)

    def test_wphi_degrades_on_coarse_mesh_and_is_the_norm(self):
        wl, w = _window(nLambda=11, qcore=2.5, qwing=10.0)
        dop = np.array([0.4 * W0 * XI_REF / CST.c_])  # narrower than the spacing
        phi, wphi = ProfileTable.line_profile_table_(wl, w, W0, dop, np.zeros(1), GAUSS)
        assert abs(wphi[0] - 1.0) > 1e-2  # visibly wrong area ...
        assert wphi[0] == pytest.approx(np.sum(w * phi[:, 0]), rel=1e-14)  # ... exactly recorded

    def test_same_columns_all_depths(self):
        # the whole point: rows share columns; nothing depth-local moves the axis
        wl, w = _window()
        dop = np.array([0.5e-8, 0.9e-8, 1.4e-8])
        phi, _ = ProfileTable.line_profile_table_(wl, w, W0, dop, np.zeros(3), GAUSS)
        assert phi.shape == (wl.shape[0], 3)
        # peak column is the same (center) for every static depth
        assert np.all(np.argmax(phi, axis=0) == wl.shape[0] // 2)


class TestVoigt:
    def test_matches_production_voigt(self):
        wl, w = _window()
        dop = np.array([0.8e-8])
        adamp = np.array([0.05])
        phi, _ = ProfileTable.line_profile_table_(wl, w, W0, dop, adamp, VOIGT)
        x = (wl - W0) / dop[0]
        ref = Profile.voigt_(adamp[0], x) / dop[0]
        assert np.allclose(phi[:, 0], ref, rtol=1e-13)

    def test_zero_damping_close_to_gaussian(self):
        wl, w = _window()
        dop = np.array([0.8e-8])
        phi_v, _ = ProfileTable.line_profile_table_(wl, w, W0, dop, np.zeros(1), VOIGT)
        phi_g, _ = ProfileTable.line_profile_table_(wl, w, W0, dop, np.zeros(1), GAUSS)
        # a=0 Voigt is exactly Gaussian analytically; the polynomial fit keeps ~1e-6
        assert np.allclose(phi_v[:, 0], phi_g[:, 0], rtol=1e-5)
