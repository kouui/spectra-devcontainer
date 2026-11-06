"""Regression tests for spectra.RadiativeTransfer.Tau"""

import numpy as np

from spectra.RadiativeTransfer import Tau

from .conftest import assert_close


class TestMakeTau:
    def test_logscale(self, ref):
        tau = Tau.make_tau_(21, 2.0, 0.0)
        assert_close(tau, ref["Tau.make_tau_21_e2_r0"])

    def test_relaxed(self, ref):
        tau = Tau.make_tau_(21, 2.0, 0.5)
        assert_close(tau, ref["Tau.make_tau_21_e2_r05"])

    def test_starts_at_zero(self):
        tau = Tau.make_tau_(21, 2.0, 0.0)
        assert tau[0] == 0.0

    def test_monotonic(self):
        tau = Tau.make_tau_(21, 2.0, 0.0)
        assert np.all(np.diff(tau) > 0)


class TestZToTau:
    def test_constant_opacity(self):
        # trapezoid rule is exact for a constant opacity; the surface point
        # carries half of the last interval
        Z = np.array([0.0, 1.0, 3.0, 4.0])
        xi = np.full(4, 2.0)
        tau = Tau.z_to_tau_(Z, xi)
        np.testing.assert_allclose(tau, [9.0, 7.0, 3.0, 1.0], rtol=1e-14)

    def test_experimental_alias_is_same_function(self):
        from spectra.Experimental import ExTau

        assert ExTau.tau_ is Tau.z_to_tau_
