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
