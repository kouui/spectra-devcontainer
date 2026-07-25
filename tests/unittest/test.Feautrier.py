"""Unit tests for spectra.RadiativeTransfer.Feautrier.

The formal solver is checked against closed-form solutions only. The
two-level-atom solution quoted in older notebooks is an *Eddington-approximation*
result, not an exact one, so it is deliberately not used as a reference here.
"""

import numpy as np
import pytest

from spectra.Enums import E_FEAUTRIER_ORDER
from spectra.Math import GaussLeg
from spectra.RadiativeTransfer import Feautrier

SECOND = E_FEAUTRIER_ORDER.SECOND
HERMITE = E_FEAUTRIER_ORDER.HERMITE


# ---------------------------------------------------------------------------
# closed-form reference solutions
# ---------------------------------------------------------------------------
def _j_const_source(tau, T, mu, S0):
    """Finite slab, S = const, no incident radiation."""
    return S0 * (1.0 - 0.5 * (np.exp(-tau / mu) + np.exp(-(T - tau) / mu)))


def _j_attenuation(tau, T, mu, h0, hn):
    """S = 0, incident h0 at the upper and hn at the lower boundary."""
    return 0.5 * (h0 * np.exp(-tau / mu) + hn * np.exp(-(T - tau) / mu))


def _j_exp_source(tau, T, mu, beta):
    """Finite slab, S = exp(-beta*tau), no incident radiation. Requires beta*mu != 1."""
    Im = (np.exp(-beta * tau) - np.exp(-tau / mu)) / (1.0 - beta * mu)
    Ip = (np.exp(-beta * tau) - np.exp(-beta * T) * np.exp(-(T - tau) / mu)) / (1.0 + beta * mu)
    return 0.5 * (Ip + Im)


def _fit_order(hs, errs):
    return np.polyfit(np.log(hs), np.log(errs), 1)[0]


# ---------------------------------------------------------------------------
# exact-solution agreement
# ---------------------------------------------------------------------------
class TestExactSolutions:
    @pytest.mark.parametrize("mu", [1.0, 0.7, 0.2])
    @pytest.mark.parametrize("order", [SECOND, HERMITE])
    def test_const_source(self, mu, order):
        T, S0 = 3.0, 2.5
        tau = np.linspace(0.0, T, 401)
        S = np.full(tau.size, S0)
        res = Feautrier.formal_improved_RH_(tau, S, mu, 0.0, 0.0, 0.0, 0.0, order, False)
        # The tolerance is set by the largest mu tested: dtau is divided by mu, so a
        # grazing ray sees a coarser effective step on the same tau grid.
        assert np.allclose(res.j, _j_const_source(tau, T, mu, S0), rtol=0.0, atol=5e-4)
        assert res.I_emergent == pytest.approx(S0 * (1.0 - np.exp(-T / mu)), abs=5e-4)

    @pytest.mark.parametrize("order", [SECOND, HERMITE])
    def test_pure_attenuation(self, order):
        T, mu, h0, hn = 3.0, 0.7, 1.3, 2.7
        tau = np.linspace(0.0, T, 401)
        S = np.zeros(tau.size)
        res = Feautrier.formal_improved_RH_(tau, S, mu, 0.0, h0, 0.0, hn, order, False)
        assert np.allclose(res.j, _j_attenuation(tau, T, mu, h0, hn), rtol=0.0, atol=1e-4)
        assert res.I_emergent == pytest.approx(hn * np.exp(-T / mu), abs=1e-4)

    @pytest.mark.parametrize("mu", [1.0, 0.6, 0.3])
    def test_linear_source_eddington_barbier(self, mu):
        # S = a + b*tau with the diffusion-limit lower boundary gives I(+)(0) = a + b*mu
        # exactly, for any slab thickness.
        a, b = 1.0, 0.3
        tau = np.linspace(0.0, 20.0, 2001)
        S = a + b * tau
        res = Feautrier.formal_improved_RH_(tau, S, mu, 0.0, 0.0, 0.0, S[-1] + mu * b, SECOND, False)
        assert res.I_emergent == pytest.approx(a + b * mu, rel=1e-4)

    def test_reflective_equals_double_slab(self):
        # A reflecting lower boundary must reproduce the upper half of a slab of
        # twice the thickness carrying a source function symmetric about its centre.
        n, Th, mu = 201, 2.0, 0.7
        tau_full = np.linspace(0.0, 2.0 * Th, 2 * n - 1)
        S_full = 1.0 + np.exp(-tau_full) + np.exp(-(2.0 * Th - tau_full))
        full = Feautrier.formal_improved_RH_(tau_full, S_full, mu, 0.0, 0.0, 0.0, 0.0, SECOND, False)
        half = Feautrier.formal_improved_RH_(tau_full[:n], S_full[:n], mu, 0.0, 0.0, 1.0, 0.0, SECOND, False)
        assert np.allclose(half.j, full.j[:n], rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("order", [SECOND, HERMITE])
    def test_emergent_intensity_identity(self, order):
        # I(+) inverts j[0] = 0.5*((1+r0)*I(+) + h0) for every boundary flavour.
        tau = np.linspace(0.0, 4.0, 121)
        S = np.exp(-0.5 * tau)
        for r0, h0 in ((0.0, 0.0), (0.0, 1.7), (1.0, 0.0)):
            res = Feautrier.formal_improved_RH_(tau, S, 0.6, r0, h0, 0.0, 0.9, order, False)
            assert res.I_emergent == pytest.approx((2.0 * res.j[0] - h0) / (1.0 + r0), rel=1e-13)


# ---------------------------------------------------------------------------
# convergence order
# ---------------------------------------------------------------------------
class TestConvergenceOrder:
    T, MU, BETA = 5.0, 0.6, 0.7
    NDS = (41, 81, 161, 321)

    def _errors(self, order):
        hs, errs = [], []
        for ND in self.NDS:
            tau = np.linspace(0.0, self.T, ND)
            S = np.exp(-self.BETA * tau)
            j = Feautrier.formal_improved_RH_(tau, S, self.MU, 0.0, 0.0, 0.0, 0.0, order, False).j
            hs.append(self.T / (ND - 1))
            errs.append(np.max(np.abs(j - _j_exp_source(tau, self.T, self.MU, self.BETA))))
        return hs, errs

    def test_second_order_is_two(self):
        assert _fit_order(*self._errors(SECOND)) == pytest.approx(2.0, abs=0.15)

    def test_hermite_order_is_three(self):
        # Hermite raises the interior rows to 4th order but its boundary rows gain
        # only one order, and the boundary caps the global rate at 3. Asserting 4
        # here would be asserting the reference's claim rather than its behaviour.
        assert _fit_order(*self._errors(HERMITE)) == pytest.approx(3.0, abs=0.15)

    def test_hermite_is_much_more_accurate(self):
        _, e2 = self._errors(SECOND)
        _, e4 = self._errors(HERMITE)
        assert e4[-1] < e2[-1] / 50.0


# ---------------------------------------------------------------------------
# diagonal operator
# ---------------------------------------------------------------------------
class TestPsi:
    @staticmethod
    def _setup():
        return np.linspace(0.0, 10.0, 61), 0.5

    def test_not_computed_when_not_requested(self):
        tau, mu = self._setup()
        res = Feautrier.formal_improved_RH_(tau, np.ones(tau.size), mu, 0.0, 0.0, 0.0, 0.0, SECOND, False)
        assert res.Psi.shape == (0,)

    @pytest.mark.parametrize("k", [0, 1, 17, 30, 59, 60])
    def test_second_order_psi_is_dj_dS(self, k):
        # With zero boundary terms the right hand side is S itself, so solving with
        # S = e_k returns column k of T^-1 and j[k] is exactly diag(T^-1)[k].
        # This checks Psi without re-deriving any coefficient formula.
        tau, mu = self._setup()
        Psi = Feautrier.formal_improved_RH_(tau, np.zeros(tau.size), mu, 0.0, 0.0, 0.0, 0.0, SECOND, True).Psi
        e = np.zeros(tau.size)
        e[k] = 1.0
        j = Feautrier.formal_improved_RH_(tau, e, mu, 0.0, 0.0, 0.0, 0.0, SECOND, False).j
        assert j[k] == pytest.approx(Psi[k], rel=1e-12)

    def test_hermite_psi_deviates_as_documented(self):
        # Hermite folds S[k+-1] into the right hand side, so dj/dS != diag(T^-1).
        # Psi is returned anyway (the ALI fixed point does not depend on it); this
        # pins the size of that deviation so it cannot drift unnoticed.
        tau, mu = self._setup()
        Psi = Feautrier.formal_improved_RH_(tau, np.zeros(tau.size), mu, 0.0, 0.0, 0.0, 0.0, HERMITE, True).Psi
        devs = []
        for k in (0, 17, 30, 60):
            e = np.zeros(tau.size)
            e[k] = 1.0
            j = Feautrier.formal_improved_RH_(tau, e, mu, 0.0, 0.0, 0.0, 0.0, HERMITE, False).j
            devs.append(abs(j[k] / Psi[k] - 1.0))
        assert 0.01 < max(devs) < 0.5

    @pytest.mark.parametrize("order", [SECOND, HERMITE])
    def test_bounds(self, order):
        # Psi <= 1 keeps the ALI denominator 1-(1-eps)*Lambda_star >= eps > 0.
        # Both boundary intervals are kept below sqrt(6) so the Hermite boundary
        # coefficients stay positive.
        tau = np.linspace(0.0, 10.0, 101)
        S = np.exp(-0.3 * tau)
        Psi = Feautrier.formal_improved_RH_(tau, S, 0.5, 0.0, 0.0, 0.0, 0.0, order, True).Psi
        assert (Psi > 0.0).all()
        assert (Psi <= 1.0).all()

    def test_hermite_bound_breaks_on_coarse_boundary_interval(self):
        # Documented limitation: A1[-1] = 2/dtau_m[-1]**2 - 1/3 turns negative once a
        # boundary interval exceeds sqrt(6), and Psi <= 1 no longer holds.
        tau = np.logspace(-4.0, 2.0, 41)
        S = np.linspace(0.5, 2.0, tau.size)
        Psi = Feautrier.formal_improved_RH_(tau, S, 0.5, 0.0, 1.0, 0.0, 2.0, HERMITE, True).Psi
        assert Psi.max() > 1.0
        Psi2 = Feautrier.formal_improved_RH_(tau, S, 0.5, 0.0, 1.0, 0.0, 2.0, SECOND, True).Psi
        assert (Psi2 <= 1.0).all()


# ---------------------------------------------------------------------------
# two-level atom: direct solver and ALI
# ---------------------------------------------------------------------------
class TestTwoLevelAtom:
    EPS = 1e-2

    @staticmethod
    def _atmosphere(ND=61):
        tau = np.logspace(-4.0, 4.0, ND)
        B = np.ones(ND)
        eps = np.full(ND, TestTwoLevelAtom.EPS)
        return tau, B, eps

    def test_direct_reproduces_sqrt_eps_law(self):
        tau, B, eps = self._atmosphere(101)
        S = Feautrier.direct_feautrier_(tau, np.zeros(4), np.full(4, B[-1]), B, eps)
        assert S[0] == pytest.approx(np.sqrt(self.EPS), rel=5e-3)

    def test_direct_accepts_other_angle_counts(self):
        # At the surface the residual departure from the sqrt(eps) law is set by the
        # depth grid rather than the angle count, so 4 and 8 angles land on the same
        # S[0]; deeper in, the two quadratures do differ.
        tau, B, eps = self._atmosphere(101)
        S4 = Feautrier.direct_feautrier_(tau, np.zeros(4), np.full(4, B[-1]), B, eps)
        S8 = Feautrier.direct_feautrier_(tau, np.zeros(8), np.full(8, B[-1]), B, eps, n_angle=8)
        assert S8[0] == pytest.approx(np.sqrt(self.EPS), rel=5e-3)
        assert S8[0] == pytest.approx(S4[0], rel=1e-4)

    @staticmethod
    def _ali(tau, B, eps, lambda_star_scale=1.0, n_iter=600, tol=1e-10):
        """ALI with the angle-averaged diagonal operator; returns (S, iterations)."""
        mus, ws = GaussLeg.gauss_quad_coe_(0.0, 1.0, 4)
        S = B.copy()
        for it in range(n_iter):
            J_fs = np.zeros(S.size)
            L_star = np.zeros(S.size)
            for i in range(mus.size):
                res = Feautrier.formal_improved_RH_(tau, S, mus[i], 0.0, 0.0, 0.0, B[-1], SECOND, True)
                J_fs += ws[i] * res.j
                L_star += ws[i] * res.Psi
            L_star = L_star * lambda_star_scale
            S_new = ((1.0 - eps) * (J_fs - L_star * S) + eps * B) / (1.0 - (1.0 - eps) * L_star)
            change = np.max(np.abs(S_new / S - 1.0))
            S = S_new
            if change < tol:
                return S, it + 1
        return S, n_iter

    def test_ali_matches_direct(self):
        tau, B, eps = self._atmosphere()
        S_direct = Feautrier.direct_feautrier_(tau, np.zeros(4), np.full(4, B[-1]), B, eps)
        S_ali, iters = self._ali(tau, B, eps)
        assert iters < 600
        assert np.allclose(S_ali, S_direct, rtol=1e-6, atol=0.0)

    def test_ali_fixed_point_independent_of_lambda_star(self):
        # The Lambda_star terms cancel once S_new == S_old, so a deliberately wrong
        # operator must reach the same solution -- only the rate changes. This is
        # what licenses returning an approximate Psi for HERMITE.
        tau, B, eps = self._atmosphere()
        S_exact_op, it_exact = self._ali(tau, B, eps, lambda_star_scale=1.0)
        S_wrong_op, it_wrong = self._ali(tau, B, eps, lambda_star_scale=0.5)
        assert np.allclose(S_wrong_op, S_exact_op, rtol=1e-5, atol=0.0)
        assert it_wrong > it_exact

    def test_lambda_iteration_stalls(self):
        # Lambda_star = 0 degenerates the ALI update to plain Lambda-iteration, which
        # is still far from converged when ALI has already finished.
        tau, B, eps = self._atmosphere()
        S_direct = Feautrier.direct_feautrier_(tau, np.zeros(4), np.full(4, B[-1]), B, eps)
        _, it_ali = self._ali(tau, B, eps)
        S_li, _ = self._ali(tau, B, eps, lambda_star_scale=0.0, n_iter=it_ali)
        assert np.max(np.abs(S_li / S_direct - 1.0)) > 1e-3
