from fractions import Fraction
import math
import unittest

from spectra import Constants as CST
from spectra.Atomic import Hydrogen
from spectra.Math import Special

# The principal quantum number enters the kernels as int64, so n**5 wraps for n >= 6209,
# n**4 differences for n >= 55109 and n**2 for n >= 3.04e9.
_N_WRAP_POW5 = 6209
_N_WRAP_POW4 = 60000
_N_WRAP_POW2 = 2_500_000_000


def _gaunt_float(n: float, x: float) -> float:
    g0 = 0.9935 + (0.2328 - 0.1296 / n) / n
    g1 = -(0.6282 - (0.5598 - 0.5299 / n) / n) / n
    g2 = (0.3887 - (1.181 - 1.470 / n) / n) / n / n
    return g0 + (g1 + g2 / x) / x


def _PI_cross_section_float(n: float, x: float, Z: int) -> float:
    Eik = CST.E_Rydberg_H_ / n**2
    v = x * Eik / CST.h_
    return 2.815e29 * Z**4 / (v**3 * n**5) * _gaunt_float(n, x)


class Test_PI_cross_section_large_n(unittest.TestCase):
    def test_PI_cross_section_x_matches_float_formula(self):
        for n in (_N_WRAP_POW5 - 1, _N_WRAP_POW5, 3 * _N_WRAP_POW5):
            with self.subTest(n=n):
                alpha = Hydrogen.PI_cross_section_(n, 2.0, 1)
                self.assertGreater(alpha, 0.0)
                self.assertAlmostEqual(alpha / _PI_cross_section_float(float(n), 2.0, 1), 1.0, delta=1e-12)

    def test_PI_cross_section_cm_matches_float_formula(self):
        for n in (_N_WRAP_POW5 - 1, _N_WRAP_POW5, 3 * _N_WRAP_POW5):
            with self.subTest(n=n):
                w = 0.5 * CST.ni2cm_ * n * n
                x = CST.h_ * (CST.c_ / w) / (CST.E_Rydberg_H_ / float(n) ** 2)
                alpha = Hydrogen.PI_cross_section_cm_(n, w, 1)
                self.assertGreater(alpha, 0.0)
                self.assertAlmostEqual(alpha / _PI_cross_section_float(float(n), x, 1), 1.0, delta=1e-12)

    def test_PI_cross_section_scales_smoothly_across_wrap_threshold(self):
        # at fixed x the cross section grows like n (v**-3 ~ n**6, divided by n**5)
        ratio = Hydrogen.PI_cross_section_(_N_WRAP_POW5, 2.0, 1) / Hydrogen.PI_cross_section_(_N_WRAP_POW5 - 1, 2.0, 1)
        self.assertAlmostEqual(ratio, _N_WRAP_POW5 / (_N_WRAP_POW5 - 1), delta=1e-6)


def _Res_and_Van_exact(ni: int, nj: int, nH: float, Te: float) -> float:
    cvdw = 1.61e-33 * float(nj**4 - ni**4)  # Python ints do not wrap
    psi_w = 17.0 * cvdw**0.4 * (20596.0 * Te**0.5) ** 0.6
    return nH * psi_w / (4.0 * CST.pi_)


def _LinearStark_exact(ni: int, nj: int, Ne: float) -> float:
    a1 = 0.642 if nj - ni == 1 else 1.0
    return 0.255 * a1 * float(nj * nj - ni * ni) * Ne ** (2.0 / 3.0)


class Test_collisional_broadening_large_n(unittest.TestCase):
    def test_Res_and_Van_matches_exact_integer_difference(self):
        nH, Te = 1e12, 7000.0
        for ni, nj in ((_N_WRAP_POW4, 2 * _N_WRAP_POW4), (10**8, 10**8 + 1), (2**53, 2**53 + 1)):
            with self.subTest(ni=ni, nj=nj):
                gamma = Hydrogen.collisional_broadening_Res_and_Van_(ni, nj, nH, Te)
                self.assertGreater(gamma, 0.0)
                self.assertAlmostEqual(gamma / _Res_and_Van_exact(ni, nj, nH, Te), 1.0, delta=1e-12)

    def test_LinearStark_matches_exact_integer_difference(self):
        Ne = 1e12
        for ni, nj in ((_N_WRAP_POW2, 2 * _N_WRAP_POW2), (10**8, 10**8 + 1), (2**53, 2**53 + 1)):
            with self.subTest(ni=ni, nj=nj):
                gamma = Hydrogen.collisional_broadening_LinearStark_(ni, nj, Ne)
                self.assertGreater(gamma, 0.0)
                self.assertAlmostEqual(gamma / _LinearStark_exact(ni, nj, Ne), 1.0, delta=1e-12)


def _x_and_Eij_exact(ni: int, nj: int) -> tuple[float, float]:
    # exact rational arithmetic: the difference of adjacent levels is not rounded away
    diff = nj * nj - ni * ni
    return float(Fraction(diff, nj * nj)), CST.E_Rydberg_H_ * float(Fraction(diff, ni * ni * nj * nj))


def _oscillator_strength_exact(ni: int, nj: int) -> float:
    x, _ = _x_and_Eij_exact(ni, nj)
    return 32.0 / (3.0 * CST.sqrt3_ * CST.pi_) * ni / (x * nj) ** 3


def _einstein_A_exact(ni: int, nj: int) -> float:
    x, Eij = _x_and_Eij_exact(ni, nj)
    fij = _oscillator_strength_exact(ni, nj) * _gaunt_float(float(ni), x)
    w = CST.h_ * CST.c_ / Eij
    return (ni / nj) ** 2 * 0.667025 / (w * w) * fij


def _CE_exact(ni: int, nj: int, Te: float) -> float:
    x, Eij = _x_and_Eij_exact(ni, nj)
    x_m = 1.0 / x
    fij = _oscillator_strength_exact(ni, nj) * _gaunt_float(float(ni), x)
    Aij = 2.0 * ni**2 * x_m * fij
    ni_m = 1.0 / ni
    bi = ni_m * (4.0 + ni_m * (-18.63 + ni_m * (36.24 - ni_m * 28.09)))
    ri = 1.94 * ni_m**1.57
    Bij = 4.0 * ni * (ni / nj) ** 3 * x_m**2 * (1.0 + x_m * (4 / 3 + x_m * bi))
    y = Eij / (CST.k_ * Te)
    z = ri * x + y
    term1 = Aij * ((1.0 / y + 0.5) * Special.E1_(y) - (1.0 / z + 0.5) * Special.E1_(z))
    term2 = (Bij - Aij * math.log(2.0 * ni * ni * x_m)) * (Special.E2_(y) / y - Special.E2_(z) / z)
    return CST.C0_ * Te**0.5 * 2 * ni * ni * x_m * y * y * (term1 + term2)


# adjacent levels: 3.1e9 squares beyond int64, 1e8 and 2**53 where float copies of the
# neighbouring levels lose the difference
_ADJACENT_LEVELS = ((_N_WRAP_POW2 + 600_000_000, _N_WRAP_POW2 + 600_000_001), (10**8, 10**8 + 1), (2**53, 2**53 + 1))


class Test_rate_coefficients_huge_n(unittest.TestCase):
    def test_oscillator_strength_matches_exact_difference(self):
        for ni, nj in _ADJACENT_LEVELS:
            with self.subTest(ni=ni):
                value = Hydrogen.absorption_oscillator_strength_(ni, nj)
                self.assertAlmostEqual(value / _oscillator_strength_exact(ni, nj), 1.0, delta=1e-12)

    def test_einstein_A_matches_exact_difference(self):
        for ni, nj in _ADJACENT_LEVELS:
            with self.subTest(ni=ni):
                value = Hydrogen.einstein_A_coefficient_(ni, nj)
                self.assertGreater(value, 0.0)
                self.assertAlmostEqual(value / _einstein_A_exact(ni, nj), 1.0, delta=1e-9)

    def test_CE_matches_exact_difference(self):
        for ni, nj in _ADJACENT_LEVELS:
            with self.subTest(ni=ni):
                value = Hydrogen.CE_rate_coe_(ni, nj, 7000.0)
                self.assertGreater(value, 0.0)
                self.assertAlmostEqual(value / _CE_exact(ni, nj, 7000.0), 1.0, delta=1e-9)

    def test_ionization_rates_are_finite_and_positive_beyond_int64_square_wrap(self):
        n = _N_WRAP_POW2 + 600_000_000
        rates = {
            "CI": Hydrogen.CI_rate_coe_(n, 7000.0),
            "Rki_spon": Hydrogen.Rki_spon_rate_coe_(n, 7000.0),
        }
        for name, value in rates.items():
            with self.subTest(kernel=name):
                self.assertTrue(math.isfinite(value) and value > 0.0, value)


if __name__ == "__main__":
    unittest.main()
