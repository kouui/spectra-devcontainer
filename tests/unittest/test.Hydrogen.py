import math
import unittest

from spectra import Constants as CST
from spectra.Atomic import Hydrogen

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


class Test_collisional_broadening_large_n(unittest.TestCase):
    def test_Res_and_Van_matches_exact_integer_difference(self):
        ni, nj = _N_WRAP_POW4, 2 * _N_WRAP_POW4
        nH, Te = 1e12, 7000.0
        cvdw = 1.61e-33 * float(nj**4 - ni**4)  # Python ints do not wrap
        psi_w = 17.0 * cvdw**0.4 * (20596.0 * Te**0.5) ** 0.6
        expected = nH * psi_w / (4.0 * CST.pi_)
        gamma = Hydrogen.collisional_broadening_Res_and_Van_(ni, nj, nH, Te)
        self.assertAlmostEqual(gamma / expected, 1.0, delta=1e-12)

    def test_LinearStark_matches_exact_integer_difference(self):
        ni, nj = _N_WRAP_POW2, 2 * _N_WRAP_POW2
        Ne = 1e12
        expected = 0.255 * float(nj * nj - ni * ni) * Ne ** (2.0 / 3.0)
        gamma = Hydrogen.collisional_broadening_LinearStark_(ni, nj, Ne)
        self.assertAlmostEqual(gamma / expected, 1.0, delta=1e-12)


class Test_rate_coefficients_huge_n(unittest.TestCase):
    def test_rates_are_finite_and_positive_beyond_int64_square_wrap(self):
        n = 3_100_000_000
        rates = {
            "einstein_A": Hydrogen.einstein_A_coefficient_(n, n + 1),
            "CE": Hydrogen.CE_rate_coe_(n, n + 1, 7000.0),
            "CI": Hydrogen.CI_rate_coe_(n, 7000.0),
            "Rki_spon": Hydrogen.Rki_spon_rate_coe_(n, 7000.0),
        }
        for name, value in rates.items():
            with self.subTest(kernel=name):
                self.assertTrue(math.isfinite(value) and value > 0.0, value)


if __name__ == "__main__":
    unittest.main()
