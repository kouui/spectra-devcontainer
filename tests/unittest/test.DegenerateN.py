import unittest

import numpy as np

from spectra import Constants as CST
from spectra.Function.Hydrogen import DegenerateN

# ni**2 wraps int64 for ni >= 3.04e9
_N_WRAP_POW2 = 3_100_000_000


class Test_ratio_Etran_to_Eionize_large_n(unittest.TestCase):
    def test_ratio_equals_ionization_wavelength_over_wavelength(self):
        for n in (10, 1000, _N_WRAP_POW2):
            with self.subTest(n=n):
                w_limit = CST.h_ * CST.c_ / CST.E_Rydberg_H_ * float(n) ** 2
                wave = w_limit * np.array([[1.0, 0.5, 0.25]])
                ratio = DegenerateN.ratio_Etran_to_Eionize_(np.array([n]), wave)
                np.testing.assert_allclose(ratio, [[1.0, 2.0, 4.0]], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
