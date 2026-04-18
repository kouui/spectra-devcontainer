import unittest

import numpy as _numpy
from numpy import allclose as _ALLCLOSE

from spectra.ImportAll import *
from spectra.RadiativeTransfer import Profile as _Profile

_KWGS_CLOSE = {"rtol": 1.0e-05, "atol": 1.0e-20}


class Test_Profile(unittest.TestCase):
    def test_voigt_gaussian(self):

        x = _numpy.linspace(-3, 3, 61)
        a = 0
        gaussian = _Profile.gaussian_(x)
        voigt = _Profile.voigt_(a, x)
        self.assertTrue(_ALLCLOSE(voigt, gaussian, **_KWGS_CLOSE))

    def test_voigt_hf(self):

        x = _numpy.linspace(-3, 3, 61)
        a = 0.1
        h, f = _Profile.hf_(a, x)
        voigt = _Profile.voigt_(a, x)
        self.assertTrue(_ALLCLOSE(voigt, h, **{"rtol": 1.0e-03, "atol": 1.0e-20}))


if __name__ == "__main__":
    unittest.main()
