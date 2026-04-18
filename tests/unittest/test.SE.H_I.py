import unittest

from numpy import allclose as _ALLCLOSE
from numpy import array as _array
from numpy import isclose as _ISCLOSE

from spectra.Function.SEquil import SELib
from spectra.ImportAll import *
from spectra.Struct import Atmosphere, Atom, Radiation

_KWGS_CLOSE = {"rtol": 1.0e-05, "atol": 1.0e-20}


class Test_SE_With_H_I(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        conf_path = CFG._ROOT_DIR / "data/conf/H.conf"
        atom, wMesh, _path_dict = Atom.init_Atom_(conf_path, is_hydrogen=True)

        atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_(atmos, wMesh, 0.5)
        SE_con, Rate_con = SELib.cal_SE_with_Nh_Te_(atom, atmos, wMesh, radiation, None)

        self.atmos = atmos
        self.SE_con = SE_con
        self.Rate_con = Rate_con

    def test_LTE(self):

        n_LTE = self.SE_con.n_LTE
        n_LTE_correct = _array(
            [
                3.70187280e-01,
                6.72414805e-08,
                6.60749362e-09,
                3.92638972e-09,
                3.69429734e-09,
                4.03864392e-09,
                4.65558107e-09,
                5.45922548e-09,
                6.29812624e-01,
            ],
            dtype=DT_NB_FLOAT,
        )
        self.assertTrue(_ALLCLOSE(n_LTE, n_LTE_correct, **_KWGS_CLOSE))

    def test_SE(self):

        n_SE = self.SE_con.n_SE
        n_SE_correct = _array(
            [
                8.66282973e-01,
                3.38419084e-07,
                3.09782647e-09,
                1.12989862e-09,
                8.75871458e-10,
                8.37963993e-10,
                8.98266198e-10,
                1.03867939e-09,
                1.33716680e-01,
            ],
            dtype=DT_NB_FLOAT,
        )
        self.assertTrue(_ALLCLOSE(n_SE, n_SE_correct, **_KWGS_CLOSE))

    def test_Ne(self):

        Ne = self.atmos.Ne
        Ne_correct = 134_244_009_880.22995
        self.assertTrue(_ISCLOSE(Ne, Ne_correct))


if __name__ == "__main__":
    unittest.main()
