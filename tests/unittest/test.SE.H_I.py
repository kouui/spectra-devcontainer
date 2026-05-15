import unittest

from numpy import allclose as _ALLCLOSE
from numpy import array as _array
from numpy import isclose as _ISCLOSE

from spectra.Function.SEquil import SELib
from spectra.ImportAll import *
from spectra.Struct import Atmosphere, Atom, Container, Radiation

_KWGS_CLOSE = {"rtol": 1.0e-05, "atol": 1.0e-20}


class Test_SE_With_H_I(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        conf_path = CFG._ROOT_DIR / "data/conf/H.conf"
        atom, wMesh, _path_dict = Atom.init_Atom_(str(conf_path), is_hydrogen=True)

        atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, Rate_con = SELib.cal_SE_with_Nh_Te_(
            atom, atmos, wMesh, radiation, None, Container.SE_Params_Container()
        )

        self.atmos = atmos
        self.SE_con = SE_con
        self.Rate_con = Rate_con

    def test_LTE(self):

        n_LTE = self.SE_con.n_LTE
        n_LTE_correct = _array(
            [
                5.39021329e-01,
                9.79087996e-08,
                9.62102208e-09,
                5.71712731e-09,
                5.37918284e-09,
                5.88057811e-09,
                6.77888634e-09,
                7.94905479e-09,
                9.34360041e-09,
                1.09411860e-08,
                4.60978511e-01,
            ],
            dtype=DT_NB_FLOAT,
        )
        self.assertTrue(_ALLCLOSE(n_LTE, n_LTE_correct, **_KWGS_CLOSE))  # type: ignore[arg-type]

    def test_SE(self):

        n_SE = self.SE_con.n_SE
        n_SE_correct = _array(
            [
                7.35056964e-01,
                5.74037856e-07,
                6.88963678e-09,
                3.24425198e-09,
                2.97126798e-09,
                3.14435863e-09,
                3.59807095e-09,
                4.27101377e-09,
                5.06465646e-09,
                5.94507968e-09,
                2.64942427e-01,
            ],
            dtype=DT_NB_FLOAT,
        )
        self.assertTrue(_ALLCLOSE(n_SE, n_SE_correct, **_KWGS_CLOSE))  # type: ignore[arg-type]

    def test_Ne(self):

        Ne = self.atmos.Ne
        Ne_correct = 266_476_643_284.7473
        self.assertTrue(_ISCLOSE(Ne, Ne_correct))


if __name__ == "__main__":
    unittest.main()
