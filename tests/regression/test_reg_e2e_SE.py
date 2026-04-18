"""End-to-end regression tests for Statistical Equilibrium pipeline"""

from spectra.Function.SEquil import SELib
from spectra.ImportAll import *
from spectra.Struct import Atmosphere, Atom, Radiation

from .conftest import assert_close


class TestHydrogenSE:
    def _load_H(self):
        conf_path = str(CFG._ROOT_DIR / "data/conf/H.conf")
        return Atom.init_Atom_(conf_path, is_hydrogen=True)

    def test_SE_with_Nh_Te(self, ref):
        atom, wMesh, _ = self._load_H()
        atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_(atmos, wMesh)
        SE_con, _ = SELib.cal_SE_with_Nh_Te_(atom, atmos, wMesh, radiation, None)

        assert_close(SE_con.n_SE, ref["E2E.H_SE_Nh_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.H_SE_Nh_Te.n_LTE"], rtol=1e-8)
        assert_close(atmos.Ne, ref["E2E.H_SE_Nh_Te.Ne"], rtol=1e-8)

    def test_SE_with_Ne_Te(self, ref):
        atom, wMesh, _ = self._load_H()
        atmos = Atmosphere.Atmosphere0D(Nh=1.0e11, Ne=5.0e10, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_(atmos, wMesh)
        SE_con, _ = SELib.cal_SE_with_Ne_Te_(atom, atmos, wMesh, radiation, None)

        assert_close(SE_con.n_SE, ref["E2E.H_SE_Ne_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.H_SE_Ne_Te.n_LTE"], rtol=1e-8)


class TestHeliumSE:
    def test_SE_with_Ne_Te(self, ref):
        conf_path = str(CFG._ROOT_DIR / "data/conf/He.conf")
        atom, wMesh, _ = Atom.init_Atom_(conf_path, is_hydrogen=False)

        atmos = Atmosphere.Atmosphere0D(Nh=1.0e11, Ne=5.0e10, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_(atmos, wMesh)
        SE_con, _ = SELib.cal_SE_with_Ne_Te_(atom, atmos, wMesh, radiation, None)

        assert_close(SE_con.n_SE, ref["E2E.He_SE_Ne_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.He_SE_Ne_Te.n_LTE"], rtol=1e-8)
