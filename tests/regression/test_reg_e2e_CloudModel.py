"""End-to-end regression tests for Cloud Model pipeline"""

from spectra.Function import SlabModel
from spectra.Function.SEquil import SELib
from spectra.ImportAll import *
from spectra.Struct import Atmosphere, Atom, Radiation

from .conftest import assert_close


class TestHydrogenCloudModel:
    def test_slab_0D(self, ref):
        conf_path = str(CFG._ROOT_DIR / "data/conf/H.conf")
        atom, wMesh, _ = Atom.init_Atom_(conf_path, is_hydrogen=True)

        atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_(atmos, wMesh)
        SE_con, _ = SELib.cal_SE_with_Nh_Te_(atom, atmos, wMesh, radiation, None)

        Cloud_con = SlabModel.SE_to_slab_0D_(atom, atmos, SE_con, depth=1.0e3 * 1.0e5)

        assert_close(Cloud_con.w0, ref["E2E.H_CloudModel.w0"], rtol=1e-8)
        assert_close(Cloud_con.tau_max, ref["E2E.H_CloudModel.tau_max"], rtol=1e-8)
        assert_close(Cloud_con.Ibar, ref["E2E.H_CloudModel.Ibar"], rtol=1e-8)
