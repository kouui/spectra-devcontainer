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
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Nh_Te_(atom, atmos, wMesh, radiation, None)

        assert_close(SE_con.n_SE, ref["E2E.H_SE_Nh_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.H_SE_Nh_Te.n_LTE"], rtol=1e-8)
        assert_close(atmos.Ne, ref["E2E.H_SE_Nh_Te.Ne"], rtol=1e-8)

    def test_SE_with_Ne_Te(self, ref):
        atom, wMesh, _ = self._load_H()
        atmos = Atmosphere.Atmosphere0D(Nh=1.0e11, Ne=5.0e10, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Ne_Te_(atom, atmos, wMesh, radiation, None)

        assert_close(SE_con.n_SE, ref["E2E.H_SE_Ne_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.H_SE_Ne_Te.n_LTE"], rtol=1e-8)

    def test_SE_with_Pg_Te(self, ref):
        atom, wMesh, _ = self._load_H()
        # Nh/Ne are overwritten by cal_SE_with_Pg_Te_ for H (is_hydrogen=True
        # branch sets both from Pg); values here are placeholder only.
        atmos = Atmosphere.Atmosphere0D(Pg=1.8, Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Pg_Te_(atom, atmos, wMesh, radiation, None)

        assert_close(SE_con.n_SE, ref["E2E.H_SE_Pg_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.H_SE_Pg_Te.n_LTE"], rtol=1e-8)
        assert_close(atmos.Ne, ref["E2E.H_SE_Pg_Te.Ne"], rtol=1e-8)
        assert_close(SE_con.Ntotal, ref["E2E.H_SE_Pg_Te.Ntotal"], rtol=1e-8)


class TestHeliumSE:
    def _load_He(self):
        conf_path = str(CFG._ROOT_DIR / "data/conf/He.conf")
        return Atom.init_Atom_(conf_path, is_hydrogen=False)

    def test_SE_with_Ne_Te(self, ref):
        atom, wMesh, _ = self._load_He()

        atmos = Atmosphere.Atmosphere0D(Nh=1.0e11, Ne=5.0e10, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Ne_Te_(atom, atmos, wMesh, radiation, None)

        assert_close(SE_con.n_SE, ref["E2E.He_SE_Ne_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.He_SE_Ne_Te.n_LTE"], rtol=1e-8)

    def test_SE_with_Nh_Te(self, ref):
        atom, wMesh, _ = self._load_He()
        atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Nh_Te_(atom, atmos, wMesh, radiation, None)

        # atmos.Ne is unchanged by cal_SE_with_Nh_Te_ for non-H atoms, so
        # asserting it would only re-confirm the input — skipped.
        assert_close(SE_con.n_SE, ref["E2E.He_SE_Nh_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.He_SE_Nh_Te.n_LTE"], rtol=1e-8)
        assert_close(SE_con.Ntotal, ref["E2E.He_SE_Nh_Te.Ntotal"], rtol=1e-8)

    def test_SE_with_Pg_Te(self, ref):
        atom, wMesh, _ = self._load_He()
        # Pg is unused by cal_SE_with_Pg_Te_ for non-H atoms; Nh/Ne drive the result.
        atmos = Atmosphere.Atmosphere0D(Pg=1.8, Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Pg_Te_(atom, atmos, wMesh, radiation, None)

        assert_close(SE_con.n_SE, ref["E2E.He_SE_Pg_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.He_SE_Pg_Te.n_LTE"], rtol=1e-8)
        assert_close(SE_con.Ntotal, ref["E2E.He_SE_Pg_Te.Ntotal"], rtol=1e-8)


class TestCaIISE:
    def _load_Ca_II(self):
        conf_path = str(CFG._ROOT_DIR / "data/conf/Ca_II.conf")
        return Atom.init_Atom_(conf_path, is_hydrogen=False)

    def test_SE_with_Nh_Te(self, ref):
        atom, wMesh, _ = self._load_Ca_II()
        atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Nh_Te_(atom, atmos, wMesh, radiation, None)

        # atmos.Ne unchanged by cal_SE_with_Nh_Te_ for non-H; skipped.
        assert_close(SE_con.n_SE, ref["E2E.Ca_II_SE_Nh_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.Ca_II_SE_Nh_Te.n_LTE"], rtol=1e-8)
        assert_close(SE_con.Ntotal, ref["E2E.Ca_II_SE_Nh_Te.Ntotal"], rtol=1e-8)

    def test_SE_with_Ne_Te(self, ref):
        atom, wMesh, _ = self._load_Ca_II()
        atmos = Atmosphere.Atmosphere0D(Nh=1.0e11, Ne=5.0e10, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Ne_Te_(atom, atmos, wMesh, radiation, None)

        # Matches the existing H/He Ne_Te pattern: n_SE + n_LTE only. atmos.Ne
        # is the input (not mutated), and SE_con.Ntotal = atmos.Nh / atom.Abun
        # where atmos.Nh is overwritten to 2*atmos.Ne in the non-H branch; the
        # n_SE array already encodes that population, so Ntotal would add little.
        assert_close(SE_con.n_SE, ref["E2E.Ca_II_SE_Ne_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.Ca_II_SE_Ne_Te.n_LTE"], rtol=1e-8)

    def test_SE_with_Pg_Te(self, ref):
        atom, wMesh, _ = self._load_Ca_II()
        atmos = Atmosphere.Atmosphere0D(Pg=1.8, Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Pg_Te_(atom, atmos, wMesh, radiation, None)

        # atmos.Ne unchanged by cal_SE_with_Pg_Te_ for non-H; skipped.
        assert_close(SE_con.n_SE, ref["E2E.Ca_II_SE_Pg_Te.n_SE"], rtol=1e-8)
        assert_close(SE_con.n_LTE, ref["E2E.Ca_II_SE_Pg_Te.n_LTE"], rtol=1e-8)
        assert_close(SE_con.Ntotal, ref["E2E.Ca_II_SE_Pg_Te.Ntotal"], rtol=1e-8)
