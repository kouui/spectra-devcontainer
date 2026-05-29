"""End-to-end regression tests for Cloud Model pipeline"""

import numpy as _numpy

from spectra.Function import SlabModel
from spectra.Function.SEquil import SELib
from spectra.ImportAll import *
from spectra.Struct import Atmosphere, Atom, Radiation

from .conftest import assert_close


class TestHydrogenCloudModel:
    def test_slab_0D(self, ref):
        conf_path = str(CFG._ROOT_DIR / "data/conf/H.conf")
        atom, wMesh, _ = Atom.init_Atom_(conf_path, is_hydrogen=True)

        atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Nh_Te_(atom, atmos, wMesh, radiation, None)

        Cloud_con = SlabModel.SE_to_slab_0D_(atom, atmos, SE_con, depth=1.0e3 * 1.0e5)

        assert_close(Cloud_con.w0, ref["E2E.H_CloudModel.w0"], rtol=1e-8)
        assert_close(Cloud_con.tau_max, ref["E2E.H_CloudModel.tau_max"], rtol=1e-8)
        assert_close(Cloud_con.Ibar, ref["E2E.H_CloudModel.Ibar"], rtol=1e-8)

    def test_slab_0D_population_inversion(self):
        """Lock in the abs() semantics of `tau_max` under population inversion.

        Issue #18: when `n_upper > n_lower` (per-degeneracy), `alp0 < 0` and
        `tau` becomes negative. Plain `tau.max()` then returns the least-negative
        value, masking the strongest |tau|. The fix takes `|tau|.max()`.

        Synthetic setup (not physically self-consistent — only `n_SE` is mutated;
        absorb_prof_1d / Line_mesh_idxs come from the proper SE pass and stay
        valid because they don't depend on populations).
        """
        conf_path = str(CFG._ROOT_DIR / "data/conf/H.conf")
        atom, wMesh, _ = Atom.init_Atom_(conf_path, is_hydrogen=True)

        atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vt=5.0e5)
        radiation = Radiation.init_Radiation_()
        SE_con, _ = SELib.cal_SE_with_Nh_Te_(atom, atmos, wMesh, radiation, None)

        # Force inversion on line 0 by swapping J/I populations. After the swap
        # n_upper >> n_lower (in thermal H I the lower level dominates), giving
        # Bji*nj > Bij*ni regardless of degeneracy weights.
        j0 = int(atom.Line["idxJ"][0])
        i0 = int(atom.Line["idxI"][0])
        SE_con.n_SE[j0], SE_con.n_SE[i0] = SE_con.n_SE[i0], SE_con.n_SE[j0]

        # exp(-tau) overflows when |tau| is large negative (synthetic extreme,
        # not a real-world concern). prof_1D is irrelevant for this test.
        with _numpy.errstate(over="ignore"):
            Cloud_con = SlabModel.SE_to_slab_0D_(atom, atmos, SE_con, depth=1.0e3 * 1.0e5)

        # Sanity: line 0 actually went negative.
        i1, i2 = int(Cloud_con.Line_mesh_idxs[0, 0]), int(Cloud_con.Line_mesh_idxs[0, 1])
        assert _numpy.any(Cloud_con.tau_1D[i1:i2] < 0), "expected negative tau on inverted line 0"

        # Contract: tau_max[k] == max(|tau_1D[i1:i2]|) for every line, signed or not.
        # Without abs() in the implementation, this would fail on line 0 (where
        # tau.max() picks the least-negative value, not the largest magnitude).
        for k in range(atom.nLine):
            i1, i2 = int(Cloud_con.Line_mesh_idxs[k, 0]), int(Cloud_con.Line_mesh_idxs[k, 1])
            expected = _numpy.abs(Cloud_con.tau_1D[i1:i2]).max()
            assert_close(Cloud_con.tau_max[k], expected, rtol=1e-12)

        assert _numpy.all(Cloud_con.tau_max >= 0.0)
