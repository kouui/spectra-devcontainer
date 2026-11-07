"""Tests of the RH atom reader and `Atom.init_Atom_from_RH_` on the 6-level
hydrogen atom of the FALC case (`tests/data/rh_falc/H_6_CRD.atom`).

Expected values are written out from the file (or derived independently of the
reader); the file is the single source. Fast, so they also run under
NUMBA_DISABLE_JIT=1.
"""

from pathlib import Path

import numpy as np
import pytest

from spectra import Constants as CST
from spectra.Function.SEquil import SELib
from spectra.Struct import Atmosphere, Radiation
from spectra.Struct import Atom as Atom_
from spectra.Util.AtomUtils import RHAtomIO

RH_ATOM = Path(__file__).resolve().parents[1] / "data" / "rh_falc" / "H_6_CRD.atom"
_E_CM = np.array([0.0, 82258.211, 97491.219, 102822.766, 105290.508, 109677.617])


@pytest.fixture(scope="module")
def atom():
    return Atom_.init_Atom_from_RH_(RH_ATOM)


@pytest.fixture(scope="module")
def atom_text():
    return RH_ATOM.read_text()


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "x.atom"
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------


def test_counts(atom):
    assert (atom.nLevel, atom.nBoundLevel, atom.nLine, atom.nCont, atom.nTran, atom.nRL) == (6, 5, 10, 5, 15, 10)
    assert atom._has_continuum
    assert atom.Z == 1
    assert atom._atom_type.name == "HYDROGEN"


def test_level_values(atom):
    # same product, same order as the file's cm^-1 -> erg conversion
    np.testing.assert_array_equal(atom.Level["erg"], _E_CM * CST.h_ * CST.c_)
    np.testing.assert_array_equal(atom.Level["g"], [2, 8, 18, 32, 50, 1])
    assert atom.Level["g"].dtype.kind == "i"
    np.testing.assert_array_equal(atom.Level["stage"], [1, 1, 1, 1, 1, 2])
    np.testing.assert_array_equal(atom.Level["isGround"], [True, False, False, False, False, True])
    # n comes from the label ('H I 3D 2DE' -> 3); the bare proton has none, although its label reads 6H
    np.testing.assert_array_equal(atom.Level["n"], [1, 2, 3, 4, 5, 0])
    np.testing.assert_array_equal(atom.Cont["nj"], np.zeros(5, dtype=int))


def test_level_gamma_is_the_summed_Aji(atom):
    # level 1 (2P) decays only to 1S: gamma = A(2P->1S), which is the radiative
    # damping the file lists for Lyman alpha (4.70e8)
    assert atom.Level["gamma"][0] == 0.0
    assert atom.Level["gamma"][1] == pytest.approx(atom.Line["AJI"][0])
    assert atom.Level["gamma"][1] == pytest.approx(4.70e8, rel=1e-2)


def test_rh_level_data(atom):
    assert atom.RH is not None
    np.testing.assert_array_equal(atom.RH.level_E_cm, _E_CM)
    np.testing.assert_array_equal(atom.RH.level_no, np.arange(6))
    assert atom.RH.level_label[0] == "H I 1S 2SE"
    assert atom.RH.level_label[5] == "H I 6H 2HO"


@pytest.mark.parametrize(
    ("label", "n"), [("H I 1S 2SE", 1), ("H I 12K 2KE", 12), ("Ca II 3D 2DE", 3)], ids=["1S", "12K", "CaII"]
)
def test_principal_n_from_label(label, n):
    assert Atom_._principal_n_from_label_(label) == n


def test_principal_n_without_a_number_raises():
    with pytest.raises(ValueError, match="principal quantum number"):
        Atom_._principal_n_from_label_("H I S 2SE")


# ---------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------

_LINES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]  # (lower, upper)


def test_only_listed_lines_in_file_order(atom):
    np.testing.assert_array_equal(atom.Line["idxI"], [i for i, _ in _LINES])
    np.testing.assert_array_equal(atom.Line["idxJ"], [j for _, j in _LINES])
    assert atom._idx_table.Line == tuple(_LINES)
    assert atom._ctj_table.Line[0] == (("H I 1S 2SE", "-", "-"), ("H I 2P 2PO", "-", "-"))


def test_line_physics(atom):
    wl = 1.0 / (_E_CM[1] - _E_CM[0])  # [cm]
    # Lyman alpha, f = 0.4162: A = 2 pi e^2 f / (eps0 m c lambda^2) * gi / gj
    aji = 0.4162 * (6.670192e-5 / (wl * 1.0e-2) ** 2) * (2.0 / 8.0)
    assert atom.Line["AJI"][0] == pytest.approx(aji, rel=1e-14)
    assert atom.Line["w0"][0] == pytest.approx(wl, rel=1e-12)
    np.testing.assert_array_equal(atom.Line["gi"][:2], [2, 2])
    np.testing.assert_array_equal(atom.Line["ni"][:5], [1, 1, 1, 1, 2])
    np.testing.assert_array_equal(atom.Line["nj"][:5], [2, 3, 4, 5, 3])


def test_rh_line_values(atom):
    rh = atom.RH
    assert rh is not None
    np.testing.assert_array_equal(
        rh.line_Grad, [4.70e8, 9.98e7, 3.02e7, 1.15e7, 9.98e7, 3.02e7, 1.15e7, 3.02e7, 1.15e7, 1.15e7]
    )
    np.testing.assert_array_equal(rh.line_cStark, np.ones(10))
    assert rh.line_vdW_type == ("UNSOLD",) * 10
    assert rh.line_cvdW.shape == (10, 4)
    np.testing.assert_array_equal(rh.line_cvdW, np.tile([1.0, 0.0, 1.0, 0.0], (10, 1)))
    np.testing.assert_array_equal(rh.line_Nlambda, [100, 50, 20, 20, 70, 40, 40, 20, 20, 20])
    assert rh.line_symm == ("ASYMM", "ASYMM", "SYMM", "SYMM", "ASYMM", "SYMM", "SYMM", "SYMM", "SYMM", "SYMM")
    np.testing.assert_array_equal(rh.line_qcore, [15.0, 10.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0])
    np.testing.assert_array_equal(rh.line_qwing, [600.0, 250.0, 100.0, 100.0, 250.0, 250.0, 250.0, 30.0, 30.0, 30.0])


# ---------------------------------------------------------------------------
# continua and PI
# ---------------------------------------------------------------------------


def test_continua(atom):
    assert atom._idx_table.Cont == tuple((i, 5) for i in range(5))
    np.testing.assert_array_equal(atom.Cont["idxI"], np.arange(5))
    np.testing.assert_array_equal(atom.Cont["idxJ"], np.full(5, 5))
    rh = atom.RH
    assert rh is not None
    np.testing.assert_array_equal(rh.cont_alpha0_m2, [6.152e-22, 1.379e-21, 2.149e-21, 2.923e-21, 3.699e-21])
    np.testing.assert_array_equal(rh.cont_lamb_min_nm, [22.794, 91.176, 205.147, 364.705, 569.852])
    np.testing.assert_array_equal(rh.cont_Nlambda, np.full(5, 20))
    assert rh.cont_wavedep == ("HYDROGENIC",) * 5


def test_PI_is_the_default_mesh_hydrogenic_cross_section(atom):
    mesh = atom._wave_mesh.Cont_mesh
    assert mesh.shape == (5, 41)
    np.testing.assert_array_equal(mesh[:, 0], atom.Cont["w0"])  # edge first
    assert atom.PI.alpha_interp.shape == (5, 41)
    assert np.all(atom.PI.alpha_interp > 0.0)
    # the edge value is the project's formula, within 3 % of the file's alpha0 (n = 1)
    np.testing.assert_allclose(atom.PI.alpha_interp[:, 0], atom.PI.Coe["alpha0"], rtol=0.03)
    np.testing.assert_array_equal(atom.PI.Coe["alpha0"], atom.RH.cont_alpha0_m2 * 1.0e4)  # m^2 -> cm^2
    np.testing.assert_array_equal(atom.PI.alpha_table[1], atom.PI.alpha_interp.ravel())
    np.testing.assert_array_equal(atom.PI.alpha_table_idxs[:, 1] - atom.PI.alpha_table_idxs[:, 0], np.full(5, 41))
    np.testing.assert_array_equal(atom.PI.Coe["dEij"], atom.Level["erg"][5] - atom.Level["erg"][:5])


# ---------------------------------------------------------------------------
# collisions
# ---------------------------------------------------------------------------


def test_collision_rows_in_cgs_and_file_order(atom):
    rh = atom.RH
    assert rh is not None
    np.testing.assert_array_equal(rh.Te_table, [3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0])
    # CE rows: upper level first
    np.testing.assert_array_equal(
        rh.CE_idx, [[1, 0], [2, 0], [3, 0], [4, 0], [2, 1], [3, 1], [4, 1], [3, 2], [4, 2], [4, 3]]
    )
    np.testing.assert_array_equal(
        rh.CE_coe_cgs[0], np.array([9.750e-16, 6.098e-16, 4.535e-16, 3.365e-16, 2.008e-16, 1.560e-16]) * 1.0e6
    )
    np.testing.assert_array_equal(rh.CE_coe_cgs[-1][-1:], np.array([6.310e-14]) * 1.0e6)
    # CI rows: lower level first
    np.testing.assert_array_equal(rh.CI_idx, [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5]])
    np.testing.assert_array_equal(
        rh.CI_coe_cgs[0], np.array([2.635e-17, 2.864e-17, 3.076e-17, 3.365e-17, 4.138e-17, 4.703e-17]) * 1.0e6
    )
    np.testing.assert_array_equal(rh.CI_coe_cgs[-1][-1:], np.array([1.601e-14]) * 1.0e6)


def test_CE_CI_are_empty_placeholders(atom):
    for c in (atom.CE, atom.CI):
        assert c.Te_table.size == 0
        assert c.Omega_table.shape[1] == 0
    assert atom.CE.Coe.shape == (10,)
    assert atom.CI.Coe.shape == (5,)


# ---------------------------------------------------------------------------
# wavelength mesh built from RL.Coe
# ---------------------------------------------------------------------------


def test_wave_mesh_line_coe(atom):
    coe = atom._wave_mesh.Line_Coe
    # ASYMM counts the full profile, SYMM one half (2N - 1 points); every count is odd
    n_expected = [101, 51, 39, 39, 71, 79, 79, 39, 39, 39]
    np.testing.assert_array_equal(coe["nLambda"], n_expected)
    np.testing.assert_array_equal(coe["qcore"], atom.RH.line_qcore)
    np.testing.assert_array_equal(coe["qwing"], atom.RH.line_qwing)
    np.testing.assert_array_equal(coe["ProfileType"], np.zeros(10, dtype=int))  # VOIGT
    np.testing.assert_array_equal(coe["idxI"], atom.Line["idxI"])
    np.testing.assert_array_equal(coe["idxJ"], atom.Line["idxJ"])
    np.testing.assert_array_equal(coe["w0"], atom.Line["w0"])

    idxs = atom._wave_mesh.Line_mesh_idxs
    assert atom._wave_mesh.Line_mesh.size == sum(n_expected)
    np.testing.assert_array_equal(idxs[:, 1] - idxs[:, 0], n_expected)
    # symmetric mesh: the last point of line 0 is +qwing, the first -qwing
    first = atom._wave_mesh.Line_mesh[idxs[0, 0] : idxs[0, 1]]
    assert first[0] == pytest.approx(-600.0)
    assert first[-1] == pytest.approx(600.0)

    np.testing.assert_array_equal(atom.RL.Coe["lineIndex"], np.arange(10))
    assert atom.RL.nRadiativeLine == 10


# ---------------------------------------------------------------------------
# reader: unsupported input
# ---------------------------------------------------------------------------


def test_unknown_keyword_raises_naming_it(atom_text, tmp_path):
    text = atom_text.replace("\nEND", "\n OHM      1  0    1.0e-16  1.0e-16\nEND")
    assert text != atom_text
    with pytest.raises(NotImplementedError, match="OHM"):
        RHAtomIO.read_RH_atom_(_write(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "name"),
    [
        ("HYDROGENIC       22.794", "EXPLICIT        22.794", "EXPLICIT"),
        ("VOIGT    100", "PRD_VOIGT 100", "PRD_VOIGT"),
        ("UNSOLD   1.000  0.000  1.000  0.000  4.70E+08", "BARKLEM  1.000  0.000  1.000  0.000  4.70E+08", "BARKLEM"),
        ("ASYMM  15.0", "RANDOM 15.0", "RANDOM"),
        ("    6       10       5        0", "    6       10       5        1", "fixed transitions"),
    ],
    ids=["continuum", "line-type", "vdW", "symmetry", "fixed"],
)
def test_unsupported_features_raise_naming_them(atom_text, tmp_path, old, new, name):
    assert old in atom_text
    with pytest.raises(NotImplementedError, match=name):
        RHAtomIO.read_RH_atom_(_write(tmp_path, atom_text.replace(old, new, 1)))


def test_file_without_end_raises(atom_text, tmp_path):
    with pytest.raises(ValueError, match="END"):
        RHAtomIO.read_RH_atom_(_write(tmp_path, atom_text.replace("END", "")))


def test_decreasing_stage_raises(atom_text, tmp_path):
    # swap the stages of the first and the last level: the levels of a stage must be contiguous
    text = atom_text.replace("2SE          '    0       0", "2SE          '    1       0").replace(
        "2HO          '    1       5", "2HO          '    0       5"
    )
    with pytest.raises(ValueError, match="stages"):
        Atom_.init_Atom_from_RH_(_write(tmp_path, text))


def test_collision_row_without_a_listed_transition_raises(atom_text, tmp_path):
    text = atom_text.replace(" CE      4  3", " CE      5  3", 1)
    with pytest.raises(ValueError, match="CE row"):
        Atom_.init_Atom_from_RH_(_write(tmp_path, text))


# ---------------------------------------------------------------------------
# the old SE path
# ---------------------------------------------------------------------------


def test_old_SE_path_refuses_an_RH_atom(atom):
    atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vt=5.0e5)
    radiation = Radiation.init_Radiation_()
    for call in (
        lambda: SELib.cal_SE_with_Nh_Te_(atom, atmos, radiation, None),
        lambda: SELib.cal_SE_with_Ne_Te_(atom, atmos, radiation, None),
        lambda: SELib.cal_SE_with_Pg_Te_(atom, atmos, radiation, None),
        lambda: SELib.cal_SE_(atom, atmos, radiation, None),
    ):
        with pytest.raises(ValueError, match="RH"):
            call()


def test_conf_atoms_have_no_RH_data():
    from spectra.ImportAll import CFG

    atom = Atom_.init_Atom_(str(CFG._ROOT_DIR / "data/conf/H6.conf"), is_hydrogen=True)
    assert atom.RH is None
