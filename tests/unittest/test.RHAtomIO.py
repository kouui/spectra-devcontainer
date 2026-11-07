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
from spectra.Enums import E_COLLISIONAL_TRANSITION_FORMULA as E_FORMULA
from spectra.Enums import E_VDW_TYPE
from spectra.Function.SEquil import SELib
from spectra.Struct import Atmosphere, Radiation
from spectra.Struct import Atom as Atom_
from spectra.Util import MeshUtil
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


def test_level_labels_are_the_ctj_info(atom):
    assert atom._ctj_table.Level[0] == ("H I 1S 2SE", "-", "-")
    assert atom._ctj_table.Level[5] == ("H I 6H 2HO", "-", "-")


def test_level_energy_round_trips_to_cm_inverse(atom):
    # a later stage recovers the file's cm^-1 from erg
    back = atom.Level["erg"] / (CST.h_ * CST.c_)
    np.testing.assert_allclose(back, _E_CM, rtol=1e-15, atol=0.0)


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


def test_line_broadening_in_RL_coe(atom):
    coe = atom.RL.Coe
    np.testing.assert_array_equal(coe["lineIndex"], np.arange(10))
    np.testing.assert_array_equal(coe["idxI"], atom.Line["idxI"])
    np.testing.assert_array_equal(coe["idxJ"], atom.Line["idxJ"])
    # Grad [s^-1]: the file's value, no conversion
    np.testing.assert_array_equal(
        coe["Grad"], [4.70e8, 9.98e7, 3.02e7, 1.15e7, 9.98e7, 3.02e7, 1.15e7, 3.02e7, 1.15e7, 1.15e7]
    )
    assert np.all(coe["vdW_type"] == int(E_VDW_TYPE.UNSOLD))
    # UNSOLD multipliers: dimensionless, the file's value
    np.testing.assert_array_equal(coe["cvdW"], np.tile([1.0, 0.0, 1.0, 0.0], (10, 1)))
    # positive cStark: dimensionless multiplier, unchanged
    np.testing.assert_array_equal(coe["cStark"], np.ones(10))


def test_negative_cStark_is_converted_and_keeps_its_sign(atom_text, tmp_path):
    # RH: cStark < 0 means Gamma / Ne = |cStark| in m^3 s^-1 (broad.c, Stark())
    old = "4.70E+08  1.0E+00"
    assert old in atom_text
    changed = Atom_.init_Atom_from_RH_(_write(tmp_path, atom_text.replace(old, "4.70E+08 -2.5E-15")))
    assert changed.RL.Coe["cStark"][0] == pytest.approx(-2.5e-15 * 1.0e6, rel=1e-15)  # cm^3 s^-1
    np.testing.assert_array_equal(changed.RL.Coe["cStark"][1:], np.ones(9))


def test_zero_Grad_in_an_RH_file_is_stored_as_zero_with_vdW_type_set(atom_text, tmp_path):
    # in an RH file Grad = 0 is "no radiative damping", not "use Line.Gamma"
    old = "4.70E+08  1.0E+00"
    assert old in atom_text
    a = Atom_.init_Atom_from_RH_(_write(tmp_path, atom_text.replace(old, "0.00E+00  1.0E+00")))
    assert a.RL.Coe["Grad"][0] == 0.0
    assert a.RL.Coe["vdW_type"][0] == int(E_VDW_TYPE.UNSOLD)


def test_mesh_counts_follow_the_file(atom):
    # raw Nlambda 100 50 20 20 70 40 40 20 20 20 with ASYMM, ASYMM, SYMM, SYMM, ASYMM, SYMM...:
    # the file's count for ASYMM (+1 when even), 2N - 1 for SYMM
    np.testing.assert_array_equal(atom.RL.Coe["nLambda"], [101, 51, 39, 39, 71, 79, 79, 39, 39, 39])


# ---------------------------------------------------------------------------
# continua and PI
# ---------------------------------------------------------------------------


_LAMB_MIN_NM = [22.794, 91.176, 205.147, 364.705, 569.852]
_ALPHA0_M2 = [6.152e-22, 1.379e-21, 2.149e-21, 2.923e-21, 3.699e-21]


def test_continua(atom):
    assert atom._idx_table.Cont == tuple((i, 5) for i in range(5))
    np.testing.assert_array_equal(atom.Cont["idxI"], np.arange(5))
    np.testing.assert_array_equal(atom.Cont["idxJ"], np.full(5, 5))
    np.testing.assert_array_equal(atom.Cont["nj"], np.zeros(5, dtype=int))


def test_alpha0_is_stored_in_cm2(atom):
    np.testing.assert_array_equal(atom.PI.Coe["alpha0"], np.array(_ALPHA0_M2) * 1.0e4)


def test_lamb_min_is_stored_in_cm_and_ends_the_mesh(atom):
    cc = atom._wave_mesh.Cont_Coe
    np.testing.assert_array_equal(cc["lamb_min"], np.array(_LAMB_MIN_NM) * 1.0e-7)
    mesh = atom._wave_mesh.Cont_mesh
    np.testing.assert_array_equal(mesh[:, 0], atom.Cont["w0"])  # edge first
    np.testing.assert_allclose(mesh[:, -1], cc["lamb_min"], rtol=1e-14)
    assert np.all(np.diff(mesh, axis=1) < 0.0)


def test_mesh_without_lamb_min_is_the_default_one():
    from spectra.ImportAll import CFG

    h = Atom_.init_Atom_(str(CFG._ROOT_DIR / "data/conf/H6.conf"), is_hydrogen=True)
    np.testing.assert_array_equal(h._wave_mesh.Cont_Coe["lamb_min"], np.zeros(h.nCont))
    expected = h.Cont["w0"][:, None] * MeshUtil.make_continuum_mesh_(41)[None, :]
    np.testing.assert_array_equal(h._wave_mesh.Cont_mesh, expected)


def test_non_positive_lamb_min_in_an_RH_file_raises(atom_text, tmp_path):
    bad = atom_text.replace("HYDROGENIC       22.794", "HYDROGENIC        0.000")
    with pytest.raises(ValueError, match="lamb_min"):
        Atom_.init_Atom_from_RH_(_write(tmp_path, bad))


def test_lamb_min_beyond_the_edge_raises(atom_text, tmp_path):
    bad = atom_text.replace("HYDROGENIC       22.794", "HYDROGENIC       220.794")
    with pytest.raises(ValueError, match="lamb_min"):
        Atom_.init_Atom_from_RH_(_write(tmp_path, bad))


def test_PI_is_the_hydrogenic_cross_section_on_the_continuum_mesh(atom):
    assert atom.PI.alpha_interp.shape == (5, 41)
    assert np.all(atom.PI.alpha_interp > 0.0)
    # the edge value is the project's formula, within 3 % of the file's alpha0 (n = 1)
    np.testing.assert_allclose(atom.PI.alpha_interp[:, 0], atom.PI.Coe["alpha0"], rtol=0.03)
    np.testing.assert_array_equal(atom.PI.alpha_table[1], atom.PI.alpha_interp.ravel())
    np.testing.assert_array_equal(atom.PI.alpha_table[0], atom._wave_mesh.Cont_mesh.ravel())
    np.testing.assert_array_equal(atom.PI.Coe["dEij"], atom.Level["erg"][5] - atom.Level["erg"][:5])


# ---------------------------------------------------------------------------
# collisions
# ---------------------------------------------------------------------------


_CE_ROW_0 = [9.750e-16, 6.098e-16, 4.535e-16, 3.365e-16, 2.008e-16, 1.560e-16]  # "CE 1 0"
_CE_LAST = [7.728e-14, 7.317e-14, 7.199e-14, 7.109e-14, 6.752e-14, 6.310e-14]  # "CE 4 3"
_CI_ROW_0 = [2.635e-17, 2.864e-17, 3.076e-17, 3.365e-17, 4.138e-17, 4.703e-17]  # "CI 0 5"
_CI_LAST = [1.342e-14, 1.588e-14, 1.710e-14, 1.786e-14, 1.743e-14, 1.601e-14]  # "CI 4 5"


def test_collisions_in_cgs(atom):
    for tran in (atom.CE, atom.CI):
        np.testing.assert_array_equal(tran.Te_table, [3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0])
    assert atom.CE._transition_formula == E_FORMULA.RH_CE
    assert atom.CI._transition_formula == E_FORMULA.RH_CI
    # SI m^3 -> cgs cm^3: x1e6, once
    assert atom.CE.Omega_table.shape == (10, 6)
    np.testing.assert_array_equal(atom.CE.Omega_table[0], np.array(_CE_ROW_0) * 1.0e6)
    np.testing.assert_array_equal(atom.CE.Omega_table[-1], np.array(_CE_LAST) * 1.0e6)
    assert atom.CI.Omega_table.shape == (5, 6)
    np.testing.assert_array_equal(atom.CI.Omega_table[0], np.array(_CI_ROW_0) * 1.0e6)
    np.testing.assert_array_equal(atom.CI.Omega_table[-1], np.array(_CI_LAST) * 1.0e6)


def test_collision_coe_has_the_lower_level_first_for_CE_and_CI(atom):
    # the file lists CE with the upper level first ("CE 1 0") and CI with the lower first ("CI 0 5")
    np.testing.assert_array_equal(atom.CE.Coe["idxI"], atom.Line["idxI"])
    np.testing.assert_array_equal(atom.CE.Coe["idxJ"], atom.Line["idxJ"])
    np.testing.assert_array_equal(atom.CI.Coe["idxI"], atom.Cont["idxI"])
    np.testing.assert_array_equal(atom.CI.Coe["idxJ"], atom.Cont["idxJ"])
    assert np.all(atom.CE.Coe["idxI"] < atom.CE.Coe["idxJ"])
    np.testing.assert_array_equal(atom.CE.Coe["gi"], atom.Line["gi"])
    np.testing.assert_array_equal(atom.CE.Coe["gj"], atom.Line["gj"])
    np.testing.assert_array_equal(atom.CE.Coe["f1"], np.ones(10, dtype=int))
    np.testing.assert_array_equal(atom.CE.Coe["f2"], np.ones(10, dtype=int))
    np.testing.assert_array_equal(
        atom.CE.Coe["dEij"], atom.Level["erg"][atom.Line["idxJ"]] - atom.Level["erg"][atom.Line["idxI"]]
    )
    np.testing.assert_array_equal(atom.CI.Coe["dEij"], atom.PI.Coe["dEij"])


def test_rows_are_placed_by_transition_not_by_file_order(atom_text, tmp_path):
    # move the last CE row ("CE 4 3") to the front and drop "CE 2 0": the table must not follow the file order
    rows = atom_text.splitlines()
    k_last = next(i for i, r in enumerate(rows) if r.startswith(" CE      4  3"))
    k_first = next(i for i, r in enumerate(rows) if r.startswith(" CE      1  0"))
    k_drop = next(i for i, r in enumerate(rows) if r.startswith(" CE      2  0"))
    moved = rows[k_last]
    rows[k_last] = ""
    rows[k_drop] = ""
    rows.insert(k_first, moved)
    a = Atom_.init_Atom_from_RH_(_write(tmp_path, "\n".join(rows)))
    np.testing.assert_array_equal(a.CE.Omega_table[9], np.array(_CE_LAST) * 1.0e6)  # line (3, 4)
    np.testing.assert_array_equal(a.CE.Omega_table[0], np.array(_CE_ROW_0) * 1.0e6)  # line (0, 1)
    np.testing.assert_array_equal(a.CE.Omega_table[1], np.zeros(6))  # no row for (0, 2)


# ---------------------------------------------------------------------------
# wavelength mesh built from RL.Coe
# ---------------------------------------------------------------------------


def test_wave_mesh_line_coe(atom):
    coe = atom._wave_mesh.Line_Coe
    # ASYMM counts the full profile, SYMM one half (2N - 1 points); every count is odd
    n_expected = [101, 51, 39, 39, 71, 79, 79, 39, 39, 39]
    np.testing.assert_array_equal(coe["nLambda"], n_expected)
    np.testing.assert_array_equal(coe["qcore"], [15.0, 10.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0])
    np.testing.assert_array_equal(coe["qwing"], [600.0, 250.0, 100.0, 100.0, 250.0, 250.0, 250.0, 30.0, 30.0, 30.0])
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


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        # RH accepts the pair in either order; the reader requires (upper, lower)
        ("  1   0  4.162E-01", "  0   1  4.162E-01", "upward"),
        # RH accepts a 16th column (g_Lande_eff)
        ("4.70E+08  1.0E+00", "4.70E+08  1.0E+00  0.0", "columns"),
        # RH accepts level rows without levelNo
        ("2SE          '    0       0", "2SE          '    0", "levelNo"),
        # RH sums repeated CE rows; the reader refuses them
        (
            " CE      1  0",
            " CE      1  0    1.0e-16  1.0e-16  1.0e-16  1.0e-16  1.0e-16  1.0e-16\n CE      1  0",
            "twice",
        ),
    ],
    ids=["swapped-pair", "16th-column", "no-levelNo", "duplicate-CE"],
)
def test_input_that_RH_accepts_but_the_reader_does_not(atom_text, tmp_path, old, new, match):
    assert old in atom_text
    with pytest.raises(ValueError, match=match):
        Atom_.init_Atom_from_RH_(_write(tmp_path, atom_text.replace(old, new, 1)))


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


def _fresh_atmos(**kw):
    return Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vt=5.0e5, **kw)


@pytest.mark.parametrize(
    "call",
    [
        lambda atom, atmos, rad: SELib.cal_SE_with_Nh_Te_(atom, atmos, rad, None),
        lambda atom, atmos, rad: SELib.cal_SE_with_Ne_Te_(atom, atmos, rad, None),
        lambda atom, atmos, rad: SELib.cal_SE_with_Pg_Te_(atom, atmos, rad, None),
        lambda atom, atmos, rad: SELib.cal_SE_with_Pg_Te_single_Atom_(atom, atmos, rad),
        lambda atom, atmos, rad: SELib.cal_SE_(atom, atmos, rad, None),
    ],
    ids=["Nh_Te", "Ne_Te", "Pg_Te", "Pg_Te_single_Atom", "cal_SE_"],
)
def test_old_SE_path_refuses_an_RH_atom_before_touching_the_atmosphere(atom, call):
    atmos = _fresh_atmos(Pg=1.8)
    before = dict(vars(atmos))
    with pytest.raises(ValueError, match="RH_CE"):
        call(atom, atmos, Radiation.init_Radiation_())
    assert dict(vars(atmos)) == before


def test_conf_atoms_have_default_broadening_and_omega_collisions():
    from spectra.ImportAll import CFG

    atom = Atom_.init_Atom_(str(CFG._ROOT_DIR / "data/conf/H6.conf"), is_hydrogen=True)
    assert atom.CE._transition_formula == E_FORMULA.OMEGA
    assert atom.CI._transition_formula == E_FORMULA.OMEGA
    coe = atom.RL.Coe
    assert np.all(coe["vdW_type"] == int(E_VDW_TYPE.NONE))
    for f in ("Grad", "cvdW", "cStark"):
        assert not np.any(coe[f])
