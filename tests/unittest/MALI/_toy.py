# -------------------------------------------------------------------------------
# toy builders for the MALI tests
#
# the toy atoms are fabricated instead of loaded from data/atom/** : every
# oracle in the tests is analytic or structural, and real atomic data would add
# failure modes (IO, continua, damping) without adding verification power.
# -------------------------------------------------------------------------------

import numpy as _numpy

from spectra.Atomic import LTELib as _LTELib
from spectra.Function.MALI.Setup import Atmos1D, Toy_Atom
from spectra.ImportAll import *
from spectra.Util import MeshUtil as _MeshUtil

# level/line table layouts mirror the fields the reused SELib primitives read
# (_ni_nj_LTE_ : Level["isGround"], Line["gi","gj","f0","idxI","idxJ"]).
_LEVEL_DTYPE = _numpy.dtype([("g", DT_NB_FLOAT), ("isGround", "?"), ("erg", DT_NB_FLOAT)])
_LINE_DTYPE = _numpy.dtype(
    [
        ("idxI", DT_NB_INT),
        ("idxJ", DT_NB_INT),
        ("gi", DT_NB_FLOAT),
        ("gj", DT_NB_FLOAT),
        ("f0", DT_NB_FLOAT),  # [hz]
        ("w0", DT_NB_FLOAT),  # [cm]
        ("AJI", DT_NB_FLOAT),  # [s^-1]
        ("BJI", DT_NB_FLOAT),  # wavelength-base, from einsteinA_to_einsteinBs_cm_
        ("BIJ", DT_NB_FLOAT),
        ("ProfileType", DT_NB_INT),  # E_ABSORPTION_PROFILE_TYPE value
    ]
)
# _ni_nj_LTE_ takes a Cont table; the toys are line-only so it is empty but must
# carry the fields the function reads inside its (skipped) nCont > 0 branch.
_CONT_DTYPE = _numpy.dtype(
    [("idxI", DT_NB_INT), ("idxJ", DT_NB_INT), ("gi", DT_NB_FLOAT), ("gj", DT_NB_FLOAT), ("f0", DT_NB_FLOAT)]
)


def make_toy_atom_(
    level_g: T_ARRAY,
    level_erg: T_ARRAY,
    line_pairs: T_LIST,
    line_Aji: T_ARRAY,
    line_Cij_coe: T_ARRAY,
    am: T_FLOAT = 1.0,
    proftype: T_E_ABSORPTION_PROFILE_TYPE = E_ABSORPTION_PROFILE_TYPE.GAUSSIAN,
) -> Toy_Atom:
    """Fabricate a line-only toy atom.

    Line wavelengths derive from the level energies (w0 = h*c/(Ej-Ei)), so a
    3-level toy is automatically Rydberg-consistent: 1/w02 = 1/w01 + 1/w12.

    Input:
        level_g: (nLevel,), statistical weights
        level_erg: (nLevel,), level energies, [erg], ascending, level_erg[0] = 0
        line_pairs: list of (i, j) level-index tuples, i < j
        line_Aji: (nLine,), Einstein A per line, [s^-1]
        line_Cij_coe: (nLine,), collisional excitation coefficient per line, [cm^3 s^-1]
        am: (,), relative atomic mass
        proftype: absorption profile type for every line
    """
    nLevel = len(level_g)
    nLine = len(line_pairs)

    Level = _numpy.zeros(nLevel, dtype=_LEVEL_DTYPE)
    Level["g"][:] = level_g
    Level["erg"][:] = level_erg
    Level["isGround"][:] = False
    Level["isGround"][0] = True

    Line = _numpy.zeros(nLine, dtype=_LINE_DTYPE)
    for k, (i, j) in enumerate(line_pairs):
        Eji = float(level_erg[j] - level_erg[i])
        w0 = CST.h_ * CST.c_ / Eji
        Bji, Bij = _LTELib.einsteinA_to_einsteinBs_cm_(float(line_Aji[k]), w0, int(level_g[i]), int(level_g[j]))
        Line["idxI"][k] = i
        Line["idxJ"][k] = j
        Line["gi"][k] = level_g[i]
        Line["gj"][k] = level_g[j]
        Line["f0"][k] = Eji / CST.h_
        Line["w0"][k] = w0
        Line["AJI"][k] = line_Aji[k]
        Line["BJI"][k] = Bji
        Line["BIJ"][k] = Bij
        Line["ProfileType"][k] = int(proftype)

    return Toy_Atom(
        Level=Level,
        Line=Line,
        Cont=_numpy.empty(0, dtype=_CONT_DTYPE),
        Cij_coe=_numpy.asarray(line_Cij_coe, dtype=DT_NB_FLOAT),
        Cont_mesh=_numpy.empty((0, 0), dtype=DT_NB_FLOAT),
        alpha=_numpy.empty((0, 0), dtype=DT_NB_FLOAT),
        am=am,
        nLevel=nLevel,
        nLine=nLine,
        nCont=0,
    )


def make_toy_atom_2lv_(
    w0_cm: T_FLOAT = 5000.0e-8,
    Aji: T_FLOAT = 1.0e8,
    Cij_coe: T_FLOAT = 1.0e-8,
    am: T_FLOAT = 1.0,
) -> Toy_Atom:
    E1 = CST.h_ * CST.c_ / w0_cm
    return make_toy_atom_(
        level_g=_numpy.array([1.0, 3.0]),
        level_erg=_numpy.array([0.0, E1]),
        line_pairs=[(0, 1)],
        line_Aji=_numpy.array([Aji]),
        line_Cij_coe=_numpy.array([Cij_coe]),
        am=am,
    )


def make_toy_atom_3lv_(
    w0_01_cm: T_FLOAT = 5000.0e-8,
    w0_12_cm: T_FLOAT = 12000.0e-8,
    Aji: T_ARRAY | None = None,  # (A10, A21, A20)
    Cij_coe: T_ARRAY | None = None,  # (C01, C12, C02)
    am: T_FLOAT = 1.0,
) -> Toy_Atom:
    E1 = CST.h_ * CST.c_ / w0_01_cm
    E2 = E1 + CST.h_ * CST.c_ / w0_12_cm
    if Aji is None:
        Aji = _numpy.array([1.0e8, 3.0e7, 1.0e7])
    if Cij_coe is None:
        Cij_coe = _numpy.array([1.0e-8, 1.0e-8, 1.0e-8])
    return make_toy_atom_(
        level_g=_numpy.array([1.0, 3.0, 5.0]),
        level_erg=_numpy.array([0.0, E1, E2]),
        line_pairs=[(0, 1), (1, 2), (0, 2)],
        line_Aji=Aji,
        line_Cij_coe=Cij_coe,
        am=am,
    )


_N_CONT_MESH: T_INT = 41


def make_toy_atom_2lv_cont_(
    w0_cm: T_FLOAT = 5000.0e-8,
    Aji: T_FLOAT = 1.0e8,
    Cij_line: T_FLOAT = 1.0e-8,
    CI_coe: T_ARRAY | None = None,  # (C_0k, C_1k)
    alpha0: T_FLOAT = 1.0e-18,
    E_ion_over_E1: T_FLOAT = 1.5,
    am: T_FLOAT = 1.0,
) -> Toy_Atom:
    """2 bound levels + 1 continuum level (the ion ground).

    The line (0,1) and the two b-f transitions (0,k), (1,k) are all active on
    the shared axis; the (1,k) continuum range contains the line, so this toy
    exercises the line-inside-continuum overlap. cross sections are
    hydrogenic-shaped, alpha = alpha0 * (wl/w0_threshold)^3.
    """
    E1 = CST.h_ * CST.c_ / w0_cm
    E_ion = E_ion_over_E1 * E1
    if CI_coe is None:
        CI_coe = _numpy.array([1.0e-9, 1.0e-9])

    atom = make_toy_atom_(
        level_g=_numpy.array([1.0, 3.0, 1.0]),
        level_erg=_numpy.array([0.0, E1, E_ion]),
        line_pairs=[(0, 1)],
        line_Aji=_numpy.array([Aji]),
        line_Cij_coe=_numpy.array([Cij_line]),
        am=am,
    )
    # the continuum level is the ground of the next ionization stage
    atom.Level["isGround"][2] = True

    nCont = 2
    Cont = _numpy.zeros(nCont, dtype=_CONT_DTYPE)
    Cont_mesh = _numpy.empty((nCont, _N_CONT_MESH), dtype=DT_NB_FLOAT)
    alpha = _numpy.empty((nCont, _N_CONT_MESH), dtype=DT_NB_FLOAT)
    for kC, i in enumerate((0, 1)):
        chi_ion = E_ion - float(atom.Level["erg"][i])
        w_threshold = CST.h_ * CST.c_ / chi_ion
        Cont["idxI"][kC] = i
        Cont["idxJ"][kC] = 2
        Cont["gi"][kC] = atom.Level["g"][i]
        Cont["gj"][kC] = atom.Level["g"][2]
        Cont["f0"][kC] = chi_ion / CST.h_
        Cont_mesh[kC, :] = w_threshold * _MeshUtil.make_continuum_mesh_(_N_CONT_MESH)
        alpha[kC, :] = alpha0 * (Cont_mesh[kC, :] / w_threshold) ** 3

    atom.Cont = Cont
    atom.Cont_mesh = Cont_mesh
    atom.alpha = alpha
    atom.Cij_coe = _numpy.concatenate([atom.Cij_coe, _numpy.asarray(CI_coe, dtype=DT_NB_FLOAT)])
    atom.nCont = nCont
    return atom


def make_toy_atom_overlap_(
    w0_cm: T_FLOAT = 5000.0e-8,
    split: T_FLOAT = 3.0e-5,
    Aji: T_ARRAY | None = None,  # (A10, A20)
    Cij_coe: T_ARRAY | None = None,  # (C01, C02, C0k, C1k, C2k)
    alpha0: T_FLOAT = 1.0e-18,
    E_ion_over_E1: T_FLOAT = 1.5,
    am: T_FLOAT = 1.0,
) -> Toy_Atom:
    """A resonance doublet under continua: lines (0,1) and (0,2) share their
    LOWER level and their upper levels are split by `split` (relative to E1),
    so the two windows overlap on the axis for the usual +-10 ruler units at
    2.5 km/s (8e-5); the ion level 3 carries three b-f transitions. this is
    the strong-overlap case the self-term operator is weakest on. (the shared
    level must be the ground: the production LTE builder chains populations
    from ground levels only.)
    """
    E1 = CST.h_ * CST.c_ / w0_cm
    E2 = E1 * (1.0 + split)
    E_ion = E_ion_over_E1 * E1
    if Aji is None:
        Aji = _numpy.array([1.0e8, 5.0e7])
    if Cij_coe is None:
        Cij_coe = _numpy.array([1.0e-8, 1.0e-8, 1.0e-9, 1.0e-9, 1.0e-9])

    atom = make_toy_atom_(
        level_g=_numpy.array([1.0, 3.0, 5.0, 1.0]),
        level_erg=_numpy.array([0.0, E1, E2, E_ion]),
        line_pairs=[(0, 1), (0, 2)],
        line_Aji=_numpy.asarray(Aji, dtype=DT_NB_FLOAT),
        line_Cij_coe=_numpy.asarray(Cij_coe[:2], dtype=DT_NB_FLOAT),
        am=am,
    )
    atom.Level["isGround"][3] = True

    nCont = 3
    Cont = _numpy.zeros(nCont, dtype=_CONT_DTYPE)
    Cont_mesh = _numpy.empty((nCont, _N_CONT_MESH), dtype=DT_NB_FLOAT)
    alpha = _numpy.empty((nCont, _N_CONT_MESH), dtype=DT_NB_FLOAT)
    for kC, i in enumerate((0, 1, 2)):
        chi_ion = E_ion - float(atom.Level["erg"][i])
        w_threshold = CST.h_ * CST.c_ / chi_ion
        Cont["idxI"][kC] = i
        Cont["idxJ"][kC] = 3
        Cont["gi"][kC] = atom.Level["g"][i]
        Cont["gj"][kC] = atom.Level["g"][3]
        Cont["f0"][kC] = chi_ion / CST.h_
        Cont_mesh[kC, :] = w_threshold * _MeshUtil.make_continuum_mesh_(_N_CONT_MESH)
        alpha[kC, :] = alpha0 * (Cont_mesh[kC, :] / w_threshold) ** 3

    atom.Cont = Cont
    atom.Cont_mesh = Cont_mesh
    atom.alpha = alpha
    atom.Cij_coe = _numpy.concatenate([atom.Cij_coe, _numpy.asarray(Cij_coe[2:], dtype=DT_NB_FLOAT)])
    atom.nCont = nCont
    return atom


def make_toy_atmos_(
    ND: T_INT,
    thickness_cm: T_FLOAT,
    Te_top: T_FLOAT = 6.0e3,
    Te_bottom: T_FLOAT | None = None,
    Ne: T_FLOAT = 1.0e10,
    Vt: T_FLOAT = 5.0e5,
    Nt: T_FLOAT = 1.0e10,
) -> Atmos1D:
    """Plane-parallel toy stratification; Te_bottom=None gives a uniform slab."""
    Z = _numpy.linspace(0.0, thickness_cm, ND)
    if Te_bottom is None:
        Te = _numpy.full(ND, Te_top, dtype=DT_NB_FLOAT)
    else:
        Te = Te_top + (Te_bottom - Te_top) * Z / thickness_cm
    return Atmos1D(
        Z=Z,
        Te=Te,
        Ne=_numpy.full(ND, Ne, dtype=DT_NB_FLOAT),
        Vt=_numpy.full(ND, Vt, dtype=DT_NB_FLOAT),
        Nt=_numpy.full(ND, Nt, dtype=DT_NB_FLOAT),
        ND=ND,
    )
