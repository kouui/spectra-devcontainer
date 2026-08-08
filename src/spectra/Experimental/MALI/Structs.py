# -------------------------------------------------------------------------------
# containers and toy builders for the MALI prototype
#
# build-once tier: everything here runs a single time per model configuration,
# so it stays interpreted and is free to use dataclasses and python loops. the
# per-iteration tier (ProfileTable.py, Loop.py kernels) receives only the plain
# arrays stored here.
#
# the toy atoms are fabricated instead of loaded from data/atom/** : every
# oracle in the tests is analytic or structural, and real atomic data would add
# failure modes (IO, continua, damping) without adding verification power.
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

import numpy as _numpy

from ...Atomic import BasicP as _BasicP
from ...Atomic import Collision as _Collision
from ...Atomic import LTELib as _LTELib
from ...Function.SEquil import SELib as _SELib
from ...ImportAll import *
from ...Util import MeshUtil as _MeshUtil
from . import GlobalMesh as _GlobalMesh
from . import ProfileTable as _ProfileTable

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


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class Toy_Atom:
    Level: T_ARRAY  # (nLevel,), _LEVEL_DTYPE
    Line: T_ARRAY  # (nLine,), _LINE_DTYPE
    Cont: T_ARRAY  # (nCont,), _CONT_DTYPE; empty for line-only toys
    # fabricated collisional coefficients, transition-ordered (lines then
    # continua), multiply by Ne for the rate. SELib._get_Cij_ is NOT reused
    # here: it rejects atoms without continuum, and the toys must stay free to
    # be line-only. detailed balance still uses the real Collision.Cij_to_Cji_.
    Cij_coe: T_ARRAY  # (nTran,), [cm^3 s^-1]
    # passive-continuum machinery (empty for line-only toys): the b-f
    # transitions contribute RATES to the SE (from prescribed radiation), never
    # opacity to the RT -- RH's PASSIVE transitions.
    Cont_mesh: T_ARRAY  # (nCont, nContMesh), [cm], descending from the threshold
    alpha: T_ARRAY  # (nCont, nContMesh), photoionization cross section, [cm^2]
    am: T_FLOAT  # atomic mass relative to hydrogen
    nLevel: T_INT
    nLine: T_INT
    nCont: T_INT


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

    The line (0,1) is the active transition; the two b-f transitions (0,k) and
    (1,k) are passive: their rates come from a prescribed radiation field via
    SELib._bf_R_rate_ and enter Gamma unpreconditioned -- the active/passive
    seam of full MALI, at toy scale. cross sections are hydrogenic-shaped,
    alpha = alpha0 * (wl/w0_threshold)^3.
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


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class Atmos1D:
    Z: T_ARRAY  # (ND,), depth below the upper surface, [cm], Z[0] = 0, ascending
    Te: T_ARRAY  # (ND,), [K]
    Ne: T_ARRAY  # (ND,), [cm^-3]
    Vt: T_ARRAY  # (ND,), [cm/s]
    Nt: T_ARRAY  # (ND,), total number density of the toy species, [cm^-3]
    ND: T_INT


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


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class MALI_Precompute:
    """Loop-invariant tier: everything here depends only on (Te, Ne, Vt) per
    depth -- never on populations -- so it is computed once and read by every
    MALI iteration. valid while Ne is fixed (no charge-conservation iteration).
    """

    n_LTE: T_ARRAY  # (ND, nLevel), normalized LTE populations
    nj_by_ni: T_ARRAY  # (ND, nTran), LTE ratio per transition (lines then continua)
    Cji_coe: T_ARRAY  # (ND, nTran), downward collisional coefficient [cm^3 s^-1]
    dopWidth_cm: T_ARRAY  # (ND, nLine), [cm]
    adamp: T_ARRAY  # (ND, nLine), Voigt damping parameter (0 for Gaussian toys)
    planck_w0: T_ARRAY  # (ND, nLine), Planck function at line center, cm-base
    # profile tables, flattened over the per-line windows of the global mesh:
    # rows win_off[kL,0]:win_off[kL,1] of `phi` belong to line kL and align with
    # global wavelength indices Nblue[kL] : Nblue[kL]+span[kL].
    phi: T_ARRAY  # (nWin_total, ND), [cm^-1]
    wphi: T_ARRAY  # (ND, nLine), numerical profile norm on the window quadrature
    weight: T_ARRAY  # (nWin_total,), per-window trapezoidal weights, [cm]
    win_off: T_ARRAY  # (nLine, 2), [start, stop) row range of each line in `phi`
    # passive b-f rates from PRESCRIBED radiation (Planck at the local Te here),
    # therefore loop-invariant; (ND, 0) for line-only toys
    Rik: T_ARRAY  # (ND, nCont), radiative ionization rate
    Rki_stim: T_ARRAY  # (ND, nCont), stimulated radiative recombination rate
    Rki_spon: T_ARRAY  # (ND, nCont), spontaneous radiative recombination rate


def precompute_(
    atom: Toy_Atom,
    atmos: Atmos1D,
    mesh: _GlobalMesh.Global_Mesh,
    adamp_const: T_FLOAT = 0.0,
) -> MALI_Precompute:
    ND = atmos.ND
    nLine = atom.nLine
    nCont = atom.nCont
    nTran = nLine + nCont

    n_LTE = _numpy.empty((ND, atom.nLevel), dtype=DT_NB_FLOAT)
    nj_by_ni = _numpy.empty((ND, nTran), dtype=DT_NB_FLOAT)
    Cji_coe = _numpy.empty((ND, nTran), dtype=DT_NB_FLOAT)
    dopWidth_cm = _numpy.empty((ND, nLine), dtype=DT_NB_FLOAT)
    planck_w0 = _numpy.empty((ND, nLine), dtype=DT_NB_FLOAT)
    Rik = _numpy.zeros((ND, nCont), dtype=DT_NB_FLOAT)
    Rki_stim = _numpy.zeros((ND, nCont), dtype=DT_NB_FLOAT)
    Rki_spon = _numpy.zeros((ND, nCont), dtype=DT_NB_FLOAT)
    for k in range(ND):
        ni, ratio = _SELib._ni_nj_LTE_(atom.Level, atom.Line, atom.Cont, atmos.Te[k], atmos.Ne[k])
        n_LTE[k, :] = ni
        nj_by_ni[k, :] = ratio
        Cji_coe[k, :] = _Collision.Cij_to_Cji_(atom.Cij_coe[:], ratio[:])
        dopWidth_cm[k, :] = _BasicP.doppler_width_(atom.Line["w0"][:], atmos.Te[k], atmos.Vt[k], atom.am)
        planck_w0[k, :] = _LTELib.planck_cm_(atom.Line["w0"][:], atmos.Te[k])
        if nCont > 0:
            # prescribed thermal radiation drives the passive b-f transitions
            PI_intensity = _LTELib.planck_cm_(atom.Cont_mesh[:, :], atmos.Te[k])
            Rik[k, :], Rki_stim[k, :], Rki_spon[k, :] = _SELib._bf_R_rate_(
                atom.Cont, atom.Cont_mesh, atmos.Te[k], nj_by_ni[k, nLine:], atom.alpha, PI_intensity
            )

    adamp = _numpy.full((ND, nLine), adamp_const, dtype=DT_NB_FLOAT)

    win_off = _numpy.empty((nLine, 2), dtype=DT_NB_INT)
    stop = 0
    for kL in range(nLine):
        win_off[kL, 0] = stop
        stop += int(mesh.span[kL])
        win_off[kL, 1] = stop

    phi = _numpy.empty((stop, ND), dtype=DT_NB_FLOAT)
    weight = _numpy.empty(stop, dtype=DT_NB_FLOAT)
    wphi = _numpy.empty((ND, nLine), dtype=DT_NB_FLOAT)
    for kL in range(nLine):
        wl_win = mesh.wl[mesh.Nblue[kL] : mesh.Nblue[kL] + mesh.span[kL]]
        weight_win = _GlobalMesh.trapezoidal_weight_(wl_win)
        phi_win, wphi_line = _ProfileTable.line_profile_table_(
            wl_win,
            weight_win,
            atom.Line["w0"][kL],
            dopWidth_cm[:, kL].copy(),
            adamp[:, kL].copy(),
            atom.Line["ProfileType"][kL],
        )
        phi[win_off[kL, 0] : win_off[kL, 1], :] = phi_win
        weight[win_off[kL, 0] : win_off[kL, 1]] = weight_win
        wphi[:, kL] = wphi_line

    return MALI_Precompute(
        n_LTE=n_LTE,
        nj_by_ni=nj_by_ni,
        Cji_coe=Cji_coe,
        dopWidth_cm=dopWidth_cm,
        adamp=adamp,
        planck_w0=planck_w0,
        phi=phi,
        wphi=wphi,
        weight=weight,
        win_off=win_off,
        Rik=Rik,
        Rki_stim=Rki_stim,
        Rki_spon=Rki_spon,
    )
