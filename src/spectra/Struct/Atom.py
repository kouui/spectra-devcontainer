# -------------------------------------------------------------------------------
# definition of struct for Atom
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
import re as _re

import numpy as _numpy

from ..ImportAll import *
from ..Util import ElementUtil as _ElementUtil
from ..Util.AtomUtils import AtomIO as _AtomIO
from ..Util.AtomUtils import RHAtomIO as _RHAtomIO
from . import WavelengthMesh as _WavelengthMesh

# -------------------------------------------------------------------------------
# struct
# -------------------------------------------------------------------------------


@_dataclass(**STRUCT_KWGS)
class Radiative_Line:
    nRadiativeLine: T_INT
    Coe: T_ARRAY  # struct array


@_dataclass(**STRUCT_KWGS)
class Collisional_Transition:
    _transition_type: T_E_COLLISIONAL_TRANSITION
    _transition_source: T_E_COLLISIONAL_TRANSITION_SOURCE
    _transition_formula: T_E_COLLISIONAL_TRANSITION_FORMULA

    Te_table: T_ARRAY  # 1d
    Omega_table: T_ARRAY  # 2d
    Coe: T_ARRAY  # struct array


@_dataclass(**STRUCT_KWGS)
class Photo_Ionization:
    alpha_table: T_ARRAY  # 2d,  (2, ?)
    alpha_table_idxs: T_ARRAY  # 2d,  (nCont, 2)
    Coe: T_ARRAY  # struct

    # intensity   : T_ARRAY # 2d   (nCont, nContMesh)

    alpha_interp: T_ARRAY  # 2d   (nCont, nContMesh)
    # interpolated photoionization cross section
    # no matter there is doppler shift, alpha_interp always
    # starts from the cross section at ionization limit


@_dataclass(**STRUCT_KWGS)
class ATOMIC_DATA_SOURCE:
    AJI: T_E_ATOMIC_DATA_SOURCE
    CE: T_E_ATOMIC_DATA_SOURCE
    CI: T_E_ATOMIC_DATA_SOURCE
    PI: T_E_ATOMIC_DATA_SOURCE


@_dataclass(**STRUCT_KWGS)
class CTJ_Table:
    Level: T_CTJ_TABLE
    Line: T_CTJ_PAIR_TABLE
    Cont: T_CTJ_PAIR_TABLE


@_dataclass(**STRUCT_KWGS)
class Index_Table:
    Line: T_IDX_PAIR_TABLE
    Cont: T_IDX_PAIR_TABLE


@_dataclass(**STRUCT_KWGS)
class RH_Data:
    """Values of an RH atom file that the `Atom` fields cannot hold.

    Everything is kept as written in the file (cm^-1, m^2, nm, raw ``Nlambda``)
    so that a later stage decides its own discretization and constants; the one
    exception is the collision coefficients, see below. Rows and indices are in
    file order: level indices are positions in the level list, and
    ``line_*``/``cont_*`` follow ``Atom.Line``/``Atom.Cont`` one to one.

    The raw cm^-1 energies are kept next to ``Atom.Level["erg"]`` on purpose:
    the broadening assembly uses RH's own constants with them, and the two
    energy scales differ by up to 1e-6 (relative).
    """

    # --- levels
    level_E_cm: T_ARRAY  # (nLevel,) [cm^-1]
    level_label: T_TUPLE[T_STR, ...]
    level_no: T_ARRAY  # (nLevel,) RH levelNo column; not the principal quantum number

    # --- lines (one per Atom.Line)
    line_Grad: T_ARRAY  # (nLine,) radiative broadening [s^-1]
    line_vdW_type: T_TUPLE[T_STR, ...]
    line_cvdW: T_ARRAY  # (nLine, 4)
    line_cStark: T_ARRAY  # (nLine,)
    line_Nlambda: T_ARRAY  # (nLine,) raw: half profile for SYMM, full profile for ASYMM
    line_symm: T_TUPLE[T_STR, ...]  # "SYMM" | "ASYMM"
    line_qcore: T_ARRAY  # (nLine,)
    line_qwing: T_ARRAY  # (nLine,)

    # --- continua (one per Atom.Cont)
    cont_alpha0_m2: T_ARRAY  # (nCont,) edge cross section [m^2]
    cont_lamb_min_nm: T_ARRAY  # (nCont,) [nm]
    cont_Nlambda: T_ARRAY  # (nCont,) raw
    cont_wavedep: T_TUPLE[T_STR, ...]  # "HYDROGENIC"

    # --- collisions (RH table format, evaluated by the solver, not by SELib)
    # The coefficients are converted once, SI -> CGS (x1e6, m^3 -> cm^3), so an
    # evaluator must not multiply again.
    # CE rows list the UPPER level first, CI rows the LOWER level first, as in
    # the file; the two conventions are easy to mix up.
    Te_table: T_ARRAY  # (nTemp,) [K]
    CE_idx: T_ARRAY  # (nCE, 2) (upper, lower)
    CE_coe_cgs: T_ARRAY  # (nCE, nTemp)
    CI_idx: T_ARRAY  # (nCI, 2) (lower, upper)
    CI_coe_cgs: T_ARRAY  # (nCI, nTemp)


@_dataclass(**STRUCT_KWGS)
class Atom:
    # Title : T_STR
    # Element : T_STR

    Z: T_INT
    Mass: T_FLOAT
    Abun: T_FLOAT

    nLevel: T_INT  # total levels, bound + continuum
    nBoundLevel: T_INT  # bound levels only (excludes the continuum level)
    nLine: T_INT
    nCont: T_INT
    nTran: T_INT
    nRL: T_INT

    Level: T_ARRAY  # struct array
    Line: T_ARRAY  # struct array
    Cont: T_ARRAY  # struct array

    _has_continuum: T_BOOL

    _atomic_data_source: ATOMIC_DATA_SOURCE
    _atom_type: T_E_ATOM

    _ctj_table: CTJ_Table
    _idx_table: Index_Table

    CE: Collisional_Transition
    CI: Collisional_Transition
    PI: Photo_Ionization
    RL: Radiative_Line

    _wave_mesh: _WavelengthMesh.Wavelength_Mesh

    # None for atoms read from the conf/Level/Aji files. An atom with RH data
    # carries placeholders in CE/CI and must not enter the SE solver of
    # `Function.SEquil.SELib`, which would compute collisions from formulas.
    RH: RH_Data | None = None


# -------------------------------------------------------------------------------
# init function
# -------------------------------------------------------------------------------


def init_Atom_(conf_path: T_STR, is_hydrogen: T_BOOL = False) -> Atom:
    """given the *.conf file, create the Atom struct

    Parameters
    ----------
    conf_path : T_STR
        path to the *.conf file
    is_hydrogen : T_BOOL, optional
        whether is Hydrogen atom, by default False

    Returns
    -------
    atom : Atom
        the Atom struct (the wavelength mesh is stored on ``atom._wave_mesh``)

    Notes
    -----
    If the data-file paths are needed, obtain them separately via
    ``_AtomIO.read_conf_(conf_path)``.
    """

    # path dict
    # --------------------
    path_dict = _AtomIO.read_conf_(conf_path)
    _atom_type: T_E_ATOM
    if is_hydrogen:
        _atom_type = E_ATOM.HYDROGEN
    else:
        _atom_type = E_ATOM.NORMAL

    # read Level
    # --------------------
    if path_dict["Level"] is None:
        raise ValueError("Lack of .Level file")

    Z, Mass, Abun, nLevel, Level, Level_info_table = _AtomIO.make_Atom_Level_(path_dict["Level"])
    # nTran nLine nCont
    # --------------------
    nLine, nCont, nTran, _has_continuum = _AtomIO.nLine_nCont_nTran_(Level["stage"])
    nBoundLevel = nLevel - 1 if _has_continuum else nLevel
    # if not _has_continuum:
    #    raise ValueError("Currently we don't support Atomic Model without comtinuum.")

    # ctj and idx table
    # --------------------
    Line_idx_table, Line_ctj_table, Cont_idx_table, Cont_ctj_table = _AtomIO.prepare_idx_ctj_mapping_(
        Level_info_table, Level["stage"], Level["isGround"], nLine, nCont
    )

    _ctj_table = CTJ_Table(Level=Level_info_table, Line=Line_ctj_table, Cont=Cont_ctj_table)
    _idx_table = Index_Table(Line=Line_idx_table, Cont=Cont_idx_table)

    # make Cont
    # --------------------
    Cont = _AtomIO.make_Atom_Cont_(nCont, Cont_idx_table, Level)

    # read Aji
    # --------------------
    Line, data_source_Aji = _AtomIO.make_Atom_Line_(path_dict["Aji"], Level, Line_idx_table, Line_ctj_table, _atom_type)
    # read CE
    # --------------------
    Te_table, Omega_table, Coe, _transition_type, _transition_source, _transition_formula = _AtomIO.make_Atom_CECI_(
        path_dict["CEe"], "CE", nLine, Line, Level, Level_info_table, Line_ctj_table
    )
    CE = Collisional_Transition(
        _transition_type=_transition_type,
        _transition_source=_transition_source,
        _transition_formula=_transition_formula,
        Te_table=Te_table,
        Omega_table=Omega_table,
        Coe=Coe,
    )
    data_source_CE: T_E_ATOMIC_DATA_SOURCE
    if Te_table.size == 0:
        data_source_CE = E_ATOMIC_DATA_SOURCE.CALCULATE
    else:
        data_source_CE = E_ATOMIC_DATA_SOURCE.EXPERIMENT

    del Te_table, Omega_table, Coe, _transition_type, _transition_source, _transition_formula
    # read CI
    # --------------------
    Te_table, Omega_table, Coe, _transition_type, _transition_source, _transition_formula = _AtomIO.make_Atom_CECI_(
        path_dict["CIe"], "CI", nCont, Cont, Level, Level_info_table, Cont_ctj_table
    )
    CI = Collisional_Transition(
        _transition_type=_transition_type,
        _transition_source=_transition_source,
        _transition_formula=_transition_formula,
        Te_table=Te_table,
        Omega_table=Omega_table,
        Coe=Coe,
    )
    data_source_CI: T_E_ATOMIC_DATA_SOURCE
    if Te_table.size == 0:
        data_source_CI = E_ATOMIC_DATA_SOURCE.CALCULATE
    else:
        data_source_CI = E_ATOMIC_DATA_SOURCE.EXPERIMENT

    del Te_table, Omega_table, Coe, _transition_type, _transition_source, _transition_formula

    # read radiative line
    # --------------------
    Coe, nRadiativeLine = _AtomIO.make_Atom_RL_(path_dict["RadiativeLine"], Level_info_table, Line_ctj_table)
    RL = Radiative_Line(nRadiativeLine=nRadiativeLine, Coe=Coe)
    nRL = nRadiativeLine
    del Coe, nRadiativeLine

    # make mesh
    # --------------------
    waveMesh = _WavelengthMesh.init_Wave_Mesh_(Cont, Line, RL.Coe)

    # read PI
    # --------------------
    Cont_mesh: T_ARRAY = waveMesh.Cont_mesh
    alpha_table, alpha_table_idxs, Coe, alpha_interp, data_source_PI = _AtomIO.make_Atom_PI_(
        path_dict["PI"], Level, Cont, Cont_mesh, _atom_type, Level_info_table, Cont_ctj_table
    )
    PI = Photo_Ionization(
        alpha_table=alpha_table, alpha_table_idxs=alpha_table_idxs, Coe=Coe, alpha_interp=alpha_interp
    )
    del alpha_table, alpha_table_idxs, Coe, alpha_interp

    # make ATOMIC_DATA_SOURCE
    _atomic_data_source = ATOMIC_DATA_SOURCE(
        AJI=data_source_Aji, CE=data_source_CE, CI=data_source_CI, PI=data_source_PI
    )

    atom = Atom(
        Z=Z,
        Mass=Mass,
        Abun=Abun,
        nLevel=nLevel,
        nBoundLevel=nBoundLevel,
        nLine=nLine,
        nCont=nCont,
        nTran=nTran,
        nRL=nRL,
        Level=Level,
        Line=Line,
        Cont=Cont,
        _has_continuum=_has_continuum,
        _atomic_data_source=_atomic_data_source,
        _atom_type=_atom_type,
        _ctj_table=_ctj_table,
        _idx_table=_idx_table,
        CE=CE,
        CI=CI,
        PI=PI,
        RL=RL,
        _wave_mesh=waveMesh,
    )
    return atom


# Conversion constants of the RH atom format, shared by the Aji and the
# collision handling below.
# RH's f -> Aji factor (2 pi e^2 / (eps0 m_e c), SI [m^2 s^-1]); it is the
# constant the RH atom files assume, kept as RH has it instead of rebuilding it
# from spectra.Constants so that Aji agrees with RH's to the last digit.
_RH_F_TO_AJI_SI = 6.670192e-5
_CM_PER_M_CUBED = 1.0e6  # m^3 -> cm^3, applied once to the CE/CI coefficients


def _principal_n_from_label_(label: T_STR) -> T_INT:
    """Principal quantum number from an RH level label, e.g. ``'H I 1S 2SE'`` -> 1.

    The configuration is the third word and starts with n. RH's ``levelNo`` is
    only a running index.
    """
    words = label.split()
    m = _re.match(r"\d+", words[2]) if len(words) > 2 else None
    if m is None:
        raise ValueError(f"cannot read the principal quantum number from RH level label {label!r}")
    return int(m.group())


def _full_profile_points_(Nlambda: T_INT, symm: T_STR) -> T_INT:
    """Points of the full (two-sided) line profile mesh for an RH ``Nlambda``.

    For SYMM RH counts one half of the profile; ASYMM counts the whole. The
    mesh builder needs an odd count, so an even one gets a point added.
    """
    n = Nlambda if symm == "ASYMM" else 2 * Nlambda - 1
    return n + 1 if n % 2 == 0 else n


def _rh_make_Level_(rh: _RHAtomIO.RH_Atom_File, Z: T_INT) -> T_TUPLE[T_ARRAY, T_CTJ_TABLE]:
    """Level struct array and the (label, "-", "-") info table of an RH atom.

    A level of stage Z + 1 (1-based) is the bare nucleus and gets ``n = 0``, as
    in `init_theoretical_hydrogen_atom_`: it has no principal quantum number,
    and the RH label of such a level (``'H I 6H 2HO'`` for the proton) is only
    a name. Every other level takes n from its label.
    """
    nLevel = rh.level_E_cm.size
    stage = rh.level_stage + 1
    if _numpy.any(_numpy.diff(stage) < 0):
        raise ValueError("RH atom file: level stages must not decrease along the level list")
    g_int = _numpy.rint(rh.level_g).astype(DT_NB_INT)
    if not _numpy.array_equal(g_int, rh.level_g):
        raise ValueError("RH atom file: statistical weights must be integers")

    Level = _numpy.zeros(nLevel, dtype=_AtomIO.LEVEL_DTYPE)
    Level["erg"][:] = rh.level_E_cm * CST.h_ * CST.c_
    Level["g"][:] = g_int
    Level["stage"][:] = stage
    Level["n"][:] = [
        0 if st == Z + 1 else _principal_n_from_label_(lab) for st, lab in zip(stage, rh.level_label, strict=True)
    ]
    for s in _numpy.unique(stage):
        members = _numpy.flatnonzero(stage == s)
        Level["isGround"][members[_numpy.argmin(Level["erg"][members])]] = True

    # one tuple per level, as the (configuration, term, J) of the other readers;
    # an RH label is not split, so the whole label is the configuration
    Level_info_table: T_CTJ_TABLE = tuple((lab, "-", "-") for lab in rh.level_label)
    return Level, Level_info_table


def _rh_make_Line_(rh: _RHAtomIO.RH_Atom_File, Level: T_ARRAY) -> T_TUPLE[T_ARRAY, T_IDX_PAIR_TABLE]:
    """Line struct array of the lines RH lists, in file order, and their (lower, upper) pairs."""
    nLine = rh.line_i.size
    idx_table: T_IDX_PAIR_TABLE = tuple(zip(rh.line_i.tolist(), rh.line_j.tolist(), strict=True))
    for i, j in idx_table:
        if Level["stage"][i] != Level["stage"][j] or not rh.level_E_cm[j] > rh.level_E_cm[i]:
            raise ValueError(f"RH atom file: line ({j}, {i}) is not an upward transition within one stage")
    if len(set(idx_table)) != nLine:
        raise ValueError("RH atom file: a line is listed twice")

    Line = _numpy.zeros(nLine, dtype=_AtomIO.LINE_DTYPE)
    Line["idxI"][:] = rh.line_i
    Line["idxJ"][:] = rh.line_j
    for k, (i, j) in enumerate(idx_table):
        wl_cm = 1.0 / (rh.level_E_cm[j] - rh.level_E_cm[i])
        Line["AJI"][k] = rh.line_f[k] * (_RH_F_TO_AJI_SI / (wl_cm * 1.0e-2) ** 2) * (rh.level_g[i] / rh.level_g[j])
    _AtomIO.finalize_Line_(Line, Level)
    return Line, idx_table


def _rh_make_Cont_(rh: _RHAtomIO.RH_Atom_File, Level: T_ARRAY) -> T_TUPLE[T_ARRAY, T_IDX_PAIR_TABLE]:
    """Cont struct array of the continua RH lists, in file order, and their (lower, upper) pairs."""
    idx_table: T_IDX_PAIR_TABLE = tuple(zip(rh.cont_i.tolist(), rh.cont_j.tolist(), strict=True))
    for i, j in idx_table:
        if Level["stage"][j] != Level["stage"][i] + 1 or not Level["isGround"][j]:
            raise ValueError(f"RH atom file: continuum ({j}, {i}) must end on the ground level of the next stage")
    return _AtomIO.make_Atom_Cont_(len(idx_table), idx_table, Level), idx_table


def _rh_check_collision_rows_(
    rh: _RHAtomIO.RH_Atom_File, Line_idx_table: T_IDX_PAIR_TABLE, Cont_idx_table: T_IDX_PAIR_TABLE
) -> None:
    """Every CE row must name a listed line and every CI row a listed continuum."""
    line_set, cont_set = set(Line_idx_table), set(Cont_idx_table)
    for upper, lower in rh.CE_idx.tolist():
        if (lower, upper) not in line_set:
            raise ValueError(f"RH atom file: CE row ({upper}, {lower}) matches no listed line")
    for lower, upper in rh.CI_idx.tolist():
        if (lower, upper) not in cont_set:
            raise ValueError(f"RH atom file: CI row ({lower}, {upper}) matches no listed continuum")


def _rh_placeholder_collisions_(
    tran_type: T_STR, Tran: T_ARRAY, Level: T_ARRAY, Level_info_table: T_CTJ_TABLE, ctj: T_CTJ_PAIR_TABLE
) -> Collisional_Transition:
    Te_table, Omega_table, Coe, _tt, _ts, _tf = _AtomIO.make_Atom_CECI_(
        None, tran_type, Tran.shape[0], Tran, Level, Level_info_table, ctj
    )
    return Collisional_Transition(
        _transition_type=_tt,
        _transition_source=_ts,
        _transition_formula=_tf,
        Te_table=Te_table,
        Omega_table=Omega_table,
        Coe=Coe,
    )


def _rh_make_RL_Coe_(rh: _RHAtomIO.RH_Atom_File, Line: T_ARRAY) -> T_ARRAY:
    nLine = Line.shape[0]
    Coe = _numpy.zeros(nLine, dtype=_AtomIO.RL_DTYPE)
    Coe["idxI"][:] = Line["idxI"]
    Coe["idxJ"][:] = Line["idxJ"]
    Coe["lineIndex"][:] = _numpy.arange(nLine)
    Coe["ProfileType"][:] = int(E_ABSORPTION_PROFILE_TYPE.VOIGT)  # the reader accepts VOIGT only
    Coe["qcore"][:] = rh.line_qcore
    Coe["qwing"][:] = rh.line_qwing
    Coe["nLambda"][:] = [_full_profile_points_(int(n), s) for n, s in zip(rh.line_Nlambda, rh.line_symm, strict=True)]
    return Coe


def _rh_make_RH_Data_(rh: _RHAtomIO.RH_Atom_File) -> RH_Data:
    return RH_Data(
        level_E_cm=rh.level_E_cm,
        level_label=rh.level_label,
        level_no=rh.level_no,
        line_Grad=rh.line_Grad,
        line_vdW_type=rh.line_vdW_type,
        line_cvdW=rh.line_cvdW,
        line_cStark=rh.line_cStark,
        line_Nlambda=rh.line_Nlambda,
        line_symm=rh.line_symm,
        line_qcore=rh.line_qcore,
        line_qwing=rh.line_qwing,
        cont_alpha0_m2=rh.cont_alpha0_m2,
        cont_lamb_min_nm=rh.cont_lamb_min_nm,
        cont_Nlambda=rh.cont_Nlambda,
        cont_wavedep=rh.cont_wavedep,
        Te_table=rh.Te_table,
        CE_idx=rh.CE_idx,
        CE_coe_cgs=rh.CE_coe * _CM_PER_M_CUBED,
        CI_idx=rh.CI_idx,
        CI_coe_cgs=rh.CI_coe * _CM_PER_M_CUBED,
    )


def init_Atom_from_RH_(path: T_STR | _Path) -> Atom:
    """Create the Atom struct from an RH ``*.atom`` file.

    Parameters
    ----------
    path : T_STR | Path
        path to the RH atom file; see `RHAtomIO` for the supported subset

    Returns
    -------
    atom : Atom
        ``atom.RH`` holds the raw file values (see `RH_Data`).

    Notes
    -----
    - Stage = RH stage + 1, ``isGround`` is the lowest level of each stage.
      Stages must not decrease along the file, so the levels of one stage
      are contiguous.
    - Only the lines and continua the file lists exist, in file order (lines
      first in the transition order).
    - ``Level["erg"]`` is ``E[cm^-1] * h_ * c_`` with project constants.
    - ``Level["n"]`` is read from the label; a bare nucleus (stage Z + 1) has 0.
    - ``RL.Coe`` carries the number of points of the full profile mesh
      (see `_full_profile_points_`), ``qcore`` and ``qwing``, so that
      `init_Wave_Mesh_` builds ``Line_Coe``/``Line_mesh`` from them.
    - ``PI`` holds the project's hydrogenic cross section on the default
      continuum mesh (edge-concentrated, ``w0`` times `make_continuum_mesh_`),
      exactly as for a "Hydrogenic" entry of a PI file. It is not RH's
      cross section: the edge value differs from ``alpha0`` by up to 3 %
      (n = 1), the mesh does not start at ``lamb_min``, and the Gaunt factor
      differs. A solver that must reproduce RH builds its cross sections from
      ``atom.RH`` on its own mesh and does not read ``atom.PI``.
    - ``CE``/``CI`` are empty placeholders and their data source says
      CALCULATE. The collisions are in ``atom.RH``; `SELib.cal_SE_` refuses
      an atom with ``RH`` set, so the placeholders cannot be used by mistake.
    """
    rh = _RHAtomIO.read_RH_atom_(path)

    sym = _ElementUtil.format_sym_(rh.element)
    Z: T_INT = _ElementUtil.sym_to_z_(sym)

    Level, Level_info_table = _rh_make_Level_(rh, Z)
    nLevel = Level.shape[0]
    Line, Line_idx_table = _rh_make_Line_(rh, Level)
    Cont, Cont_idx_table = _rh_make_Cont_(rh, Level)
    _rh_check_collision_rows_(rh, Line_idx_table, Cont_idx_table)
    nLine, nCont = Line.shape[0], Cont.shape[0]
    _has_continuum = nCont > 0

    Line_ctj_table: T_CTJ_PAIR_TABLE = tuple((Level_info_table[i], Level_info_table[j]) for i, j in Line_idx_table)
    Cont_ctj_table: T_CTJ_PAIR_TABLE = tuple((Level_info_table[i], Level_info_table[j]) for i, j in Cont_idx_table)
    _ctj_table = CTJ_Table(Level=Level_info_table, Line=Line_ctj_table, Cont=Cont_ctj_table)
    _idx_table = Index_Table(Line=Line_idx_table, Cont=Cont_idx_table)

    CE = _rh_placeholder_collisions_("CE", Line, Level, Level_info_table, Line_ctj_table)
    CI = _rh_placeholder_collisions_("CI", Cont, Level, Level_info_table, Cont_ctj_table)

    RL = Radiative_Line(nRadiativeLine=nLine, Coe=_rh_make_RL_Coe_(rh, Line))
    waveMesh = _WavelengthMesh.init_Wave_Mesh_(Cont, Line, RL.Coe)

    alpha_table, alpha_table_idxs, PI_Coe, alpha_interp = _AtomIO.make_Atom_PI_hydrogenic_(
        Level, Cont, waveMesh.Cont_mesh, rh.cont_alpha0_m2 * 1.0e4
    )
    PI = Photo_Ionization(
        alpha_table=alpha_table, alpha_table_idxs=alpha_table_idxs, Coe=PI_Coe, alpha_interp=alpha_interp
    )

    # CALCULATE for CE/CI: no table is stored in CE/CI (see Notes)
    _atomic_data_source = ATOMIC_DATA_SOURCE(
        AJI=E_ATOMIC_DATA_SOURCE.EXPERIMENT,
        CE=E_ATOMIC_DATA_SOURCE.CALCULATE,
        CI=E_ATOMIC_DATA_SOURCE.CALCULATE,
        PI=E_ATOMIC_DATA_SOURCE.EXPERIMENT,
    )

    return Atom(
        Z=Z,
        Mass=_ElementUtil.sym_to_mass_(sym),
        Abun=_ElementUtil.sym_to_abun_(sym),
        nLevel=nLevel,
        nBoundLevel=nLevel - 1 if _has_continuum else nLevel,
        nLine=nLine,
        nCont=nCont,
        nTran=nLine + nCont,
        nRL=nLine,
        Level=Level,
        Line=Line,
        Cont=Cont,
        _has_continuum=_has_continuum,
        _atomic_data_source=_atomic_data_source,
        _atom_type=E_ATOM.HYDROGEN if Z == 1 else E_ATOM.NORMAL,
        _ctj_table=_ctj_table,
        _idx_table=_idx_table,
        CE=CE,
        CI=CI,
        PI=PI,
        RL=RL,
        _wave_mesh=waveMesh,
        RH=_rh_make_RH_Data_(rh),
    )


def init_theoretical_hydrogen_atom_(
    nLevel: T_INT = 8,
) -> Atom:
    """Create an `Atom` struct for hydrogen from purely theoretical atomic
    data. No file I/O, no conf parsing, no delegation to `init_Atom_`.

    The Level numpy array and CTJ table are constructed directly from the
    hydrogenic energy formula, then the remaining atomic data (Aji, CE,
    CI, PI, RL) are obtained by calling the existing `AtomIO.make_Atom_*_`
    helpers with `path=None`, which already route hydrogen to the analytic
    formulas in `spectra.Atomic.Hydrogen` and
    `spectra.Function.Hydrogen.DegenerateN`.

    Parameters
    ----------
    nLevel : T_INT, optional
        Total number of levels including the continuum. Must be >= 3.
        Default is 8 (n = 1..7 bound + continuum).

    Returns
    -------
    atom : Atom
        the Atom struct (the wavelength mesh is stored on ``atom._wave_mesh``)
    """
    import numpy as _numpy

    from .. import Constants as _CST
    from ..Util import ElementUtil as _ElementUtil

    if nLevel < 3:
        raise ValueError("`nLevel` must be >= 3 (n=1 bound + >=1 excited + continuum)")

    Z: T_INT = 1
    element: T_STR = "H"
    Mass: T_FLOAT = _ElementUtil.sym_to_mass_(element)
    Abun: T_FLOAT = _ElementUtil.sym_to_abun_(element)

    # build Level numpy array directly (mirrors the Level struct built by
    # AtomIO.make_Atom_Level_, but from the analytic hydrogenic formula)
    _level_dtype = _numpy.dtype(
        [
            ("erg", T_FLOAT),
            ("g", T_INT),
            ("stage", T_INT),
            ("gamma", T_FLOAT),
            ("isGround", T_BOOL),
            ("n", T_INT),
        ]
    )
    Level = _numpy.zeros(nLevel, dtype=_level_dtype)
    _Level_info_list: T_LIST[T_TUPLE[T_STR, T_STR, T_STR]] = []

    Ry = _CST.E_Rydberg_H_  # Rydberg energy unit [erg], proton-mass corrected

    # n=1 ground state (1s 2S 1/2), stage=1
    Level[0]["erg"] = 0.0
    Level[0]["g"] = 2
    Level[0]["stage"] = 1
    Level[0]["n"] = 1
    Level[0]["isGround"] = 1
    _Level_info_list.append(("1s", "2S", "1/2"))

    # n=2..nLevel-1 bound states, stage=1
    for k in range(1, nLevel - 1):
        n = k + 1
        Level[k]["erg"] = Ry - Ry * (1.0 / n**2)
        Level[k]["g"] = 2 * n * n
        Level[k]["stage"] = 1
        Level[k]["n"] = n
        Level[k]["isGround"] = 0
        _Level_info_list.append((f"{n}", "-", "-"))

    # continuum, stage=2
    Level[nLevel - 1]["erg"] = Ry
    Level[nLevel - 1]["g"] = 1
    Level[nLevel - 1]["stage"] = 2
    Level[nLevel - 1]["n"] = 0
    Level[nLevel - 1]["isGround"] = 1
    _Level_info_list.append(("-", "-", "-"))

    Level_info_table: T_TUPLE[T_TUPLE[T_STR, T_STR, T_STR], ...] = tuple(_Level_info_list)

    _atom_type = E_ATOM.HYDROGEN

    # counts and CTJ/index mapping tables
    nLine, nCont, nTran, _has_continuum = _AtomIO.nLine_nCont_nTran_(Level["stage"])
    nBoundLevel = nLevel - 1 if _has_continuum else nLevel
    Line_idx_table, Line_ctj_table, Cont_idx_table, Cont_ctj_table = _AtomIO.prepare_idx_ctj_mapping_(
        Level_info_table, Level["stage"], Level["isGround"], nLine, nCont
    )
    _ctj_table = CTJ_Table(Level=Level_info_table, Line=Line_ctj_table, Cont=Cont_ctj_table)
    _idx_table = Index_Table(Line=Line_idx_table, Cont=Cont_idx_table)

    # Cont
    Cont = _AtomIO.make_Atom_Cont_(nCont, Cont_idx_table, Level)

    # Line (path=None -> Aji computed from Hydrogen.einstein_A_coefficient_)
    Line, data_source_Aji = _AtomIO.make_Atom_Line_(None, Level, Line_idx_table, Line_ctj_table, _atom_type)

    # CE (path=None -> empty tables, CALCULATE source; rates computed in SELib)
    Te_table_CE, Omega_table_CE, Coe_CE, _tt, _ts, _tf = _AtomIO.make_Atom_CECI_(
        None, "CE", nLine, Line, Level, Level_info_table, Line_ctj_table
    )
    CE = Collisional_Transition(
        _transition_type=_tt,
        _transition_source=_ts,
        _transition_formula=_tf,
        Te_table=Te_table_CE,
        Omega_table=Omega_table_CE,
        Coe=Coe_CE,
    )
    data_source_CE = E_ATOMIC_DATA_SOURCE.CALCULATE

    # CI (path=None -> empty tables, CALCULATE source)
    Te_table_CI, Omega_table_CI, Coe_CI, _tt, _ts, _tf = _AtomIO.make_Atom_CECI_(
        None, "CI", nCont, Cont, Level, Level_info_table, Cont_ctj_table
    )
    CI = Collisional_Transition(
        _transition_type=_tt,
        _transition_source=_ts,
        _transition_formula=_tf,
        Te_table=Te_table_CI,
        Omega_table=Omega_table_CI,
        Coe=Coe_CI,
    )
    data_source_CI = E_ATOMIC_DATA_SOURCE.CALCULATE

    # RL (path=None -> empty)
    Coe_RL, nRadiativeLine = _AtomIO.make_Atom_RL_(None, Level_info_table, Line_ctj_table)
    RL = Radiative_Line(nRadiativeLine=nRadiativeLine, Coe=Coe_RL)

    # wavelength mesh
    waveMesh = _WavelengthMesh.init_Wave_Mesh_(Cont, Line, RL.Coe)

    # PI (path=None + HYDROGEN -> cross section from DegenerateN)
    alpha_table, alpha_table_idxs, Coe_PI, alpha_interp, data_source_PI = _AtomIO.make_Atom_PI_(
        None, Level, Cont, waveMesh.Cont_mesh, _atom_type, Level_info_table, Cont_ctj_table
    )
    PI = Photo_Ionization(
        alpha_table=alpha_table,
        alpha_table_idxs=alpha_table_idxs,
        Coe=Coe_PI,
        alpha_interp=alpha_interp,
    )

    _atomic_data_source = ATOMIC_DATA_SOURCE(
        AJI=data_source_Aji,
        CE=data_source_CE,
        CI=data_source_CI,
        PI=data_source_PI,
    )

    atom = Atom(
        Z=Z,
        Mass=Mass,
        Abun=Abun,
        nLevel=nLevel,
        nBoundLevel=nBoundLevel,
        nLine=nLine,
        nCont=nCont,
        nTran=nTran,
        nRL=nRadiativeLine,
        Level=Level,
        Line=Line,
        Cont=Cont,
        _has_continuum=_has_continuum,
        _atomic_data_source=_atomic_data_source,
        _atom_type=_atom_type,
        _ctj_table=_ctj_table,
        _idx_table=_idx_table,
        CE=CE,
        CI=CI,
        PI=PI,
        RL=RL,
        _wave_mesh=waveMesh,
    )
    return atom
