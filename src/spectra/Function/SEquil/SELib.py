# -------------------------------------------------------------------------------
# definition of functions to perform statistical equilibrium
# -------------------------------------------------------------------------------

from collections import namedtuple as _namedtuple

import numpy as _numpy

from ...Atomic import BasicP as _BasicP
from ...Atomic import Collision as _Collision
from ...Atomic import Hydrogen as _Hydrogen
from ...Atomic import LTELib as _LTELib
from ...Atomic import PhotoIonize as _PhotoIonize
from ...Atomic import SEsolver as _SEsolver
from ...Elements import TOTAL_ABUN as _TOTAL_ABUN
from ...Elements import WEIGHTED_TOTAL_MASS as _WEIGHTED_TOTAL_MASS
from ...ImportAll import *
from ...Math import Integrate as _Integrate
from ...RadiativeTransfer import Profile as _Profile
from ...Struct import Atmosphere as _Atmosphere

# from ...Struct.WavelengthMesh import _N_LINE_MESH, _LINE_MESH_QCORE, _LINE_MESH_QWING, _LINE_MESH_TYPE
# -----------------------------------------------------------------------------
# high level functions with struct as function argument
# -----------------------------------------------------------------------------
from ...Struct import Atom as _Atom
from ...Struct import Container as _Container
from ...Struct import Radiation as _Radiation
from ...Util import MeshUtil as _MeshUtil


def cal_SE_with_Pg_Te_single_Atom_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    radiation: _Radiation.Radiation,
    se_params: _Container.SE_Params_Container | None = None,
):
    Pg = atmos.Pg
    Te = atmos.Te
    _Vt = atmos.Vt
    kT = CST.k_ * Te

    is_hydrogen = atom._atom_type == E_ATOM.HYDROGEN

    Ne2Ng = 0.1
    Ne2Ng_prev = Ne2Ng
    Nh_SE = None

    Ng = Pg / ((1.0 + Ne2Ng) * kT)  # ( 0.5 * atom.Mass * CST.mH_ * Vt*Vt + ( 1. + Ne2Ng ) * kT)
    atmos.Ne = Ng * Ne2Ng

    atmos.Nh = Ng if is_hydrogen else 0.0

    PI_intensity: T_ARRAY | None = None
    while True:
        print(f"Ne2Ng={Ne2Ng:.3E}  Ng={Ng:.3E}  Ne={atmos.Ne:.3E}")
        SE_con, tran_rate_con = cal_SE_(atom, atmos, radiation, Nh_SE, se_params, PI_intensity=PI_intensity)
        PI_intensity = SE_con.PI_intensity
        n_SE = SE_con.n_SE
        Ne2Ng = 2 * n_SE[-1] + n_SE[-7:-1].sum()  ##: for He
        Ne_SE = Ng * Ne2Ng
        Ne_new = 0.5 * (Ne_SE + atmos.Ne)
        # print("ratio: ", Ne2Nh, Ne2Nh_prev)
        if (abs(Ne2Ng - Ne2Ng_prev) / Ne2Ng_prev) < 0.01:
            atmos.Ne = Ne_new
            break
        else:
            Ng = Pg / ((1.0 + Ne2Ng) * kT)  # ( 0.5 * atom.Mass * CST.mH_ * Vt*Vt + ( 1. + Ne2Ng ) * kT)
            atmos.Ne = Ne_new
            Ne2Ng_prev = Ne2Ng

        if is_hydrogen:
            atmos.Nh = Ng
            Nh_SE = n_SE

    atmos.Nh = Ng

    return SE_con, tran_rate_con


def cal_SE_with_Pg_Te_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    radiation: _Radiation.Radiation,
    Nh_SE: T_ARRAY | None,
    se_params: _Container.SE_Params_Container | None = None,
) -> T_TUPLE[_Container.SE_Container, _Container.TranRates_Container]:
    Pg = atmos.Pg
    Te = atmos.Te
    Vt = atmos.Vt
    kT = CST.k_ * Te

    is_hydrogen = atom._atom_type == E_ATOM.HYDROGEN

    Ne2Nh = 0.5
    Ne2Nh_prev = Ne2Nh
    if is_hydrogen:
        atmos.Nh = Pg / (0.5 * _WEIGHTED_TOTAL_MASS * CST.mH_ * Vt * Vt + (_TOTAL_ABUN + Ne2Nh) * kT)
        atmos.Ne = atmos.Nh * Ne2Nh

    PI_intensity: T_ARRAY | None = None
    while True:
        # print(f"Ne2Nh={Ne2Nh}, Ne={atmos.Ne:.2E}")
        SE_con, tran_rate_con = cal_SE_(atom, atmos, radiation, Nh_SE, se_params, PI_intensity=PI_intensity)
        PI_intensity = SE_con.PI_intensity
        n_SE = SE_con.n_SE

        if is_hydrogen:
            # print(f"{n_SE[0]:.2E}, {n_SE[-1]:.2E}")
            Ne2Nh = n_SE[-1] + 1.0e-4  ##: bubble effect
            Ne_SE = atmos.Nh * Ne2Nh
            Ne_new = 0.5 * (Ne_SE + atmos.Ne)
            # print("ratio: ", Ne2Nh, Ne2Nh_prev)
            if (abs(Ne2Nh - Ne2Nh_prev) / Ne2Nh_prev) < 0.01:
                atmos.Ne = Ne_new
                break
            else:
                atmos.Nh = Pg / (0.5 * _WEIGHTED_TOTAL_MASS * CST.mH_ * Vt * Vt + (_TOTAL_ABUN + Ne2Nh) * kT)
                atmos.Ne = Ne_new
                Ne2Nh_prev = Ne2Nh
        else:
            break

    SE_con.Nh = atmos.Nh
    SE_con.Ne = atmos.Ne
    if is_hydrogen:
        SE_con.Ntotal = atmos.Nh
    else:
        SE_con.Ntotal = atmos.Nh * atom.Abun

    return SE_con, tran_rate_con


def cal_SE_with_Nh_Te_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    radiation: _Radiation.Radiation,
    Nh_SE: T_ARRAY | None,
    se_params: _Container.SE_Params_Container | None = None,
) -> T_TUPLE[_Container.SE_Container, _Container.TranRates_Container]:

    Nh = atmos.Nh  # [/cm^{3}]
    Ne0 = 1.0e-4 * Nh  # [/cm^{3}]
    if Nh_SE is None:
        atmos.Ne = 0.5 * Nh  # [/cm^{3}]
    # else:
    #     atmos.Ne = Ne0 + Nh * Nh_SE[-1]   # [/cm^{3}]

    is_hydrogen = atom._atom_type == E_ATOM.HYDROGEN

    PI_intensity: T_ARRAY | None = None
    while True:
        # print(f"Ne={atmos.Ne:.2E}")
        SE_con, tran_rate_con = cal_SE_(atom, atmos, radiation, Nh_SE, se_params, PI_intensity=PI_intensity)
        PI_intensity = SE_con.PI_intensity

        n_SE = SE_con.n_SE

        if is_hydrogen:
            Ne_SE = Ne0 + Nh * n_SE[-1]
            Ne_new = 0.5 * (Ne_SE + atmos.Ne)
            # Ne_new = Ne_SE
            if (abs(Ne_new - atmos.Ne) / atmos.Ne) < 0.01:
                atmos.Ne = Ne_new
                break
            else:
                atmos.Ne = Ne_new
        else:
            break

    SE_con.Nh = atmos.Nh
    SE_con.Ne = atmos.Ne
    if is_hydrogen:
        SE_con.Ntotal = atmos.Nh
    else:
        SE_con.Ntotal = atmos.Nh * atom.Abun

    return SE_con, tran_rate_con


def cal_SE_with_Ne_Te_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    radiation: _Radiation.Radiation,
    Nh_SE: T_ARRAY | None,
    se_params: _Container.SE_Params_Container | None = None,
    is_single_element: bool = False,
    rate_only: T_BOOL = False,
) -> T_TUPLE[_Container.SE_Container, _Container.TranRates_Container]:
    r"""
    if is_single_element=False, then atmos.Nh must be provided when it is not hydrogen atom
    """
    ##    is_hydrogen = ( atom._atom_type ==  E_ATOM.HYDROGEN )

    ## : this comment out block tries to compute Nh with iteration
    ##   currently, we assume Nh does not change much in the iteration
    ##   (in collisional brodenning functions), so we fix Nh
    ##   maybe we need this iteration when we include collisional with proton and H I
    ##
    ##    if (Nh_SE is None) & (~is_hydrogen) :
    ##        raise ValueError("could not perform SE for non-hydrogen atom without given `Nh_SE`.")
    ##
    ##    if (Nh_SE is None) & (is_hydrogen):
    ##            atmos.Nh = 2 * atmos.Ne
    ##    else:
    ##        atmos.Nh  = atmos.Ne / ( 1.E-4 + Nh_SE[-1] )          # [cm^{-3}]

    is_hydrogen = atom._atom_type == E_ATOM.HYDROGEN

    if Nh_SE is None:
        atmos.Nh = 2 * atmos.Ne
    else:
        atmos.Nh = atmos.Ne / (1.0e-4 + Nh_SE[-1])

    if is_single_element:
        atmos.Nh = 0.0
        if (Nh_SE is not None) and (is_hydrogen):
            atmos.Nh = atmos.Ne / (Nh_SE[-1])

    SE_con, tran_rate_con = cal_SE_(atom, atmos, radiation, Nh_SE, se_params, rate_only=rate_only)
    if rate_only:
        return SE_con, tran_rate_con

    if is_hydrogen:
        if is_single_element:
            raise ValueError("Please keep is_single_element=False when is_hydrogen=True")
        atmos.Nh = atmos.Ne / (1.0e-4 + SE_con.n_SE[-1])
        SE_con.Ntotal = atmos.Nh
        SE_con.Nh = SE_con.Ntotal
    else:
        if is_single_element:
            atmos.Nh = atmos.Ne / SE_con.n_SE[-1]
            SE_con.Ntotal = atmos.Nh
            SE_con.Nh = 0.0
        else:
            SE_con.Ntotal = atmos.Nh / atom.Abun
            SE_con.Nh = atmos.Nh

    return SE_con, tran_rate_con


def cal_SE_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    radiation: _Radiation.Radiation,
    Nh_SE: T_ARRAY | None,
    se_params: _Container.SE_Params_Container | None = None,
    rate_only: T_BOOL = False,
    PI_intensity: T_ARRAY | None = None,
) -> T_TUPLE[_Container.SE_Container, _Container.TranRates_Container]:
    ##: TODO: instead of using background radiation in radiation struct
    ##        use an updatable MeanIntensity struct for lines and PI_intensity

    ## : extract variable from structs

    Mass = atom.Mass

    atom_type = atom._atom_type

    Level = atom.Level
    Line = atom.Line
    Cont = atom.Cont

    nLevel = atom.nLevel
    nLine = atom.nLine
    nCont = atom.nCont

    data_src_CE = atom._atomic_data_source.CE
    data_src_CI = atom._atomic_data_source.CI

    CE_Omega_table = atom.CE.Omega_table
    CE_Te_table = atom.CE.Te_table
    CE_Coe = atom.CE.Coe

    CI_Omega_table = atom.CI.Omega_table
    CI_Te_table = atom.CI.Te_table
    CI_Coe = atom.CI.Coe

    wMesh = atom._wave_mesh

    Cont_mesh = wMesh.Cont_mesh

    alpha_interp = atom.PI.alpha_interp

    solar = radiation.solar

    Line_mesh_Coe = wMesh.Line_Coe
    Line_mesh = wMesh.Line_mesh
    Line_mesh_idxs = wMesh.Line_mesh_idxs

    Te = atmos.Te
    Ne = atmos.Ne
    Vt = atmos.Vt
    Vd_sun = atmos.Vd_sun

    if se_params is None:
        se_params = _Container.SE_Params_Container()

    # Tr=None => use radiation.solar; not-None (including 0.0) => planck(Tr).
    # Tr_val is unused in the radiation.solar branch but must hold a numeric
    # type for the primitive _B_Jbar_ signature.
    use_Tr: T_BOOL = se_params.Tr is not None
    Tr: T_FLOAT = se_params.Tr if se_params.Tr is not None else 0.0

    if se_params.doppler_shift_continuum:
        raise NotImplementedError("Doppler shift of continuum wavelength mesh not yet implemented.")

    # Local placeholder: future continuum-shift work replaces this with a shifted
    # array. Today the continuum mesh is fixed (the flag above gates that path),
    # so we use wMesh.Cont_mesh directly. Exported via SE_Container.cont_wm_cm.
    cont_wave_mesh = Cont_mesh

    if PI_intensity is None:
        if use_Tr:
            PI_intensity = _LTELib.planck_cm_(cont_wave_mesh[:, :], Tr)
        else:
            PI_intensity = _PhotoIonize.interpolate_PI_intensity_(solar[:, :], cont_wave_mesh[:, :])

    Nh_I_ground: T_FLOAT
    if Nh_SE is None:
        Nh_I_ground = 0.5 * atmos.Nh  # half hydrogen atoms are in its H I ground Level
    else:
        Nh_I_ground = atmos.Nh * Nh_SE[0] / Nh_SE.sum()

    Aji = Line["AJI"][:]

    ## : append idxI, idxJ
    idxI = _numpy.empty(nLine + nCont, dtype=DT_NB_INT)
    idxJ = _numpy.empty(nLine + nCont, dtype=DT_NB_INT)
    idxI[:nLine] = Line["idxI"][:]
    idxJ[:nLine] = Line["idxJ"][:]
    idxI[nLine:] = Cont["idxI"][:]
    idxJ[nLine:] = Cont["idxJ"][:]

    ## : Given ..., perform SE to calculate n_SE

    n_LTE, nj_by_ni = _ni_nj_LTE_(Level, Line, Cont, Te, Ne)
    # nj_by_ni_Line = nj_by_ni[:nLine]
    nj_by_ni_Cont = nj_by_ni[nLine:]

    Rik, Rki_stim, Rki_spon = _bf_R_rate_(
        Cont,
        cont_wave_mesh[:, :],
        Te,
        nj_by_ni_Cont[:],
        alpha_interp[:, :],
        PI_intensity[:, :],
    )

    bj = _B_Jbar_(
        Line,
        Line_mesh_Coe,
        Line_mesh[:],
        Line_mesh_idxs[:, :],
        Te,
        Vt,
        Vd_sun,
        Ne,
        Nh_I_ground,
        Mass,
        atom_type,
        solar[:, :],
        Tr,
        use_Tr,
    )

    Cij = _get_Cij_(
        Line,
        Cont,
        Te,
        atom_type,
        CE_Omega_table,
        CE_Te_table,
        CE_Coe,
        data_src_CE,
        CI_Omega_table,
        CI_Te_table,
        CI_Coe,
        data_src_CI,
    )
    Cji = _Collision.Cij_to_Cji_(Cij[:], nj_by_ni[:])

    Rij, Rji_stim, Rji_spon = _make_Rji_Rij_(Aji[:], bj.Bji_Jbar[:], bj.Bij_Jbar[:], Rki_spon[:], Rki_stim[:], Rik[:])
    n_SE, Rmat, Cmat = _solve_SE_(
        nLevel, idxI[:], idxJ[:], Rji_spon[:], Rji_stim[:], Rij[:], Cji[:], Cij[:], Ne, rate_only
    )

    se_bb_con = _Container.SE_BB_Container(
        wm_cm_shifted_1d=bj.wm_cm_shifted_all,
        solar_intensity_shifted_1d=bj.solar_intensity_shifted_all,
    )

    SE_con = _Container.SE_Container(
        n_SE=n_SE,
        n_LTE=n_LTE,
        nj_by_ni=nj_by_ni,
        se_bb_con=se_bb_con,
        absorb_prof_1d=bj.absorb_prof_cm_all,
        wm_cm_1d=bj.wm_cm_all,
        Line_mesh_idxs=Line_mesh_idxs,
        Jbar=bj.Jbar_all,
        PI_intensity=PI_intensity,
        cont_wm_cm=cont_wave_mesh,
        Ntotal=0.0,
        Nh=0.0,
        Ne=Ne,
        Te=Te,
    )

    tran_rate_con = _Container.TranRates_Container(
        Rji_spon=Rji_spon[:],
        Rji_stim=Rji_stim[:],
        Rij=Rij[:],
        Cji_Ne=Cji[:] * Ne,
        Cij_Ne=Cij[:] * Ne,
        Rmat=Rmat,
        Cmat=Cmat,
    )

    return SE_con, tran_rate_con


# -----------------------------------------------------------------------------
# mid level functions with array as function argument
# -----------------------------------------------------------------------------


def _ni_nj_LTE_(Level: T_ARRAY, Line: T_ARRAY, Cont: T_ARRAY, Te: T_FLOAT, Ne: T_FLOAT) -> T_TUPLE[T_ARRAY, T_ARRAY]:

    _nLevel = Level.shape[0]
    nLine = Line.shape[0]
    nCont = Cont.shape[0]
    nTran = nLine + nCont

    ## : initilize _nj_by_ni
    nj_by_ni = _numpy.empty(nTran, dtype=DT_NB_FLOAT)
    idxI = _numpy.empty(nTran, dtype=DT_NB_INT)
    idxJ = _numpy.empty(nTran, dtype=DT_NB_INT)

    ## : for line transitions
    gi = Line["gi"][:]
    gj = Line["gj"][:]
    Eji = CST.h_ * Line["f0"][:]

    nj_by_ni[:nLine] = _LTELib.boltzmann_distribution_(gi[:], gj[:], Eji[:], Te)
    idxI[:nLine] = Line["idxI"][:]
    idxJ[:nLine] = Line["idxJ"][:]

    ## : if there is continuum transition
    if nCont > 0:
        gi = Cont["gi"][:]
        gj = Cont["gj"][:]
        chi = CST.h_ * Cont["f0"][:]

        nj_by_ni[nLine:] = _LTELib.saha_distribution_(gi[:], gj[:], chi[:], Ne, Te)
        idxI[nLine:] = Cont["idxI"][:]
        idxJ[nLine:] = Cont["idxJ"][:]

    isGround = Level["isGround"][:]

    ni = _nj_by_ni_To_ni_(nj_by_ni[:], idxI[:], idxJ[:], isGround[:], nLine)

    return ni, nj_by_ni


def _nj_by_ni_To_ni_(nj_by_ni: T_ARRAY, idxI: T_ARRAY, idxJ: T_ARRAY, isGround: T_ARRAY, nLine: T_INT) -> T_ARRAY:

    nLevel = isGround.shape[0]
    nTran = idxI.shape[0]

    ni = _numpy.empty(nLevel, dtype=DT_NB_FLOAT)
    ni[0] = 1.0

    # loop through continuum transition first,
    # must be sorted from lower ionization stage to higher ionization stage
    # if so, we have all ground level population relative to the very first ground level
    for k in range(nLine, nTran):
        i = idxI[k]
        j = idxJ[k]
        if isGround[i]:
            ni[j] = nj_by_ni[k] * ni[i]

    # loop through line transition
    for k in range(nLine):
        i = idxI[k]
        j = idxJ[k]
        if isGround[i]:
            ni[j] = nj_by_ni[k] * ni[i]

    return ni[:] / ni.sum(axis=0)


def _bf_R_rate_(
    Cont: T_ARRAY,
    Cont_mesh: T_ARRAY,
    Te: T_FLOAT,
    nj_by_ni_Cont: T_ARRAY,
    alpha_interp: T_ARRAY,
    PI_intensity: T_ARRAY,
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    # -------------------------------------------------------------------------
    # we compute/interpolate photoionizatoin cross section only once
    # and assume that while suffering Doppler shift
    #    - continuum wavelength mesh might shift
    #        (but for the sake of simplicity, we assume they do not shift)
    #    - photoionizatoin cross section keep constant
    # -------------------------------------------------------------------------
    nCont = Cont.shape[0]

    Rik = _numpy.empty(nCont, dtype=DT_NB_FLOAT)
    Rki_stim = _numpy.empty(nCont, dtype=DT_NB_FLOAT)
    Rki_spon = _numpy.empty(nCont, dtype=DT_NB_FLOAT)
    ## loop over continuum transition
    for kL in range(nCont):
        res = _PhotoIonize.bound_free_radiative_transition_coefficient_(
            wave=Cont_mesh[kL, ::-1],
            J=PI_intensity[kL, ::-1],
            alpha=alpha_interp[kL, ::-1],
            Te=Te,
            nk_by_ni_LTE=nj_by_ni_Cont[kL],
        )
        Rik[kL] = res[0]
        Rki_stim[kL] = res[1]
        Rki_spon[kL] = res[2]

    return Rik, Rki_stim, Rki_spon


# NamedTuple return type for _B_Jbar_. Module-scope so numba can capture it
# at @njit-compile time (numba supports collections.namedtuple as a tuple
# with attribute access in nopython mode).
_B_Jbar_Result = _namedtuple(
    "_B_Jbar_Result",
    [
        "Bij_Jbar",  # 1d (nLine,)
        "Bji_Jbar",  # 1d (nLine,)
        "absorb_prof_cm_all",  # 1d (sum_of_line_wavelength_mesh,)
        "wm_cm_all",  # 1d (sum_of_line_wavelength_mesh,)
        "wm_cm_shifted_all",  # 1d (sum_of_line_wavelength_mesh,)
        "solar_intensity_shifted_all",  # 1d (sum_of_line_wavelength_mesh,)
        "Jbar_all",  # 1d (nLine,)
    ],
)


def _B_Jbar_(
    Line: T_ARRAY,
    Line_mesh_Coe: T_ARRAY,
    Line_mesh: T_ARRAY,
    Line_mesh_idxs: T_ARRAY,
    Te: T_FLOAT,
    Vt: T_FLOAT,
    Vd_sun: T_FLOAT,
    Ne: T_FLOAT,
    Nh_I_ground: T_FLOAT,
    Mass: T_FLOAT,
    atom_type: T_E_ATOM,
    backRad: T_ARRAY,
    Tr: T_FLOAT,
    use_Tr: T_BOOL,
) -> "_B_Jbar_Result":
    ##: TODO: add input argument for PRD correlation matrix and PRD/CRD binary indicator for lines
    ##        this requires adding one more ProfileType called PRD

    # Mesh-shift mechanic (see docs/tasks/009-doppler-velocity-split/refactor_04.md):
    #   absorb_prof_cm_all  : sigma(wm) / dopWidth_cm — unshifted base profile
    #                         (atom rest frame). Consumed by downstream forward
    #                         models (slab/cloud), which apply their own Vd_obs
    #                         shift via the output wavelength axis.
    #   wm_cm_all           : sun-frame wavelength labels in cm
    #                         (= wm * dopWidth_cm + w0). Unshifted; exported so
    #                         the cloud model can build its observer-frame mesh
    #                         without recomputing dopWidth_cm.
    #   wm_cm_shifted_all   : sun-frame wavelengths the atom samples after a
    #                         Doppler boost (= wm_cm - w0*Vd_sun/c). Used to
    #                         interpolate the solar spectrum without shifting
    #                         the profile off-mesh at large |Vd_sun|.
    #   solar_intensity_shifted_all : backRad / planck evaluated at
    #                         wm_cm_shifted_all (per-line). Stored so the SE
    #                         radiation field is inspectable at the same
    #                         wavelengths the Jbar integral consumed.
    # Sign convention: +Vd_sun = OUTWARDS from the sun, so the sun-frame
    # absorption line center sits at w0 - w0*Vd_sun/c (blue of w0). Sampling
    # the sun at the atom's rest-frame wavelengths therefore requires querying
    # the solar spectrum at wm_cm - w0*Vd_sun/c.

    nLine = Line.shape[0]

    absorb_prof_cm_all: T_ARRAY = _numpy.empty_like(Line_mesh)
    # Sun-frame wavelength labels in cm, parallel to absorb_prof_cm_all. Exposed
    # so the cloud model can build its observer-frame mesh without recomputing
    # dopWidth_cm (Te/Vt-dependent, identical to the value used here).
    wm_cm_all: T_ARRAY = _numpy.empty_like(Line_mesh)
    # Shifted-mesh diagnostics. Every slice is filled in the loop below:
    # active lines with their shifted mesh/intensity, inactive (f0<=0) lines
    # in the early-skip block (wavelength -> inf, intensity -> 0).
    wm_cm_shifted_all: T_ARRAY = _numpy.zeros_like(Line_mesh)
    solar_intensity_shifted_all: T_ARRAY = _numpy.zeros_like(Line_mesh)
    Jbar_all: T_ARRAY = _numpy.empty(nLine, dtype=DT_NB_FLOAT)

    Bji_Jbar = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    Bij_Jbar = _numpy.empty(nLine, dtype=DT_NB_FLOAT)

    absorb_prof_cm: T_ARRAY
    for k in range(nLine):
        i_start, i_end = Line_mesh_idxs[k, :]

        if Line["f0"][k] <= 0:
            # inactive line (degenerate levels, w0=inf): wavelength-like
            # slices are inf (no finite wavelength exists), radiation-like
            # slices are physically 0. Downstream consumers must exclude
            # these slices from arithmetic (label/diagnostic use only).
            Jbar_all[k] = 0.0
            Bij_Jbar[k] = 0.0
            Bji_Jbar[k] = 0.0
            wm_cm_all[i_start:i_end] = _numpy.inf
            wm_cm_shifted_all[i_start:i_end] = _numpy.inf
            absorb_prof_cm_all[i_start:i_end] = 0.0
            solar_intensity_shifted_all[i_start:i_end] = 0.0
            continue

        ## collisional broadening line width
        gamma: T_FLOAT = Line["Gamma"][k]
        if atom_type == E_ATOM.HYDROGEN:
            gamma += _Hydrogen.collisional_broadening_Res_and_Van_(Line["ni"][k], Line["nj"][k], Nh_I_ground, Te)
            gamma += _Hydrogen.collisional_broadening_LinearStark_(Line["ni"][k], Line["nj"][k], Ne)
        ##: TODO: how about the collisional broadening of non-hydrogen atom

        Bij = Line["BIJ"][k]
        Bji = Line["BJI"][k]
        w0 = Line["w0"][k]
        f0 = Line["f0"][k]
        dopWidth_cm = _BasicP.doppler_width_(w0, Te, Vt, Mass)

        # Line_mesh[i_start:i_end]                                       ##: Line_mesh not used?
        proftype = Line_mesh_Coe["ProfileType"][k]
        nLambda = Line_mesh_Coe["nLambda"][k]
        qcore = Line_mesh_Coe["qcore"][k]
        qwing = Line_mesh_Coe["qwing"][k]

        # wm : wave mesh [dop_width_cm]
        ## TODO : since we have already calculated line mesh in dopller width unit,
        #         this line could be replaced by
        #         wm = Line_mesh[i_start:i_end]
        wm = _MeshUtil.make_full_line_mesh_(nLambda, qcore, qwing)
        wm_cm_all[i_start:i_end] = wm[:] * dopWidth_cm + w0

        if proftype == E_ABSORPTION_PROFILE_TYPE.VOIGT:
            dopWidth_hz = dopWidth_cm * f0 / w0
            a = gamma / (4.0 * CST.pi_ * dopWidth_hz)
            absorb_prof_cm = _Profile.voigt_(a, wm[:])

        elif proftype == E_ABSORPTION_PROFILE_TYPE.GAUSSIAN:
            absorb_prof_cm = _Profile.gaussian_(wm[:])

        else:
            raise ValueError("Only 'VOIGT' and 'GAUSSIAN' are valid E_ABSORPTION_PROFILE_TYPE")

        # normalize: (prof_cm * wm_cm) integrates to 1 over each line's mesh
        absorb_prof_cm[:] = absorb_prof_cm[:] / dopWidth_cm
        absorb_prof_cm_all[i_start:i_end] = absorb_prof_cm[:]

        # Sun-frame unshifted mesh in [cm] — aliased to the already-stored
        # wm_cm_all segment for clarity inside the integrand expressions below.
        wm_cm = wm_cm_all[i_start:i_end]

        # Mesh-shift: query the solar spectrum at sun-frame wavelengths that
        # account for the atom's Doppler boost. wm_cm_shifted differs from
        # wm_cm by a constant offset (w0*Vd_sun/c is line-dependent but
        # wm-independent), so d(wm_cm_shifted) = d(wm_cm) and the trapezoid
        # measure is identical to the unshifted-mesh case.
        wm_cm_shifted = wm_cm[:] - w0 * Vd_sun / CST.c_
        wm_cm_shifted_all[i_start:i_end] = wm_cm_shifted[:]

        if use_Tr:
            # SE uses planck(Tr) as a scalar; broadcast across the per-line
            # slice for the debug field, then evaluate Jbar with the same
            # scalar pulled out of the trapze integrand.
            planck_val = _LTELib.planck_cm_(w0, Tr)
            solar_intensity_shifted_all[i_start:i_end] = planck_val
            Jbar0 = _Integrate.trapze_(absorb_prof_cm[:], wm_cm_shifted[:]) * planck_val
        else:
            solar_intensity_shifted = _numpy.interp(wm_cm_shifted[:], backRad[0, :], backRad[1, :])
            solar_intensity_shifted_all[i_start:i_end] = solar_intensity_shifted[:]
            integrand = solar_intensity_shifted[:] * absorb_prof_cm[:]
            Jbar0 = _Integrate.trapze_(integrand[:], wm_cm_shifted[:])

        Jbar_all[k] = Jbar0

        Bij_Jbar[k] = Bij * Jbar0
        Bji_Jbar[k] = Bji * Jbar0

    return _B_Jbar_Result(
        Bij_Jbar,
        Bji_Jbar,
        absorb_prof_cm_all,
        wm_cm_all,
        wm_cm_shifted_all,
        solar_intensity_shifted_all,
        Jbar_all,
    )


def _get_Cij_(  # noqa: C901
    Line: T_ARRAY,
    Cont: T_ARRAY,
    Te: T_FLOAT,
    atom_type: T_E_ATOM,
    CE_Omega_table: T_ARRAY,
    CE_Te_table: T_ARRAY,
    CE_Coe: T_ARRAY,
    data_src_CE: T_E_ATOMIC_DATA_SOURCE,
    CI_Omega_table: T_ARRAY,
    CI_Te_table: T_ARRAY,
    CI_Coe: T_ARRAY,
    data_src_CI: T_E_ATOMIC_DATA_SOURCE,
):

    nLine = Line.shape[0]
    nCont = Cont.shape[0]
    Cij = _numpy.empty(nLine + nCont, dtype=DT_NB_FLOAT)

    ## : for line transition
    if data_src_CE == E_ATOMIC_DATA_SOURCE.EXPERIMENT:
        for k in range(nLine):
            omega = _Collision.interp_omega_(CE_Omega_table[k, :], Te, CE_Te_table[:], CE_Coe["f1"][k], CE_Coe["f2"][k])
            Cij[k] = _Collision.CE_rate_coe_(omega, Te, CE_Coe["gi"][k], CE_Coe["dEij"][k])

    elif data_src_CE == E_ATOMIC_DATA_SOURCE.CALCULATE:
        if atom_type != E_ATOM.HYDROGEN:
            raise ValueError("we don't have function to calculate collisional rate coefficient for non-hydrogen atom.")

        for k in range(nLine):
            Cij[k] = _Hydrogen.CE_rate_coe_(Line["ni"][k], Line["nj"][k], Te)

    else:
        raise ValueError("only 'CALCULATE' and 'EXPERIMENT' are valid E_ATOMIC_DATA_SOURCE.")

    if nCont > 0:
        ## : for line transition
        if data_src_CI == E_ATOMIC_DATA_SOURCE.EXPERIMENT:
            for k in range(nCont):
                omega = _Collision.interp_omega_(CI_Omega_table[k, :], Te, CI_Te_table[:], 1.0, CI_Coe["f2"][k])
                Cij[k + nLine] = _Collision.CI_rate_coe_(omega, Te, CI_Coe["dEij"][k])

        elif data_src_CI == E_ATOMIC_DATA_SOURCE.CALCULATE:
            if atom_type != E_ATOM.HYDROGEN:
                raise ValueError(
                    "we don't have function to calculate collisional rate coefficient for non-hydrogen atom."
                )

            for k in range(nCont):
                Cij[k + nLine] = _Hydrogen.CI_rate_coe_(Cont["ni"][k], Te)

        else:
            raise ValueError("only 'CALCULATE' and 'EXPERIMENT' are valid E_ATOMIC_DATA_SOURCE.")

    else:
        raise ValueError("currently, atomic model without continuum is not yet supported.")

    return Cij


def _make_Rji_Rij_(
    Aji: T_ARRAY, Bji_Jbar: T_ARRAY, Bij_Jbar: T_ARRAY, Rki_spon: T_ARRAY, Rki_stim: T_ARRAY, Rik: T_ARRAY
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    nLine = Aji.shape[0]
    nCont = Rik.shape[0]
    nTran = nLine + nCont
    Rji_spon = _numpy.empty(nTran, dtype=DT_NB_FLOAT)
    Rji_stim = _numpy.empty(nTran, dtype=DT_NB_FLOAT)
    Rij = _numpy.empty(nTran, dtype=DT_NB_FLOAT)

    Rji_spon[:nLine] = Aji[:]
    Rji_spon[nLine:] = Rki_spon[:]
    Rji_stim[:nLine] = Bji_Jbar[:]
    Rji_stim[nLine:] = Rki_stim[:]
    Rij[:nLine] = Bij_Jbar[:]
    Rij[nLine:] = Rik[:]

    return Rij, Rji_stim, Rji_spon


def _solve_SE_(
    nLevel: T_INT,
    idxI: T_ARRAY,
    idxJ: T_ARRAY,
    Rji_spon: T_ARRAY,
    Rji_stim: T_ARRAY,
    Rij: T_ARRAY,
    Cji: T_ARRAY,
    Cij: T_ARRAY,
    Ne: T_FLOAT,
    rate_only: T_BOOL,
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:

    Cmat = _numpy.zeros((nLevel, nLevel), dtype=DT_NB_FLOAT)
    _SEsolver.set_matrixC_(Cmat[:, :], Cji[:], Cij[:], idxI[:], idxJ, Ne)

    Rmat = _numpy.zeros((nLevel, nLevel), dtype=DT_NB_FLOAT)
    _SEsolver.set_matrixR_(Rmat[:, :], Rji_spon[:], Rji_stim[:], Rij[:], idxI[:], idxJ[:])

    # n_SE = _SEsolver.solve_SE_(Rmat, Cmat)
    if rate_only:
        n_SE = _numpy.zeros(nLevel, dtype=DT_NB_FLOAT)
    else:
        n_SE = _SEsolver.solve_SE_(Rmat, Cmat)

    return n_SE, Rmat, Cmat


def Rmat_Cmat_to_Pmat_(Rmat: T_ARRAY, Cmat: T_ARRAY) -> T_ARRAY:
    r"""get  P matrix from R matrix, C matrix.

    Parameters
    ----------

    Rmat : T_ARRAY, (nLevel,nLevel)
        radiative transition rate matrix,
        [:math:`s^{-1}`]

    Cmat : T_ARRAY, (nLevel,nLevel)
        collisional transition rate matrix,
        [:math:`s^{-1}`]

    Returns
    -------

    A : T_ARRAY, (nLevel,nLevel)
        combined transition matrix with non-zero diagonal components (without abundance definition equation).
        [:math:`cm^{-3}`]

    """

    nLevel = Rmat.shape[0]
    A = Cmat[:, :] + Rmat[:, :]
    # b = _numpy.zeros(nLevel, dtype=DT_NB_FLOAT)

    # -------------------------------------------------------------
    # diagnal components
    # -------------------------------------------------------------
    for k in range(nLevel):
        A[k, k] = -A[:, k].sum()

    # -------------------------------------------------------------
    # abundance definition equation
    # -------------------------------------------------------------
    # A[-1,:] = 1.
    # b[-1] = 1.

    # nArr = _numpy.linalg.solve(A, b)

    return A


# -------------------------------------------------------------------------------
# numba optimization
# -------------------------------------------------------------------------------

if CFG._IS_JIT:
    _nj_by_ni_To_ni_ = nb_njit(**NB_NJIT_KWGS)(_nj_by_ni_To_ni_)
    _ni_nj_LTE_ = nb_njit(**NB_NJIT_KWGS)(_ni_nj_LTE_)
    _bf_R_rate_ = nb_njit(**NB_NJIT_KWGS)(_bf_R_rate_)
    _B_Jbar_ = nb_njit(**NB_NJIT_KWGS)(_B_Jbar_)
    _get_Cij_ = nb_njit(**NB_NJIT_KWGS)(_get_Cij_)
    _solve_SE_ = nb_njit(**NB_NJIT_KWGS)(_solve_SE_)
