# -------------------------------------------------------------------------------
# build-once setup of the MALI iteration: loop-invariant tables
#
# runs a single time per model configuration, so it stays interpreted and is
# free to use dataclasses and python loops. every transition -- line or
# continuum -- is ACTIVE: its opacity comes from the populations inside the
# sweep, and its rates from the radiation field solved on the one global axis.
# the background (H-minus, H f-f, ...) is what the active atom does NOT
# provide: it is thermal (eta = chi*B) plus a separate coherent-scattering
# extinction sigma that the sweep pairs with sigma*J, never with B.
# -------------------------------------------------------------------------------

from collections.abc import Callable as _Callable
from dataclasses import dataclass as _dataclass

import numpy as _numpy

from ...Atomic import BasicP as _BasicP
from ...Atomic import Collision as _Collision
from ...Atomic import LTELib as _LTELib
from ...Function.SEquil import SELib as _SELib
from ...ImportAll import *
from ...RadiativeTransfer.MALI import GlobalMesh as _GlobalMesh
from ...RadiativeTransfer.MALI import ProfileTable as _ProfileTable
from ...RadiativeTransfer.MALI.Precompute import MALI_Precompute


# Toy_Atom and Atmos1D are temporary input types: Struct.Atom.Atom and AtmosphereC1D do not yet carry the
# fabricated collision tables and the per-species density, so the solver reads these until it is moved onto them
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
    # continuum meshes and cross sections (empty for line-only toys); the
    # sweep assembles NLTE b-f opacity from these and the current populations
    Cont_mesh: T_ARRAY  # (nCont, nContMesh), [cm], descending from the threshold
    alpha: T_ARRAY  # (nCont, nContMesh), photoionization cross section, [cm^2]
    am: T_FLOAT  # atomic mass relative to hydrogen
    nLevel: T_INT
    nLine: T_INT
    nCont: T_INT


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class Atmos1D:
    Z: T_ARRAY  # (ND,), depth below the upper surface, [cm], Z[0] = 0, ascending
    Te: T_ARRAY  # (ND,), [K]
    Ne: T_ARRAY  # (ND,), [cm^-3]
    Vt: T_ARRAY  # (ND,), [cm/s]
    Nt: T_ARRAY  # (ND,), total number density of the toy species, [cm^-3]
    ND: T_INT


# temporary: the LTE populations still come from SELib._ni_nj_LTE_; the routine moves to Atomic/LTELib in a later
# step, after which this module no longer imports SELib
def precompute_(
    atom: Toy_Atom,
    atmos: Atmos1D,
    mesh: _GlobalMesh.Global_Mesh,
    adamp_const: T_FLOAT = 0.0,
    Cij_dep: T_ARRAY | None = None,
    adamp_dep: T_ARRAY | None = None,
    bg_fn: _Callable[[float], T_TUPLE[T_ARRAY, T_ARRAY]] | None = None,
) -> MALI_Precompute:
    """Optional real-atom inputs (defaults reproduce the toy behavior):
    Cij_dep: (ND, nTran), per-depth upward collisional coefficients
        (temperature-dependent tables); None broadcasts atom.Cij_coe
    adamp_dep: (ND, nLine), per-depth Voigt damping; None uses adamp_const
    bg_fn: wl_cm -> (chi_thermal (ND,), sigma (ND,)); the thermal part must
        EXCLUDE this atom's own b-f (that opacity comes from the populations
        in the sweep) and must not contain scattering (which goes in sigma:
        Thomson, and Rayleigh off the ground level -- RH multiplies the
        Rayleigh cross section by the atmosphere file's ground population
        once, before iterating, so it is build-once here as well).
        None leaves every background table at zero.
    """
    ND = atmos.ND
    nLine = atom.nLine
    nCont = atom.nCont
    nTran = nLine + nCont
    if mesh.Nblue.shape[0] != nTran:
        raise ValueError(f"mesh has {mesh.Nblue.shape[0]} windows, atom has {nTran} transitions")

    n_LTE = _numpy.empty((ND, atom.nLevel), dtype=DT_NB_FLOAT)
    nj_by_ni = _numpy.empty((ND, nTran), dtype=DT_NB_FLOAT)
    Cji_coe = _numpy.empty((ND, nTran), dtype=DT_NB_FLOAT)
    dopWidth_cm = _numpy.empty((ND, nLine), dtype=DT_NB_FLOAT)
    if Cij_dep is None:
        Cij_coe = _numpy.broadcast_to(atom.Cij_coe[:], (ND, nTran)).copy()
    else:
        Cij_coe = _numpy.ascontiguousarray(Cij_dep, dtype=DT_NB_FLOAT)
    for k in range(ND):
        ni, ratio = _SELib._ni_nj_LTE_(atom.Level, atom.Line, atom.Cont, atmos.Te[k], atmos.Ne[k])
        n_LTE[k, :] = ni
        nj_by_ni[k, :] = ratio
        Cji_coe[k, :] = _Collision.Cij_to_Cji_(Cij_coe[k, :], ratio[:])
        dopWidth_cm[k, :] = _BasicP.doppler_width_(atom.Line["w0"][:], atmos.Te[k], atmos.Vt[k], atom.am)

    if adamp_dep is None:
        adamp = _numpy.full((ND, nLine), adamp_const, dtype=DT_NB_FLOAT)
    else:
        adamp = _numpy.ascontiguousarray(adamp_dep, dtype=DT_NB_FLOAT)

    nWinLine = int(mesh.win_off[nLine - 1, 1]) if nLine > 0 else 0
    phi = _numpy.empty((nWinLine, ND), dtype=DT_NB_FLOAT)
    weight = _numpy.empty(nWinLine, dtype=DT_NB_FLOAT)
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
        phi[mesh.win_off[kL, 0] : mesh.win_off[kL, 1], :] = phi_win
        weight[mesh.win_off[kL, 0] : mesh.win_off[kL, 1]] = weight_win
        wphi[:, kL] = wphi_line

    row_cont0 = nWinLine
    # the b-f rate trapezoid needs at least two window points per continuum
    if nCont > 0 and int(mesh.span[nLine:].min()) < 2:
        raise ValueError("every continuum window must carry at least two axis points")
    nWinCont = int(mesh.win_off[nTran - 1, 1]) - row_cont0 if nCont > 0 else 0
    alpha_win = _numpy.empty(nWinCont, dtype=DT_NB_FLOAT)
    for kC in range(nCont):
        t = nLine + kC
        wl_win = mesh.wl[mesh.Nblue[t] : mesh.Nblue[t] + mesh.span[t]]
        order = _numpy.argsort(atom.Cont_mesh[kC, :])
        # log-log interpolation: alpha spans decades over a continuum range;
        # the window is exactly the mesh's closed range, so no extrapolation
        alpha_win[mesh.win_off[t, 0] - row_cont0 : mesh.win_off[t, 1] - row_cont0] = _numpy.exp(
            _numpy.interp(_numpy.log(wl_win), _numpy.log(atom.Cont_mesh[kC, order]), _numpy.log(atom.alpha[kC, order]))
        )

    hnu_k = CST.h_ * CST.c_ / (mesh.wl * CST.k_)
    exp_hnu_kT = _numpy.ascontiguousarray(_numpy.exp(-hnu_k[:, None] / atmos.Te[None, :]), dtype=DT_NB_FLOAT)
    twohc2_wl5 = _numpy.asarray(2.0 * CST.h_ * CST.c_**2 / mesh.wl**5, dtype=DT_NB_FLOAT)
    hn_bottom = _numpy.asarray(_LTELib.planck_cm_(mesh.wl[:], atmos.Te[ND - 1]), dtype=DT_NB_FLOAT)

    bg_chi, bg_eta, bg_sca = _background_tables_(mesh.wl, atmos.Te, bg_fn)

    return MALI_Precompute(
        n_LTE=n_LTE,
        nj_by_ni=nj_by_ni,
        Cij_coe=Cij_coe,
        Cji_coe=Cji_coe,
        dopWidth_cm=dopWidth_cm,
        adamp=adamp,
        phi=phi,
        wphi=wphi,
        weight=weight,
        alpha_win=alpha_win,
        row_cont0=row_cont0,
        exp_hnu_kT=exp_hnu_kT,
        twohc2_wl5=twohc2_wl5,
        hn_bottom=hn_bottom,
        bg_chi=bg_chi,
        bg_eta=bg_eta,
        bg_sca=bg_sca,
    )


def _background_tables_(
    wl: T_ARRAY, Te: T_ARRAY, bg_fn: _Callable[[float], T_TUPLE[T_ARRAY, T_ARRAY]] | None
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    """Thermal background chi/eta and scattering sigma on every axis column."""
    Nspect = wl.shape[0]
    ND = Te.shape[0]
    bg_chi = _numpy.zeros((Nspect, ND), dtype=DT_NB_FLOAT)
    bg_eta = _numpy.zeros((Nspect, ND), dtype=DT_NB_FLOAT)
    bg_sca = _numpy.zeros((Nspect, ND), dtype=DT_NB_FLOAT)
    if bg_fn is None:
        return bg_chi, bg_eta, bg_sca
    for iw in range(Nspect):
        chi_th, sigma = bg_fn(float(wl[iw]))
        chi_th = _numpy.asarray(chi_th, dtype=DT_NB_FLOAT)
        sigma = _numpy.asarray(sigma, dtype=DT_NB_FLOAT)
        # a scalar or a (1,) array would broadcast silently into a "valid" table
        if chi_th.shape != (ND,) or sigma.shape != (ND,):
            raise ValueError(f"bg_fn at wl={wl[iw]:.6e} cm must return two ({ND},) arrays")
        if not (_numpy.all(_numpy.isfinite(chi_th)) and _numpy.all(_numpy.isfinite(sigma))):
            raise ValueError(f"bg_fn at wl={wl[iw]:.6e} cm returned non-finite values")
        bg_chi[iw, :] = chi_th
        bg_eta[iw, :] = chi_th * _LTELib.planck_cm_(float(wl[iw]), Te[:])
        bg_sca[iw, :] = sigma
    return bg_chi, bg_eta, bg_sca
