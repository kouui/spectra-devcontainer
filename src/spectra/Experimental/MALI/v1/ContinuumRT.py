# -------------------------------------------------------------------------------
# active continua: radiative transfer on the merged continuum axis
#
# the passive tier prescribes the radiation driving the b-f rates; this module
# SOLVES it instead. per MALI iteration:
#     populations -> NLTE b-f opacity/emissivity on the continuum union axis
#     -> per-(wavelength, mu) Feautrier solve -> J_cont(wl, z)
#     -> Rik / Rki per continuum (the production rate integral)
# the rates enter the SE unpreconditioned (Lambda-iteration on the continua) --
# acceptable because the b-f photon pools are collisionally and thermally
# anchored far more strongly than resonance lines; measured on FALC hydrogen
# before being trusted further.
#
# overlap handling: continua of the same atom overlap each other (e.g. the
# Balmer and Paschen ranges share 205-365 nm) and every overlapping continuum
# contributes opacity at a shared wavelength. LINE opacity inside continuum
# ranges (Lyman lines sit in the Balmer continuum range) is neglected: a line
# core occupies a sliver of the continuum integration range, and the b-f rate
# integrand is smooth there.
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

import numpy as _numpy

from ....Atomic import LTELib as _LTELib
from ....Atomic import PhotoIonize as _PhotoIonize
from ....ImportAll import *
from ....RadiativeTransfer import Feautrier as _Feautrier
from . import Structs as _Structs

if CFG._IS_JIT:
    _formal_rh_ = _Feautrier.formal_improved_RH_
else:
    _formal_rh_ = nb_njit(**NB_NJIT_KWGS)(_Feautrier.formal_improved_RH_)


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class Cont_RT:
    """Build-once tier of the active-continuum solver."""

    wl_ax: T_ARRAY  # (nAx,), merged continuum axis, [cm], ascending
    # per-continuum cross section on the axis, 0 outside [lamb_min, edge]
    alpha_ax: T_ARRAY  # (nCont, nAx), [cm^2]
    in_range: T_ARRAY  # (nCont, nAx), bool, alpha support of each continuum
    exp_hnu_kT: T_ARRAY  # (nAx, ND), exp(-h nu/kT) stimulated factor
    twohc2_wl5: T_ARRAY  # (nAx,), 2 h c^2 / wl^5, cm-base emission factor
    bg_chi: T_ARRAY  # (nAx, ND), background extinction (must EXCLUDE the
    #   atom's own b-f -- that opacity comes from the populations here)
    bg_eta: T_ARRAY  # (nAx, ND), thermal background emissivity
    hn_bottom: T_ARRAY  # (nAx,), Planck at the lower boundary


def build_cont_rt_(
    atom: _Structs.Toy_Atom,
    atmos: _Structs.Atmos1D,
    bg_chi_fn=None,
) -> Cont_RT:
    """Merge the per-continuum meshes into one axis and tabulate everything
    populations never touch.

    bg_chi_fn: wl_cm -> (ND,) background extinction on the continuum axis;
        it must not include this atom's own bound-free.
    """
    nCont = atom.nCont
    ND = atmos.ND
    wl_ax = _numpy.unique(_numpy.concatenate([atom.Cont_mesh[kC, :] for kC in range(nCont)]))
    nAx = wl_ax.shape[0]

    alpha_ax = _numpy.zeros((nCont, nAx), dtype=DT_NB_FLOAT)
    in_range = _numpy.zeros((nCont, nAx), dtype=_numpy.bool_)
    for kC in range(nCont):
        mesh = atom.Cont_mesh[kC, :]
        lo, hi = float(mesh.min()), float(mesh.max())
        mask = (wl_ax >= lo) & (wl_ax <= hi)
        in_range[kC, :] = mask
        # log-log interpolation: alpha spans decades over a continuum range
        alpha_ax[kC, mask] = _numpy.exp(
            _numpy.interp(
                _numpy.log(wl_ax[mask]),
                _numpy.log(mesh[_numpy.argsort(mesh)]),
                _numpy.log(atom.alpha[kC, _numpy.argsort(mesh)]),
            )
        )

    hnu_k = CST.h_ * CST.c_ / (wl_ax * CST.k_)
    exp_hnu_kT = _numpy.exp(-hnu_k[:, None] / atmos.Te[None, :])
    twohc2_wl5 = 2.0 * CST.h_ * CST.c_**2 / wl_ax**5

    bg_chi = _numpy.zeros((nAx, ND), dtype=DT_NB_FLOAT)
    bg_eta = _numpy.zeros((nAx, ND), dtype=DT_NB_FLOAT)
    if bg_chi_fn is not None:
        for iw in range(nAx):
            bg_chi[iw, :] = bg_chi_fn(float(wl_ax[iw]))
            bg_eta[iw, :] = bg_chi[iw, :] * _LTELib.planck_cm_(float(wl_ax[iw]), atmos.Te[:])

    hn_bottom = _numpy.asarray(_LTELib.planck_cm_(wl_ax[:], atmos.Te[ND - 1]), dtype=DT_NB_FLOAT)

    return Cont_RT(
        wl_ax=wl_ax,
        alpha_ax=alpha_ax,
        in_range=in_range,
        exp_hnu_kT=_numpy.ascontiguousarray(exp_hnu_kT, dtype=DT_NB_FLOAT),
        twohc2_wl5=twohc2_wl5,
        bg_chi=bg_chi,
        bg_eta=bg_eta,
        hn_bottom=hn_bottom,
    )


def continuum_sweep_(
    Z: T_ARRAY,
    n_abs: T_ARRAY,
    idxI: T_ARRAY,
    idxJ: T_ARRAY,
    nj_by_ni: T_ARRAY,
    alpha_ax: T_ARRAY,
    exp_hnu_kT: T_ARRAY,
    twohc2_wl5: T_ARRAY,
    bg_chi: T_ARRAY,
    bg_eta: T_ARRAY,
    hn_bottom: T_ARRAY,
    mus: T_ARRAY,
    wmus: T_ARRAY,
) -> T_ARRAY:
    """Solve J on the continuum axis from the CURRENT populations.

    NLTE bound-free opacity and emissivity per axis wavelength iw, depth k:
        n_dag   = n_k(upper) / (nj/ni)_LTE        (LTE-consistent lower pop.)
        chi    += alpha * (n_i - n_dag * exp(-h nu/kT))
        eta    += alpha * (2 h c^2/wl^5) * n_dag * exp(-h nu/kT)
    summed over every continuum whose range contains iw, plus the background.

    Input:
        Z: (ND,), depth, [cm], ascending
        n_abs: (ND, nLevel), ABSOLUTE populations, [cm^-3]
        idxI, idxJ: (nCont,), continuum level indices
        nj_by_ni: (ND, nCont), LTE upper/lower ratio per continuum
        alpha_ax .. hn_bottom: the Cont_RT tables
        mus, wmus: angle quadrature on (0, 1), weights sum to 1

    Output:
        J: (nAx, ND), mean intensity, cm-base
    """
    nCont = idxI.shape[0]
    nAx = alpha_ax.shape[1]
    ND = Z.shape[0]
    n_mu = mus.shape[0]

    J = _numpy.zeros((nAx, ND), dtype=DT_NB_FLOAT)
    chi = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    eta = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    S = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    tau = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    for iw in range(nAx):
        for k in range(ND):
            chi[k] = bg_chi[iw, k]
            eta[k] = bg_eta[iw, k]
            for kC in range(nCont):
                a = alpha_ax[kC, iw]
                if a <= 0.0:
                    continue
                n_dag = n_abs[k, idxJ[kC]] / nj_by_ni[k, kC]
                stim = n_dag * exp_hnu_kT[iw, k]
                chi[k] += a * (n_abs[k, idxI[kC]] - stim)
                eta[k] += a * twohc2_wl5[iw] * stim
            S[k] = eta[k] / chi[k]
        tau[0] = 0.0
        for k in range(1, ND):
            tau[k] = tau[k - 1] + 0.5 * (chi[k - 1] + chi[k]) * (Z[k] - Z[k - 1])
        for im in range(n_mu):
            res = _formal_rh_(tau, S, mus[im], 0.0, 0.0, 0.0, hn_bottom[iw], E_FEAUTRIER_ORDER.SECOND, False)
            for k in range(ND):
                J[iw, k] += wmus[im] * res.j[k]
    return J


def bf_rates_(
    cont_rt: Cont_RT,
    atom: _Structs.Toy_Atom,
    atmos: _Structs.Atmos1D,
    nj_by_ni_Cont: T_ARRAY,
    n_pop: T_ARRAY,
    mus: T_ARRAY,
    wmus: T_ARRAY,
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    """One active-continuum step: solve J, then integrate the b-f rates.

    Input:
        nj_by_ni_Cont: (ND, nCont), LTE ratio of the continuum transitions
        n_pop: (ND, nLevel), normalized populations (times atmos.Nt inside)

    Output:
        Rik, Rki_stim, Rki_spon: (ND, nCont), [s^-1]
    """
    ND = atmos.ND
    nCont = atom.nCont
    n_abs = n_pop * atmos.Nt[:, None]
    J = continuum_sweep_(
        atmos.Z, n_abs,
        _numpy.ascontiguousarray(atom.Cont["idxI"][:]),
        _numpy.ascontiguousarray(atom.Cont["idxJ"][:]),
        nj_by_ni_Cont, cont_rt.alpha_ax, cont_rt.exp_hnu_kT, cont_rt.twohc2_wl5,
        cont_rt.bg_chi, cont_rt.bg_eta, cont_rt.hn_bottom, mus, wmus,
    )  # fmt: skip

    Rik = _numpy.empty((ND, nCont), dtype=DT_NB_FLOAT)
    Rki_stim = _numpy.empty((ND, nCont), dtype=DT_NB_FLOAT)
    Rki_spon = _numpy.empty((ND, nCont), dtype=DT_NB_FLOAT)
    for kC in range(nCont):
        mask = cont_rt.in_range[kC, :]
        wave = cont_rt.wl_ax[mask]  # ascending, as the rate integral expects
        alpha = cont_rt.alpha_ax[kC, mask]
        for k in range(ND):
            res = _PhotoIonize.bound_free_radiative_transition_coefficient_(
                wave=wave,
                J=J[mask, k],
                alpha=alpha,
                Te=atmos.Te[k],
                nk_by_ni_LTE=nj_by_ni_Cont[k, kC],
            )
            Rik[k, kC] = res[0]
            Rki_stim[k, kC] = res[1]
            Rki_spon[k, kC] = res[2]
    return Rik, Rki_stim, Rki_spon


# -------------------------------------------------------------------------------
# numba optimization : per-iteration kernels compile unconditionally
# -------------------------------------------------------------------------------

continuum_sweep_ = nb_njit(**NB_NJIT_KWGS)(continuum_sweep_)
