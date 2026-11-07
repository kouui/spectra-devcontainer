# -------------------------------------------------------------------------------
# per-iteration kernels of the MALI iteration on the unified axis
#
# structure of every iteration (the only tier that sees populations):
#     populations -> per-transition coefficients (line chi/S, continuum n_dag)
#     -> per column: chi/eta summed over the transitions covering it + background
#        -> per-mu Feautrier solve -> J, Psi (angle-averaged), chi_tot
#     -> per transition: Jbar/Lstar over its window, b-f rates over its window
#     -> preconditioned update
# the formal solve is organized by COLUMN (every axis wavelength solved exactly
# once, with all of its opacity) and the rate integrals by WINDOW; the sweep
# stores J/Psi/chi_tot on the full axis so the window integrals run outside
# the column loop (no reduction across transitions inside it).
# the kernels take plain arrays only, so this module imports no input struct.
# -------------------------------------------------------------------------------

import numpy as _numpy

from ...Atomic import SEsolver as _SEsolver
from ...Atomic import emisivity as _emisivity
from ...Atomic import extinction as _extinction
from ...ImportAll import *
from .. import Feautrier as _Feautrier

# cap on the diagonal scattering-operator weight Psi*sigma/chi (see
# Function/MALI/Loop.mali_multilevel_): 1e3 overrelaxation at most, rounding amplified by 1e3
W_SCA_MAX: T_FLOAT = 1.0 - 1.0e-3

# the kernel bodies call these names; keeping the aliases keeps the bodies identical to the frozen
# Experimental v2 copy they were relocated from.
_formal_rh_ = _Feautrier.formal_improved_RH_
_set_matrixR_ = _SEsolver.set_matrixR_
_set_matrixC_ = _SEsolver.set_matrixC_
_solve_SE_ = _SEsolver.solve_SE_

# the production b-b opacity/emissivity conventions (numba DUFuncs, callable from
# njit code) -- reused rather than re-derived so the convention cannot drift from
# Atomic/{extinction,emisivity}.
_bb_extinction_ = _extinction.bb_extinction_
_bb_emissivity_ = _emisivity.bb_emissivity_


@nb_njit(**NB_NJIT_KWGS)
def line_coefficients_(
    n_abs: T_ARRAY,
    w0: T_ARRAY,
    Aji: T_ARRAY,
    Bji: T_ARRAY,
    Bij: T_ARRAY,
    idxI: T_ARRAY,
    idxJ: T_ARRAY,
) -> T_TUPLE[T_ARRAY, T_ARRAY]:
    """Per line and depth, the population-dependent scalars the column
    assembly multiplies by the profile: the profile-integrated opacity at line
    center and the (CRD, wavelength-independent) line source function.

        chi = h*nu/(4*pi) * (ni*Bij*phi - nj*Bji*psi),   psi = phi (CRD)
        eta = h*nu/(4*pi) * nj*Aji * psi
    evaluated with psi = phi = 1 (bb_extinction_ / bb_emissivity_ bindings).

    Input:
        n_abs: (ND, nLevel), ABSOLUTE populations, [cm^-3]
        w0, Aji, Bji, Bij, idxI, idxJ: (nLine,), line coefficients

    Output:
        chi_int: (nLine, ND), [cm^-1 * cm]
        S_line: (nLine, ND)
    """
    nLine = w0.shape[0]
    ND = n_abs.shape[0]
    chi_int = _numpy.empty((nLine, ND), dtype=DT_NB_FLOAT)
    S_line = _numpy.empty((nLine, ND), dtype=DT_NB_FLOAT)
    for kL in range(nLine):
        for k in range(ND):
            ni = n_abs[k, idxI[kL]]
            nj = n_abs[k, idxJ[kL]]
            chi_int[kL, k] = _bb_extinction_(w0[kL], Bji[kL], Bij[kL], nj, ni, 1.0, 1.0)
            S_line[kL, k] = _bb_emissivity_(w0[kL], Aji[kL], nj) / chi_int[kL, k]
    return chi_int, S_line


@nb_njit(**NB_NJIT_KWGS)
def continuum_coefficients_(
    n_abs: T_ARRAY,
    nj_by_ni: T_ARRAY,
    idxI: T_ARRAY,
    idxJ: T_ARRAY,
) -> T_TUPLE[T_ARRAY, T_ARRAY]:
    """Per continuum and depth, the populations the NLTE b-f column assembly
    needs: the lower level and the LTE-consistent lower population implied by
    the upper (ion) level, n_dag = n_upper / (nj/ni)_LTE.

    Input:
        n_abs: (ND, nLevel), ABSOLUTE populations, [cm^-3]
        nj_by_ni: (ND, nCont), LTE ratio per continuum
        idxI, idxJ: (nCont,), continuum level indices

    Output:
        n_low, n_dag: (nCont, ND), [cm^-3]
    """
    nCont = idxI.shape[0]
    ND = n_abs.shape[0]
    n_low = _numpy.empty((nCont, ND), dtype=DT_NB_FLOAT)
    n_dag = _numpy.empty((nCont, ND), dtype=DT_NB_FLOAT)
    for kC in range(nCont):
        for k in range(ND):
            n_low[kC, k] = n_abs[k, idxI[kC]]
            n_dag[kC, k] = n_abs[k, idxJ[kC]] / nj_by_ni[k, kC]
    return n_low, n_dag


@nb_njit(**NB_NJIT_KWGS)
def unified_sweep_(  # noqa: C901
    Z: T_ARRAY,
    wl: T_ARRAY,
    col_ptr: T_ARRAY,
    col_tran: T_ARRAY,
    col_row: T_ARRAY,
    nLine: T_INT,
    w0: T_ARRAY,
    phi: T_ARRAY,
    chi_int: T_ARRAY,
    S_line: T_ARRAY,
    alpha_win: T_ARRAY,
    row_cont0: T_INT,
    n_low: T_ARRAY,
    n_dag: T_ARRAY,
    exp_hnu_kT: T_ARRAY,
    twohc2_wl5: T_ARRAY,
    bg_chi: T_ARRAY,
    bg_eta: T_ARRAY,
    sigma: T_ARRAY,
    J_dag: T_ARRAY,
    hn_bottom: T_ARRAY,
    mus: T_ARRAY,
    wmus: T_ARRAY,
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    """One formal sweep over the whole axis: every column assembled from ALL
    transitions covering it plus the thermal background and the coherent
    scattering, solved once per angle.

    Per column iw and depth k:
        line t (CRD):  chi_l = (w0/wl) * chi_int[t,k] * phi[row,k]
                       eta_l = chi_l * S_line[t,k]
        continuum c:   stim  = n_dag[c,k] * exp(-h nu/kT)
                       chi_c = alpha * (n_low[c,k] - stim)
                       eta_c = alpha * 2hc^2/wl^5 * stim
        chi_tot = sum + bg_chi + sigma
        S_col   = (sum eta + bg_eta + sigma * J_dag) / chi_tot
    the (w0/wl) factor mirrors the production per-wavelength h*nu convention
    for a coefficient evaluated at line center. the scattering emissivity
    uses the PREVIOUS iteration's field J_dag (RH's treatment): it enters as
    a fixed source, so Psi stays the derivative with respect to the thermal
    part of S_col only and the caller must not put sigma*J_dag into any
    operator correction. sigma is counted once, here -- bg_chi is thermal.

    Input:
        Z: (ND,), depth, [cm], ascending, Z[0] = 0
        wl, col_ptr, col_tran, col_row: the axis and its CSR membership
        nLine: transitions t < nLine are lines, the rest continua
        w0: (nLine,); phi: (nWinLine, ND); chi_int, S_line: (nLine, ND)
        alpha_win: (nWinCont,); row_cont0: first continuum row
        n_low, n_dag: (nCont, ND)
        exp_hnu_kT: (Nspect, ND); twohc2_wl5, hn_bottom: (Nspect,)
        bg_chi, bg_eta: (Nspect, ND), thermal background
        sigma: (Nspect, ND), coherent-scattering extinction, [cm^-1]
        J_dag: (Nspect, ND), mean intensity the scattering re-emits
        mus, wmus: angle quadrature on (0, 1), weights sum to 1

    Output:
        J: (Nspect, ND), mean intensity, cm-base
        Psi: (Nspect, ND), angle-averaged diagonal operator d J / d S_col
        chi_tot: (Nspect, ND), total extinction, [cm^-1]
    """
    Nspect = wl.shape[0]
    ND = Z.shape[0]
    n_mu = mus.shape[0]

    J = _numpy.zeros((Nspect, ND), dtype=DT_NB_FLOAT)
    Psi = _numpy.zeros((Nspect, ND), dtype=DT_NB_FLOAT)
    chi_tot = _numpy.empty((Nspect, ND), dtype=DT_NB_FLOAT)
    eta = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    S_col = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    tau = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    for iw in range(Nspect):
        for k in range(ND):
            chi_tot[iw, k] = bg_chi[iw, k] + sigma[iw, k]
            eta[k] = bg_eta[iw, k] + sigma[iw, k] * J_dag[iw, k]
        for m in range(col_ptr[iw], col_ptr[iw + 1]):
            t = col_tran[m]
            row = col_row[m]
            if t < nLine:
                nu_ratio = w0[t] / wl[iw]
                for k in range(ND):
                    chi_l = nu_ratio * chi_int[t, k] * phi[row, k]
                    chi_tot[iw, k] += chi_l
                    eta[k] += chi_l * S_line[t, k]
            else:
                kC = t - nLine
                a = alpha_win[row - row_cont0]
                for k in range(ND):
                    stim = n_dag[kC, k] * exp_hnu_kT[iw, k]
                    chi_tot[iw, k] += a * (n_low[kC, k] - stim)
                    eta[k] += a * twohc2_wl5[iw] * stim
        for k in range(ND):
            S_col[k] = eta[k] / chi_tot[iw, k]
        tau[0] = 0.0
        for k in range(1, ND):
            tau[k] = tau[k - 1] + 0.5 * (chi_tot[iw, k - 1] + chi_tot[iw, k]) * (Z[k] - Z[k - 1])
        for im in range(n_mu):
            res = _formal_rh_(tau, S_col, mus[im], 0.0, 0.0, 0.0, hn_bottom[iw], E_FEAUTRIER_ORDER.SECOND, True)
            for k in range(ND):
                J[iw, k] += wmus[im] * res.j[k]
                Psi[iw, k] += wmus[im] * res.Psi[k]
    return J, Psi, chi_tot


@nb_njit(**NB_NJIT_KWGS)
def line_rates_(
    J: T_ARRAY,
    Psi: T_ARRAY,
    chi_tot: T_ARRAY,
    wl: T_ARRAY,
    Nblue: T_ARRAY,
    span: T_ARRAY,
    win_off: T_ARRAY,
    phi: T_ARRAY,
    weight: T_ARRAY,
    wphi: T_ARRAY,
    w0: T_ARRAY,
    chi_int: T_ARRAY,
) -> T_TUPLE[T_ARRAY, T_ARRAY]:
    """Per line, the profile-weighted window integrals of the solved field:

        Jbar[k]  = sum_iw weight*phi * J(iw,k)                        / wphi[k]
        Lstar[k] = sum_iw weight*phi * Psi(iw,k) * chi_l/chi_tot(iw,k) / wphi[k]

    the division by wphi (the numerical profile norm on this window) cancels
    the quadrature's profile-area error identically (RH's wphi trick). Lstar
    is the exact d Jbar / d S_line of the discrete system with every OTHER
    source in the column frozen: chi_l/chi_tot is d S_col / d S_line of the
    mixed column. this is the self-term operator only -- the cross terms
    between transitions sharing a level (full RH92) are deliberately not
    formed; they change the convergence rate, never the fixed point.

    Input:
        J, Psi, chi_tot: (Nspect, ND), from unified_sweep_
        wl, Nblue, span, win_off: axis and line windows (first nLine entries)
        phi, weight, wphi: line profile tables; w0, chi_int: line coefficients

    Output:
        Jbar, Lstar: (nLine, ND)
    """
    nLine = w0.shape[0]
    ND = J.shape[1]
    Jbar = _numpy.zeros((nLine, ND), dtype=DT_NB_FLOAT)
    Lstar = _numpy.zeros((nLine, ND), dtype=DT_NB_FLOAT)
    for kL in range(nLine):
        for i in range(span[kL]):
            iw = Nblue[kL] + i
            row = win_off[kL, 0] + i
            nu_ratio = w0[kL] / wl[iw]
            for k in range(ND):
                coe = weight[row] * phi[row, k] / wphi[k, kL]
                chi_l = nu_ratio * chi_int[kL, k] * phi[row, k]
                Jbar[kL, k] += coe * J[iw, k]
                Lstar[kL, k] += coe * Psi[iw, k] * chi_l / chi_tot[iw, k]
    return Jbar, Lstar


@nb_njit(**NB_NJIT_KWGS)
def bf_rates_kernel_(
    J: T_ARRAY,
    wl: T_ARRAY,
    Nblue: T_ARRAY,
    span: T_ARRAY,
    win_off: T_ARRAY,
    alpha_win: T_ARRAY,
    row_cont0: T_INT,
    nLine: T_INT,
    Te: T_ARRAY,
    nj_by_ni: T_ARRAY,
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    """Per continuum and depth, the radiative rates from the solved J on the
    continuum's own window (the closed range on the axis, ascending):

        R_ik      = 4 pi int alpha/(h nu) J dlambda
        R_ki,stim = (ni/nk)_LTE 4 pi int alpha/(h nu) J e^(-h nu/kT) dlambda
        R_ki,spon = (ni/nk)_LTE 4 pi int alpha/(h nu) (2hc^2/lambda^5) e^(-h nu/kT) dlambda

    trapezoid on the window points -- the same formulas as the production
    PhotoIonize.bound_free_radiative_transition_coefficient_, which is the
    oracle this kernel is tested against.

    Input:
        J: (Nspect, ND); wl: (Nspect,); Nblue, span, win_off: all transitions
        alpha_win: (nWinCont,), row_cont0: first continuum row
        nLine: (,); Te: (ND,); nj_by_ni: (ND, nTran), LTE ratios (continua at nLine:)

    Output:
        Rik, Rki_stim, Rki_spon: (ND, nCont), [s^-1]
    """
    ND = Te.shape[0]
    nCont = Nblue.shape[0] - nLine
    Rik = _numpy.zeros((ND, nCont), dtype=DT_NB_FLOAT)
    Rki_stim = _numpy.zeros((ND, nCont), dtype=DT_NB_FLOAT)
    Rki_spon = _numpy.zeros((ND, nCont), dtype=DT_NB_FLOAT)
    hc = CST.h_ * CST.c_
    twohc2 = 2.0 * CST.h_ * CST.c_ * CST.c_
    for kC in range(nCont):
        t = nLine + kC
        i0 = Nblue[t]
        nw = span[t]
        r0 = win_off[t, 0] - row_cont0
        for k in range(ND):
            s_ik = 0.0
            s_stim = 0.0
            s_spon = 0.0
            # trapezoid: f[i] weights 0.5*(x[i+1]-x[i-1]) inside, half-intervals at the ends
            for i in range(nw):
                iw = i0 + i
                lam = wl[iw]
                a_hv = alpha_win[r0 + i] * lam / hc  # alpha / (h nu)
                expo = _numpy.exp(-hc / (lam * CST.k_ * Te[k]))
                if i == 0:
                    w = 0.5 * (wl[iw + 1] - lam)
                elif i == nw - 1:
                    w = 0.5 * (lam - wl[iw - 1])
                else:
                    w = 0.5 * (wl[iw + 1] - wl[iw - 1])
                s_ik += w * a_hv * J[iw, k]
                s_stim += w * a_hv * J[iw, k] * expo
                s_spon += w * a_hv * twohc2 / lam**5 * expo
            factor = 1.0 / nj_by_ni[k, t]
            Rik[k, kC] = 4.0 * CST.pi_ * s_ik
            Rki_stim[k, kC] = factor * 4.0 * CST.pi_ * s_stim
            Rki_spon[k, kC] = factor * 4.0 * CST.pi_ * s_spon
    return Rik, Rki_stim, Rki_spon


@nb_njit(**NB_NJIT_KWGS)
def update_populations_(
    Jbar: T_ARRAY,
    Lstar: T_ARRAY,
    S_line: T_ARRAY,
    Aji: T_ARRAY,
    Bji: T_ARRAY,
    Bij: T_ARRAY,
    idxI: T_ARRAY,
    idxJ: T_ARRAY,
    Cij_coe: T_ARRAY,
    Cji_coe: T_ARRAY,
    Rik: T_ARRAY,
    Rki_stim: T_ARRAY,
    Rki_spon: T_ARRAY,
    Ne: T_ARRAY,
    nLevel: T_INT,
    lstar_scale: T_FLOAT,
) -> T_ARRAY:
    """Preconditioned SE solve, depth by depth.

    Rybicki & Hummer (1992) preconditioning with a diagonal operator is
    form-preserving: per line the effective rates are
        Rji_spon = Aji * (1 - Lstar)
        Jbar_eff = Jbar - Lstar * S_line          (both rate directions)
    so the standard rate-matrix assembly (SEsolver.set_matrixR_/set_matrixC_)
    consumes them unchanged. the b-f rates enter unpreconditioned (plain
    Lambda-iteration on the continua).

    Lstar is diagonal in depth, so each depth solves an independent
    nLevel x nLevel system: the population update stays LOCAL, which is the
    entire cost advantage of MALI over complete linearization.

    Input:
        Jbar, Lstar, S_line: (nLine, ND)
        Aji, Bji, Bij: (nLine,); idxI, idxJ: (nTran,), lines then continua
        Cij_coe, Cji_coe: (ND, nTran), collisional coefficients
        Rik, Rki_stim, Rki_spon: (ND, nCont), b-f rates from the solved field
        Ne: (ND,); nLevel: (,)
        lstar_scale: (,), deliberate operator mis-scaling (tests only; 1.0 normally)

    Output:
        n_new: (ND, nLevel), normalized populations
    """
    nLine = Aji.shape[0]
    nCont = Rik.shape[1]
    nTran = nLine + nCont
    ND = Ne.shape[0]

    n_new = _numpy.empty((ND, nLevel), dtype=DT_NB_FLOAT)
    Rji_spon = _numpy.empty(nTran, dtype=DT_NB_FLOAT)
    Rji_stim = _numpy.empty(nTran, dtype=DT_NB_FLOAT)
    Rij = _numpy.empty(nTran, dtype=DT_NB_FLOAT)
    for k in range(ND):
        for kL in range(nLine):
            L = lstar_scale * Lstar[kL, k]
            Jbar_eff = Jbar[kL, k] - L * S_line[kL, k]
            Rji_spon[kL] = Aji[kL] * (1.0 - L)
            Rji_stim[kL] = Bji[kL] * Jbar_eff
            Rij[kL] = Bij[kL] * Jbar_eff
        for kC in range(nCont):
            Rji_spon[nLine + kC] = Rki_spon[k, kC]
            Rji_stim[nLine + kC] = Rki_stim[k, kC]
            Rij[nLine + kC] = Rik[k, kC]
        Rmat = _numpy.zeros((nLevel, nLevel), dtype=DT_NB_FLOAT)
        Cmat = _numpy.zeros((nLevel, nLevel), dtype=DT_NB_FLOAT)
        _set_matrixR_(Rmat, Rji_spon, Rji_stim, Rij, idxI, idxJ)
        _set_matrixC_(Cmat, Cji_coe[k, :], Cij_coe[k, :], idxI, idxJ, Ne[k])
        n_new[k, :] = _solve_SE_(Rmat, Cmat)
    return n_new
