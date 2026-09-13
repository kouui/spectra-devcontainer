# -------------------------------------------------------------------------------
# the MALI iteration on the unified axis: one formal sweep over every column
#
# structure of every iteration (the only tier that sees populations):
#     populations -> per-transition coefficients (line chi/S, continuum n_dag)
#     -> per column: chi/eta summed over the transitions covering it + background
#        -> per-mu Feautrier solve -> J, Psi (angle-averaged), chi_tot
#     -> per transition: Jbar/Lstar over its window, b-f rates over its window
#     -> preconditioned update -> convergence check
# the formal solve is organized by COLUMN (every axis wavelength solved exactly
# once, with all of its opacity) and the rate integrals by WINDOW; the sweep
# stores J/Psi/chi_tot on the full axis so the window integrals run outside
# the column loop (no reduction across transitions inside it).
# the sweep kernels are compiled unconditionally (hot path); the outer
# convergence loop is interpreted -- it does bookkeeping, not arithmetic.
# -------------------------------------------------------------------------------

from collections import namedtuple as _namedtuple

import numpy as _numpy

from ....Atomic import LTELib as _LTELib
from ....Atomic import SEsolver as _SEsolver
from ....Atomic import emisivity as _emisivity
from ....Atomic import extinction as _extinction
from ....ImportAll import *
from ....Math import GaussLeg as _GaussLeg
from ....RadiativeTransfer import Feautrier as _Feautrier
from . import GlobalMesh as _GlobalMesh
from . import Structs as _Structs

# cap on the diagonal scattering-operator weight Psi*sigma/chi (see
# mali_multilevel_): 1e3 overrelaxation at most, rounding amplified by 1e3
_W_SCA_MAX: T_FLOAT = 1.0 - 1.0e-3

# jitted callers need jitted callees; production compiles these only under
# CFG._IS_JIT, so bind compiled references here otherwise.
if CFG._IS_JIT:
    _formal_rh_ = _Feautrier.formal_improved_RH_
    _set_matrixR_ = _SEsolver.set_matrixR_
    _set_matrixC_ = _SEsolver.set_matrixC_
    _solve_SE_ = _SEsolver.solve_SE_
else:
    _formal_rh_ = nb_njit(**NB_NJIT_KWGS)(_Feautrier.formal_improved_RH_)
    _set_matrixR_ = nb_njit(**NB_NJIT_KWGS)(_SEsolver.set_matrixR_)
    _set_matrixC_ = nb_njit(**NB_NJIT_KWGS)(_SEsolver.set_matrixC_)
    _solve_SE_ = nb_njit(**NB_NJIT_KWGS)(_SEsolver.solve_SE_)

# the production b-b opacity/emissivity conventions, compiled from the raw
# python functions behind their numpy.vectorize wrappers -- reused rather than
# re-derived so the convention cannot drift from Atomic/{extinction,emisivity}.
_bb_extinction_ = nb_njit(**NB_NJIT_KWGS)(_extinction.bb_extinction_.pyfunc)  # type: ignore[attr-defined]
_bb_emissivity_ = nb_njit(**NB_NJIT_KWGS)(_emisivity.bb_emissivity_.pyfunc)  # type: ignore[attr-defined]


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
    oracle this kernel is tested against; re-stated here because the
    production function is compiled only under the global JIT flag and its
    nested trapezoid cannot be bound from a jitted caller otherwise.

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


def bf_rates_(
    J: T_ARRAY,
    mesh: _GlobalMesh.Global_Mesh,
    pre: _Structs.MALI_Precompute,
    atom: _Structs.Toy_Atom,
    atmos: _Structs.Atmos1D,
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    """bf_rates_kernel_ on the structs (see there).

    Output:
        Rik, Rki_stim, Rki_spon: (ND, nCont), [s^-1]
    """
    return bf_rates_kernel_(
        J, mesh.wl, mesh.Nblue, mesh.span, mesh.win_off, pre.alpha_win, pre.row_cont0, atom.nLine,
        atmos.Te, pre.nj_by_ni,
    )  # fmt: skip


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


MALIml_Result = _namedtuple(
    "MALIml_Result", ["n", "S_line", "Jbar", "Lstar", "J", "niter", "converged", "dn_history", "dJ_history"]
)


def mali_multilevel_(
    atom: _Structs.Toy_Atom,
    atmos: _Structs.Atmos1D,
    mesh: _GlobalMesh.Global_Mesh,
    pre: _Structs.MALI_Precompute,
    n_angle: T_INT = 4,
    tol: T_FLOAT = 1.0e-8,
    itmax: T_INT = 2000,
    use_lstar: T_BOOL = True,
    lstar_scale: T_FLOAT = 1.0,
    n_init: T_ARRAY | None = None,
    tol_J: T_FLOAT = 1.0e-6,
    J_init: T_ARRAY | None = None,
) -> MALIml_Result:
    """Multilevel MALI driver on the unified axis.

    interpreted orchestration: unpacks the structs into the plain arrays the
    jitted kernels take, starts from LTE populations (or n_init, (ND, nLevel)
    normalized -- e.g. a reference solution for a fixed-point check), and
    iterates
        coefficients -> unified sweep -> window rates -> per-depth SE
    until BOTH max|dn| < tol (absolute, populations normalized to 1 per
    depth) and max|dJ|/max|J| (relative, per column) < tol_J, so a
    still-moving radiation field cannot end the loop early. the two
    tolerances are separate on purpose: a trace level at 1e-11 moves by a
    few percent under an absolute dn of 1e-12, and J reports that relative
    motion faithfully -- one tolerance for both would never be met. after the
    loop one more sweep is run on the FINAL populations (no SE update, not
    counted in niter) so that the returned J, Jbar, Lstar and S_line are the
    radiation field OF the returned n, not of the state one update earlier.
    coherent scattering (the build-once background sigma) re-emits the
    field of the previous sweep, J_dag, which starts at B_lambda(Te) (or
    J_init). the sweep's J is then corrected with the diagonal operator for
    the scattering source,
        J = (J_sweep - Psi*(sigma/chi)*J_dag) / (1 - Psi*sigma/chi),
    the local ALI of Olson, Auer & Buchler for a linear scattering term.
    RH Lambda-iterates that term instead and stops on the populations;
    on FALC that leaves its 122-160 nm field a factor 2 short of the fixed
    point, and plain Lambda-iteration here reported converged=True at
    tol_J = 1e-6 with J 36% short at 128 nm (sigma/chi = 0.9998, tau = 75).
    the fixed point does not depend on the start or the operator, only the
    iteration count does; dJ < tol_J then certifies J_dag has stopped
    moving. the corrected J feeds the rates: at the fixed point it equals
    the sweep's J, and on the way it is the better estimate; its derivative
    with respect to the thermal source is Psi/(1 - Psi*sigma/chi), which is
    what the line operator Lstar is built from, so Lstar stays the exact
    diagonal of the corrected system. the weight is capped below 1: an
    operator smaller than the true diagonal only slows the acceleration,
    while the uncapped division amplifies rounding by 1/(1 - w) and hits
    0/0 where Psi rounds to 1 (a thick pure-scattering column); the weight
    is also floored at 0 for a column whose net extinction went negative
    (population inversion), where no acceleration is attempted.

    Input:
        atom: Structs.Toy_Atom
        atmos: Structs.Atmos1D
        mesh: GlobalMesh.Global_Mesh (windows for every transition)
        pre: Structs.MALI_Precompute
        use_lstar: (,), False -> plain (preconditioner-free) iteration
        lstar_scale: (,), deliberate operator mis-scaling (tests only)
        tol_J: (,), relative tolerance on the per-column change of J
        J_init: (Nspect, ND), starting field for the scattering emissivity
            (e.g. a warm start); None uses B_lambda(Te)

    Output: MALIml_Result(n, S_line, Jbar, Lstar, J, niter, converged, dn_history, dJ_history)
    """
    mus, wmus = _GaussLeg.gauss_quad_coe_(0.0, 1.0, n_angle)
    nLine, nCont = atom.nLine, atom.nCont
    nTran = nLine + nCont
    idxI = _numpy.empty(nTran, dtype=DT_NB_INT)
    idxJ = _numpy.empty(nTran, dtype=DT_NB_INT)
    idxI[:nLine] = atom.Line["idxI"][:]
    idxJ[:nLine] = atom.Line["idxJ"][:]
    if nCont > 0:
        idxI[nLine:] = atom.Cont["idxI"][:]
        idxJ[nLine:] = atom.Cont["idxJ"][:]
    idxI_c = _numpy.ascontiguousarray(idxI[nLine:])
    idxJ_c = _numpy.ascontiguousarray(idxJ[nLine:])
    nj_by_ni_c = _numpy.ascontiguousarray(pre.nj_by_ni[:, nLine:])

    w0 = _numpy.ascontiguousarray(atom.Line["w0"][:])
    Aji = _numpy.ascontiguousarray(atom.Line["AJI"][:])
    Bji = _numpy.ascontiguousarray(atom.Line["BJI"][:])
    Bij = _numpy.ascontiguousarray(atom.Line["BIJ"][:])

    scale = lstar_scale if use_lstar else 0.0
    n = pre.n_LTE.copy() if n_init is None else _numpy.ascontiguousarray(n_init, dtype=DT_NB_FLOAT).copy()
    if J_init is None:
        J = _numpy.empty((mesh.wl.shape[0], atmos.ND), dtype=DT_NB_FLOAT)
        for k in range(atmos.ND):
            J[:, k] = _LTELib.planck_cm_(mesh.wl[:], atmos.Te[k])
    else:
        J = _numpy.ascontiguousarray(J_init, dtype=DT_NB_FLOAT).copy()
    S_line = _numpy.zeros((nLine, atmos.ND), dtype=DT_NB_FLOAT)
    Jbar = _numpy.zeros_like(S_line)
    Lstar = _numpy.zeros_like(S_line)
    dn_history = []
    dJ_history = []
    niter = 0
    converged = False

    def radiation_step(n_cur, J_dag):
        n_abs = n_cur * atmos.Nt[:, None]
        chi_int, S_cur = line_coefficients_(n_abs, w0, Aji, Bji, Bij, idxI[:nLine], idxJ[:nLine])
        n_low, n_dag = continuum_coefficients_(n_abs, nj_by_ni_c, idxI_c, idxJ_c)
        J_cur, Psi, chi_tot = unified_sweep_(
            atmos.Z, mesh.wl, mesh.col_ptr, mesh.col_tran, mesh.col_row, nLine,
            w0, pre.phi, chi_int, S_cur, pre.alpha_win, pre.row_cont0, n_low, n_dag,
            pre.exp_hnu_kT, pre.twohc2_wl5, pre.bg_chi, pre.bg_eta, pre.bg_sca, J_dag, pre.hn_bottom, mus, wmus,
        )  # fmt: skip
        w_sca = _numpy.clip(Psi * pre.bg_sca / chi_tot, 0.0, _W_SCA_MAX)
        J_cur = (J_cur - w_sca * J_dag) / (1.0 - w_sca)
        Jbar_cur, Lstar_cur = line_rates_(
            J_cur, Psi / (1.0 - w_sca), chi_tot, mesh.wl, mesh.Nblue, mesh.span, mesh.win_off,
            pre.phi, pre.weight, pre.wphi, w0, chi_int,
        )  # fmt: skip
        return J_cur, S_cur, Jbar_cur, Lstar_cur, bf_rates_(J_cur, mesh, pre, atom, atmos)

    for it in range(1, itmax + 1):
        niter = it
        J_new, S_line, Jbar, Lstar, (Rik, Rki_stim, Rki_spon) = radiation_step(n, J)
        n_new = update_populations_(
            Jbar, Lstar, S_line, Aji, Bji, Bij, idxI, idxJ,
            pre.Cij_coe, pre.Cji_coe, Rik, Rki_stim, Rki_spon,
            atmos.Ne, atom.nLevel, scale,
        )  # fmt: skip
        dn = float(_numpy.abs(n_new - n).max())
        # per-column relative change; an all-zero column (no radiation at all)
        # cannot move and must not turn the criterion into 0/0
        J_scale = _numpy.abs(J_new).max(axis=1)
        live = J_scale > 0.0
        dJ = float((_numpy.abs(J_new - J).max(axis=1)[live] / J_scale[live]).max()) if live.any() else 0.0
        dn_history.append(dn)
        dJ_history.append(dJ)
        n = n_new
        J = J_new
        if dn < tol and dJ < tol_J:
            converged = True
            break
    # diagnostics consistent with the returned populations (and with the
    # last field as J_dag, which dJ < tol_J has certified stationary)
    J, S_line, Jbar, Lstar, _ = radiation_step(n, J)
    return MALIml_Result(
        n=n, S_line=S_line, Jbar=Jbar, Lstar=Lstar, J=J, niter=niter, converged=converged,
        dn_history=_numpy.asarray(dn_history), dJ_history=_numpy.asarray(dJ_history),
    )  # fmt: skip


MALI2lv_Result = _namedtuple("MALI2lv_Result", ["S", "Jbar", "Lstar", "niter", "dS_history"])


def mali_two_level_(
    Z: T_ARRAY,
    chi0: T_ARRAY,
    eps: T_ARRAY,
    B: T_ARRAY,
    wl_win: T_ARRAY,
    phi_win: T_ARRAY,
    weight_win: T_ARRAY,
    wphi_line: T_ARRAY,
    n_angle: T_INT = 4,
    tol: T_FLOAT = 1.0e-8,
    itmax: T_INT = 20000,
    use_lstar: T_BOOL = True,
    lstar_scale: T_FLOAT = 1.0,
) -> MALI2lv_Result:
    """Two-level-atom MALI in (eps, B) form, run through the unified sweep:
    one line, no continua, no background, no scattering, the whole axis =
    the line window.

        S = (1 - eps) * Jbar + eps * B,
    every oracle (sqrt(eps) law, Lambda-iteration fixed point) is analytic.
    populations enter only through chi0 (the profile-integrated opacity at
    line center), held fixed. the preconditioned update is

        S_new = ((1-eps)*(Jbar - Lstar*S_old) + eps*B) / (1 - (1-eps)*Lstar)

    use_lstar=False degenerates to plain Lambda-iteration: the convergence
    crawls but the fixed point is identical -- Lstar (even a deliberately
    scaled one, see lstar_scale) cancels at S_new = S_old. the returned
    Jbar/Lstar come from one extra sweep on the returned S (not counted).

    Input:
        Z: (ND,), depth, [cm], ascending, Z[0] = 0
        chi0: (ND,), profile-integrated line opacity, [cm^-1 * cm]
        eps: (ND,), photon destruction probability
        B: (ND,), Planck function at line center (also the lower boundary)
        wl_win, phi_win, weight_win, wphi_line: the window (= axis) and its
            profile table (nw,), (nw, ND), (nw,), (ND,)
        n_angle, tol, itmax, use_lstar, lstar_scale: see mali_multilevel_

    Output: MALI2lv_Result(S, Jbar, Lstar, niter, dS_history)
    """
    ND = Z.shape[0]
    nw = wl_win.shape[0]
    mus, wmus = _GaussLeg.gauss_quad_coe_(0.0, 1.0, n_angle)
    # a one-line axis: every column belongs to line 0, row = column
    w0 = _numpy.array([wl_win[nw // 2]], dtype=DT_NB_FLOAT)
    Nblue = _numpy.zeros(1, dtype=DT_NB_INT)
    span = _numpy.array([nw], dtype=DT_NB_INT)
    win_off = _numpy.array([[0, nw]], dtype=DT_NB_INT)
    col_ptr = _numpy.arange(nw + 1, dtype=DT_NB_INT)
    col_tran = _numpy.zeros(nw, dtype=DT_NB_INT)
    col_row = _numpy.arange(nw, dtype=DT_NB_INT)
    chi_int = _numpy.ascontiguousarray(chi0[None, :], dtype=DT_NB_FLOAT)
    n_c = _numpy.empty((0, ND), dtype=DT_NB_FLOAT)
    alpha_win = _numpy.empty(0, dtype=DT_NB_FLOAT)
    ones = _numpy.ones((nw, ND), dtype=DT_NB_FLOAT)  # exp_hnu_kT unused without continua
    zeros_ax = _numpy.zeros(nw, dtype=DT_NB_FLOAT)
    zeros_bg = _numpy.zeros((nw, ND), dtype=DT_NB_FLOAT)
    hn_bottom = _numpy.full(nw, B[ND - 1], dtype=DT_NB_FLOAT)
    wphi = _numpy.ascontiguousarray(wphi_line[:, None], dtype=DT_NB_FLOAT)

    S = B.copy()
    dS_history = []
    niter = 0
    Jbar = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
    Lstar = _numpy.zeros(ND, dtype=DT_NB_FLOAT)

    def sweep(S_cur):
        S_line = _numpy.ascontiguousarray(S_cur[None, :])
        J, Psi, chi_tot = unified_sweep_(
            Z, wl_win, col_ptr, col_tran, col_row, 1, w0, phi_win, chi_int, S_line, alpha_win, nw,
            n_c, n_c, ones, zeros_ax, zeros_bg, zeros_bg, zeros_bg, zeros_bg, hn_bottom, mus, wmus,
        )  # fmt: skip
        Jbar2, Lstar2 = line_rates_(
            J, Psi, chi_tot, wl_win, Nblue, span, win_off, phi_win, weight_win, wphi, w0, chi_int
        )
        return Jbar2[0, :], Lstar2[0, :]

    for it in range(1, itmax + 1):
        niter = it
        Jbar, Lstar = sweep(S)
        L = lstar_scale * Lstar if use_lstar else _numpy.zeros(ND, dtype=DT_NB_FLOAT)
        S_new = ((1.0 - eps) * (Jbar - L * S) + eps * B) / (1.0 - (1.0 - eps) * L)
        dS = float(_numpy.abs(S_new - S).max() / _numpy.abs(S_new).max())
        dS_history.append(dS)
        S = S_new
        if dS < tol:
            break
    Jbar, Lstar = sweep(S)
    return MALI2lv_Result(S=S, Jbar=Jbar, Lstar=Lstar, niter=niter, dS_history=_numpy.asarray(dS_history))


# -------------------------------------------------------------------------------
# numba optimization : per-iteration kernels compile unconditionally
# -------------------------------------------------------------------------------

line_coefficients_ = nb_njit(**NB_NJIT_KWGS)(line_coefficients_)
continuum_coefficients_ = nb_njit(**NB_NJIT_KWGS)(continuum_coefficients_)
unified_sweep_ = nb_njit(**NB_NJIT_KWGS)(unified_sweep_)
line_rates_ = nb_njit(**NB_NJIT_KWGS)(line_rates_)
bf_rates_kernel_ = nb_njit(**NB_NJIT_KWGS)(bf_rates_kernel_)
update_populations_ = nb_njit(**NB_NJIT_KWGS)(update_populations_)
