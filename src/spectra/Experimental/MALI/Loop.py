# -------------------------------------------------------------------------------
# the MALI iteration: formal sweep, Jbar/Lstar accumulation, source update
#
# structure of every iteration (the only tier that sees populations):
#     S / chi  ->  per-(wavelength, mu) Feautrier solve  ->  Jbar, Lstar
#     ->  preconditioned update  ->  convergence check
# the sweep kernels are compiled unconditionally (hot path); the outer
# convergence loop is interpreted -- it does bookkeeping, not arithmetic.
# -------------------------------------------------------------------------------

from collections import namedtuple as _namedtuple

import numpy as _numpy

from ...Atomic import SEsolver as _SEsolver
from ...Atomic import emisivity as _emisivity
from ...Atomic import extinction as _extinction
from ...ImportAll import *
from ...Math import GaussLeg as _GaussLeg
from ...RadiativeTransfer import Feautrier as _Feautrier
from . import GlobalMesh as _GlobalMesh
from . import Structs as _Structs

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


def two_level_sweep_(
    Z: T_ARRAY,
    chi0: T_ARRAY,
    S: T_ARRAY,
    hn: T_FLOAT,
    phi_win: T_ARRAY,
    weight_win: T_ARRAY,
    wphi_line: T_ARRAY,
    mus: T_ARRAY,
    wmus: T_ARRAY,
) -> T_TUPLE[T_ARRAY, T_ARRAY]:
    """One formal sweep of a single line: accumulate Jbar and Lambda_star.

    Per window wavelength iw and angle mu, solve the Feautrier system on the
    vertical tau scale of that wavelength and accumulate the profile-weighted
    angle/wavelength quadrature:

        Jbar[k]  = sum_mu w_mu sum_iw weight*phi * j(iw,k)   / wphi[k]
        Lstar[k] = sum_mu w_mu sum_iw weight*phi * Psi(iw,k) / wphi[k]

    the division by wphi (the numerical profile norm on this window) cancels
    the quadrature's profile-area error identically -- with it, a constant
    radiation field J is averaged to exactly J on any mesh (RH's wphi trick).
    Lstar uses the SAME weighting: it must be the exact derivative
    d Jbar[k] / d S[k] of the discrete Jbar actually computed, since the line
    source S is constant across the (CRD) profile.

    Input:
        Z: (ND,), depth below the surface, [cm], ascending, Z[0] = 0
        chi0: (ND,), profile-integrated line opacity, [cm^-1 * cm]
        S: (ND,), current line source function
        hn: (,), incident intensity at the lower boundary (thermalized: B at bottom)
        phi_win: (nw, ND), profile table of this line's window, [cm^-1]
        weight_win: (nw,), window quadrature weights, [cm]
        wphi_line: (ND,), numerical profile norm on this window
        mus: (n_mu,), angle quadrature nodes on (0, 1)
        wmus: (n_mu,), angle quadrature weights, sum to 1

    Output:
        Jbar: (ND,), profile-weighted mean intensity
        Lstar: (ND,), diagonal approximate operator d Jbar / d S
    """
    nw = phi_win.shape[0]
    ND = Z.shape[0]
    n_mu = mus.shape[0]

    Jbar = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
    Lstar = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
    tau = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    for iw in range(nw):
        # vertical tau scale of this wavelength column (trapezoid over depth)
        tau[0] = 0.0
        for k in range(1, ND):
            chi_m = 0.5 * (chi0[k - 1] * phi_win[iw, k - 1] + chi0[k] * phi_win[iw, k])
            tau[k] = tau[k - 1] + chi_m * (Z[k] - Z[k - 1])
        for im in range(n_mu):
            res = _formal_rh_(tau, S, mus[im], 0.0, 0.0, 0.0, hn, E_FEAUTRIER_ORDER.SECOND, True)
            for k in range(ND):
                coe = wmus[im] * weight_win[iw] * phi_win[iw, k] / wphi_line[k]
                Jbar[k] += coe * res.j[k]
                Lstar[k] += coe * res.Psi[k]
    return Jbar, Lstar


MALI2lv_Result = _namedtuple("MALI2lv_Result", ["S", "Jbar", "Lstar", "niter", "dS_history"])


def mali_two_level_(
    Z: T_ARRAY,
    chi0: T_ARRAY,
    eps: T_ARRAY,
    B: T_ARRAY,
    phi_win: T_ARRAY,
    weight_win: T_ARRAY,
    wphi_line: T_ARRAY,
    n_angle: T_INT = 4,
    tol: T_FLOAT = 1.0e-8,
    itmax: T_INT = 20000,
    use_lstar: T_BOOL = True,
    lstar_scale: T_FLOAT = 1.0,
) -> MALI2lv_Result:
    """Two-level-atom MALI on the toy pipeline.

    The two-level problem is stated directly in (eps, B) form,
        S = (1 - eps) * Jbar + eps * B,
    so every oracle (sqrt(eps) law, Lambda-iteration fixed point) is analytic.
    populations enter only through chi0, which is held fixed here.

    the preconditioned update (operator splitting Jbar = Jbar_FS - Lstar*S_old
    + Lstar*S_new) is

        S_new = ((1-eps)*(Jbar - Lstar*S_old) + eps*B) / (1 - (1-eps)*Lstar)

    use_lstar=False degenerates to plain Lambda-iteration (Lstar = 0): the
    convergence crawls but the fixed point is identical -- Lstar (even a
    deliberately scaled one, see lstar_scale) cancels at S_new = S_old, so an
    imperfect operator changes the rate, never the answer.

    Input:
        Z, chi0, phi_win, weight_win, wphi_line: see two_level_sweep_
        eps: (ND,), photon destruction probability
        B: (ND,), Planck function at line center
        n_angle: (,), Gauss-Legendre angle points on (0, 1)
        tol: (,), convergence threshold on max|dS|/max(S)
        itmax: (,), iteration cap
        use_lstar: (,), False -> plain Lambda-iteration
        lstar_scale: (,), deliberate mis-scaling of the operator (tests only)

    Output: MALI2lv_Result(S, Jbar, Lstar, niter, dS_history)
    """
    ND = Z.shape[0]
    mus, wmus = _GaussLeg.gauss_quad_coe_(0.0, 1.0, n_angle)

    S = B.copy()
    hn = float(B[ND - 1])
    dS_history = []
    niter = 0
    Jbar = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
    Lstar = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
    for it in range(1, itmax + 1):
        niter = it
        Jbar, Lstar = two_level_sweep_(Z, chi0, S, hn, phi_win, weight_win, wphi_line, mus, wmus)
        if use_lstar:
            L = lstar_scale * Lstar
        else:
            L = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
        S_new = ((1.0 - eps) * (Jbar - L * S) + eps * B) / (1.0 - (1.0 - eps) * L)
        dS = float(_numpy.abs(S_new - S).max() / _numpy.abs(S_new).max())
        dS_history.append(dS)
        S = S_new
        if dS < tol:
            break
    return MALI2lv_Result(S=S, Jbar=Jbar, Lstar=Lstar, niter=niter, dS_history=_numpy.asarray(dS_history))


def multilevel_sweep_(
    Z: T_ARRAY,
    n_pop: T_ARRAY,
    Nt: T_ARRAY,
    wl: T_ARRAY,
    Nblue: T_ARRAY,
    span: T_ARRAY,
    win_off: T_ARRAY,
    phi: T_ARRAY,
    weight: T_ARRAY,
    wphi: T_ARRAY,
    w0: T_ARRAY,
    Aji: T_ARRAY,
    Bji: T_ARRAY,
    Bij: T_ARRAY,
    idxI: T_ARRAY,
    idxJ: T_ARRAY,
    planck_w0: T_ARRAY,
    mus: T_ARRAY,
    wmus: T_ARRAY,
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    """One formal sweep over every active line: Jbar, Lstar, and the line
    source function from the CURRENT populations.

    Opacity and emissivity are the production formulas themselves --
    bb_extinction_ / bb_emissivity_ (compiled bindings of
    Atomic/extinction.py, Atomic/emisivity.py):
        chi = h*nu/(4*pi) * (ni*Bij*phi - nj*Bji*psi),   psi = phi (CRD)
        eta = h*nu/(4*pi) * nj*Aji * psi
    under CRD their ratio is wavelength-independent inside a window, so the
    line source function is the per-depth scalar eta/chi evaluated with the
    profile-integrated coefficients (psi = phi = 1) at the line center.
    the toys carry no background continuum, and the toy windows are disjoint:
    each column belongs to exactly one line (overlap handling is full-MALI work).

    Input:
        Z: (ND,), depth, [cm], ascending, Z[0] = 0
        n_pop: (ND, nLevel), normalized populations
        Nt: (ND,), total species number density, [cm^-3]
        wl, Nblue, span: the global axis and per-line windows
        win_off, phi, weight, wphi: profile-table tier (see MALI_Precompute)
        w0, Aji, Bji, Bij, idxI, idxJ: (nLine,), line coefficients
        planck_w0: (ND, nLine), Planck at line center (lower-boundary intensity)
        mus, wmus: angle quadrature on (0, 1), weights sum to 1

    Output:
        Jbar: (nLine, ND)
        Lstar: (nLine, ND), diagonal operator d Jbar / d S_line
        S_line: (nLine, ND)
    """
    nLine = w0.shape[0]
    ND = Z.shape[0]
    n_mu = mus.shape[0]

    Jbar = _numpy.zeros((nLine, ND), dtype=DT_NB_FLOAT)
    Lstar = _numpy.zeros((nLine, ND), dtype=DT_NB_FLOAT)
    S_line = _numpy.empty((nLine, ND), dtype=DT_NB_FLOAT)
    chi_int = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    tau = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    for kL in range(nLine):
        for k in range(ND):
            ni = Nt[k] * n_pop[k, idxI[kL]]
            nj = Nt[k] * n_pop[k, idxJ[kL]]
            # profile-integrated (psi = phi = 1) coefficients at line center;
            # the spectral values are these times phi at each column
            chi_int[k] = _bb_extinction_(w0[kL], Bji[kL], Bij[kL], nj, ni, 1.0, 1.0)
            S_line[kL, k] = _bb_emissivity_(w0[kL], Aji[kL], nj) / chi_int[k]
        hn = planck_w0[ND - 1, kL]
        for iw in range(span[kL]):
            row = win_off[kL, 0] + iw
            # nu = c/wl of THIS column, mirroring the production per-wavelength
            # convention: rescale the line-center h*nu factor inside chi_int
            nu_ratio = w0[kL] / wl[Nblue[kL] + iw]
            tau[0] = 0.0
            for k in range(1, ND):
                chi_m = 0.5 * nu_ratio * (chi_int[k - 1] * phi[row, k - 1] + chi_int[k] * phi[row, k])
                tau[k] = tau[k - 1] + chi_m * (Z[k] - Z[k - 1])
            for im in range(n_mu):
                res = _formal_rh_(tau, S_line[kL, :], mus[im], 0.0, 0.0, 0.0, hn, E_FEAUTRIER_ORDER.SECOND, True)
                for k in range(ND):
                    coe = wmus[im] * weight[row] * phi[row, k] / wphi[k, kL]
                    Jbar[kL, k] += coe * res.j[k]
                    Lstar[kL, k] += coe * res.Psi[k]
    return Jbar, Lstar, S_line


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
    consumes them unchanged. passive b-f transitions enter unpreconditioned --
    their radiation is prescribed, there is no self-feedback to remove.

    Lstar is diagonal in depth, so each depth solves an independent
    nLevel x nLevel system: the population update stays LOCAL, which is the
    entire cost advantage of MALI over complete linearization.

    Input:
        Jbar, Lstar, S_line: (nLine, ND), from multilevel_sweep_
        Aji, Bji, Bij: (nLine,); idxI, idxJ: (nTran,), lines then continua
        Cij_coe: (nTran,); Cji_coe: (ND, nTran), collisional coefficients
        Rik, Rki_stim, Rki_spon: (ND, nCont), passive b-f rates
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
        _set_matrixC_(Cmat, Cji_coe[k, :], Cij_coe, idxI, idxJ, Ne[k])
        n_new[k, :] = _solve_SE_(Rmat, Cmat)
    return n_new


MALIml_Result = _namedtuple("MALIml_Result", ["n", "S_line", "Jbar", "Lstar", "niter", "dn_history"])


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
) -> MALIml_Result:
    """Multilevel MALI driver on the toy pipeline.

    interpreted orchestration: unpacks the structs into the plain arrays the
    jitted kernels take, starts from LTE populations, and iterates
        sweep -> preconditioned per-depth SE -> convergence check
    until max|dn| < tol (populations are normalized to 1 per depth).

    Input:
        atom: Structs.Toy_Atom
        atmos: Structs.Atmos1D
        mesh: GlobalMesh.Global_Mesh
        pre: Structs.MALI_Precompute
        use_lstar: (,), False -> plain (preconditioner-free) iteration
        lstar_scale: (,), deliberate operator mis-scaling (tests only)

    Output: MALIml_Result(n, S_line, Jbar, Lstar, niter, dn_history)
    """
    mus, wmus = _GaussLeg.gauss_quad_coe_(0.0, 1.0, n_angle)
    nTran = atom.nLine + atom.nCont
    idxI = _numpy.empty(nTran, dtype=DT_NB_INT)
    idxJ = _numpy.empty(nTran, dtype=DT_NB_INT)
    idxI[: atom.nLine] = atom.Line["idxI"][:]
    idxJ[: atom.nLine] = atom.Line["idxJ"][:]
    if atom.nCont > 0:
        idxI[atom.nLine :] = atom.Cont["idxI"][:]
        idxJ[atom.nLine :] = atom.Cont["idxJ"][:]

    w0 = _numpy.ascontiguousarray(atom.Line["w0"][:])
    Aji = _numpy.ascontiguousarray(atom.Line["AJI"][:])
    Bji = _numpy.ascontiguousarray(atom.Line["BJI"][:])
    Bij = _numpy.ascontiguousarray(atom.Line["BIJ"][:])

    scale = lstar_scale if use_lstar else 0.0
    n = pre.n_LTE.copy()
    S_line = _numpy.zeros((atom.nLine, atmos.ND), dtype=DT_NB_FLOAT)
    Jbar = _numpy.zeros_like(S_line)
    Lstar = _numpy.zeros_like(S_line)
    dn_history = []
    niter = 0
    for it in range(1, itmax + 1):
        niter = it
        Jbar, Lstar, S_line = multilevel_sweep_(
            atmos.Z, n, atmos.Nt, mesh.wl, mesh.Nblue, mesh.span, pre.win_off,
            pre.phi, pre.weight, pre.wphi, w0, Aji, Bji, Bij, idxI[: atom.nLine], idxJ[: atom.nLine],
            pre.planck_w0, mus, wmus,
        )  # fmt: skip
        n_new = update_populations_(
            Jbar, Lstar, S_line, Aji, Bji, Bij, idxI, idxJ,
            atom.Cij_coe, pre.Cji_coe, pre.Rik, pre.Rki_stim, pre.Rki_spon,
            atmos.Ne, atom.nLevel, scale,
        )  # fmt: skip
        dn = float(_numpy.abs(n_new - n).max())
        dn_history.append(dn)
        n = n_new
        if dn < tol:
            break
    return MALIml_Result(n=n, S_line=S_line, Jbar=Jbar, Lstar=Lstar, niter=niter, dn_history=_numpy.asarray(dn_history))


# -------------------------------------------------------------------------------
# numba optimization : per-iteration kernels compile unconditionally
# -------------------------------------------------------------------------------

two_level_sweep_ = nb_njit(**NB_NJIT_KWGS)(two_level_sweep_)
multilevel_sweep_ = nb_njit(**NB_NJIT_KWGS)(multilevel_sweep_)
update_populations_ = nb_njit(**NB_NJIT_KWGS)(update_populations_)
