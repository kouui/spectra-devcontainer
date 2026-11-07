# -------------------------------------------------------------------------------
# two-level-atom MALI driver, a test oracle for the kernels
#
# it runs the unified sweep with one line, no continua, no background and no
# scattering, so that every result has an analytic reference. production code
# does not use it.
# -------------------------------------------------------------------------------

from collections import namedtuple as _namedtuple

import numpy as _numpy

from spectra.ImportAll import *
from spectra.Math import GaussLeg as _GaussLeg
from spectra.RadiativeTransfer.MALI.Kernels import line_rates_, unified_sweep_

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
        n_angle, tol, itmax, use_lstar, lstar_scale: see Function/MALI/Loop.mali_multilevel_

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
