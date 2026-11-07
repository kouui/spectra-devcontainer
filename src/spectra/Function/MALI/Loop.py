# -------------------------------------------------------------------------------
# the MALI iteration driver: interpreted orchestration around the jitted kernels
#
# it does bookkeeping, not arithmetic: unpack the structs into plain arrays,
# call the kernels of RadiativeTransfer/MALI/Kernels.py, check convergence.
# -------------------------------------------------------------------------------

from collections import namedtuple as _namedtuple

import numpy as _numpy

from ...Atomic import LTELib as _LTELib
from ...ImportAll import *
from ...Math import GaussLeg as _GaussLeg
from ...RadiativeTransfer.MALI import GlobalMesh as _GlobalMesh
from ...RadiativeTransfer.MALI import Precompute as _Precompute
from ...RadiativeTransfer.MALI.Kernels import (
    W_SCA_MAX,
    bf_rates_kernel_,
    continuum_coefficients_,
    line_coefficients_,
    line_rates_,
    unified_sweep_,
    update_populations_,
)
from . import Setup as _Setup


def bf_rates_(
    J: T_ARRAY,
    mesh: _GlobalMesh.Global_Mesh,
    pre: _Precompute.MALI_Precompute,
    atom: _Setup.Toy_Atom,
    atmos: _Setup.Atmos1D,
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    """bf_rates_kernel_ on the structs (see there).

    Output:
        Rik, Rki_stim, Rki_spon: (ND, nCont), [s^-1]
    """
    return bf_rates_kernel_(
        J, mesh.wl, mesh.Nblue, mesh.span, mesh.win_off, pre.alpha_win, pre.row_cont0, atom.nLine,
        atmos.Te, pre.nj_by_ni,
    )  # fmt: skip


MALIml_Result = _namedtuple(
    "MALIml_Result", ["n", "S_line", "Jbar", "Lstar", "J", "niter", "converged", "dn_history", "dJ_history"]
)


def mali_multilevel_(
    atom: _Setup.Toy_Atom,
    atmos: _Setup.Atmos1D,
    mesh: _GlobalMesh.Global_Mesh,
    pre: _Precompute.MALI_Precompute,
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
        atom: Setup.Toy_Atom
        atmos: Setup.Atmos1D
        mesh: GlobalMesh.Global_Mesh (windows for every transition)
        pre: Precompute.MALI_Precompute
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
        w_sca = _numpy.clip(Psi * pre.bg_sca / chi_tot, 0.0, W_SCA_MAX)
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
