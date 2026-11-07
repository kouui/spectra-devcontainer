# -------------------------------------------------------------------------------
# loop-invariant tier of the MALI iteration
#
# build-once tier: filled once per model configuration by Function/MALI/Setup.py
# and only read by the per-iteration kernels. a plain container of arrays so
# that RadiativeTransfer/MALI never has to import an input struct.
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

from ...ImportAll import *


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class MALI_Precompute:
    """Loop-invariant tier: everything here depends only on (Te, Ne, Vt) per
    depth -- never on populations -- so it is computed once and read by every
    MALI iteration. valid while Ne is fixed (no charge-conservation iteration).
    """

    n_LTE: T_ARRAY  # (ND, nLevel), normalized LTE populations
    nj_by_ni: T_ARRAY  # (ND, nTran), LTE ratio per transition (lines then continua)
    Cij_coe: T_ARRAY  # (ND, nTran), upward collisional coefficient [cm^3 s^-1]
    Cji_coe: T_ARRAY  # (ND, nTran), downward collisional coefficient [cm^3 s^-1]
    dopWidth_cm: T_ARRAY  # (ND, nLine), [cm]
    adamp: T_ARRAY  # (ND, nLine), Voigt damping parameter (0 for Gaussian toys)
    # line profile tables, flattened over the line windows of the global mesh:
    # rows mesh.win_off[kL,0]:mesh.win_off[kL,1] belong to line kL (lines come
    # first in win_off, so these rows start at 0)
    phi: T_ARRAY  # (nWinLine, ND), [cm^-1]
    wphi: T_ARRAY  # (ND, nLine), numerical profile norm on the window quadrature
    weight: T_ARRAY  # (nWinLine,), per-window trapezoidal weights, [cm]
    # continuum cross sections on the continuum windows: row r of the global
    # window tables (r >= row_cont0) maps to alpha_win[r - row_cont0]
    alpha_win: T_ARRAY  # (nWinCont,), [cm^2]
    row_cont0: T_INT  # first continuum row = mesh.win_off[nLine, 0] (nWinLine)
    # per axis column, populations never touch these
    exp_hnu_kT: T_ARRAY  # (Nspect, ND), exp(-h nu/kT), stimulated factor
    twohc2_wl5: T_ARRAY  # (Nspect,), 2 h c^2 / wl^5, cm-base emission factor
    hn_bottom: T_ARRAY  # (Nspect,), Planck at the lower boundary, per column
    # background of everything the active atom does not provide
    bg_chi: T_ARRAY  # (Nspect, ND), THERMAL extinction, [cm^-1]
    bg_eta: T_ARRAY  # (Nspect, ND), bg_chi * B_lambda(Te)
    bg_sca: T_ARRAY  # (Nspect, ND), coherent-scattering extinction, [cm^-1]
