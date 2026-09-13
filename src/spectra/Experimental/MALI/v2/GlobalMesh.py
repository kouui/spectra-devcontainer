# -------------------------------------------------------------------------------
# global (shared) wavelength mesh for MALI -- ONE axis for every transition
#
# the axis is anchored with a SCALAR reference velocity (xi_ref) instead of the
# local Doppler width, so one wavelength array is valid at every depth point --
# the precondition for any 1/2/3-D transfer solve. depth-dependent physics never
# touches this axis; it enters later as profile *values* on these fixed columns
# (see ProfileTable.py).
#
# lines and continua are merged into the same axis (lines first, then continua,
# in the atom's own order), so a column inside several transitions' ranges is
# solved ONCE with all of their opacity. the reverse map -- which transitions
# cover column iw -- is stored as a CSR table so the per-iteration sweep never
# searches: rows of the window tables are found by arithmetic on win_off.
#
# a continuum's threshold column belongs to that continuum (closed interval
# [lamb_min, edge]); the axis carries one float per wavelength, so only the
# blue-side (b-f active) state of the edge exists -- the red-side state RH keeps
# as a second record is dropped. rates integrate over each continuum's own
# window and never see the red side, so this only affects the emergent
# spectrum at exactly the edge wavelength.
#
# build-once code: runs a single time per atom/mesh configuration, so it stays
# interpreted (no njit) and may use lists and dataclasses freely.
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

import numpy as _numpy

from ....ImportAll import *


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class Global_Mesh:
    wl: T_ARRAY  # (Nspect,), absolute wavelength [cm], sorted ascending
    # per transition (lines then continua, atom order): window on the axis
    Nblue: T_ARRAY  # (nTran,), global index of each transition's bluemost point
    span: T_ARRAY  # (nTran,), number of consecutive global points per transition
    # [start, stop) rows of each transition in the flattened window tables
    # (row of column iw of transition t = win_off[t, 0] + iw - Nblue[t])
    win_off: T_ARRAY  # (nTran, 2)
    # CSR reverse map: transitions covering column iw are
    # col_tran[col_ptr[iw] : col_ptr[iw+1]], and col_row holds their window rows
    col_ptr: T_ARRAY  # (Nspect + 1,)
    col_tran: T_ARRAY  # (nnz,)
    col_row: T_ARRAY  # (nnz,)


def anchor_line_mesh_(q: T_ARRAY, w0: T_FLOAT, xi_ref: T_FLOAT) -> T_ARRAY:
    """Anchor a dimensionless line mesh template to absolute wavelength.

    One unit of q corresponds to w0*xi_ref/c in wavelength. xi_ref is a ruler,
    not a physical width: no profile is ever evaluated "with" xi_ref, so its
    value only decides where the fixed grid points sit (velocity coverage
    per q unit), never the correctness of any downstream value.

    Input:
        q: (nLambda,), dimensionless symmetric line mesh (MeshUtil.make_full_line_mesh_)
        w0: (,), line center wavelength, [cm]
        xi_ref: (,), reference velocity ruler, [cm/s]

    Output:
        wl: (nLambda,), absolute wavelength, [cm], ascending
    """
    return w0 * (1.0 + q[:] * xi_ref / CST.c_)


def merge_meshes_(meshes: T_LIST) -> Global_Mesh:
    """Merge per-transition absolute-wavelength arrays into one sorted global axis.

    Points from different transitions closer than a tolerance are merged into
    one shared grid point (overlap consistency: transitions overlapping in
    wavelength must literally share grid points). The tolerance is taken
    relative to the smallest spacing INSIDE any single mesh, so two distinct
    points of the same transition can never collapse.

    Input:
        meshes: list of (nLambda_i,) STRICTLY ASCENDING absolute-wavelength
            arrays [cm], lines first then continua, in the atom's order. the
            production continuum mesh is descending: the caller reverses it.

    Output:
        Global_Mesh
    """
    nTran = len(meshes)
    for t, wl_t in enumerate(meshes):
        if wl_t.ndim != 1 or wl_t.shape[0] == 0 or not _numpy.all(_numpy.isfinite(wl_t)):
            raise ValueError(f"mesh {t}: expected a non-empty finite 1-D array")
        if wl_t.shape[0] > 1 and not _numpy.all(_numpy.diff(wl_t) > 0.0):
            raise ValueError(f"mesh {t}: must be strictly ascending (continuum meshes are descending by default)")

    min_spacing = _numpy.inf
    for wl_line in meshes:
        if wl_line.shape[0] > 1:
            min_spacing = min(min_spacing, _numpy.diff(wl_line).min())
    eps = 1.0e-3 * min_spacing if _numpy.isfinite(min_spacing) else 0.0

    wl_all = _numpy.sort(_numpy.concatenate(meshes))
    # keep the first point of every cluster of near-duplicates
    keep = _numpy.empty(wl_all.shape[0], dtype=bool)
    keep[0] = True
    keep[1:] = _numpy.diff(wl_all) > eps
    wl = wl_all[keep]

    Nblue = _numpy.empty(nTran, dtype=DT_NB_INT)
    span = _numpy.empty(nTran, dtype=DT_NB_INT)
    for k, wl_line in enumerate(meshes):
        i0 = _nearest_index_(wl, wl_line[0], eps)
        i1 = _nearest_index_(wl, wl_line[-1], eps)
        Nblue[k] = i0
        span[k] = i1 - i0 + 1

    win_off = _numpy.empty((nTran, 2), dtype=DT_NB_INT)
    stop = 0
    for t in range(nTran):
        win_off[t, 0] = stop
        stop += int(span[t])
        win_off[t, 1] = stop

    col_ptr, col_tran, col_row = _column_membership_(wl.shape[0], Nblue, span, win_off)
    return Global_Mesh(
        wl=wl, Nblue=Nblue, span=span, win_off=win_off, col_ptr=col_ptr, col_tran=col_tran, col_row=col_row
    )


def _column_membership_(
    Nspect: T_INT, Nblue: T_ARRAY, span: T_ARRAY, win_off: T_ARRAY
) -> T_TUPLE[T_ARRAY, T_ARRAY, T_ARRAY]:
    """CSR table of the transitions covering every axis column, in transition
    order within a column (so lines precede continua, matching the atom)."""
    nTran = Nblue.shape[0]
    count = _numpy.zeros(Nspect, dtype=DT_NB_INT)
    for t in range(nTran):
        count[Nblue[t] : Nblue[t] + span[t]] += 1
    col_ptr = _numpy.zeros(Nspect + 1, dtype=DT_NB_INT)
    col_ptr[1:] = _numpy.cumsum(count)
    nnz = int(col_ptr[-1])
    col_tran = _numpy.empty(nnz, dtype=DT_NB_INT)
    col_row = _numpy.empty(nnz, dtype=DT_NB_INT)
    fill = col_ptr[:-1].copy()
    for t in range(nTran):
        for iw in range(int(Nblue[t]), int(Nblue[t] + span[t])):
            col_tran[fill[iw]] = t
            col_row[fill[iw]] = win_off[t, 0] + iw - Nblue[t]
            fill[iw] += 1
    return col_ptr, col_tran, col_row


def _nearest_index_(wl: T_ARRAY, value: T_FLOAT, eps: T_FLOAT) -> T_INT:
    idx = int(_numpy.searchsorted(wl, value))
    if idx > 0 and (idx == wl.shape[0] or value - wl[idx - 1] <= wl[idx] - value):
        idx -= 1
    if abs(wl[idx] - value) > eps:
        raise ValueError("wavelength not found in merged global mesh")
    return idx


def trapezoidal_weight_(wl: T_ARRAY) -> T_ARRAY:
    """Trapezoidal quadrature weights of a (non-uniform) ascending axis.

    sum_j weight[j]*f[j] equals the trapezoid integral of f over the axis.

    Input:
        wl: (n,), ascending axis

    Output:
        weight: (n,), same unit as wl
    """
    n = wl.shape[0]
    weight = _numpy.empty(n, dtype=DT_NB_FLOAT)
    if n == 1:
        weight[0] = 1.0
        return weight
    weight[0] = 0.5 * (wl[1] - wl[0])
    weight[-1] = 0.5 * (wl[-1] - wl[-2])
    weight[1:-1] = 0.5 * (wl[2:] - wl[:-2])
    return weight
