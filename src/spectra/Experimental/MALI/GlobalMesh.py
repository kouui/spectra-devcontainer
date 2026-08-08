# -------------------------------------------------------------------------------
# global (shared) wavelength mesh for MALI
#
# the axis is anchored with a SCALAR reference velocity (xi_ref) instead of the
# local Doppler width, so one wavelength array is valid at every depth point --
# the precondition for any 1/2/3-D transfer solve. depth-dependent physics never
# touches this axis; it enters later as profile *values* on these fixed columns
# (see ProfileTable.py).
#
# build-once code: runs a single time per atom/mesh configuration, so it stays
# interpreted (no njit) and may use lists and dataclasses freely.
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

import numpy as _numpy

from ...ImportAll import *


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class Global_Mesh:
    wl: T_ARRAY  # (Nspect,), absolute wavelength [cm], sorted ascending
    Nblue: T_ARRAY  # (nLine,), global index of each line's bluemost point
    span: T_ARRAY  # (nLine,), number of consecutive global points per line


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


def merge_meshes_(line_meshes: T_LIST) -> Global_Mesh:
    """Merge per-line absolute-wavelength arrays into one sorted global axis.

    Points from different lines closer than a tolerance are merged into one
    shared grid point (overlap consistency: transitions overlapping in
    wavelength must literally share grid points). The tolerance is taken
    relative to the smallest spacing INSIDE any single line mesh, so two
    distinct points of the same line can never collapse.

    Input:
        line_meshes: list of (nLambda_i,) ascending absolute-wavelength arrays [cm]

    Output:
        Global_Mesh(wl, Nblue, span, weight)
    """
    nLine = len(line_meshes)

    min_spacing = _numpy.inf
    for wl_line in line_meshes:
        if wl_line.shape[0] > 1:
            min_spacing = min(min_spacing, _numpy.diff(wl_line).min())
    eps = 1.0e-3 * min_spacing if _numpy.isfinite(min_spacing) else 0.0

    wl_all = _numpy.sort(_numpy.concatenate(line_meshes))
    # keep the first point of every cluster of near-duplicates
    keep = _numpy.empty(wl_all.shape[0], dtype=bool)
    keep[0] = True
    keep[1:] = _numpy.diff(wl_all) > eps
    wl = wl_all[keep]

    Nblue = _numpy.empty(nLine, dtype=DT_NB_INT)
    span = _numpy.empty(nLine, dtype=DT_NB_INT)
    for k, wl_line in enumerate(line_meshes):
        i0 = _nearest_index_(wl, wl_line[0], eps)
        i1 = _nearest_index_(wl, wl_line[-1], eps)
        Nblue[k] = i0
        span[k] = i1 - i0 + 1

    return Global_Mesh(wl=wl, Nblue=Nblue, span=span)


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
