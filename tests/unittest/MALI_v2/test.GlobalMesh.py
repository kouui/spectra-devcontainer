"""Unit tests for spectra.Experimental.MALI.v2.GlobalMesh (RH stages B+C).

The global axis is pure geometry: anchoring a dimensionless template with a
scalar ruler, merging per-transition arrays into one sorted deduplicated axis,
locating each transition as an (offset, span) window, and the CSR reverse map
from column to covering transitions. No atmosphere physics enters.
"""

import numpy as np
import pytest

from spectra import Constants as CST
from spectra.Experimental.MALI.v2 import GlobalMesh
from spectra.Util import MeshUtil

XI_REF = 2.5e5  # 2.5 km/s in cm/s


def _line_mesh(w0, nLambda=21, qcore=2.5, qwing=10.0):
    q = MeshUtil.make_full_line_mesh_(nLambda, qcore, qwing)
    return GlobalMesh.anchor_line_mesh_(q, w0, XI_REF)


class TestAnchor:
    def test_center_and_extent(self):
        w0 = 5000.0e-8  # 5000 AA in cm
        wl = _line_mesh(w0, nLambda=21, qcore=2.5, qwing=10.0)
        assert wl[10] == pytest.approx(w0, rel=1e-15)
        # q = +-qwing maps to +-qwing * w0*xi_ref/c
        assert wl[-1] - w0 == pytest.approx(10.0 * w0 * XI_REF / CST.c_, rel=1e-12)
        assert wl[0] - w0 == pytest.approx(-10.0 * w0 * XI_REF / CST.c_, rel=1e-12)
        assert np.all(np.diff(wl) > 0)

    def test_axis_is_depth_independent(self):
        # the whole point of the scalar ruler: no local quantity enters
        wl_a = _line_mesh(5000.0e-8)
        wl_b = _line_mesh(5000.0e-8)
        assert np.array_equal(wl_a, wl_b)


class TestMerge:
    def test_single_line_round_trip(self):
        wl_line = _line_mesh(5000.0e-8)
        mesh = GlobalMesh.merge_meshes_([wl_line])
        assert mesh.wl.shape[0] == wl_line.shape[0]
        assert mesh.Nblue[0] == 0
        assert mesh.span[0] == wl_line.shape[0]
        assert np.array_equal(mesh.wl[mesh.Nblue[0] : mesh.Nblue[0] + mesh.span[0]], wl_line)

    def test_disjoint_lines_round_trip(self):
        wl_a = _line_mesh(4000.0e-8)
        wl_b = _line_mesh(6000.0e-8)
        mesh = GlobalMesh.merge_meshes_([wl_a, wl_b])
        assert mesh.wl.shape[0] == wl_a.shape[0] + wl_b.shape[0]
        got_a = mesh.wl[mesh.Nblue[0] : mesh.Nblue[0] + mesh.span[0]]
        got_b = mesh.wl[mesh.Nblue[1] : mesh.Nblue[1] + mesh.span[1]]
        assert np.array_equal(got_a, wl_a)
        assert np.array_equal(got_b, wl_b)

    def test_identical_lines_deduplicate(self):
        wl_a = _line_mesh(5000.0e-8)
        mesh = GlobalMesh.merge_meshes_([wl_a, wl_a.copy()])
        assert mesh.wl.shape[0] == wl_a.shape[0]
        assert mesh.Nblue[0] == mesh.Nblue[1] == 0
        assert mesh.span[0] == mesh.span[1] == wl_a.shape[0]

    def test_overlapping_lines_share_axis(self):
        # second line centered inside the first line's red wing
        w0_a = 5000.0e-8
        wl_a = _line_mesh(w0_a)
        w0_b = wl_a[-3]  # a wavelength the first line already samples
        wl_b = _line_mesh(w0_b)
        mesh = GlobalMesh.merge_meshes_([wl_a, wl_b])

        # every original point of both lines exists on the global axis
        for wl_line, k in ((wl_a, 0), (wl_b, 1)):
            window = mesh.wl[mesh.Nblue[k] : mesh.Nblue[k] + mesh.span[k]]
            for v in wl_line:
                assert np.min(np.abs(window - v)) <= 1e-3 * np.min(np.diff(wl_line))

        # line B's window covers points contributed by line A (span > own count)
        assert mesh.span[1] > wl_b.shape[0]
        # the exactly-shared point (b's center == a's sample) was merged, not doubled
        n_shared = np.sum(np.abs(mesh.wl - w0_b) <= 1e-16)
        assert n_shared == 1

    def test_windows_are_contiguous_and_sorted(self):
        meshes = [_line_mesh(w0) for w0 in (4998.0e-8, 5000.0e-8, 5001.0e-8)]
        mesh = GlobalMesh.merge_meshes_([meshes[1], meshes[0], meshes[2]])
        assert np.all(np.diff(mesh.wl) > 0)
        for k in range(3):
            assert mesh.Nblue[k] >= 0
            assert mesh.Nblue[k] + mesh.span[k] <= mesh.wl.shape[0]

    def test_uniform_degenerate_mesh(self):
        # qwing <= 2*qcore: template degenerates to (near) uniform spacing --
        # the recommended configuration for large-velocity simulations
        q = MeshUtil.make_full_line_mesh_(21, qcore=10.0, qwing=10.0)
        d = np.diff(q)
        assert d.max() / d.min() == pytest.approx(1.0, abs=1e-6)
        wl = GlobalMesh.anchor_line_mesh_(q, 5000.0e-8, XI_REF)
        mesh = GlobalMesh.merge_meshes_([wl])
        assert np.array_equal(mesh.wl, wl)


class TestMergeWithContinua:
    """The axis is transition-complete: merge_meshes_ is agnostic to what an
    ascending wavelength array represents, so continuum meshes get (Nblue,
    span) windows exactly like lines. A consumer may ignore windows (the MALI
    toy keeps continua passive), but the builder must handle them all."""

    def _cont_mesh(self, w0, nLambda=41):
        # production convention: threshold times a descending template
        return (w0 * MeshUtil.make_continuum_mesh_(nLambda))[::-1].copy()

    def test_continuum_window_swallows_interleaved_lines(self):
        cont = self._cont_mesh(5000.0e-8)
        wl_a = _line_mesh(3000.0e-8)
        wl_b = _line_mesh(4000.0e-8)
        mesh = GlobalMesh.merge_meshes_([wl_a, wl_b, cont])
        # every input point is present in its own window
        for k, src in enumerate((wl_a, wl_b, cont)):
            window = mesh.wl[mesh.Nblue[k] : mesh.Nblue[k] + mesh.span[k]]
            assert np.all(np.isin(src, window))
        # both lines sit inside the continuum's range: its window includes them
        assert mesh.span[2] >= cont.shape[0] + wl_a.shape[0] + wl_b.shape[0]
        # the narrow line windows contain no continuum points by accident here
        assert np.all(np.diff(mesh.wl) > 0)

    def test_exact_shared_point_deduplicates_across_types(self):
        # a continuum-like coarse mesh sharing one point exactly with a line
        wl_line = _line_mesh(5000.0e-8)
        cont = np.array([wl_line[-1], 5100.0e-8, 5200.0e-8])
        mesh = GlobalMesh.merge_meshes_([wl_line, cont])
        assert mesh.wl.shape[0] == wl_line.shape[0] + cont.shape[0] - 1
        got_line = mesh.wl[mesh.Nblue[0] : mesh.Nblue[0] + mesh.span[0]]
        got_cont = mesh.wl[mesh.Nblue[1] : mesh.Nblue[1] + mesh.span[1]]
        # both windows locate the one shared axis point
        assert got_line[-1] == got_cont[0]

    def test_mixed_scales_keep_distinct_points(self):
        # eps comes from the finest intra-mesh spacing; a coarse-mesh point
        # near (but beyond eps of) a line point must remain its own axis point
        wl_line = _line_mesh(5000.0e-8)
        eps = 1.0e-3 * np.min(np.diff(wl_line))
        cont = np.array([wl_line[0] + 10.0 * eps, 5100.0e-8, 5200.0e-8])
        mesh = GlobalMesh.merge_meshes_([wl_line, cont])
        assert mesh.wl.shape[0] == wl_line.shape[0] + cont.shape[0]


class TestInputContract:
    def test_descending_mesh_rejected(self):
        cont = 5000.0e-8 * MeshUtil.make_continuum_mesh_(41)  # production: descending
        with pytest.raises(ValueError, match="ascending"):
            GlobalMesh.merge_meshes_([_line_mesh(3000.0e-8), cont])

    def test_empty_or_nonfinite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            GlobalMesh.merge_meshes_([np.empty(0)])
        with pytest.raises(ValueError, match="finite"):
            GlobalMesh.merge_meshes_([np.array([1.0, np.nan, 2.0])])


def _membership_reference(wl, meshes):
    """Independent oracle from the RAW input meshes: column iw belongs to
    transition t iff its wavelength lies inside t's closed range (to the merge
    tolerance); rows count that transition's columns from its bluest one."""
    eps = 1.0e-3 * min(np.min(np.diff(m)) for m in meshes if m.shape[0] > 1)
    ref = [[] for _ in range(wl.shape[0])]
    row = 0
    for t, m in enumerate(meshes):
        for iw in range(wl.shape[0]):
            if m[0] - eps <= wl[iw] <= m[-1] + eps:
                ref[iw].append((t, row))
                row += 1
    return ref


def _assert_csr_matches(mesh, meshes):
    ref = _membership_reference(mesh.wl, meshes)
    for iw in range(mesh.wl.shape[0]):
        lo, hi = mesh.col_ptr[iw], mesh.col_ptr[iw + 1]
        got = list(zip(mesh.col_tran[lo:hi], mesh.col_row[lo:hi], strict=True))
        assert got == ref[iw], iw
    assert mesh.col_ptr[-1] == sum(len(r) for r in ref)


def _cont_mesh(w_edge, nLambda=41):
    return (w_edge * MeshUtil.make_continuum_mesh_(nLambda))[::-1].copy()


def _hydrogen_like_meshes(n_max=5):
    """Rydberg ladder: lines from levels 1..n_max-1 to every higher level, then
    one continuum per level -- the FALC 6-level hydrogen axis layout without
    the data files (Ly-alpha inside the Balmer range, Balmer inside Paschen)."""
    w_edge = [CST.h_ * CST.c_ / (CST.E_Rydberg_H_ / n**2) for n in range(1, n_max + 1)]
    lines = []
    for i in range(1, n_max):
        for j in range(i + 1, n_max + 1):
            lines.append(_line_mesh(1.0 / (1.0 / w_edge[i - 1] - 1.0 / w_edge[j - 1]), nLambda=31, qwing=50.0))
    conts = [_cont_mesh(w) for w in w_edge]
    return lines, conts


class TestWindowsAndMembership:
    def _mesh(self):
        # line inside continuum B; continuum A's range partly inside B's;
        # a second line redward of B's edge so the edge has a red neighbour
        cont_a, cont_b = _cont_mesh(3333.0e-8), _cont_mesh(10000.0e-8)
        meshes = [_line_mesh(5000.0e-8), _line_mesh(10020.0e-8), cont_a, cont_b]
        return GlobalMesh.merge_meshes_(meshes), meshes

    def test_win_off_is_prefix_sum_of_span(self):
        mesh, _ = self._mesh()
        assert mesh.win_off[0, 0] == 0
        for t in range(mesh.span.shape[0]):
            assert mesh.win_off[t, 1] - mesh.win_off[t, 0] == mesh.span[t]
            if t > 0:
                assert mesh.win_off[t, 0] == mesh.win_off[t - 1, 1]

    def test_csr_matches_independent_oracle(self):
        mesh, meshes = self._mesh()
        _assert_csr_matches(mesh, meshes)

    def test_csr_matches_oracle_on_hydrogen_like_axis(self):
        lines, conts = _hydrogen_like_meshes()
        meshes = lines + conts
        mesh = GlobalMesh.merge_meshes_(meshes)
        _assert_csr_matches(mesh, meshes)
        nL = len(lines)

        def members(wl_value):
            iw = int(np.argmin(np.abs(mesh.wl - wl_value)))
            return set(mesh.col_tran[mesh.col_ptr[iw] : mesh.col_ptr[iw + 1]].tolist())

        # Ly-alpha (line 0) center, 121.6 nm: the line + the Balmer continuum
        # only -- the Lyman edge is at 91.2 nm and the production continuum
        # mesh reaches down to ~0.16 x edge, so Paschen starts at ~128 nm
        assert conts[2][0] > lines[0][-1]
        assert members(lines[0][15]) == {0, nL + 1}
        # Balmer edge column (364.6 nm): Balmer (closed interval) + Paschen,
        # Brackett, Pfund whose ranges contain it; no Lyman, no line
        assert members(conts[1][-1]) == {nL + n for n in range(1, 5)}

    def test_overlap_cases(self):
        mesh, meshes = self._mesh()
        line_red, cont_a, cont_b = meshes[1], meshes[2], meshes[3]

        def members(wl_value):
            iw = int(np.argmin(np.abs(mesh.wl - wl_value)))
            return set(mesh.col_tran[mesh.col_ptr[iw] : mesh.col_ptr[iw + 1]].tolist())

        assert members(5000.0e-8) == {0, 3}  # line inside continuum B only
        assert members(2000.0e-8) == {2, 3}  # A's point inside B's range
        assert members(600.0e-8) == {2}  # A alone below B's range
        assert members(cont_a[-1]) == {2, 3}  # A's edge: closed interval, inside B
        # B's edge column is B's (closed interval); the next column redward --
        # the bluest point of the red line -- is not
        iw_edge = int(np.argmin(np.abs(mesh.wl - cont_b[-1])))
        assert members(cont_b[-1]) == {3}
        assert mesh.wl[iw_edge + 1] == line_red[0]
        assert members(line_red[0]) == {1}

    def test_line_only_columns_belong_to_exactly_one(self):
        mesh = GlobalMesh.merge_meshes_([_line_mesh(4000.0e-8), _line_mesh(6000.0e-8)])
        assert np.all(np.diff(mesh.col_ptr) == 1)


class TestTrapezoidalWeight:
    def test_matches_numpy_trapezoid(self):
        rng = np.random.default_rng(7)
        x = np.sort(rng.uniform(0.0, 1.0, 31))
        f = np.sin(3.0 * x) + 2.0
        w = GlobalMesh.trapezoidal_weight_(x)
        assert np.sum(w * f) == pytest.approx(np.trapezoid(f, x), rel=1e-14)

    def test_total_weight_is_axis_length(self):
        x = np.linspace(2.0, 5.0, 11)
        w = GlobalMesh.trapezoidal_weight_(x)
        assert w.sum() == pytest.approx(3.0, rel=1e-14)
