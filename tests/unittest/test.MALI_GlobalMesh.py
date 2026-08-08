"""Unit tests for spectra.Experimental.MALI.GlobalMesh (RH stages B+C).

The global axis is pure geometry: anchoring a dimensionless template with a
scalar ruler, merging per-line arrays into one sorted deduplicated axis, and
locating each line as an (offset, span) window. No atmosphere physics enters.
"""

import numpy as np
import pytest

from spectra import Constants as CST
from spectra.Experimental.MALI import GlobalMesh
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
