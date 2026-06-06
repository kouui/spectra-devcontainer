# Behavioral tests that lock in the per-field semantics of the new
# Atmosphere.Vd_obs (observer-frame; shifts slab output wavelength mesh)
# and Atmosphere.Vd_sun (sun-rest-frame; SE samples the solar spectrum on a
# Vd_sun-shifted mesh, profile stays at w0) split. All pre-existing
# regression tests use Vd=0 — a silent swap of these two fields would pass
# undetected without these locks.

import numpy as np
import pytest

from spectra.Atomic import BasicP
from spectra.Experimental import ExFAL
from spectra.Function.SEquil import SELib
from spectra.Function.SlabModel import CloudModel as _CloudModel
from spectra.ImportAll import *
from spectra.Struct import Atmosphere, Atom, Radiation

_CONF_PATH = str(CFG._ROOT_DIR / "data/conf/H.conf")
_FAL_PATH = CFG._ROOT_DIR / "data/atmos/FAL/FALC_82.atmos"


def _hydrogen_setup(Vd_obs: float = 0.0, Vd_sun: float = 0.0):
    atom, _ = Atom.init_Atom_(_CONF_PATH, is_hydrogen=True)
    wMesh = atom._wave_mesh
    atmos = Atmosphere.Atmosphere0D(
        Nh=1.0e12,
        Ne=1.0e11,
        Te=7.0e3,
        Vd_obs=Vd_obs,
        Vd_sun=Vd_sun,
        Vt=5.0e5,
    )
    radiation = Radiation.init_Radiation_()
    SE_con, _ = SELib.cal_SE_with_Nh_Te_(atom, atmos, radiation, None)
    return atom, wMesh, atmos, radiation, SE_con


def test_slab_wl_1D_matches_Vd_obs_formula():
    # Vd_obs must reproduce the observer-frame mesh formula
    # wl = Line_mesh * dopWidth + w0 + w0 * Vd_obs / c, line-by-line.
    # Sign (astronomy radial-velocity convention): +Vd_obs = atom AWAY from
    # observer (source receding) → observer sees red shift, so the
    # observer-frame line center sits at w0 + w0*Vd_obs/c.
    Vd_obs = 3.0e5  # cm/s
    atom, wMesh, atmos, _, SE_con = _hydrogen_setup(Vd_obs=Vd_obs, Vd_sun=0.0)

    Cloud_con = _CloudModel._SE_to_slab_0D_bb_(atom, atmos, SE_con, depth=1.0e3 * 1.0e5)

    nLine = atom.nLine
    checked = 0
    for k in range(nLine):
        if atom.Line["f0"][k] <= 0:
            continue
        i1 = int(wMesh.Line_mesh_idxs[k, 0])
        i2 = int(wMesh.Line_mesh_idxs[k, 1])
        w0 = float(atom.Line["w0"][k])
        dopWidth = BasicP.doppler_width_(w0, atmos.Te, atmos.Vt, atom.Mass)
        expected = wMesh.Line_mesh[i1:i2] * dopWidth + w0 + w0 * Vd_obs / CST.c_
        # atol=0 forces the comparison to be purely relative; numpy's default
        # atol=1e-8 would mask the ~1e-10 cm shift on UV lines and let a
        # missing Vd_obs term pass silently.
        assert np.allclose(Cloud_con.wl_1D[i1:i2], expected, rtol=1e-12, atol=0.0), (
            f"line {k}: wl_1D mismatch with Vd_obs formula"
        )
        checked += 1

    assert checked > 0, "no lines with f0 > 0 were checked"


def test_se_wm_cm_shifted_1d_tracks_Vd_sun():
    # Mesh-shift mechanic: the shifted-mesh debug field stores the sun-frame
    # wavelengths the atom samples = wm_cm_1d - w0 * Vd_sun / c.
    Vd_sun = 3.0e6  # cm/s
    atom, _, _, _, SE_con = _hydrogen_setup(Vd_obs=0.0, Vd_sun=Vd_sun)
    nLine = atom.nLine
    for k in range(nLine):
        if atom.Line["f0"][k] <= 0:
            continue
        i1, i2 = SE_con.Line_mesh_idxs[k]
        w0 = atom.Line["w0"][k]
        expected = SE_con.wm_cm_1d[i1:i2] - w0 * Vd_sun / CST.c_
        np.testing.assert_allclose(SE_con.se_bb_con.wm_cm_shifted_1d[i1:i2], expected, rtol=1e-10)


def test_se_solar_intensity_shifted_1d_matches_interp():
    Vd_sun = 3.0e6  # cm/s
    atom, _, _, radiation, SE_con = _hydrogen_setup(Vd_obs=0.0, Vd_sun=Vd_sun)
    nLine = atom.nLine
    for k in range(nLine):
        if atom.Line["f0"][k] <= 0:
            continue
        i1, i2 = SE_con.Line_mesh_idxs[k]
        expected = np.interp(
            SE_con.se_bb_con.wm_cm_shifted_1d[i1:i2],
            radiation.solar[0, :],
            radiation.solar[1, :],
        )
        np.testing.assert_allclose(SE_con.se_bb_con.solar_intensity_shifted_1d[i1:i2], expected, rtol=1e-10)


def test_se_jbar_robust_at_large_Vd_sun():
    # Locks the off-mesh peak-truncation fix. At Vd_sun = 2e8 cm/s, the
    # profile-shift mechanic would push the peak ~170 Doppler widths away
    # from w0 — outside the default qwing=10 mesh — and Jbar would collapse
    # to ~0 (Voigt wings * truncated mesh ≈ machine noise) on every line.
    # Mesh-shift keeps the normalized profile centered on the shifted solar
    # sample, so Jbar stays a finite, positive, physically-sourced number.
    # NOTE: tight closeness to the Vd_sun=0 baseline is NOT a valid lock —
    # the H solar spectrum changes by orders of magnitude across the ~Å shift
    # this Vd implies, especially for UV (Lyman) and Balmer-line cores. Lock
    # only the bug signature: jb finite, > 0, and not collapsed.
    atom, _, _, _, SE_con_zero = _hydrogen_setup(Vd_sun=0.0)
    _, _, _, _, SE_con_big = _hydrogen_setup(Vd_sun=2.0e8)
    nLine = atom.nLine
    checked = 0
    for k in range(nLine):
        if atom.Line["f0"][k] <= 0:
            continue
        j0 = SE_con_zero.Jbar[k]
        jb = SE_con_big.Jbar[k]
        assert np.isfinite(jb), f"Jbar non-finite at large Vd_sun on line {k}"
        assert jb > 0.0, f"Jbar collapsed to zero at large Vd_sun on line {k} (off-mesh bug?)"
        # Empirical floor: the off-mesh bug would push jb/j0 to ~machine zero
        # on ALL lines uniformly. Physical Vd_sun=2e8 keeps every H line's
        # ratio above 1e-4 (worst case observed: Ly-alpha at ~1e-3 — solar UV
        # spectrum dropoff, NOT profile truncation).
        assert jb / j0 > 1.0e-4, (
            f"line {k}: jb/j0={jb / j0:.2e} below physical floor; off-mesh profile truncation likely back"
        )
        checked += 1
    assert checked > 0, "no active lines exercised"


def test_se_wm_cm_shifted_within_backRad_at_large_Vd_sun():
    _, _, _, radiation, SE_con = _hydrogen_setup(Vd_sun=2.0e8)
    # Note: inactive-line slices in wm_cm_shifted_1d are zero by design
    # (see refactor_04.md §2.6). Filter those out before bounds-checking.
    active = SE_con.se_bb_con.wm_cm_shifted_1d[SE_con.se_bb_con.wm_cm_shifted_1d > 0]
    assert active.size > 0
    assert active.min() > radiation.solar[0, :].min(), "shifted mesh underflows backRad"
    assert active.max() < radiation.solar[0, :].max(), "shifted mesh overflows backRad"


def test_se_absorb_prof_1d_is_unshifted():
    # The canonical absorb_prof_1d is the unshifted base profile in the atom
    # rest frame. Its shape MUST be identical across two SE runs that differ
    # only in Vd_sun — Vd_sun lives only in the debug field. This is the
    # contract the cloud model relies on (cloud applies its own Vd_obs shift
    # via the output wavelength axis, never touches Vd_sun).
    _, wMesh_a, _, _, SE_con_zero = _hydrogen_setup(Vd_obs=0.0, Vd_sun=0.0)
    _, wMesh_b, _, _, SE_con_shift = _hydrogen_setup(Vd_obs=0.0, Vd_sun=1.0e6)

    assert np.array_equal(wMesh_a.Line_mesh_idxs, wMesh_b.Line_mesh_idxs)
    # rtol=1e-12, atol=0: the base profile depends only on Te, Vt, atom params,
    # all unchanged between the two runs. Any drift signals a leak of Vd_sun
    # into the unshifted field.
    assert np.allclose(SE_con_zero.absorb_prof_1d, SE_con_shift.absorb_prof_1d, rtol=1e-12, atol=0.0), (
        "absorb_prof_1d (unshifted) changed when only Vd_sun varied"
    )


def test_cloud_tau_peak_tracks_Vd_obs_only():
    # The slab tau_1D peak position (as a wavelength label in wl_1D) must
    # depend on Vd_obs (the observer-frame velocity) and NOT on Vd_sun
    # (the SE-internal shift). This guards against the original bug where the
    # cloud model paired a Vd_sun-shifted profile with a Vd_obs-shifted mesh
    # label, double-counting the line displacement and giving the wrong peak
    # position whenever Vd_sun != Vd_obs.
    Vd_obs = 3.0e5
    atom_a, wMesh_a, atmos_a, _, SE_con_a = _hydrogen_setup(Vd_obs=Vd_obs, Vd_sun=0.0)
    _atom_b, _wMesh_b, atmos_b, _, SE_con_b = _hydrogen_setup(Vd_obs=Vd_obs, Vd_sun=1.0e6)

    Cloud_a = _CloudModel._SE_to_slab_0D_bb_(atom_a, atmos_a, SE_con_a, depth=1.0e3 * 1.0e5)
    Cloud_b = _CloudModel._SE_to_slab_0D_bb_(atom_a, atmos_b, SE_con_b, depth=1.0e3 * 1.0e5)

    # Pick the strongest line in either run to keep the test robust.
    k = int(np.argmax(np.abs(Cloud_a.tau_max)))
    i1 = int(wMesh_a.Line_mesh_idxs[k, 0])
    i2 = int(wMesh_a.Line_mesh_idxs[k, 1])
    wl_peak_a = float(Cloud_a.wl_1D[i1 + int(np.argmax(np.abs(Cloud_a.tau_1D[i1:i2])))])
    wl_peak_b = float(Cloud_b.wl_1D[i1 + int(np.argmax(np.abs(Cloud_b.tau_1D[i1:i2])))])

    w0 = float(atom_a.Line["w0"][k])
    expected_peak = w0 + w0 * Vd_obs / CST.c_

    # Semantic peak-side check: under the astronomy radial-velocity convention,
    # +Vd_obs (atom receding) must move the peak to the RED side of w0. Catches
    # double-negative typos (formula + comment flipped together) with a clearer
    # failure than an argmax-index mismatch.
    assert expected_peak > w0, f"expected_peak {expected_peak} should be > w0 {w0} for +Vd_obs"
    assert wl_peak_a > w0, f"measured peak {wl_peak_a} should be > w0 {w0} for +Vd_obs"
    assert wl_peak_b > w0, f"measured peak {wl_peak_b} should be > w0 {w0} for +Vd_obs"

    # Both runs share the same Vd_obs; their peak wavelength labels must agree
    # tightly and equal w0 + w0*Vd_obs/c. Tolerance is one mesh spacing (the
    # argmax can only resolve to the nearest sample).
    dop = BasicP.doppler_width_(w0, atmos_a.Te, atmos_a.Vt, atom_a.Mass)
    mesh_step = float(wMesh_a.Line_mesh[i1 + 1] - wMesh_a.Line_mesh[i1]) * dop
    assert abs(wl_peak_a - expected_peak) <= mesh_step, (
        f"Vd_sun=0 cloud tau peak {wl_peak_a} not within one mesh step of {expected_peak}"
    )
    assert abs(wl_peak_b - expected_peak) <= mesh_step, (
        f"Vd_sun!=0 cloud tau peak {wl_peak_b} not within one mesh step of {expected_peak}"
    )


def test_cloud_tau_peak_negative_Vd_obs_blue_shift():
    # Symmetric coverage of the positive-Vd test: with Vd_obs = -3 km/s
    # (atom approaching observer), the observer-frame peak must sit on the
    # BLUE side of w0. Guards against an off-by-sign bug that a +Vd-only
    # test would not catch (e.g. an abs(Vd_obs) creeping in).
    Vd_obs = -3.0e5  # cm/s, -3 km/s
    atom, wMesh, atmos, _, SE_con = _hydrogen_setup(Vd_obs=Vd_obs, Vd_sun=0.0)
    Cloud = _CloudModel._SE_to_slab_0D_bb_(atom, atmos, SE_con, depth=1.0e3 * 1.0e5)

    k = int(np.argmax(np.abs(Cloud.tau_max)))
    i1 = int(wMesh.Line_mesh_idxs[k, 0])
    i2 = int(wMesh.Line_mesh_idxs[k, 1])
    wl_peak = float(Cloud.wl_1D[i1 + int(np.argmax(np.abs(Cloud.tau_1D[i1:i2])))])
    w0 = float(atom.Line["w0"][k])
    expected_peak = w0 + w0 * Vd_obs / CST.c_

    assert expected_peak < w0, f"expected_peak {expected_peak} should be < w0 {w0} for -Vd_obs"
    assert wl_peak < w0, f"measured peak {wl_peak} should be < w0 {w0} for -Vd_obs"

    dop = BasicP.doppler_width_(w0, atmos.Te, atmos.Vt, atom.Mass)
    mesh_step = float(wMesh.Line_mesh[i1 + 1] - wMesh.Line_mesh[i1]) * dop
    assert abs(wl_peak - expected_peak) <= mesh_step, (
        f"-Vd_obs cloud tau peak {wl_peak} not within one mesh step of {expected_peak}"
    )


def test_atmos0d_default_doppler_zero():
    # Atmosphere0D.Vd_obs / Vd_sun default to 0.0 — a construction with
    # only the non-Doppler kwargs must yield zero Doppler velocities.
    # Also assert non-Doppler kwargs round-trip correctly (Vt, Te), so a
    # silent field reorder that re-aliased Vt to a default would fail
    # loudly rather than producing a confusing downstream physics result.
    Vt_in = 4.2e5
    Te_in = 7.5e3
    atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=Te_in, Vt=Vt_in)
    assert atmos.Vd_obs == 0.0
    assert atmos.Vd_sun == 0.0
    assert atmos.Vt == Vt_in
    assert atmos.Te == Te_in


def test_se_container_no_mesh_fields():
    # The wave_mesh_shifted_1d / cont_wave_mesh_shifted attributes were
    # removed when shift moved from mesh-shift to profile-shift; ensure they
    # don't sneak back via the dataclass.
    _, _, _, _, SE_con = _hydrogen_setup()
    assert not hasattr(SE_con, "wave_mesh_shifted_1d")
    assert not hasattr(SE_con, "cont_wave_mesh_shifted")


def test_init_VAL_doppler_fields():
    # VAL atmospheres must expose both velocity fields, shape-aligned with
    # Vt, and default to all-zero (VAL data has no per-depth flow velocity).
    atmos = Atmosphere.init_VAL_("C")
    assert atmos.Vd_sun.shape == atmos.Vt.shape
    assert atmos.Vd_obs.shape == atmos.Vt.shape
    assert (atmos.Vd_sun == 0).all()
    assert (atmos.Vd_obs == 0).all()


def test_init_FAL_doppler_fields():
    # FAL loader must populate Vd_sun from the file's vertical flow velocity
    # column and leave Vd_obs at zero (observer-frame geometry is caller's
    # responsibility). Vd_sun all-zero is not asserted — FAL files can carry
    # real flow velocities; FALC_82 happens to be hydrostatic.
    if not _FAL_PATH.exists():
        pytest.skip("no FAL fixture")
    _, atmos = ExFAL.init_FAL_(str(_FAL_PATH))
    assert atmos.Vd_sun.shape == atmos.Vt.shape
    assert atmos.Vd_obs.shape == atmos.Vt.shape
    assert (atmos.Vd_obs == 0).all()


def test_jit_smoke_b_jbar_nonzero_vd_sun():
    # Numba-compiled _B_Jbar_ must accept Vd_sun != 0 without compile errors
    # and yield a normalized n_SE. Only meaningful when JIT is enabled.
    if not CFG._IS_JIT:
        pytest.skip("JIT disabled; numba codepath not exercised")
    _, _, _, _, SE_con = _hydrogen_setup(Vd_obs=0.0, Vd_sun=5.0e5)
    total = float(SE_con.n_SE.sum())
    assert np.isfinite(total)
    assert np.isclose(total, 1.0, rtol=0.05)
