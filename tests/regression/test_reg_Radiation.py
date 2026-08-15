"""Regression tests for the shipped solar atlas and its load-time NaN fill.

The atlas is read only through ``numpy.interp``, which does not skip NaN: a
query in an interval *adjacent* to a NaN sample returns NaN, and that NaN
propagates through Jbar into the SE rate matrix, making every level population
of the atom NaN. ``init_Radiation_`` therefore bridges the bad samples on load.

These tests pin the defect itself, not just the fix. If a re-issued atlas grows
a new NaN sample, ``test_atlas_nan_samples_are_the_known_four`` fails loudly
instead of the fill silently absorbing it -- a wider gap may need real data
rather than linear interpolation.
"""

import numpy as np
import pytest

from spectra.ImportAll import CFG
from spectra.Struct import Radiation

_ATLAS_PATH = CFG._ROOT_DIR / "data" / "intensity" / "atlas" / "QS" / "atlas_QS.20221118.npy"

# Air wavelengths [AA] of every NaN intensity sample in the shipped atlas.
_KNOWN_NAN_AA = np.array([1799.875, 2201.125, 2245.375, 2707.625])

# The atlas grid step is 0.25 AA; 1e-3 pins the sample without pinning float noise.
_WAVELENGTH_ATOL_AA = 1.0e-3


@pytest.fixture(scope="module")
def raw_atlas() -> np.ndarray:
    """The atlas exactly as stored on disk, before ``init_Radiation_`` fills it."""
    return np.load(_ATLAS_PATH)


def test_atlas_nan_samples_are_the_known_four(raw_atlas: np.ndarray) -> None:
    bad_AA = raw_atlas[0, np.isnan(raw_atlas[1, :])] * 1.0e8

    assert bad_AA.size == _KNOWN_NAN_AA.size, (
        f"expected {_KNOWN_NAN_AA.size} NaN samples in the atlas, found {bad_AA.size} at {bad_AA} AA"
    )
    assert np.allclose(np.sort(bad_AA), _KNOWN_NAN_AA, atol=_WAVELENGTH_ATOL_AA, rtol=0.0)


def test_atlas_wavelength_axis_has_no_nan(raw_atlas: np.ndarray) -> None:
    # The fill interpolates *against* row 0, so a NaN there would poison every
    # filled value rather than a single sample.
    assert not np.isnan(raw_atlas[0, :]).any()


def test_init_Radiation_fills_every_nan() -> None:
    assert not np.isnan(Radiation.init_Radiation_().solar).any()


def test_fill_stays_between_its_neighbours(raw_atlas: np.ndarray) -> None:
    # Linear interpolation is only defensible while the gaps stay one sample
    # wide; a bracketing check catches a widened gap that a NaN count would not.
    solar = Radiation.init_Radiation_().solar
    for i in np.flatnonzero(np.isnan(raw_atlas[1, :])):
        lo, hi = sorted((raw_atlas[1, i - 1], raw_atlas[1, i + 1]))
        assert lo <= solar[1, i] <= hi
