# -------------------------------------------------------------------------------
# definition of struct for storing radiation
# -------------------------------------------------------------------------------


from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path

import numpy as _numpy

from ..ImportAll import *


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class Radiation:
    solar: T_ARRAY  # 2d, (2, n_wavelength); row 0 wavelength [cm], row 1 intensity [erg/cm^2/Sr/cm/s]


def init_Radiation_(path: _Path | None = None) -> Radiation:
    """Load the solar atlas spectrum into a Radiation struct.

    Continuum PI intensity is built on demand by `cal_SE_` from `radiation.solar`
    and `wMesh.Cont_mesh` (or from `planck(Tr)` when `se_params.Tr is not None`);
    it is not cached here.

    NaN intensity samples in the atlas are bridged linearly on load. The atlas
    is consumed exclusively through `numpy.interp`, which does not treat NaN as
    a gap: a query landing in an interval that merely *touches* a NaN sample
    returns NaN. That NaN then travels Jbar -> the SE rate matrix -> a linear
    solve, so a single bad sample turns *every* level population of the atom
    into NaN. Which lines reach a bad sample depends on the Doppler width, so
    left unfilled the defect surfaces as a silent blow-up above some threshold
    in Te/Vt rather than as a localized wavelength artifact.
    """
    if path is None:
        path = CFG._ROOT_DIR / "data" / "intensity" / "atlas" / "QS" / "atlas_QS.20221118.npy"
    solar: T_ARRAY = _numpy.load(path)

    bad = _numpy.isnan(solar[1, :])
    if bad.any():
        solar[1, bad] = _numpy.interp(solar[0, bad], solar[0, ~bad], solar[1, ~bad])

    return Radiation(solar=solar)
