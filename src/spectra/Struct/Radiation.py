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
    """
    if path is None:
        path = CFG._ROOT_DIR / "data" / "intensity" / "atlas" / "QS" / "atlas_QS.20221118.npy"
    return Radiation(solar=_numpy.load(path))
