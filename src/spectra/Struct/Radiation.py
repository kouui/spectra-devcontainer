# -------------------------------------------------------------------------------
# definition of struct for storing radiation
# -------------------------------------------------------------------------------
# VERSION
# 0.1.0
#    2021/05/18   u.k.   spectra-re
# 0.1.1
#    2022/01/07   u.k.   modified atlas(backRad)
# 0.1.2
#    2022/08/01   u.k.   modified atlas(added 10000-11000A absorption lines into backRad)
# -------------------------------------------------------------------------------


from dataclasses import dataclass as _dataclass

from ..ImportAll import *


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class Radiation:
    backRad: T_ARRAY  # 2d, (2, n_wavelength)

    PI_intensity: T_ARRAY  # 2d, (nCont, _N_CONT_MESH )


import numpy as _numpy

from ..Atomic import LTELib as _LTELib
from ..Atomic import PhotoIonize as _PhotoIonize
from . import Atmosphere as _Atmosphere
from . import WavelengthMesh as _WavelengthMesh


def init_Radiation_(
    atmos: T_UNION[_Atmosphere.Atmosphere0D, _Atmosphere.AtmosphereC1D], wMesh: _WavelengthMesh.Wavelength_Mesh
) -> Radiation:

    backRad = _numpy.load(CFG._ROOT_DIR / "data" / "intensity" / "atlas" / "QS" / "atlas_QS.20221118.npy")
    # backRad[0,:] *= 1E-8
    # backRad[1,:] *= 2.5*intensity_fac

    Cont_mesh = wMesh.Cont_mesh
    if atmos.use_Tr:
        Tr = atmos.Tr
        PI_intensity = _LTELib.planck_cm_(Cont_mesh[:, :], Tr)
    else:
        #  TODO : average backRad for PI_intensity ?
        PI_intensity = _PhotoIonize.interpolate_PI_intensity_(backRad[:, :], Cont_mesh[:, :])

    radiation = Radiation(
        backRad=backRad,
        PI_intensity=PI_intensity,
    )

    return radiation
