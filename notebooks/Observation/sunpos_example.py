
from sunposition import sunpos
import datetime

import numpy as np

def parallactic_angle(_lat, _ha, _zenith, _isRad=False):
    r"""

    Parameters
    ----------
    _lat : float, [:math:`rad`] if _isRad else [:math:`deg`]
        latitude of the telescope

    _ha : float, [:math:`rad`] if _isRad else [:math:`deg`]
        Solar hour angle

    _zenith : float, [:math:`rad`] if _isRad else [:math:`deg`]
        zenith distance

    _isRad : bool,
        whether the unit of input arguments are `rad`

    Returns
    ----------
    _pa, float, [:math:`rad`] if _isRad else [:math:`deg`]

    Notes
    --------
    Makita, M., Funakoshi, Y., Tomura, I., Kawakami, S.,Hanaoka, Y., Kawai, G.,
    1996, Technical Reports from Kwasan and Hida Observatories Faculty of science,
    Kyoto University, 7: 1-27

    """
    _deg2rad = np.pi / 180.
    if not _isRad:
        _lat_rad = _lat * _deg2rad
        _ha_rad  = _ha  * _deg2rad
        _zenith_rad = _zenith * _deg2rad
    else:
        _lat_rad = _lat
        _ha_rad  = _ha
        _zenith_rad = _zenith

    _sin_pa = np.cos(_lat_rad) * np.sin(_ha_rad) / np.sin(_zenith_rad)
    _pa_rad = np.arcsin( _sin_pa )

    if not _isRad:
        _pa = _pa_rad / _deg2rad
    else:
        _pa = _pa_rad
    return _pa



if __name__ == "__main__":

    args = {
        't'   : None,        # UTC
        'lat' : 36.15,       # [deg]
        'lon' : 137.36,      # [deg]
        'elev': 1300,        # [m]
        'temp': 21.71,       # [deg C]
        'p'   : 823.4,       # [mbar]
        'dt'  : 0.0,         # time shift, fix to 0.
        'rad' : False        # output [rad] instead of [deg]
    }

    args['t'] = datetime.datetime(2020,12,4,1,30,0) # Year, Month, Date, Hour, Minute, Second

    # unit : args['rad']==False --> [deg]
    # az : observed azimuth angle, measured eastward from north
    # zen: observed zenith angle, measured down from vertical
    # ra : topocentric right ascension
    # dec: topocentric declination
    # h  : topocentric hour angle
    az, zen, ra, dec, h =     sunpos(args['t'], args['lat'], args['lon'],
                                     args['elev'], args['temp'], args['p'],
                                     args['dt'], args['rad'])
    pa = parallactic_angle(args['lat'], h, zen, _isRad=args['rad'])

    print(az, zen, ra, dec, h, pa)
