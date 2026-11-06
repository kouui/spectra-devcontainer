# -------------------------------------------------------------------------------
# function definition of basic math
# -------------------------------------------------------------------------------

from ..ImportAll import *


@nb_vec(**NB_VEC_KWGS)
def is_odd_(n: T_INT) -> T_INT:
    """A fast method to check whether a number is odd.

    Parameters
    ----------
    n : T_INT
        input integer number

    Returns
    -------
    T_INT
        1 if input number is an odd number, otherwise 0
    """
    return n & 0x1
