import statsmodels.api as sm
from scipy.optimize import curve_fit

import math


import numpy as np
import scipy as sp
import scipy.signal

from typing import Tuple, Sequence, Optional

try: 
    import cupy as cp
    HAS_CUPY = True
except:
    HAS_CUPY = False

import numpy as np

def get_array_module(arr):
    if type(arr) is np.ndarray:
        return np
    elif HAS_CUPY:
        return cp
    else:
        print(type(arr))
        raise RuntimeError('Unknown array type!')
        

def _normalize_signal(x: np.ndarray) -> np.ndarray:
    eps = 1e-10
    xp = get_array_module(x)

    x_abs = xp.abs(x)

    x_norm = x.copy()
    x_norm /= (x_abs + eps)

    return x_norm
