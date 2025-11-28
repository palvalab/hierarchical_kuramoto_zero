import numpy as np
import numpy.typing as npt

import scipy as sp

from typing import List, Union, Any, Optional, TypeVar

T = TypeVar("T", bound=np.generic)

try:
    import cupy as cp
    import cupy.typing as cpt
    HAS_CUPY = True

    # NDArray can be either a NumPy or CuPy array of dtype T
    NDArray = Union[npt.NDArray[T], cpt.NDArray[T]]
except Exception:
    # cupy not available
    HAS_CUPY = False
    NDArray = npt.NDArray[T]

def get_array_module(arr):
    if type(arr) is np.ndarray:
        return np
    elif HAS_CUPY:
        return cp
    else:
        print(type(arr))
        raise RuntimeError('Unknown array type!')