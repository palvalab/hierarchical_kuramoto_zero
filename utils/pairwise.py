from .utils import NDArray, get_array_module, _normalize_signal

def cplv(x: NDArray, y: Optional[NDArray]=None, is_normed: bool=False) -> NDArray:
    """
        computes cPLV either for a given pair of complex signals or for all possible pairs of channels if x is 2d matrix
    :param x: 1d or 2d array of complex values
    :param y: Optional, 1d or 2d array of complex values
    :return: complex phase locking value
    """
    xp = get_array_module(x)

    n_ts = x.shape[1]
    
    if is_normed:
        x_norm = x
        y_norm = x_norm if y is None else y
    else:
        x_norm = _normalize_signal(x)
        y_norm = x_norm if y is None else _normalize_signal(y)

    avg_diff = xp.inner(x_norm, xp.conj(y_norm)) / n_ts

    return avg_diff