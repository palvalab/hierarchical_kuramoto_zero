
import numpy as np

from bisect import bisect_left

from typing import Tuple, Sequence, Optional

from .utils import get_array_module
from .stats import fit_tukey, fit_weighted


def _dfa_boxcar(data_orig, win_lengths, xp):
    '''            
    Computes DFA using FFT-based method. (Nolte 2019 Sci Rep)
    Input: 
        data_orig:   1D array of amplitude time series.
        win_lenghts: 1D array of window lengths in samples.
    Output:
        fluct: Fluctuation function.
        slope: Slopes.
    
    '''
    data = xp.array(data_orig, copy=True)
    win_arr = xp.array(win_lengths)
    
    data -= data.mean(axis=1, keepdims=True)
    data_fft = xp.fft.fft(data)

    n_chans, n_ts = data.shape
    is_odd = n_ts % 2 == 1

    nx = (n_ts + 1)//2 if is_odd else n_ts//2 + 1
    data_power = 2*xp.abs(data_fft[:, 1:nx])**2

    if is_odd == False:
        data_power[:,~0] /= 2
        
    ff = xp.arange(1, nx)
    g_sin = xp.sin(xp.pi*ff/n_ts)
    
    hsin = xp.sin(xp.pi*xp.outer(win_arr, ff)/n_ts)
    hcos = xp.cos(xp.pi*xp.outer(win_arr, ff)/n_ts)

    hx = 1 - hsin/xp.outer(win_arr, g_sin)
    h = (hx / (2*g_sin.reshape(1, -1)))**2

    f2 = xp.inner(data_power, h)

    fluct = xp.sqrt(f2)/n_ts

    hy = -hx*(hcos*xp.pi*ff/n_ts - hsin/win_arr.reshape(-1,1)) / xp.outer(win_arr, g_sin)
    h3 = hy/(4*g_sin**2)

    slope = xp.inner(data_power, h3) / f2*win_arr
    
    return fluct, slope


def dfa(data: np.ndarray, window_lengths: Sequence, method: str='boxcar', 
            use_gpu: bool=False, fitting ='Tukey', weighting = 'sq1ox') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:    
    """
    Compute DFA with conventional (windowed) or FFT-based 'boxcar' method.
    
    INPUT:
        data:           2D array of size [N_channels x N_samples]. Should be normalized data!!
        window_lengths: sequence of window sizes, should be in samples.
        method:         either 'conv' or 'boxcar' 
        use_gpu:        If True, input np.array is converted to cp in function
        fitting:        'linfit' for regular unweighted linear fit, 
                        'Tukey' for biweight/bisquare,
                        'weighted' for weighted linear fit.
        weighting:      'sq1ox' or '1ox' 
                    
    OUTPUT:        
        fluctuation: 2D array of size N_channels x N_windows), 
        slope:       2D array of size N_channels x N_windows), 
        DFA:         1D vector of size N_channels 
        residuals:   1D vector of size N_channels        
    """

    module = get_array_module(data)
    
    N_samp = data.shape[1]
    
    allowed_methods = ('boxcar','conv' )
    if not(method in allowed_methods):
        raise RuntimeError('Method {} is not allowed! Only {} are available'.format(method, ','.join(allowed_methods)))

    allowed_weightings = ('sq1ox', '1ox')
    if not(weighting in allowed_weightings):
        raise RuntimeError('Weighting {} is not allowed! Only {} are available'.format(weighting, ','.join(allowed_weightings)))

    fluct, slope =  _dfa_boxcar(data, window_lengths, xp=module) 
        
    if not(module is np):
        fluct = module.asnumpy(fluct)
        slope = module.asnumpy(slope)    
    
    if fitting == 'weighted':
        dfa_values, residuals = fit_weighted(weighting, fluct, N_samp, window_lengths)

    elif fitting == 'Tukey':
        dfa_values, residuals = fit_tukey(weighting, fluct, N_samp, window_lengths)
        
    elif fitting == 'linfit':           
        dfa_values, residuals = np.polyfit(np.log2(window_lengths), np.log2(fluct.T), 1)
    
    return fluct, slope, dfa_values, residuals


def _cdf_na(beta: float, l: float, L: float) -> float:
    """Theoretical CDF F^{NA}(beta) for a -3/2 power-law PDF on [l, L]."""
    if beta <= l:
        return 0.0
    if beta >= L:
        return 1.0
    denom = (l ** -0.5) - (L ** -0.5)
    if denom == 0:
        return np.nan
    return ((l ** -0.5) - (beta ** -0.5)) / denom

def kappa_k(sizes, m: int = 10, bounds = None) -> float:
    """
    κ (K measure) from Shew et al., J. Neurosci. (2009)

    Args:
        sizes: Positive cluster/avalanche sizes. Non-positive entries are ignored.
        m: Number of logarithmically spaced probe points β_k between l and L (default 10).
        bounds: Optional (l, L) to set the reference power-law support. By default l=min(sizes), L=max(sizes).

    Returns:
        k (< 1 subcritical; ~ 1 critical;  > 1 supercritical).
    """
    xs = [float(x) for x in sizes if x is not None and x > 0 and np.isfinite(float(x))]
    if len(xs) < 2:
        return np.nan
    xs.sort()
    l = bounds[0] if bounds is not None else xs[0]
    L = bounds[1] if bounds is not None else xs[-1]
    if l <= 0 or L <= 0 or L <= l:
        return np.nan

    if m < 2:
        m = 2
    log_l = np.log(l)
    log_L = np.log(L)
    betas = [np.exp(log_l + (log_L - log_l) * k / (m - 1)) for k in range(m)]

    n = len(xs)
    total = 0.0
    for beta in betas:
        count_lt = bisect_left(xs, beta) 
        F_emp = count_lt / n
        F_na = _cdf_na(beta, l, L)
        total += (F_na - F_emp)

    return 1.0 + (total / m)