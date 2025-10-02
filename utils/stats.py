import statsmodels.api as sm
from scipy.optimize import curve_fit

import numpy as np
import scipy as sp
import scipy.signal

from typing import Tuple, Sequence, Optional

class CustomNorm(sm.robust.norms.TukeyBiweight):
    ''' 
    Custom Norm Class using Tukey's biweight (bisquare).
    '''
    
    def __init__(self, weights, c=4.685, **kwargs):
        super().__init__(**kwargs)
        self.weights_vector = np.array(weights)
        self.flag = 0
        self.c = c
        
    def weights(self, z):
        """
            Instead of weights equal to one return custom
        INPUT:
            z : 1D array or list
        OUTPUT:
            weights: ndarray
        """
        if self.flag == 0:
            self.flag = 1
            return self.weights_vector.copy()
        else:
            subset = self._subset(z)
            return (1 - (z / self.c)**2)**2 * subset



def _tukey_regression(x,y,weights):
    '''
    Fit using Tukey's biweight function.    
    '''
    
    N = len(y)
    X = np.array([[1, x[i]] for i in range(N)])
    
    rlm_model = sm.RLM(y,X,M=CustomNorm(weights=weights,c=4.685))
    rlm_results = rlm_model.fit()
    
    b,a = rlm_results.params    
    return a,b

def fit_tukey(weighting: str, fluct: np.ndarray, N_samp: int, window_lengths: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    # I am not sure if this is the correct way to get weights for Tukey!
    N_CH = fluct.shape[0]
    if weighting == 'sq1ox':
        weights = [np.sqrt(1/x) for x in N_samp/window_lengths]
    elif weighting == '1ox':
        weights = [(1/x) for x in N_samp/window_lengths]  
        
    dfa_values = np.zeros(N_CH)
    residuals  = np.zeros(N_CH)  
    x = np.log2(window_lengths)         
    for i in range(N_CH):
        y = np.log2(fluct[i])
        dfa_values[i], residuals[i] = _tukey_regression(x,y,weights)

    return dfa_values, residuals

def fit_weighted(weighting: str, fluct: np.ndarray, N_samp: int, window_lengths: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    
    N_CH = fluct.shape[0]
    if weighting == 'sq1ox':
        sigma = [np.sqrt(1/x) for x in N_samp/window_lengths]
    elif weighting == '1ox':
        sigma = [(1/x) for x in N_samp/window_lengths]
    
    dfa_values = np.zeros(N_CH)
    residuals  = np.zeros(N_CH)        
    p0 = 0.7,-8                    # might have to be parametrized? or not?     
    x=np.log2(window_lengths) 
    for i in range(N_CH):
        y=np.log2(fluct[i])
        popt, pcov = curve_fit(f, x , y, p0, sigma = sigma,absolute_sigma=True)
        dfa_values[i], residuals[i] = popt

    return dfa_values, residuals