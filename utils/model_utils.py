import numpy as np
import cupy as cp
import scipy as sp

from .kuramoto import KuramotoFast

def simulate_single_run(k, node_frequencies, time=350, n_oscillators=500, frequency_spread=3, normalize_external=False, aggregate='mean',
                       noise_sigma=3.0, sr=250, weight_matrix=None, omegas=None, use_tqdm=True, omega_seed=42, random_seed=42):
    np.random.seed(random_seed)
    cp.random.seed(random_seed)
    n_nodes = len(node_frequencies)
    if weight_matrix is None:
        w = np.zeros((n_nodes, n_nodes))
    else:
        w = weight_matrix.copy()
        
    if not(type(k) in (list, np.ndarray)):
        k_values = [k]*n_nodes
    else:
        k_values = k
        

    model = KuramotoFast(n_oscillators=n_oscillators, sampling_rate=sr,
                 k_list=k_values, weight_matrix=w, use_tqdm=use_tqdm,
                 node_frequencies=node_frequencies,
                 scale=frequency_spread, custom_omegas=omegas, noise_scale=noise_sigma, use_cuda=True)

    return model.simulate(time=time, random_seed=random_seed, aggregate=aggregate)
