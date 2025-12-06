# Imports:
import pickle
import os
import re
import glob 

import argparse
from pathlib import Path

import numpy as np
import scipy as sp
import cupy as cp

from utils.model_utils import simulate_single_run
# from utils.pyutils import dfa, cplv, compute_avalanche_stats
from utils.pairwise import cplv
from utils.kuramoto import KuramotoFast
from utils.criticality import kappa_k, dfa
from utils.avalanches import extract_avalanche_properties_spiking

PROJECT_CODE = os.environ['PROJECT_CODE']

def shuffle_connectome(sc):
    res = sc.copy().flatten()

    np.random.shuffle(res)

    res = res.reshape(sc.shape)

    res = (res + res.T)/2

    return res

def mean_norm_connectome(sc):
    return sc / sc.mean()

def log_scale_connectome(sc):
    return np.log(sc + 1)

def load_connectome(subject_id, preprocessing_list = list()):
    subject_path = f'/projappl/{PROJECT_CODE}/kuramoto_fitting/SCs/controls/sub-CON{subject_id}_yeo17_200Parcels_for_fit'
    
    with open(subject_path, 'rb') as f:
        subject = pickle.load(f)

    orig_sc = subject["connectome"] 

    sc_transformed = orig_sc.copy()

    for p in preprocessing_list:
        sc_transformed = p(sc_transformed)
    
    return sc_transformed

def main(args):
    ###### 
    # KURAMOTO PARAMETERS:
    n_nodes = 200
    num_oscillators = 500
    sr = 250
    f_nodes = [10] * n_nodes
    f_spread = 1.0
    noise_sigma = 3.0
    time_s = 360
    cutoff_time = 60

    window_sizes = np.geomspace(20 / 10 * sr, (time_s - cutoff_time) * sr * 0.2, 30).astype(int)
    freqs =  [10] * n_nodes
    ###### 
    
    nk = 40

    k_values = np.linspace(0, 8, nk)
    l_values = np.linspace(0, 8, nk)

    start_size = args.kl_idx * args.grid_size

    subject_indices = sorted([re.search(r'CON(\d+)', fpath).group(1) for fpath in glob.glob(f'/projappl/{PROJECT_CODE}/kuramoto_fitting/SCs/controls/*')])

    for subject_idx in subject_indices:               
        connectome = load_connectome(subject_idx, [mean_norm_connectome])
        sim = f'{subject_idx}'
        root_dir = f'/scratch/{PROJECT_CODE}/pnas_sims_new/{sim}'
        
        print(f'Starting a simulation with sim={sim}, start_idx={start_size}')

        if args.data_psd:
            psd_spectra = pickle.load(open(f'data/sub-{args.subject}_psd.pickle', 'rb'))
            freqs_sampled = np.array([np.random.choice(psd_spectra['frequencies'], size=num_oscillators, 
                                    p=psd_spectra['foofed_spectra'][chan_idx]) for chan_idx in range(n_nodes)])
        else:
            freqs_sampled = None
    
        for current_idx in range(start_size, start_size + args.grid_size):
            k_idx = current_idx % k_values.shape[0]
            l_idx = current_idx // l_values.shape[0]
            
            if (k_idx >= k_values.shape[0]) or (l_idx >= l_values.shape[0]):
                continue
    
            k = k_values[k_idx]
            l = l_values[l_idx]

            if args.aggregate == 'mean':
                aggregator_func = 'mean'
            elif args.aggregate == 'avalanches':
                aggregator_func = SpikeAggregator()
            else:
                raise RuntimeError(f'Unknown aggregator: {args.aggregate}')

            observables_fname = f'observables_{args.aggregate}_subject_{sim}_K-{k}_L-{l}.npy'

            observables_dir = os.path.join(root_dir, 'observables')
            observables_fpath = os.path.join(observables_dir, observables_fname)

            # if os.path.exists(observables_fpath):
            #     continue
    
            print(f'Running a simulation with K={k}, L={l}')
            
            mdl = KuramotoFast(num_oscillators, sr, [k]*n_nodes, connectome*l, f_nodes, noise_sigma, scale=f_spread, 
                                use_tqdm=False, custom_omegas=freqs_sampled)
            data_sim = mdl.simulate(time_s, aggregate=aggregator_func)
            data_sim = data_sim[..., cutoff_time*sr:]


            if args.aggregate == 'mean':
                data_envelope = np.abs(data_sim)
    
                model_order_vals = data_envelope.mean(axis=-1)
                model_dfa_vals = dfa(data_envelope, window_sizes, use_gpu=True)[2]
                model_cc_vals = np.corrcoef(data_envelope)
                model_plv_vals = np.abs(cplv(data_sim))

                model_psd_f, model_psd_vals = sp.signal.welch(data_sim.real, fs=sr, nperseg=256*4)

                model_observables = {'cc_values': model_cc_vals, 'plv_values': model_plv_vals,
                                     'order': model_order_vals, 'dfa': model_dfa_vals,
                                     'psd_freqs': model_psd_f, 'psd_values': model_psd_vals}
            elif args.aggregate == 'avalanches':
                _, avalanche_sizes = extract_avalanche_properties_spiking(data_sim.real)

                sizes_kappa_vals = np.array([kappa_k(node_avalanche_sizes) for node_avalanche_sizes in avalanche_sizes])

                model_observables = {'kappa_sizes': sizes_kappa_vals}
    
            Path(observables_dir).mkdir(parents=True, exist_ok=True)
    
            pickle.dump(model_observables, open(observables_fpath, 'wb'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-kl_idx', '--kl_idx', type=int)  
    parser.add_argument('-grid_size', '--grid_size', type=int)
    parser.add_argument('-data_psd', '--data_psd', action='store_true')  
    parser.add_argument('-aggregate', '--aggregate', type=str, default='mean')
     
    args = parser.parse_args()

    main(args)
