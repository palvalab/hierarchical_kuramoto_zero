import numpy as np

def detect_spikes_phase_crossing(data, target_phase=0):
    data_shifted = data*np.exp(1j*target_phase)
    data_phase = np.angle(data_shifted)
    phase_diff_non_circ = np.diff(data_phase, axis=-1, prepend=1)
    spike_events = (phase_diff_non_circ < 0)

    return spike_events

# @numba.jit
def extract_avalanche_properties_spiking(data_spikes, t=None, use_total=False):
    if (t is None):
        t = np.median(data_spikes, axis=-1, keepdims=True)
    
    avalanche_lengths_channelwise = [list() for i in range(data_spikes.shape[0])]
    avalanche_sizes_channelwise = [list() for i in range(data_spikes.shape[0])]
    
    for i in range(data_spikes.shape[0]):
        avalanche_lengths_list = list()
        avalanche_sizes_list = list()
        
        running_size = 0
        running_length = 0

        for j in range(data_spikes.shape[1]):
            if data_spikes[i,j] > t[i]:
                if use_total:
                    running_size += data_spikes[i,j]
                else:
                    running_size += data_spikes[i,j] - t[i]
        
                running_length += 1
            else:
                if running_length > 0:
                    avalanche_lengths_list.append(running_length)
                    avalanche_sizes_list.append(running_size)
                
                running_size = 0
                running_length = 0

        if running_length > 0:
            avalanche_lengths_list.append(running_length)
            avalanche_sizes_list.append(running_size)
        
        avalanche_lengths_channelwise[i] = avalanche_lengths_list
        avalanche_sizes_channelwise[i] = avalanche_sizes_list

    return avalanche_lengths_channelwise, avalanche_sizes_channelwise