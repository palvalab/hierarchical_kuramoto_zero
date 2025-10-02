def extract_avalanche_properties_spiking(data_spikes, t=None):
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
            if data_spikes[i,j] >= t[i]:
                running_size += data_spikes[i,j]
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
        
        avalanche_lengths_channelwise[i].extend(avalanche_lengths_list)
        avalanche_sizes_channelwise[i].extend(avalanche_sizes_list)

    return avalanche_lengths_channelwise, avalanche_sizes_channelwise