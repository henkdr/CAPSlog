from calc_mem_stats import partitionings_to_cutpoints, \
    find_mem_isolated, find_mem_added, get_mem_stats, do_completeness_check, print_results
import numpy as np


def evenly_distribute(n_gpus, n_layers):
    if n_gpus == 0 or n_layers == 0:
        return []

    partitioning = []
    
    l_per_gpu = n_layers // n_gpus
    rest = n_layers % n_gpus

    # Give all but last gpu l_per_gpu layers (+ 1 of the rest if necessary)
    for gpu in range(n_gpus):
        if gpu < rest:
            partitioning.append(l_per_gpu+1)
        else:
            partitioning.append(l_per_gpu)

    return partitioning

# Fill the given n_gpus, by placing l_per_gpu layers on each GPU as
# long as there are layers. The remainder fgoes to the last GPU(s).
def evenly_distribute_fill(n_gpus, n_layers, l_per_gpu):
    if n_gpus == 0 or n_layers == 0:
        return []

    partitioning = []
    
    # Give all but last gpu l_per_gpu layers (+ 1 of the rest if necessary)
    for gpu in range(n_gpus):
        if n_layers >= l_per_gpu:
            partitioning.append(l_per_gpu)
            n_layers -= l_per_gpu
        else:
            partitioning.append(n_layers)
            n_layers -= n_layers

    return partitioning

def fill(n_gpus, l_per_gpu):
    return [l_per_gpu] * n_gpus

def get_mCAP_partitionings(n_gpus, n_layers):
    if n_layers < 2 * n_gpus - 2:
        raise ValueError("For profiling, n_layers must be >= 2 * n_gpus - 2")

    partitionings = []

    l_per_gpu = (n_layers - 2) // (n_gpus - 2) + 1
    # print("l_per_gpu", l_per_gpu)

    # Go through the layers and generate a set of partitionings
    # with for each layer 'l' at least partitionings where:
    # 1. The layer is isolated on a GPU.
    # 2. There is a set of layers [l-n, l-1] on a GPU.
    # 3. There is a set of layers [l-n, l] on a GPU.
    # Mem isolated and mem added can be extracted from this
    # set of 'profiling partitionings'.
    for layer in range(1, n_layers - 1):
        gpu = (layer // l_per_gpu)
        partitioning = evenly_distribute_fill(gpu+1, layer, l_per_gpu)

        # Special case: add one extra layer to last gpu and add single-layer GPU after that:
        if partitioning[-1] == 0:
            partitioning[-2] += 1
            partitioning[-1] = 1

        partitioning += [1]

        gpus_left = n_gpus - len(partitioning)
        layers_left = n_layers - sum(partitioning)

        # Skip this partitioning if there are not enough layers left for the remaining GPUs.
        if layers_left < gpus_left:
            # print("Skipping... ", partitioning + evenly_distribute(gpus_left, layers_left), layer) 
            continue

        partitioning += evenly_distribute(gpus_left, layers_left)
        partitionings.append(partitioning)

    # If last generated partitioning looked like: [x1, ... ,1, 1, 1], we can terminate.
    # Else (if the last one looks like [x1, ... xn, 1, 1]), mem added cannot be
    # extracted for the second to last layer, so add one last set of
    # profiling partitionings for that layer, with:
    # [x1, ... xn-1, 1, 2] and [x1, ... xn-1, 2, 1]
    if partitionings[-1][-3:] != [1, 1, 1]:
        partitioning = partitionings[-1][:]
        partitioning2 = partitionings[-1][:]        
        partitioning[-3] -= 1
        partitioning[-1] += 1
        partitioning2[-3] -= 1
        partitioning2[-2] += 1
        partitionings.append(partitioning)
        partitionings.append(partitioning2)

    return partitionings

# Converts a list of partitionings (each in format: [n_layers_gpu_0, ..., n_layers_gpu_k])
# to a list of forward layer ids as accepted by Alpa (e.g. [[0, 1], ..., [n-1, n]).
def convert_to_forward_layers(partitionings):
    new_partitionings = []

    for partitioning in partitionings:
        last = 0
        new = []
        for x in partitioning:
            new.append(list(np.arange(last, last+x)))
            last += x 
        new_partitionings.append(new)

    return new_partitionings

# Generates partitionings for a trimmed model.
# Ensures that at most 2 intact layers are on a GPU at once.
# Returns a list of partitionings, and a corresponding list of contentful stages to be profiled;
# stages not listed in profiling_stages are to be trimmed during profiling.
def get_trimmed_partitionings(n_gpus, n_layers):
    if n_layers < 6:
        raise ValueError("For profiling, n_layers must be >= 2 * n_gpus - 2")

    partitionings = []
    profiling_stages = []

    single_layers = [1 for i in range(n_gpus-1)]
    single_layers.append(n_layers-(n_gpus-1))
    partitionings.append(single_layers)
    profiling_stages.append([i for i in range(n_gpus-1)])

    even_layers = [2 for i in range(n_gpus-1)]
    even_layers.append(n_layers-sum(even_layers))
    partitionings.append(even_layers)
    profiling_stages.append([i for i in range(n_gpus-1)])

    odd_layers = [1]
    odd_layers.extend(2 for i in range(n_gpus-2))
    odd_layers.append(n_layers-sum(odd_layers))
    partitionings.append(odd_layers)
    profiling_stages.append([i for i in range(1,n_gpus-1)])

    # PARTITIONINGS FOR MEM ISOLATED
    layers_remaining = single_layers[-1]
    start = sum(single_layers[:-1])
    while layers_remaining > 0:
        end = min(start + (n_gpus-2), n_layers)
        partitioning = [start]
        partitioning.extend(1 for i in range(start,end))
        profiling_stages.append([i for i in range(1,len(partitioning))])

        layers_remaining = n_layers-sum(partitioning)
        if layers_remaining > 0:
            partitioning.append(layers_remaining)
        partitionings.append(partitioning)
        start = sum(partitioning[:-1])

    # PARTITIONINGS FOR MEM ADDED
    # Go through both even and odd layers
    for parity in [even_layers, odd_layers]:
        layers_remaining = parity[-1]
        start = sum(parity[:-1])
        while layers_remaining > 1:
            end = min(start + 2*(n_gpus-2), n_layers-1)
            partitioning = [start]
            partitioning.extend(2 for i in range(start,end,2))
            profiling_stages.append([i for i in range(1,len(partitioning))])

            layers_remaining = n_layers-sum(partitioning)
            if layers_remaining > 0:
                partitioning.append(layers_remaining)
            partitionings.append(partitioning)
            start = sum(partitioning[:-1])

    return partitionings, profiling_stages

# Generates profiling partitionings for given n_gpus and n_layers as a test.
# Then runs a validity check to see if mem isolated and mem added can indeed
# be extracted from the generated profiling partitionings.
def main(n_gpus, n_layers):
    print("Predicting for", n_gpus, "gpus and", n_layers, "layers")
    ps, stages = get_trimmed_partitionings(n_gpus, n_layers)

    for i,p in enumerate(ps):
        print(p, stages[i])
        assert sum(p) == n_layers
        assert len(p) <= n_gpus

    # Validity check:
    partitionings = convert_to_forward_layers(ps)
    for p in partitionings: print(p)

    partitionings = [partitionings_to_cutpoints(p) for p in partitionings]
    n_layers = partitionings[0][-1]

    # Put cutpoints and mocked(!) memory results in dict for validity check.
    data = []
    for i,p in enumerate(partitionings):
        contentful_stages = stages[i]
        mem = []
        for j in range(len(p)):
            if j in contentful_stages:
                mem.append(1)
            else:
                mem.append(0)
        data.append({"partitioning": p, "mem": mem})

    # Check if indeed al profiling data can be extracted from the generated partitionings.
    print("Running validity check...")
    results = get_mem_stats(data, n_layers)
    do_completeness_check(results, n_layers)

if __name__ == "__main__":
    main(n_gpus=8, n_layers=42)
