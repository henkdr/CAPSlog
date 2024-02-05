import re
from collections import defaultdict

# Filters the logged stages line starting with a given preamble
# In args:
#   line: the line from the profiling logs listing to be filtered.
#       For example: "Stage to cut is: [0, 13, 14, 15, 16, 17, 18, 19]"
#
#   preamble: the preamble of the line to be filtered away.
#       For example: "Stage to cut is: "
# Out args:
#   line: an integer list of the stages listed in the given line.
#       For example: [0, 13, 14, 15, 16, 17, 18, 19]
def filter_stages_line(line, preamble):
    if not (line.startswith(preamble)):
        line = line.split(preamble)[1]
    else:
        line = line.replace(preamble, "")
    line = line.split("]")[0]
    line = [int(s) for s in line.split(', ')]
    return line

# Filters the logged memory lines from profiling output.
# In args:
#   line: the loggedd memory line as a String, which has the format
#       "Memory allocated on rank W after X iterations | peak allocated: Y | peak reserved: Z"
# Out args:
#   line: list of integers of the format [STAGE, ITERATION, PEAK ALLOCATED MEM, PEAK RESERVED MEM]
def filter_mem_line(line):
    # Memory line includes RANK#, ITERATION#, PEAK_ALLOC#, PEAK_RES#
    line = [int(x) for x in re.findall(r'\d+', line)]
    return line

# Sort the peak memory statistics per stage
# In args:
#   extracted_mems: list of contents of the logged memory lines from profiling output.
#       Each entry in the list is a 4-item list with the following contents:
#       [STAGE, ITERATION, PEAK ALLOCATED MEM, PEAK RESERVED MEM]
# Out args:
#   mems: A list of the peak allocated memory for each stage in the profiling run,
#       arranged in ascending order of stage ID.
def sort_mems(extracted_mems):
    max_mems = defaultdict(int)

    for line in extracted_mems:
        rank = line[0]
        mem = line[2]
        max_mems[rank] = max(max_mems[rank], mem)

    mems = [max_mems[i] for i in range(len(max_mems))]
    return mems

# Voids the memory statistics for trimmed stages
# In args:
#   contentful_stages: the list of non-trimmed stages for each profiling run
#
#   mem: the list of per-stage memory statistics for each profiling run
# Out args:
#   mem: the mem input argument where the trimmed stages have the memory statistic set to -1
def clear_trimmed_stages(contentful_stages, mem):
    for i, stages in enumerate(contentful_stages):
        maxi = len(mem[i])
        for s in range(maxi):
            if not s in stages:
                # stage was not contentful, set mem to -1
                mem[i][s] = -1
    return mem

# Read input file containing the output of multiple profiling runs performed in Varuna.
# In args:
#   slurm_filename: file to output of the profiling run, e.g. ssh_out_<jobid>
# Out args:
#   partitionings: list of partitionings that were found in the input file,
#       each partitioning is a list of the first cutpoint included in each stage,
#       appended with the total number of cutpoints at the end.
#       [ first cutpoint on stage 0, ..., first cutpoint on stage n, total nr cutpoints ]
#       Example: [0, 2, 3, 7, 11, 15, 18, 21, 24]
#
#   mem: list of peak memory usage (in bytes) of each GPU during the
#       runs described by 'partitionings'. Each element looks like:
#       [peak_mem_gpu_0, ..., peak_mem_gpu_k]
def read_input_varuna(slurm_filename):
    num_cutpoints = 0
    partitionings = []
    contentful_stages = []
    mem = []

    failed_partitionings = []
    gpu_mem = [] # list of list of rank/iter/alloc/res

    with open(slurm_filename, 'r') as f:
        lines = f.readlines()

    skip = False

    for l in lines:
        if "Num cutpoints is" in l:
            l = l.split("Num cutpoints is")[1]
            num_cutpoints = int(re.findall(r'\d+', l)[0])
        elif "Stage to cut is: " in l:
            if skip:
                skip = False
            partitioning = filter_stages_line(l, "Stage to cut is: [")
            partitioning.append(num_cutpoints+1)
            partitionings.append(partitioning)

        elif "Profiling stages" in l:
            contentful_stages.append(filter_stages_line(l, "PROFILING MODE; Profiling stages: ["))
        if skip:
            continue
        elif l.startswith("Process done with return code 0"):
            # reached end of profiling run
            mem.append(sort_mems(gpu_mem))
            gpu_mem = []
            skip = True

        elif l.startswith("Memory allocated on rank"):
            if "Epoch:" in l:
                l = l.split("Epoch:")[0]

            filtered_line = filter_mem_line(l)
            if len(filtered_line) != 4: # in case multiple gpus printed on same line
                for i in range(0, len(filtered_line), 4):
                    gpu_mem.append(filtered_line[i:i+4])
            else:
                gpu_mem.append(filtered_line)

        elif l.startswith("Process done with return code"):
            # Skip this partitioning, because this run went OOM or another error occurred.
            skip = True
            if len(contentful_stages) > 0:
                contentful_stages.pop()
            failed_partitionings.append(partitionings.pop())
            gpu_mem = []

    if len(failed_partitionings) > 0:
        print("FAILED PARTITIONINGS: ", failed_partitionings)

    # If profiling was done using trimming, void the trimmed memory statistics
    if len(contentful_stages) > 0:
        mem = clear_trimmed_stages(contentful_stages, mem)

    return partitionings, mem
