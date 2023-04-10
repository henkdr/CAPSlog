import re

def filter_partitioning_line(line):
    if not (line.startswith("Stage to cut is")):
        line = line.split("Stage to cut is: [")[1]
    else:
        line = line.replace("Stage to cut is: [", "")
    line = line.split("]")[0]
    line = [int(s) for s in line.split(', ')]
    line.append(24) # NOTE: HARDCODED NR OF CUTPOINTS !!!
    return line

def filter_mem_line(line):
    line = [int(x) for x in re.findall(r'\d+', line)]
    return line

def sort_mems(extracted_mems):
    extracted_mems.sort(key = lambda x: x[0])
    mems = [mem[2] for mem in extracted_mems] # TODO: check that we want allocated mem, not reserved
    return mems

# Read input file containing the output of multiple profiling runs performed in Varuna.
# In args:
#   slurm_filename: file to output of the profiling run, e.g. ssh_out_<jobid>
# Out args: 
#   partitionings: list of partitionings that were found in the input file,
#   each partitioning is a list of the first cutpoint included in each stage, 
#   appended with the total number of cutpoints at the end.
#   [ first cutpoint on stage 0, ..., first cutpoint on stage n, total nr cutpoints ]
#   Example: [0, 2, 3, 7, 11, 15, 18, 21, 24]
#
#   mem: list of peak memory usage (in bytes) of each GPU during the
#   runs described by 'partitionings'. Each element looks like:
#   [peak_mem_gpu_0, ..., peak_mem_gpu_k]
def read_input_varuna(slurm_filename):
    partitionings = []
    mem = []

    gpu_mem = []

    with open(slurm_filename, 'r') as f:
        lines = f.readlines()

    skip = False
    grab_stat = False # ignore iteration 1

    for l in lines:
        if "Stage to cut is: " in l:
            if skip:
                skip = False
            else:
                partitionings.append(filter_partitioning_line(l))
        elif l.startswith(" iteration"):
            if grab_stat:
                # reached end of memory-collection of iteration 2
                mem.append(sort_mems(gpu_mem))
                gpu_mem = []
            grab_stat = not grab_stat
        
        elif l.startswith("Memory allocated on rank"):
            if grab_stat:
                filtered_line = filter_mem_line(l)
                if len(filtered_line) != 4: # in case multiple gpus printed on same line
                    for i in range(0, len(filtered_line), 4):
                        gpu_mem.append(filtered_line[i:i+4])
                else: 
                    gpu_mem.append(filtered_line)

        # TODO: figure out
        # elif "ran out of memory" in l:
        #     # Skip next  line that describes partitionings, because this run went OOM.
        #     skip = True

    return partitionings, mem

if __name__ == "__main__":
    partitioning, mem = read_input_varuna("ssh_out_49086")
    partition_lengths = []
    mem_lengths = []
    for p in partitioning:
        partition_lengths.append(len(p))
    for m in mem:
        mem_lengths.append(len(m))
    print(partition_lengths)
    print(mem_lengths)
    print(len(partition_lengths))