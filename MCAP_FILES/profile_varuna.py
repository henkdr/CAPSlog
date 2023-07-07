import os, sys, subprocess
from argparse import ArgumentParser, REMAINDER
from mcap_utils import get_trimmed_partitionings, get_mCAP_partitionings
from datetime import datetime

def get_varuna_partitionings(n_layers, n_gpus, trimmed):
    if trimmed:
        mcap_partitionings, profiling_stages = get_trimmed_partitionings(n_gpus, n_layers)
    else:
        mcap_partitionings = get_mCAP_partitionings(n_gpus, n_layers)
        profiling_stages = None
    varuna_partitionings = convert_to_varuna_cutpoints(mcap_partitionings)

    for partitioning in varuna_partitionings:
        assert len(partitioning) <= n_gpus, "partitioning must have max n_gpus-many stages"

    print("NUMBER OF PARTITIONINGS: ", len(varuna_partitionings), flush=True)
    
    return varuna_partitionings, profiling_stages

def convert_to_varuna_cutpoints(mcap_partitionings):
    cps = []
    for partitioning in mcap_partitionings:
        last = 0
        new = []
        for x in partitioning:
            new.append(last)
            last += x 
        cps.append(new)

    return cps

def launch_cmd(args, partitioning, contentful_stages):
    stage_to_cut = ','.join(str(p) for p in partitioning)
    if contentful_stages is not None:
        profiling_stages = ','.join(str(s) for s in contentful_stages)
    num_stages = len(partitioning)
    launch_cmd = [sys.executable]
    launch_cmd.append("-m")
    launch_cmd.append("varuna.run_varuna")
    launch_cmd.append("--stage_to_cut={}".format(str(stage_to_cut)))
    launch_cmd.append("--nstages={}".format(str(num_stages)))
    launch_cmd.append("--chunk_size={}".format(str(args.chunk_size)))
    launch_cmd.append("--batch_size={}".format(str(args.batch_size)))
    launch_cmd.append("--gpus_per_node={}".format(str(args.gpus_per_node)))
    launch_cmd.append("--job_id={}".format(str(args.job_id)))
    launch_cmd.append("--no_morphing")
    launch_cmd.append("--machine_list={}".format(str(args.machine_list)))
    launch_cmd.append("--manager_ip={}".format(str(args.manager_ip)))
    launch_cmd.append("--code_dir={}".format(str(args.code_dir)))
    launch_cmd.append(args.training_script)
    if contentful_stages is not None:
        launch_cmd.append("--profiling_stages={}".format(str(profiling_stages)))
    launch_cmd.extend(args.training_script_args)
    return launch_cmd

def main(args):
    n_layers = args.n_cutpoints + 1
    partitionings, profiling_stages = get_varuna_partitionings(n_layers, args.n_gpus, args.trimmed)

    current_env = os.environ.copy()
    processes = []
    
    start_time = datetime.now()    
    for i, partitioning in enumerate(partitionings):
        if args.trimmed:
            stages = profiling_stages[i]
        else:
            stages = None

        cmd = launch_cmd(args, partitioning, stages)
        
        process = subprocess.Popen(cmd, env=current_env, stdout=sys.stdout, stderr=sys.stdout)
        processes.append(process)
        process.wait()
        print("Process done with return code", process.returncode)
    end_time = datetime.now()
    
    execution_time = end_time - start_time
    print("END-TO-END PROFILING TIME: ", execution_time)

def parse_args():
    parser = ArgumentParser()

    parser = ArgumentParser(description="mCAP profiler for Varuna framework")
    parser.add_argument("--job_id", type=str, default=None, help= "SLURM job ID.")
    parser.add_argument("--trimmed", required=False, action="store_true", help = "Use model trimming in profiling.")
    parser.add_argument('--n_gpus', type=int, default=8, help = "number of GPUs to profile for")
    parser.add_argument('--n_cutpoints', type=int, default=24, help = "number of Varuna cutpoints in the model")
    parser.add_argument("--machine_list", type=str, help = "path to a file with reachable IPs written line-wise.")
    parser.add_argument("--manager_ip", type=str, default=None,
                            help= "IP address for long-living manager, used for varuna morphing.")
    parser.add_argument("--batch_size", default=None, type=int, help="Total effective batch size for training")
    parser.add_argument("--chunk_size", type=int, default=None, help="Micro-batch size per mini-batch")
    parser.add_argument("--gpus_per_node", type=int, default=4, help = "number of GPUs per machine")
    parser.add_argument("--code_dir", default=None, type=str,
                        help="Path on all machines of directory to run training in.")
    parser.add_argument("training_script", type=str, default=None, nargs='?', 
                            help="The full path to the training program/script "
                             "followed by all its arguments")
    parser.add_argument('training_script_args', nargs=REMAINDER)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)