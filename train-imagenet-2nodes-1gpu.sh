#!/bin/bash
#SBATCH --nodes 2             
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --time=00:15:00

. /etc/bashrc
. ~/.bashrc
conda activate varuna
module load cuda11.7/toolkit

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

export MASTER_PORT=3456
export WORLD_SIZE=2

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

nodelist=`scontrol show hostnames $SLURM_NODELIST | tr ' \n' ' '`
echo nodelist: $nodelist
mkdir -p $HOME/tmp
NODEFILE=$HOME/tmp/hosts.$SLURM_JOB_ID

# Configure specified number of CPUs per node:
( for i in $nodelist; do echo $i; done) > $NODEFILE
cat $NODEFILE


dir=/home/als271/varuna/varuna/examples/EfficientNet-PyTorch/imagenet

python -m varuna.run_varuna  \
    --machine_list $NODEFILE \
    --manager_ip $MASTER_ADDR \
    --no_morphing \
    --nstages 2 \
    --batch_size 2 \
    --chunk_size 1 \
    --gpus_per_node 1 \
    --code_dir $dir \
    main.py \
    data \
    -e -a 'efficientnet-b0' \
    --pretrained \
    --varuna \
    --lr 0.001 \
    --epochs 1