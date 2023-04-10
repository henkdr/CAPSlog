#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -C A4000
#SBATCH --gres=gpu:4

. /etc/bashrc
. ~/.bashrc
conda activate varuna
module load cuda11.7/toolkit

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=3456

nodelist=`scontrol show hostnames $SLURM_NODELIST | tr ' \n' ' '`
echo nodelist: $nodelist
mkdir -p $HOME/tmp
NODEFILE=$HOME/tmp/hosts.$SLURM_JOB_ID
( for i in $nodelist; do echo $i; done) > $NODEFILE

DATA_PATH=/var/scratch/als271/tft_data/data/processed/electricity_bin
RESULTS_PATH=/home/als271/DeepLearningExamples/PyTorch/Forecasting/TFT/results



# NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 \
NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO \
python -m varuna.run_varuna --nstages 4 --chunk_size 4 --batch_size 256 \
        --gpus_per_node 4 --no_morphing --machine_list $NODEFILE \
        --manager_ip $MASTER_ADDR \
        --code_dir '/home/als271/DeepLearningExamples/PyTorch/Forecasting/TFT' train.py \
        --dataset electricity \
        --data_path $DATA_PATH \
        --batch_size 1024 \
        --sample 450000 50000 \
        --lr 1e-3 \
        --epochs 30 \
        --seed 1 \
        --results $RESULTS_PATH \
        --varuna
