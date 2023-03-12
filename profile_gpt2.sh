#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH -N 2
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

DATA_PATH=/var/scratch/als271/Megatron-LM/my-gpt2_text_document
GPUS_PER_SERVER=4

NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO \
python -m varuna.run_varuna --nstages 1 --chunk_size 1 --batch_size 256 \
        --gpus_per_node $GPUS_PER_SERVER --no_morphing --machine_list $NODEFILE \
        --manager_ip $MASTER_ADDR \
        --code_dir '/var/scratch/als271/Megatron-LM/' pretrain_gpt2.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --train-iters 100 \
        --lr-decay-iters 100 \
        --data-path $DATA_PATH \
        --distributed-backend gloo \
        --vocab-file gpt2-vocab.json \
        --merge-file gpt2-merges.txt \
        --save /var/scratch/als271/text \
        --save-interval 1000 \
        --data-impl mmap \
        --split 949,50,1 \
        --lr 0.00001 \
        --min-lr 1e-5 \
        --lr-decay-style cosine \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --use-cpu-initialization \
        --warmup .05 \
        --varuna \
        --profiling