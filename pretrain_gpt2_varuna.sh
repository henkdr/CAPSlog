#!/bin/bash
#SBATCH --time=00:45:00
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

DATA_PATH=/var/scratch/als271/Megatron-LM/my-gpt2_text_document
CHECKPOINT_PATH=/home/als271/varuna/megatron_pretrain

# 355m model
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16

# # 1.5bn model
# $NUM_LAYERS=48
# $HIDDEN_SIZE=1600
# $NUM_ATTENTION_HEADS=16

# # 2.5bn model
# $NUM_LAYERS=54
# $HIDDEN_SIZE=1920
# $NUM_ATTENTION_HEADS=20

# #8.3bn model
# $NUM_LAYERS=72
# $HIDDEN_SIZE=3072
# $NUM_ATTENTION_HEADS=32


# NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 \
NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO \
python -m varuna.run_varuna --nstages 4 --chunk_size 4 --batch_size 256 \
        --gpus_per_node 4 --no_morphing --machine_list $NODEFILE \
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
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
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
        --log-interval 1 \
        --exit-interval 100 \
        --save-interval 1000 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --varuna
