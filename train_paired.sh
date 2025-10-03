#!/bin/bash

# Paired Channel EnCodec Training Script
# This script trains the model on paired channels with similarity and diversity losses

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory
mkdir -p checkpoints_paired_nq2

# Run paired training
python train_paired_gpu.py \
    --config-name=config_paired_nq2 \
    datasets.data_root="/scratch/eigenscape" \
    datasets.batch_size=8 \
    datasets.fixed_length=13800 \
    optimization.lr=3e-4 \
    optimization.disc_lr=3e-4 \
    common.max_epoch=300 \
    common.seed=3401 \
    checkpoint.save_folder="./checkpoints_paired_nq2/" \
    pairwise.sim_weight=0.5 \
    pairwise.div_weight=0.5 \
    wandb.enabled=true \
    wandb.project="paired-encodec-nq2" \
    wandb.name="paired_encodec_bs8_lr3e-4"

echo "Paired training completed!"
