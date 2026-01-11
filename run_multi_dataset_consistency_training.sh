#!/bin/bash

# Multi-dataset EnCodec training script with consistency losses
# This script runs training with multiple datasets (jamendo, common_voice, fsd50k, dns_challenge4)
# and includes consistency losses (slice consistency and augmentation consistency)
#
# Prerequisites:
#   1. Ensure datasets are available with CSV files:
#      - jamendo: /scratch/iran/jamendo/
#      - common_voice: /scratch/iran/common_voice/
#      - fsd50k: /scratch/iran/fsd50k/
#      - dns_challenge4: /scratch/iran/dns_challenge4/

echo "Starting multi-dataset EnCodec training with consistency losses..."
echo "Target bandwidths: 1.5, 3.0, 6.0, 12.0, 24.0 kbps"
echo ""

# Check if virtual environment exists
if [ ! -d "venv_mono_encodec" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   ./setup_mono_training.sh"
    exit 1
fi

# Quick check for datasets
echo "Checking datasets..."
MISSING_DATASETS=0

# Check for CSV files
DATASETS=("jamendo" "common_voice" "fsd50k" "dns_challenge4")
BASE_PATH="/scratch/iran"

for dataset in "${DATASETS[@]}"; do
    train_csv="${BASE_PATH}/${dataset}/${dataset}_train.csv"
    if [ ! -f "$train_csv" ]; then
        echo "⚠️  Warning: ${dataset} train CSV not found at: ${train_csv}"
        MISSING_DATASETS=1
    else
        echo "✓ Found ${dataset} train CSV"
    fi
done

if [ $MISSING_DATASETS -eq 1 ]; then
    echo ""
    echo "⚠️  Some datasets are missing. Training will continue with available datasets."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv_mono_encodec/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
echo "Starting training with consistency losses..."
python train_multi_dataset_consistency.py \
    --config-name=config_multi_dataset_consistency \
    common.max_epoch=200 \
    common.val_interval=5 \
    datasets.batch_size=8 \
    datasets.fixed_length=16000 \
    model.sample_rate=24000 \
    model.channels=1 \
    model.target_bandwidths=[1.5,3.0,6.0,12.0,24.0] \
    model.slice_consistency.slice_interval_type=random \
    model.slice_consistency.split_interval_percentage=0.2 \
    model.slice_consistency.feature_types=["quant_in"] \
    model.perturb_encoder.perturb_methods=["volume_aug","inversion_aug"] \
    model.perturb_encoder.volume_aug_config.gain_range=[0.5,2.0] \
    model.perturb_encoder.volume_aug_config.apply_prob=0.5 \
    model.perturb_encoder.inversion_aug_config.apply_prob=0.5 \
    model.perturb_encoder.perturb_all_audio=true \
    model.perturb_encoder.perturb_slice_audio=true \
    wandb.enabled=true \
    wandb.project=multi-dataset-encodec-consistency \
    wandb.name=multi_dataset_consistency_bs8_epochs200_24khz

echo "Training completed!"

