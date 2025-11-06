#!/bin/bash

# Multi-dataset EnCodec training script with slice consistency
# This script runs training with multiple datasets (jamendo, common_voice, etc.)
# and includes slice consistency loss

echo "Starting multi-dataset EnCodec training with slice consistency..."
echo "Target bandwidths: 1.5, 3.0, 6.0, 12.0, 24.0 kbps"

# Check if virtual environment exists
if [ ! -d "venv_mono_encodec" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "./setup_mono_training.sh"
    exit 1
fi

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

# Create output directory if it doesn't exist
mkdir -p ./checkpoints_multi_dataset_slice_consistency/

# Run training
echo "Starting training with slice consistency..."
python train_slice_consistency.py \
    --config-name=config_slice_consistency \
    common.max_epoch=400 \
    datasets.batch_size=16 \
    datasets.fixed_length=32000 \
    model.sample_rate=24000 \
    model.channels=1 \
    model.target_bandwidths=[1.5,3.0,6.0,12.0,24.0] \
    model.slice_consistency.slice_interval_type=random \
    model.slice_consistency.split_interval_percentage=0.2 \
    model.slice_consistency.feature_types=["quant_in"] \
    model.slice_consistency.loss_weights=[20.0] \
    model.perturb_encoder.perturb_methods=["volume_aug","inversion_aug"] \
    model.perturb_encoder.perturb_all_audio=true \
    model.perturb_encoder.perturb_slice_audio=true \
    wandb.enabled=true \
    wandb.project=multi-dataset-encodec-slice-consistency \
    wandb.name=multi_dataset_slice_consistency_bs16_epochs400_24khz_mono

echo "Training completed!"
