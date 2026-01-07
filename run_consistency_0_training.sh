#!/bin/bash

# Eigenscape EnCodec training script with codebook 0 consistency
# This script runs training with Eigenscape multi-channel dataset
# and includes consistency losses for codebook 0 (augmentation and slice consistency)

echo "Starting Eigenscape EnCodec training with codebook 0 consistency..."
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

# Run training
echo "Starting training with codebook 0 consistency..."
python train_consistency_0.py \
    --config-name=config_consistency_0 \
    common.max_epoch=200 \
    common.val_interval=5 \
    datasets.batch_size=4 \
    datasets.fixed_length=16000 \
    model.sample_rate=24000 \
    model.channels=2 \
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
    wandb.project=eigenscape-encodec-consistency-0 \
    wandb.name=eigenscape_consistency_0_bs4_epochs200_24khz

echo "Training completed!"
