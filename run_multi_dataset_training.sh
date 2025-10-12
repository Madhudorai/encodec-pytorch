#!/bin/bash

# Multi-dataset EnCodec training script
# This script runs training with multiple datasets (jamendo, common_voice, etc.)

echo "Starting multi-dataset EnCodec training..."
echo "Configuration: 64 batch size, 300 epochs, 2k updates per epoch, 24kHz mono"
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
mkdir -p ./checkpoints_multi_dataset/

# Run training
echo "Starting training..."
python train_multi_dataset.py \
    --config-name=config_multi_dataset \
    common.max_epoch=300 \
    datasets.batch_size=16 \
    datasets.fixed_length=32000 \
    model.sample_rate=24000 \
    model.channels=1 \
    model.target_bandwidths=[1.5,3.0,6.0,12.0,24.0] \
    wandb.enabled=true \
    wandb.project=multi-dataset-encodec \
    wandb.name=multi_dataset_bs64_epochs300_24khz_mono

echo "Training completed!"
