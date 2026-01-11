#!/bin/bash

# Eigenscape EnCodec training script with spatial consistency
# This script runs training with Eigenscape multi-channel dataset
# and includes spatial consistency losses (inter-channel, augmentation, and slice consistency)
#
# Prerequisites:
#   1. Run setup_spatial_training.sh to download and prepare datasets
#   2. Ensure datasets are available:
#      - Eigenscape: /scratch/eigenscape
#      - Spatial datasets: /scratch/spatial_audio_data (with CSV files in csvs/ folder)

echo "Starting Eigenscape EnCodec training with spatial consistency..."
echo "Target bandwidths: 1.5, 3.0, 6.0, 12.0, 24.0 kbps"
echo ""

# Check if virtual environment exists
if [ ! -d "venv_mono_encodec" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   ./setup_spatial_training.sh"
    exit 1
fi

# Quick check for datasets
SPATIAL_DATA_PATH="${SPATIAL_DATA_BASE_PATH:-/scratch/spatial_audio_data}"
EIGENSCAPE_PATH="/scratch/eigenscape"

echo "Checking datasets..."
MISSING_DATASETS=0

# Check spatial datasets CSV files
if [ ! -d "${SPATIAL_DATA_PATH}/csvs" ]; then
    echo "⚠️  Warning: Spatial dataset CSV files not found at: ${SPATIAL_DATA_PATH}/csvs"
    echo "   Run: python datasets/generate_spatial_dataset_csvs.py"
    MISSING_DATASETS=1
else
    CSV_COUNT=$(ls -1 "${SPATIAL_DATA_PATH}/csvs"/*.csv 2>/dev/null | wc -l)
    if [ "$CSV_COUNT" -eq 0 ]; then
        echo "⚠️  Warning: No CSV files found in ${SPATIAL_DATA_PATH}/csvs"
        echo "   Run: python datasets/generate_spatial_dataset_csvs.py"
        MISSING_DATASETS=1
    else
        echo "✓ Found ${CSV_COUNT} CSV files for spatial datasets"
    fi
fi

# Check Eigenscape (optional but recommended)
if [ ! -d "${EIGENSCAPE_PATH}" ]; then
    echo "⚠️  Warning: Eigenscape dataset not found at: ${EIGENSCAPE_PATH}"
    echo "   Training will use spatial datasets only"
else
    echo "✓ Eigenscape dataset found"
fi

if [ $MISSING_DATASETS -eq 1 ]; then
    echo ""
    echo "❌ Missing required datasets. Please run setup first:"
    echo "   ./setup_spatial_training.sh"
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
echo "Starting training with spatial consistency..."
python train_spatial_consistency.py \
    --config-name=config_spatial_consistency \
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
    wandb.project=eigenscape-encodec-spatial-consistency \
    wandb.name=eigenscape_spatial_consistency_bs4_epochs200_24khz

echo "Training completed!"
