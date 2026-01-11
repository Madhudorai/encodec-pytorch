#!/bin/bash

# Setup script for spatial consistency training
# This script prepares all required datasets before training

set -e  # Exit on error

echo "=========================================="
echo "Spatial Consistency Training Setup"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv_mono_encodec" ]; then
    echo "❌ Virtual environment not found. Please create it first:"
    echo "   python -m venv venv_mono_encodec"
    echo "   source venv_mono_encodec/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_mono_encodec/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Check for required directories
SPATIAL_DATA_PATH="${SPATIAL_DATA_BASE_PATH:-/scratch/spatial_audio_data}"
EIGENSCAPE_PATH="/scratch/eigenscape"

echo "Checking dataset paths..."
echo "  Spatial data path: ${SPATIAL_DATA_PATH}"
echo "  Eigenscape path: ${EIGENSCAPE_PATH}"
echo ""

# Step 1: Prepare spatial audio datasets
echo "=========================================="
echo "Step 1: Preparing Spatial Audio Datasets"
echo "=========================================="
echo "This will download and process:"
echo "  - STARSS"
echo "  - DCASE"
echo "  - LOCATA"
echo "  - MARCO"
echo ""
echo "⚠️  This may take a long time and requires significant disk space!"
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running data preparation..."
    echo "This will:"
    echo "  1. Download datasets from Zenodo/HuggingFace"
    echo "  2. Extract WAV files"
    echo "  3. Copy WAV files to organized directories"
    echo "  4. Generate CSV files with train/val/test splits"
    echo ""
    python datasets/prepare_all_data.py --base_path "${SPATIAL_DATA_PATH}"
    
    if [ $? -eq 0 ]; then
        echo "✓ Spatial audio datasets prepared successfully"
    else
        echo "❌ Failed to prepare spatial audio datasets"
        exit 1
    fi
else
    echo "Skipping spatial dataset preparation..."
    echo "⚠️  Make sure datasets are already prepared at: ${SPATIAL_DATA_PATH}"
    echo ""
    echo "If you have WAV files already, you can generate CSVs only:"
    echo "  python datasets/prepare_all_data.py --base_path ${SPATIAL_DATA_PATH} --skip_download"
fi

echo ""

# Step 2: Generate CSV files
echo "=========================================="
echo "Step 2: Generating CSV Files"
echo "=========================================="
echo "This will create train/val/test CSV files for all spatial datasets"
echo ""

# Check if spatial data exists (new simplified structure or old structure)
if [ ! -d "${SPATIAL_DATA_PATH}/wav_files" ] && [ ! -d "${SPATIAL_DATA_PATH}/input/mic_dev" ]; then
    echo "⚠️  Warning: Spatial data directory not found"
    echo "   Expected: ${SPATIAL_DATA_PATH}/wav_files/ (new structure)"
    echo "   Or: ${SPATIAL_DATA_PATH}/input/mic_dev/ (old structure)"
    echo "   Make sure Step 1 completed successfully or datasets are already prepared"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# CSV generation is now part of prepare_all_data.py, but we can also run it separately
if [ -d "${SPATIAL_DATA_PATH}/wav_files" ] || [ -d "${SPATIAL_DATA_PATH}/input/mic_dev" ]; then
    echo "Generating CSV files from existing WAV files..."
    python datasets/generate_spatial_dataset_csvs.py --base_path "${SPATIAL_DATA_PATH}"
    
    if [ $? -eq 0 ]; then
        echo "✓ CSV files generated successfully"
        echo "   Location: ${SPATIAL_DATA_PATH}/csvs/"
    else
        echo "⚠️  CSV generation had issues, but continuing..."
    fi
else
    echo "⚠️  Skipping CSV generation (no WAV files found)"
fi

echo ""

# Step 3: Check Eigenscape dataset
echo "=========================================="
echo "Step 3: Checking Eigenscape Dataset"
echo "=========================================="
echo ""

if [ ! -d "${EIGENSCAPE_PATH}" ]; then
    echo "⚠️  Warning: Eigenscape dataset not found at: ${EIGENSCAPE_PATH}"
    echo "   The training script will use spatial datasets only if Eigenscape is missing"
    echo "   To use Eigenscape, make sure the dataset is available at the path above"
else
    echo "✓ Eigenscape dataset found"
    # Count folders
    FOLDER_COUNT=$(find "${EIGENSCAPE_PATH}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "   Found ${FOLDER_COUNT} folders"
fi

echo ""

# Summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo "✓ Virtual environment: Ready"
echo ""

# Check spatial datasets
if [ -d "${SPATIAL_DATA_PATH}/csvs" ]; then
    CSV_COUNT=$(ls -1 "${SPATIAL_DATA_PATH}/csvs"/*.csv 2>/dev/null | wc -l)
    echo "✓ Spatial datasets: ${CSV_COUNT} CSV files found"
else
    echo "⚠️  Spatial datasets: CSV files not found"
fi

# Check Eigenscape
if [ -d "${EIGENSCAPE_PATH}" ]; then
    echo "✓ Eigenscape dataset: Found"
else
    echo "⚠️  Eigenscape dataset: Not found (training will use spatial datasets only)"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run training with:"
echo "  ./run_spatial_consistency_training.sh"
echo ""
echo "Or manually:"
echo "  source venv_mono_encodec/bin/activate"
echo "  python train_spatial_consistency.py --config-name=config_spatial_consistency"
echo ""

