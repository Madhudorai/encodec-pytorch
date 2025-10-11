#!/bin/bash

# MTG-Jamendo Dataset Download Script
# Downloads the MTG-Jamendo dataset using the official Python downloader
# Dataset URL: https://zenodo.org/record/3826813

set -e  # Exit on any error

# Configuration
DATASET_DIR="/scratch/iran/jamendo"
REPO_URL="https://github.com/MTG/mtg-jamendo-dataset.git"
TEMP_DIR="/tmp/mtg-jamendo-dataset"

echo "Starting MTG-Jamendo dataset download to: $DATASET_DIR"

# Create dataset directory
mkdir -p "$DATASET_DIR"

# Check if Python 3.7+ is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.7+ is required, but found version $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"

# Check if distutils is available (removed in Python 3.12+)
if ! python3 -c "import distutils" 2>/dev/null; then
    echo "Warning: distutils not available in Python $PYTHON_VERSION"
    echo "This may cause pip installation issues. Trying alternative approach..."
    USE_ALTERNATIVE_METHOD=true
else
    USE_ALTERNATIVE_METHOD=false
fi


# Clone the repository if it doesn't exist
if [ ! -d "$TEMP_DIR" ]; then
    echo "Cloning MTG-Jamendo dataset repository..."
    git clone "$REPO_URL" "$TEMP_DIR"
else
    echo "Repository already exists, updating..."
    cd "$TEMP_DIR"
    git pull
    cd - > /dev/null
fi

# Create virtual environment
echo "Setting up virtual environment..."
cd "$TEMP_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install only the packages needed for download.py
echo "Installing minimal packages needed for download script..."
echo "Required packages: gdown, requests, tqdm"

# Upgrade pip first
pip install --upgrade pip

# Install only the essential packages using --only-binary to avoid distutils issues
echo "Installing packages with --only-binary to avoid distutils issues..."
pip install --only-binary=all gdown requests tqdm || {
    echo "Binary installation failed, trying with --no-build-isolation..."
    pip install --no-build-isolation gdown requests tqdm || {
        echo "All installation methods failed. Proceeding anyway - download may fail."
    }
}

echo "✓ Virtual environment setup complete"

# Function to download dataset with retry
download_dataset() {
    local dataset="$1"
    local data_type="$2"
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Downloading $dataset ($data_type) - attempt $attempt/$max_attempts..."
        
        if python3 scripts/download/download.py \
            --dataset "$dataset" \
            --type "$data_type" \
            --from mtg-fast \
            --unpack \
            --remove \
            "$DATASET_DIR"; then
            echo "✓ Downloaded $dataset ($data_type)"
            return 0
        else
            echo "✗ Failed to download $dataset ($data_type) (attempt $attempt/$max_attempts)"
            attempt=$((attempt + 1))
            if [ $attempt -le $max_attempts ]; then
                echo "Retrying in 10 seconds..."
                sleep 10
            fi
        fi
    done
    
    echo "✗ Failed to download $dataset ($data_type) after $max_attempts attempts"
    return 1
}

# Download the autotagging-moodtheme dataset (smaller subset used for EnCodec training)
echo "Downloading autotagging-moodtheme dataset (mood/theme subset)..."
echo "This will download ~152 GB of data (much smaller than full dataset)"
echo "This subset contains ~919 hours of audio, similar to what's used in EnCodec training"
echo "Press Ctrl+C within 10 seconds to cancel..."
sleep 10

if download_dataset "autotagging_moodtheme" "audio"; then
    echo "✓ autotagging-moodtheme audio dataset downloaded successfully!"
else
    echo "✗ Failed to download autotagging-moodtheme audio dataset"
    exit 1
fi

# Verify the download
echo "Verifying download..."
if [ -d "$DATASET_DIR/autotagging_moodtheme" ]; then
    echo "✓ MTG-Jamendo dataset downloaded and extracted successfully!"
    echo "Dataset location: $DATASET_DIR"
    echo "Audio files: $(find "$DATASET_DIR" -name "*.mp3" | wc -l)"
    echo "Total size: $(du -sh "$DATASET_DIR" | cut -f1)"
else
    echo "✗ Error: Dataset extraction failed"
    exit 1
fi

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
