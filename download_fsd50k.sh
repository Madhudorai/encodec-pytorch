#!/bin/bash

# FSD50K Dataset Download Script
# Downloads the FSD50K dataset from Zenodo
# Dataset URL: https://zenodo.org/api/records/4060432/files-archive

set -e  # Exit on any error

# Configuration
DATASET_DIR="/scratch/iran/fsd50k"
BASE_URL="https://zenodo.org/record/4060432/files"

# Create dataset directory
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Starting FSD50K dataset download to: $DATASET_DIR"

# Function to download file with retry
download_with_retry() {
    local url="$1"
    local filename="$2"
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Downloading $filename (attempt $attempt/$max_attempts)..."
        if wget -c -O "$filename" "$url"; then
            echo "✓ Downloaded $filename"
            return 0
        else
            echo "✗ Failed to download $filename (attempt $attempt/$max_attempts)"
            attempt=$((attempt + 1))
            if [ $attempt -le $max_attempts ]; then
                echo "Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
    
    echo "✗ Failed to download $filename after $max_attempts attempts"
    return 1
}

# Download development audio files (split into 6 parts)
echo "Downloading development audio files..."
DEV_FILES=(
    "FSD50K.dev_audio.z01"
    "FSD50K.dev_audio.z02" 
    "FSD50K.dev_audio.z03"
    "FSD50K.dev_audio.z04"
    "FSD50K.dev_audio.z05"
    "FSD50K.dev_audio.zip"
)

for file in "${DEV_FILES[@]}"; do
    download_with_retry "$BASE_URL/$file" "$file"
done

# Download evaluation audio files (split into 2 parts)
echo "Downloading evaluation audio files..."
EVAL_FILES=(
    "FSD50K.eval_audio.z01"
    "FSD50K.eval_audio.zip"
)

for file in "${EVAL_FILES[@]}"; do
    download_with_retry "$BASE_URL/$file" "$file"
done


# Merge and extract development audio files
echo "Merging development audio files..."
if [ -f "FSD50K.dev_audio.zip" ]; then
    # Merge split files
    zip -s 0 FSD50K.dev_audio.zip --out FSD50K.dev_audio_merged.zip
    echo "Extracting development audio files..."
    unzip -q FSD50K.dev_audio_merged.zip
    rm FSD50K.dev_audio_merged.zip
    echo "✓ Development audio files extracted"
else
    echo "Error: FSD50K.dev_audio.zip not found"
    exit 1
fi

# Merge and extract evaluation audio files
echo "Merging evaluation audio files..."
if [ -f "FSD50K.eval_audio.zip" ]; then
    # Merge split files
    zip -s 0 FSD50K.eval_audio.zip --out FSD50K.eval_audio_merged.zip
    echo "Extracting evaluation audio files..."
    unzip -q FSD50K.eval_audio_merged.zip
    rm FSD50K.eval_audio_merged.zip
    echo "✓ Evaluation audio files extracted"
else
    echo "Error: FSD50K.eval_audio.zip not found"
    exit 1
fi

# Extract other components
echo "Extracting ground truth, metadata, and documentation..."
unzip -q FSD50K.ground_truth.zip
unzip -q FSD50K.metadata.zip
unzip -q FSD50K.doc.zip

# Clean up zip files to save space
echo "Cleaning up zip files..."
rm -f FSD50K.*.zip FSD50K.*.z01 FSD50K.*.z02 FSD50K.*.z03 FSD50K.*.z04 FSD50K.*.z05

# Verify the download
echo "Verifying download..."
if [ -d "FSD50K.dev_audio" ] && [ -d "FSD50K.eval_audio" ] && [ -d "FSD50K.ground_truth" ]; then
    echo "✅ FSD50K dataset downloaded and extracted successfully!"
    echo "Dataset location: $DATASET_DIR"
    echo ""
    echo "Directory structure:"
    ls -la
    echo ""
    echo "Development audio files: $(find FSD50K.dev_audio -name "*.wav" | wc -l)"
    echo "Evaluation audio files: $(find FSD50K.eval_audio -name "*.wav" | wc -l)"
else
    echo "❌ Error: Dataset extraction failed"
    exit 1
fi
