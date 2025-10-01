#!/bin/bash

# Training script that activates virtual environment and runs training

echo "Starting mono audio EnCodec training..."

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

# Run training
echo "Starting training..."
python train_single_gpu.py

echo "Training completed!"
