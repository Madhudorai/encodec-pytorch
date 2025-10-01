#!/bin/bash

# Setup script for mono audio EnCodec training with n_q=2

echo "Setting up mono audio EnCodec training..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv_mono_encodec" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_mono_encodec
    echo "✓ Virtual environment created: venv_mono_encodec"
else
    echo "✓ Virtual environment already exists: venv_mono_encodec"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_mono_encodec/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required dependencies
echo "Installing required dependencies..."
pip install -r requirements_mono.txt

# Create checkpoint directory
mkdir -p checkpoints_mono_nq2

# Setup complete
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update data path in config/config_mono_nq2.yaml"
echo "2. Login to wandb: wandb login"
echo "3. Start training: ./run_training.sh"
echo ""
echo "Training will:"
echo "- Use mono audio from 32-channel recordings"
echo "- Randomly select 1 channel per sample"
echo "- Extract 1-second segments"
echo "- Train with n_q=2 quantizers"
echo "- Save checkpoints to ./checkpoints_mono_nq2/"
echo "- Log everything to wandb for monitoring"
