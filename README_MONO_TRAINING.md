# Mono Audio EnCodec Training with n_q=2

This setup allows you to train a mono audio EnCodec model with 2 quantizers (n_q=2) using your 32-channel audio dataset.

## Setup

### 1. Dataset Structure
Your dataset should be organized as follows:
```
/scratch/eigenscape/
├── Beach/
├── Busy Street/
├── Park/
├── Pedestrian Zone/
├── Quiet Street/
├── Shopping Centre/
├── Woodland/
└── Train Station/
```

- **Training folders**: Beach, Busy Street, Park, Pedestrian Zone, Quiet Street, Shopping Centre
- **Validation folders**: Woodland, Train Station
- Each folder should contain 32-channel audio files (WAV, FLAC, MP3, M4A supported)

### 2. Installation
```bash
# Install required dependencies
pip install soundfile

# Make setup script executable
chmod +x setup_mono_training.sh

# Run setup
./setup_mono_training.sh
```

### 3. Configuration
The training uses `config/config_mono_nq2.yaml` with the following key settings:
- **Channels**: 1 (mono)
- **Sample rate**: 24,000 Hz
- **n_q**: 2 (number of quantizers)
- **Batch size**: 8 (optimized for single GPU)
- **Tensor cut**: 24,000 samples (1 second)
- **Data root**: `/scratch/eigenscape`

## Training

### Start Training
```bash
python train_single_gpu.py
```

### What happens during training:
1. **Data Loading**: Randomly selects one channel (1-32) from each 32-channel audio file
2. **Segment Extraction**: Extracts 1-second segments from the selected channel
3. **Model Training**: Trains EnCodec with 2 quantizers for mono audio compression
4. **Checkpoint Saving**: Saves model checkpoints every 2 epochs to `./checkpoints_mono_nq2/`

### Monitor Training
- **TensorBoard**: Logs are saved to `./checkpoints_mono_nq2/runs/`
- **Console Output**: Training progress and losses are logged to console and file
- **Audio Samples**: Ground truth and reconstructed audio samples are saved during testing

## Files Created

### New Files:
- `multi_channel_dataset.py`: Custom dataset for 32-channel audio with mono extraction
- `train_single_gpu.py`: Single GPU training script
- `test_setup.py`: Setup verification script
- `config/config_mono_nq2.yaml`: Configuration for mono training with n_q=2
- `setup_mono_training.sh`: Setup script

### Modified Files:
- `model.py`: Added n_q parameter support to model creation

## Key Features

### Dataset (`multi_channel_dataset.py`):
- Loads 32-channel audio files using `soundfile`
- Randomly selects one channel (1-32) per sample
- Extracts 1-second segments
- Handles various audio formats (WAV, FLAC, MP3, M4A)
- Supports both training and validation splits

### Model Configuration:
- **Mono audio**: 1 channel input/output
- **2 quantizers**: Reduced from default (usually 4-8)
- **24kHz sampling**: Standard for speech/audio compression
- **1-second segments**: Fixed length for consistent training

### Training Features:
- **Single GPU**: Optimized for single GPU training
- **Mixed Precision**: Optional AMP support
- **Loss Balancing**: Configurable loss weights
- **Discriminator Training**: Adversarial training with discriminator
- **Checkpointing**: Automatic model saving and resuming

## Usage Examples

### Test Setup
```bash
python test_setup.py
```

### Start Training
```bash
python train_single_gpu.py
```

### Resume Training
Edit `config/config_mono_nq2.yaml`:
```yaml
checkpoint:
  resume: True
  checkpoint_path: './checkpoints_mono_nq2/epoch10_lr3e-4.pt'
  disc_checkpoint_path: './checkpoints_mono_nq2/epoch10_disc_lr3e-4.pt'
```

### Monitor with TensorBoard
```bash
tensorboard --logdir=./checkpoints_mono_nq2/runs/
```

## Troubleshooting

### Common Issues:
1. **Data path not found**: Update `data_root` in config file
2. **CUDA out of memory**: Reduce `batch_size` in config
3. **No audio files found**: Check folder structure and file extensions
4. **Import errors**: Install missing dependencies with `pip install soundfile`

### Performance Tips:
- Use `num_workers=4` for faster data loading
- Enable `pin_memory=True` for GPU acceleration
- Monitor GPU memory usage and adjust batch size accordingly
- Use mixed precision (`amp: True`) for faster training (if supported)

## Expected Results

After training, you should have:
- A mono audio EnCodec model with 2 quantizers
- Checkpoints saved every 2 epochs
- TensorBoard logs for monitoring
- Audio samples for quality assessment
- Model ready for compression/decompression tasks

The model will be optimized for mono audio compression at 24kHz with 2 quantizers, providing a good balance between compression ratio and audio quality.
