# Paired Channel EnCodec Training

This document describes the new paired channel training mode for the EnCodec model, which learns to encode shared content information in the first token and channel-specific information in the second token.

## Overview

The paired training mode processes audio pairs from the same time segment but different microphone channels, with the following objectives:

1. **Similarity Loss**: First token should be similar for the same content (different channels, same file/segment)
2. **Diversity Loss**: Second token should be different for different channels (encouraging channel-specific encoding)
3. **Reconstruction Quality**: Maintain high-quality audio reconstruction for both channels

## Architecture

- **Same Model Weights**: Both channels use identical encoder/decoder weights
- **Residual Vector Quantization**: 2 codebooks of 1024 entries each
- **Multi-Scale Discriminator**: 3-scale STFT discriminator for adversarial training
- **Paired Losses**: Additional similarity and diversity losses on quantized embeddings

## Loss Functions

### Existing Losses (per channel)
- **l_t**: Time domain L1 loss (weight: 0.1)
- **l_f**: Frequency domain loss (weight: 1.0)
- **l_g**: Generator adversarial loss (weight: 3.0)
- **l_feat**: Feature matching loss (weight: 3.0)
- **loss_w**: Vector quantization commitment loss (weight: 1.0)

### New Pairwise Losses
- **l_sim**: Similarity loss between first tokens (weight: 0.5)
- **l_div**: Diversity loss between second tokens (weight: 0.5)

## Files Added/Modified

### New Files
- `paired_channel_dataset.py`: Dataset class for loading paired channel audio
- `pairwise_losses.py`: Similarity and diversity loss functions
- `train_paired_gpu.py`: Training script for paired mode
- `config/config_paired_nq2.yaml`: Configuration for paired training
- `train_paired.sh`: Shell script to run paired training

### Modified Files
- `model.py`: Added `return_embeddings` parameter to forward method

## Usage

### Training Paired Model

```bash
# Run paired training
./train_paired.sh

# Or run directly with Python
python train_paired_gpu.py --config-name=config_paired_nq2
```

### Configuration

Key parameters in `config_paired_nq2.yaml`:

```yaml
datasets:
  batch_size: 8  # Reduced for paired training
  fixed_length: 13800  # Half of original (pairs = 2x data)

pairwise:
  sim_weight: 0.5  # Weight for similarity loss
  div_weight: 0.5  # Weight for diversity loss

model:
  n_q: 2  # Two codebooks
  target_bandwidths: [1.5, 3., 6., 12., 24.]
```

### Backward Compatibility

The original training scripts remain unchanged:
- `train_single_gpu.py`: Original mono training
- `run_training.sh`: Original training script
- `config/config_mono_nq2.yaml`: Original configuration

## Data Structure

### Training Data
- **Source**: Same eigenscape dataset
- **Folders**: Beach, Busy Street, Park, Pedestrian Zone, Quiet Street, Shopping Centre
- **Sampling**: Random file, random channel pair, random 1-second segment
- **Variation**: Different every epoch

### Validation Data
- **Source**: Woodland, Train Station
- **Consistency**: Fixed pairs every epoch (reproducible)
- **Segments**: 100 fixed validation pairs

## Expected Behavior

1. **First Token (Content)**: Should be similar for same content across different channels
2. **Second Token (Channel)**: Should be different for different channels
3. **Reconstruction**: High-quality reconstruction for both channels
4. **Convergence**: Similarity loss decreases, diversity loss increases (becomes less negative)

## Monitoring

The training logs include:
- Individual reconstruction losses for both channels
- Pairwise similarity and diversity losses
- Discriminator losses
- Audio samples for both channels in WandB

## Troubleshooting

### Common Issues
1. **Memory**: Reduce batch size if OOM errors occur
2. **Convergence**: Adjust similarity/diversity loss weights
3. **Data Loading**: Ensure eigenscape dataset path is correct

### Debugging
- Check that both channels are different in validation samples
- Monitor similarity loss trend (should decrease)
- Monitor diversity loss trend (should become less negative)

## Future Extensions

Potential improvements:
1. **Adaptive Weights**: Dynamic adjustment of similarity/diversity weights
2. **Multi-Channel**: Extend to more than 2 channels
3. **Content-Aware**: Different similarity weights based on content type
4. **Channel Classification**: Add auxiliary task to classify channel types
