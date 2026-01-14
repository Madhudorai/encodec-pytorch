# Codebook 0 Consistency Analysis

This directory contains scripts for post-training analysis of codebook 0 consistency.

## Overview

The analysis tests three types of consistency:

1. **Slice Consistency**: Full audio vs random slices
2. **Augmentation Consistency**: Original vs gain-adjusted/inverted audio
3. **Time Delay Consistency**: Original vs time-delayed audio (up to 10ms)

## Structure

- `extract_codebook_indices.py`: Helper to extract codebook 0 indices from model
- `metrics.py`: Metric computation functions (match rate, differences, boundary analysis)
- `test_slice_consistency.py`: Slice consistency tests
- `test_augmentation_consistency.py`: Augmentation consistency tests
- `test_time_delay_consistency.py`: Time delay consistency tests
- `visualizations.py`: Visualization functions (heatmaps, plots)
- `main_analysis.py`: Main script that runs all tests

## Usage

### Basic Usage (Single Audio File)

```bash
python main_analysis.py \
    --audio /path/to/test_audio.wav \
    --output_dir ./analysis_results
```

### Process All Demo Folder Audio Files

```bash
python main_analysis.py \
    --demo_dir ./demo \
    --output_dir ./analysis_results
```

This will automatically find all `*_gt.wav` files in the demo directory and process each one. Results are saved per audio file, plus an aggregated summary.

### With Baseline Comparison

```bash
python main_analysis.py \
    --demo_dir ./demo \
    --output_dir ./analysis_results \
    --compare_baseline
```

The script uses default checkpoint paths:
- **Trained model**: `/user/i/iran/encodec-pytorch/outputs/2026-01-11/07-57-22/checkpoints_multi_dataset_consistency/bs8_cut24000_length16000_epoch150_lr0.0003.pt`
- **Baseline model**: `/user/i/iran/encodec-pytorch/outputs/2025-10-21/23-31-14/checkpoints_multi_dataset/bs16_cut24000_length32000_epoch300_lr0.0003.pt`

### With Baseline Comparison

```bash
python main_analysis.py \
    --audio /path/to/test_audio.wav \
    --output_dir ./analysis_results \
    --compare_baseline
```

This will analyze both:
- **Trained model** (with consistency training): epoch 150
- **Baseline model** (pre-consistency): epoch 300 from config

### Custom Checkpoints

```bash
python main_analysis.py \
    --checkpoint /path/to/trained_checkpoint.pt \
    --baseline_checkpoint /path/to/baseline_checkpoint.pt \
    --audio /path/to/test_audio.wav \
    --output_dir ./analysis_results \
    --compare_baseline
```

## Arguments

- `--checkpoint`: Path to trained model checkpoint (default: epoch 150 consistency model)
- `--baseline_checkpoint`: Path to baseline model for comparison (default: epoch 300 pre-consistency model)
- `--compare_baseline`: Run analysis on both models and generate comparison report
- `--audio`: Path to single test audio file (use this OR `--demo_dir`, not both)
- `--demo_dir`: Path to demo directory with ground truth files (processes all `*_gt.wav` files recursively)
- `--output_dir`: Directory to save analysis results (default: `./analysis_results`)
- `--bandwidth`: Target bandwidth for analysis (default: 6.0)
- `--sample_rate`: Audio sample rate (default: 24000)
- `--channels`: Number of audio channels (default: 1)

## Output

### Single Audio File

Results are saved in `{output_dir}/trained_model/` (and `{output_dir}/baseline_model/` if comparing):

1. **Visualizations**:
   - `time_delay_heatmap.png`: Heatmap showing index differences vs delay and position
   - `match_rate_vs_delay.png`: Match rate curve vs delay amount
   - `boundary_vs_center.png`: Boundary vs center comparison
   - `gain_consistency.png`: Match rate vs gain multiplier
   - `delay_difference_histogram.png`: Distribution of index differences

2. **Summary**:
   - `summary.txt`: Text summary of all metrics

### Demo Folder (Multiple Audio Files)

When using `--demo_dir`, the script processes each audio file and saves results in:
- `{output_dir}/trained_model/{audio_name}/` - Individual results per audio file
- `{output_dir}/baseline_model/{audio_name}/` - Baseline results (if comparing)

Plus an **aggregated summary** at the root:
- `{output_dir}/aggregated_summary.txt`: Statistics across all audio files (mean, std, min, max) and improvement metrics if comparing

3. **Comparison** (if `--compare_baseline` is used):
   - Aggregated comparison metrics showing improvement percentages across all audio files

## Metrics

- **Exact Match Rate**: Percentage of positions where codebook indices match exactly
- **Index Differences**: Absolute differences when indices don't match
- **Boundary vs Center**: Separate metrics for boundary regions (first/last 10%) vs center (middle 80%)

## Customization

You can modify the test parameters in `main_analysis.py`:

- Slice ratios: `[0.1, 0.2, 0.3, 0.5]`
- Gain values: `[0.5, 0.75, 1.0, 1.5, 2.0]`
- Delay values: `[1.0, 2.0, 5.0, 7.5, 10.0]` ms
- Number of slices per ratio: `5`
