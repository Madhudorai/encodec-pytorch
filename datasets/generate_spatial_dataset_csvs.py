#!/usr/bin/env python3
"""
Generate CSV files for spatial audio datasets (STARSS, DCASE, LOCATA, MARCO).

This script scans the prepared spatial audio data directories and creates CSV files
listing all WAV files, split into train/val/test based on fold naming conventions.

Usage:
    python datasets/generate_spatial_dataset_csvs.py [--base_path /scratch/spatial_audio_data]
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from glob import glob
import re

# Fold naming conventions for each dataset
# Format: {dataset_name: {split: [fold_prefixes]}}
FOLD_MAPPING = {
    'starss': {
        'train': [],  # All files not in test are train
        'test': ['fold15'],
        'val': []  # No explicit val split
    },
    'dcase': {
        'train': [],  # Need to check actual fold naming
        'test': [],
        'val': []
    },
    'aug_locata': {
        'train': [],  # Files not in test/val are train
        'test': ['fold11', 'fold13'],  # Augmented folds
        'val': []
    },
    'aug_marco': {
        'train': ['fold9'],
        'test': ['fold10'],
        'val': ['fold17']
    }
}


def get_fold_from_filename(filename):
    """Extract fold prefix from filename (e.g., 'fold7_file.wav' -> 'fold7')."""
    match = re.match(r'^(fold\d+)_', filename)
    if match:
        return match.group(1)
    return None


def generate_csv_for_dataset(dataset_name, mic_dir, output_dir, fold_mapping=None):
    """Generate train/val/test CSV files for a spatial audio dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'starss', 'aug_marco')
        mic_dir: Directory containing WAV files
        output_dir: Directory to save CSV files
        fold_mapping: Dictionary mapping splits to fold prefixes
    """
    if not os.path.exists(mic_dir):
        print(f"Warning: Directory {mic_dir} does not exist. Skipping {dataset_name}.")
        return
    
    # Get all WAV files
    wav_files = glob(os.path.join(mic_dir, '*.wav'))
    if len(wav_files) == 0:
        print(f"Warning: No WAV files found in {mic_dir}. Skipping {dataset_name}.")
        return
    
    print(f"\nProcessing {dataset_name}:")
    print(f"  Found {len(wav_files)} WAV files in {mic_dir}")
    
    # Use provided fold mapping or default
    if fold_mapping is None:
        fold_mapping = FOLD_MAPPING.get(dataset_name, {
            'train': [],
            'test': [],
            'val': []
        })
    
    # Organize files by split
    splits = {'train': [], 'val': [], 'test': []}
    
    for wav_file in wav_files:
        filename = os.path.basename(wav_file)
        fold = get_fold_from_filename(filename)
        
        # Determine which split this file belongs to
        assigned = False
        for split_name, fold_prefixes in fold_mapping.items():
            if fold and fold in fold_prefixes:
                splits[split_name].append(wav_file)
                assigned = True
                break
        
        # If not assigned and we have train/test/val mappings, assign to train by default
        if not assigned:
            # For datasets with explicit fold mappings, unassigned files go to train
            if any(fold_mapping.values()):
                splits['train'].append(wav_file)
            else:
                # For datasets without fold mappings, try to infer from filename
                # or assign all to train
                splits['train'].append(wav_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files for each split
    for split_name, files in splits.items():
        if len(files) > 0:
            csv_path = os.path.join(output_dir, f'{dataset_name}_{split_name}.csv')
            df = pd.DataFrame(files)
            df.to_csv(csv_path, index=False, header=False)
            print(f"  {split_name}: {len(files)} files -> {csv_path}")
        else:
            print(f"  {split_name}: 0 files (skipped)")
    
    # Also create a combined CSV for convenience
    all_files = [f for files in splits.values() for f in files]
    if len(all_files) > 0:
        csv_path = os.path.join(output_dir, f'{dataset_name}_all.csv')
        df = pd.DataFrame(all_files)
        df.to_csv(csv_path, index=False, header=False)
        print(f"  all: {len(all_files)} files -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate CSV files for spatial audio datasets'
    )
    parser.add_argument(
        '--base_path',
        type=str,
        default=os.environ.get('SPATIAL_DATA_BASE_PATH', '/scratch/spatial_audio_data'),
        help='Base path for spatial audio data (default: /scratch/spatial_audio_data or SPATIAL_DATA_BASE_PATH env var)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for CSV files (default: {base_path}/csvs)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['starss', 'dcase', 'aug_locata', 'aug_marco'],
        help='List of datasets to process (default: all)'
    )
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    # Check both possible locations (new simplified structure and old structure)
    wav_files_dir = base_path / 'wav_files'
    old_mic_dir = base_path / 'input' / 'mic_dev'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_path / 'csvs'
    
    print(f"Base path: {base_path}")
    print(f"Output directory: {output_dir}")
    print(f"Processing datasets: {args.datasets}")
    
    # Process each dataset
    for dataset_name in args.datasets:
        # Try new simplified structure first, fallback to old structure
        mic_dir = wav_files_dir / dataset_name
        if not mic_dir.exists():
            mic_dir = old_mic_dir / dataset_name
        if not mic_dir.exists():
            # Also try without 'aug_' prefix for some datasets
            if dataset_name.startswith('aug_'):
                alt_name = dataset_name.replace('aug_', '')
                mic_dir = wav_files_dir / alt_name
                if not mic_dir.exists():
                    mic_dir = old_mic_dir / alt_name
        
        print(f"\nLooking for {dataset_name} in: {mic_dir}")
        generate_csv_for_dataset(
            dataset_name,
            str(mic_dir),
            str(output_dir),
            FOLD_MAPPING.get(dataset_name)
        )
    
    print(f"\n✓ CSV generation complete!")
    print(f"CSV files saved to: {output_dir}")


if __name__ == '__main__':
    main()

