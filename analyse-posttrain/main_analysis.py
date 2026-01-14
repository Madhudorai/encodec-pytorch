"""Main analysis script for codebook 0 consistency evaluation.

This script tests:
1. Slice consistency (full audio vs random slices)
2. Augmentation consistency (original vs gain-adjusted/inverted)
3. Time delay consistency (original vs time-delayed, up to 10ms)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
import torchaudio
import numpy as np
from collections import defaultdict
import soundfile as sf

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_consistency_0 import EncodecModelWithSliceConsistency
from utils import convert_audio

# Import analysis modules
from extract_codebook_indices import extract_codebook0_indices
from test_slice_consistency import test_slice_consistency
from test_augmentation_consistency import test_gain_consistency, test_inversion_consistency
from test_time_delay_consistency import test_time_delay_consistency
from visualizations import (
    plot_time_delay_heatmap,
    plot_match_rate_vs_delay,
    plot_boundary_vs_center,
    plot_gain_consistency,
    plot_index_difference_histogram
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config: Dict[str, Any]) -> EncodecModelWithSliceConsistency:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration dictionary
    
    Returns:
        model: Loaded model in eval mode
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Get model configuration
    target_bandwidths = config.get('target_bandwidths', [1.5, 3.0, 6.0, 12.0, 24.0])
    sample_rate = config.get('sample_rate', 24000)
    channels = config.get('channels', 1)
    causal = config.get('causal', True)
    norm = config.get('norm', 'weight_norm')
    audio_normalize = config.get('audio_normalize', True)
    segment = config.get('segment', None)
    name = config.get('name', 'encodec_consistency')
    ratios = config.get('ratios', [8, 5, 4, 2])
    
    # Slice consistency config (if used during training)
    slice_consistency_config = config.get('slice_consistency', None)
    perturb_encoder_config = config.get('perturb_encoder', None)
    inter_channel_consistency_config = config.get('inter_channel_consistency', None)
    
    # Create model
    model = EncodecModelWithSliceConsistency._get_model(
        target_bandwidths=target_bandwidths,
        sample_rate=sample_rate,
        channels=channels,
        causal=causal,
        model_norm=norm,
        audio_normalize=audio_normalize,
        segment=segment,
        name=name,
        ratios=ratios,
        slice_consistency=slice_consistency_config,
        perturb_encoder=perturb_encoder_config,
        inter_channel_consistency=inter_channel_consistency_config,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    
    model.eval()
    logger.info("Model loaded successfully")
    
    return model


def load_audio(audio_path: str, target_sr: int, target_channels: int) -> torch.Tensor:
    """Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        target_channels: Target number of channels
    
    Returns:
        audio: Audio tensor [C, T]
    """
    logger.info(f"Loading audio from {audio_path}")
    
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        # Common failure mode on some setups: torchaudio tries to use torchcodec/ffmpeg
        # and fails to load libtorchcodec. Fall back to soundfile, which can read wavs
        # without ffmpeg.
        logger.warning(f"torchaudio.load failed ({type(e).__name__}: {e}). Falling back to soundfile...")
        audio_np, sr = sf.read(audio_path, always_2d=True)  # [T, C]
        # Convert to torch [C, T]
        wav = torch.from_numpy(audio_np.T).float()
        # Resample if needed (prefer torchaudio.functional.resample which doesn't require ffmpeg)
        if sr != target_sr:
            try:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
                sr = target_sr
            except Exception as re:
                raise RuntimeError(
                    f"Audio loaded via soundfile at sr={sr}, but resampling to {target_sr} failed. "
                    f"Original error: {re}"
                ) from re
    
    audio = convert_audio(wav, sr, target_sr, target_channels)
    
    logger.info(f"Audio loaded: shape={audio.shape}, sample_rate={target_sr}")
    
    return audio


def run_analysis(model: EncodecModelWithSliceConsistency, audio: torch.Tensor,
                bandwidth: float, sample_rate: int, output_dir: Path,
                audio_name: str = "audio"):
    """Run all consistency analysis tests.
    
    Args:
        model: Trained model
        audio: Input audio [C, T]
        bandwidth: Target bandwidth for analysis
        sample_rate: Audio sample rate
        output_dir: Directory to save results
        audio_name: Name identifier for this audio (for logging)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info(f"Starting Codebook 0 Consistency Analysis: {audio_name}")
    logger.info("=" * 60)
    
    all_results = {}
    
    # 1. Slice Consistency Test
    logger.info("\n[1/3] Testing Slice Consistency...")
    slice_results = test_slice_consistency(
        model, audio, bandwidth,
        slice_ratios=[0.1, 0.2, 0.3, 0.5],
        num_slices_per_ratio=5
    )
    all_results['slice_consistency'] = slice_results
    
    # Compute average match rate
    avg_slice_match = np.mean([r['match_rate'] for r in slice_results['match_rates']])
    logger.info(f"  Average slice match rate: {avg_slice_match*100:.2f}%")
    
    # 2. Augmentation Consistency Test
    logger.info("\n[2/3] Testing Augmentation Consistency...")
    
    # Gain adjustment
    gain_results = test_gain_consistency(
        model, audio, bandwidth,
        gain_values=[0.5, 0.75, 1.0, 1.5, 2.0]
    )
    all_results['gain_consistency'] = gain_results
    
    avg_gain_match = np.mean([r['match_rate'] for r in gain_results['match_rates']])
    logger.info(f"  Average gain match rate: {avg_gain_match*100:.2f}%")
    
    # Inversion
    inversion_results = test_inversion_consistency(model, audio, bandwidth)
    all_results['inversion_consistency'] = inversion_results
    
    logger.info(f"  Inversion match rate: {inversion_results['match_rate']*100:.2f}%")
    
    # 3. Time Delay Consistency Test
    logger.info("\n[3/3] Testing Time Delay Consistency...")
    delay_results = test_time_delay_consistency(
        model, audio, bandwidth, sample_rate,
        delay_values_ms=[1.0, 2.0, 5.0, 7.5, 10.0]
    )
    all_results['time_delay_consistency'] = delay_results
    
    avg_delay_match = np.mean([r['match_rate'] for r in delay_results['match_rates']])
    logger.info(f"  Average delay match rate: {avg_delay_match*100:.2f}%")
    
    # Generate Visualizations
    logger.info("\nGenerating visualizations...")
    
    # Time delay heatmap
    if len(delay_results['heatmap_data']) > 0:
        plot_time_delay_heatmap(
            delay_results['heatmap_data'],
            delay_results['delay_values_ms'],
            save_path=output_dir / 'time_delay_heatmap.png'
        )
        logger.info("  ✓ Time delay heatmap saved")
    
    # Match rate vs delay
    plot_match_rate_vs_delay(
        delay_results['match_rates'],
        save_path=output_dir / 'match_rate_vs_delay.png'
    )
    logger.info("  ✓ Match rate vs delay plot saved")
    
    # Boundary vs center
    if delay_results['boundary_metrics']:
        plot_boundary_vs_center(
            delay_results['boundary_metrics'],
            save_path=output_dir / 'boundary_vs_center.png'
        )
        logger.info("  ✓ Boundary vs center plot saved")
    
    # Gain consistency
    plot_gain_consistency(
        gain_results['match_rates'],
        save_path=output_dir / 'gain_consistency.png'
    )
    logger.info("  ✓ Gain consistency plot saved")
    
    # Index difference histograms
    if delay_results['index_differences']:
        all_differences = np.concatenate([
            d['differences'] for d in delay_results['index_differences']
        ])
        plot_index_difference_histogram(
            all_differences,
            condition_name='Time Delay',
            save_path=output_dir / 'delay_difference_histogram.png'
        )
        logger.info("  ✓ Delay difference histogram saved")
    
    # Save results summary
    summary = {
        'slice_consistency': {
            'average_match_rate': float(avg_slice_match),
            'num_tests': len(slice_results['match_rates'])
        },
        'gain_consistency': {
            'average_match_rate': float(avg_gain_match),
            'num_tests': len(gain_results['match_rates'])
        },
        'inversion_consistency': {
            'match_rate': float(inversion_results['match_rate'])
        },
        'time_delay_consistency': {
            'average_match_rate': float(avg_delay_match),
            'num_tests': len(delay_results['match_rates'])
        }
    }
    
    # Save summary as text
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Codebook 0 Consistency Analysis Summary: {audio_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Slice Consistency:\n")
        f.write(f"  Average Match Rate: {avg_slice_match*100:.2f}%\n")
        f.write(f"  Number of Tests: {len(slice_results['match_rates'])}\n\n")
        f.write(f"Gain Consistency:\n")
        f.write(f"  Average Match Rate: {avg_gain_match*100:.2f}%\n")
        f.write(f"  Number of Tests: {len(gain_results['match_rates'])}\n\n")
        f.write(f"Inversion Consistency:\n")
        f.write(f"  Match Rate: {inversion_results['match_rate']*100:.2f}%\n\n")
        f.write(f"Time Delay Consistency:\n")
        f.write(f"  Average Match Rate: {avg_delay_match*100:.2f}%\n")
        f.write(f"  Number of Tests: {len(delay_results['match_rates'])}\n")
    
    logger.info(f"\nSummary saved to {summary_path}")
    logger.info("\n" + "=" * 60)
    logger.info(f"Analysis Complete for {audio_name}!")
    logger.info("=" * 60)
    
    return summary  # Return summary dict for easy comparison


def find_demo_audio_files(demo_dir: Path) -> List[Path]:
    """Find all ground truth audio files in demo directory.
    
    Args:
        demo_dir: Path to demo directory
    
    Returns:
        List of paths to *_gt.wav files
    """
    audio_files = []
    for gt_file in demo_dir.rglob("*_gt.wav"):
        audio_files.append(gt_file)
    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(description='Codebook 0 Consistency Analysis')
    parser.add_argument('--checkpoint', type=str, 
                       default=str((Path(__file__).parent.parent / 'checkpoints' / 'bs8_cut24000_length16000_epoch150_lr0.0003.pt').resolve()),
                       help='Path to model checkpoint (default: ./checkpoints/bs8_cut24000_length16000_epoch150_lr0.0003.pt)')
    parser.add_argument('--baseline_checkpoint', type=str,
                       default=str((Path(__file__).parent.parent / 'checkpoints' / 'bs16_cut24000_length32000_epoch300_lr0.0003.pt').resolve()),
                       help='Path to baseline model checkpoint for comparison (default: ./checkpoints/bs16_cut24000_length32000_epoch300_lr0.0003.pt)')
    parser.add_argument('--compare_baseline', action='store_true',
                       help='Run analysis on both trained and baseline models for comparison')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to single test audio file (ignored if --demo_dir is used)')
    parser.add_argument('--demo_dir', type=str, default=None,
                       help='Path to demo directory with ground truth files (processes all *_gt.wav files)')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Output directory for results')
    parser.add_argument('--bandwidth', type=float, default=6.0,
                       help='Target bandwidth for analysis (default: 6.0)')
    parser.add_argument('--sample_rate', type=int, default=24000,
                       help='Audio sample rate (default: 24000)')
    parser.add_argument('--channels', type=int, default=1,
                       help='Number of audio channels (default: 1)')
    
    args = parser.parse_args()
    
    # Model configuration (from config_multi_dataset_consistency.yaml)
    config = {
        'target_bandwidths': [1.5, 3.0, 6.0, 12.0, 24.0],
        'sample_rate': args.sample_rate,
        'channels': args.channels,
        'causal': True,
        'norm': 'weight_norm',
        'audio_normalize': True,
        'segment': None,
        'name': 'multi_dataset_encodec_consistency',
        'ratios': [8, 5, 4, 2],
        'slice_consistency': {
            'slice_interval_type': 'random',
            'split_interval_percentage': 0.2,
            'feature_types': ['quant_in'],
            'loss_types': ['mse_loss'],
            'augmentation_constraint_loss_weight': 1.0,
            'slice_consistency_constraint_loss_weight': 1.0,
            'target_sr': 24000,
            'ds_rate': 320,
            'mse_loss_reduction': 'mean'
        },
        'perturb_encoder': {
            'perturb_methods': ['volume_aug', 'inversion_aug'],
            'volume_aug_config': {'gain_range': [0.5, 2.0], 'apply_prob': 0.5},
            'inversion_aug_config': {'apply_prob': 0.5},
            'perturb_all_audio': True,
            'perturb_slice_audio': True
        },
        'inter_channel_consistency': None,  # Disabled for multi-dataset (mono)
    }
    
    # Baseline config (no consistency training)
    baseline_config = {
        'target_bandwidths': [1.5, 3.0, 6.0, 12.0, 24.0],
        'sample_rate': args.sample_rate,
        'channels': args.channels,
        'causal': True,
        'norm': 'weight_norm',
        'audio_normalize': True,
        'segment': None,
        'name': 'multi_dataset_encodec',
        'ratios': [8, 5, 4, 2],
        'slice_consistency': None,
        'perturb_encoder': None,
        'inter_channel_consistency': None,
    }
    
    # Determine audio files to process
    if args.demo_dir:
        demo_dir = Path(args.demo_dir)
        if not demo_dir.exists():
            logger.error(f"Demo directory not found: {demo_dir}")
            return
        audio_files = find_demo_audio_files(demo_dir)
        if not audio_files:
            logger.error(f"No *_gt.wav files found in {demo_dir}")
            return
        logger.info(f"Found {len(audio_files)} audio files in demo directory")
    elif args.audio:
        audio_files = [Path(args.audio)]
    else:
        logger.error("Either --audio or --demo_dir must be provided")
        return
    
    # Aggregate results across all audio files
    all_trained_results = []
    all_baseline_results = []
    
    # Load models once
    logger.info("\n" + "=" * 60)
    logger.info("Loading Models")
    logger.info("=" * 60)
    model = load_model(args.checkpoint, config)
    baseline_model = None
    if args.compare_baseline:
        baseline_model = load_model(args.baseline_checkpoint, baseline_config)
    
    # Process each audio file
    for audio_file in audio_files:
        audio_name = audio_file.stem.replace('_gt', '')  # Remove _gt suffix
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {audio_file.name} ({audio_name})")
        logger.info(f"{'='*60}")
        
        # Load audio
        audio = load_audio(str(audio_file), args.sample_rate, args.channels)
        
        # Run analysis on trained model
        output_dir = Path(args.output_dir) / 'trained_model' / audio_name
        results_trained = run_analysis(model, audio, args.bandwidth, args.sample_rate, output_dir, audio_name)
        all_trained_results.append((audio_name, results_trained))
        
        # Run analysis on baseline model if requested
        if args.compare_baseline and baseline_model:
            baseline_output_dir = Path(args.output_dir) / 'baseline_model' / audio_name
            results_baseline = run_analysis(baseline_model, audio, args.bandwidth, args.sample_rate, baseline_output_dir, audio_name)
            all_baseline_results.append((audio_name, results_baseline))
    
    # Generate aggregated summary
    logger.info("\n" + "=" * 60)
    logger.info("Generating Aggregated Summary")
    logger.info("=" * 60)
    
    # Aggregate trained model results
    trained_aggregated = {
        'slice_consistency': [],
        'gain_consistency': [],
        'inversion_consistency': [],
        'time_delay_consistency': []
    }
    
    for audio_name, results in all_trained_results:
        trained_aggregated['slice_consistency'].append(results['slice_consistency']['average_match_rate'])
        trained_aggregated['gain_consistency'].append(results['gain_consistency']['average_match_rate'])
        trained_aggregated['inversion_consistency'].append(results['inversion_consistency']['match_rate'])
        trained_aggregated['time_delay_consistency'].append(results['time_delay_consistency']['average_match_rate'])
    
    # Aggregate baseline results if available
    baseline_aggregated = None
    if all_baseline_results:
        baseline_aggregated = {
            'slice_consistency': [],
            'gain_consistency': [],
            'inversion_consistency': [],
            'time_delay_consistency': []
        }
        for audio_name, results in all_baseline_results:
            baseline_aggregated['slice_consistency'].append(results['slice_consistency']['average_match_rate'])
            baseline_aggregated['gain_consistency'].append(results['gain_consistency']['average_match_rate'])
            baseline_aggregated['inversion_consistency'].append(results['inversion_consistency']['match_rate'])
            baseline_aggregated['time_delay_consistency'].append(results['time_delay_consistency']['average_match_rate'])
    
    # Save aggregated summary
    summary_path = Path(args.output_dir) / 'aggregated_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Codebook 0 Consistency Analysis - Aggregated Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of audio files analyzed: {len(all_trained_results)}\n")
        f.write(f"Audio files: {[name for name, _ in all_trained_results]}\n\n")
        
        f.write("TRAINED MODEL (with consistency training):\n")
        f.write("-" * 60 + "\n")
        f.write(f"Slice Consistency:\n")
        f.write(f"  Mean: {np.mean(trained_aggregated['slice_consistency'])*100:.2f}%\n")
        f.write(f"  Std:  {np.std(trained_aggregated['slice_consistency'])*100:.2f}%\n")
        f.write(f"  Min:  {np.min(trained_aggregated['slice_consistency'])*100:.2f}%\n")
        f.write(f"  Max:  {np.max(trained_aggregated['slice_consistency'])*100:.2f}%\n\n")
        
        f.write(f"Gain Consistency:\n")
        f.write(f"  Mean: {np.mean(trained_aggregated['gain_consistency'])*100:.2f}%\n")
        f.write(f"  Std:  {np.std(trained_aggregated['gain_consistency'])*100:.2f}%\n")
        f.write(f"  Min:  {np.min(trained_aggregated['gain_consistency'])*100:.2f}%\n")
        f.write(f"  Max:  {np.max(trained_aggregated['gain_consistency'])*100:.2f}%\n\n")
        
        f.write(f"Inversion Consistency:\n")
        f.write(f"  Mean: {np.mean(trained_aggregated['inversion_consistency'])*100:.2f}%\n")
        f.write(f"  Std:  {np.std(trained_aggregated['inversion_consistency'])*100:.2f}%\n")
        f.write(f"  Min:  {np.min(trained_aggregated['inversion_consistency'])*100:.2f}%\n")
        f.write(f"  Max:  {np.max(trained_aggregated['inversion_consistency'])*100:.2f}%\n\n")
        
        f.write(f"Time Delay Consistency:\n")
        f.write(f"  Mean: {np.mean(trained_aggregated['time_delay_consistency'])*100:.2f}%\n")
        f.write(f"  Std:  {np.std(trained_aggregated['time_delay_consistency'])*100:.2f}%\n")
        f.write(f"  Min:  {np.min(trained_aggregated['time_delay_consistency'])*100:.2f}%\n")
        f.write(f"  Max:  {np.max(trained_aggregated['time_delay_consistency'])*100:.2f}%\n\n")
        
        if baseline_aggregated:
            f.write("\nBASELINE MODEL (pre-consistency training):\n")
            f.write("-" * 60 + "\n")
            f.write(f"Slice Consistency:\n")
            f.write(f"  Mean: {np.mean(baseline_aggregated['slice_consistency'])*100:.2f}%\n")
            f.write(f"  Std:  {np.std(baseline_aggregated['slice_consistency'])*100:.2f}%\n\n")
            
            f.write(f"Gain Consistency:\n")
            f.write(f"  Mean: {np.mean(baseline_aggregated['gain_consistency'])*100:.2f}%\n")
            f.write(f"  Std:  {np.std(baseline_aggregated['gain_consistency'])*100:.2f}%\n\n")
            
            f.write(f"Inversion Consistency:\n")
            f.write(f"  Mean: {np.mean(baseline_aggregated['inversion_consistency'])*100:.2f}%\n")
            f.write(f"  Std:  {np.std(baseline_aggregated['inversion_consistency'])*100:.2f}%\n\n")
            
            f.write(f"Time Delay Consistency:\n")
            f.write(f"  Mean: {np.mean(baseline_aggregated['time_delay_consistency'])*100:.2f}%\n")
            f.write(f"  Std:  {np.std(baseline_aggregated['time_delay_consistency'])*100:.2f}%\n\n")
            
            f.write("\nIMPROVEMENT (Trained - Baseline):\n")
            f.write("-" * 60 + "\n")
            slice_improvement = (np.mean(trained_aggregated['slice_consistency']) - 
                               np.mean(baseline_aggregated['slice_consistency'])) * 100
            gain_improvement = (np.mean(trained_aggregated['gain_consistency']) - 
                              np.mean(baseline_aggregated['gain_consistency'])) * 100
            inv_improvement = (np.mean(trained_aggregated['inversion_consistency']) - 
                             np.mean(baseline_aggregated['inversion_consistency'])) * 100
            delay_improvement = (np.mean(trained_aggregated['time_delay_consistency']) - 
                               np.mean(baseline_aggregated['time_delay_consistency'])) * 100
            
            f.write(f"Slice Consistency:     {slice_improvement:+.2f}%\n")
            f.write(f"Gain Consistency:     {gain_improvement:+.2f}%\n")
            f.write(f"Inversion Consistency: {inv_improvement:+.2f}%\n")
            f.write(f"Time Delay Consistency: {delay_improvement:+.2f}%\n")
    
    logger.info(f"Aggregated summary saved to: {summary_path}")
    logger.info(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
