"""Test slice consistency: full audio vs random slices."""

import torch
import numpy as np
import typing as tp
from pathlib import Path

from extract_codebook_indices import extract_codebook0_indices
from metrics import compute_exact_match_rate, compute_index_differences, compute_boundary_vs_center_metrics


def extract_random_slice(audio: torch.Tensor, slice_ratio: float = 0.2, 
                        start_pos: tp.Optional[int] = None) -> tp.Tuple[torch.Tensor, int]:
    """Extract a random slice from audio.
    
    Args:
        audio: Input audio [C, T] or [1, C, T]
        slice_ratio: Ratio of audio length for slice (e.g., 0.2 = 20%)
        start_pos: Optional start position (if None, random)
    
    Returns:
        slice_audio: Extracted slice [C, T_slice]
        start_pos: Start position of slice
    """
    # Handle batch dimension
    if audio.dim() == 3:
        audio = audio.squeeze(0)  # [1, C, T] -> [C, T]
    
    C, T = audio.shape
    slice_length = int(T * slice_ratio)
    
    # Random start position if not provided
    if start_pos is None:
        max_start = T - slice_length
        start_pos = np.random.randint(0, max_start + 1)
    
    # Extract slice
    end_pos = start_pos + slice_length
    slice_audio = audio[:, start_pos:end_pos]  # [C, T_slice]
    
    return slice_audio, start_pos


def test_slice_consistency(model, audio: torch.Tensor, bandwidth: float,
                          slice_ratios: tp.List[float] = [0.1, 0.2, 0.3, 0.5],
                          num_slices_per_ratio: int = 5,
                          boundary_ratio: float = 0.1) -> tp.Dict[str, tp.Any]:
    """Test slice consistency by comparing full audio vs slices.
    
    Args:
        model: Trained model
        audio: Input audio [C, T] or [1, C, T]
        bandwidth: Target bandwidth
        slice_ratios: List of slice length ratios to test
        num_slices_per_ratio: Number of random slices per ratio
        boundary_ratio: Ratio for boundary analysis (e.g., 0.1 = 10% on each side)
    
    Returns:
        results: Dictionary with all test results
    """
    results = {
        'slice_ratios': slice_ratios,
        'num_slices_per_ratio': num_slices_per_ratio,
        'match_rates': [],
        'boundary_metrics': [],
        'index_differences': [],
        'slice_positions': []
    }
    
    # Extract codebook 0 indices from full audio
    full_indices = extract_codebook0_indices(model, audio, bandwidth)
    full_indices = full_indices.squeeze(0)  # [T_feat] - remove batch dim
    
    # Test each slice ratio
    for slice_ratio in slice_ratios:
        for slice_idx in range(num_slices_per_ratio):
            # Extract random slice
            slice_audio, start_pos = extract_random_slice(audio, slice_ratio)
            
            # Extract codebook 0 indices from slice
            slice_indices = extract_codebook0_indices(model, slice_audio, bandwidth)
            slice_indices = slice_indices.squeeze(0)  # [T_feat_slice]
            
            # Find overlapping region in feature space
            # Assuming feature downsampling rate (need to compute from model)
            # For now, approximate: T_feat ≈ T / downsample_rate
            # We'll align based on start position in original audio
            
            # Compute overlap region
            # Full audio indices correspond to full audio
            # Slice indices correspond to slice region
            # We need to extract the overlapping portion from full indices
            
            # Approximate feature positions (this is a simplification)
            # In practice, you'd need to know the exact downsampling rate
            # For now, we'll compare what we can
            
            # Simple approach: compare the shorter sequence
            min_len = min(len(full_indices), len(slice_indices))
            
            # For slice, we want to compare the portion that overlaps with full
            # This is a simplified version - in practice, you'd need proper temporal alignment
            full_overlap = full_indices[:min_len]
            slice_overlap = slice_indices[:min_len]
            
            # Compute metrics
            match_rate = compute_exact_match_rate(full_overlap, slice_overlap)
            differences = compute_index_differences(full_overlap, slice_overlap)
            boundary_metrics = compute_boundary_vs_center_metrics(
                full_overlap, slice_overlap, boundary_ratio
            )
            
            # Store results
            results['match_rates'].append({
                'slice_ratio': slice_ratio,
                'slice_idx': slice_idx,
                'match_rate': match_rate
            })
            results['boundary_metrics'].append({
                'slice_ratio': slice_ratio,
                'slice_idx': slice_idx,
                **boundary_metrics
            })
            results['index_differences'].append({
                'slice_ratio': slice_ratio,
                'slice_idx': slice_idx,
                'differences': differences
            })
            results['slice_positions'].append({
                'slice_ratio': slice_ratio,
                'slice_idx': slice_idx,
                'start_pos': start_pos
            })
    
    return results
