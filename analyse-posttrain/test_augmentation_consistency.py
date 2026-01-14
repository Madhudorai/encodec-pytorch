"""Test augmentation consistency: original vs augmented audio."""

import torch
import numpy as np
import typing as tp

from extract_codebook_indices import extract_codebook0_indices
from metrics import compute_exact_match_rate, compute_index_differences


def apply_gain_adjustment(audio: torch.Tensor, gain: float) -> torch.Tensor:
    """Apply gain adjustment to audio.
    
    Args:
        audio: Input audio [C, T] or [1, C, T]
        gain: Gain multiplier (e.g., 0.5, 1.5, 2.0)
    
    Returns:
        augmented_audio: Gain-adjusted audio
    """
    if audio.dim() == 3:
        return audio * gain
    else:
        return audio * gain


def apply_inversion(audio: torch.Tensor) -> torch.Tensor:
    """Apply audio inversion (multiply by -1).
    
    Args:
        audio: Input audio [C, T] or [1, C, T]
    
    Returns:
        inverted_audio: Inverted audio
    """
    return -audio


def test_gain_consistency(model, audio: torch.Tensor, bandwidth: float,
                          gain_values: tp.List[float] = [0.5, 0.75, 1.0, 1.5, 2.0]) -> tp.Dict[str, tp.Any]:
    """Test consistency under gain adjustments.
    
    Args:
        model: Trained model
        audio: Input audio [C, T] or [1, C, T]
        bandwidth: Target bandwidth
        gain_values: List of gain values to test
    
    Returns:
        results: Dictionary with all test results
    """
    results = {
        'gain_values': gain_values,
        'match_rates': [],
        'index_differences': []
    }
    
    # Extract codebook 0 indices from original audio
    original_indices = extract_codebook0_indices(model, audio, bandwidth)
    original_indices = original_indices.squeeze(0)  # [T_feat]
    
    # Test each gain value
    for gain in gain_values:
        # Apply gain adjustment
        augmented_audio = apply_gain_adjustment(audio, gain)
        
        # Extract codebook 0 indices from augmented audio
        augmented_indices = extract_codebook0_indices(model, augmented_audio, bandwidth)
        augmented_indices = augmented_indices.squeeze(0)  # [T_feat]
        
        # Ensure same length
        min_len = min(len(original_indices), len(augmented_indices))
        orig_comp = original_indices[:min_len]
        aug_comp = augmented_indices[:min_len]
        
        # Compute metrics
        match_rate = compute_exact_match_rate(orig_comp, aug_comp)
        differences = compute_index_differences(orig_comp, aug_comp)
        
        # Store results
        results['match_rates'].append({
            'gain': gain,
            'match_rate': match_rate
        })
        results['index_differences'].append({
            'gain': gain,
            'differences': differences
        })
    
    return results


def test_inversion_consistency(model, audio: torch.Tensor, bandwidth: float) -> tp.Dict[str, tp.Any]:
    """Test consistency under audio inversion.
    
    Args:
        model: Trained model
        audio: Input audio [C, T] or [1, C, T]
        bandwidth: Target bandwidth
    
    Returns:
        results: Dictionary with test results
    """
    # Extract codebook 0 indices from original audio
    original_indices = extract_codebook0_indices(model, audio, bandwidth)
    original_indices = original_indices.squeeze(0)  # [T_feat]
    
    # Apply inversion
    inverted_audio = apply_inversion(audio)
    
    # Extract codebook 0 indices from inverted audio
    inverted_indices = extract_codebook0_indices(model, inverted_audio, bandwidth)
    inverted_indices = inverted_indices.squeeze(0)  # [T_feat]
    
    # Ensure same length
    min_len = min(len(original_indices), len(inverted_indices))
    orig_comp = original_indices[:min_len]
    inv_comp = inverted_indices[:min_len]
    
    # Compute metrics
    match_rate = compute_exact_match_rate(orig_comp, inv_comp)
    differences = compute_index_differences(orig_comp, inv_comp)
    
    results = {
        'match_rate': match_rate,
        'index_differences': differences
    }
    
    return results
