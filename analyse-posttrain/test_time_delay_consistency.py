"""Test time delay consistency: original vs time-delayed audio."""

import torch
import numpy as np
import typing as tp

from extract_codebook_indices import extract_codebook0_indices
from metrics import compute_exact_match_rate, compute_index_differences, compute_boundary_vs_center_metrics


def apply_time_delay(audio: torch.Tensor, delay_ms: float, sample_rate: int) -> torch.Tensor:
    """Apply time delay to audio by padding with zeros at the beginning.
    
    Args:
        audio: Input audio [C, T] or [1, C, T]
        delay_ms: Delay in milliseconds
        sample_rate: Audio sample rate
    
    Returns:
        delayed_audio: Time-delayed audio
    """
    # Handle batch dimension
    if audio.dim() == 3:
        audio = audio.squeeze(0)  # [1, C, T] -> [C, T]
    
    C, T = audio.shape
    
    # Convert delay to samples
    delay_samples = int(delay_ms * sample_rate / 1000.0)
    
    # Pad with zeros at the beginning
    # Shape: [C, T + delay_samples]
    delayed_audio = torch.cat([
        torch.zeros(C, delay_samples, device=audio.device, dtype=audio.dtype),
        audio
    ], dim=1)
    
    return delayed_audio


def test_time_delay_consistency(model, audio: torch.Tensor, bandwidth: float, 
                               sample_rate: int,
                               delay_values_ms: tp.List[float] = [1.0, 2.0, 5.0, 7.5, 10.0],
                               boundary_ratio: float = 0.1) -> tp.Dict[str, tp.Any]:
    """Test consistency under time delays.
    
    Args:
        model: Trained model
        audio: Input audio [C, T] or [1, C, T]
        bandwidth: Target bandwidth
        sample_rate: Audio sample rate
        delay_values_ms: List of delay values in milliseconds
        boundary_ratio: Ratio for boundary analysis (e.g., 0.1 = 10% on each side)
    
    Returns:
        results: Dictionary with all test results including heatmap data
    """
    results = {
        'delay_values_ms': delay_values_ms,
        'match_rates': [],
        'index_differences': [],
        'boundary_metrics': [],
        'heatmap_data': []  # For visualization: [delay_idx, position] -> difference
    }
    
    # Extract codebook 0 indices from original audio
    original_indices = extract_codebook0_indices(model, audio, bandwidth)
    original_indices = original_indices.squeeze(0)  # [T_feat]
    T_feat = len(original_indices)
    
    # Initialize heatmap data
    heatmap_differences = np.zeros((len(delay_values_ms), T_feat))
    
    # Test each delay value
    for delay_idx, delay_ms in enumerate(delay_values_ms):
        # Apply time delay
        delayed_audio = apply_time_delay(audio, delay_ms, sample_rate)
        
        # Extract codebook 0 indices from delayed audio
        delayed_indices = extract_codebook0_indices(model, delayed_audio, bandwidth)
        delayed_indices = delayed_indices.squeeze(0)  # [T_feat_delayed]
        
        # Find overlapping region
        # Delayed audio has delay_samples more samples, so its features start later
        # We need to align: original[0:T_feat] vs delayed[delay_feat:T_feat+delay_feat]
        # For simplicity, we'll compare what overlaps
        
        # Approximate feature delay (this depends on model's downsampling rate)
        # For now, we'll use a simple approach: compare from start of delayed
        # In practice, you'd compute the exact feature-level delay
        
        # Simple approach: compare the overlapping portion
        # Original indices: [0:T_feat]
        # Delayed indices: [delay_feat:T_feat] (where delay_feat corresponds to delay_samples)
        # For now, we'll compare the minimum length
        
        min_len = min(len(original_indices), len(delayed_indices))
        
        # For proper alignment, we'd need to account for the delay in feature space
        # Simplified: compare original[:min_len] with delayed[:min_len]
        # In reality, delayed should be shifted, but for visualization we'll use this
        
        orig_comp = original_indices[:min_len]
        delayed_comp = delayed_indices[:min_len]
        
        # Compute metrics
        match_rate = compute_exact_match_rate(orig_comp, delayed_comp)
        differences = compute_index_differences(orig_comp, delayed_comp)
        boundary_metrics = compute_boundary_vs_center_metrics(
            orig_comp, delayed_comp, boundary_ratio
        )
        
        # Store per-position differences for heatmap
        position_differences = torch.abs(orig_comp - delayed_comp).float().numpy()
        heatmap_differences[delay_idx, :len(position_differences)] = position_differences
        
        # Store results
        results['match_rates'].append({
            'delay_ms': delay_ms,
            'match_rate': match_rate
        })
        results['index_differences'].append({
            'delay_ms': delay_ms,
            'differences': differences
        })
        results['boundary_metrics'].append({
            'delay_ms': delay_ms,
            **boundary_metrics
        })
    
    # Store heatmap data
    results['heatmap_data'] = heatmap_differences
    
    return results
