"""Metrics computation for consistency analysis."""

import numpy as np
import torch
import typing as tp


def compute_exact_match_rate(indices1: torch.Tensor, indices2: torch.Tensor, 
                             valid_mask: tp.Optional[torch.Tensor] = None) -> float:
    """Compute exact match rate between two codebook index sequences.
    
    Args:
        indices1: First index sequence [B, T] or [T]
        indices2: Second index sequence [B, T] or [T]
        valid_mask: Optional mask for valid positions [B, T] or [T]
    
    Returns:
        match_rate: Percentage of positions where indices match exactly
    """
    # Flatten if needed
    if indices1.dim() > 1:
        indices1 = indices1.flatten()
        indices2 = indices2.flatten()
        if valid_mask is not None:
            valid_mask = valid_mask.flatten()
    
    # Compute matches
    matches = (indices1 == indices2)
    
    # Apply valid mask if provided
    if valid_mask is not None:
        matches = matches & valid_mask.bool()
        total_valid = valid_mask.sum().item()
    else:
        total_valid = len(indices1)
    
    if total_valid == 0:
        return 0.0
    
    match_rate = matches.sum().item() / total_valid
    return match_rate


def compute_index_differences(indices1: torch.Tensor, indices2: torch.Tensor,
                              valid_mask: tp.Optional[torch.Tensor] = None) -> np.ndarray:
    """Compute absolute differences between codebook indices.
    
    Args:
        indices1: First index sequence [B, T] or [T]
        indices2: Second index sequence [B, T] or [T]
        valid_mask: Optional mask for valid positions [B, T] or [T]
    
    Returns:
        differences: Array of absolute index differences
    """
    # Flatten if needed
    if indices1.dim() > 1:
        indices1 = indices1.flatten()
        indices2 = indices2.flatten()
        if valid_mask is not None:
            valid_mask = valid_mask.flatten()
    
    # Compute differences
    differences = torch.abs(indices1 - indices2).float()
    
    # Apply valid mask if provided
    if valid_mask is not None:
        differences = differences[valid_mask.bool()]
    
    return differences.numpy()


def compute_boundary_vs_center_metrics(indices1: torch.Tensor, indices2: torch.Tensor,
                                       boundary_ratio: float = 0.1) -> tp.Dict[str, float]:
    """Compute metrics separately for boundary and center regions.
    
    Args:
        indices1: First index sequence [T]
        indices2: Second index sequence [T]
        boundary_ratio: Ratio of boundary region (e.g., 0.1 = 10% on each side)
    
    Returns:
        metrics: Dictionary with boundary and center metrics
    """
    T = len(indices1)
    boundary_size = int(T * boundary_ratio)
    
    # Define regions
    left_boundary = slice(0, boundary_size)
    center = slice(boundary_size, T - boundary_size)
    right_boundary = slice(T - boundary_size, T)
    
    # Compute metrics for each region
    metrics = {}
    
    # Left boundary
    left_match = compute_exact_match_rate(
        indices1[left_boundary], 
        indices2[left_boundary]
    )
    metrics['left_boundary_match_rate'] = left_match
    
    # Center
    center_match = compute_exact_match_rate(
        indices1[center], 
        indices2[center]
    )
    metrics['center_match_rate'] = center_match
    
    # Right boundary
    right_match = compute_exact_match_rate(
        indices1[right_boundary], 
        indices2[right_boundary]
    )
    metrics['right_boundary_match_rate'] = right_match
    
    # Combined boundary
    boundary_indices1 = torch.cat([indices1[left_boundary], indices1[right_boundary]])
    boundary_indices2 = torch.cat([indices2[left_boundary], indices2[right_boundary]])
    boundary_match = compute_exact_match_rate(boundary_indices1, boundary_indices2)
    metrics['boundary_match_rate'] = boundary_match
    
    # Boundary degradation
    metrics['boundary_degradation'] = center_match - boundary_match
    
    return metrics
