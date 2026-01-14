"""Visualization functions for consistency analysis."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import typing as tp
from pathlib import Path


def plot_slice_consistency_heatmap(full_indices: torch.Tensor, slice_indices: torch.Tensor,
                                   save_path: tp.Optional[Path] = None):
    """Plot heatmap comparing full audio vs slice codebook indices.
    
    Args:
        full_indices: Full audio codebook 0 indices [T_feat]
        slice_indices: Slice codebook 0 indices [T_feat_slice]
        save_path: Optional path to save figure
    """
    # Compute differences
    min_len = min(len(full_indices), len(slice_indices))
    full_comp = full_indices[:min_len].numpy()
    slice_comp = slice_indices[:min_len].numpy()
    
    differences = np.abs(full_comp - slice_comp)
    matches = (full_comp == slice_comp).astype(float)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot 1: Full indices
    axes[0].plot(full_comp, label='Full Audio', alpha=0.7)
    axes[0].set_ylabel('Codebook Index')
    axes[0].set_title('Full Audio Codebook 0 Indices')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Slice indices
    axes[1].plot(slice_comp, label='Slice', alpha=0.7, color='orange')
    axes[1].set_ylabel('Codebook Index')
    axes[1].set_title('Slice Codebook 0 Indices')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Differences (heatmap style)
    axes[2].imshow(differences.reshape(1, -1), aspect='auto', cmap='Reds', 
                   interpolation='nearest', vmin=0, vmax=differences.max() if len(differences) > 0 else 1)
    axes[2].set_ylabel('Difference')
    axes[2].set_xlabel('Temporal Position (feature frames)')
    axes[2].set_title('Index Differences (Red = Larger Difference)')
    axes[2].set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_time_delay_heatmap(heatmap_data: np.ndarray, delay_values_ms: tp.List[float],
                            save_path: tp.Optional[Path] = None):
    """Plot heatmap of index differences vs delay amount.
    
    Args:
        heatmap_data: 2D array [num_delays, num_positions] with index differences
        delay_values_ms: List of delay values in milliseconds
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create heatmap
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', 
                   interpolation='nearest', vmin=0, vmax=heatmap_data.max() if heatmap_data.size > 0 else 1)
    
    # Set labels
    ax.set_xlabel('Temporal Position (feature frames)', fontsize=12)
    ax.set_ylabel('Delay (ms)', fontsize=12)
    ax.set_title('Codebook 0 Index Differences: Delay vs Position', fontsize=14)
    
    # Set y-axis ticks
    ax.set_yticks(range(len(delay_values_ms)))
    ax.set_yticklabels([f'{d:.1f}' for d in delay_values_ms])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Index Difference', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_match_rate_vs_delay(match_rates: tp.List[tp.Dict[str, float]],
                             save_path: tp.Optional[Path] = None):
    """Plot match rate vs delay amount.
    
    Args:
        match_rates: List of dicts with 'delay_ms' and 'match_rate' keys
        save_path: Optional path to save figure
    """
    delays = [m['delay_ms'] for m in match_rates]
    rates = [m['match_rate'] * 100 for m in match_rates]  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(delays, rates, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Delay (ms)', fontsize=12)
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax.set_title('Codebook 0 Consistency: Match Rate vs Time Delay', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_boundary_vs_center(boundary_metrics: tp.List[tp.Dict[str, float]],
                            save_path: tp.Optional[Path] = None):
    """Plot boundary vs center match rates.
    
    Args:
        boundary_metrics: List of dicts with boundary/center metrics
        save_path: Optional path to save figure
    """
    if not boundary_metrics:
        return
    
    # Extract metrics
    boundary_rates = [m.get('boundary_match_rate', 0) * 100 for m in boundary_metrics]
    center_rates = [m.get('center_match_rate', 0) * 100 for m in boundary_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(boundary_metrics))
    width = 0.35
    
    ax.bar(x - width/2, boundary_rates, width, label='Boundary', alpha=0.8)
    ax.bar(x + width/2, center_rates, width, label='Center', alpha=0.8)
    
    ax.set_xlabel('Test Condition', fontsize=12)
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax.set_title('Boundary vs Center Match Rates', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Test {i+1}' for i in range(len(boundary_metrics))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_gain_consistency(match_rates: tp.List[tp.Dict[str, float]],
                          save_path: tp.Optional[Path] = None):
    """Plot match rate vs gain value.
    
    Args:
        match_rates: List of dicts with 'gain' and 'match_rate' keys
        save_path: Optional path to save figure
    """
    gains = [m['gain'] for m in match_rates]
    rates = [m['match_rate'] * 100 for m in match_rates]  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(gains, rates, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Gain Multiplier', fontsize=12)
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax.set_title('Codebook 0 Consistency: Match Rate vs Gain Adjustment', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Original (gain=1.0)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_index_difference_histogram(differences: np.ndarray, condition_name: str,
                                    save_path: tp.Optional[Path] = None):
    """Plot histogram of index differences.
    
    Args:
        differences: Array of index differences
        condition_name: Name of test condition (for title)
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(differences, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Index Difference', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of Index Differences: {condition_name}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_diff = np.mean(differences)
    median_diff = np.median(differences)
    ax.axvline(mean_diff, color='r', linestyle='--', label=f'Mean: {mean_diff:.2f}')
    ax.axvline(median_diff, color='g', linestyle='--', label=f'Median: {median_diff:.2f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
