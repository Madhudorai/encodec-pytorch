"""Slice consistency module for EnCodec training.

This module implements slice consistency loss, which encourages the model
to produce consistent features when encoding the full audio vs. a slice of the audio.
This helps improve the robustness of the codec to variable-length inputs.

Based on ConsistencyInNeuralCodec implementation.
"""

import typing as tp
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SliceInterval:
    """Data class to hold slice interval information.
    
    Attributes:
        start_point: Start position in audio space (samples) [B]
        end_point: End position in audio space (samples) [B]
        start_split: Start position in feature space (tokens) [B]
        end_split: End position in feature space (tokens) [B]
        split_interval_lengths: Length of split interval in feature space [B]
    """
    start_point: torch.Tensor  # [B] - audio space
    end_point: torch.Tensor  # [B] - audio space
    start_split: torch.Tensor  # [B] - feature space
    end_split: torch.Tensor  # [B] - feature space
    split_interval_lengths: torch.Tensor  # [B] - feature space


class SliceConsistency(nn.Module):
    """Slice consistency loss module.
    
    Computes consistency loss between features extracted from full audio
    and features extracted from a slice of the audio. The loss encourages
    the model to produce consistent features at overlapping positions.
    
    Based on the ConsistencyInNeuralCodec implementation.
    
    Args:
        slice_interval_type: Type of slice interval ("random", "static", "static_duration")
        split_interval_percentage: Percentage of audio to use for slice (0.0-1.0)
        feature_types: List of feature types to compare ["quant_in", "quant_out", "sub_quants"]
        loss_types: List of loss types to use ["mse_loss"]
        loss_weights: List of weights for each loss type
        target_sr: Target sample rate
        ds_rate: Downsampling rate (encoder hop length)
        mse_loss_reduction: Reduction method for MSE loss ("mean" or "sum")
    """
    
    def __init__(
        self,
        slice_interval_type: str = "random",
        split_interval_percentage: float = 0.2,
        feature_types: tp.List[str] = None,
        loss_types: tp.List[str] = None,
        loss_weights: tp.List[float] = None,
        target_sr: int = 24000,
        ds_rate: int = 320,
        mse_loss_reduction: str = "mean",
    ):
        super().__init__()
        
        self.slice_interval_type = slice_interval_type
        self.split_interval_percentage = split_interval_percentage
        self.feature_types = feature_types if feature_types is not None else ["quant_in"]
        self.loss_types = loss_types if loss_types is not None else ["mse_loss"]
        self.loss_weights = loss_weights if loss_weights is not None else [10.0]
        self.target_sr = target_sr
        self.ds_rate = ds_rate
        self.mse_loss_reduction = mse_loss_reduction
        
        # Validate inputs
        assert len(self.feature_types) == len(self.loss_types), \
            f"feature_types and loss_types must have same length, got {len(self.feature_types)} and {len(self.loss_types)}"
        assert len(self.loss_types) == len(self.loss_weights), \
            f"loss_types and loss_weights must have same length, got {len(self.loss_types)} and {len(self.loss_weights)}"
    
    def gather_features(
        self,
        features: torch.Tensor,
        slice_features: torch.Tensor,
        slice_interval: SliceInterval,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather features from full audio at overlapping positions with slice.
        
        Uses torch.gather with attention masks to handle variable-length sequences.
        
        Args:
            features: Full audio features [B, T, D] or [K, B, T, D]
            slice_features: Slice audio features [B, T_slice, D] or [K, B, T_slice, D]
            slice_interval: SliceInterval with alignment information
            
        Returns:
            Tuple of (gathered_features, aligned_slice_features, attention_mask)
            - gathered_features: Features from full audio at overlapping positions
            - aligned_slice_features: Slice features aligned to same positions
            - attention_mask: Mask indicating valid positions [B, T] or [K, B, T]
        """
        if features.dim() == 3:
            # Regular features: [B, T, D]
            batch_size, seq_len, dim = features.shape
            sub_quant_first = False
        elif features.dim() == 4:
            # Sub-quantizer features: [K, B, T, D]
            num_sub_quants, batch_size, seq_len, dim = features.shape
            sub_quant_first = True
        else:
            raise ValueError(f"Expected 3D or 4D features, got {features.dim()}D")
        
        device = features.device
        start_split = slice_interval.start_split  # [B]
        end_split = slice_interval.end_split  # [B]
        split_interval_lengths = slice_interval.split_interval_lengths  # [B]
        
        # Maximum interval length across batch (for padding to same length)
        interval_length = split_interval_lengths.max().item()
        
        # Create indices for gathering
        if not sub_quant_first:
            # [B, interval_length]
            indices = torch.arange(interval_length, device=device).expand(batch_size, interval_length)
        else:
            # [K, B, interval_length]
            indices = torch.arange(interval_length, device=device).expand(num_sub_quants, batch_size, interval_length)
        
        # Adjust indices by start_split
        start_split = start_split.unsqueeze(-1)  # [B, 1]
        if not sub_quant_first:
            adjusted_indices = start_split + indices  # [B, interval_length]
        else:
            adjusted_indices = start_split.unsqueeze(0) + indices  # [K, B, interval_length]
        
        # Create attention mask (valid positions)
        end_split = end_split.unsqueeze(-1)  # [B, 1]
        if not sub_quant_first:
            attention_mask = adjusted_indices < end_split.expand(batch_size, interval_length)
        else:
            attention_mask = adjusted_indices < end_split.unsqueeze(0).expand(num_sub_quants, batch_size, interval_length)
        
        # Apply attention mask to indices (set invalid positions to 0)
        attention_masked_indices = adjusted_indices * attention_mask
        
        # Gather features from full audio
        if not sub_quant_first:
            # Expand indices for gather: [B, interval_length, D]
            gather_indices = attention_masked_indices.unsqueeze(-1).expand(-1, -1, dim)
            gathered_features = torch.gather(features, 1, gather_indices)
        else:
            # Expand indices for gather: [K, B, interval_length, D]
            gather_indices = attention_masked_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
            gathered_features = torch.gather(features, 2, gather_indices)
        
        # Align slice features to the same length (take first interval_length elements)
        if not sub_quant_first:
            # [B, T_slice, D] -> [B, interval_length, D]
            slice_feat_length = slice_features.shape[1]
            if slice_feat_length >= interval_length:
                aligned_slice_features = slice_features[:, :interval_length, :]
            else:
                # Pad if slice is shorter
                padding = torch.zeros(
                    batch_size, interval_length - slice_feat_length, dim,
                    device=device, dtype=slice_features.dtype
                )
                aligned_slice_features = torch.cat([slice_features, padding], dim=1)
        else:
            # [K, B, T_slice, D] -> [K, B, interval_length, D]
            slice_feat_length = slice_features.shape[2]
            if slice_feat_length >= interval_length:
                aligned_slice_features = slice_features[:, :, :interval_length, :]
            else:
                # Pad if slice is shorter
                padding = torch.zeros(
                    num_sub_quants, batch_size, interval_length - slice_feat_length, dim,
                    device=device, dtype=slice_features.dtype
                )
                aligned_slice_features = torch.cat([slice_features, padding], dim=2)
        
        # Apply attention mask to both features
        if not sub_quant_first:
            attention_mask_expanded = attention_mask.unsqueeze(-1).type_as(gathered_features)
            gathered_features = gathered_features * attention_mask_expanded
            aligned_slice_features = aligned_slice_features * attention_mask_expanded
        else:
            attention_mask_expanded = attention_mask.unsqueeze(-1).type_as(gathered_features)
            gathered_features = gathered_features * attention_mask_expanded
            aligned_slice_features = aligned_slice_features * attention_mask_expanded
        
        return gathered_features, aligned_slice_features, attention_mask
    
    def compute_consistency_loss(
        self,
        full_features: tp.Dict[str, torch.Tensor],
        slice_features: tp.Dict[str, torch.Tensor],
        feature_lengths: torch.Tensor,
        codebook_indices: tp.Optional[torch.Tensor] = None,
        slice_codebook_indices: tp.Optional[torch.Tensor] = None,
        slice_interval: tp.Optional[SliceInterval] = None,
    ) -> tp.Dict[str, tp.Any]:
        """Compute slice consistency loss.
        
        Args:
            full_features: Dictionary of full audio features
                - quant_in: [B, T, D] - quantizer input features
                - quant_out: [B, T, D] - quantizer output features
                - sub_quants: [K, B, T, D] - individual quantizer outputs (optional)
            slice_features: Dictionary of slice audio features (same structure as full_features)
            feature_lengths: Length of full audio features [B]
            codebook_indices: Full audio codebook indices [K, B, T] (optional)
            slice_codebook_indices: Slice audio codebook indices [K, B, T_slice] (optional)
            slice_interval: SliceInterval object with alignment information
            
        Returns:
            Dictionary with:
                - 'loss': Total weighted consistency loss [scalar tensor]
                - 'loss_dict': Dictionary of individual loss components
        """
        device = next(iter(full_features.values())).device
        
        # Initialize loss components
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if slice_interval is None:
            raise ValueError("slice_interval must be provided")
        
        # For each feature type, compute consistency loss
        for i, feature_type in enumerate(self.feature_types):
            if feature_type not in full_features or feature_type not in slice_features:
                continue
            
            full_feat = full_features[feature_type]  # [B, T, D] or [K, B, T, D]
            slice_feat = slice_features[feature_type]  # [B, T_slice, D] or [K, B, T_slice, D]
            
            # Gather features using proper attention masking
            gathered_features, aligned_slice_features, attention_mask = self.gather_features(
                full_feat, slice_feat, slice_interval
            )
            
            # Compute loss based on loss type
            if self.loss_types[i] == "mse_loss":
                # Compute MSE loss
                diff = gathered_features - aligned_slice_features
                mse = diff ** 2  # [B, T, D] or [K, B, T, D]
                
                # Apply attention mask
                if attention_mask.dim() == 2:
                    # [B, T] -> [B, T, 1]
                    attention_mask_expanded = attention_mask.unsqueeze(-1).type_as(mse)
                else:
                    # [K, B, T] -> [K, B, T, 1]
                    attention_mask_expanded = attention_mask.unsqueeze(-1).type_as(mse)
                
                mse = mse * attention_mask_expanded
                
                # Compute total valid elements
                tot = attention_mask.sum()
                
                if self.mse_loss_reduction == "mean":
                    feature_loss = mse.sum() / tot / mse.shape[-1]  # Divide by valid elements and feature dim
                elif self.mse_loss_reduction == "sum":
                    feature_loss = mse.sum()
                else:
                    raise ValueError(f"Unknown mse_loss_reduction: {self.mse_loss_reduction}")
                
                # For sub_quants, average across quantizers
                if feature_type == "sub_quants" and gathered_features.dim() == 4:
                    # [K, B, T, D] -> average across K dimension
                    feature_loss = feature_loss / gathered_features.shape[0]
                
            else:
                raise ValueError(f"Unknown loss type: {self.loss_types[i]}")
            
            # Weight the loss
            weighted_loss = feature_loss * self.loss_weights[i]
            loss_dict[f"{feature_type}_{self.loss_types[i]}"] = weighted_loss
            total_loss = total_loss + weighted_loss
        
        return {
            'loss': total_loss,
            'loss_dict': loss_dict,
        }
