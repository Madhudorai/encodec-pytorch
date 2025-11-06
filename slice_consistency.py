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
        self.loss_weights = loss_weights if loss_weights is not None else [20.0]
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
            
            # Check if both features are the same length (slice vs slice comparison)
            # If so, we can directly compare without gathering
            if full_feat.shape[1] == slice_feat.shape[1] and full_feat.dim() == slice_feat.dim():
                # Direct comparison: slice from original vs slice from perturbed (same length)
                gathered_features = full_feat
                aligned_slice_features = slice_feat
                # Create attention mask for all valid positions
                if full_feat.dim() == 3:
                    # [B, T, D]
                    attention_mask = torch.ones(full_feat.shape[0], full_feat.shape[1], device=full_feat.device, dtype=torch.bool)
                else:
                    # [K, B, T, D]
                    attention_mask = torch.ones(full_feat.shape[0], full_feat.shape[1], full_feat.shape[2], device=full_feat.device, dtype=torch.bool)
            else:
                # Gather features using proper attention masking (full vs slice comparison)
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
        
        # Compute codebook consistency accuracy if codebook indices are provided
        codebook_accuracy = None
        first_codebook_accuracy = None
        codebook_accuracies = {}  # Store accuracy for each codebook
        
        if codebook_indices is not None and slice_codebook_indices is not None:
            # codebook_indices: [K, B, T] - could be from full audio or slice from original
            # slice_codebook_indices: [K, B, T_slice] - could be from slice or slice from perturbed
            # slice_interval: has start_split, end_split, split_interval_lengths
            
            K, B, T = codebook_indices.shape
            _, _, T_slice = slice_codebook_indices.shape
            
            # Check if both are the same length (slice vs slice comparison)
            if T == T_slice:
                # Direct comparison: slice from original vs slice from perturbed (same length)
                gathered_full_indices = codebook_indices  # [K, B, T]
                aligned_slice_indices = slice_codebook_indices  # [K, B, T_slice] where T == T_slice
                # Create attention mask for all valid positions
                attention_mask = torch.ones(K, B, T, device=device, dtype=torch.bool)
            else:
                # Gather codebook indices from full audio at overlapping positions
                start_split = slice_interval.start_split  # [B]
                end_split = slice_interval.end_split  # [B]
                split_interval_lengths = slice_interval.split_interval_lengths  # [B]
                max_interval_length = split_interval_lengths.max().item()
                
                # Gather codebook indices from full audio at overlapping positions
                # Create indices for gathering: [K, B, interval_length]
                indices = torch.arange(max_interval_length, device=device).expand(K, B, max_interval_length)
                start_split_expanded = start_split.unsqueeze(0).unsqueeze(-1)  # [1, B, 1]
                adjusted_indices = start_split_expanded + indices  # [K, B, interval_length]
                
                # Create attention mask
                end_split_expanded = end_split.unsqueeze(0).unsqueeze(-1)  # [1, B, 1]
                attention_mask = adjusted_indices < end_split_expanded.expand(K, B, max_interval_length)
                
                # Gather full codebook indices at overlapping positions
                gathered_full_indices = torch.gather(
                    codebook_indices,  # [K, B, T]
                    2,  # gather along time dimension
                    adjusted_indices.clamp(0, T - 1)  # [K, B, interval_length]
                )  # [K, B, interval_length]
                
                # Align slice codebook indices to same length
                if T_slice >= max_interval_length:
                    aligned_slice_indices = slice_codebook_indices[:, :, :max_interval_length]  # [K, B, interval_length]
                else:
                    # Pad if slice is shorter (pad with -1 to indicate invalid)
                    padding = torch.full(
                        (K, B, max_interval_length - T_slice),
                        -1, device=device, dtype=slice_codebook_indices.dtype
                    )
                    aligned_slice_indices = torch.cat([slice_codebook_indices, padding], dim=2)
            
            # Apply attention mask - need to handle per-codebook mask
            # For each codebook, create its own attention mask
            codebook_attention_mask = attention_mask  # [K, B, interval_length]
            
            # Compute accuracy for each codebook
            for k in range(K):
                # Get attention mask for this codebook (first dimension is batch, second is time)
                cb_attention = codebook_attention_mask[k]  # [B, interval_length]
                
                # Get indices for this codebook
                gathered_cb = gathered_full_indices[k]  # [B, interval_length]
                aligned_cb = aligned_slice_indices[k]  # [B, interval_length]
                
                # Compute matches: indices match AND both are valid (not -1) AND within attention mask
                valid_mask = (gathered_cb >= 0) & (aligned_cb >= 0) & cb_attention
                matches = (gathered_cb == aligned_cb) & valid_mask
                
                # Compute accuracy for this codebook
                total_valid = valid_mask.sum().item()
                if total_valid > 0:
                    acc = matches.sum().item() / total_valid
                    codebook_accuracies[f'codebook{k}_accuracy'] = acc
                else:
                    codebook_accuracies[f'codebook{k}_accuracy'] = 0.0
            
            # Overall accuracy across all codebooks
            all_valid = ((gathered_full_indices >= 0) & (aligned_slice_indices >= 0) & codebook_attention_mask)
            all_matches = (gathered_full_indices == aligned_slice_indices) & all_valid
            total_valid = all_valid.sum().item()
            if total_valid > 0:
                codebook_accuracy = all_matches.sum().item() / total_valid
            else:
                codebook_accuracy = 0.0
            
            # First codebook accuracy (most important)
            if K > 0 and 'codebook0_accuracy' in codebook_accuracies:
                first_codebook_accuracy = codebook_accuracies['codebook0_accuracy']
            
            # First 3 codebooks average accuracy
            if K >= 3:
                first_3_accs = [codebook_accuracies.get(f'codebook{k}_accuracy', 0.0) for k in range(3)]
                codebook_accuracies['first_3_codebooks_accuracy'] = sum(first_3_accs) / 3.0
        
        result = {
            'loss': total_loss,
            'loss_dict': loss_dict,
        }
        
        if codebook_accuracy is not None:
            result['codebook_accuracy'] = codebook_accuracy
        if first_codebook_accuracy is not None:
            result['first_codebook_accuracy'] = first_codebook_accuracy
        
        # Add all codebook accuracies to result
        result.update(codebook_accuracies)
        
        return result
    
    def compute_augmentation_consistency(
        self,
        original_codebook_indices: torch.Tensor,
        perturbed_codebook_indices: torch.Tensor,
        attention_mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Dict[str, float]:
        """Compute augmentation consistency accuracy.
        
        Compares codebook indices between original full audio and perturbed full audio.
        This measures how consistent the model is when the same audio is perturbed.
        
        Args:
            original_codebook_indices: Original full audio codebook indices [K, B, T]
            perturbed_codebook_indices: Perturbed full audio codebook indices [K, B, T]
            attention_mask: Optional attention mask [B, T] or [K, B, T]
            
        Returns:
            Dictionary with augmentation consistency accuracy metrics:
                - 'augmentation_consistency_accuracy': Overall accuracy across all codebooks
                - 'augmentation_consistency_codebook0_accuracy': First codebook accuracy
                - 'augmentation_consistency_first_3_codebooks_accuracy': Average of first 3 codebooks
                - 'augmentation_consistency_codebook{k}_accuracy': Accuracy for each codebook k
        """
        device = original_codebook_indices.device
        K, B, T = original_codebook_indices.shape
        
        # Ensure shapes match
        if perturbed_codebook_indices.shape != original_codebook_indices.shape:
            # If lengths differ, use minimum length
            min_T = min(T, perturbed_codebook_indices.shape[2])
            original_codebook_indices = original_codebook_indices[:, :, :min_T]
            perturbed_codebook_indices = perturbed_codebook_indices[:, :, :min_T]
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, :min_T]
                else:
                    attention_mask = attention_mask[:, :, :min_T]
            T = min_T
        
        # Create attention mask if not provided (all positions are valid)
        if attention_mask is None:
            attention_mask = torch.ones(K, B, T, device=device, dtype=torch.bool)
        elif attention_mask.dim() == 2:
            # [B, T] -> [K, B, T]
            attention_mask = attention_mask.unsqueeze(0).expand(K, -1, -1)
        
        # Compute accuracy for each codebook
        augmentation_accuracies = {}
        
        for k in range(K):
            # Get codebook indices for this codebook
            orig_cb = original_codebook_indices[k]  # [B, T]
            pert_cb = perturbed_codebook_indices[k]  # [B, T]
            cb_attention = attention_mask[k]  # [B, T]
            
            # Compute matches: indices match AND within attention mask
            valid_mask = cb_attention
            matches = (orig_cb == pert_cb) & valid_mask
            
            # Compute accuracy for this codebook
            total_valid = valid_mask.sum().item()
            if total_valid > 0:
                acc = matches.sum().item() / total_valid
                augmentation_accuracies[f'augmentation_consistency_codebook{k}_accuracy'] = acc
            else:
                augmentation_accuracies[f'augmentation_consistency_codebook{k}_accuracy'] = 0.0
        
        # Overall accuracy across all codebooks
        all_valid = attention_mask
        all_matches = (original_codebook_indices == perturbed_codebook_indices) & all_valid
        total_valid = all_valid.sum().item()
        if total_valid > 0:
            augmentation_accuracies['augmentation_consistency_accuracy'] = all_matches.sum().item() / total_valid
        else:
            augmentation_accuracies['augmentation_consistency_accuracy'] = 0.0
        
        # First codebook accuracy (most important)
        if K > 0 and 'augmentation_consistency_codebook0_accuracy' in augmentation_accuracies:
            augmentation_accuracies['augmentation_consistency_codebook0_accuracy'] = augmentation_accuracies['augmentation_consistency_codebook0_accuracy']
        
        # First 3 codebooks average accuracy
        if K >= 3:
            first_3_accs = [augmentation_accuracies.get(f'augmentation_consistency_codebook{k}_accuracy', 0.0) for k in range(3)]
            augmentation_accuracies['augmentation_consistency_first_3_codebooks_accuracy'] = sum(first_3_accs) / 3.0
        
        return augmentation_accuracies
