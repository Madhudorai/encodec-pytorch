"""EnCodec model with consistency losses for Eigenscape dataset."""

import math
from pathlib import Path
import typing as tp

import numpy as np
import torch
from torch import nn

import quantization as qt
import modules as m
from utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url
import random

from model import EncodecModel
from consistency_0 import SliceConsistency
from perturb_encoder import PerturbEncoder

ROOT_URL = 'https://dl.fbaipublicfiles.com/encodec/v0/'


class EncodecModelWithSliceConsistency(EncodecModel):
    """EnCodec model with slice consistency loss.
    
    Extends the base EnCodecModel to support slice consistency training.
    During training, it encodes both the full audio and a random slice,
    then computes consistency loss between features at overlapping positions.
    Also supports perturbation augmentation (volume/gain, inversion) for consistency.
    """
    
    def __init__(
        self,
        encoder: m.SEANetEncoder,
        decoder: m.SEANetDecoder,
        quantizer: qt.ResidualVectorQuantizer,
        target_bandwidths: tp.List[float],
        sample_rate: int,
        channels: int,
        normalize: bool = False,
        segment: tp.Optional[float] = None,
        overlap: float = 0.01,
        name: str = 'unset',
        # Slice consistency parameters
        slice_consistency: tp.Optional[dict] = None,
        # Perturbation parameters
        perturb_encoder: tp.Optional[dict] = None,
        # Inter-channel consistency parameters
        inter_channel_consistency: tp.Optional[dict] = None,
    ):
        super().__init__(
            encoder, decoder, quantizer, target_bandwidths,
            sample_rate, channels, normalize, segment, overlap, name
        )
        
        # Initialize slice consistency module
        if slice_consistency is not None:
            # Extract weights for constraint losses before passing to SliceConsistency
            self.augmentation_constraint_loss_weight = slice_consistency.get('augmentation_constraint_loss_weight', 15.0)
            self.slice_consistency_constraint_loss_weight = slice_consistency.get('slice_consistency_constraint_loss_weight', 15.0)
            # Remove these from slice_consistency dict before passing to SliceConsistency
            slice_consistency_for_init = {k: v for k, v in slice_consistency.items() 
                                         if k not in ['augmentation_constraint_loss_weight', 'slice_consistency_constraint_loss_weight']}
            self.slice_consistency = SliceConsistency(**slice_consistency_for_init)
            self.use_slice_consistency = True
        else:
            self.slice_consistency = None
            self.use_slice_consistency = False
            self.augmentation_constraint_loss_weight = 15.0
            self.slice_consistency_constraint_loss_weight = 15.0
        
        # Initialize perturb encoder
        if perturb_encoder is not None:
            self.perturb_encoder = PerturbEncoder(**perturb_encoder)
        else:
            self.perturb_encoder = None
        
        # Initialize inter-channel consistency
        if inter_channel_consistency is not None and inter_channel_consistency.get('enabled', False):
            self.inter_channel_consistency = inter_channel_consistency
            self.use_inter_channel_consistency = True
        else:
            self.inter_channel_consistency = None
            self.use_inter_channel_consistency = False
    
    def _quantize_from_codebook(self, emb: torch.Tensor, sample_rate: int, bandwidth: float, start_codebook: int = 1):
        """Quantize embeddings starting from a specific codebook index.
        
        Args:
            emb: Encoder embeddings [B, D, T]
            sample_rate: Sample rate
            bandwidth: Target bandwidth
            start_codebook: Codebook index to start from (0-indexed, default 1 to skip codebook 0)
            
        Returns:
            QuantizedResult with quantized embeddings, codes, and loss
        """
        from quantization.vq import QuantizedResult
        
        bw_per_q = self.quantizer.get_bandwidth_per_quantizer(sample_rate)
        n_q_total = self.quantizer.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        
        # Only use codebooks from start_codebook onwards
        n_q_used = max(1, n_q_total - start_codebook)  # At least 1 codebook
        
        # Manually quantize starting from start_codebook
        residual = emb
        quantized_out = torch.zeros_like(emb)
        all_indices = []
        all_losses = []
        
        for i in range(start_codebook, start_codebook + n_q_used):
            if i >= len(self.quantizer.vq.layers):
                break
            layer = self.quantizer.vq.layers[i]
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)
        
        if all_indices:
            out_indices = torch.stack(all_indices)  # [n_q_used, B, T]
            out_losses = torch.stack(all_losses)
            commit_loss = torch.mean(out_losses)
            bw = torch.tensor(n_q_used * bw_per_q).to(emb)
            return QuantizedResult(quantized_out, out_indices, bw, penalty=commit_loss)
        else:
            # Fallback: return unquantized
            dummy_indices = torch.zeros((1, emb.shape[0], emb.shape[2]), dtype=torch.long, device=emb.device)
            return QuantizedResult(emb, dummy_indices, torch.tensor(0.0).to(emb), penalty=torch.tensor(0.0).to(emb))
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_embeddings: bool = False,
        return_slice_consistency: bool = False,
        batch_idx: tp.Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass with optional slice consistency computation.
        
        Args:
            x: Input audio tensor [B, C, T] where C can be > 1 for multi-channel
            return_embeddings: Whether to return intermediate embeddings
            return_slice_consistency: Whether to compute slice consistency loss
            
        Returns:
            If return_slice_consistency is False:
                - output: Reconstructed audio [B, C, T]
                - loss_w: Commitment loss
                - frames: Encoded frames
            If return_slice_consistency is True:
                - output: Reconstructed audio [B, C, T]
                - loss_w: Commitment loss
                - frames: Encoded frames
                - slice_consistency_output: Dictionary with slice consistency loss (includes intra and inter-channel)
        """
        # Check if we have multi-channel input (C > 1)
        B, C, T = x.shape
        is_multi_channel = C > 1
        
        # For multi-channel, reshape to process each channel separately
        # [B, C, T] -> [B*C, 1, T]
        # NOTE: We do separate forward passes because encoding is NOT context-independent.
        # If we concatenated channels in time [B, 1, 2*T], the encoding of [T-2T] would be
        # affected by the audio in [0-T], which would break inter-channel consistency.
        if is_multi_channel:
            x_reshaped = x.reshape(B * C, 1, T)  # [B*C, 1, T]
        else:
            x_reshaped = x
        
        # Store original audio if we need it for slice extraction or augmentation consistency
        # We need it if: slice consistency is enabled AND we're perturbing full audio
        # (because we want to extract slice from original, not perturbed, so slice can be augmented independently)
        # OR if we want to compute augmentation consistency (original vs perturbed)
        need_original = (
            return_slice_consistency and 
            self.use_slice_consistency and 
            self.perturb_encoder is not None and 
            self.perturb_encoder.perturb_all_audio
        )
        orig_audio = x_reshaped.clone() if need_original else None
        
        # Compute original codebook indices for augmentation consistency (before perturbation)
        # We'll compute this after we know the bandwidth (in training/eval mode)
        original_codebook_indices = None
        
        # Apply perturbation to full audio if enabled
        # When slice consistency is enabled, we want to calculate consistency on augmented audio
        # So if perturb_all_audio is True, we always augment the full audio
        # (Individual augmentations have their own apply_prob for probabilistic application)
        # For multi-channel, apply perturbations per-channel (each channel gets independent perturbations)
        if self.perturb_encoder is not None and self.perturb_encoder.perturb_all_audio:
            # Only create sample indices for deterministic behavior (validation), skip in training for speed
            if batch_idx is not None:
                batch_size = x_reshaped.shape[0]
                sample_indices = torch.arange(batch_size, device=x_reshaped.device)
                x_reshaped = self.perturb_encoder(x_reshaped, batch_idx=batch_idx, sample_indices=sample_indices)
            else:
                # Training mode: use simple call without extra parameters (faster)
                # Each channel gets independent perturbations
                x_reshaped = self.perturb_encoder(x_reshaped)
        
        frames = self.encode(x_reshaped)
        
        if self.training:
            # Standard training forward
            loss_w = torch.tensor([0.0], device=x_reshaped.device, requires_grad=True)
            codes = []
            quantized_embeddings = []
            
            # Random bandwidth selection
            index = torch.tensor(random.randint(0, len(self.target_bandwidths)-1), device=x.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(index, src=0)
            bw = self.target_bandwidths[index.item()]
            
            # Extract codebook 0 indices for inter-channel consistency
            # We'll extract from full_codebook_indices after the main forward pass
            # to avoid redundant forward passes
            inter_channel_cb0_indices = None
            # Will be set after we get full_codebook_indices from main forward pass
            
            # Compute original codebook indices for augmentation consistency (before perturbation)
            # Use the same bandwidth as perturbed audio
            if need_original and return_slice_consistency and self.use_slice_consistency:
                # In training mode, encode() returns embeddings; in eval mode, it returns codes
                # We need embeddings here, so get them directly from the encoder
                if self.training:
                    orig_frames = self.encode(orig_audio)
                    for emb_orig, scale_orig in orig_frames:
                        qv_orig = self.quantizer(emb_orig, self.frame_rate, bw)
                        if original_codebook_indices is None:
                            original_codebook_indices = qv_orig.codes  # [K, B, T]
                        break  # Only need first frame
                else:
                    # In eval mode, encode() returns codes, not embeddings
                    # Get embeddings directly from encoder to avoid dimension mismatch
                    # Handle segmentation like encode() does
                    _, channels, length = orig_audio.shape
                    segment_length = self.segment_length
                    if segment_length is None:
                        segment_length = length
                        stride = length
                    else:
                        stride = self.segment_stride
                        assert stride is not None
                    
                    # Get first frame only
                    offset = 0
                    frame = orig_audio[:, :, offset: offset + segment_length]
                    # Handle normalization
                    if self.normalize:
                        mono = frame.mean(dim=1, keepdim=True)
                        volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                        scale = 1e-8 + volume
                        frame = frame / scale
                        scale = scale.view(-1, 1)
                    else:
                        scale = None
                    # Get embeddings from encoder
                    emb_orig = self.encoder(frame)  # [B, D, T]
                    qv_orig = self.quantizer(emb_orig, self.frame_rate, bw)
                    if original_codebook_indices is None:
                        original_codebook_indices = qv_orig.codes  # [K, B, T]
            
            # Encode full audio
            full_features = {}
            full_codebook_indices = None
            full_sub_quants = []
            
            for emb, scale in frames:
                # Get encoder output (quant_in) - store for first frame only for consistency
                if "quant_in" not in full_features:
                    full_features["quant_in"] = emb  # [B, D, T]
                
                # Normal quantization (all codebooks including 0)
                qv = self.quantizer(emb, self.frame_rate, bw)
                loss_w = loss_w + qv.penalty
                codes.append((qv.quantized, scale))
                
                # Get quantized output (quant_out) - store for first frame only
                if "quant_out" not in full_features:
                    full_features["quant_out"] = qv.quantized  # [B, D, T]
                
                # Get codebook indices - store for first frame only
                # qv.codes is [K, B, T] where K is number of quantizers used
                if full_codebook_indices is None:
                    full_codebook_indices = qv.codes
                
                # Get individual quantizer outputs (sub_quants) for first frame only
                if return_slice_consistency and self.use_slice_consistency and len(full_sub_quants) == 0:
                    residual = emb
                    frame_sub_quants = []
                    n_q_used = min(len(self.quantizer.vq.layers), qv.codes.shape[0])
                    for i in range(n_q_used):
                        layer = self.quantizer.vq.layers[i]
                        quantized, indices, layer_loss = layer(residual)
                        frame_sub_quants.append(quantized)
                        residual = residual - quantized.detach()
                    if frame_sub_quants:
                        full_sub_quants.append(torch.stack(frame_sub_quants))  # [K, B*C, D, T] (after reshape)
            
            # Decode full audio
            output = self.decode(codes)[:, :, :x_reshaped.shape[-1]]
            
            # Reshape output back to [B, C, T] if multi-channel
            if is_multi_channel:
                output = output.reshape(B, C, output.shape[-1])  # [B, C, T]
                
                # Extract inter-channel codebook 0 indices from full_codebook_indices
                # full_codebook_indices is [K, B*C, T_feat] where:
                # - First B samples (0 to B-1) are channel 0
                # - Next B samples (B to 2B-1) are channel 1
                if self.use_inter_channel_consistency and C == 2 and full_codebook_indices is not None:
                    # Extract codebook 0 (first codebook) indices
                    cb0_indices = full_codebook_indices[0]  # [B*C, T_feat]
                    ch0_cb0_indices = cb0_indices[:B, :]  # [B, T_feat] - channel 0
                    ch1_cb0_indices = cb0_indices[B:, :]  # [B, T_feat] - channel 1
                    inter_channel_cb0_indices = (ch0_cb0_indices, ch1_cb0_indices)
            
            # Compute slice consistency if requested (allow in eval mode for validation metrics)
            slice_consistency_output = None
            if return_slice_consistency and self.use_slice_consistency:
                # Get feature lengths (time dimension)
                # emb shape is [B, D, T], so T is at dimension 2
                if len(frames) > 0:
                    emb, _ = frames[0]
                    batch_size = emb.shape[0]
                    feature_lengths = torch.tensor([emb.shape[2]] * batch_size, device=emb.device, dtype=torch.long)
                    
                    # CRITICAL FIX: Sample slice interval in FEATURE SPACE first (like ConsistencyInNeuralCodec)
                    # This ensures alignment between feature positions
                    ds_rate = self.slice_consistency.ds_rate  # Downsampling rate (320)
                    tokens_per_second = int(self.sample_rate / ds_rate)
                    
                    # Sample split intervals in feature space
                    split_interval_lengths = (feature_lengths * self.slice_consistency.split_interval_percentage).floor().long()
                    split_interval_lengths = torch.where(
                        split_interval_lengths == 0, 
                        feature_lengths, 
                        split_interval_lengths
                    )
                    
                    # Deterministic start positions in feature space (for validation) or random (for training)
                    max_start_feature = feature_lengths - split_interval_lengths
                    max_start_feature = torch.clamp(max_start_feature, min=0)
                    if batch_idx is not None and not self.training:
                        # Deterministic mode: use batch_idx and sample indices to create deterministic positions
                        generator = torch.Generator(device=emb.device)
                        start_positions_feature = torch.zeros(batch_size, device=emb.device, dtype=torch.long)
                        for b in range(batch_size):
                            # Create deterministic seed from batch_idx and sample index
                            seed = (batch_idx * 1000 + b) % (2**31)
                            generator.manual_seed(seed)
                            start_pos = (torch.rand(1, generator=generator, device=emb.device) * max_start_feature[b].float()).long().item()
                            start_positions_feature[b] = start_pos
                    else:
                        # Random mode (training)
                        start_positions_feature = (torch.rand(batch_size, device=emb.device) * max_start_feature.float()).long()
                    end_positions_feature = torch.clamp(start_positions_feature + split_interval_lengths, max=feature_lengths)
                    
                    # Convert feature positions to audio positions
                    start_positions_audio = start_positions_feature * ds_rate
                    end_positions_audio = end_positions_feature * ds_rate
                    # For separate channel processing, each channel has its own T samples
                    # batch_size here is B*C (from reshaped [B*C, 1, T])
                    # We need to extract slices from original x [B, C, T] format
                    # and ensure slices stay within each channel's boundary (0-T for each channel)
                    audio_length = T  # Each channel has T samples
                    end_positions_audio = torch.clamp(end_positions_audio, max=audio_length)
                    
                    # Extract the SAME slice from both original and perturbed full audio
                    # This aligns latent representations from slice-consistency and perturbation-consistency methods
                    # IMPORTANT: For separate channel processing [B*C, 1, T], we extract slices
                    # from the original x [B, C, T] format, ensuring slices stay within [0, T] for each channel
                    
                    # batch_size is B*C, so we need to map back to [B, C, T] format
                    # For each sample in batch, extract slice from its corresponding channel
                    if orig_audio is not None:
                        # orig_audio is [B*C, 1, T] - reshape back to [B, C, T] for slice extraction
                        orig_audio_reshaped = orig_audio.reshape(B, C, T)  # [B, C, T]
                    else:
                        orig_audio_reshaped = x  # [B, C, T]
                    
                    # x_reshaped is [B*C, 1, T] - reshape back to [B, C, T] for slice extraction
                    x_reshaped_back = x_reshaped.reshape(B, C, T)  # [B, C, T]
                    
                    # Extract slice from original full audio [B, C, T]
                    slice_from_original_list = []
                    for bc_idx in range(batch_size):
                        # Map bc_idx back to (b, c)
                        b = bc_idx // C
                        c = bc_idx % C
                        start_pos = start_positions_audio[bc_idx].item()
                        end_pos = end_positions_audio[bc_idx].item()
                        # Extract slice from channel c of batch b
                        slice_from_original_list.append(orig_audio_reshaped[b:b+1, c:c+1, start_pos:end_pos])
                    slice_from_original = torch.cat(slice_from_original_list, dim=0)  # [B*C, 1, T_slice]
                    
                    # Extract the SAME slice from perturbed full audio
                    slice_from_perturbed_list = []
                    for bc_idx in range(batch_size):
                        # Map bc_idx back to (b, c)
                        b = bc_idx // C
                        c = bc_idx % C
                        start_pos = start_positions_audio[bc_idx].item()
                        end_pos = end_positions_audio[bc_idx].item()
                        # Extract slice from channel c of batch b
                        slice_from_perturbed_list.append(x_reshaped_back[b:b+1, c:c+1, start_pos:end_pos])
                    slice_from_perturbed = torch.cat(slice_from_perturbed_list, dim=0)  # [B*C, 1, T_slice]
                    
                    # Encode both slices
                    slice_original_frames = self.encode(slice_from_original)
                    slice_perturbed_frames = self.encode(slice_from_perturbed)
                    
                    # Get features from both slices
                    slice_original_features = {}
                    slice_perturbed_features = {}
                    slice_original_codebook_indices = []
                    slice_perturbed_codebook_indices = []
                    slice_original_sub_quants = []
                    slice_perturbed_sub_quants = []
                    
                    for (emb_slice_orig, scale_slice_orig), (emb_slice_pert, scale_slice_pert) in zip(slice_original_frames, slice_perturbed_frames):
                        slice_original_features["quant_in"] = emb_slice_orig  # [B, D, T_slice]
                        slice_perturbed_features["quant_in"] = emb_slice_pert  # [B, D, T_slice]
                        
                        # Quantize both slices with same bandwidth
                        qv_slice_orig = self.quantizer(emb_slice_orig, self.frame_rate, bw)
                        qv_slice_pert = self.quantizer(emb_slice_pert, self.frame_rate, bw)
                        
                        slice_original_features["quant_out"] = qv_slice_orig.quantized  # [B, D, T_slice]
                        slice_perturbed_features["quant_out"] = qv_slice_pert.quantized  # [B, D, T_slice]
                        
                        slice_original_codebook_indices.append(qv_slice_orig.codes)  # [K, B, T_slice]
                        slice_perturbed_codebook_indices.append(qv_slice_pert.codes)  # [K, B, T_slice]
                        
                        # Get sub_quants for both slices
                        for emb_slice, qv_slice, sub_quants_list in [
                            (emb_slice_orig, qv_slice_orig, slice_original_sub_quants),
                            (emb_slice_pert, qv_slice_pert, slice_perturbed_sub_quants)
                        ]:
                            residual = emb_slice
                            frame_sub_quants = []
                            n_q_used = min(len(self.quantizer.vq.layers), qv_slice.codes.shape[0])
                            for i in range(n_q_used):
                                layer = self.quantizer.vq.layers[i]
                                quantized, indices, layer_loss = layer(residual)
                                frame_sub_quants.append(quantized)
                                residual = residual - quantized.detach()
                            if frame_sub_quants:
                                sub_quants_list.append(torch.stack(frame_sub_quants))  # [K, B, D, T_slice]
                    
                    # Convert features to [B, T, D] format for consistency computation
                    # Slice from original audio features
                    slice_orig_quant_in = slice_original_features["quant_in"].transpose(1, 2)  # [B, T_slice, D]
                    slice_orig_quant_out = slice_original_features["quant_out"].transpose(1, 2)  # [B, T_slice, D]
                    
                    # Slice from perturbed audio features
                    slice_pert_quant_in = slice_perturbed_features["quant_in"].transpose(1, 2)  # [B, T_slice, D]
                    slice_pert_quant_out = slice_perturbed_features["quant_out"].transpose(1, 2)  # [B, T_slice, D]
                    
                    # Handle sub_quants - convert from [K, B, D, T] to [K, B, T, D]
                    slice_orig_sub_quants_t = None
                    if slice_original_sub_quants and len(slice_original_sub_quants) > 0:
                        slice_orig_sub_quants_t = slice_original_sub_quants[0].transpose(2, 3)  # [K, B, T_slice, D]
                    
                    slice_pert_sub_quants_t = None
                    if slice_perturbed_sub_quants and len(slice_perturbed_sub_quants) > 0:
                        slice_pert_sub_quants_t = slice_perturbed_sub_quants[0].transpose(2, 3)  # [K, B, T_slice, D]
                    
                    # Prepare features dictionary for loss computation
                    # Compare slice from original vs slice from perturbed (same slice positions)
                    slice_original_features_dict = {
                        "quant_in": slice_orig_quant_in,
                        "quant_out": slice_orig_quant_out,
                    }
                    if slice_orig_sub_quants_t is not None:
                        slice_original_features_dict["sub_quants"] = slice_orig_sub_quants_t
                    
                    slice_perturbed_features_dict = {
                        "quant_in": slice_pert_quant_in,
                        "quant_out": slice_pert_quant_out,
                    }
                    if slice_pert_sub_quants_t is not None:
                        slice_perturbed_features_dict["sub_quants"] = slice_pert_sub_quants_t
                    
                    # Create slice interval for consistency computation
                    # Both slices are the same length, so we can use a simple interval
                    from consistency_0 import SliceInterval
                    slice_feature_lengths = torch.tensor([slice_orig_quant_in.shape[1]] * batch_size, device=emb.device, dtype=torch.long)
                    slice_interval = SliceInterval(
                        start_point=start_positions_audio,
                        end_point=end_positions_audio,
                        start_split=torch.zeros(batch_size, device=emb.device, dtype=torch.long),
                        end_split=slice_feature_lengths,
                        split_interval_lengths=slice_feature_lengths,
                    )
                    
                    # Compute consistency losses using codebook 0 only
                    # Initialize output dictionary
                    slice_consistency_output = {
                        'loss': torch.tensor(0.0, device=emb.device, requires_grad=True),
                        'loss_dict': {},
                    }
                    
                    # Extract first codebook quantized embeddings (Q1) for constraint losses
                    # NOTE: We compare the quantized VECTORS (embeddings), NOT the codebook indices
                    # The quantized vectors use straight-through estimator, so gradients flow back to encoder
                    # 1. Original full audio Q1
                    original_full_q1 = None
                    if orig_audio is not None and original_codebook_indices is not None:
                        # Get original full audio first codebook embeddings
                        if self.training:
                            orig_frames = self.encode(orig_audio)
                        else:
                            # In eval mode, get embeddings directly from encoder
                            _, channels, length = orig_audio.shape
                            segment_length = self.segment_length
                            if segment_length is None:
                                segment_length = length
                                stride = length
                            else:
                                stride = self.segment_stride
                                assert stride is not None
                            
                            offset = 0
                            frame = orig_audio[:, :, offset: offset + segment_length]
                            if self.normalize:
                                mono = frame.mean(dim=1, keepdim=True)
                                volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                                scale = 1e-8 + volume
                                frame = frame / scale
                                scale = scale.view(-1, 1)
                            else:
                                scale = None
                            emb_orig = self.encoder(frame)
                            orig_frames = [(emb_orig, scale)]
                        
                        for emb_orig, scale_orig in orig_frames:
                            residual = emb_orig
                            # Get first codebook only
                            if len(self.quantizer.vq.layers) > 0:
                                layer = self.quantizer.vq.layers[0]
                                quantized, indices, layer_loss = layer(residual)
                                original_full_q1 = quantized  # [B, D, T]
                            break
                    
                    # 2. Perturbed full audio Q1 (from full_sub_quants)
                    perturbed_full_q1 = None
                    if len(full_sub_quants) > 0:
                        full_sub_quants_t = full_sub_quants[0]  # [K, B, D, T]
                        if full_sub_quants_t.shape[0] > 0:
                            perturbed_full_q1 = full_sub_quants_t[0]  # [B, D, T] - first codebook
                    
                    # 3. Original slice Q1 (from slice_original_sub_quants)
                    original_slice_q1 = None
                    if slice_original_sub_quants and len(slice_original_sub_quants) > 0:
                        slice_orig_sub_quants_t = slice_original_sub_quants[0]  # [K, B, D, T_slice]
                        if slice_orig_sub_quants_t.shape[0] > 0:
                            original_slice_q1 = slice_orig_sub_quants_t[0]  # [B, D, T_slice] - first codebook
                    
                    # Compute augmentation constraint loss: original full vs perturbed full
                    loss_augmentation_constraint = None
                    if original_full_q1 is not None and perturbed_full_q1 is not None:
                        loss_augmentation_constraint_raw = self.slice_consistency.compute_augmentation_constraint_loss(
                            original_first_codebook=original_full_q1,
                            perturbed_first_codebook=perturbed_full_q1,
                        )
                        # Apply weight
                        loss_augmentation_constraint = loss_augmentation_constraint_raw * self.augmentation_constraint_loss_weight
                        slice_consistency_output['loss'] = slice_consistency_output['loss'] + loss_augmentation_constraint
                        slice_consistency_output['loss_dict']['augmentation_constraint_loss'] = loss_augmentation_constraint
                    
                    # Compute slice consistency constraint loss: original full vs original slice
                    loss_slice_consistency_constraint = None
                    if original_full_q1 is not None and original_slice_q1 is not None:
                        # Create slice interval for original full vs original slice
                        from consistency_0 import SliceInterval
                        orig_slice_interval = SliceInterval(
                            start_point=start_positions_audio,
                            end_point=end_positions_audio,
                            start_split=start_positions_feature,
                            end_split=end_positions_feature,
                            split_interval_lengths=split_interval_lengths,
                        )
                        loss_slice_consistency_constraint_raw = self.slice_consistency.compute_slice_consistency_constraint_loss(
                            full_first_codebook=original_full_q1,
                            slice_first_codebook=original_slice_q1,
                            slice_interval=orig_slice_interval,
                        )
                        # Apply weight
                        loss_slice_consistency_constraint = loss_slice_consistency_constraint_raw * self.slice_consistency_constraint_loss_weight
                        slice_consistency_output['loss'] = slice_consistency_output['loss'] + loss_slice_consistency_constraint
                        slice_consistency_output['loss_dict']['slice_consistency_constraint_loss'] = loss_slice_consistency_constraint
                    
                    # Compute metrics for logging (original full vs original slice)
                    if original_codebook_indices is not None and orig_audio is not None:
                        # Get full original audio features (need to encode original audio)
                        # In eval mode, encode() returns codes, not embeddings - get embeddings directly
                        if self.training:
                            orig_frames = self.encode(orig_audio)
                        else:
                            # In eval mode, get embeddings directly from encoder
                            # Handle segmentation like encode() does
                            _, channels, length = orig_audio.shape
                            segment_length = self.segment_length
                            if segment_length is None:
                                segment_length = length
                                stride = length
                            else:
                                stride = self.segment_stride
                                assert stride is not None
                            
                            # Get first frame only
                            offset = 0
                            frame = orig_audio[:, :, offset: offset + segment_length]
                            # Handle normalization
                            if self.normalize:
                                mono = frame.mean(dim=1, keepdim=True)
                                volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                                scale = 1e-8 + volume
                                frame = frame / scale
                                scale = scale.view(-1, 1)
                            else:
                                scale = None
                            # Get embeddings from encoder
                            emb_orig = self.encoder(frame)  # [B, D, T]
                            orig_frames = [(emb_orig, scale)]
                        
                        orig_full_features = {}
                        orig_full_codebook_indices = None
                        for emb_orig, scale_orig in orig_frames:
                            if "quant_in" not in orig_full_features:
                                orig_full_features["quant_in"] = emb_orig  # [B, D, T]
                            qv_orig = self.quantizer(emb_orig, self.frame_rate, bw)
                            if "quant_out" not in orig_full_features:
                                orig_full_features["quant_out"] = qv_orig.quantized
                            if orig_full_codebook_indices is None:
                                orig_full_codebook_indices = qv_orig.codes
                            break  # Only need first frame
                        
                        # Convert to [B, T, D] format
                        orig_full_quant_in = orig_full_features["quant_in"].transpose(1, 2)  # [B, T, D]
                        orig_full_quant_out = orig_full_features["quant_out"].transpose(1, 2)  # [B, T, D]
                        
                        orig_full_features_dict = {
                            "quant_in": orig_full_quant_in,
                            "quant_out": orig_full_quant_out,
                        }
                        
                        # Create slice interval for original full vs original slice
                        from consistency_0 import SliceInterval
                        orig_slice_interval = SliceInterval(
                            start_point=start_positions_audio,
                            end_point=end_positions_audio,
                            start_split=start_positions_feature,
                            end_split=end_positions_feature,
                            split_interval_lengths=split_interval_lengths,
                        )
                        
                        # Compute slice consistency metric (original full vs original slice) - CODEBOOK 0 ONLY
                        # Extract only first codebook indices [1, B, T] from [K, B, T]
                        orig_full_cb0_indices = orig_full_codebook_indices[0:1] if orig_full_codebook_indices is not None else None  # [1, B, T]
                        slice_orig_cb0_indices = slice_original_codebook_indices[0][0:1] if slice_original_codebook_indices and len(slice_original_codebook_indices) > 0 else None  # [1, B, T_slice]
                        
                        slice_consistency_metric = self.slice_consistency.compute_consistency_loss(
                            full_features=orig_full_features_dict,
                            slice_features=slice_original_features_dict,
                            feature_lengths=feature_lengths,
                            codebook_indices=orig_full_cb0_indices,  # Only codebook 0
                            slice_codebook_indices=slice_orig_cb0_indices,  # Only codebook 0
                            slice_interval=orig_slice_interval,
                        )
                        
                        # Store only codebook 0 accuracy metrics
                        if 'first_codebook_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_codebook0_accuracy'] = slice_consistency_metric['first_codebook_accuracy']
                        elif 'codebook0_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_codebook0_accuracy'] = slice_consistency_metric['codebook0_accuracy']
                    
                    # Compute augmentation consistency (original full vs perturbed full) - CODEBOOK 0 ONLY
                    if original_codebook_indices is not None and full_codebook_indices is not None:
                        # Extract only first codebook indices [1, B, T] from [K, B, T]
                        orig_cb0_indices = original_codebook_indices[0:1]  # [1, B, T]
                        pert_cb0_indices = full_codebook_indices[0:1]  # [1, B, T]
                        
                        augmentation_consistency = self.slice_consistency.compute_augmentation_consistency(
                            original_codebook_indices=orig_cb0_indices,  # Only codebook 0
                            perturbed_codebook_indices=pert_cb0_indices,  # Only codebook 0
                        )
                        # Store only codebook 0 accuracy
                        if 'augmentation_consistency_codebook0_accuracy' in augmentation_consistency:
                            slice_consistency_output['augmentation_consistency_codebook0_accuracy'] = augmentation_consistency['augmentation_consistency_codebook0_accuracy']
                    
                    # Compute inter-channel consistency loss (codebook 0 only)
                    if is_multi_channel and self.use_inter_channel_consistency and C == 2 and inter_channel_cb0_indices is not None:
                        ch0_cb0_indices, ch1_cb0_indices = inter_channel_cb0_indices
                        ch0_cb0 = ch0_cb0_indices.unsqueeze(0)  # [1, B, T]
                        ch1_cb0 = ch1_cb0_indices.unsqueeze(0)  # [1, B, T]
                        
                        inter_channel_acc = self.slice_consistency.compute_inter_channel_consistency(
                            channel_a_codebook_indices=ch0_cb0,
                            channel_b_codebook_indices=ch1_cb0,
                        )
                        
                        # Compute loss using indices matching (1 - accuracy)
                        matches = (ch0_cb0_indices == ch1_cb0_indices).float()  # [B, T]
                        inter_channel_loss = 1.0 - matches.mean()  # MSE-like loss (1 - accuracy)
                        
                        inter_channel_weight = self.inter_channel_consistency.get('loss_weight', 15.0)
                        weighted_inter_channel_loss = inter_channel_loss * inter_channel_weight
                        
                        # Add to slice_consistency_output
                        if slice_consistency_output is None:
                            slice_consistency_output = {
                                'loss': torch.tensor(0.0, device=output.device, requires_grad=True),
                                'loss_dict': {},
                            }
                        slice_consistency_output['loss'] = slice_consistency_output['loss'] + weighted_inter_channel_loss
                        slice_consistency_output['loss_dict']['inter_channel_consistency_loss'] = weighted_inter_channel_loss
                        
                        # Store accuracy
                        if 'inter_channel_consistency_codebook0_accuracy' in inter_channel_acc:
                            slice_consistency_output['inter_channel_consistency_codebook0_accuracy'] = inter_channel_acc['inter_channel_consistency_codebook0_accuracy']
                        else:
                            slice_consistency_output['inter_channel_consistency_codebook0_accuracy'] = matches.mean().item()
                    
                    # Legacy inter-channel consistency (using full_codebook_indices - may not work if codebook 0 is skipped)
                    if is_multi_channel and self.use_inter_channel_consistency and full_codebook_indices is not None and inter_channel_cb0_indices is None:
                        # full_codebook_indices is [K, B*C, T] where B*C is batch*channels
                        K, BC, T = full_codebook_indices.shape
                        C = x.shape[1]  
                        B = BC // C  
                        
                        # Reshape to [K, B, C, T]
                        full_codebook_indices_reshaped = full_codebook_indices.reshape(K, B, C, T)
                        
                        # Get codebook index to compare (default: 0)
                        codebook_idx = self.inter_channel_consistency.get('codebook_index', 0)
                        if codebook_idx < K:
                            # Extract codebook 0 indices for all channels [B, C, T]
                            cb0_indices = full_codebook_indices_reshaped[codebook_idx]  # [B, C, T]
                            
                            # Compare each pair of channels for each batch item
                            inter_channel_accuracies_list = []
                            for b in range(B):
                                # Get all channels for this batch item [C, T]
                                batch_channels = cb0_indices[b]  # [C, T]
                                
                                # Compare each pair of channels
                                for c1 in range(C):
                                    for c2 in range(c1 + 1, C):
                                        ch1_indices = batch_channels[c1:c1+1]  # [1, T]
                                        ch2_indices = batch_channels[c2:c2+1]  # [1, T]
                                        
                                        # Compute consistency accuracy between these two channels
                                        # Convert to [1, 1, T] format for the function
                                        ch1_cb = ch1_indices.unsqueeze(0)  # [1, 1, T]
                                        ch2_cb = ch2_indices.unsqueeze(0)  # [1, 1, T]
                                        
                                        inter_channel_acc = self.slice_consistency.compute_inter_channel_consistency(
                                            channel_a_codebook_indices=ch1_cb,  # [1, 1, T]
                                            channel_b_codebook_indices=ch2_cb,  # [1, 1, T]
                                        )
                                        
                                        # Store codebook 0 accuracy
                                        if 'inter_channel_consistency_codebook0_accuracy' in inter_channel_acc:
                                            inter_channel_accuracies_list.append(inter_channel_acc['inter_channel_consistency_codebook0_accuracy'])
                            
                            # Average inter-channel accuracies across all pairs
                            if inter_channel_accuracies_list:
                                avg_inter_channel_acc = sum(inter_channel_accuracies_list) / len(inter_channel_accuracies_list)
                                slice_consistency_output['inter_channel_consistency_codebook0_accuracy'] = avg_inter_channel_acc
                    
                    # Compute inter-channel consistency loss if multi-channel and enabled
                    if is_multi_channel and self.use_inter_channel_consistency and full_sub_quants:
                        # Get first codebook embeddings for all channels
                        # full_sub_quants[0] is [K, B*C, D, T] after reshape
                        # We need to reshape back to [K, B, C, D, T] to separate channels
                        full_sub_quants_t = full_sub_quants[0]  # [K, B*C, D, T]
                        K, BC, D, T_feat = full_sub_quants_t.shape
                        
                        # Reshape to [K, B, C, D, T]
                        full_sub_quants_reshaped = full_sub_quants_t.reshape(K, B, C, D, T_feat)
                        
                        # Get codebook 0 embeddings [B, C, D, T]
                        codebook_idx = self.inter_channel_consistency.get('codebook_index', 0)
                        if codebook_idx < K:
                            channel_codebooks = full_sub_quants_reshaped[codebook_idx]  # [B, C, D, T]
                            
                            # Compute inter-channel consistency between all pairs of channels
                            # For each batch item, compare channel 0 with channel 1, channel 0 with channel 2, etc.
                            inter_channel_losses = []
                            for b in range(B):
                                # Get all channels for this batch item [C, D, T]
                                batch_channels = channel_codebooks[b]  # [C, D, T]
                                
                                # Compare each pair of channels
                                for c1 in range(C):
                                    for c2 in range(c1 + 1, C):
                                        ch1_codebook = batch_channels[c1]  # [D, T]
                                        ch2_codebook = batch_channels[c2]  # [D, T]
                                        
                                        # Compute consistency loss between these two channels
                                        loss_inter = self.slice_consistency.compute_inter_channel_consistency_loss(
                                            channel_a_codebook=ch1_codebook.unsqueeze(0),  # [1, D, T]
                                            channel_b_codebook=ch2_codebook.unsqueeze(0),  # [1, D, T]
                                            codebook_index=codebook_idx,
                                            mse_loss_reduction=self.inter_channel_consistency.get('mse_loss_reduction', 'mean'),
                                        )
                                        inter_channel_losses.append(loss_inter)
                            
                            # Average inter-channel losses and weight them
                            if inter_channel_losses:
                                inter_channel_loss = sum(inter_channel_losses) / len(inter_channel_losses)
                                inter_channel_weight = self.inter_channel_consistency.get('loss_weight', 10.0)
                                weighted_inter_channel_loss = inter_channel_loss * inter_channel_weight
                                
                                # Add to slice_consistency_output
                                if slice_consistency_output is None:
                                    slice_consistency_output = {
                                        'loss': torch.tensor(0.0, device=output.device, requires_grad=True),
                                        'loss_dict': {},
                                    }
                                slice_consistency_output['loss'] = slice_consistency_output['loss'] + weighted_inter_channel_loss
                                slice_consistency_output['loss_dict']['inter_channel_consistency_loss'] = weighted_inter_channel_loss
            
            if return_embeddings:
                return output, loss_w, frames, quantized_embeddings, slice_consistency_output
            elif return_slice_consistency:
                return output, loss_w, frames, slice_consistency_output
            else:
                return output, loss_w, frames
        else:
            # Evaluation mode - can compute slice consistency for validation metrics
            # Store original audio if we need it for slice extraction (when augmentations are enabled)
            need_original_for_slice = (
                return_slice_consistency and 
                self.use_slice_consistency and 
                self.perturb_encoder is not None and 
                self.perturb_encoder.perturb_all_audio
            )
            orig_audio = x_reshaped.clone() if need_original_for_slice else None
            
            # Apply perturbation to full audio if enabled (even in eval mode for validation metrics)
            if self.perturb_encoder is not None and self.perturb_encoder.perturb_all_audio:
                # Only create sample indices for deterministic behavior (validation), skip in training for speed
                if batch_idx is not None:
                    batch_size = x_reshaped.shape[0]
                    sample_indices = torch.arange(batch_size, device=x_reshaped.device)
                    x_reshaped = self.perturb_encoder(x_reshaped, batch_idx=batch_idx, sample_indices=sample_indices)
                else:
                    # Training mode: use simple call without extra parameters (faster)
                    x_reshaped = self.perturb_encoder(x_reshaped)
            
            # In eval mode, encode() returns codes, not embeddings
            # We need to get embeddings directly from the encoder
            if self.training:
                frames = self.encode(x_reshaped)
            else:
                # In eval mode, get embeddings directly from encoder
                # Handle segmentation like encode() does
                _, channels, length = x_reshaped.shape
                segment_length = self.segment_length
                if segment_length is None:
                    segment_length = length
                    stride = length
                else:
                    stride = self.segment_stride
                    assert stride is not None
                
                frames = []
                for offset in range(0, length, stride):
                    frame = x_reshaped[:, :, offset: offset + segment_length]
                    # Handle normalization
                    if self.normalize:
                        mono = frame.mean(dim=1, keepdim=True)
                        volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                        scale = 1e-8 + volume
                        frame = frame / scale
                        scale = scale.view(-1, 1)
                    else:
                        scale = None
                    # Get embeddings from encoder
                    emb = self.encoder(frame)  # [B, D, T]
                    frames.append((emb, scale))
            
            loss_w = torch.tensor([0.0], device=x_reshaped.device, requires_grad=False)
            codes = []
            quantized_embeddings = []
            
            # Use the set bandwidth (from model.bandwidth) or default to last bandwidth
            bw = getattr(self, 'bandwidth', self.target_bandwidths[-1])
            
            # Compute original codebook indices for augmentation consistency (before perturbation)
            # Use the same bandwidth as perturbed audio
            original_codebook_indices = None
            if need_original_for_slice and return_slice_consistency and self.use_slice_consistency:
                # In training mode, encode() returns embeddings; in eval mode, it returns codes
                # We need embeddings here, so get them directly from the encoder
                if self.training:
                    orig_frames = self.encode(orig_audio)
                    for emb_orig, scale_orig in orig_frames:
                        qv_orig = self.quantizer(emb_orig, self.frame_rate, bw)
                        if original_codebook_indices is None:
                            original_codebook_indices = qv_orig.codes  # [K, B, T]
                        break  # Only need first frame
                else:
                    # In eval mode, get embeddings directly from encoder
                    # Handle segmentation like encode() does
                    _, channels, length = orig_audio.shape
                    segment_length = self.segment_length
                    if segment_length is None:
                        segment_length = length
                        stride = length
                    else:
                        stride = self.segment_stride
                        assert stride is not None
                    
                    # Get first frame only
                    offset = 0
                    frame = orig_audio[:, :, offset: offset + segment_length]
                    # Handle normalization
                    if self.normalize:
                        mono = frame.mean(dim=1, keepdim=True)
                        volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                        scale = 1e-8 + volume
                        frame = frame / scale
                        scale = scale.view(-1, 1)
                    else:
                        scale = None
                    # Get embeddings from encoder
                    emb_orig = self.encoder(frame)  # [B, D, T]
                    qv_orig = self.quantizer(emb_orig, self.frame_rate, bw)
                    if original_codebook_indices is None:
                        original_codebook_indices = qv_orig.codes  # [K, B, T]
            
            # Encode full audio
            full_features = {}
            full_codebook_indices = None
            full_sub_quants = []
            
            for emb, scale in frames:
                # Quantize
                qv = self.quantizer(emb, self.frame_rate, bw)
                loss_w = loss_w + qv.penalty
                # In training mode, decode expects embeddings; in eval mode, it expects codebook indices
                if self.training:
                    codes.append((qv.quantized, scale))
                else:
                    # In eval mode, decode expects [B, K, T] codebook indices (long/int type)
                    codes_transposed = qv.codes.transpose(0, 1)  # [K, B, T] -> [B, K, T]
                    # Ensure codes are long/int type for embedding lookup
                    if codes_transposed.dtype != torch.long:
                        codes_transposed = codes_transposed.long()
                    codes.append((codes_transposed, scale))
                
                # Store features for slice consistency if needed
                if return_slice_consistency and self.use_slice_consistency:
                    if "quant_in" not in full_features:
                        full_features["quant_in"] = emb  # [B, D, T]
                    if "quant_out" not in full_features:
                        full_features["quant_out"] = qv.quantized  # [B, D, T]
                    if full_codebook_indices is None:
                        full_codebook_indices = qv.codes
                    if len(full_sub_quants) == 0:
                        residual = emb
                        frame_sub_quants = []
                        n_q_used = min(len(self.quantizer.vq.layers), qv.codes.shape[0])
                        for i in range(n_q_used):
                            layer = self.quantizer.vq.layers[i]
                            quantized, indices, layer_loss = layer(residual)
                            frame_sub_quants.append(quantized)
                            residual = residual - quantized.detach()
                        if frame_sub_quants:
                            full_sub_quants.append(torch.stack(frame_sub_quants))  # [K, B, D, T]
            
            # Decode full audio
            output = self.decode(codes)[:, :, :x_reshaped.shape[-1]]
            
            # Reshape output back to [B, C, T] if multi-channel
            if is_multi_channel:
                output = output.reshape(B, C, output.shape[-1])  # [B, C, T]
            
            # Compute slice consistency if requested
            slice_consistency_output = None
            if return_slice_consistency and self.use_slice_consistency:
                # Get feature lengths
                if len(frames) > 0:
                    emb, _ = frames[0]
                    batch_size = emb.shape[0]
                    feature_lengths = torch.tensor([emb.shape[2]] * batch_size, device=emb.device, dtype=torch.long)
                    
                    # Sample slice interval in feature space
                    ds_rate = self.slice_consistency.ds_rate
                    tokens_per_second = int(self.sample_rate / ds_rate)
                    
                    split_interval_lengths = (feature_lengths * self.slice_consistency.split_interval_percentage).floor().long()
                    split_interval_lengths = torch.where(
                        split_interval_lengths == 0, 
                        feature_lengths, 
                        split_interval_lengths
                    )
                    
                    # Deterministic start positions in feature space (for validation) or random (for training)
                    max_start_feature = feature_lengths - split_interval_lengths
                    max_start_feature = torch.clamp(max_start_feature, min=0)
                    if batch_idx is not None and not self.training:
                        # Deterministic mode: use batch_idx and sample indices to create deterministic positions
                        generator = torch.Generator(device=emb.device)
                        start_positions_feature = torch.zeros(batch_size, device=emb.device, dtype=torch.long)
                        for b in range(batch_size):
                            # Create deterministic seed from batch_idx and sample index
                            seed = (batch_idx * 1000 + b) % (2**31)
                            generator.manual_seed(seed)
                            start_pos = (torch.rand(1, generator=generator, device=emb.device) * max_start_feature[b].float()).long().item()
                            start_positions_feature[b] = start_pos
                    else:
                        # Random mode (training)
                        start_positions_feature = (torch.rand(batch_size, device=emb.device) * max_start_feature.float()).long()
                    end_positions_feature = torch.clamp(start_positions_feature + split_interval_lengths, max=feature_lengths)
                    
                    # Convert to audio positions
                    start_positions_audio = start_positions_feature * ds_rate
                    end_positions_audio = end_positions_feature * ds_rate
                    # Use original audio length if available, otherwise use perturbed audio length
                    audio_length = orig_audio.shape[2] if orig_audio is not None else x.shape[2]
                    end_positions_audio = torch.clamp(end_positions_audio, max=audio_length)
                    
                    # Extract slice from original audio and encode it standalone
                    # This is Z_slice_original: slice cut from original → encoded standalone
                    slice_from_original_list = []
                    for b in range(batch_size):
                        start_pos = start_positions_audio[b].item()
                        end_pos = end_positions_audio[b].item()
                        # Use original audio if available, otherwise use x (which might be perturbed)
                        source_orig = orig_audio if orig_audio is not None else x
                        max_pos = min(end_pos, source_orig.shape[2])
                        if start_pos < max_pos:
                            slice_from_original_list.append(source_orig[b:b+1, :, start_pos:max_pos])
                        else:
                            slice_from_original_list.append(source_orig[b:b+1, :, start_pos:start_pos+1])
                    slice_from_original = torch.cat(slice_from_original_list, dim=0)  # [B, C, T_slice]
                    
                    # Encode slice from original standalone
                    if self.training:
                        slice_original_frames = self.encode(slice_from_original)
                    else:
                        # In eval mode, get embeddings directly from encoder
                        slice_orig_frame = slice_from_original
                        if self.normalize:
                            mono_orig = slice_orig_frame.mean(dim=1, keepdim=True)
                            volume_orig = mono_orig.pow(2).mean(dim=2, keepdim=True).sqrt()
                            scale_orig = 1e-8 + volume_orig
                            slice_orig_frame = slice_orig_frame / scale_orig
                            scale_orig = scale_orig.view(-1, 1)
                        else:
                            scale_orig = None
                        emb_slice_orig = self.encoder(slice_orig_frame)  # [B, D, T]
                        slice_original_frames = [(emb_slice_orig, scale_orig)]
                    
                    # Extract corresponding slice from already-encoded full perturbed audio
                    # This is Z_slice_from_full_perturbed: full perturbed encoded → extract slice from full encoding
                    # full_features["quant_in"] is [B, D, T] where T is feature length
                    # Extract slice using feature positions (start_positions_feature, end_positions_feature)
                    full_quant_in = full_features["quant_in"]  # [B, D, T]
                    full_quant_out = full_features["quant_out"]  # [B, D, T]
                    
                    # Extract slice from full encoding for each batch
                    slice_perturbed_features = {}
                    for b in range(batch_size):
                        start_feat = start_positions_feature[b].item()
                        end_feat = end_positions_feature[b].item()
                        # Extract from full encoding: [D, T] -> [D, T_slice]
                        if b == 0:
                            slice_perturbed_quant_in = full_quant_in[b:b+1, :, start_feat:end_feat]  # [1, D, T_slice]
                            slice_perturbed_quant_out = full_quant_out[b:b+1, :, start_feat:end_feat]  # [1, D, T_slice]
                        else:
                            slice_perturbed_quant_in = torch.cat([
                                slice_perturbed_quant_in, 
                                full_quant_in[b:b+1, :, start_feat:end_feat]
                            ], dim=0)
                            slice_perturbed_quant_out = torch.cat([
                                slice_perturbed_quant_out,
                                full_quant_out[b:b+1, :, start_feat:end_feat]
                            ], dim=0)
                    
                    # Extract codebook indices from full encoding
                    slice_perturbed_codebook_indices = []
                    if full_codebook_indices is not None:
                        # full_codebook_indices is [K, B, T]
                        for b in range(batch_size):
                            start_feat = start_positions_feature[b].item()
                            end_feat = end_positions_feature[b].item()
                            if b == 0:
                                slice_perturbed_codes = full_codebook_indices[:, b:b+1, start_feat:end_feat]  # [K, 1, T_slice]
                            else:
                                slice_perturbed_codes = torch.cat([
                                    slice_perturbed_codes,
                                    full_codebook_indices[:, b:b+1, start_feat:end_feat]
                                ], dim=1)
                        slice_perturbed_codebook_indices.append(slice_perturbed_codes)  # [K, B, T_slice]
                    
                    # Extract sub_quants from full encoding if available
                    slice_perturbed_sub_quants = []
                    if len(full_sub_quants) > 0:
                        full_sub_quants_t = full_sub_quants[0]  # [K, B, D, T]
                        for b in range(batch_size):
                            start_feat = start_positions_feature[b].item()
                            end_feat = end_positions_feature[b].item()
                            if b == 0:
                                slice_perturbed_sub_quants_t = full_sub_quants_t[:, b:b+1, :, start_feat:end_feat]  # [K, 1, D, T_slice]
                            else:
                                slice_perturbed_sub_quants_t = torch.cat([
                                    slice_perturbed_sub_quants_t,
                                    full_sub_quants_t[:, b:b+1, :, start_feat:end_feat]
                                ], dim=1)
                        slice_perturbed_sub_quants.append(slice_perturbed_sub_quants_t)  # [K, B, D, T_slice]
                    
                    # Get features from original slice (encoded standalone)
                    slice_original_features = {}
                    slice_original_codebook_indices = []
                    slice_original_sub_quants = []
                    
                    for emb_slice_orig, scale_slice_orig in slice_original_frames:
                        slice_original_features["quant_in"] = emb_slice_orig
                        
                        # Quantize original slice with same bandwidth
                        qv_slice_orig = self.quantizer(emb_slice_orig, self.frame_rate, bw)
                        slice_original_features["quant_out"] = qv_slice_orig.quantized
                        slice_original_codebook_indices.append(qv_slice_orig.codes)
                        
                        # Get sub_quants for original slice
                        residual = emb_slice_orig
                        frame_sub_quants = []
                        n_q_used = min(len(self.quantizer.vq.layers), qv_slice_orig.codes.shape[0])
                        for i in range(n_q_used):
                            layer = self.quantizer.vq.layers[i]
                            quantized, indices, layer_loss = layer(residual)
                            frame_sub_quants.append(quantized)
                            residual = residual - quantized.detach()
                        if frame_sub_quants:
                            slice_original_sub_quants.append(torch.stack(frame_sub_quants))
                    
                    # Set perturbed features (already extracted from full encoding)
                    slice_perturbed_features["quant_in"] = slice_perturbed_quant_in  # [B, D, T_slice]
                    slice_perturbed_features["quant_out"] = slice_perturbed_quant_out  # [B, D, T_slice]
                    
                    # Convert features to [B, T, D] format for consistency computation
                    # Slice from original audio features
                    slice_orig_quant_in = slice_original_features["quant_in"].transpose(1, 2)
                    slice_orig_quant_out = slice_original_features["quant_out"].transpose(1, 2)
                    
                    # Slice from perturbed audio features
                    slice_pert_quant_in = slice_perturbed_features["quant_in"].transpose(1, 2)
                    slice_pert_quant_out = slice_perturbed_features["quant_out"].transpose(1, 2)
                    
                    # Handle sub_quants - convert from [K, B, D, T] to [K, B, T, D]
                    slice_orig_sub_quants_t = None
                    if slice_original_sub_quants and len(slice_original_sub_quants) > 0:
                        slice_orig_sub_quants_t = slice_original_sub_quants[0].transpose(2, 3)
                    
                    slice_pert_sub_quants_t = None
                    if slice_perturbed_sub_quants and len(slice_perturbed_sub_quants) > 0:
                        slice_pert_sub_quants_t = slice_perturbed_sub_quants[0].transpose(2, 3)
                    
                    # Prepare features dictionary for loss computation
                    # Compare slice from original vs slice from perturbed (same slice positions)
                    slice_original_features_dict = {
                        "quant_in": slice_orig_quant_in,
                        "quant_out": slice_orig_quant_out,
                    }
                    if slice_orig_sub_quants_t is not None:
                        slice_original_features_dict["sub_quants"] = slice_orig_sub_quants_t
                    
                    slice_perturbed_features_dict = {
                        "quant_in": slice_pert_quant_in,
                        "quant_out": slice_pert_quant_out,
                    }
                    if slice_pert_sub_quants_t is not None:
                        slice_perturbed_features_dict["sub_quants"] = slice_pert_sub_quants_t
                    
                    # Create slice interval for consistency computation
                    # Both slices are the same length, so we can use a simple interval
                    from consistency_0 import SliceInterval
                    slice_feature_lengths = torch.tensor([slice_orig_quant_in.shape[1]] * batch_size, device=emb.device, dtype=torch.long)
                    slice_interval = SliceInterval(
                        start_point=start_positions_audio,
                        end_point=end_positions_audio,
                        start_split=torch.zeros(batch_size, device=emb.device, dtype=torch.long),
                        end_split=slice_feature_lengths,
                        split_interval_lengths=slice_feature_lengths,
                    )
                    
                    # Compute consistency losses using codebook 0 only
                    # Initialize output dictionary
                    slice_consistency_output = {
                        'loss': torch.tensor(0.0, device=emb.device, requires_grad=False),
                        'loss_dict': {},
                    }
                    
                    # Extract first codebook quantized embeddings (Q1) for constraint losses
                    # 1. Original full audio Q1
                    original_full_q1 = None
                    if orig_audio is not None and original_codebook_indices is not None:
                        # Get original full audio first codebook embeddings
                        if self.training:
                            orig_frames = self.encode(orig_audio)
                        else:
                            # In eval mode, get embeddings directly from encoder
                            _, channels, length = orig_audio.shape
                            segment_length = self.segment_length
                            if segment_length is None:
                                segment_length = length
                                stride = length
                            else:
                                stride = self.segment_stride
                                assert stride is not None
                            
                            offset = 0
                            frame = orig_audio[:, :, offset: offset + segment_length]
                            if self.normalize:
                                mono = frame.mean(dim=1, keepdim=True)
                                volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                                scale = 1e-8 + volume
                                frame = frame / scale
                                scale = scale.view(-1, 1)
                            else:
                                scale = None
                            emb_orig = self.encoder(frame)
                            orig_frames = [(emb_orig, scale)]
                        
                        for emb_orig, scale_orig in orig_frames:
                            residual = emb_orig
                            # Get first codebook only
                            if len(self.quantizer.vq.layers) > 0:
                                layer = self.quantizer.vq.layers[0]
                                quantized, indices, layer_loss = layer(residual)
                                original_full_q1 = quantized  # [B, D, T]
                            break
                    
                    # 2. Perturbed full audio Q1 (from full_sub_quants)
                    perturbed_full_q1 = None
                    if len(full_sub_quants) > 0:
                        full_sub_quants_t = full_sub_quants[0]  # [K, B, D, T]
                        if full_sub_quants_t.shape[0] > 0:
                            perturbed_full_q1 = full_sub_quants_t[0]  # [B, D, T] - first codebook
                    
                    # 3. Original slice Q1 (from slice_original_sub_quants)
                    original_slice_q1 = None
                    if slice_original_sub_quants and len(slice_original_sub_quants) > 0:
                        slice_orig_sub_quants_t = slice_original_sub_quants[0]  # [K, B, D, T_slice]
                        if slice_orig_sub_quants_t.shape[0] > 0:
                            original_slice_q1 = slice_orig_sub_quants_t[0]  # [B, D, T_slice] - first codebook
                    
                    # Compute augmentation constraint loss: original full vs perturbed full
                    loss_augmentation_constraint = None
                    if original_full_q1 is not None and perturbed_full_q1 is not None:
                        loss_augmentation_constraint_raw = self.slice_consistency.compute_augmentation_constraint_loss(
                            original_first_codebook=original_full_q1,
                            perturbed_first_codebook=perturbed_full_q1,
                        )
                        # Apply weight
                        loss_augmentation_constraint = loss_augmentation_constraint_raw * self.augmentation_constraint_loss_weight
                        slice_consistency_output['loss'] = slice_consistency_output['loss'] + loss_augmentation_constraint
                        slice_consistency_output['loss_dict']['augmentation_constraint_loss'] = loss_augmentation_constraint
                    
                    # Compute slice consistency constraint loss: original full vs original slice
                    loss_slice_consistency_constraint = None
                    if original_full_q1 is not None and original_slice_q1 is not None:
                        # Create slice interval for original full vs original slice
                        from consistency_0 import SliceInterval
                        orig_slice_interval = SliceInterval(
                            start_point=start_positions_audio,
                            end_point=end_positions_audio,
                            start_split=start_positions_feature,
                            end_split=end_positions_feature,
                            split_interval_lengths=split_interval_lengths,
                        )
                        loss_slice_consistency_constraint_raw = self.slice_consistency.compute_slice_consistency_constraint_loss(
                            full_first_codebook=original_full_q1,
                            slice_first_codebook=original_slice_q1,
                            slice_interval=orig_slice_interval,
                        )
                        # Apply weight
                        loss_slice_consistency_constraint = loss_slice_consistency_constraint_raw * self.slice_consistency_constraint_loss_weight
                        slice_consistency_output['loss'] = slice_consistency_output['loss'] + loss_slice_consistency_constraint
                        slice_consistency_output['loss_dict']['slice_consistency_constraint_loss'] = loss_slice_consistency_constraint
                    
                    # Compute metrics for logging (original full vs original slice)
                    if original_codebook_indices is not None and orig_audio is not None:
                        # Get full original audio features (need to encode original audio)
                        # In eval mode, encode() returns codes, not embeddings - get embeddings directly
                        if self.training:
                            orig_frames = self.encode(orig_audio)
                        else:
                            # In eval mode, get embeddings directly from encoder
                            # Handle segmentation like encode() does
                            _, channels, length = orig_audio.shape
                            segment_length = self.segment_length
                            if segment_length is None:
                                segment_length = length
                                stride = length
                            else:
                                stride = self.segment_stride
                                assert stride is not None
                            
                            # Get first frame only
                            offset = 0
                            frame = orig_audio[:, :, offset: offset + segment_length]
                            # Handle normalization
                            if self.normalize:
                                mono = frame.mean(dim=1, keepdim=True)
                                volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                                scale = 1e-8 + volume
                                frame = frame / scale
                                scale = scale.view(-1, 1)
                            else:
                                scale = None
                            # Get embeddings from encoder
                            emb_orig = self.encoder(frame)  # [B, D, T]
                            orig_frames = [(emb_orig, scale)]
                        
                        orig_full_features = {}
                        orig_full_codebook_indices = None
                        for emb_orig, scale_orig in orig_frames:
                            if "quant_in" not in orig_full_features:
                                orig_full_features["quant_in"] = emb_orig  # [B, D, T]
                            qv_orig = self.quantizer(emb_orig, self.frame_rate, bw)
                            if "quant_out" not in orig_full_features:
                                orig_full_features["quant_out"] = qv_orig.quantized
                            if orig_full_codebook_indices is None:
                                orig_full_codebook_indices = qv_orig.codes
                            break  # Only need first frame
                        
                        # Convert to [B, T, D] format
                        orig_full_quant_in = orig_full_features["quant_in"].transpose(1, 2)  # [B, T, D]
                        orig_full_quant_out = orig_full_features["quant_out"].transpose(1, 2)  # [B, T, D]
                        
                        orig_full_features_dict = {
                            "quant_in": orig_full_quant_in,
                            "quant_out": orig_full_quant_out,
                        }
                        
                        # Create slice interval for original full vs original slice
                        from consistency_0 import SliceInterval
                        orig_slice_interval = SliceInterval(
                            start_point=start_positions_audio,
                            end_point=end_positions_audio,
                            start_split=start_positions_feature,
                            end_split=end_positions_feature,
                            split_interval_lengths=split_interval_lengths,
                        )
                        
                        # Compute slice consistency metric (original full vs original slice) - CODEBOOK 0 ONLY
                        # Extract only first codebook indices [1, B, T] from [K, B, T]
                        orig_full_cb0_indices = orig_full_codebook_indices[0:1] if orig_full_codebook_indices is not None else None  # [1, B, T]
                        slice_orig_cb0_indices = slice_original_codebook_indices[0][0:1] if slice_original_codebook_indices and len(slice_original_codebook_indices) > 0 else None  # [1, B, T_slice]
                        
                        slice_consistency_metric = self.slice_consistency.compute_consistency_loss(
                            full_features=orig_full_features_dict,
                            slice_features=slice_original_features_dict,
                            feature_lengths=feature_lengths,
                            codebook_indices=orig_full_cb0_indices,  # Only codebook 0
                            slice_codebook_indices=slice_orig_cb0_indices,  # Only codebook 0
                            slice_interval=orig_slice_interval,
                        )
                        
                        # Store only codebook 0 accuracy metrics
                        if 'first_codebook_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_codebook0_accuracy'] = slice_consistency_metric['first_codebook_accuracy']
                        elif 'codebook0_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_codebook0_accuracy'] = slice_consistency_metric['codebook0_accuracy']
                    
                    # Compute augmentation consistency (original full vs perturbed full) - CODEBOOK 0 ONLY
                    if original_codebook_indices is not None and full_codebook_indices is not None:
                        # Extract only first codebook indices [1, B, T] from [K, B, T]
                        orig_cb0_indices = original_codebook_indices[0:1]  # [1, B, T]
                        pert_cb0_indices = full_codebook_indices[0:1]  # [1, B, T]
                        
                        augmentation_consistency = self.slice_consistency.compute_augmentation_consistency(
                            original_codebook_indices=orig_cb0_indices,  # Only codebook 0
                            perturbed_codebook_indices=pert_cb0_indices,  # Only codebook 0
                        )
                        # Store only codebook 0 accuracy
                        if 'augmentation_consistency_codebook0_accuracy' in augmentation_consistency:
                            slice_consistency_output['augmentation_consistency_codebook0_accuracy'] = augmentation_consistency['augmentation_consistency_codebook0_accuracy']
                    
                    # Compute inter-channel consistency accuracy if multi-channel and enabled - CODEBOOK 0 ONLY
                    if is_multi_channel and self.use_inter_channel_consistency and full_codebook_indices is not None:
                        # full_codebook_indices is [K, B*C, T] where B*C is batch*channels
                        # We need to reshape to [K, B, C, T] to separate channels
                        K, BC, T = full_codebook_indices.shape
                        # B is the batch size, C is the number of channels
                        # In eval mode, use C from the original input x (available at function start)
                        C = x.shape[1]  # Number of channels from original input
                        B = BC // C  # Batch size
                        
                        # Reshape to [K, B, C, T]
                        full_codebook_indices_reshaped = full_codebook_indices.reshape(K, B, C, T)
                        
                        # Get codebook index to compare (default: 0)
                        codebook_idx = self.inter_channel_consistency.get('codebook_index', 0)
                        if codebook_idx < K:
                            # Extract codebook 0 indices for all channels [B, C, T]
                            cb0_indices = full_codebook_indices_reshaped[codebook_idx]  # [B, C, T]
                            
                            # Compare each pair of channels for each batch item
                            inter_channel_accuracies_list = []
                            for b in range(B):
                                # Get all channels for this batch item [C, T]
                                batch_channels = cb0_indices[b]  # [C, T]
                                
                                # Compare each pair of channels
                                for c1 in range(C):
                                    for c2 in range(c1 + 1, C):
                                        ch1_indices = batch_channels[c1:c1+1]  # [1, T]
                                        ch2_indices = batch_channels[c2:c2+1]  # [1, T]
                                        
                                        # Compute consistency accuracy between these two channels
                                        # Convert to [1, 1, T] format for the function
                                        ch1_cb = ch1_indices.unsqueeze(0)  # [1, 1, T]
                                        ch2_cb = ch2_indices.unsqueeze(0)  # [1, 1, T]
                                        
                                        inter_channel_acc = self.slice_consistency.compute_inter_channel_consistency(
                                            channel_a_codebook_indices=ch1_cb,  # [1, 1, T]
                                            channel_b_codebook_indices=ch2_cb,  # [1, 1, T]
                                        )
                                        
                                        # Store codebook 0 accuracy
                                        if 'inter_channel_consistency_codebook0_accuracy' in inter_channel_acc:
                                            inter_channel_accuracies_list.append(inter_channel_acc['inter_channel_consistency_codebook0_accuracy'])
                            
                            # Average inter-channel accuracies across all pairs
                            if inter_channel_accuracies_list:
                                avg_inter_channel_acc = sum(inter_channel_accuracies_list) / len(inter_channel_accuracies_list)
                                slice_consistency_output['inter_channel_consistency_codebook0_accuracy'] = avg_inter_channel_acc
            
            if return_embeddings:
                return output, loss_w, frames, quantized_embeddings, slice_consistency_output
            elif return_slice_consistency:
                return output, loss_w, frames, slice_consistency_output
            else:
                return output, loss_w, frames
    
    @staticmethod
    def _get_model(
        target_bandwidths: tp.List[float],
        sample_rate: int = 24_000,
        channels: int = 1,
        causal: bool = True,
        model_norm: str = 'weight_norm',
        audio_normalize: bool = False,
        segment: tp.Optional[float] = None,
        name: str = 'unset',
        ratios=[8, 5, 4, 2],
        n_q: tp.Optional[int] = None,
        slice_consistency: tp.Optional[dict] = None,
        perturb_encoder: tp.Optional[dict] = None,
        inter_channel_consistency: tp.Optional[dict] = None,
    ):
        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal, ratios=ratios)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal, ratios=ratios)
        
        if n_q is None:
            n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))
        
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        
        model = EncodecModelWithSliceConsistency(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
            slice_consistency=slice_consistency,
            perturb_encoder=perturb_encoder,
            inter_channel_consistency=inter_channel_consistency,
        )
        return model
