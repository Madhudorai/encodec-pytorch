"""EnCodec model with slice consistency support."""

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
from slice_consistency import SliceConsistency
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
    ):
        super().__init__(
            encoder, decoder, quantizer, target_bandwidths,
            sample_rate, channels, normalize, segment, overlap, name
        )
        
        # Initialize slice consistency module
        if slice_consistency is not None:
            self.slice_consistency = SliceConsistency(**slice_consistency)
            self.use_slice_consistency = True
        else:
            self.slice_consistency = None
            self.use_slice_consistency = False
        
        # Initialize perturb encoder
        if perturb_encoder is not None:
            self.perturb_encoder = PerturbEncoder(**perturb_encoder)
        else:
            self.perturb_encoder = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_embeddings: bool = False,
        return_slice_consistency: bool = False,
        batch_idx: tp.Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass with optional slice consistency computation.
        
        Args:
            x: Input audio tensor [B, C, T]
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
                - slice_consistency_output: Dictionary with slice consistency loss
        """
        # FORWARD PASS FLOW:
        # 1. If slice consistency is enabled: Store original audio for slice extraction
        #    (We extract slice from original audio, then augment it independently)
        # 2. Augment full audio if enabled (for slice consistency on augmented audio)
        # 3. Encode augmented full audio -> get features
        # 4. Extract slice from original (unperturbed) audio
        # 5. Augment slice independently if enabled
        # 6. Encode augmented slice -> get features  
        # 7. Compute slice consistency between full and slice features
        
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
        orig_audio = x.clone() if need_original else None
        
        # Compute original codebook indices for augmentation consistency (before perturbation)
        # We'll compute this after we know the bandwidth (in training/eval mode)
        original_codebook_indices = None
        
        # Apply perturbation to full audio if enabled
        # When slice consistency is enabled, we want to calculate consistency on augmented audio
        # So if perturb_all_audio is True, we always augment the full audio
        # (Individual augmentations have their own apply_prob for probabilistic application)
        if self.perturb_encoder is not None and self.perturb_encoder.perturb_all_audio:
            # Only create sample indices for deterministic behavior (validation), skip in training for speed
            if batch_idx is not None:
                batch_size = x.shape[0]
                sample_indices = torch.arange(batch_size, device=x.device)
                x = self.perturb_encoder(x, batch_idx=batch_idx, sample_indices=sample_indices)
            else:
                # Training mode: use simple call without extra parameters (faster)
                x = self.perturb_encoder(x)
        
        frames = self.encode(x)
        
        if self.training:
            # Standard training forward
            loss_w = torch.tensor([0.0], device=x.device, requires_grad=True)
            codes = []
            quantized_embeddings = []
            
            # Random bandwidth selection
            index = torch.tensor(random.randint(0, len(self.target_bandwidths)-1), device=x.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(index, src=0)
            bw = self.target_bandwidths[index.item()]
            
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
                
                # Quantize
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
                        full_sub_quants.append(torch.stack(frame_sub_quants))  # [K, B, D, T]
            
            # Decode full audio
            output = self.decode(codes)[:, :, :x.shape[-1]]
            
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
                    audio_length = x.shape[2]  # [B, C, T]
                    end_positions_audio = torch.clamp(end_positions_audio, max=audio_length)
                    
                    # Extract the SAME slice from both original and perturbed full audio
                    # This aligns latent representations from slice-consistency and perturbation-consistency methods
                    
                    # Extract slice from original full audio
                    slice_from_original_list = []
                    for b in range(batch_size):
                        start_pos = start_positions_audio[b].item()
                        end_pos = end_positions_audio[b].item()
                        # Use original audio if available, otherwise use x (which might be perturbed)
                        source_orig = orig_audio if orig_audio is not None else x
                        slice_from_original_list.append(source_orig[b:b+1, :, start_pos:end_pos])
                    slice_from_original = torch.cat(slice_from_original_list, dim=0)  # [B, C, T_slice]
                    
                    # Extract the SAME slice from perturbed full audio
                    slice_from_perturbed_list = []
                    for b in range(batch_size):
                        start_pos = start_positions_audio[b].item()
                        end_pos = end_positions_audio[b].item()
                        slice_from_perturbed_list.append(x[b:b+1, :, start_pos:end_pos])
                    slice_from_perturbed = torch.cat(slice_from_perturbed_list, dim=0)  # [B, C, T_slice]
                    
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
                    from slice_consistency import SliceInterval
                    slice_feature_lengths = torch.tensor([slice_orig_quant_in.shape[1]] * batch_size, device=emb.device, dtype=torch.long)
                    slice_interval = SliceInterval(
                        start_point=start_positions_audio,
                        end_point=end_positions_audio,
                        start_split=torch.zeros(batch_size, device=emb.device, dtype=torch.long),
                        end_split=slice_feature_lengths,
                        split_interval_lengths=slice_feature_lengths,
                    )
                    
                    # Compute consistency loss: slice from original vs slice from perturbed
                    # This aligns latent representations from slice-consistency and perturbation-consistency methods
                    slice_consistency_output = self.slice_consistency.compute_consistency_loss(
                        full_features=slice_original_features_dict,  # Slice from original
                        slice_features=slice_perturbed_features_dict,  # Same slice from perturbed
                        feature_lengths=slice_feature_lengths,
                        codebook_indices=slice_original_codebook_indices[0] if slice_original_codebook_indices else None,  # [K, B, T_slice]
                        slice_codebook_indices=slice_perturbed_codebook_indices[0] if slice_perturbed_codebook_indices else None,  # [K, B, T_slice]
                        slice_interval=slice_interval,
                    )
                    
                    # The loss computed above compares slice from original vs slice from perturbed
                    # But for metrics, we need to compute:
                    # 1. Slice consistency: original full vs original slice
                    # 2. Augmentation consistency: original full vs perturbed full
                    
                    # Compute slice consistency metric: original full vs original slice
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
                        from slice_consistency import SliceInterval
                        orig_slice_interval = SliceInterval(
                            start_point=start_positions_audio,
                            end_point=end_positions_audio,
                            start_split=start_positions_feature,
                            end_split=end_positions_feature,
                            split_interval_lengths=split_interval_lengths,
                        )
                        
                        # Compute slice consistency metric (original full vs original slice)
                        slice_consistency_metric = self.slice_consistency.compute_consistency_loss(
                            full_features=orig_full_features_dict,
                            slice_features=slice_original_features_dict,
                            feature_lengths=feature_lengths,
                            codebook_indices=orig_full_codebook_indices,
                            slice_codebook_indices=slice_original_codebook_indices[0] if slice_original_codebook_indices else None,
                            slice_interval=orig_slice_interval,
                        )
                        
                        # Rename to slice_consistency_accuracy for metrics
                        if 'codebook_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_accuracy'] = slice_consistency_metric['codebook_accuracy']
                        if 'first_codebook_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_codebook0_accuracy'] = slice_consistency_metric['first_codebook_accuracy']
                        if 'first_3_codebooks_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_first_3_codebooks_accuracy'] = slice_consistency_metric['first_3_codebooks_accuracy']
                    
                    # Compute augmentation consistency (original full vs perturbed full)
                    if original_codebook_indices is not None and full_codebook_indices is not None:
                        augmentation_consistency = self.slice_consistency.compute_augmentation_consistency(
                            original_codebook_indices=original_codebook_indices,
                            perturbed_codebook_indices=full_codebook_indices,
                        )
                        slice_consistency_output.update(augmentation_consistency)
            
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
            orig_audio = x.clone() if need_original_for_slice else None
            
            # Apply perturbation to full audio if enabled (even in eval mode for validation metrics)
            if self.perturb_encoder is not None and self.perturb_encoder.perturb_all_audio:
                # Only create sample indices for deterministic behavior (validation), skip in training for speed
                if batch_idx is not None:
                    batch_size = x.shape[0]
                    sample_indices = torch.arange(batch_size, device=x.device)
                    x = self.perturb_encoder(x, batch_idx=batch_idx, sample_indices=sample_indices)
                else:
                    # Training mode: use simple call without extra parameters (faster)
                    x = self.perturb_encoder(x)
            
            # In eval mode, encode() returns codes, not embeddings
            # We need to get embeddings directly from the encoder
            if self.training:
                frames = self.encode(x)
            else:
                # In eval mode, get embeddings directly from encoder
                # Handle segmentation like encode() does
                _, channels, length = x.shape
                segment_length = self.segment_length
                if segment_length is None:
                    segment_length = length
                    stride = length
                else:
                    stride = self.segment_stride
                    assert stride is not None
                
                frames = []
                for offset in range(0, length, stride):
                    frame = x[:, :, offset: offset + segment_length]
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
            
            loss_w = torch.tensor([0.0], device=x.device, requires_grad=False)
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
            output = self.decode(codes)[:, :, :x.shape[-1]]
            
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
                    
                    # Extract the SAME slice from both original and perturbed full audio
                    # This aligns latent representations from slice-consistency and perturbation-consistency methods
                    
                    # Extract slice from original full audio
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
                    
                    # Extract the SAME slice from perturbed full audio
                    slice_from_perturbed_list = []
                    for b in range(batch_size):
                        start_pos = start_positions_audio[b].item()
                        end_pos = end_positions_audio[b].item()
                        max_pos = min(end_pos, x.shape[2])
                        if start_pos < max_pos:
                            slice_from_perturbed_list.append(x[b:b+1, :, start_pos:max_pos])
                        else:
                            slice_from_perturbed_list.append(x[b:b+1, :, start_pos:start_pos+1])
                    slice_from_perturbed = torch.cat(slice_from_perturbed_list, dim=0)  # [B, C, T_slice]
                    
                    # Encode both slices
                    # In eval mode, encode() returns codes, not embeddings - get embeddings directly
                    if self.training:
                        slice_original_frames = self.encode(slice_from_original)
                        slice_perturbed_frames = self.encode(slice_from_perturbed)
                    else:
                        # In eval mode, get embeddings directly from encoder
                        # Handle normalization if needed
                        slice_orig_frame = slice_from_original
                        slice_pert_frame = slice_from_perturbed
                        if self.normalize:
                            mono_orig = slice_orig_frame.mean(dim=1, keepdim=True)
                            volume_orig = mono_orig.pow(2).mean(dim=2, keepdim=True).sqrt()
                            scale_orig = 1e-8 + volume_orig
                            slice_orig_frame = slice_orig_frame / scale_orig
                            scale_orig = scale_orig.view(-1, 1)
                            
                            mono_pert = slice_pert_frame.mean(dim=1, keepdim=True)
                            volume_pert = mono_pert.pow(2).mean(dim=2, keepdim=True).sqrt()
                            scale_pert = 1e-8 + volume_pert
                            slice_pert_frame = slice_pert_frame / scale_pert
                            scale_pert = scale_pert.view(-1, 1)
                        else:
                            scale_orig = None
                            scale_pert = None
                        # Get embeddings from encoder
                        emb_slice_orig = self.encoder(slice_orig_frame)  # [B, D, T]
                        emb_slice_pert = self.encoder(slice_pert_frame)  # [B, D, T]
                        slice_original_frames = [(emb_slice_orig, scale_orig)]
                        slice_perturbed_frames = [(emb_slice_pert, scale_pert)]
                    
                    # Get features from both slices
                    slice_original_features = {}
                    slice_perturbed_features = {}
                    slice_original_codebook_indices = []
                    slice_perturbed_codebook_indices = []
                    slice_original_sub_quants = []
                    slice_perturbed_sub_quants = []
                    
                    for (emb_slice_orig, scale_slice_orig), (emb_slice_pert, scale_slice_pert) in zip(slice_original_frames, slice_perturbed_frames):
                        slice_original_features["quant_in"] = emb_slice_orig
                        slice_perturbed_features["quant_in"] = emb_slice_pert
                        
                        # Quantize both slices with same bandwidth
                        qv_slice_orig = self.quantizer(emb_slice_orig, self.frame_rate, bw)
                        qv_slice_pert = self.quantizer(emb_slice_pert, self.frame_rate, bw)
                        
                        slice_original_features["quant_out"] = qv_slice_orig.quantized
                        slice_perturbed_features["quant_out"] = qv_slice_pert.quantized
                        
                        slice_original_codebook_indices.append(qv_slice_orig.codes)
                        slice_perturbed_codebook_indices.append(qv_slice_pert.codes)
                        
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
                                sub_quants_list.append(torch.stack(frame_sub_quants))
                    
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
                    from slice_consistency import SliceInterval
                    slice_feature_lengths = torch.tensor([slice_orig_quant_in.shape[1]] * batch_size, device=emb.device, dtype=torch.long)
                    slice_interval = SliceInterval(
                        start_point=start_positions_audio,
                        end_point=end_positions_audio,
                        start_split=torch.zeros(batch_size, device=emb.device, dtype=torch.long),
                        end_split=slice_feature_lengths,
                        split_interval_lengths=slice_feature_lengths,
                    )
                    
                    # Compute consistency loss: slice from original vs slice from perturbed
                    # This aligns latent representations from slice-consistency and perturbation-consistency methods
                    slice_consistency_output = self.slice_consistency.compute_consistency_loss(
                        full_features=slice_original_features_dict,  # Slice from original
                        slice_features=slice_perturbed_features_dict,  # Same slice from perturbed
                        feature_lengths=slice_feature_lengths,
                        codebook_indices=slice_original_codebook_indices[0] if slice_original_codebook_indices else None,
                        slice_codebook_indices=slice_perturbed_codebook_indices[0] if slice_perturbed_codebook_indices else None,
                        slice_interval=slice_interval,
                    )
                    
                    # The loss computed above compares slice from original vs slice from perturbed
                    # But for metrics, we need to compute:
                    # 1. Slice consistency: original full vs original slice
                    # 2. Augmentation consistency: original full vs perturbed full
                    
                    # Compute slice consistency metric: original full vs original slice
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
                        from slice_consistency import SliceInterval
                        orig_slice_interval = SliceInterval(
                            start_point=start_positions_audio,
                            end_point=end_positions_audio,
                            start_split=start_positions_feature,
                            end_split=end_positions_feature,
                            split_interval_lengths=split_interval_lengths,
                        )
                        
                        # Compute slice consistency metric (original full vs original slice)
                        slice_consistency_metric = self.slice_consistency.compute_consistency_loss(
                            full_features=orig_full_features_dict,
                            slice_features=slice_original_features_dict,
                            feature_lengths=feature_lengths,
                            codebook_indices=orig_full_codebook_indices,
                            slice_codebook_indices=slice_original_codebook_indices[0] if slice_original_codebook_indices else None,
                            slice_interval=orig_slice_interval,
                        )
                        
                        # Rename to slice_consistency_accuracy for metrics
                        if 'codebook_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_accuracy'] = slice_consistency_metric['codebook_accuracy']
                        if 'first_codebook_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_codebook0_accuracy'] = slice_consistency_metric['first_codebook_accuracy']
                        if 'first_3_codebooks_accuracy' in slice_consistency_metric:
                            slice_consistency_output['slice_consistency_first_3_codebooks_accuracy'] = slice_consistency_metric['first_3_codebooks_accuracy']
                    
                    # Compute augmentation consistency (original full vs perturbed full)
                    if original_codebook_indices is not None and full_codebook_indices is not None:
                        augmentation_consistency = self.slice_consistency.compute_augmentation_consistency(
                            original_codebook_indices=original_codebook_indices,
                            perturbed_codebook_indices=full_codebook_indices,
                        )
                        slice_consistency_output.update(augmentation_consistency)
            
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
    ):
        """Create model with slice consistency support.
        
        Args:
            slice_consistency: Dictionary with slice consistency parameters
                - slice_interval_type: "random"
                - split_interval_percentage: 0.2
                - feature_types: ["quant_in"]
                - loss_types: ["mse_loss"]
                - loss_weights: [20.0]
                - target_sr: 24000
                - ds_rate: 320
                - mse_loss_reduction: "mean"
            perturb_encoder: Dictionary with perturbation parameters
                - perturb_methods: ["volume_aug", "inversion_aug"]
                - volume_aug_config: {"gain_range": (0.5, 2.0), "apply_prob": 0.5}
                - inversion_aug_config: {"apply_prob": 0.5}
                - perturb_all_audio: True
                - perturb_slice_audio: True
        """
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
        )
        return model
