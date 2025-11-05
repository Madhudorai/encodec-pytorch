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
        
        # Store original audio if we need it for slice extraction
        # We need it if: slice consistency is enabled AND we're perturbing full audio
        # (because we want to extract slice from original, not perturbed, so slice can be augmented independently)
        need_original_for_slice = (
            self.training and 
            return_slice_consistency and 
            self.use_slice_consistency and 
            self.perturb_encoder is not None and 
            self.perturb_encoder.perturb_all_audio
        )
        orig_audio = x.clone() if need_original_for_slice else None
        
        # Apply perturbation to full audio if enabled
        # When slice consistency is enabled, we want to calculate consistency on augmented audio
        # So if perturb_all_audio is True, we always augment the full audio
        # (Individual augmentations have their own apply_prob for probabilistic application)
        if self.training and self.perturb_encoder is not None and self.perturb_encoder.perturb_all_audio:
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
            
            # Compute slice consistency if requested
            slice_consistency_output = None
            if return_slice_consistency and self.use_slice_consistency and self.training:
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
                    
                    # Random start positions in feature space
                    max_start_feature = feature_lengths - split_interval_lengths
                    max_start_feature = torch.clamp(max_start_feature, min=0)
                    start_positions_feature = (torch.rand(batch_size, device=emb.device) * max_start_feature.float()).long()
                    end_positions_feature = torch.clamp(start_positions_feature + split_interval_lengths, max=feature_lengths)
                    
                    # Convert feature positions to audio positions
                    start_positions_audio = start_positions_feature * ds_rate
                    end_positions_audio = end_positions_feature * ds_rate
                    audio_length = x.shape[2]  # [B, C, T]
                    end_positions_audio = torch.clamp(end_positions_audio, max=audio_length)
                    
                    # Extract slices for each batch item based on audio positions
                    # Extract from original audio (before perturbation) if available, otherwise from x
                    # This ensures slice is extracted from unperturbed audio, then perturbed independently
                    slice_source = orig_audio if orig_audio is not None else x
                    
                    slice_audio_list = []
                    for b in range(batch_size):
                        start_pos = start_positions_audio[b].item()
                        end_pos = end_positions_audio[b].item()
                        slice_audio_list.append(slice_source[b:b+1, :, start_pos:end_pos])
                    slice_audio = torch.cat(slice_audio_list, dim=0)  # [B, C, T_slice]
                    
                    # Apply perturbation to slice audio if enabled
                    # Note: Individual augmentations (volume_aug, inversion_aug) have their own apply_prob
                    # So even if perturb_slice_audio is True, each augmentation may be applied probabilistically
                    if self.perturb_encoder is not None and self.perturb_encoder.perturb_slice_audio:
                        slice_audio = self.perturb_encoder(slice_audio)
                    
                    # Encode slice
                    slice_frames = self.encode(slice_audio)
                    
                    slice_features = {}
                    slice_codebook_indices = []
                    slice_sub_quants = []
                    
                    for emb_slice, scale_slice in slice_frames:
                        slice_features["quant_in"] = emb_slice  # [B, D, T_slice]
                        
                        # Quantize slice with same bandwidth
                        qv_slice = self.quantizer(emb_slice, self.frame_rate, bw)
                        slice_features["quant_out"] = qv_slice.quantized  # [B, D, T_slice]
                        slice_codebook_indices.append(qv_slice.codes)  # [K, B, T_slice]
                        
                        # Get sub_quants for slice (individual quantizer outputs)
                        residual = emb_slice
                        frame_sub_quants = []
                        n_q_used = min(len(self.quantizer.vq.layers), qv_slice.codes.shape[0])
                        for i in range(n_q_used):
                            layer = self.quantizer.vq.layers[i]
                            quantized, indices, layer_loss = layer(residual)
                            frame_sub_quants.append(quantized)
                            residual = residual - quantized.detach()
                        if frame_sub_quants:
                            slice_sub_quants.append(torch.stack(frame_sub_quants))  # [K, B, D, T_slice]
                    
                    # Convert features to [B, T, D] format for consistency computation
                    # Full audio features
                    full_quant_in = full_features["quant_in"].transpose(1, 2)  # [B, T, D]
                    full_quant_out = full_features["quant_out"].transpose(1, 2)  # [B, T, D]
                    
                    # Handle sub_quants - convert from [K, B, D, T] to [K, B, T, D]
                    full_sub_quants_t = None
                    if full_sub_quants and len(full_sub_quants) > 0:
                        full_sub_quants_t = full_sub_quants[0].transpose(2, 3)  # [K, B, T, D]
                    
                    # Slice features
                    slice_quant_in = slice_features["quant_in"].transpose(1, 2)  # [B, T_slice, D]
                    slice_quant_out = slice_features["quant_out"].transpose(1, 2)  # [B, T_slice, D]
                    
                    slice_sub_quants_t = None
                    if slice_sub_quants and len(slice_sub_quants) > 0:
                        slice_sub_quants_t = slice_sub_quants[0].transpose(2, 3)  # [K, B, T_slice, D]
                    
                    # Prepare features dictionary
                    full_features_dict = {
                        "quant_in": full_quant_in,
                        "quant_out": full_quant_out,
                    }
                    if full_sub_quants_t is not None:
                        full_features_dict["sub_quants"] = full_sub_quants_t
                    
                    slice_features_dict = {
                        "quant_in": slice_quant_in,
                        "quant_out": slice_quant_out,
                    }
                    if slice_sub_quants_t is not None:
                        slice_features_dict["sub_quants"] = slice_sub_quants_t
                    
                    # Create slice interval for consistency computation
                    # Use the feature-space positions we computed earlier
                    from slice_consistency import SliceInterval
                    slice_interval = SliceInterval(
                        start_point=start_positions_audio,
                        end_point=end_positions_audio,
                        start_split=start_positions_feature,
                        end_split=end_positions_feature,
                        split_interval_lengths=split_interval_lengths,
                    )
                    
                    # Compute slice consistency loss
                    # Note: full_codebook_indices is [K, B, T] from first frame, not a list
                    # Pass pre-computed slice_interval to ensure correct alignment
                    slice_consistency_output = self.slice_consistency.compute_consistency_loss(
                        full_features=full_features_dict,
                        slice_features=slice_features_dict,
                        feature_lengths=feature_lengths,
                        codebook_indices=full_codebook_indices if full_codebook_indices is not None else None,  # [K, B, T]
                        slice_codebook_indices=slice_codebook_indices[0] if slice_codebook_indices else None,  # [K, B, T_slice]
                        slice_interval=slice_interval,  # Use pre-computed interval
                    )
            
            if return_embeddings:
                return output, loss_w, frames, quantized_embeddings, slice_consistency_output
            elif return_slice_consistency:
                return output, loss_w, frames, slice_consistency_output
            else:
                return output, loss_w, frames
        else:
            # Evaluation mode - no slice consistency
            if return_embeddings:
                output = self.decode(frames)[:, :, :x.shape[-1]]
                quantized_embeddings = []
                for emb, scale in frames:
                    encoded_frame = self.quantizer.encode(emb, self.frame_rate, self.target_bandwidths[-1])
                    quantized_emb = self.quantizer.decode(encoded_frame)
                    quantized_embeddings.append(quantized_emb)
                return output, quantized_embeddings, None
            else:
                return self.decode(frames)[:, :, :x.shape[-1]]
    
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
                - loss_weights: [10.0]
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
