"""Perturb encoder for audio augmentation (volume/gain and inversion)."""

import torch
import torch.nn as nn
from typing import Optional, Sequence, Union, Dict
from dataclasses import dataclass


@dataclass
class VolumeAugConfig:
    """Configuration for volume/gain augmentation."""
    gain_range: tuple = (0.5, 2.0)  # (min_gain, max_gain) - gain multiplier
    apply_prob: float = 0.5  # Probability of applying volume augmentation


@dataclass
class InversionAugConfig:
    """Configuration for audio inversion augmentation."""
    apply_prob: float = 0.5  # Probability of applying inversion


class PerturbEncoder(nn.Module):
    """Perturb encoder that applies data augmentations to audio.
    
    Supports:
    - Volume/gain augmentation: Randomly scales audio by a gain factor
    - Audio inversion: Inverts audio (multiply by -1)
    
    Similar to ConsistencyInNeuralCodec's PerturbEncoder but with simpler augmentations.
    """
    
    def __init__(
        self,
        perturb_methods: Optional[Sequence[str]] = None,
        volume_aug_config: Optional[Union[Dict, VolumeAugConfig]] = None,
        inversion_aug_config: Optional[Union[Dict, InversionAugConfig]] = None,
        perturb_all_audio: bool = True,
        perturb_slice_audio: Optional[bool] = None,
    ):
        """
        Args:
            perturb_methods: List of perturbation methods to apply ["volume_aug", "inversion_aug"]
            volume_aug_config: Configuration for volume augmentation
            inversion_aug_config: Configuration for inversion augmentation
            perturb_all_audio: Whether to apply perturbations to full audio
            perturb_slice_audio: Whether to apply perturbations to slice audio (if None, uses perturb_all_audio)
        """
        super().__init__()
        
        if perturb_methods is None:
            perturb_methods = []
        
        self.perturb_methods = perturb_methods
        self.perturb_all_audio = perturb_all_audio
        self.perturb_slice_audio = perturb_slice_audio if perturb_slice_audio is not None else perturb_all_audio
        
        # Parse volume augmentation config
        if volume_aug_config is None:
            volume_aug_config = VolumeAugConfig()
        elif isinstance(volume_aug_config, dict):
            # Convert list to tuple for gain_range if needed
            if 'gain_range' in volume_aug_config and isinstance(volume_aug_config['gain_range'], list):
                volume_aug_config['gain_range'] = tuple(volume_aug_config['gain_range'])
            volume_aug_config = VolumeAugConfig(**volume_aug_config)
        self.volume_aug_config = volume_aug_config
        
        # Parse inversion augmentation config
        if inversion_aug_config is None:
            inversion_aug_config = InversionAugConfig()
        elif isinstance(inversion_aug_config, dict):
            inversion_aug_config = InversionAugConfig(**inversion_aug_config)
        self.inversion_aug_config = inversion_aug_config
    
    def forward_volume_aug(self, audio: torch.Tensor, batch_idx: Optional[int] = None, sample_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply volume/gain augmentation to audio.
        
        Args:
            audio: Audio tensor [B, C, T] or [B, T]
            batch_idx: Optional batch index for deterministic behavior (validation)
            sample_indices: Optional sample indices [B] for deterministic behavior (validation)
            
        Returns:
            Perturbed audio with random gain applied
        """
        # Apply perturbations during both training and validation (for metric computation)
        # Deterministic for validation, random for training
        batch_size = audio.shape[0]
        if batch_idx is not None and sample_indices is not None:
            # Deterministic mode: use batch_idx and sample indices
            generator = torch.Generator(device=audio.device)
            for b in range(batch_size):
                seed = (batch_idx * 1000 + sample_indices[b].item()) % (2**31)
                generator.manual_seed(seed)
                if torch.rand(1, generator=generator, device=audio.device).item() < self.volume_aug_config.apply_prob:
                    # Sample deterministic gain
                    min_gain, max_gain = self.volume_aug_config.gain_range
                    gain = torch.rand(1, generator=generator, device=audio.device).item() * (max_gain - min_gain) + min_gain
                    # Apply gain (works for both [B, C, T] and [B, T] shapes)
                    if audio.dim() == 3:  # [B, C, T]
                        audio[b:b+1] = audio[b:b+1] * gain
                    else:  # [B, T]
                        audio[b] = audio[b] * gain
        else:
            # Random mode (training)
            if torch.rand(1).item() < self.volume_aug_config.apply_prob:
                # Sample random gain
                min_gain, max_gain = self.volume_aug_config.gain_range
                gain = torch.rand(1, device=audio.device).item() * (max_gain - min_gain) + min_gain
                # Apply gain (works for both [B, C, T] and [B, T] shapes)
                audio = audio * gain
        return audio
    
    def forward_inversion_aug(self, audio: torch.Tensor, batch_idx: Optional[int] = None, sample_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply audio inversion (multiply by -1).
        
        Args:
            audio: Audio tensor [B, C, T] or [B, T]
            batch_idx: Optional batch index for deterministic behavior (validation)
            sample_indices: Optional sample indices [B] for deterministic behavior (validation)
            
        Returns:
            Inverted audio (if randomly selected)
        """
        # Apply perturbations during both training and validation (for metric computation)
        # Deterministic for validation, random for training
        batch_size = audio.shape[0]
        if batch_idx is not None and sample_indices is not None:
            # Deterministic mode: use batch_idx and sample indices
            generator = torch.Generator(device=audio.device)
            for b in range(batch_size):
                seed = (batch_idx * 1000 + sample_indices[b].item() + 100000) % (2**31)  # Different seed offset
                generator.manual_seed(seed)
                if torch.rand(1, generator=generator, device=audio.device).item() < self.inversion_aug_config.apply_prob:
                    # Apply inversion
                    if audio.dim() == 3:  # [B, C, T]
                        audio[b:b+1] = -audio[b:b+1]
                    else:  # [B, T]
                        audio[b] = -audio[b]
        else:
            # Random mode (training)
            if torch.rand(1).item() < self.inversion_aug_config.apply_prob:
                audio = -audio
        return audio
    
    def forward(self, audio: torch.Tensor, batch_idx: Optional[int] = None, sample_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply perturbations to audio.
        
        Args:
            audio: Audio tensor [B, C, T] or [B, T]
            batch_idx: Optional batch index for deterministic behavior (validation)
            sample_indices: Optional sample indices [B] for deterministic behavior (validation)
            
        Returns:
            Perturbed audio
        """
        for perturb_method in self.perturb_methods:
            forward_method = getattr(self, f"forward_{perturb_method}")
            audio = forward_method(audio, batch_idx, sample_indices)
        return audio
    
    def __call__(self, audio: torch.Tensor, batch_idx: Optional[int] = None, sample_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self.forward(audio, batch_idx, sample_indices)
