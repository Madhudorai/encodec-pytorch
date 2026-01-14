"""Helper module to extract codebook 0 indices from model."""

import torch
import typing as tp


def extract_codebook0_indices(model, audio: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Extract codebook 0 indices from model forward pass.
    
    Args:
        model: EncodecModelWithSliceConsistency model
        audio: Input audio tensor [B, C, T] or [1, C, T] or [C, T]
        bandwidth: Target bandwidth for quantization
    
    Returns:
        codebook0_indices: Codebook 0 indices [B, T_feat] where T_feat is feature length
    """
    model.eval()
    model.bandwidth = bandwidth
    
    # Ensure audio has batch dimension
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)  # [C, T] -> [1, C, T]
    
    B, C, T = audio.shape
    
    # Move to device
    if torch.cuda.is_available():
        audio = audio.cuda()
    
    with torch.no_grad():
        # Reshape for multi-channel if needed
        if C > 1:
            audio_reshaped = audio.reshape(B * C, 1, T)  # [B*C, 1, T]
        else:
            audio_reshaped = audio
        
        # In eval mode, model.encode() returns codes, not embeddings
        # We need to get embeddings directly from the encoder
        # Handle segmentation like encode() does
        _, channels, length = audio_reshaped.shape
        segment_length = model.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = model.segment_stride
            if stride is None:
                stride = segment_length
        
        # Get embeddings directly from encoder (like model does in eval mode)
        codebook_indices_list = []
        
        for offset in range(0, length, stride):
            frame = audio_reshaped[:, :, offset: offset + segment_length]
            
            # Handle normalization
            if model.normalize:
                mono = frame.mean(dim=1, keepdim=True)
                volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
                scale = 1e-8 + volume
                frame = frame / scale
            else:
                scale = None
            
            # Get embeddings from encoder directly
            emb = model.encoder(frame)  # [B, D, T] where D is feature dimension (e.g., 128)
            
            # Use quantizer.encode() like the model does in eval mode
            # This returns codes [K, B, T_feat] where K is number of codebooks
            codes = model.quantizer.encode(emb, model.frame_rate, bandwidth)
            # codes is [K, B, T_feat] where K is number of codebooks
            if len(codebook_indices_list) == 0:
                # Extract codebook 0 (first codebook)
                codebook0 = codes[0]  # [B, T_feat] or [B*C, T_feat] if multi-channel
                codebook_indices_list.append(codebook0)
                break  # Only need first frame for analysis
        
        if len(codebook_indices_list) == 0:
            raise ValueError("Failed to extract codebook indices")
        
        codebook0_indices = codebook_indices_list[0]  # [B, T_feat] or [B*C, T_feat]
        
        # Reshape back if multi-channel
        if C > 1:
            # codebook0_indices is [B*C, T_feat], reshape to [B, C, T_feat]
            codebook0_indices = codebook0_indices.reshape(B, C, -1)
            # For analysis, we typically use first channel
            codebook0_indices = codebook0_indices[:, 0, :]  # [B, T_feat]
    
    return codebook0_indices.cpu()  # [B, T_feat]
