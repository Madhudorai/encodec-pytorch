import torch
import torch.nn.functional as F


def similarity_loss(embeddings1, embeddings2, token_idx=0):
    """Compute similarity loss between first token embeddings of paired channels.
    
    Args:
        embeddings1 (list): List of quantized embeddings for channel 1
        embeddings2 (list): List of quantized embeddings for channel 2  
        token_idx (int): Which token to compare (0 for first token)
    
    Returns:
        torch.Tensor: L2 loss between first token embeddings
    """
    if len(embeddings1) <= token_idx or len(embeddings2) <= token_idx:
        return torch.tensor(0.0, device=embeddings1[0].device, requires_grad=True)
    
    # Get the first token embeddings (shape: [B, D, T])
    token1 = embeddings1[token_idx].float()  # First codebook embedding
    token2 = embeddings2[token_idx].float()  # First codebook embedding
    
    # Compute L2 loss between the embeddings
    l2_loss = F.mse_loss(token1, token2)
    
    return l2_loss


def diversity_loss(embeddings1, embeddings2, token_idx=1):
    """Compute diversity loss between second token embeddings of paired channels.
    
    Args:
        embeddings1 (list): List of quantized embeddings for channel 1
        embeddings2 (list): List of quantized embeddings for channel 2
        token_idx (int): Which token to compare (1 for second token)
    
    Returns:
        torch.Tensor: Negative similarity loss (encourages difference)
    """
    if len(embeddings1) <= token_idx or len(embeddings2) <= token_idx:
        return torch.tensor(0.0, device=embeddings1[0].device, requires_grad=True)
    
    # Get the second token embeddings (shape: [B, D, T])
    token1 = embeddings1[token_idx].float()  # Second codebook embedding
    token2 = embeddings2[token_idx].float()  # Second codebook embedding
    
    # Compute negative L2 loss (maximize distance)
    # We use negative MSE to encourage diversity
    l2_similarity = F.mse_loss(token1, token2)
    diversity_loss = -l2_similarity  # Negative to encourage difference
    
    return diversity_loss


def pairwise_losses(embeddings1, embeddings2, sim_weight=0.5, div_weight=0.5):
    """Compute both similarity and diversity losses for paired channels.
    
    Args:
        embeddings1 (list): List of quantized embeddings for channel 1
        embeddings2 (list): List of quantized embeddings for channel 2
        sim_weight (float): Weight for similarity loss
        div_weight (float): Weight for diversity loss
    
    Returns:
        dict: Dictionary containing similarity and diversity losses
    """
    losses = {}
    
    # Similarity loss for first token (encourage same content)
    losses['l_sim'] = similarity_loss(embeddings1, embeddings2, token_idx=0) * sim_weight
    
    # Diversity loss for second token (encourage different channel characteristics)
    losses['l_div'] = diversity_loss(embeddings1, embeddings2, token_idx=1) * div_weight
    
    return losses


def total_pairwise_loss(embeddings1, embeddings2, sim_weight=0.5, div_weight=0.5):
    """Compute total pairwise loss combining similarity and diversity.
    
    Args:
        embeddings1 (list): List of quantized embeddings for channel 1
        embeddings2 (list): List of quantized embeddings for channel 2
        sim_weight (float): Weight for similarity loss
        div_weight (float): Weight for diversity loss
    
    Returns:
        torch.Tensor: Total pairwise loss
    """
    losses = pairwise_losses(embeddings1, embeddings2, sim_weight, div_weight)
    return losses['l_sim'] + losses['l_div']


def test_pairwise_losses():
    """Test function for pairwise losses."""
    # Create dummy embeddings
    batch_size = 2
    embedding_dim = 128
    time_steps = 32
    
    # Simulate 2 codebooks (first and second token)
    embeddings1 = [
        torch.randn(batch_size, embedding_dim, time_steps),  # First token
        torch.randn(batch_size, embedding_dim, time_steps),  # Second token
    ]
    
    embeddings2 = [
        torch.randn(batch_size, embedding_dim, time_steps),  # First token
        torch.randn(batch_size, embedding_dim, time_steps),  # Second token
    ]
    
    # Test similarity loss (should be positive)
    sim_loss = similarity_loss(embeddings1, embeddings2, token_idx=0)
    print(f"Similarity loss: {sim_loss.item():.4f}")
    
    # Test diversity loss (should be negative)
    div_loss = diversity_loss(embeddings1, embeddings2, token_idx=1)
    print(f"Diversity loss: {div_loss.item():.4f}")
    
    # Test total pairwise loss
    total_loss = total_pairwise_loss(embeddings1, embeddings2)
    print(f"Total pairwise loss: {total_loss.item():.4f}")
    
    # Test with identical first tokens (similarity should be 0)
    embeddings2[0] = embeddings1[0].clone()
    sim_loss_identical = similarity_loss(embeddings1, embeddings2, token_idx=0)
    print(f"Similarity loss (identical first tokens): {sim_loss_identical.item():.4f}")


if __name__ == '__main__':
    test_pairwise_losses()
