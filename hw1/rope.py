from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device

    # Compute position indices
    pos_indices = torch.arange(0, seqlen, device=device).float()
    
    # Compute sinusoidal frequencies
    freqs = torch.pow(theta, -torch.arange(0, head_dim, 2).float() / head_dim).to(device)
    angles = pos_indices[:, None] * freqs[None, :]
    sine, cosine = torch.sin(angles), torch.cos(angles)

    # Expand dims to match query/key tensor shape for broadcasting
    sine = sine.view(1, seqlen, 1, -1).repeat(query.shape[0], 1, query.shape[2], 1)
    cosine = cosine.view(1, seqlen, 1, -1).repeat(query.shape[0], 1, query.shape[2], 1)

    # Apply rotation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # Rotated embeddings
    query_rotated_real = cosine * query_real - sine * query_imag
    query_rotated_imag = sine * query_real + cosine * query_imag
    key_rotated_real = cosine * key_real - sine * key_imag
    key_rotated_imag = sine * key_real + cosine * key_imag

    # Combine back the rotated real and imaginary components
    query_out = torch.stack((query_rotated_real, query_rotated_imag), dim=-1).view(query.shape)
    key_out = torch.stack((key_rotated_real, key_rotated_imag), dim=-1).view(key.shape)

    return query_out, key_out
