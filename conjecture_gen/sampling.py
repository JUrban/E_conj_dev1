"""
Sampling utilities for conjecture generation: top-k, nucleus (top-p),
and temperature scaling.
"""

import torch
import torch.nn.functional as F


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0,
                       top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Sample from logits with temperature, top-k, and/or nucleus (top-p) filtering.

    Args:
        logits: (batch, vocab) raw logits
        temperature: temperature scaling (1.0 = no change, <1 = sharper, >1 = flatter)
        top_k: keep only top-k logits (0 = disabled)
        top_p: keep smallest set of logits with cumulative prob >= top_p (0.0 = disabled)

    Returns:
        (batch,) sampled indices
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        threshold = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # Nucleus (top-p) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative probability above the threshold
        # Shift right so that the first token above threshold is kept
        remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove_mask] = float('-inf')
        # Scatter back to original order
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
