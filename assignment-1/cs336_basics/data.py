"""Data loading utilities for language-model training."""

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (input, next-token-label) sequences from a token array.

    Picks `batch_size` random start positions (uniform, with replacement) such
    that both the length-`context_length` input window and its 1-shifted label
    window fit inside `dataset`. Returns two (batch_size, context_length) Long
    tensors on `device`; the labels are the inputs shifted one token ahead.
    """
    # Highest valid start leaves room for the label's extra shifted token.
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    # (batch, context_length) index grid: each row is start .. start+L-1.
    idx = starts[:, None] + np.arange(context_length)

    x = torch.tensor(dataset[idx], dtype=torch.long, device=device)
    y = torch.tensor(dataset[idx + 1], dtype=torch.long, device=device)
    return x, y
