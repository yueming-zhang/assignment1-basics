"""Autoregressive decoding from a trained language model.

Supports temperature scaling (eq. 23) and top-p / nucleus sampling (eq. 24).
"""

import torch

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of highest-prob tokens with cumulative mass >= top_p.

    Zeros the truncated tail and renormalizes so the kept tokens sum to 1.
    """
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Keep token i if the mass strictly before it is still < top_p; this includes
    # the first token that pushes the cumulative sum to >= top_p (smallest set).
    keep = (cumsum - sorted_probs) < top_p
    keep[0] = True  # always keep at least the most probable token

    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = sorted_probs[keep]
    return filtered / filtered.sum()


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt: list[int] | torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    device: str | torch.device | None = None,
) -> list[int]:
    """Sample a completion of `prompt` from `model`, returning the new token ids.

    Stops at `eos_token_id` (if given) or after `max_new_tokens`. temperature=0
    means greedy (argmax) decoding; top_p (if set) restricts each step to the
    nucleus before sampling. The context is cropped to the model's context_length.
    """
    model.eval()
    if not torch.is_tensor(prompt):
        prompt = torch.tensor(prompt, dtype=torch.long)
    x = prompt.to(device).view(1, -1)  # (1, t)

    ctx = model.context_length
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(x[:, -ctx:])  # (1, seq, vocab)
        logits = logits[0, -1]  # next-token logits at the last position

        if temperature == 0:
            next_id = torch.argmax(logits)
        else:
            probs = softmax(logits / temperature, dim=-1)
            if top_p is not None:
                probs = top_p_filter(probs, top_p)
            next_id = torch.multinomial(probs, num_samples=1).squeeze()

        token = int(next_id.item())
        generated.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        x = torch.cat([x, next_id.view(1, 1)], dim=1)

    return generated


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token: str = "<|endoftext|>",
    device: str | torch.device | None = None,
) -> str:
    """Tokenize `prompt`, sample a completion, and decode it back to a string.

    Stops at the `eos_token` if it is in the vocabulary. The returned string is
    the completion only (not including the prompt), with any trailing EOS removed.
    """
    prompt_ids = tokenizer.encode(prompt)
    eos_id = tokenizer.bytes_to_id.get(eos_token.encode("utf-8"))

    new_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_id,
        device=device,
    )
    if eos_id is not None and new_ids and new_ids[-1] == eos_id:
        new_ids = new_ids[:-1]
    return tokenizer.decode(new_ids)
