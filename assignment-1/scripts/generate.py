"""Generate text from a trained Transformer LM checkpoint.

Reconstructs the model from the training run's config.json, loads the checkpoint
weights and the BPE tokenizer, then samples a completion for a prompt.

Example:
    uv run scripts/generate.py \
        --config experiments/logs/baseline/config.json \
        --checkpoint checkpoints/baseline.pt \
        --vocab scripts/ts_bpe_output/vocab.json \
        --merges scripts/ts_bpe_output/merges.json \
        --prompt "Once upon a time" \
        --temperature 0.8 --top-p 0.95 --max-new-tokens 256
"""

import argparse
import json

import torch

from cs336_basics.decode import generate_text
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

MODEL_KEYS = (
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "rope_theta",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate from a trained LM")
    p.add_argument("--config", type=str, required=True, help="Training run's config.json")
    p.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint .pt")
    p.add_argument("--vocab", type=str, required=True, help="BPE vocab .json")
    p.add_argument("--merges", type=str, required=True, help="BPE merges .json")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--special-token", type=str, default="<|endoftext|>")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)
    model_args = {k: config[k] for k in MODEL_KEYS}

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [args.special_token])

    model = TransformerLM(**model_args, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    completion = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token=args.special_token,
        device=args.device,
    )
    print(args.prompt + completion)


if __name__ == "__main__":
    main()
