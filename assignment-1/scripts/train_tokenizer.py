"""Train a BPE tokenizer on a text file and save vocab.json + merges.json.

Usage:
    # TinyStories (10K vocab)
    uv run python scripts/train_tokenizer.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 --out-dir scripts/ts_bpe_output

    # OpenWebText (32K vocab)
    uv run python scripts/train_tokenizer.py \
        --input data/owt_train.txt \
        --vocab-size 32000 --out-dir scripts/owt_bpe_output
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.adapters import run_train_bpe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a BPE tokenizer")
    p.add_argument("--input", type=Path, required=True, help="Training text file")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True, help="Dir for vocab.json + merges.json")
    p.add_argument("--special-token", action="append", default=None,
                   help="Special token (repeatable). Defaults to ['<|endoftext|>'].")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = args.special_token or ["<|endoftext|>"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE on {args.input} with vocab_size={args.vocab_size} ...")
    start = time.time()
    vocab, merges = run_train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )
    print(f"Done in {time.time() - start:.1f}s. Vocab size: {len(vocab)}, Merges: {len(merges)}")

    vocab_path = args.out_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({k: v.decode("latin-1") for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)
    print(f"Vocab saved to {vocab_path}")

    merges_path = args.out_dir / "merges.json"
    with open(merges_path, "w") as f:
        json.dump([[a.decode("latin-1"), b.decode("latin-1")] for a, b in merges], f, ensure_ascii=False, indent=2)
    print(f"Merges saved to {merges_path}")

    longest = max(vocab.values(), key=len)
    print(f"Longest token ({len(longest)} bytes): {longest}")


if __name__ == "__main__":
    main()
