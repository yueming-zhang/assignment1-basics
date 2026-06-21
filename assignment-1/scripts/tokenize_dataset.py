"""Encode a raw text corpus to a uint16 .npy token array for training.

Loads a trained BPE tokenizer (vocab + merges) and runs Tokenizer.encode_file,
which tokenizes in parallel and memory-maps the output, so large corpora never
need to fit in RAM.

Example:
    uv run scripts/tokenize_dataset.py \
        --vocab scripts/ts_bpe_output/vocab.json \
        --merges scripts/ts_bpe_output/merges.json \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --output data/ts_train.npy
"""

import argparse

from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenize a text file to a .npy token array")
    p.add_argument("--vocab", required=True, help="BPE vocab .json")
    p.add_argument("--merges", required=True, help="BPE merges .json")
    p.add_argument("--input", required=True, help="Raw .txt corpus")
    p.add_argument("--output", required=True, help="Destination .npy")
    p.add_argument("--special-token", default="<|endoftext|>")
    p.add_argument("--num-processes", type=int, default=None, help="Workers (default: all cores)")
    args = p.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [args.special_token])
    out = tokenizer.encode_file(args.input, args.output, num_processes=args.num_processes)
    print(f"Wrote token ids to {out}")


if __name__ == "__main__":
    main()
