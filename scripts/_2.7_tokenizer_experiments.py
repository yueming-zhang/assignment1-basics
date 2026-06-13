"""Tokenizer experiments (Problem tokenizer_experiments).

Parts:
  (a) Compression ratio for TS and OWT tokenizers on their respective corpora
  (b) Cross-tokenize: OWT sample with TS tokenizer
  (c) Throughput estimate and time to tokenize the Pile (825 GB)
  (d) Encode full train/valid datasets to uint16 numpy arrays

Usage:
    uv run python scripts/tokenizer_experiments.py
    uv run python scripts/tokenizer_experiments.py --skip-encode  # skip part (d)
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizer import Tokenizer

# Paths
TS_VOCAB   = Path("scripts/ts_bpe_output/vocab.json")
TS_MERGES  = Path("scripts/ts_bpe_output/merges.json")
OWT_VOCAB  = Path("profile/owt_bpe_output/vocab.json")
OWT_MERGES = Path("profile/owt_bpe_output/merges.json")

DATA = {
    "ts":  {"train": Path("data/TinyStoriesV2-GPT4-train.txt"),
            "valid": Path("data/TinyStoriesV2-GPT4-valid.txt")},
    "owt": {"train": Path("data/owt_train.txt"),
            "valid": Path("data/owt_valid.txt")},
}

SPECIAL_TOKENS = ["<|endoftext|>"]
PILE_BYTES = 825 * 1024 ** 3  # 825 GB


def load_tokenizer(vocab_path: Path, merges_path: Path) -> Tokenizer:
    return Tokenizer.from_files(str(vocab_path), str(merges_path), SPECIAL_TOKENS)


def sample_documents(path: Path, n: int = 10, seed: int = 42,
                     prefix_bytes: int = 50 * 1024 * 1024) -> list[str]:
    """Return n randomly sampled documents (split on <|endoftext|>).

    Only reads a bounded prefix of the file (default 50 MB) rather than the
    whole thing — the corpora are multi-GB and reading them fully would exhaust
    memory. A 50 MB prefix holds far more than n documents.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read(prefix_bytes)
    docs = [d.strip() for d in text.split("<|endoftext|>") if d.strip()]
    # Drop the last doc: it may be truncated by the prefix cutoff.
    if len(docs) > 1:
        docs = docs[:-1]
    rng = random.Random(seed)
    return rng.sample(docs, min(n, len(docs)))


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """bytes / token across all docs."""
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    return total_bytes / total_tokens


# ---------------------------------------------------------------------------
# Part (a): compression ratios on matching corpora
# ---------------------------------------------------------------------------
def part_a(ts_tok: Tokenizer, owt_tok: Tokenizer):
    print("\n=== (a) Compression ratio ===")

    ts_docs  = sample_documents(DATA["ts"]["train"])
    owt_docs = sample_documents(DATA["owt"]["train"])

    ts_ratio  = compression_ratio(ts_tok,  ts_docs)
    owt_ratio = compression_ratio(owt_tok, owt_docs)

    print(f"TinyStories tokenizer on TS  corpus: {ts_ratio:.2f} bytes/token")
    print(f"OWT tokenizer       on OWT corpus: {owt_ratio:.2f} bytes/token")


# ---------------------------------------------------------------------------
# Part (b): cross-tokenization
# ---------------------------------------------------------------------------
def part_b(ts_tok: Tokenizer, owt_tok: Tokenizer):
    print("\n=== (b) Cross-tokenization: OWT sample with TS tokenizer ===")

    owt_docs = sample_documents(DATA["owt"]["train"])

    owt_with_owt_tok = compression_ratio(owt_tok, owt_docs)
    owt_with_ts_tok  = compression_ratio(ts_tok,  owt_docs)

    print(f"OWT sample with OWT tokenizer: {owt_with_owt_tok:.2f} bytes/token")
    print(f"OWT sample with TS  tokenizer: {owt_with_ts_tok:.2f} bytes/token")
    print(
        "The TS tokenizer produces more tokens per byte on OWT text because its "
        "vocabulary was learned from children's stories and lacks the web-text "
        "patterns (URLs, code, punctuation clusters) common in OWT."
    )


# ---------------------------------------------------------------------------
# Part (c): throughput
# ---------------------------------------------------------------------------
def part_c(owt_tok: Tokenizer):
    print("\n=== (c) Throughput estimate ===")

    # Use a ~1 MB chunk of OWT training text for the benchmark
    with open(DATA["owt"]["train"], encoding="utf-8") as f:
        sample_text = f.read(1024 * 1024)  # 1 MB

    sample_bytes = len(sample_text.encode("utf-8"))

    start = time.perf_counter()
    _ = owt_tok.encode(sample_text)
    elapsed = time.perf_counter() - start

    throughput_bps = sample_bytes / elapsed
    throughput_mbps = throughput_bps / 1024 ** 2

    pile_seconds = PILE_BYTES / throughput_bps
    pile_hours   = pile_seconds / 3600

    print(f"Sample size : {sample_bytes / 1024 ** 2:.2f} MB")
    print(f"Elapsed     : {elapsed:.3f}s")
    print(f"Throughput  : {throughput_mbps:.1f} MB/s ({throughput_bps / 1e6:.1f} bytes/s)")
    print(f"Pile (825 GB) estimate: {pile_seconds:.0f}s = {pile_hours:.1f} hours")


# ---------------------------------------------------------------------------
# Part (d): encode full datasets to uint16 numpy arrays
# ---------------------------------------------------------------------------
def encode_file_to_uint16(tokenizer: Tokenizer, src: Path, dst: Path,
                          batch: int = 1 << 20):
    """Stream-encode src, collect into a uint16 array, save as .npy.

    Token IDs are flushed into uint16 numpy chunks every `batch` tokens instead
    of accumulating in a Python list. A list of boxed ints costs ~28 B/token
    (~80+ GB for the 11.9 GB OWT corpus, which OOM-kills the WSL VM); the chunked
    form keeps only one batch of Python ints alive at a time, and the final array
    is 2 B/token.
    """
    print(f"  Encoding {src} -> {dst} ...")
    start = time.perf_counter()

    chunks: list[np.ndarray] = []
    buf: list[int] = []
    with open(src, encoding="utf-8") as f:
        for tid in tokenizer.encode_iterable(f):
            buf.append(tid)
            if len(buf) >= batch:
                chunks.append(np.array(buf, dtype=np.uint16))
                buf.clear()
    if buf:
        chunks.append(np.array(buf, dtype=np.uint16))

    arr = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr)

    elapsed = time.perf_counter() - start
    print(f"    {len(arr):,} tokens, saved {dst.stat().st_size / 1024**2:.1f} MB in {elapsed:.1f}s")


def part_d(ts_tok: Tokenizer, owt_tok: Tokenizer):
    print("\n=== (d) Encode full datasets to uint16 ===")
    print(
        "uint16 supports values 0–65535, which covers both the 10K TS vocab "
        "and the 32K OWT vocab while using only 2 bytes per token (half of int32)."
    )

    out = Path("data/tokenized")
    encode_file_to_uint16(ts_tok,  DATA["ts"]["train"],  out / "ts_train.npy")
    encode_file_to_uint16(ts_tok,  DATA["ts"]["valid"],  out / "ts_valid.npy")
    encode_file_to_uint16(owt_tok, DATA["owt"]["train"], out / "owt_train.npy")
    encode_file_to_uint16(owt_tok, DATA["owt"]["valid"], out / "owt_valid.npy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-encode", action="store_true",
                        help="Skip part (d) — full dataset encoding")
    args = parser.parse_args()

    for path, name in [(TS_VOCAB, "TS vocab"), (TS_MERGES, "TS merges"),
                       (OWT_VOCAB, "OWT vocab"), (OWT_MERGES, "OWT merges")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}. "
                "Run scripts/train_ts_tokenizer.py first."
            )

    print("Loading tokenizers ...")
    ts_tok  = load_tokenizer(TS_VOCAB,  TS_MERGES)
    owt_tok = load_tokenizer(OWT_VOCAB, OWT_MERGES)
    print(f"  TS  vocab size: {len(ts_tok.vocab)}")
    print(f"  OWT vocab size: {len(owt_tok.vocab)}")

    part_a(ts_tok, owt_tok)
    part_b(ts_tok, owt_tok)
    part_c(owt_tok)

    if not args.skip_encode:
        part_d(ts_tok, owt_tok)
    else:
        print("\n(d) skipped via --skip-encode")


if __name__ == "__main__":
    main()
