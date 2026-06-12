"""Train BPE on OpenWebText and serialize vocab/merges to disk.

Usage:
    uv run python profile/run_owt_bpe.py
"""
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.adapters import run_train_bpe

INPUT_PATH = Path("data/owt_train.txt")
VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]
OUT_DIR = Path("profile/owt_bpe_output")

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Training BPE on {INPUT_PATH} with vocab_size={VOCAB_SIZE} ...")
start = time.time()
vocab, merges = run_train_bpe(
    input_path=INPUT_PATH,
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
)
elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s. Vocab size: {len(vocab)}, Merges: {len(merges)}")

# Serialize vocab: {id -> token hex string}
vocab_path = OUT_DIR / "vocab.json"
with open(vocab_path, "w") as f:
    json.dump({k: v.decode("latin-1") for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)
print(f"Vocab saved to {vocab_path}")

# Serialize merges: list of [token1, token2] pairs
merges_path = OUT_DIR / "merges.json"
with open(merges_path, "w") as f:
    json.dump([[a.decode("latin-1"), b.decode("latin-1")] for a, b in merges], f, ensure_ascii=False, indent=2)
print(f"Merges saved to {merges_path}")

# Report the longest token
longest = max(vocab.values(), key=len)
print(f"\nLongest token ({len(longest)} bytes): {longest}")
