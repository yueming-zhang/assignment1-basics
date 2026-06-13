"""Train BPE tokenizer on TinyStories with 10K vocab and save to scripts/ts_bpe_output/.

Usage:
    uv run python scripts/train_ts_tokenizer.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.adapters import run_train_bpe

INPUT_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]
OUT_DIR = Path("scripts/ts_bpe_output")

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

vocab_path = OUT_DIR / "vocab.json"
with open(vocab_path, "w") as f:
    json.dump({k: v.decode("latin-1") for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)
print(f"Vocab saved to {vocab_path}")

merges_path = OUT_DIR / "merges.json"
with open(merges_path, "w") as f:
    json.dump([[a.decode("latin-1"), b.decode("latin-1")] for a, b in merges], f, ensure_ascii=False, indent=2)
print(f"Merges saved to {merges_path}")

longest = max(vocab.values(), key=len)
print(f"Longest token ({len(longest)} bytes): {longest}")
