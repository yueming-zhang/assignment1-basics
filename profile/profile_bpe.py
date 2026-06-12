"""Profile BPE training. Run with: uv run python profile_bpe.py [corpus|tinystories]"""
import cProfile
import pstats
import sys
from pathlib import Path

FIXTURES = Path("tests/fixtures")

CONFIGS = {
    "corpus": (FIXTURES / "corpus.en", 500),
    "tinystories": (FIXTURES / "tinystories_sample_5M.txt", 1000),
}

target = sys.argv[1] if len(sys.argv) > 1 else "corpus"
if target not in CONFIGS:
    print(f"Usage: uv run python profile_bpe.py [{' | '.join(CONFIGS)}]")
    sys.exit(1)

input_path, vocab_size = CONFIGS[target]
print(f"Profiling train_bpe on {input_path} (vocab_size={vocab_size}) ...")

from tests.adapters import run_train_bpe

pr = cProfile.Profile()
pr.enable()
run_train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
pr.disable()

stats = pstats.Stats(pr)
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_stats(20)
