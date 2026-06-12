import os
import re
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def _pretokenize_chunk(args: tuple) -> dict[tuple[bytes, ...], int]:
    file_path, start, end, special_pattern = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    parts = re.split(special_pattern, chunk) if special_pattern else [chunk]
    word_counts: dict[tuple[bytes, ...], int] = {}
    for part in parts:
        for match in regex.finditer(PAT, part):
            token = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            word_counts[token] = word_counts.get(token, 0) + 1
    return word_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Vocabulary: IDs 0-255 are initial bytes, then special tokens, then merges
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for idx, token in enumerate(special_tokens):
        vocab[256 + idx] = token.encode("utf-8")

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # Chunk file at special-token boundaries for parallel pre-tokenization
    split_token = (special_tokens[0] if special_tokens else "<|endoftext|>").encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    special_pattern: str | None = (
        "|".join(re.escape(t) for t in special_tokens) if special_tokens else None
    )
    chunk_args = [
        (str(input_path), start, end, special_pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_processes) as pool:
        chunk_results = pool.map(_pretokenize_chunk, chunk_args)

    # Merge per-chunk counts
    word_counts: dict[tuple[bytes, ...], int] = {}
    for result in chunk_results:
        for word, count in result.items():
            word_counts[word] = word_counts.get(word, 0) + count

    # Build pair frequency index
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    # Maps each pair to the set of words (as tuples) that contain it
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    merges: list[tuple[bytes, bytes]] = []
    next_id = max(vocab) + 1

    for _ in tqdm(range(num_merges), desc="Merges", unit="merge"):
        if not pair_counts:
            break

        # Max by (count, pair) — breaks ties by lexicographically greater pair
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] <= 0:
            break

        merged = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[next_id] = merged
        next_id += 1

        words_to_update = list(pair_to_words.pop(best_pair, set()))
        del pair_counts[best_pair]

        for word in words_to_update:
            count = word_counts.pop(word)

            # Build new word by replacing every occurrence of best_pair with merged
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)

            # Decrement counts for all pairs in the old word
            for j in range(len(word) - 1):
                old_pair = (word[j], word[j + 1])
                if old_pair in pair_counts:
                    pair_counts[old_pair] -= count
                    if pair_counts[old_pair] == 0:
                        del pair_counts[old_pair]
                        pair_to_words.pop(old_pair, None)
                    else:
                        pair_to_words[old_pair].discard(word)

            # Increment counts for all pairs in the new word
            for j in range(len(new_word_tuple) - 1):
                new_pair = (new_word_tuple[j], new_word_tuple[j + 1])
                pair_counts[new_pair] += count
                pair_to_words[new_pair].add(new_word_tuple)

            word_counts[new_word_tuple] = word_counts.get(new_word_tuple, 0) + count

    return vocab, merges


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    import multiprocessing

    INPUT_PATH = Path("data/owt_train.txt")
    VOCAB_SIZE = 32000
    SPECIAL_TOKENS = ["<|endoftext|>"]
    OUT_DIR = Path("profile/owt_bpe_output")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NUM_PROCESSES = multiprocessing.cpu_count()

    print(f"Input:      {INPUT_PATH.resolve()} ({INPUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Output dir: {OUT_DIR.resolve()}")
    print(f"Processes:  {NUM_PROCESSES}")
    print()

    start = time.time()
    vocab, merges = train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS, num_processes=NUM_PROCESSES)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min). Vocab: {len(vocab)}, Merges: {len(merges)}")

    vocab_path = OUT_DIR / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({k: v.decode("latin-1") for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)
    print(f"Vocab saved to {vocab_path}")

    merges_path = OUT_DIR / "merges.json"
    with open(merges_path, "w") as f:
        json.dump([[a.decode("latin-1"), b.decode("latin-1")] for a, b in merges], f, ensure_ascii=False, indent=2)
    print(f"Merges saved to {merges_path}")

    longest = max(vocab.values(), key=len)
    print(f"\nLongest token ({len(longest)} bytes): {longest}")
