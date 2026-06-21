import heapq
import os
import re
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex
from tqdm import tqdm


class _PairKey:
    """Max-heap entry: higher count wins; on a tie the lexicographically greater
    byte-pair wins. Pairs are stored as integer ids (for fast lookup/validation),
    plus the byte-pair the ids map to (only used for the rare tie-break compare)."""

    __slots__ = ("neg_count", "ids", "bytes_pair")

    def __init__(self, count: int, ids: tuple[int, int], bytes_pair: tuple[bytes, bytes]) -> None:
        self.neg_count = -count
        self.ids = ids
        self.bytes_pair = bytes_pair

    def __lt__(self, other: "_PairKey") -> bool:
        if self.neg_count != other.neg_count:
            return self.neg_count < other.neg_count
        return self.bytes_pair > other.bytes_pair  # larger byte-pair wins on tie


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


def _pretokenize_chunk(args: tuple) -> dict[tuple[int, ...], int]:
    file_path, start, end, special_pattern = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    parts = re.split(special_pattern, chunk) if special_pattern else [chunk]
    word_counts: dict[tuple[int, ...], int] = {}
    for part in parts:
        for match in regex.finditer(PAT, part):
            # Each word is a tuple of byte values (ints 0-255), not bytes objects:
            # iterating a bytes object yields ints, so tuple(b"ab") == (97, 98).
            token = tuple(match.group().encode("utf-8"))
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

    # Merge per-chunk counts. Words are tuples of integer byte values (0-255).
    merged_counts: dict[tuple[int, ...], int] = {}
    for result in chunk_results:
        for word, count in result.items():
            merged_counts[word] = merged_counts.get(word, 0) + count

    # Intern each distinct word to a stable integer id. A word's count never
    # changes during training (merges only relabel its symbol sequence), so we
    # keep counts in a parallel list and rewrite words[wid] in place on each merge.
    words: list[tuple[int, ...]] = list(merged_counts.keys())
    word_count: list[int] = list(merged_counts.values())
    del merged_counts

    # Build pair index over integer-id pairs. pair_to_words holds word *ids*
    # (ints), so its set operations never hash full word tuples.
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wid, word in enumerate(words):
        c = word_count[wid]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += c
            pair_to_words[pair].add(wid)

    # Heap key stores the byte-pair (vocab[a], vocab[b]) only for the tie-break compare.
    heap = [_PairKey(count, p, (vocab[p[0]], vocab[p[1]])) for p, count in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    next_id = max(vocab) + 1

    for _ in tqdm(range(num_merges), desc="Merges", unit="merge"):
        # Pop stale entries until we find one whose count still matches pair_counts
        best_pair = None
        while heap:
            entry = heapq.heappop(heap)
            if pair_counts.get(entry.ids, 0) == -entry.neg_count:
                best_pair = entry.ids
                break
        if best_pair is None:
            break

        a, b = best_pair
        merged_id = next_id
        vocab[merged_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))
        next_id += 1

        words_to_update = pair_to_words.pop(best_pair, set())
        del pair_counts[best_pair]

        # Track pairs whose count changed this merge, then push each once at the end.
        # A frequent pair touched by many words would otherwise be pushed many times.
        changed_pairs: set[tuple[int, int]] = set()

        for wid in words_to_update:
            word = words[wid]
            count = word_count[wid]

            # Build new word by replacing every occurrence of (a, b) with merged_id
            new_word: list[int] = []
            i = 0
            n = len(word)
            while i < n:
                if i + 1 < n and word[i] == a and word[i + 1] == b:
                    new_word.append(merged_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)

            # Decrement counts for pairs in the old word (sets hash ints, not tuples)
            for j in range(len(word) - 1):
                old_pair = (word[j], word[j + 1])
                if old_pair in pair_counts:
                    pair_counts[old_pair] -= count
                    if pair_counts[old_pair] == 0:
                        del pair_counts[old_pair]
                        pair_to_words.pop(old_pair, None)
                    else:
                        pair_to_words[old_pair].discard(wid)
                        changed_pairs.add(old_pair)

            # Increment counts for pairs in the new word
            for j in range(len(new_word_tuple) - 1):
                new_pair = (new_word_tuple[j], new_word_tuple[j + 1])
                pair_counts[new_pair] += count
                pair_to_words[new_pair].add(wid)
                changed_pairs.add(new_pair)

            words[wid] = new_word_tuple

        # One push per distinct changed pair (final count), instead of one per update
        for pair in changed_pairs:
            if pair in pair_counts:
                heapq.heappush(heap, _PairKey(pair_counts[pair], pair, (vocab[pair[0]], vocab[pair[1]])))

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

    top10 = sorted(vocab.values(), key=len, reverse=True)[:10]
    print("\nTop 10 longest tokens:")
    for rank, tok in enumerate(top10, 1):
        decoded = tok.decode("utf-8", errors="replace")
        print(f"{rank:2}. {len(tok):3} bytes  {tok!r}  ->  {decoded!r}")
