from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import regex


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Return byte offsets that split a file into <= desired_num_chunks pieces,
    each boundary landing at the start of a ``split_special_token`` occurrence.

    Splitting on the special token guarantees no pre-token (and no multi-byte
    UTF-8 char, since the token is ASCII) straddles a boundary, so encoding the
    chunks independently and concatenating the ids equals encoding the whole.

    Unlike the assignment's example helper, the search carries the trailing
    ``len(token)-1`` bytes across reads so a token spanning a read boundary is
    still found.
    """
    assert isinstance(split_special_token, bytes)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size == 0 or desired_num_chunks <= 1:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    tok_len = len(split_special_token)
    mini = 4096
    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        carry = b""
        while True:
            buf = file.read(mini)
            if not buf:
                boundaries[bi] = file_size
                break
            hay = carry + buf
            found = hay.find(split_special_token)
            if found != -1:
                boundaries[bi] = pos - len(carry) + found
                break
            pos += len(buf)
            carry = hay[-(tok_len - 1):] if tok_len > 1 else b""

    return sorted(set(boundaries))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = list(special_tokens) if special_tokens else []

        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Append any special tokens not already in vocab
        for token_str in self.special_tokens:
            token_bytes = token_str.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id

        # merge rank: (bytes, bytes) -> index in merges list
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        self._special_tokens_set: set[str] = set(self.special_tokens)

        # Memoize BPE result per pre-token. Correct because _encode_pretoken is a
        # pure function of the (fixed) merges/vocab, and natural text is Zipfian so
        # a handful of pre-tokens dominate the call count.
        self._pretoken_cache: dict[bytes, list[int]] = {}

        if self.special_tokens:
            # Sort longest first so overlapping tokens are greedily matched
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern: re.Pattern[str] | None = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
            )
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath) as f:
            raw_vocab = json.load(f)
        vocab = {int(k): v.encode("latin-1") for k, v in raw_vocab.items()}

        with open(merges_filepath) as f:
            raw_merges = json.load(f)
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken_bytes: bytes) -> list[int]:
        """Return cached BPE ids for a pre-token, computing on a cache miss."""
        cached = self._pretoken_cache.get(pretoken_bytes)
        if cached is not None:
            return cached
        ids = self._bpe_pretoken(pretoken_bytes)
        self._pretoken_cache[pretoken_bytes] = ids
        return ids

    def _bpe_pretoken(self, pretoken_bytes: bytes) -> list[int]:
        """Apply BPE merges to a single pre-token by repeatedly merging the
        lowest-rank adjacent pair.

        O(n^2) but with a tiny constant: no heap, linked list, or alive array.
        Real pre-tokens are short (~99% are <=12 bytes, max ~142), so the low
        constant beats the heap's O(n log n) object churn on every realistic input.
        """
        if not pretoken_bytes:
            return []

        tokens = [bytes([b]) for b in pretoken_bytes]
        merge_rank = self.merge_rank

        while len(tokens) >= 2:
            # Find the adjacent pair with the lowest (earliest-learned) merge rank.
            best_rank = None
            best_i = -1
            for i in range(len(tokens) - 1):
                rank = merge_rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i

            if best_i < 0:
                break  # no remaining adjacent pair is mergeable

            tokens[best_i : best_i + 2] = [tokens[best_i] + tokens[best_i + 1]]

        bytes_to_id = self.bytes_to_id
        return [bytes_to_id[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs, handling special tokens atomically."""
        if not text:
            return []

        ids: list[int] = []

        parts = self._special_pattern.split(text) if self._special_pattern else [text]

        for part in parts:
            if not part:
                continue
            if part in self._special_tokens_set:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                for match in regex.finditer(PAT, part):
                    ids.extend(self._encode_pretoken(match.group().encode("utf-8")))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings, yielding token IDs one at a time."""
        for text in iterable:
            yield from self.encode(text)

    def encode_file(
        self,
        src: str | Path,
        dst: str | Path,
        num_processes: int | None = None,
        chunk_bytes: int = 16 * 1024 * 1024,
        dtype=np.uint16,
        split_special_token: bytes = b"<|endoftext|>",
    ) -> Path:
        """Encode a whole text file to a uint16 ``.npy`` array using a process pool.

        The file is cut into many ~``chunk_bytes`` pieces at ``split_special_token``
        boundaries; ``num_processes`` workers each tokenize a piece and write a
        uint16 ``.npy`` shard. The main process then concatenates the shards into
        ``dst`` through a memmap, so peak RAM stays ~one chunk rather than the whole
        corpus. Returns ``dst``.

        Correctness: because every boundary sits at a special-token start, the
        concatenated shard ids equal ``encode`` of the entire file.
        """
        src = str(src)
        dst = Path(dst)
        if num_processes is None:
            num_processes = os.cpu_count() or 1

        file_size = os.path.getsize(src)
        desired = max(num_processes, math.ceil(file_size / chunk_bytes)) if file_size else 1
        with open(src, "rb") as f:
            boundaries = find_chunk_boundaries(f, desired, split_special_token)

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmpdir = tempfile.mkdtemp(prefix="enc_shards_", dir=str(dst.parent))
        try:
            tasks = [
                (i, s, e, os.path.join(tmpdir, f"shard_{i:06d}.npy"))
                for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))
                if e > s
            ]

            if num_processes <= 1 or len(tasks) <= 1:
                # Avoid pool overhead for tiny inputs / single core.
                results = [
                    (i, path, _encode_range_to_shard(self, src, s, e, path, np.dtype(dtype)))
                    for (i, s, e, path) in tasks
                ]
            else:
                initargs = (
                    self.vocab,
                    self.merges,
                    self.special_tokens,
                    src,
                    np.dtype(dtype).str,
                )
                with mp.Pool(num_processes, initializer=_pool_init, initargs=initargs) as pool:
                    results = pool.map(_pool_worker, tasks)

            results.sort(key=lambda r: r[0])  # restore document order

            total = sum(n for _, _, n in results)
            out = np.lib.format.open_memmap(str(dst), mode="w+", dtype=dtype, shape=(total,))
            off = 0
            for _idx, shard_path, n in results:
                if n:
                    out[off : off + n] = np.load(shard_path, mmap_mode="r")
                    off += n
            out.flush()
            del out
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return dst

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")


def _encode_range_to_shard(
    tokenizer: Tokenizer, src: str, start: int, end: int, shard_path: str, dtype
) -> int:
    """Encode bytes [start, end) of ``src`` and save them as a uint16 shard.

    Returns the token count. ``errors='ignore'`` is defensive only — boundaries
    sit on ASCII special-token starts, so no multi-byte char is ever split.
    """
    with open(src, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    ids = tokenizer.encode(raw.decode("utf-8", errors="ignore"))
    arr = np.asarray(ids, dtype=dtype)
    np.save(shard_path, arr)
    return arr.shape[0]


# Per-worker state, populated once by the Pool initializer to avoid re-pickling
# the (large) vocab/merges on every task.
_WORKER_STATE: dict = {}


def _pool_init(vocab, merges, special_tokens, src, dtype_str) -> None:
    _WORKER_STATE["tokenizer"] = Tokenizer(vocab, merges, special_tokens)
    _WORKER_STATE["src"] = src
    _WORKER_STATE["dtype"] = np.dtype(dtype_str)


def _pool_worker(task):
    idx, start, end, shard_path = task
    n = _encode_range_to_shard(
        _WORKER_STATE["tokenizer"], _WORKER_STATE["src"], start, end, shard_path, _WORKER_STATE["dtype"]
    )
    return idx, shard_path, n
