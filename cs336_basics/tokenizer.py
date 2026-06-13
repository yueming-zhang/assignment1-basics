from __future__ import annotations

import heapq
import json
import re
from collections.abc import Iterable, Iterator

import regex


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
    ) -> "Tokenizer":
        with open(vocab_filepath) as f:
            raw_vocab = json.load(f)
        vocab = {int(k): v.encode("latin-1") for k, v in raw_vocab.items()}

        with open(merges_filepath) as f:
            raw_merges = json.load(f)
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken_bytes: bytes) -> list[int]:
        """Apply BPE merges to a single pre-token using a priority queue."""
        if not pretoken_bytes:
            return []

        tokens = [bytes([b]) for b in pretoken_bytes]
        n = len(tokens)

        if n == 1:
            return [self.bytes_to_id[tokens[0]]]

        # Doubly-linked list via index arrays (no class overhead)
        prev = list(range(-1, n - 1))   # prev[i] = previous live index
        next_ = list(range(1, n + 1))   # next_[i] = next live index; n = sentinel

        alive = [True] * n  # False once a position is absorbed as the RIGHT element of a merge

        heap: list[tuple[int, int]] = []
        for i in range(n - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = self.merge_rank.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        while heap:
            rank, i = heapq.heappop(heap)
            if not alive[i]:
                continue  # position i was absorbed as a right element in a previous merge
            j = next_[i]
            if j >= n:
                continue  # right neighbour already merged away
            pair = (tokens[i], tokens[j])
            if self.merge_rank.get(pair) != rank:
                continue  # stale entry: tokens[i] changed since this was pushed

            # Absorb j into i
            tokens[i] = tokens[i] + tokens[j]
            alive[j] = False
            next_[i] = next_[j]
            if next_[j] < n:
                prev[next_[j]] = i

            # Push new candidate pairs involving the updated token at i
            if prev[i] != -1:
                left = prev[i]
                new_pair = (tokens[left], tokens[i])
                r = self.merge_rank.get(new_pair)
                if r is not None:
                    heapq.heappush(heap, (r, left))

            if next_[i] < n:
                right = next_[i]
                new_pair = (tokens[i], tokens[right])
                r = self.merge_rank.get(new_pair)
                if r is not None:
                    heapq.heappush(heap, (r, i))

        # Walk linked list to collect result
        result: list[int] = []
        i = 0
        while i < n:
            result.append(self.bytes_to_id[tokens[i]])
            i = next_[i]
        return result

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

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")
