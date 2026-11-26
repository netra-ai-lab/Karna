"""A character n-gram language model in numpy (no compiled dependency), for CTC beam search.

The model is a backoff n-gram over the characters a Karna model can emit in ``auto`` mode
(``bias.lm_alphabet``: Khmer, ASCII and the space), built by ``scripts/build_char_lm.py`` with
interpolated modified Kneser-Ney and count pruning. Character level fits Khmer, which is
written without spaces between words.

Symbols: 0 is the sentence boundary (``<s>`` as history, ``</s>`` as a prediction), and
1..K are the alphabet's characters. An n-gram of up to 8 symbols is one uint64 key, its
symbols one byte each, oldest first. File (``.npz``):

    keys{k}     uint64 [M_k]  sorted keys of the k-grams kept
    logp{k}     float16 [M_k] ln p(last symbol | the k-1 before it)
    bow{k}      float16 [M_k] ln backoff weight of the k-gram as a history (0 if it has none)
    meta        JSON: order, alphabet (the K characters, in symbol order), sources, build settings

A query backs off as in an ARPA file: p(w | h) is the longest stored h'w (h' a suffix of h),
times the backoff weights of the longer histories skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BOUNDARY = 0


class CharLM:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path) as data:
            self.meta = json.loads(str(data["meta"]))
            self.order = int(self.meta["order"])
            self.keys = [None] + [data[f"keys{k}"].astype(np.uint64) for k in range(1, self.order + 1)]
            self.logp = [None] + [data[f"logp{k}"].astype(np.float32) for k in range(1, self.order + 1)]
            self.bow = [None] + [data[f"bow{k}"].astype(np.float32) for k in range(1, self.order + 1)]
        self.alphabet: list[str] = list(self.meta["alphabet"])
        self.size = len(self.alphabet) + 1  # symbols, with the boundary
        self._symbols = np.arange(self.size, dtype=np.uint64)
        self._mask = (1 << (8 * (self.order - 1))) - 1
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    # ---- states: the last (order - 1) symbols as one int, and how many of them are real

    def start(self) -> tuple[int, int]:
        """The state at the start of a sentence (after ``<s>``)."""
        return (BOUNDARY, 1)

    def advance(self, state: tuple[int, int], symbol: int) -> tuple[int, int]:
        key, length = state
        return (((key << 8) | symbol) & self._mask, min(length + 1, self.order - 1))

    # ---- scores

    def _find(self, k: int, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keys = self.keys[k]
        idx = np.searchsorted(keys, query)
        idx = np.minimum(idx, len(keys) - 1)
        return idx, keys[idx] == query

    def scores(self, state: tuple[int, int]) -> np.ndarray:
        """ln p(s | state) for every symbol s (index 0 is ``</s>``), float32 [size]. Cached."""
        cached = self._cache.get(state)
        if cached is not None:
            return cached
        key, length = state
        out = np.zeros(self.size, np.float32)
        found = np.zeros(self.size, bool)
        backoff = 0.0
        for k in range(min(length + 1, self.order), 0, -1):
            ctx = key & ((1 << (8 * (k - 1))) - 1)
            idx, hit = self._find(k, (np.uint64(ctx) << np.uint64(8)) | self._symbols)
            new = hit & ~found
            out[new] = self.logp[k][idx[new]] + backoff
            found |= new
            if found.all() or k == 1:
                break
            # the history ctx (k-1 symbols) was not enough for the rest: back off past it
            hidx, hhit = self._find(k - 1, np.array([ctx], np.uint64))
            if hhit[0]:
                backoff += float(self.bow[k - 1][hidx[0]])
        out[~found] = -30.0  # a symbol never seen at all (the unigrams cover every symbol)
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[state] = out
        return out

    def score_text(self, symbols: list[int]) -> float:
        """ln p of a sentence's symbols (1..K), ``</s>`` included."""
        state, total = self.start(), 0.0
        for s in [*symbols, BOUNDARY]:
            total += float(self.scores(state)[s])
            state = self.advance(state, s)
        return total


def symbol_map(lm: CharLM, pieces: list[str], alphabet_ids: list[int]) -> np.ndarray:
    """LM symbol for each column of ``sub_logp`` (column 0, the blank, maps to -1). The LM's
    alphabet is matched by character, so an LM built from another vocabulary order still works."""
    from ..text import SPACE_PIECE

    index = {ch: i + 1 for i, ch in enumerate(lm.alphabet)}
    out = np.full(len(alphabet_ids) + 1, -1, np.int64)
    for col, vid in enumerate(alphabet_ids, start=1):
        ch = " " if pieces[vid] == SPACE_PIECE else pieces[vid]
        out[col] = index.get(ch, -1)
    return out
