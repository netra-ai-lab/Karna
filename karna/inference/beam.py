"""CTC prefix beam search with a character language model (shallow fusion), in numpy.

Input is one utterance's ``sub_logp`` (``bias.py``): [T, 1 + K] log-probabilities over the blank
(column 0) and the alphabet. A prefix is scored as

    ln p_ctc(prefix) + lm_weight * ln p_lm(prefix) + sum of insertion_bonus over its characters

and the LM's end-of-sentence probability is added once at the end. Frames where the blank's
probability is above ``blank_skip`` only move every prefix to "ends in blank" (CTC output is
mostly blank, so this skips most frames), and each frame only extends prefixes with its
``top_k`` likeliest characters above ``token_min`` (log-probability).
"""

from __future__ import annotations

import math

import numpy as np

from .lm import CharLM

NEG = -1e30


def _lse(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b <= NEG:
        return a
    return a + math.log1p(math.exp(b - a))


class _Beam:
    __slots__ = ("pb", "pnb", "lm", "state", "last", "bonus")

    def __init__(self, pb: float, pnb: float, lm: float, state, last: int, bonus: float = 0.0) -> None:
        self.pb, self.pnb, self.lm, self.state, self.last, self.bonus = pb, pnb, lm, state, last, bonus


def beam_search(logp: np.ndarray, lm: CharLM | None = None, symbols: np.ndarray | None = None,
                beam: int = 8, lm_weight: float = 0.5, insertion_bonus=0.0, top_k: int = 8,
                token_min: float = -10.0, blank_skip: float = 0.999) -> list[int]:
    """The best column sequence (1..K, collapsed, blanks removed) for ``logp`` [T, 1 + K].

    ``symbols`` maps each column to its LM symbol (``lm.symbol_map``); columns the LM does not
    know get no LM score. With ``lm=None`` this is plain CTC beam search. ``insertion_bonus`` is
    one number or one per column (Khmer is written with more characters per second of speech
    than English, so it takes a larger bonus).
    """
    logp = np.asarray(logp, np.float32)
    bonus_of = (np.broadcast_to(np.asarray(insertion_bonus, np.float64), (logp.shape[1],)).tolist()
                if np.ndim(insertion_bonus) else [float(insertion_bonus)] * logp.shape[1])
    skip = math.log(blank_skip)
    use_lm = lm is not None and lm_weight != 0
    beams: dict[tuple, _Beam] = {(): _Beam(0.0, NEG, 0.0, lm.start() if use_lm else None, -1)}

    def score(prefix: tuple, b: _Beam) -> float:
        return _lse(b.pb, b.pnb) + lm_weight * b.lm + b.bonus

    for t in range(logp.shape[0]):
        row = logp[t]
        blank = float(row[0])
        if blank > skip:
            for b in beams.values():
                b.pb, b.pnb = _lse(b.pb, b.pnb) + blank, NEG
            continue
        cand = np.argpartition(-row[1:], min(top_k, row.size - 2))[:top_k] + 1
        cand = [int(c) for c in cand if row[c] > token_min]
        nxt: dict[tuple, _Beam] = {}

        def add(prefix, pb, pnb, parent: _Beam, last: int, lm_score: float, state, bonus: float) -> None:
            b = nxt.get(prefix)
            if b is None:
                nxt[prefix] = _Beam(pb, pnb, lm_score, state, last, bonus)
            else:
                b.pb, b.pnb = _lse(b.pb, pb), _lse(b.pnb, pnb)

        for prefix, b in beams.items():
            total = _lse(b.pb, b.pnb)
            # stay: a blank, or the last character held for another frame
            stay_nb = b.pnb + float(row[b.last]) if b.last > 0 else NEG
            add(prefix, total + blank, stay_nb, b, b.last, b.lm, b.state, b.bonus)
            lm_row = lm.scores(b.state) if use_lm else None
            for c in cand:
                p = float(row[c])
                # the same character again only counts as new after a blank
                new_nb = (b.pb + p) if c == b.last else (total + p)
                if new_nb <= NEG:
                    continue
                if use_lm:
                    s = int(symbols[c]) if symbols is not None else c
                    lm_score = b.lm + (float(lm_row[s]) if s >= 0 else 0.0)
                    state = lm.advance(b.state, s) if s >= 0 else b.state
                else:
                    lm_score, state = 0.0, None
                add(prefix + (c,), NEG, new_nb, b, c, lm_score, state, b.bonus + bonus_of[c])
        if len(nxt) > beam:
            ranked = sorted(nxt.items(), key=lambda kv: score(kv[0], kv[1]), reverse=True)[:beam]
            nxt = dict(ranked)
        beams = nxt

    def final(item) -> float:
        prefix, b = item
        end = lm_weight * float(lm.scores(b.state)[0]) if use_lm else 0.0
        return score(prefix, b) + end

    best = max(beams.items(), key=final)[0]
    return list(best)
