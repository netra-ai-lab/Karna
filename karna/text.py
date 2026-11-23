"""Text side: the Omnilingual char vocabulary, normalisation, CTC targets and greedy decoding.

``omniASR_tokenizer`` is a character-level SentencePiece model (9,812 entries: Latin,
Khmer and ~1,600 other languages' characters). It is case-sensitive with lowercase
entries only, maps a space to its own token, and has no entry for Khmer punctuation
(``។``, ``៕``) or zero-width characters, which it turns into ``<unk>``. Omnilingual's
own training pipeline drops ``<unk>`` from targets, so this module does the same.

The vocabulary was read once from fairseq2 (``vocab.json``) and is kept as a plain ``{char: id}`` table:
a dict lookup per character is exactly what the char model does, and it is safe to
use in forked DataLoader workers.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

BLANK = 0  # fairseq2's ctc_loss uses the default blank, index 0 (<s>)
SPECIAL = {0, 1, 2, 3}  # <s>, <pad>, </s>, <unk>
UNK = 3
SPACE_PIECE = "▁"
#: ``omniASR_tokenizer``'s pieces in id order, exported once with ``karna_train.convert.vocab_from_fairseq2``
VOCAB_PATH = Path(__file__).with_name("vocab.json")

_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str, lang: str) -> str:
    """NFC, zero-width characters removed, whitespace collapsed; English lowercased."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    if lang == "en":
        text = text.lower()
    return _SPACES.sub(" ", text).strip()


class CharVocab:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces
        self.index = {}
        for i, p in enumerate(pieces):
            if i in SPECIAL:
                continue
            self.index.setdefault(" " if p == SPACE_PIECE else p, i)

    @classmethod
    def default(cls) -> CharVocab:
        """The Omnilingual vocabulary shipped with this package (``karna/vocab.json``)."""
        return cls.load(VOCAB_PATH)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.pieces, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CharVocab:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self.pieces)

    def encode(self, text: str) -> tuple[list[int], int]:
        """Token ids for normalised ``text`` and the number of characters dropped as unknown."""
        ids, unk = [], 0
        for ch in text:
            i = self.index.get(ch)
            if i is None:
                unk += 1
            else:
                ids.append(i)
        # dropping a character can leave double or edge spaces
        out: list[int] = []
        sp = self.index[" "]
        for i in ids:
            if i == sp and (not out or out[-1] == sp):
                continue
            out.append(i)
        while out and out[-1] == sp:
            out.pop()
        return out, unk

    def filter(self, text: str) -> str:
        """``text`` with the characters the model cannot emit removed (reference side of CER)."""
        return self.decode(self.encode(text)[0])

    def decode(self, ids: list[int]) -> str:
        return "".join(" " if self.pieces[i] == SPACE_PIECE else self.pieces[i] for i in ids if i not in SPECIAL)

    def greedy(self, frame_ids: list[int]) -> str:
        """CTC greedy decode of one utterance's per-frame argmax ids."""
        out, prev = [], None
        for i in frame_ids:
            if i != prev and i not in SPECIAL:
                out.append(i)
            prev = i
        return self.decode(out)


# ---- clean-up of Khmer output (``Karna.transcribe`` and live transcription apply it by default).
# On the golden km clips, live, it took karna-large's CER from 0.195 to 0.179 in auto mode, and it
# removes every unwritable sequence; it changed nothing else measured (scripts/stream_eval.py).

_K_SIGN = "ា-៑៓៝"          # dependent vowels and signs: they follow a consonant
_K_VOWEL = "ា-ៅ"                     # dependent vowels: one per syllable
_REPEAT_SIGN = re.compile(rf"([{_K_SIGN}])\1+")
_VOWEL_RUN = re.compile(f"([{_K_VOWEL}])[{_K_VOWEL}]+")
_LOOSE_COENG = re.compile("្(?![ក-អ])")
_LEADING_SIGN = re.compile(rf"(^|\s)[{_K_SIGN}្]+")
_SPLIT_VOWELS = (("េា", "ោ"), ("េី", "ើ"))
_KHMER = re.compile("[ក-៿᧠-᧿]")


def clean_khmer(text: str) -> str:
    """Repair what greedy CTC leaves in Khmer: sequences that can't be written, shown as ◌.

    - a vowel or sign repeated back to back (``ាា``) is kept once;
    - e + aa (or e + ii) written as two vowels becomes the one vowel they spell (``ោ``, ``ើ``);
    - of two dependent vowels in a row, the first is kept (a syllable has one);
    - a coeng not followed by a consonant, and vowels or signs at the start of a word, are dropped;
    - runs of spaces become one.
    Text without Khmer is returned with only its spaces normalised.
    """
    if _KHMER.search(text):
        text = _REPEAT_SIGN.sub(r"\1", text)
        for split, whole in _SPLIT_VOWELS:       # e + aa written as two vowels: the one vowel they make
            text = text.replace(split, whole)
        text = _VOWEL_RUN.sub(r"\1", text)
        text = _LOOSE_COENG.sub("", text)
        text = _LEADING_SIGN.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()
