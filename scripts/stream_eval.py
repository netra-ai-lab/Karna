"""Measure live (streamed) transcription against offline, for tuning ``karna/streaming.py``.

Streams audio in 100 ms chunks through the same Endpointer the server uses and decodes each
closed segment as a final, with and without left context (the end of the previous segment
prepended, its frames dropped from the output). Two inputs:

1. The Khmer golden clips stitched with 1.5 s pauses (plus -90 dBFS hiss), at three loudness levels: CER against
   tests/golden/references.tsv, and the same for offline decoding of each clip.
2. Any extra recordings given with --audio (no reference): the share of Latin letters in the
   output, which for Khmer speech is all error, and CER against --pseudo-model offline on 20 s windows.

    KARNA_BUNDLES=training/runs/onnx python scripts/stream_eval.py --audio live.wav
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import karna
from karna.streaming import Context, Endpointer, pad
from karna.text import clean_khmer

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
RATE = 16000
SILENCE = np.zeros(RATE // 2, np.float32)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def cer(hyp: str, ref: str) -> float:
    return edit_distance(norm(hyp), norm(ref)) / max(1, len(norm(ref)))


def latin_share(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(c.isascii() and c.isalpha() for c in chars) / max(1, len(chars))


def segments(audio: np.ndarray, chunk: int = RATE // 10):
    ep = Endpointer(sample_rate=RATE)
    out = []
    for i in range(0, len(audio), chunk):
        out += ep.push(audio[i:i + chunk])
    return [s for s in out + ep.flush() if s.keep]


INVALID = re.compile("(^|\s)[ា-៓៝]|([ា-ៅ])[ា-ៅ]|្(?![ក-អ])|([ា-៑])\3")


def invalid_rate(text: str) -> float:
    """Khmer sequences that can't be written (shown as a dotted circle), per 100 characters."""
    return 100 * len(INVALID.findall(text)) / max(1, len(text))


def stream(rec, audio: np.ndarray, context_s: float = 1.0, language: str = "auto",
           clean: bool = False) -> str:
    """Finals only, as the server decodes them (streaming.Context + Recognizer.transcribe_tail)."""
    texts, context = [], Context(int(context_s * RATE))
    for seg in segments(audio):
        piece = seg.audio
        if context.audio.size:
            wav, skip = context.wrap(piece, SILENCE)
            text = rec.transcribe_tail(wav, skip, language=language)
        else:
            text = rec.transcribe([pad(piece, SILENCE)], language=language)[0]
        texts.append(clean_khmer(text) if clean else text)
        context.update(seg)
    return " ".join(t for t in texts if t)


#: name: (language, clean). An 80 Hz high-pass before decoding was also tried (2026-11-23): +0.9 CER
#: points for karna-large on the capture, -0.3 for karna-base, golden unchanged; not kept.
SETTINGS = {
    "auto": ("auto", False),
    "auto+clean": ("auto", True),
    "km": ("km", False),
    "km+clean": ("km", True),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["karna-base", "karna-large"])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--audio", nargs="*", default=[], help="extra Khmer recordings without a reference")
    ap.add_argument("--pseudo-model", default="karna-large")
    args = ap.parse_args()

    refs = dict(line.split("	", 1) for line in (GOLDEN / "references.tsv").read_text("utf-8").splitlines() if line.strip())
    km = sorted(k for k in refs if k.startswith("fleurs-km"))
    clips = [sf.read(GOLDEN / "audio" / k, dtype="float32")[0] for k in km]
    gap = np.zeros(int(1.5 * RATE), np.float32)
    ref_all = " ".join(refs[k].strip() for k in km)
    extra = {Path(p).name: sf.read(p, dtype="float32")[0] for p in args.audio}
    hiss = lambda x: x + np.random.default_rng(0).normal(0, 3e-5, len(x)).astype(np.float32)  # noqa: E731

    recs = {m: karna.load_model(m).recognizer for m in set(args.models) | ({args.pseudo_model} if extra else set())}
    pseudo = {}
    for name, audio in extra.items():
        rec = recs[args.pseudo_model]
        pseudo[name] = " ".join(rec.transcribe([audio[i:i + 20 * RATE]], language="km")[0]
                                for i in range(0, len(audio), 20 * RATE))

    cols = ["golden km ×1 CER", "golden km ×3 CER"] + [f"{n}: Latin / invalid per 100 / CER vs pseudo" for n in extra]
    print("| model | setting | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for m in args.models:
        rec = recs[m]
        for name in args.settings:
            language, clean = SETTINGS[name]
            row = []
            for gain in (1.0, 3.0):
                stitched = hiss(np.concatenate([x for c in clips for x in (gap, np.clip(c * gain, -1, 1))] + [gap]))
                row.append(f"{cer(stream(rec, stitched, language=language, clean=clean), ref_all):.3f}")
            for n, audio in extra.items():
                out = stream(rec, audio, language=language, clean=clean)
                row.append(f"{latin_share(out):.3f} / {invalid_rate(out):.2f} / {cer(out, pseudo[n]):.3f}")
            print(f"| {m} | {name} | " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
