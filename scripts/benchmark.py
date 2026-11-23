# After Netra-OCR scripts/benchmark.py (Netra Lab kit).
"""Time transcription, stage by stage, as a real-time factor.

    python scripts/benchmark.py                                   # karna-base on tests/golden/audio
    python scripts/benchmark.py --model karna-large --device cuda --repeat 3
    python scripts/benchmark.py --inputs path/to/clips --batch-size 1

Prints seconds per minute of audio for each stage: decoding the files (load), normalising and
padding (prep), the ONNX graph (graph) and greedy CTC decoding (decode). The first pass warms the
model up and isn't counted. Compare runs on the same machine, threads and inputs; see
playbooks/INFERENCE.md §6 in the Netra Lab kit.
"""

import argparse
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

INPUTS = Path(__file__).resolve().parents[1] / "tests" / "golden" / "audio"
AUDIO = {".wav", ".flac", ".ogg", ".mp3", ".opus"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--model", default="karna-base")
    parser.add_argument("--device", help="cpu, cuda, ... (default: KARNA_DEVICE, else auto)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3, help="passes over the inputs")
    args = parser.parse_args()
    if args.device:
        os.environ["KARNA_DEVICE"] = args.device

    import karna
    from karna.audio import load_audio

    start = time.perf_counter()
    asr = karna.load_model(args.model)
    load_s = time.perf_counter() - start
    rec = asr.recognizer
    files = sorted(p for p in args.inputs.rglob("*") if p.suffix.lower() in AUDIO)
    asr.transcribe([str(p) for p in files], batch_size=args.batch_size)  # warm-up

    spent = defaultdict(float)

    def timed(name, fn):
        def wrapper(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                spent[name] += time.perf_counter() - t
        return wrapper

    rec.session.run = timed("graph", rec.session.run)
    rec.normalize = timed("prep", rec.normalize)
    rec.vocab.greedy = timed("decode", rec.vocab.greedy)

    audio_s = 0.0
    total = time.perf_counter()
    for _ in range(args.repeat):
        t = time.perf_counter()
        wavs = [load_audio(p, rec.sample_rate) for p in files]
        spent["load"] += time.perf_counter() - t
        audio_s += sum(len(w) for w in wavs) / rec.sample_rate
        rec.transcribe(wavs, batch_size=args.batch_size)
    total = time.perf_counter() - total

    per_min = 60.0 / audio_s
    print(f"{args.model} on {rec.device} ({platform.processor() or platform.machine()}, {os.cpu_count()} threads), "
          f"{len(files)} files x {args.repeat}, {audio_s / args.repeat:.1f} s of audio per pass, batch {args.batch_size}")
    print(f"model load {load_s:.2f} s")
    print(f"{'stage':<8}{'s per min of audio':>20}")
    for name in ("load", "prep", "graph", "decode"):
        print(f"{name:<8}{spent[name] * per_min:>20.3f}")
    print(f"{'other':<8}{(total - sum(spent.values())) * per_min:>20.3f}")
    print(f"{'total':<8}{total * per_min:>20.3f}   (real-time factor {total / audio_s:.3f})")


if __name__ == "__main__":
    main()
