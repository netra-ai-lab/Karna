"""Real-world proxy benchmark: how Karna does away from its clean, read-speech test sets.

The published numbers (docs/RESULTS.md) are all in-domain: FLEURS, DDD and LibriSpeech are read
speech, recorded clean, and the models were trained on the same sources without augmentation.
This script measures the conditions a microphone brings, with everything held fixed by a seed:

Natural sets (other recordings than the training data):
    slr42_dev / slr42_test   OpenSLR 42 Khmer (crowd-recorded), two halves: tune on dev, report test
    kmcorpus                 a sample of the kmcorpus Khmer test set (spontaneous speech)
    kmcorpus_dev             400 other kmcorpus clips, for tuning
    cv_km                    Common Voice Khmer test (12 clips)
Simulated sets (FLEURS-km test and LibriSpeech test-other, ``<set>@<condition>``):
    clean    as recorded
    noise    MUSAN / ESC-50 background at 5-15 dB SNR
    reverb   a simulated room (RIRS_NOISES) and 25 dB SNR background
    far      a room and 5-10 dB SNR: a laptop across the table
    phone    8 kHz, mu-law, 300 Hz high-pass, 15 dB SNR
Live streams (``--live``): 40 utterances of a simulated set stitched with 0.3-2 s gaps of the
same background, fed in 100 ms chunks through ``Karna.stream()``. Reported: CER/WER of the
finals, how far partials are from their final, decode cost per second of audio, mid-utterance cuts
and speech with no text (in no segment, or in one whose final came back empty).

    python scripts/realworld_eval.py --model karna-base --device cpu
    python scripts/realworld_eval.py --model karna-large --device cuda --live --tag baseline

Results go to training/runs/eval/realworld/<model>-<device>-<tag>.json, and a table is printed.
The decoded and corrupted audio is cached once under $KARNA_EVAL_CACHE.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import karna
from karna.text import CharVocab, normalize_text

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
DISK = Path("/run/media/pc/disk1")
ASR_ROOT = Path(os.environ.get("ASR_ROOT", DISK / "khmer-self-feedback-asr"))
MANIFESTS = ASR_ROOT / "manifests"
AUG = Path(os.environ.get("KARNA_AUG_DIR", DISK / "ml-data/datasets/diarization/train/v1/augmentation"))
CACHE = Path(os.environ.get("KARNA_EVAL_CACHE", DISK / "ml-data/cache/sota-features/realworld"))
OUT = ROOT / "training" / "runs" / "eval" / "realworld"
SEED = 20261126

NATURAL = ("slr42_dev", "slr42_test", "kmcorpus", "cv_km")
#: for tuning (decoder settings, VAD): never reported as results
DEV = ("slr42_dev", "kmcorpus_dev", "fleurs_km_dev@clean", "fleurs_km_dev@noise", "fleurs_km_dev@far",
       "libri_dev@clean", "libri_dev@noise", "libri_dev@far")
CONDITIONS = ("clean", "noise", "reverb", "far", "phone")
SIMULATED = tuple(f"{base}@{c}" for base in ("fleurs_km", "libri_other") for c in CONDITIONS)
SIZES = {"slr42": 400, "kmcorpus": 600, "fleurs_km": 200, "libri_other": 200, "fleurs_km_dev": 150, "libri_dev": 150}


# ---------------------------------------------------------------- metrics

def edit_distance(a, b) -> int:
    try:
        import editdistance

        return editdistance.eval(a, b)
    except ImportError:
        pass
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(refs: list[str], hyps: list[str], langs: list[str]) -> dict:
    """CER over characters without spaces (karna_train.evaluate.score), WER for English, and the
    share of Latin letters in the output for Khmer speech (all of it is error)."""
    ce = cn = we = wn = latin = chars = 0
    for r, h, lang in zip(refs, hyps, langs):
        rc, hc = r.replace(" ", ""), h.replace(" ", "")
        ce += edit_distance(list(rc), list(hc))
        cn += len(rc)
        if lang == "en":
            we += edit_distance(r.split(), h.split())
            wn += len(r.split())
        else:
            latin += sum(c.isascii() and c.isalpha() for c in hc)
            chars += len(hc)
    out = {"cer": ce / max(cn, 1), "n": len(refs)}
    if wn:
        out["wer"] = we / wn
    if chars:
        out["latin"] = latin / chars
    return out


# ---------------------------------------------------------------- reading the sets

def _rows(manifest: str) -> list[dict]:
    return [json.loads(line) for line in open(MANIFESTS / manifest, encoding="utf-8")]


def _parquet_audio(rows: list[dict], shard_key: str, row_key: str) -> list[np.ndarray]:
    import pyarrow.parquet as pq

    from karna.audio import load_audio

    out: dict[int, np.ndarray] = {}
    by_shard: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_shard.setdefault(r[shard_key], []).append(i)
    for shard, idx in by_shard.items():
        audio = pq.read_table(ASR_ROOT / shard, columns=["audio"]).column("audio")
        for i in idx:
            out[i] = load_audio(audio[rows[i][row_key]].as_py()["bytes"], RATE)
    return [out[i] for i in range(len(rows))]


def _sample(rows: list[dict], n: int, tag: str) -> list[dict]:
    rng = np.random.default_rng(int(hashlib.sha1(f"{SEED}-{tag}".encode()).hexdigest()[:8], 16))
    idx = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False).tolist())
    return [rows[i] for i in idx]


def load_base(name: str) -> tuple[list[np.ndarray], list[str], list[str]]:
    """(audio, raw reference text, language) for a natural set or a simulated set's clean source."""
    if name in ("slr42_dev", "slr42_test"):
        rows = _rows("test_slr42.jsonl")
        # split by speaker-free hash of the row, so the halves are fixed and disjoint
        half = [r for r in rows if (int(hashlib.sha1(f"{r['row_index']}".encode()).hexdigest(), 16) % 2 == 0)
                == (name == "slr42_dev")]
        rows = _sample(half, SIZES["slr42"], name)
        return _parquet_audio(rows, "shard_path", "row_index"), [r["transcript"] for r in rows], ["km"] * len(rows)
    if name in ("kmcorpus", "kmcorpus_dev"):
        # kmcorpus has no separate dev split (its dev manifest is a subset of test), so the dev
        # set is 400 other clips of the test manifest, disjoint from the 600 that are reported
        rows = _rows("test_kmcorpus.jsonl")
        test = _sample(rows, SIZES["kmcorpus"], "kmcorpus")
        if name == "kmcorpus_dev":
            used = {(r["shard_path"], r["row_index"]) for r in test}
            rows = _sample([r for r in rows if (r["shard_path"], r["row_index"]) not in used], 400, name)
        else:
            rows = test
        return _parquet_audio(rows, "shard_path", "row_index"), [r["transcript"] for r in rows], ["km"] * len(rows)
    if name == "cv_km":
        rows = _rows("test_cv26_km.jsonl")
        return _parquet_audio(rows, "shard_path", "row_index"), [r["transcript"] for r in rows], ["km"] * len(rows)
    if name == "fleurs_km":
        rows = [json.loads(line) for line in open(ROOT / "training/data/fleurs_km_test.jsonl", encoding="utf-8")]
        rows = _sample(rows, SIZES["fleurs_km"], name)
        return _parquet_audio(rows, "shard", "row"), [r["text"] for r in rows], ["km"] * len(rows)
    if name == "fleurs_km_dev":
        rows = [json.loads(line) for line in open(ROOT / "training/data/fleurs_km_dev.jsonl", encoding="utf-8")]
        rows = _sample(rows, SIZES[name], name)
        return _parquet_audio(rows, "shard", "row"), [r["text"] for r in rows], ["km"] * len(rows)
    if name == "libri_dev":
        rows = [json.loads(line) for line in open(ROOT / "training/data/libri_dev-clean.jsonl", encoding="utf-8")]
        rows = _sample(rows, SIZES[name], name)
        return [sf.read(r["path"], dtype="float32")[0] for r in rows], [r["text"] for r in rows], ["en"] * len(rows)
    if name == "libri_other":
        rows = [json.loads(line) for line in open(ROOT / "training/data/libri_test-other.jsonl", encoding="utf-8")]
        rows = _sample(rows, SIZES["libri_other"], name)
        return [sf.read(r["path"], dtype="float32")[0] for r in rows], [r["text"] for r in rows], ["en"] * len(rows)
    raise KeyError(name)


# ---------------------------------------------------------------- corruption

class Corpora:
    """Background noise and room impulse responses, in a fixed shuffled order."""

    def __init__(self) -> None:
        rng = np.random.default_rng(SEED)
        self.noises = sorted([*(AUG / "musan").glob("*.wav"), *(AUG / "esc50").glob("*.wav")])
        self.rirs = sorted((AUG / "rir").glob("*.wav"))
        rng.shuffle(self.noises)
        rng.shuffle(self.rirs)
        if not self.noises or not self.rirs:
            raise FileNotFoundError(f"no noise / RIR files under {AUG}")

    def noise(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` samples of background, looping a file (or two) from a random offset."""
        while True:
            wav, sr = sf.read(self.noises[int(rng.integers(len(self.noises)))], dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            if sr != RATE:
                from karna.audio import resample

                wav = resample(wav, sr, RATE)
            if wav.size > RATE // 4 and float(np.sqrt(np.mean(wav ** 2))) > 1e-4:
                break
        start = int(rng.integers(len(wav)))
        reps = int(np.ceil((n + start) / len(wav)))
        return np.tile(wav, reps)[start:start + n].astype(np.float32)

    def rir(self, rng: np.random.Generator) -> np.ndarray:
        wav, _ = sf.read(self.rirs[int(rng.integers(len(self.rirs)))], dtype="float32", always_2d=True)
        return wav[:, 0]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))) + 1e-12


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    return x + noise * (rms(x) / rms(noise) / (10 ** (snr_db / 20)))


def convolve(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Reverberant ``x``, aligned on the direct path and at the same RMS."""
    peak = int(np.argmax(np.abs(rir)))
    rir = rir[peak:] / (np.abs(rir[peak]) + 1e-9)
    n = len(x) + len(rir) - 1
    size = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(rir, size), size)[: len(x)]
    return (y * (rms(x) / rms(y))).astype(np.float32)


def highpass(x: np.ndarray, cutoff: float, rate: int = RATE) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    spec *= 1 / (1 + (cutoff / np.maximum(freqs, 1e-3)) ** 4)  # 2nd-order-like roll-off
    return np.fft.irfft(spec, len(x)).astype(np.float32)


def mu_law(x: np.ndarray, mu: int = 255) -> np.ndarray:
    peak = max(float(np.abs(x).max()), 1e-6)
    y = np.sign(x / peak) * np.log1p(mu * np.abs(x / peak)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * mu) / mu * 2 - 1
    return (np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu * peak).astype(np.float32)


def corrupt(x: np.ndarray, condition: str, corpora: Corpora, rng: np.random.Generator,
            noise: np.ndarray | None = None) -> np.ndarray:
    """``x`` under ``condition``; ``noise`` (same length) is drawn when not given."""
    x = x.astype(np.float32)
    if condition == "clean":
        return x
    noise = corpora.noise(rng, len(x)) if noise is None else noise
    if condition == "noise":
        y = add_noise(x, noise, rng.uniform(5, 15))
    elif condition == "reverb":
        y = add_noise(convolve(x, corpora.rir(rng)), noise, 25)
    elif condition == "far":
        y = add_noise(convolve(x, corpora.rir(rng)), noise, rng.uniform(5, 10))
    elif condition == "phone":
        from karna.audio import resample

        y = add_noise(x, noise, 15)
        y = resample(mu_law(highpass(resample(y, RATE, 8000), 300, 8000)), 8000, RATE)[: len(x)]
    else:
        raise ValueError(condition)
    peak = float(np.abs(y).max())
    return (y / peak * 0.9 if peak > 0.99 else y).astype(np.float32)


# ---------------------------------------------------------------- cache

def _save(path: Path, audio: list[np.ndarray], refs: list[str], langs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(a) for a in audio])
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, audio=np.concatenate(audio).astype(np.float32), offsets=offsets,
             meta=np.array(json.dumps({"refs": refs, "langs": langs}, ensure_ascii=False)))
    tmp.replace(path)


def _load(path: Path) -> tuple[list[np.ndarray], list[str], list[str]]:
    with np.load(path) as data:  # each data[key] re-reads the array from disk, so read it once
        audio, off = data["audio"], data["offsets"]
        meta = json.loads(str(data["meta"]))
    return [audio[off[i]:off[i + 1]] for i in range(len(off) - 1)], meta["refs"], meta["langs"]


def get_set(name: str, corpora: Corpora | None = None) -> tuple[list[np.ndarray], list[str], list[str]]:
    path = CACHE / f"{name.replace('@', '__')}.npz"
    if path.exists():
        return _load(path)
    base, _, condition = name.partition("@")
    if condition:
        audio, refs, langs = get_set(base) if (CACHE / f"{base}.npz").exists() else load_base(base)
        if not (CACHE / f"{base}.npz").exists():
            _save(CACHE / f"{base}.npz", audio, refs, langs)
        corpora = corpora or Corpora()
        rng = np.random.default_rng(int(hashlib.sha1(f"{SEED}-{name}".encode()).hexdigest()[:8], 16))
        audio = [corrupt(a, condition, corpora, rng) for a in audio]
    else:
        audio, refs, langs = load_base(name)
    _save(path, audio, refs, langs)
    return audio, refs, langs


# ---------------------------------------------------------------- live streams

def make_stream(name: str, n: int = 40) -> tuple[np.ndarray, list[str], list[tuple[int, int]], str]:
    """``n`` utterances of simulated set ``name`` (``base@condition``) with 0.3-2 s gaps of the same
    background noise. Returns the stream, the references, each utterance's (start, end) sample and
    the language."""
    path = CACHE / "streams" / f"{name.replace('@', '__')}-{n}.npz"
    if path.exists():
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        return data["audio"], meta["refs"], [tuple(s) for s in meta["spans"]], meta["lang"]
    base, _, condition = name.partition("@")
    audio, refs, langs = get_set(base)
    # the same utterances and gaps for every condition of a base set, so conditions compare directly
    pick = np.random.default_rng(int(hashlib.sha1(f"{SEED}-stream-{base}".encode()).hexdigest()[:8], 16))
    idx = sorted(pick.choice(len(audio), size=min(n, len(audio)), replace=False).tolist())
    gaps = [int(pick.uniform(0.3, 2.0) * RATE) for _ in range(len(idx) + 1)]
    rng = np.random.default_rng(int(hashlib.sha1(f"{SEED}-stream-{name}".encode()).hexdigest()[:8], 16))
    total = sum(len(audio[i]) for i in idx) + sum(gaps)
    clean = np.zeros(total, np.float32)
    spans, pos = [], gaps[0]
    for k, i in enumerate(idx):
        clean[pos:pos + len(audio[i])] = audio[i]
        spans.append((pos, pos + len(audio[i])))
        pos += len(audio[i]) + gaps[k + 1]
    corpora = Corpora()
    if condition == "clean":
        out = clean + rng.normal(0, 3e-5, total).astype(np.float32)  # -90 dBFS hiss, as stream_eval
    else:
        # corrupt utterance by utterance (each its own SNR / room), with one continuous background
        noise = corpora.noise(rng, total)
        speech_rms = np.mean([rms(audio[i]) for i in idx])
        out = clean.copy()
        if condition in ("reverb", "far"):  # each utterance in its own room
            for (s, e) in spans:
                out[s:e] = convolve(clean[s:e], corpora.rir(rng))
        snr = {"noise": 10, "reverb": 25, "far": 7.5, "phone": 15}[condition]
        out = out + noise * (speech_rms / rms(noise) / 10 ** (snr / 20))
        if condition == "phone":
            from karna.audio import resample

            out = resample(mu_law(highpass(resample(out, RATE, 8000), 300, 8000)), 8000, RATE)[:total]
        peak = float(np.abs(out).max())
        out = (out / peak * 0.9 if peak > 0.99 else out).astype(np.float32)
    refs_sel = [refs[i] for i in idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, audio=out, meta=np.array(json.dumps({"refs": refs_sel, "spans": spans, "lang": langs[0]},
                                                       ensure_ascii=False)))
    return out, refs_sel, spans, langs[0]


def run_live(asr, name: str, vocab: CharVocab, stream_opts: dict, chunk_s: float = 0.1) -> dict:
    audio, refs, spans, lang = make_stream(name)
    live = asr.stream(**stream_opts)
    timings = {"partial": [], "final": []}
    current = {"kind": "final"}

    def timed(fn):
        def run(*a, **k):
            t0 = time.perf_counter()
            out = fn(*a, **k)
            timings[current["kind"]].append((time.perf_counter() - t0, sum(len(x) for x in a if isinstance(x, np.ndarray))))
            return out
        return run

    for attr in ("_decode", "_decode_context", "_final_decode", "_final_decode_context"):
        if getattr(live, attr, None) is not None:
            setattr(live, attr, timed(getattr(live, attr)))
    partials: dict[int, list[str]] = {}
    finals: list[dict] = []
    chunk = int(chunk_s * RATE)
    for i in range(0, len(audio), chunk):
        # the transcriber decodes partials after pushing; finals happen inside push
        current["kind"] = "final"
        events = []
        ep = live.endpointer
        closed = ep.push(audio[i:i + chunk])
        for seg in closed:
            events += live._final(seg)
        current["kind"] = "partial"
        events += live.feed(np.zeros(0, np.float32))
        for e in events:
            if e["type"] == "partial":
                partials.setdefault(e["segment"], []).append(e["text"])
            else:
                finals.append(e)
    current["kind"] = "final"
    finals += live.finish()
    hyp = " ".join(f["text"] for f in finals if f["text"])
    ref = " ".join(vocab.filter(normalize_text(r, lang)) for r in refs)
    res = score([ref], [vocab.filter(hyp) if lang == "km" else hyp], [lang])
    # partials against their own final: how much of what the user saw changed
    churn_e = churn_n = 0
    for f in finals:
        for p in partials.get(f["segment"], []):
            churn_e += edit_distance(list(p.replace(" ", "")), list(f["text"].replace(" ", "")))
            churn_n += max(1, len(f["text"].replace(" ", "")))
    # segmentation against the known utterances
    cuts = [int(round(f["end"] * RATE)) for f in finals]
    mid = sum(any(s + 0.15 * RATE < c < e - 0.15 * RATE for (s, e) in spans) for c in cuts)
    covered = np.zeros(len(audio), bool)
    for f in finals:
        if f["text"]:
            covered[int(f["start"] * RATE):int(f["end"] * RATE)] = True
    speech = sum(e - s for s, e in spans)
    missed = sum(int((~covered[s:e]).sum()) for s, e in spans) / max(speech, 1)
    dur = len(audio) / RATE
    res.update({
        "audio_s": round(dur, 1), "segments": len(finals), "utterances": len(spans),
        "mid_utterance_cuts": mid, "missed_speech": round(missed, 4),
        "partial_vs_final_cer": round(churn_e / max(churn_n, 1), 4), "partials": sum(len(v) for v in partials.values()),
        "decode_s_per_audio_s": round(sum(t for t, _ in timings["partial"] + timings["final"]) / dur, 4),
        "final_decode_s_mean": round(float(np.mean([t for t, _ in timings["final"]] or [0])), 3),
        "final_decode_s_max": round(float(np.max([t for t, _ in timings["final"]] or [0])), 3),
        "partial_decode_s_max": round(float(np.max([t for t, _ in timings["partial"]] or [0])), 3),
    })
    return res


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="karna-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--sets", nargs="*", default=[*NATURAL, *SIMULATED])
    ap.add_argument("--live", action="store_true", help="also run the live streams")
    ap.add_argument("--live-sets", nargs="*", default=["fleurs_km@clean", "fleurs_km@noise", "fleurs_km@far",
                                                       "libri_other@clean", "libri_other@noise"])
    ap.add_argument("--no-offline", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--opt", action="append", default=[],
                    help="key=value passed to transcribe() and stream() (e.g. decoder=beam); JSON values")
    ap.add_argument("--stream-opt", action="append", default=[],
                    help="key=value passed to stream() only (e.g. vad=silero, min_silence_ms=500)")
    ap.add_argument("--lm", help="character LM (.npz) for decoder=beam")
    ap.add_argument("--beam-opt", action="append", default=[],
                    help="key=value beam settings (lm_weight, insertion_bonus, beam, ...); JSON values")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--prepare", action="store_true", help="only build the cached sets")
    args = ap.parse_args()

    opts = {}
    for kv in args.opt:
        k, _, v = kv.partition("=")
        try:
            opts[k] = json.loads(v)
        except ValueError:
            opts[k] = v
    corpora = Corpora()
    if args.prepare:
        for name in args.sets:
            t0 = time.time()
            audio, _, _ = get_set(name, corpora)
            print(f"{name}: {len(audio)} clips, {sum(map(len, audio)) / RATE / 60:.1f} min ({time.time() - t0:.0f} s)",
                  flush=True)
        for name in args.live_sets:
            audio, refs, _, _ = make_stream(name)
            print(f"stream {name}: {len(refs)} utterances, {len(audio) / RATE / 60:.1f} min", flush=True)
        return

    t0 = time.perf_counter()
    asr = karna.load_model(args.model, device=args.device)
    load_s = time.perf_counter() - t0
    vocab = asr.recognizer.vocab
    beam_opts = {}
    for kv in args.beam_opt:
        k, _, v = kv.partition("=")
        beam_opts[k] = json.loads(v)
    if args.lm or beam_opts:
        asr.set_lm(args.lm, **beam_opts)
    result = {"model": args.model, "device": asr.device, "language": args.language, "opts": opts,
              "lm": args.lm, "beam_options": beam_opts,
              "load_s": round(load_s, 2), "offline": {}, "live": {}, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"{args.model} on {asr.device} (load {load_s:.1f} s), opts {opts}", flush=True)
    if not args.no_offline:
        for name in args.sets:
            audio, refs, langs = get_set(name, corpora)
            refs = [vocab.filter(normalize_text(r, lang)) for r, lang in zip(refs, langs)]
            t0 = time.perf_counter()
            hyps = asr.transcribe(list(audio), batch_size=args.batch_size, language=args.language, **opts)
            secs = time.perf_counter() - t0
            hyps = [vocab.filter(normalize_text(h, lang)) for h, lang in zip(hyps, langs)]
            res = score(refs, hyps, langs)
            res["rtf"] = round(secs / (sum(map(len, audio)) / RATE), 4)
            res["examples"] = [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:3]]
            result["offline"][name] = res
            print(f"  {name:22s} " + "  ".join(f"{k} {v:.4f}" for k, v in res.items()
                                              if k in ("cer", "wer", "latin", "rtf")), flush=True)
    if args.live:
        stream_opts = {"language": args.language, **opts}
        for kv in args.stream_opt:
            k, _, v = kv.partition("=")
            try:
                stream_opts[k] = json.loads(v)
            except ValueError:
                stream_opts[k] = v
        result["stream_options"] = stream_opts
        for name in args.live_sets:
            res = run_live(asr, name, vocab, stream_opts)
            result["live"][name] = res
            print(f"  live {name:17s} " + "  ".join(f"{k} {v}" for k, v in res.items() if k != "n"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.model}-{result['device']}-{args.tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
