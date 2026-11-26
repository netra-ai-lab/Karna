"""Tune beam search (LM weight, insertion bonus, beam) on the dev sets of scripts/realworld_eval.py.

The acoustic model runs once per clip; the grid then only re-runs the beam search, in parallel.

    python scripts/tune_lm.py --model karna-base --lm training/runs/lm/karna-char8.npz \\
        --weights 0 0.2 0.4 0.6 0.8 --bonus 0 0.5 1 1.5 2

Prints CER (Khmer) / WER (English) per set and setting; writes
training/runs/eval/realworld/tune-<model>-<tag>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realworld_eval as rw  # noqa: E402

import karna  # noqa: E402
from karna.inference.beam import beam_search  # noqa: E402
from karna.inference.lm import CharLM, symbol_map  # noqa: E402
from karna.text import clean_khmer, normalize_text  # noqa: E402

_W: dict = {}


def _init(lm_path, pieces, alphabet_ids):
    lm = CharLM(lm_path) if lm_path else None
    khmer = np.array([False] + [("ក" <= pieces[i] <= "៿" or "᧠" <= pieces[i] <= "᧿") for i in alphabet_ids])
    _W.update(lm=lm, symbols=symbol_map(lm, pieces, alphabet_ids) if lm else None, alphabet=alphabet_ids,
              khmer=khmer)


def _decode(job):
    logp, opts = job
    opts = dict(opts)
    if "insertion_bonus_latin" in opts:
        latin = opts.pop("insertion_bonus_latin")
        opts["insertion_bonus"] = np.where(_W["khmer"], opts.get("insertion_bonus", 0.0), latin)
    cols = beam_search(logp, _W["lm"], _W["symbols"], **opts)
    return [_W["alphabet"][c - 1] for c in cols]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="karna-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lm")
    ap.add_argument("--sets", nargs="*", default=list(rw.DEV))
    ap.add_argument("--weights", type=float, nargs="*", default=[0.0, 0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--bonus", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--bonus-latin", type=float, nargs="*", default=[None],
                    help="insertion bonus for ASCII/space columns (default: same as --bonus)")
    ap.add_argument("--beam", type=int, nargs="*", default=[8])
    ap.add_argument("--language", default="auto")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    asr = karna.load_model(args.model, device=args.device)
    rec = asr.recognizer
    vocab = rec.vocab
    data = {}
    for name in args.sets:
        audio, refs, langs = rw.get_set(name)
        t0 = time.perf_counter()
        cache = rw.CACHE / "logp" / f"{args.model}-{args.language}-{name.replace('@', '__')}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                off = z["offsets"]
                flat = z["logp"].astype(np.float32)
                logps = [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
                greedy = json.loads(str(z["greedy"]))
            refs = [vocab.filter(normalize_text(r, lang)) for r, lang in zip(refs, langs)]
            data[name] = (logps, refs, langs, greedy)
            print(f"{name}: {len(audio)} clips (cached log-probabilities)", flush=True)
            continue
        logps, greedy = [None] * len(audio), [None] * len(audio)
        order = sorted(range(len(audio)), key=lambda i: len(audio[i]), reverse=True)
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            lens = np.array([len(audio[i]) for i in idx], np.int64)
            x = np.zeros((len(idx), int(lens.max())), np.float32)
            for r, i in enumerate(idx):
                x[r, :len(audio[i])] = rec.normalize(audio[i])
            out = rec.session.run(rec._inputs(x, lens, args.language))
            for r, i in enumerate(idx):
                n = int(out["frame_lens"][r])
                logps[i] = out["sub_logp"][r, :n].copy()
                greedy[i] = vocab.greedy(out["ids"][r, :n].tolist())
        refs = [vocab.filter(normalize_text(r, lang)) for r, lang in zip(refs, langs)]
        data[name] = (logps, refs, langs, greedy)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, logp=np.concatenate(logps).astype(np.float16),
                 offsets=np.cumsum([0] + [len(lp) for lp in logps]), greedy=np.array(json.dumps(greedy)))
        print(f"{name}: {len(audio)} clips, acoustic {time.perf_counter() - t0:.0f} s", flush=True)

    def post(text: str, lang: str) -> str:
        return vocab.filter(normalize_text(clean_khmer(text), lang))

    results = {"model": args.model, "lm": args.lm, "language": args.language, "greedy": {}, "grid": []}
    for name, (logps, refs, langs, greedy) in data.items():
        results["greedy"][name] = rw.score(refs, [post(h, lang) for h, lang in zip(greedy, langs)], langs)
    print("greedy  " + "  ".join(f"{n} {_fmt(r)}" for n, r in results["greedy"].items()), flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.lm, vocab.pieces, rec.alphabet_ids)) as pool:
        weights = args.weights if args.lm else [0.0]
        for w, bonus, latin, beam in itertools.product(weights, args.bonus, args.bonus_latin, args.beam):
            opts = {"lm_weight": w, "insertion_bonus": bonus, "beam": beam}
            if latin is not None:
                opts["insertion_bonus_latin"] = latin
            t0 = time.perf_counter()
            row = {"options": opts, "sets": {}}
            frames = 0
            for name, (logps, refs, langs, _) in data.items():
                ids = pool.map(_decode, [(lp, opts) for lp in logps], chunksize=4)
                hyps = [post(vocab.decode(i), lang) for i, lang in zip(ids, langs)]
                row["sets"][name] = rw.score(refs, hyps, langs)
                frames += sum(len(lp) for lp in logps)
            secs = time.perf_counter() - t0
            row["beam_ms_per_audio_s"] = round(1000 * secs * args.workers / (frames * 0.02), 2)
            results["grid"].append(row)
            print(f"w {w:.2f} b {bonus:.2f} bl {latin if latin is not None else bonus:.2f} beam {beam:2d}  " + "  ".join(
                f"{n} {_fmt(r)}" for n, r in row["sets"].items()) + f"  ({row['beam_ms_per_audio_s']} ms/s)",
                flush=True)
    out = rw.OUT / f"tune-{args.model}-{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def _fmt(r: dict) -> str:
    return f"{r['wer']:.4f}w" if "wer" in r else f"{r['cer']:.4f}"


if __name__ == "__main__":
    main()
