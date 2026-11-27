"""Build Karna's character n-gram language model from text on this machine (numpy only).

    python scripts/build_char_lm.py --out training/runs/lm/karna-char8 --order 8

1. Text. Each source is read once, line by line, and sampled to its character quota:
   Khmer from the web corpus (khmer-lm-corpus), the ASR text corpus, the Khmer side of the
   Khmer-English MT corpus and the DDD training transcripts; English from the LibriSpeech-960
   transcripts and the English side of the MT corpus. Lines are normalised as the model's
   output is (NFC, zero-width removed, lowercase), reduced to the LM alphabet
   (``bias.lm_alphabet``: Khmer, ASCII, space), and dropped when more than 10 % of their
   characters fall outside it or when they repeat a line already taken.
2. Every evaluation sentence is removed: FLEURS km/en dev and test, DDD dev and test, SLR42,
   kmcorpus, Common Voice km, LibriSpeech dev/test (clean and other) and the golden clips. A line
   is dropped when it shares any 20 non-space characters with one of them (found with 20-grams at
   a stride of 10, so every shared run of 29 or more is caught), or equals a shorter one.
3. Counts for orders 1..N, interpolated modified Kneser-Ney (continuation counts below the top
   order, three discounts per order), then count pruning (``--prune``, per order: keep n-grams
   seen more than this many times). Backoff weights are the interpolation weights.
4. ``<out>.npz`` (``karna.inference.lm.CharLM``) and ``<out>.json``, the build manifest: sources
   with their sizes and hashes, quotas, lines kept and dropped, pruning, model size and the
   per-character perplexity on the held-out evaluation references.

Peak memory is about 12 bytes per text character per order being counted; run it under a cap:
``systemd-run --user --scope -p MemoryMax=12G ...``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from karna.inference.bias import lm_alphabet
from karna.text import SPACE_PIECE, CharVocab, normalize_text

ROOT = Path(__file__).resolve().parents[1]
DISK = Path("/run/media/pc/disk1")
DATA = DISK / "ml-data" / "datasets"
ASR = DISK / "khmer-self-feedback-asr"
GRAM, STRIDE = 20, 10

# (name, language, path, reader, character quota)
SOURCES = [
    ("ddd_train", "km", DATA / "asr/khmer-asr-corpus/v1/train.tsv", "tsv_transcript", 40_000_000),
    ("khmer_web", "km", DATA / "text-corpora/khmer-lm-corpus/v1/all_text.txt", "lines", 70_000_000),
    ("khmer_asr_text", "km", DATA / "text-corpora/nmt-tokenizer-corpus/v1/asr_corpus_raw.txt", "lines", 25_000_000),
    ("mt_khmer", "km", DATA / "mt/kh-en-corpus/v1/combined_train.jsonl", "mt_km", 20_000_000),
    ("libri960", "en", ROOT / "training/data/libri_train_960.jsonl", "jsonl_text", 60_000_000),
    ("mt_english", "en", DATA / "mt/kh-en-corpus/v1/combined_train.jsonl", "mt_en", 30_000_000),
]
_TAG = re.compile(r"^<2\w+>\s*")
_KHMER = re.compile("[ក-៿᧠-᧿]")


# ---------------------------------------------------------------- text

class Encoder:
    """Normalised text -> LM symbols (one byte each: 1..K, 0 is the sentence boundary)."""

    def __init__(self) -> None:
        vocab = CharVocab.default()
        self.alphabet = [" " if vocab.pieces[i] == SPACE_PIECE else vocab.pieces[i] for i in lm_alphabet(vocab.pieces)]
        assert len(self.alphabet) < 255
        self.space = self.alphabet.index(" ") + 1
        self.table = {ord(ch): chr(i + 1) for i, ch in enumerate(self.alphabet)}
        self._spaces = re.compile(chr(self.space) + "+")

    def encode(self, text: str) -> tuple[str, float]:
        """(symbols as a str of chr(1..K), share of non-space characters dropped)."""
        text = normalize_text(text, "en")  # lowercase: the vocabulary has lowercase Latin only
        if not text:
            return "", 1.0
        enc = text.translate(self._delete_others(text))
        enc = self._spaces.sub(chr(self.space), enc).strip(chr(self.space))
        n = len(text) - text.count(" ")
        kept = len(enc) - enc.count(chr(self.space))
        return enc, 1 - kept / max(n, 1)

    def _delete_others(self, text: str) -> dict:
        table = self.table
        missing = {ord(c) for c in set(text)} - table.keys()
        if not missing:
            return table
        out = dict(table)
        for o in missing:
            out[o] = chr(self.space) if chr(o).isspace() else None
        return out


def read(reader: str, path: Path):
    """Yield raw text lines from a source."""
    with open(path, encoding="utf-8", errors="replace") as f:
        if reader == "lines":
            yield from f
        elif reader == "tsv_transcript":
            next(f)
            for line in f:
                parts = line.split("\t")
                if len(parts) > 2:
                    yield parts[2]
        elif reader == "jsonl_text":
            for line in f:
                yield json.loads(line)["text"]
        elif reader in ("mt_km", "mt_en"):
            want_km = reader == "mt_km"
            for line in f:
                row = json.loads(line)
                for side in (row.get("source", ""), row.get("target", "")):
                    side = _TAG.sub("", side)
                    if bool(_KHMER.search(side)) == want_km:
                        yield side
        else:
            raise ValueError(reader)


def eval_references() -> dict[str, list[str]]:
    """Every sentence the models are evaluated on, by set."""
    refs: dict[str, list[str]] = {}
    for name in ("fleurs_km_dev", "fleurs_km_test", "ddd_dev", "ddd_test", "fleurs_en_dev", "fleurs_en_test",
                 "libri_dev-clean", "libri_test-clean", "libri_test-other"):
        path = ROOT / "training/data" / f"{name}.jsonl"
        if path.exists():
            refs[name] = [json.loads(line)["text"] for line in open(path, encoding="utf-8")]
    for name in ("test_slr42", "test_kmcorpus", "test_cv26_km"):
        path = ASR / "manifests" / f"{name}.jsonl"
        if path.exists():
            refs[name] = [json.loads(line)["transcript"] for line in open(path, encoding="utf-8")]
    libri = DATA / "asr/librispeech/v1/LibriSpeech"
    for split in ("dev-clean", "dev-other", "test-clean", "test-other"):
        lines = []
        for trans in glob.glob(str(libri / split / "*" / "*" / "*.trans.txt")):
            lines += [line.split(" ", 1)[1] for line in open(trans, encoding="utf-8") if " " in line]
        if lines:
            refs[f"librispeech_{split}"] = lines
    golden = ROOT / "tests/golden/references.tsv"
    if golden.exists():
        refs["golden"] = [line.rstrip("\n").split("\t")[-1] for line in open(golden, encoding="utf-8")]
    return refs


class Exclusion:
    def __init__(self, enc: Encoder, refs: dict[str, list[str]]) -> None:
        sp = chr(enc.space)
        self.grams: set[str] = set()
        self.short: set[str] = set()
        for lines in refs.values():
            for r in lines:
                s = enc.encode(r)[0].replace(sp, "")
                if len(s) < GRAM:
                    if s:
                        self.short.add(s)
                    continue
                self.grams.update(s[i:i + GRAM] for i in range(len(s) - GRAM + 1))
        self.sp = sp

    def hit(self, enc_line: str) -> bool:
        s = enc_line.replace(self.sp, "")
        if len(s) < GRAM:
            return s in self.short
        grams = self.grams
        return any(s[i:i + GRAM] in grams for i in range(0, len(s) - GRAM + 1, STRIDE)) or \
            s[len(s) - GRAM:] in grams


def gather(enc: Encoder, excl: Exclusion, quota_scale: float, seed: int, log,
           only: list[str] | None = None) -> tuple[dict, dict]:
    """Per language, the training text as a uint8 stream with 0 between lines; and the manifest."""
    rng = np.random.default_rng(seed)
    streams = {"km": bytearray(b"\0"), "en": bytearray(b"\0")}
    seen: set[bytes] = set()
    manifest = {}
    for name, lang, path, reader, quota in SOURCES:
        if only and name not in only:
            continue
        if not path.exists():
            log(f"  {name}: missing {path}, skipped")
            continue
        quota = int(quota * quota_scale)
        st = path.stat()
        # characters in the file, roughly (Khmer is 3 bytes a character in UTF-8); MT has both sides
        est = st.st_size / (2.6 if lang == "km" else 1.0) / (2 if reader.startswith("mt_") else 1)
        keep_p = min(1.0, 1.3 * quota / est)
        t0 = time.time()
        stats = {"lines": 0, "kept": 0, "chars": 0, "sampled_out": 0, "foreign": 0, "duplicate": 0,
                 "eval_overlap": 0, "empty": 0}
        buf = streams[lang]
        for raw in read(reader, path):
            stats["lines"] += 1
            if keep_p < 1 and rng.random() > keep_p:
                stats["sampled_out"] += 1
                continue
            line, dropped = enc.encode(raw)
            if len(line) < 3:
                stats["empty"] += 1
                continue
            if dropped > 0.10:
                stats["foreign"] += 1
                continue
            data = line.encode("latin-1")
            h = hashlib.blake2b(data, digest_size=8).digest()
            if h in seen:
                stats["duplicate"] += 1
                continue
            if excl.hit(line):
                stats["eval_overlap"] += 1
                continue
            seen.add(h)
            buf += data
            buf += b"\0"
            stats["kept"] += 1
            stats["chars"] += len(data)
            if stats["chars"] >= quota:
                break
        manifest[name] = {"language": lang, "path": str(path), "bytes": st.st_size,
                          "mtime": int(st.st_mtime), "quota_chars": quota, "keep_probability": round(keep_p, 4),
                          **stats, "seconds": round(time.time() - t0, 1)}
        log(f"  {name}: {stats['kept']:,} lines, {stats['chars'] / 1e6:.1f} M chars "
            f"(eval overlap {stats['eval_overlap']:,}, duplicates {stats['duplicate']:,}, "
            f"foreign {stats['foreign']:,}) in {time.time() - t0:.0f} s")
    return {k: np.frombuffer(bytes(v), np.uint8) for k, v in streams.items()}, manifest


# ---------------------------------------------------------------- counting

def ngram_counts(stream: np.ndarray, k: int, chunk: int = 40_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique k-gram keys of ``stream`` and their counts. A k-gram may start with the
    boundary (``<s>``) and end with it (``</s>``), but not have it inside."""
    out_keys, out_counts = [], []
    n = len(stream)
    zero = np.concatenate([[0], np.cumsum(stream == 0, dtype=np.int32)])
    for lo in range(0, n - k + 1, chunk):
        hi = min(n - k + 1, lo + chunk)
        keys = np.zeros(hi - lo, np.uint64)
        for i in range(k):
            keys = (keys << np.uint64(8)) | stream[lo + i:hi + i].astype(np.uint64)
        if k > 2:
            starts = np.arange(lo, hi)
            valid = (zero[starts + k - 1] - zero[starts + 1]) == 0
            keys = keys[valid]
        u, c = np.unique(keys, return_counts=True)
        out_keys.append(u)
        out_counts.append(c.astype(np.int64))
        del keys
    if len(out_keys) == 1:
        return out_keys[0], out_counts[0]
    keys = np.concatenate(out_keys)
    counts = np.concatenate(out_counts)
    del out_keys, out_counts
    order = np.argsort(keys, kind="stable")
    keys, counts = keys[order], counts[order]
    first = np.concatenate([[True], keys[1:] != keys[:-1]])
    idx = np.flatnonzero(first)
    return keys[idx], np.add.reduceat(counts, idx)


def discounts(counts: np.ndarray) -> tuple[float, float, float]:
    n = [int((counts == i).sum()) for i in (1, 2, 3, 4)]
    if min(n) == 0:
        return 0.5, 1.0, 1.5
    y = n[0] / (n[0] + 2 * n[1])
    d = (1 - 2 * y * n[1] / n[0], 2 - 3 * y * n[2] / n[1], 3 - 4 * y * n[3] / n[2])
    return tuple(float(np.clip(x, 0.05, i + 1)) for i, x in enumerate(d))


def lookup(keys: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    return idx, keys[idx] == query


def build(stream: np.ndarray, order: int, prune: list[int], log) -> dict:
    """Interpolated modified Kneser-Ney with count pruning: per order, kept keys, ln p and ln bow."""
    size = int(stream.max()) + 1
    kept: dict[int, dict] = {}
    upper_keys = None  # unique keys of order k+1 (unpruned), for continuation counts
    stats = {}
    for k in range(order, 0, -1):
        t0 = time.time()
        keys, raw = ngram_counts(stream, k)
        if k == order:
            mod = raw.copy()
        else:
            suffix = upper_keys & np.uint64((1 << (8 * k)) - 1)
            su, sc = np.unique(suffix, return_counts=True)
            idx, hit = lookup(su, keys)
            mod = np.where(hit, sc[idx], 0).astype(np.int64)
            bos = (keys >> np.uint64(8 * (k - 1))) == 0  # starts with <s>: nothing precedes it
            if k == 1:
                bos[:] = False
            mod = np.where(bos | (mod == 0), raw, mod)
            del suffix, su, sc, idx, hit
        d1, d2, d3 = discounts(mod)
        disc = np.where(mod == 1, d1, np.where(mod == 2, d2, d3))
        # per history: total, and the mass the discounts free for the lower order
        hist = keys >> np.uint64(8)
        starts = np.flatnonzero(np.concatenate([[True], hist[1:] != hist[:-1]]))
        c_h = np.add.reduceat(mod, starts).astype(np.float64)
        freed = np.add.reduceat(disc, starts)
        gamma_h = freed / c_h
        lens = np.diff(np.append(starts, len(keys)))
        c_entry = np.repeat(c_h, lens)
        gamma_entry = np.repeat(gamma_h, lens)
        keep = raw > prune[k - 1] if k > 1 else np.ones(len(keys), bool)
        kept[k] = {"keys": keys[keep], "alpha": ((mod - disc) / c_entry)[keep],
                   "gamma": gamma_entry[keep], "hist_keys": hist[starts], "hist_gamma": gamma_h}
        stats[k] = {"unique": int(len(keys)), "kept": int(keep.sum()), "discounts": [round(d1, 3), round(d2, 3), round(d3, 3)],
                    "seconds": round(time.time() - t0, 1)}
        log(f"    order {k}: {len(keys):,} n-grams, kept {int(keep.sum()):,}, D {d1:.2f} {d2:.2f} {d3:.2f} "
            f"({time.time() - t0:.0f} s)")
        upper_keys = keys
        del raw, mod, disc, hist, c_entry, gamma_entry, keep
    del upper_keys
    # probabilities, low order first: p(w|h) = alpha(hw) + gamma(h) p(w|h')
    tables = {}
    for k in range(1, order + 1):
        t = kept[k]
        if k == 1:
            # unigrams interpolate with the uniform distribution over every symbol
            full = np.zeros(size, np.float64)
            full[t["keys"].astype(np.int64)] = t["alpha"]
            gamma0 = float(t["gamma"][0]) if len(t["gamma"]) else 1.0
            p = full + gamma0 / size
            keys = np.arange(size, dtype=np.uint64)
        else:
            lower = tables[k - 1]
            idx, hit = lookup(lower["keys"], t["keys"] & np.uint64((1 << (8 * (k - 1))) - 1))
            if not hit.all():
                raise RuntimeError(f"order {k}: {int((~hit).sum())} n-grams have no lower-order suffix")
            p = t["alpha"] + t["gamma"] * np.exp(lower["logp"][idx].astype(np.float64))
            keys = t["keys"]
        tables[k] = {"keys": keys, "logp": np.log(np.maximum(p, 1e-30)).astype(np.float32),
                     "bow": np.zeros(len(keys), np.float32)}
    # backoff weight of a history = its interpolation weight gamma (computed one order up)
    for k in range(1, order):
        up = kept[k + 1]
        idx, hit = lookup(tables[k]["keys"], up["hist_keys"])
        tables[k]["bow"][idx[hit]] = np.log(np.maximum(up["hist_gamma"][hit], 1e-30))
    return {"tables": tables, "stats": stats}


def save(path: Path, tables: dict, order: int, alphabet: list[str], meta: dict) -> int:
    arrays = {}
    for k in range(1, order + 1):
        arrays[f"keys{k}"] = tables[k]["keys"].astype(np.uint64)
        arrays[f"logp{k}"] = tables[k]["logp"].astype(np.float16)
        arrays[f"bow{k}"] = tables[k]["bow"].astype(np.float16)
    meta = {"order": order, "alphabet": alphabet, **meta}
    # the model file is meant to be shared: no local paths or file times (the .json manifest keeps them)
    meta["sources"] = {name: {k: v for k, v in src.items() if k not in ("path", "mtime", "bytes")}
                       for name, src in meta.get("sources", {}).items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    tmp.replace(path)
    return path.stat().st_size


def perplexity(lm, enc: Encoder, lines: list[str]) -> float:
    """Bits per character (``</s>`` counted) of ``lines`` under ``lm``."""
    total = n = 0
    for r in lines:
        s = enc.encode(r)[0]
        if not s:
            continue
        syms = [ord(c) for c in s]
        total += lm.score_text(syms)
        n += len(syms) + 1
    return float(-total / max(n, 1) / np.log(2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "training/runs/lm/karna-char8"))
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--prune", type=int, nargs="*", default=[0, 0, 0, 1, 1, 2, 3, 4],
                    help="per order 1..N: keep n-grams seen more than this many times")
    ap.add_argument("--quota-scale", type=float, default=1.0, help="multiply every source's character quota")
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--reuse-text", action="store_true", help="reuse <out>.text.npz from an earlier run")
    ap.add_argument("--sources", nargs="*", help="build from these sources only (default: all), e.g. "
                    "ddd_train libri960 for text with a known licence")
    args = ap.parse_args()
    if not 1 <= args.order <= 8:
        raise SystemExit("--order must be 1..8 (a key is 8 bytes)")
    prune = (args.prune + [args.prune[-1]] * args.order)[: args.order]
    if any(b < a for a, b in zip(prune, prune[1:])):
        raise SystemExit("--prune must not decrease with the order")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(out.with_suffix(".log"), "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_start = time.time()
    log(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} order {args.order}, prune {prune}, quota x{args.quota_scale}")
    enc = Encoder()
    refs = eval_references()
    log("evaluation references: " + ", ".join(f"{k} {len(v):,}" for k, v in refs.items()))
    text_cache = out.with_name(out.name + ".text.npz")
    if args.reuse_text and text_cache.exists():
        with np.load(text_cache) as data:
            streams = {"km": data["km"], "en": data["en"]}
            manifest = json.loads(str(data["manifest"]))
        log(f"reusing text from {text_cache}")
    else:
        excl = Exclusion(enc, refs)
        log(f"exclusion index: {len(excl.grams):,} {GRAM}-grams, {len(excl.short):,} short sentences")
        streams, manifest = gather(enc, excl, args.quota_scale, args.seed, log, args.sources)
        np.savez(text_cache, km=streams["km"], en=streams["en"], manifest=np.array(json.dumps(manifest)))
    stream = np.concatenate([streams["km"], streams["en"][1:]])
    log(f"text: Khmer {len(streams['km']) / 1e6:.1f} M, English {len(streams['en']) / 1e6:.1f} M symbols")
    del streams
    built = build(stream, args.order, prune, log)
    del stream
    meta = {"sources": manifest, "prune": prune, "orders": built["stats"], "gram_exclusion": [GRAM, STRIDE],
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": args.seed, "quota_scale": args.quota_scale}
    size = save(out.with_suffix(".npz"), built["tables"], args.order, enc.alphabet, meta)
    log(f"saved {out.with_suffix('.npz')} ({size / 1e6:.1f} MB)")
    from karna.inference.lm import CharLM

    lm = CharLM(out.with_suffix(".npz"))
    ppl = {}
    for name in ("fleurs_km_dev", "ddd_dev", "test_slr42", "test_kmcorpus", "libri_dev-clean", "fleurs_en_dev"):
        if name in refs:
            ppl[name] = round(perplexity(lm, enc, refs[name][:1500]), 4)
    log("bits per character: " + ", ".join(f"{k} {v}" for k, v in ppl.items()))
    manifest_out = {**meta, "file": str(out.with_suffix(".npz")), "bytes": size,
                    "sha256": hashlib.sha256(out.with_suffix(".npz").read_bytes()).hexdigest(),
                    "bits_per_char": ppl, "seconds": round(time.time() - t_start)}
    out.with_suffix(".json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
