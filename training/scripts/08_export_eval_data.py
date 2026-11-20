"""Pack the nine evaluation sets into one Hugging Face dataset folder (parquet + README).

    .venv/bin/python scripts/08_export_eval_data.py <out_dir> [--push user/name] [--public]

Each set becomes ``data/<set>.parquet`` with columns id, audio ({bytes, path}, the original
encoded audio: wav or flac), text (the reference as scored by 03_evaluate.py), lang, duration.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from karna_train import ASR_ROOT, PROJECT_ROOT  # noqa: E402

SETS = ["ddd_dev", "ddd_test", "fleurs_km_dev", "fleurs_km_test", "fleurs_en_dev", "fleurs_en_test",
        "libri_dev-clean", "libri_test-clean", "libri_test-other"]
HF_FEATURES = {"info": {"features": {
    "id": {"dtype": "string", "_type": "Value"}, "audio": {"_type": "Audio"},
    "text": {"dtype": "string", "_type": "Value"}, "lang": {"dtype": "string", "_type": "Value"},
    "duration": {"dtype": "float32", "_type": "Value"}}}}
SCHEMA = pa.schema(
    [("id", pa.string()), ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
     ("text", pa.string()), ("lang", pa.string()), ("duration", pa.float32())],
    metadata={"huggingface": json.dumps(HF_FEATURES)})
README = """---
license: cc-by-sa-4.0
task_categories:
- automatic-speech-recognition
language:
- km
- en
configs:
{configs}
---
# Karna evaluation sets

The nine sets that `training/scripts/03_evaluate.py` scores in
the Karna repo, one parquet file each. Audio is stored as the original
encoded bytes (DDD and FLEURS: wav, LibriSpeech: flac); `text` is the reference exactly as
scored.

| Set | Source | Utterances |
|---|---|---|
{rows}

Sources and licences: DDD from
[Digital-Divide-Data/khmer-speech-dataset](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset)
(CC BY-SA 4.0), FLEURS (CC BY 4.0), LibriSpeech (CC BY 4.0). The DDD dev and test rows were held
out of every training run. Use of the combined set is under the most restrictive licence (CC BY-SA 4.0).
"""
SOURCE = {"ddd": "DDD, Khmer", "fleurs_km": "FLEURS, Khmer", "fleurs_en": "FLEURS, English", "libri": "LibriSpeech"}


def audio_bytes(r: dict, shards: dict) -> tuple[bytes, str]:
    if "path" in r:
        return Path(r["path"]).read_bytes(), Path(r["path"]).name
    t = shards.get(r["shard"])
    if t is None:
        shards.clear()  # rows are visited shard by shard; keep one table in memory
        t = shards[r["shard"]] = pq.read_table(ASR_ROOT / r["shard"], columns=["audio"])["audio"].to_pylist()
    a = t[r["row"]]
    return a["bytes"], f"{Path(r['shard']).stem}_{r['row']}"


def duration(raw: bytes) -> float:
    i = sf.info(io.BytesIO(raw))
    return round(i.frames / i.samplerate, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--push", help="upload to this dataset repo (created private unless --public)")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in SETS:
        rows = [json.loads(l) for l in open(PROJECT_ROOT / "data" / f"{name}.jsonl")]
        shards: dict = {}
        out = []
        for r in sorted(rows, key=lambda r: (r.get("shard", ""), r.get("row", 0))):  # one read per shard
            raw, fname = audio_bytes(r, shards)
            out.append({"id": r.get("id") or f"{name}_{fname}", "audio": {"bytes": raw, "path": fname},
                        "text": r["text"], "lang": r["lang"], "duration": duration(raw)})
        pq.write_table(pa.Table.from_pylist(out, schema=SCHEMA), args.out / "data" / f"{name}.parquet",
                       row_group_size=200)
        counts[name] = len(out)
        print(name, len(out), flush=True)
    configs = "\n".join(f"- config_name: {n}\n  data_files: data/{n}.parquet" for n in SETS)
    rows = "\n".join(f"| `{n}` | {next(v for k, v in SOURCE.items() if n.startswith(k))} | {c} |" for n, c in counts.items())
    (args.out / "README.md").write_text(README.format(configs=configs, rows=rows))
    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, repo_type="dataset", private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=args.push, repo_type="dataset", folder_path=str(args.out),
                          commit_message="Add Karna evaluation sets")
        print("pushed", args.push)


if __name__ == "__main__":
    main()
