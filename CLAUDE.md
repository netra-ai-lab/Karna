# Karna: project memory

Khmer + English CTC speech recognition: karna-base (71M) and karna-large (217M), pruned from Omnilingual ASR.
Laid out per the Netra Lab kit (`playbooks/REPO_LAYOUT.md`): `karna/` is the inference-only pip package
`netra-karna` (ONNX Runtime, no torch); `training/` is the separate `karna-train` package (PyTorch),
never imported by `karna/`.

## Layout

- `karna/models.py`: short name → HF repo, the `onnx` folder in it, and the pinned `revision`. `KARNA_BUNDLES` points at local model folders.
- `karna/inference/`: `session.py` and `hub.py` from the kit (keep in sync with Netra-OCR); `recognizer.py` is Karna's own (normalise → pad → graph → greedy CTC).
- `karna/server/`: FastAPI + static UI, with one worker thread; jobs are directories under `KARNA_DATA_DIR`.
- `training/karna_train/export_onnx.py` + `training/export.sh`: PyTorch release → ONNX model folder, with a parity check. `run.sh` runs it after `06_export_pruned.py`.
- `docker/`: CPU image with both models baked in; `smoke_test.py` runs at build time.
- Docs: `docs/USAGE.md` (every flag, route and env var), `docs/RELEASING.md` (release steps), `THIRD_PARTY_LICENSES.md`.

## Decisions

- 2026-11-22 (user): keep two HF repos, `Netra-Lab/karna-base` and `Netra-Lab/karna-large`, each with a new `onnx/` folder; the root PyTorch files stay for fine-tuning.
- 2026-11-22 (user): the teachers (`teacher-300m`, `teacher-1b`) stay PyTorch-only training assets; they are not in the package's model list.
- The graph ends in the per-frame argmax (`ids`, `frame_lens`), so [N, T, 9812] logits never leave the session. The attention mask is always built in the wrapper, because `PrunableCTC.forward` builds it only when something is padded (a data-dependent branch).
- fp32 ONNX only. **int8 rejected** (2026-11-23, karna-base, `onnxruntime.quantization.quantize_dynamic`, QInt8 weights): 4.4× slower on CPU (340 s vs 77 s for the 40 parity clips), Khmer CER 0.205 → 0.248 against references, and only 6/40 transcripts identical to fp32. **fp16 storage not tried yet**: it would halve the download, but would need fp32 compute (Cast after each initializer) and identical text before shipping.
- The server refuses audio over 30 s (`KARNA_MAX_SECONDS`): training used 1–25 s clips and each file is decoded in one pass.
- The CLI keeps `tsv` (`path<TAB>text`) as its default output, so scripts written for the PyTorch CLI keep working.

## Measured

Machine: AMD Zen 3 laptop CPU (AMD64 Family 25 Model 80), 16 threads, Windows 11; torch 2.15 CPU, onnxruntime 1.32. No NVIDIA GPU on this machine, so CUDA parity and speed are **unmeasured**.

| | karna-base | karna-large |
| --- | --- | --- |
| `onnx/model.onnx` | 287 MB | 871 MB |
| Parity, ONNX vs PyTorch (CPU) | 40/40 real clips (20 FLEURS km, 20 LibriSpeech) and 16/16 random mixed-length batches identical | same |
| ONNX, `scripts/benchmark.py`, 40 clips, 506 s of audio, batch 8 | 8.66 s per minute of audio (RTF 0.144), model load 0.88 s | 17.02 s per minute (RTF 0.284), load 2.34 s |
| PyTorch, same clips, 8 threads | 8.83 s per minute (RTF 0.147) | 16.72 s per minute (RTF 0.279) |

So on CPU, ONNX runs at the same speed as PyTorch: no speedup, so don't claim one. The gains are install size (wheel 0.16 MB, about 195 MB of site-packages with `[cpu,server]`, no torch) and the serving channels. Almost all of the time is the graph; load, prep and decode are each under 0.02 s per minute.

Tests (2026-11-23): `pytest tests training/tests`: 29 passed, 8 skipped (CUDA-only). `pytest tests --slow`: 19 passed (golden transcripts identical for both models). A clean venv from the wheel passed `docker/smoke_test.py`.

## Pitfalls

- A tiny test model needs a realistic conv stride. With too small a total stride, the frame count explodes, and attention over N×H×T×T allocated 11 GB at one Add node. It looked like a broadcasting bug in the export, but wasn't one.
- `torch.onnx.export(dynamo=True)` warns that the `batch` axis name "will not be used" because it shares constraints with another axis. This is harmless.
- `onnxruntime` and `onnxruntime-gpu` must never be installed together: they share the module name.
- The exported ONNX bundles live in `training/runs/onnx/<name>/` (git-ignored). Use `KARNA_BUNDLES=training/runs/onnx` to run against them offline.
- Git Bash on Windows: run Python with `PYTHONIOENCODING=utf-8`, or Khmer output raises `UnicodeEncodeError` (cp1252).

## Pending (each outward-facing step needs the user's go-ahead)

1. Done 2026-11-23: `onnx/`, the standardised cards and `assets/header.png` are uploaded. `revision` is pinned to karna-base `563d364`, karna-large `d124eca`. `pytest --slow` passes against the published models.
2. `docker build` (the Docker daemon wasn't running on 2026-11-23), then fill the image size and memory TODOs in the README §5 table and in `docs/RELEASING.md`.
3. Measure CUDA parity and speed on a GPU machine.
4. PyPI pending publisher and the GitHub `pypi` environment (user, on the web; Chrome is not allowed on pypi.org). GHCR visibility after the first image is pushed.
5. Branch `onnx-serving` is pushed; the user opens the PR (Chrome is not signed in to GitHub, and gh is not installed). Tag `v0.2.0` after merge.
