# Karna: project memory

Khmer + English CTC speech recognition: karna-base (71M) and karna-large (217M), pruned from Omnilingual ASR.
Laid out per the Netra Lab kit (`playbooks/REPO_LAYOUT.md`): `karna/` is the inference-only pip package
`netra-karna` (ONNX Runtime, no torch); `training/` is the separate `karna-train` package (PyTorch),
never imported by `karna/`.

## Layout

- `karna/models.py`: short name → HF repo, the `onnx` folder in it, and the pinned `revision`. `KARNA_BUNDLES` points at local model folders.
- `karna/inference/`: `session.py` and `hub.py` from the kit (keep in sync with Netra-OCR); `recognizer.py` is Karna's own (normalise → pad → graph → greedy CTC).
- `karna/streaming.py`: live transcription (`Karna.stream()`): energy-VAD `Endpointer` plus partial/final re-decoding, numpy only.
- `karna/server/`: FastAPI + static UI, with one worker thread; jobs are directories under `KARNA_DATA_DIR`. `WebSocket /v1/stream` (`LiveSession` in `app.py`) runs its decodes on the same worker via `JobStore.submit` (priority queue: live first, then file jobs) and saves each session with `JobStore.record`. The browser side is `static/js/stream.js` + `pcm-worklet.js`.
- `training/karna_train/export_onnx.py` + `training/export.sh`: PyTorch release → ONNX model folder, with a parity check. `run.sh` runs it after `06_export_pruned.py`.
- `docker/`: CPU image with both models baked in; `smoke_test.py` runs at build time.
- Docs: `docs/USAGE.md` (every flag, route and env var), `docs/RELEASING.md` (release steps), `THIRD_PARTY_LICENSES.md`.

## Decisions

- 2026-11-22 (user): keep two HF repos, `Netra-Lab/karna-base` and `Netra-Lab/karna-large`, each with a new `onnx/` folder; the root PyTorch files stay for fine-tuning.
- 2026-11-22 (user): the teachers (`teacher-300m`, `teacher-1b`) stay PyTorch-only training assets; they are not in the package's model list.
- The graph ends in the per-frame argmax (`ids`, `frame_lens`), so [N, T, 9812] logits never leave the session. The attention mask is always built in the wrapper, because `PrunableCTC.forward` builds it only when something is padded (a data-dependent branch).
- fp32 ONNX only. **int8 rejected** (2026-11-23, karna-base, `onnxruntime.quantization.quantize_dynamic`, QInt8 weights): 4.4× slower on CPU (340 s vs 77 s for the 40 parity clips), Khmer CER 0.205 → 0.248 against references, and only 6/40 transcripts identical to fp32. **fp16 storage not tried yet**: it would halve the download, but would need fp32 compute (Cast after each initializer) and identical text before shipping.
- The server refuses audio over 30 s (`KARNA_MAX_SECONDS`): training used 1–25 s clips and each file is decoded in one pass.
- 2026-11-23: live transcription is **endpointed re-decoding**, not chunked state: the model has full attention and per-utterance normalisation, so there is nothing to carry between chunks. Partials re-decode the whole open utterance (latest wins); finals are offline decodes of the closed segment. Tuned on the golden clips:
  - `min_segment_s=5`: a 600 ms pause only closes a segment of 5 s or more, and 1.5 s closes any. Cutting at every 0.75 s pause gave a 2.4 s Khmer piece that karna-large returned in Latin script.
  - 0.5 s of zero padding on each side of every decode. The same quiet clip as a bare 7 s segment came back garbled from large, and exact again with padding. Cropping in more real background did not help.
  - `abs_floor=5e-4`, `ratio=4`: FLEURS km 0000 has speech at RMS 0.002–0.004, and 0.004 missed it entirely. Digital-silence frames never set the noise floor.
  - Result: 4/8 (clip, model) pairs identical to offline, the rest 1–5 characters apart (`test_streaming_matches_offline`, `--slow`).
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

Tests (2026-11-23, with streaming): `pytest tests training/tests`: 42 passed, 10 skipped (CUDA-only and `--slow`). `pytest tests --slow`: 34 passed (golden transcripts identical for both models; streaming within 15 % of the offline text).

Live, real-time pacing over the WebSocket (`/v1/stream`, same laptop): karna-base partials arrive about every 1 s and finals about 1.1 s after speech ends; karna-large finals 2–4 s after, with partials spreading out as the utterance grows. Checked in Chrome with a clip fed in as the microphone; a real microphone is **untested**. A clean venv from the wheel passed `docker/smoke_test.py`.

## Pitfalls

- A tiny test model needs a realistic conv stride. With too small a total stride, the frame count explodes, and attention over N×H×T×T allocated 11 GB at one Add node. It looked like a broadcasting bug in the export, but wasn't one.
- `torch.onnx.export(dynamo=True)` warns that the `batch` axis name "will not be used" because it shares constraints with another axis. This is harmless.
- `onnxruntime` and `onnxruntime-gpu` must never be installed together: they share the module name.
- The exported ONNX bundles live in `training/runs/onnx/<name>/` (git-ignored). Use `KARNA_BUNDLES=training/runs/onnx` to run against them offline.
- Git Bash on Windows: run Python with `PYTHONIOENCODING=utf-8`, or Khmer output raises `UnicodeEncodeError` (cp1252).

## Pending (each outward-facing step needs the user's go-ahead)

1. Done 2026-11-23: `onnx/`, the standardised cards and `assets/header.png` are uploaded. `revision` is pinned to karna-base `563d364`, karna-large `d124eca`. `pytest --slow` passes against the published models.
2. Done 2026-11-23: v0.2.0 released to PyPI (verified with a clean `[cpu]` install, golden 4/4 for both models, no torch) and GHCR (public). Measured the published image: 1.16 GB compressed, 1.42 GB unpacked, memory 0.5 GB idle / 1.7 GB with karna-base / 4.0 GB with both models (ORT keeps its arena), so the README says to give Docker 6 GB. Per 29 s clip in the container: karna-base about 4 s, karna-large about 8 s (the first karna-large call takes 19 s because it loads the model).
3. Measure CUDA parity and speed on a GPU machine.
4. Done: PyPI trusted publisher and the GitHub `pypi` environment are set up; the GHCR image is public.
5. Done: `onnx-serving` is fast-forwarded into `master`; tag `v0.2.0` is pushed. The merged `onnx-serving` branch can be deleted.
6. Done 2026-11-23: v1.0.0 (live transcription) released: tag `v1.0.0` pushed; the pypi, docker and tests workflows passed. A clean `[cpu,server]` venv from PyPI gave 1.0.0, no torch, golden 4/4 (karna-base), and a working `asr.stream()`.
7. Try Live with a real microphone, and in Firefox and Safari. So far it has only been checked in Chrome, with a clip fed in as the microphone.
