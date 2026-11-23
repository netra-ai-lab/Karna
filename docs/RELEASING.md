# Releasing Karna

A release is a version tag. It publishes:
- the package `netra-karna` to PyPI (`.github/workflows/pypi.yml`);
- the app image `ghcr.io/netra-ai-lab/karna` for `linux/amd64` and `linux/arm64` (`.github/workflows/docker.yml`).

The models live in two Hugging Face repos, [Netra-Lab/karna-base](https://huggingface.co/Netra-Lab/karna-base) and [Netra-Lab/karna-large](https://huggingface.co/Netra-Lab/karna-large):

```
Netra-Lab/karna-base/
├── onnx/                  what the package runs: model.onnx, vocab.json, netra-model.json
├── config.json  model.safetensors  vocab.json     PyTorch weights, for fine-tuning (training/)
└── README.md              the card (source: training/release/cards/karna-base.md)
```

The package loads each repo at the commit pinned in `karna/models.py` (`revision`), so uploading new weights never changes what an installed version runs.

## One-time setup

1. **PyPI Trusted Publishing.** On pypi.org → `netra-karna` → Manage → Publishing, add a GitHub publisher:
   - owner `netra-ai-lab`;
   - repository `Karna`;
   - workflow `pypi.yml`;
   - environment `pypi`.

   Before the first release, use a *pending* publisher. Then create the environment `pypi` in the GitHub repo (Settings → Environments).
2. **Make the image public** after the first release: GitHub → netra-ai-lab → Packages → `karna` → Package settings → Change visibility → Public.
3. **Docker Hub** (optional): the variable `DOCKERHUB_IMAGE` (e.g. `netralab/karna`) and the secrets `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`.

## Each release

0. **If the models changed:**
   - export and check each one: `CHECK=<real audio> training/export.sh <release folder>` (see [TRAINING.md](TRAINING.md#exporting-to-onnx));
   - log in yourself (`hf auth login`), then upload the folder as `onnx/`:

     ```bash
     hf upload Netra-Lab/karna-base runs/base/release/onnx/karna-base onnx --commit-message "ONNX model folder"
     ```

   - set `revision` for that model in `karna/models.py` to the commit the upload printed;
   - regenerate `tests/golden/golden.json` if the text changed (`python scripts/make_golden.py`).
1. **Check:**
   - `pytest tests` and `pytest tests --slow` (the published models against `tests/golden/`);
   - `python scripts/benchmark.py --device cpu`, compared with the last release.
2. **Bump** `version` in `pyproject.toml` (the only place it's written), then commit and push.
3. **Tag:**

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

4. **Watch the workflows.**
   - *docker*: builds both architectures, each running `docker/smoke_test.py` on the golden clips, then publishes `X.Y.Z`, `X.Y` and `latest`.
   - *pypi*: fails if the tag doesn't match the version.
5. **Check from a clean machine:**
   - `pip install "netra-karna[cpu]"` and the README examples;
   - `docker run --rm -p 8000:8000 ghcr.io/netra-ai-lab/karna:X.Y.Z`.

If the pypi workflow fails with `400 File already exists`, that version is already on PyPI. Release the next version instead of rerunning.

## Facts for the product page

Measured on 0.2.0 (`ghcr.io/netra-ai-lab/karna:0.2.0`, amd64), on Docker Desktop 29.8 with 16 CPUs and 7.2 GB of memory, sending 29 s clips to `/v1/transcribe?wait=true`. Memory is from `docker stats` after three requests per model; ONNX Runtime keeps its working memory, so it stays there.

| | |
| :--- | :--- |
| Package | wheel 0.16 MB (0.2.0); the models download on first use |
| Model download | karna-base 287 MB, karna-large 871 MB (fp32 ONNX) |
| Image | 1.16 GB compressed (amd64 and arm64), 1.42 GB unpacked: models 1.1 GB, Python environment 188 MB |
| Memory | 0.5 GB idle (karna-base loaded at start), 1.7 GB after karna-base requests, 4.0 GB with karna-large also in use. Give Docker 6 GB. |
| Speed | karna-base 8.7 s, karna-large 17.0 s per minute of audio on the CPU (16-thread AMD Zen 3 laptop, batch 8, `scripts/benchmark.py`); PyTorch: 8.8 s and 16.7 s |
| GPU | Not needed (CPU-only image); `pip install "netra-karna[gpu]"` for CUDA |
| Network | None after the pull: both models are inside the image |
