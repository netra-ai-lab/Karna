# Using Karna

Karna runs in four ways. All of them use the same ONNX models and give the same text.

| Way in | Install | For |
| --- | --- | --- |
| [Docker image](#docker) | `docker run ... ghcr.io/netra-ai-lab/karna` | The web app and REST API, with both models inside and no Python needed |
| [Python](#python) | `pip install "netra-karna[cpu]"` | Scripts and pipelines |
| [Command line](#command-line) | the same package | Shell scripts and batch jobs |
| [Web app and REST API](#web-app-and-rest-api) | `pip install "netra-karna[full]"`, then `karna serve` | The Docker app, run without Docker |

Pick one ONNX Runtime extra: `cpu` (`onnxruntime`) or `gpu` (`onnxruntime-gpu`, CUDA). Installing both breaks both, because the two packages share one module name. Without either, `import karna` still works, but loading a model tells you which extra to install.

## Python

```python
import karna

asr = karna.load_model("karna-base")         # loads once; reuse it
asr.transcribe("speech.wav")                 # -> "text"
asr.transcribe(["a.wav", "b.flac"])          # -> ["text", "text"], same order
karna.transcribe("speech.wav")               # one call; the model is cached per process
```

| Call | |
| --- | --- |
| `karna.load_model(name_or_path="karna-base", device=None)` | A `Karna` object. `name_or_path` is a short name (`karna-base`, `karna-large`), a Hugging Face repo id with an `onnx/` folder, an `hf://repo/folder@revision` source, or a local model folder. |
| `Karna.transcribe(audio, batch_size=8)` | Text for one input, or a list of texts for a list. Inputs are sorted by length and batched, then returned in your order. |
| `karna.transcribe(audio, model="karna-base", device=None, batch_size=8)` | Both steps in one call. |
| `karna.available_models()` | `{"karna-base": "Netra-Lab/karna-base", ...}` |
| `asr.device`, `asr.name`, `asr.num_parameters`, `asr.sample_rate` | What was loaded. `num_parameters` comes from the model folder's `netra-model.json`. |

An **input** is:
- a file path (wav, flac, ogg, mp3 and anything else libsndfile reads);
- the file's bytes;
- a mono 16 kHz waveform, as a numpy array (a torch tensor also works; Karna never imports torch);
- a `(waveform, sample_rate)` tuple, resampled to 16 kHz with soxr.

Stereo is averaged to mono. Each input is decoded in one pass, so split recordings longer than about 30 s into utterances first.

PyTorch checkpoints and release folders (`config.json` + `model.safetensors`) don't load here. They run in the training package (`karna_train.release.load_release`), and `training/export.sh` turns them into a model folder.

## Command line

```
karna transcribe AUDIO... [--model karna-base] [--device auto] [--batch-size 8] [--format tsv]
karna models
karna download [MODEL...] [--device cpu]
karna serve [--host 127.0.0.1] [--port 8000] [--model karna-base] [--device auto]
karna --version
```

| Command | |
| --- | --- |
| `transcribe` | `--format tsv` (default) prints `path<TAB>text` per file; `txt` prints the text only; `json` prints `{"model", "device", "results": [{"file", "text"}, ...]}`. |
| `models` | The released models, their size and repo. |
| `download` | Fetches models now, for offline use later (then set `HF_HUB_OFFLINE=1`). `--device` fetches only the files that device needs. |
| `serve` | The web app and REST API. Needs the `server` extra; without it, the command names the extra. |

`python -m karna` works the same as `karna`.

## Web app and REST API

`karna serve` (or the Docker image) serves the web page at `/` and the API under `/v1`. Interactive API docs are at `/docs`.

One worker thread runs the models, so jobs run one at a time, in order. A job is a directory under `KARNA_DATA_DIR`, so jobs survive restarts. The uploaded audio is deleted once the job has run.

| Route | |
| --- | --- |
| `GET /health` | `{"status": "ok"}`; the Docker HEALTHCHECK |
| `GET /v1/info` | Version, device, `models_ready`, the models and the limits. Needs no key. |
| `POST /v1/transcribe` | Form fields `file` (the audio) and `model` (optional). Returns 202 with the queued job. With `?wait=true`, it blocks for up to 5 minutes and returns the finished job. |
| `GET /v1/jobs` | All jobs, newest first |
| `GET /v1/jobs/{id}` | One job: `status` is `queued`, `running`, `done` or `failed` (with `error`); the text is in `result.text` |
| `DELETE /v1/jobs/{id}` | 204 |
| `GET /v1/jobs/{id}/export?format=txt\|json` | The text, or the whole job, as a download |

```bash
curl -F file=@speech.wav "http://localhost:8000/v1/transcribe?wait=true"
curl -H "X-API-Key: $KEY" -F file=@speech.flac -F model=karna-large http://localhost:8000/v1/transcribe
```

Errors are JSON `{"detail": "..."}`:
- **401:** a missing or wrong key;
- **404:** no such job;
- **409:** an export of an unfinished job;
- **413:** an upload over the size limit;
- **422:** audio that can't be read, audio over the duration limit, or an unknown model.

## Environment variables

| Variable | Default | |
| --- | --- | --- |
| `KARNA_DEVICE` | `auto` | The device when none is passed (see below) |
| `KARNA_BUNDLES` | *(unset)* | A folder of local model folders by short name (`<dir>/karna-base/netra-model.json`). Names missing from it come from Hugging Face. |
| `KARNA_ORT_SPIN` | `0` | `1` keeps ONNX Runtime's threads spinning between calls: a little faster under constant load, but busy CPUs when idle |
| `HF_HOME`, `HF_HUB_OFFLINE` | | Hugging Face's cache folder, and offline mode |
| `KARNA_HOST`, `KARNA_PORT` | `127.0.0.1`, `8000` | Where `karna serve` listens (the Docker image listens on `0.0.0.0`) |
| `KARNA_MODEL` | `karna-base` | The model the server loads at start; the other loads on first use |
| `KARNA_DATA_DIR` | `~/.cache/karna/server` | Where jobs live (`/data` in Docker) |
| `KARNA_API_KEY` | *(unset)* | Requires `X-API-Key: <key>` or `Authorization: Bearer <key>` on `/v1/*` except `/v1/info` |
| `KARNA_MAX_SECONDS` | `30` | Longer audio is refused with 422 |
| `KARNA_MAX_UPLOAD_MB` | `50` | Larger uploads are refused with 413 |
| `KARNA_KEEP_JOBS` | `200` | Finished jobs kept on disk; the oldest are deleted first |

## Devices

| Device | Needs | |
| --- | --- | --- |
| `cpu` | `[cpu]` or `[gpu]` | Always available |
| `cuda`, `cuda:N` | `[gpu]`, an NVIDIA driver and CUDA 12 | |
| `tensorrt`, `tensorrt:N` | `[gpu]` and TensorRT | Falls back to CUDA, then CPU, op by op. Never picked by `auto`. |
| `openvino` | `onnxruntime-openvino` | Intel CPUs and iGPUs |
| `auto` | | `cuda` if usable, else `cpu` |

If a GPU device can't start, Karna logs a warning and runs on the CPU.

## Docker

```bash
docker run -d --name karna -p 8000:8000 -v karna-data:/data ghcr.io/netra-ai-lab/karna
docker run -d -p 8000:8000 -e KARNA_API_KEY=change-me -e KARNA_MODEL=karna-large ghcr.io/netra-ai-lab/karna
docker compose up --build              # from a checkout, see docker-compose.yml
```

The image is CPU-only. It has both models inside and runs offline (`HF_HUB_OFFLINE=1`), as a non-root user, with jobs in the `/data` volume. Every image build runs `docker/smoke_test.py`, which transcribes the test clips with both models. For a GPU, install the `gpu` extra on the host instead.
