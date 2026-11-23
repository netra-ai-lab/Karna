# Using Karna

Karna runs in four ways. All of them use the same ONNX models and give the same text. Python and the server can also transcribe live, as audio arrives ([Python](#live-transcription), [WebSocket](#live-transcription-v1stream)).

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

### Live transcription

`Karna.stream()` gives text while audio is still arriving, from a microphone or a socket:

```python
live = asr.stream()                          # karna.StreamingTranscriber
for chunk in chunks:                         # mono 16 kHz float32 numpy arrays, any size
    for event in live.feed(chunk):
        print(event["type"], event["segment"], event["text"])
for event in live.finish():                  # closes the last utterance
    print(event)
print(live.text)                             # every final, joined with spaces
```

Karna is an offline model with no streaming state, so live text comes from **endpointed re-decoding**:
1. An energy VAD splits the audio into utterances at pauses.
2. While an utterance is open, all of it is decoded again after every 0.8 s of new audio and sent as a `partial`, which may still change.
3. When it closes, it is decoded once more and sent as a `final`, which never changes.

| Event | |
| --- | --- |
| `{"type": "partial", "segment": n, "text": ...}` | The open utterance so far. A newer partial or the final for `n` replaces it. |
| `{"type": "final", "segment": n, "text": ..., "start": s, "end": s}` | The utterance's text, and its place in the stream in seconds. An empty `text` means the segment was noise: clear its partial. |

An utterance closes:
- at a 600 ms pause once it holds 5 s of audio;
- at a 1.5 s pause, whatever its length;
- at 15 s, at its quietest point in the last 2 s.

Short pieces lose context: cut at every brief pause, a 2 s piece of Khmer can come back in Latin script. Each decode gets 0.5 s of silence on both sides, which brought karna-large back to the offline text on a quiet FLEURS clip.

The options are keyword arguments to `stream()`: `partial_interval_s`, `pad_ms`, `min_silence_ms`, `min_segment_s`, `long_silence_ms`, `max_segment_s`, `preroll_ms`, `tail_ms`, `min_speech_ms`, `abs_floor` and `ratio` (see `karna/streaming.py`).

How close it is: on the golden clips, fed in 100 ms chunks, 4 of 8 (clip, model) pairs give exactly the offline text, and the rest are 1–5 characters apart. A partial costs a decode of the whole open utterance, at the batch speed below: a 5 s utterance takes about 0.7 s with karna-base and 1.4 s with karna-large on the reference laptop CPU. A final arrives about 1 s (base) or 2–4 s (large) after the speaker stops.

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
| `WebSocket /v1/stream` | Live transcription (below) |

```bash
curl -F file=@speech.wav "http://localhost:8000/v1/transcribe?wait=true"
curl -H "X-API-Key: $KEY" -F file=@speech.flac -F model=karna-large http://localhost:8000/v1/transcribe
```

### Live transcription: `/v1/stream`

The web page's **Live** button uses this. Text appears as you speak, and the session is saved as a job in Recent when you stop.

1. Open `ws://host:port/v1/stream` and send `{"type": "start", "model": "karna-base", "api_key": "..."}`. `model` and `api_key` are optional. A browser can't set headers on a WebSocket, so the key goes in this message; other clients may send `X-API-Key` or `Authorization: Bearer` instead.
2. The server answers `{"type": "ready", "model", "sample_rate": 16000, "max_seconds"}`.
3. Send binary frames of **16 kHz mono little-endian int16 PCM**, of any size; about 100 ms each works well.
4. The server sends `partial` and `final` events, as in [Live transcription](#live-transcription) above.
5. Send `{"type": "stop"}`. The server closes the last utterance and sends `{"type": "done", "text", "job", "reason"}`, then closes the socket. `job` is the saved job, or `null` when no speech was heard. `reason` is `stop`, `limit` (the session reached `KARNA_MAX_STREAM_SECONDS`) or `disconnect`. Saved jobs have `"live": true` and `result.segments` with times; no audio is kept.

Live decodes run on the same worker as file jobs, ahead of any queued ones. A new partial is skipped while the last one is still decoding, so a slow machine sends fewer partials instead of falling behind. Close codes:
- **1003:** the first message wasn't a start message;
- **1008:** a missing or wrong key, or an unknown model (after an `error` event);
- **1013:** `KARNA_MAX_STREAMS` sessions are already open.

```python
import asyncio, json, websockets   # pip install websockets

async def live(pcm16: bytes):
    async with websockets.connect("ws://localhost:8000/v1/stream") as ws:
        await ws.send(json.dumps({"type": "start", "model": "karna-base"}))
        await ws.recv()                                     # ready
        for i in range(0, len(pcm16), 3200):                # 100 ms frames
            await ws.send(pcm16[i:i + 3200])
        await ws.send(json.dumps({"type": "stop"}))
        async for message in ws:
            event = json.loads(message)
            print(event["type"], event.get("text"))
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
| `KARNA_MAX_STREAM_SECONDS` | `600` | The longest live session; it then finishes as if stopped (`reason: "limit"`) |
| `KARNA_MAX_STREAMS` | `2` | Live sessions at once; more are refused with close code 1013 |

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
