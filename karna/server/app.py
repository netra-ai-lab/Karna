"""Routes. ``create_app()`` is importable for tests (pass a fake ``engine``)."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..audio import SAMPLE_RATE, load_audio
from ..inference.bias import LANGUAGES
from ..models import MODELS
from ..streaming import Segment, StreamingTranscriber
from .jobs import Engine, JobStore
from .settings import Settings

STATIC = Path(__file__).with_name("static")
logger = logging.getLogger(__name__)
#: re-decode the open segment after this much new audio (latest wins: none while one is running);
#: padding (0.5 s) and context (1 s) are StreamingTranscriber's defaults
PARTIAL_SECONDS = 0.8


class LiveSession:
    """One /v1/stream WebSocket: a ``StreamingTranscriber`` whose decodes run on the worker
    (live first, ahead of file jobs), sending partials and finals as they come."""

    def __init__(self, ws: WebSocket, model: str, store: JobStore, engine: Engine, max_seconds: float,
                 partial_model: str | None = None, language: str = "auto", decoder: str = "auto") -> None:
        self.ws, self.model, self.store, self.engine = ws, model, store, engine
        self.language = language
        self.decoder = decoder
        # Drafts come from the chosen model too. karna-base drafts for karna-large arrived every
        # ~1 s, but were mostly unreadable (capture CER 0.55 vs 0.24), so what was on screen made
        # no sense until Stop. karna-large drafts come every 1-3 s and finals 0-2 s later.
        # A client may still ask for faster drafts with ``partial_model``.
        self.partial_model = partial_model or model
        # drafts are re-decoded often, so always greedy; finals use ``decoder`` (auto: beam with an LM)
        final_opts = {} if decoder == "auto" else {"decoder": decoder}
        draft_opts = {"decoder": "greedy"}
        self.live = StreamingTranscriber(
            lambda wav: engine.transcribe(self.partial_model, wav, language, **draft_opts),
            decode_context=lambda wav, skip: engine.transcribe_tail(self.partial_model, wav, skip, language,
                                                                    **draft_opts),
            final_decode=lambda wav: engine.transcribe(model, wav, language, **final_opts),
            final_decode_context=lambda wav, skip: engine.transcribe_tail(model, wav, skip, language, **final_opts),
            partial_interval_s=PARTIAL_SECONDS, sample_rate=SAMPLE_RATE)
        self.limit = int(max_seconds * SAMPLE_RATE)
        self.send_lock = asyncio.Lock()
        self.connected = True
        self.samples = 0
        self.odd = b""                         # half of an int16 sample split across frames
        self.partial: asyncio.Task | None = None
        self.closed: set[int] = set()
        self.decode_s = 0.0

    @property
    def finals(self) -> list[dict]:
        return self.live.finals

    async def send(self, message: dict) -> None:
        if not self.connected:
            return
        async with self.send_lock:
            try:
                await self.ws.send_json(message)
            except Exception:
                self.connected = False

    async def decode(self, job, seconds: float, model: str) -> str:
        t0 = time.perf_counter()
        text = await asyncio.wrap_future(self.store.submit(job))
        took = time.perf_counter() - t0
        self.decode_s += took
        logger.info("live %s: %.1f s of audio in %.2f s", model, seconds, took)
        return text

    async def run(self) -> None:
        reason = "stop"
        await self.send({"type": "ready", "model": self.model, "partial_model": self.partial_model,
                         "language": self.language, "decoder": self.decoder,
                         "sample_rate": SAMPLE_RATE,
                         "max_seconds": self.limit / SAMPLE_RATE})
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    self.connected, reason = False, "disconnect"
                    break
                if message.get("bytes") is not None:
                    await self.audio(message["bytes"])
                    if self.samples >= self.limit:
                        reason = "limit"
                        break
                elif message.get("text"):
                    try:
                        if json.loads(message["text"]).get("type") == "stop":
                            break
                    except (ValueError, AttributeError):
                        pass  # not a control message we know: ignore it
            if self.partial is not None:
                self.partial.cancel()
            for seg in self.live.flush():
                await self.final(seg)
            text = self.live.text
            job = self.save(text) if text else None
            await self.send({"type": "done", "text": text, "job": job, "reason": reason})
        except Exception as e:
            logger.exception("live session failed")
            await self.send({"type": "error", "detail": f"{type(e).__name__}: {e}"})
        if self.connected:
            try:
                await self.ws.close()
            except Exception:
                pass

    def save(self, text: str) -> dict:
        seconds = round(self.samples / SAMPLE_RATE, 3)
        result = {"text": text, "model": self.model, "device": self.engine.device, "duration_s": seconds,
                  "seconds": round(self.decode_s, 3),
                  "segments": [{k: f[k] for k in ("text", "start", "end")} for f in self.finals if f["text"]]}
        return self.store.record(f"Live {time.strftime('%Y-%m-%d %H:%M:%S')}", self.model, seconds, result,
                                 self.language)

    async def audio(self, data: bytes) -> None:
        data = self.odd + data
        cut = len(data) - len(data) % 2
        self.odd = data[cut:]
        if not cut:
            return
        pcm = np.frombuffer(data[:cut], dtype="<i2")[: self.limit - self.samples].astype(np.float32) / 32768.0
        self.samples += len(pcm)
        for seg in self.live.push(pcm):
            await self.final(seg)
        if self.partial is not None and not self.partial.done():
            return  # latest wins: no new draft while one is being decoded
        due = self.live.due_partial()
        if due is not None:
            seg_id, audio = due
            self.partial = asyncio.create_task(self.send_partial(seg_id, self.live.job(audio), len(audio)))

    async def send_partial(self, seg_id: int, job, samples: int) -> None:
        try:
            text = await self.decode(job, samples / SAMPLE_RATE, self.partial_model)
        except Exception as e:
            await self.send({"type": "error", "detail": f"{type(e).__name__}: {e}"})
            return
        if seg_id not in self.closed:  # its final may already have been sent
            await self.send({"type": "partial", "segment": seg_id, "text": text})

    async def final(self, seg: Segment) -> None:
        self.closed.add(seg.id)
        done = self.live.close(seg)
        if done is None:
            text = await self.decode(self.live.job(seg.audio, final=True), len(seg.audio) / SAMPLE_RATE, self.model)
            done = [self.live.accept(seg, text)]
        for event in done:
            await self.send(event)


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = engine or Engine(settings.model, settings.device)
    store = JobStore(settings.data_dir, engine, keep=settings.keep_jobs)

    def decode(data: bytes):
        return load_audio(data, SAMPLE_RATE)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.start(decode)
        yield
        store.stop()

    app = FastAPI(title="Karna", version=__version__, lifespan=lifespan,
                  description="Khmer and English speech recognition. Upload audio, get text.")
    app.state.settings, app.state.store, app.state.engine = settings, store, engine

    def key_ok(headers, given: str = "") -> bool:
        if not settings.api_key:
            return True
        given = given or headers.get("x-api-key") or ""
        header = headers.get("authorization") or ""
        if header.lower().startswith("bearer "):
            given = header[7:].strip()
        return hmac.compare_digest(given.encode(), settings.api_key.encode())

    def auth(request: Request) -> None:
        if not key_ok(request.headers):
            raise HTTPException(401, "missing or wrong API key", headers={"WWW-Authenticate": "Bearer"})

    def job_or_404(job_id: str) -> dict:
        try:
            return store.read(job_id)
        except KeyError:
            raise HTTPException(404, f"no job {job_id!r}") from None

    @app.get("/health", include_in_schema=False)
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/info")
    def info() -> dict:
        """Version, device and readiness, and the options the UI may offer."""
        return {
            "name": "Karna", "version": __version__,
            "device": engine.device, "models_ready": engine.ready, "error": engine.error,
            "default_model": settings.model,
            "models": [{"name": n, "params": m["params"], "about": m["about"], "loaded": n in engine.loaded}
                       for n, m in MODELS.items()],
            "sample_rate": SAMPLE_RATE, "max_seconds": settings.max_seconds,
            "max_upload_mb": settings.max_upload_mb, "auth": bool(settings.api_key),
            "streaming": True, "max_stream_seconds": settings.max_stream_seconds, "languages": list(LANGUAGES),
        }

    @app.post("/v1/transcribe", dependencies=[Depends(auth)], status_code=202)
    async def transcribe(file: UploadFile = File(..., description="audio: wav, flac, ogg, mp3, ..."),
                         model: str | None = Form(None, description="karna-base (default) or karna-large"),
                         language: str = Form("auto", description="auto (Khmer or English), km (no Latin letters) or en"),
                         wait: bool = Query(False, description="block until the text is ready (up to 5 minutes)")):
        """Queue one audio file; returns the job. With ``wait=true``, returns the finished job."""
        model = model or settings.model
        if model not in MODELS:
            raise HTTPException(422, f"unknown model {model!r}: use one of {sorted(MODELS)}")
        if language not in LANGUAGES:
            raise HTTPException(422, f"unknown language {language!r}: use one of {list(LANGUAGES)}")
        limit = int(settings.max_upload_mb * 1024 * 1024)
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"the file is larger than {settings.max_upload_mb:g} MB")
        try:
            wav = await run_in_threadpool(decode, data)
        except Exception:
            raise HTTPException(422, "could not read the audio: send wav, flac, ogg or mp3") from None
        seconds = len(wav) / SAMPLE_RATE
        if seconds > settings.max_seconds:
            raise HTTPException(422, f"the audio is {seconds:.1f} s; the limit is {settings.max_seconds:g} s "
                                     "(KARNA_MAX_SECONDS)")
        job = store.create(data, file.filename or "audio", model, seconds, language)
        if wait:
            job = await run_in_threadpool(store.wait, job["id"], 300.0)
            return JSONResponse(job, status_code=200 if job["status"] == "done" else 202)
        return job

    live = {"sessions": 0}

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        """Live transcription: send {"type": "start", "model": ..., "api_key": ...}, then binary frames
        of 16 kHz mono little-endian int16 PCM, then {"type": "stop"}. Receive "ready", "partial" and
        "final" events, then "done" with the saved job (docs/USAGE.md)."""
        await ws.accept()
        try:
            start = await asyncio.wait_for(ws.receive_json(), 15)
        except Exception:
            start = None
        if not isinstance(start, dict) or start.get("type") != "start":
            await ws.close(1003, "send a start message first")
            return
        if not key_ok(ws.headers, str(start.get("api_key") or "")):
            await ws.close(1008, "missing or wrong API key")
            return
        model = start.get("model") or settings.model
        partial_model = start.get("partial_model") or None
        language = start.get("language") or "auto"
        decoder = start.get("decoder") or "auto"
        if decoder not in ("auto", "greedy", "beam"):
            await ws.send_json({"type": "error", "detail": f"unknown decoder {decoder!r}: use auto, greedy or beam"})
            await ws.close(1008, "unknown decoder")
            return
        if language not in LANGUAGES:
            await ws.send_json({"type": "error", "detail": f"unknown language {language!r}: use one of {list(LANGUAGES)}"})
            await ws.close(1008, "unknown language")
            return
        for name in (model, partial_model):
            if name is not None and name not in MODELS:
                await ws.send_json({"type": "error", "detail": f"unknown model {name!r}: use one of {sorted(MODELS)}"})
                await ws.close(1008, "unknown model")
                return
        if live["sessions"] >= settings.max_streams:
            await ws.close(1013, "busy: too many live sessions")
            return
        live["sessions"] += 1
        try:
            await LiveSession(ws, model, store, engine, settings.max_stream_seconds, partial_model, language,
                              decoder).run()
        finally:
            live["sessions"] -= 1

    @app.get("/v1/jobs", dependencies=[Depends(auth)])
    def jobs() -> list[dict]:
        return store.list()

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
    def job(job_id: str) -> dict:
        return job_or_404(job_id)

    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(auth)], status_code=204)
    def delete(job_id: str) -> Response:
        try:
            store.delete(job_id)
        except KeyError:
            raise HTTPException(404, f"no job {job_id!r}") from None
        return Response(status_code=204)

    @app.get("/v1/jobs/{job_id}/export", dependencies=[Depends(auth)])
    def export(job_id: str, format: str = Query("txt", pattern="^(txt|json)$")):
        job = job_or_404(job_id)
        if job["status"] != "done":
            raise HTTPException(409, f"the job is {job['status']}")
        stem = Path(job["filename"]).stem or job_id
        if format == "json":
            return JSONResponse(job, headers={"Content-Disposition": f'attachment; filename="{stem}.json"'})
        return PlainTextResponse(job["result"]["text"] + "\n",
                                 headers={"Content-Disposition": f'attachment; filename="{stem}.txt"'})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
