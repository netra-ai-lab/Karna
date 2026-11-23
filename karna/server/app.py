"""Routes. ``create_app()`` is importable for tests (pass a fake ``engine``)."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..audio import SAMPLE_RATE, load_audio
from ..models import MODELS
from .jobs import Engine, JobStore
from .settings import Settings

STATIC = Path(__file__).with_name("static")


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

    def auth(request: Request) -> None:
        if not settings.api_key:
            return
        given = request.headers.get("x-api-key") or ""
        header = request.headers.get("authorization") or ""
        if header.lower().startswith("bearer "):
            given = header[7:].strip()
        if not hmac.compare_digest(given.encode(), settings.api_key.encode()):
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
        }

    @app.post("/v1/transcribe", dependencies=[Depends(auth)], status_code=202)
    async def transcribe(file: UploadFile = File(..., description="audio: wav, flac, ogg, mp3, ..."),
                         model: str | None = Form(None, description="karna-base (default) or karna-large"),
                         wait: bool = Query(False, description="block until the text is ready (up to 5 minutes)")):
        """Queue one audio file; returns the job. With ``wait=true``, returns the finished job."""
        model = model or settings.model
        if model not in MODELS:
            raise HTTPException(422, f"unknown model {model!r}: use one of {sorted(MODELS)}")
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
            raise HTTPException(422, f"the audio is {seconds:.1f} s; the limit is {settings.max_seconds:g} s. "
                                     "Karna decodes each file in one pass: split long recordings into utterances.")
        job = store.create(data, file.filename or "audio", model, seconds)
        if wait:
            job = await run_in_threadpool(store.wait, job["id"], 300.0)
            return JSONResponse(job, status_code=200 if job["status"] == "done" else 202)
        return job

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
