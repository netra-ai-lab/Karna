"""Jobs as plain directories, and the one inference worker thread that owns every model.

    <data_dir>/<job_id>/
        job.json      id, status, model, filename, duration, times, error
        upload.bin    the upload (deleted once processed)
        result.json   {"text": ..., "model": ..., "device": ..., "duration_s": ..., "seconds": ...}

Requests enqueue a job and return at once; the worker runs jobs one at a time, so one CPU or
GPU serves one job at a time and models are never shared across threads. Jobs queued or
running when the server stopped are queued again at start.

Live transcription (``/v1/stream``) runs its decodes on the same worker through ``submit``, ahead
of queued file jobs, and saves each finished session as a done job with ``record``.
"""

from __future__ import annotations

import itertools
import json
import logging
import queue
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

JOB_ID = re.compile(r"^[0-9a-f]{12}$")

#: worker queue priorities: live decodes first, then file jobs, then the stop sentinel
LIVE, FILE, STOP = 0, 1, 2


class Engine:
    """The models, loaded and run on the worker thread only."""

    def __init__(self, default_model: str, device: str | None) -> None:
        self.default_model = default_model
        self.requested_device = device
        self.device: str | None = None
        self.ready = False
        self.error: str | None = None
        self.loaded: dict[str, Any] = {}

    def load(self, name: str):
        if name not in self.loaded:
            from ..asr import load_model

            self.loaded[name] = load_model(name, self.requested_device)
            self.device = self.loaded[name].device
        return self.loaded[name]

    def transcribe(self, name: str, wav, language: str = "auto", decoder: str = "auto") -> str:
        extra = {} if decoder == "auto" else {"decoder": decoder}
        return self.load(name).transcribe(wav, language=language, **extra)

    def transcribe_long(self, name: str, wav, language: str = "auto") -> dict:
        """``{"text", "segments"}`` for a recording longer than one pass (``Karna.transcribe_long``)."""
        return self.load(name).transcribe_long(wav, language=language)

    def transcribe_tail(self, name: str, wav, skip: int, language: str = "auto", decoder: str = "auto") -> str:
        """Text for ``wav[skip:]``, with ``wav[:skip]`` as context (``Recognizer.transcribe_tail``)."""
        from ..text import clean_khmer

        asr = self.load(name)
        decoder = asr._decoder(decoder) if hasattr(asr, "_decoder") else decoder
        extra = {} if decoder in ("auto", "greedy") else {"decoder": decoder}
        return clean_khmer(asr.recognizer.transcribe_tail(wav, skip, language=language, **extra))


class JobStore:
    def __init__(self, root: Path, engine: Engine, keep: int = 200) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.keep = keep
        self.queue: queue.PriorityQueue[tuple[int, int, Any]] = queue.PriorityQueue()
        self._seq = itertools.count()
        self.lock = threading.Lock()
        self.done_events: dict[str, threading.Event] = {}
        self.thread: threading.Thread | None = None

    # ---- files

    def path(self, job_id: str) -> Path:
        if not JOB_ID.match(job_id):
            raise KeyError(job_id)
        return self.root / job_id

    def read(self, job_id: str) -> dict:
        try:
            job = json.loads((self.path(job_id) / "job.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise KeyError(job_id) from e
        result = self.path(job_id) / "result.json"
        if job["status"] == "done" and result.exists():
            job["result"] = json.loads(result.read_text(encoding="utf-8"))
        return job

    def _write(self, job: dict) -> None:
        path = self.path(job["id"]) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _update(self, job_id: str, **fields) -> dict:
        with self.lock:
            job = json.loads((self.path(job_id) / "job.json").read_text(encoding="utf-8"))
            job.update(fields)
            self._write(job)
            return job

    # ---- API

    def _enqueue(self, priority: int, item: Any) -> None:
        self.queue.put((priority, next(self._seq), item))

    def create(self, data: bytes, filename: str, model: str, duration_s: float, language: str = "auto") -> dict:
        job_id = uuid.uuid4().hex[:12]
        folder = self.path(job_id)
        folder.mkdir()
        (folder / "upload.bin").write_bytes(data)
        job = {"id": job_id, "status": "queued", "model": model, "filename": filename,
               "language": language, "duration_s": round(duration_s, 3), "created": time.time(), "finished": None,
               "error": None}
        with self.lock:
            self._write(job)
            self.done_events[job_id] = threading.Event()
        self._enqueue(FILE, job_id)
        self._prune()
        return job

    def record(self, filename: str, model: str, duration_s: float, result: dict, language: str = "auto") -> dict:
        """Save a finished transcript (a live session) as a done job; no audio is kept."""
        job_id = uuid.uuid4().hex[:12]
        folder = self.path(job_id)
        folder.mkdir()
        now = time.time()
        (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        job = {"id": job_id, "status": "done", "model": model, "filename": filename, "live": True, "language": language,
               "duration_s": round(duration_s, 3), "created": now, "started": now, "finished": now, "error": None}
        with self.lock:
            self._write(job)
        self._prune()
        return self.read(job_id)

    def submit(self, fn: Callable[[], Any]) -> Future:
        """Run ``fn`` on the worker thread ahead of queued file jobs (live decodes)."""
        future: Future = Future()
        self._enqueue(LIVE, (fn, future))
        return future

    def list(self) -> list[dict]:
        jobs = []
        for folder in self.root.iterdir():
            if folder.is_dir() and JOB_ID.match(folder.name):
                try:
                    jobs.append(self.read(folder.name))
                except KeyError:
                    continue
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def delete(self, job_id: str) -> None:
        folder = self.path(job_id)
        if not (folder / "job.json").exists():
            raise KeyError(job_id)
        shutil.rmtree(folder, ignore_errors=True)

    def wait(self, job_id: str, timeout: float) -> dict:
        event = self.done_events.get(job_id)
        if event is not None:
            event.wait(timeout)
        return self.read(job_id)

    def _prune(self) -> None:
        finished = [j for j in self.list() if j["status"] in ("done", "failed")]
        for job in finished[self.keep:]:
            shutil.rmtree(self.path(job["id"]), ignore_errors=True)

    # ---- worker

    def start(self, decode: Callable[[bytes], object]) -> None:
        """Requeue unfinished jobs, then load the default model and run jobs on one thread."""
        for job in reversed(self.list()):
            if job["status"] in ("queued", "running"):
                if (self.path(job["id"]) / "upload.bin").exists():
                    self._update(job["id"], status="queued")
                    self.done_events[job["id"]] = threading.Event()
                    self._enqueue(FILE, job["id"])
                else:
                    self._update(job["id"], status="failed", error="the server stopped while this job ran")
        self.thread = threading.Thread(target=self._work, args=(decode,), name="karna-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._enqueue(STOP, None)

    def _work(self, decode: Callable[[bytes], object]) -> None:
        try:
            self.engine.load(self.engine.default_model)
            self.engine.ready = True
        except Exception as e:  # reported by /v1/info; jobs then fail with the same message
            logger.exception("loading %s failed", self.engine.default_model)
            self.engine.error = f"{type(e).__name__}: {e}"
        while True:
            _, _, item = self.queue.get()
            if item is None:
                return
            if isinstance(item, tuple):
                fn, future = item
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(fn())
                    except BaseException as e:  # noqa: BLE001 - handed to the waiting request
                        future.set_exception(e)
            else:
                self._run(item, decode)

    def _run(self, job_id: str, decode: Callable[[bytes], object]) -> None:
        folder = self.path(job_id)
        if not (folder / "job.json").exists():  # deleted while queued
            return
        try:
            job = self._update(job_id, status="running", started=time.time())
            start = time.perf_counter()
            wav = decode((folder / "upload.bin").read_bytes())
            from ..asr import SINGLE_PASS_S

            language = job.get("language", "auto")
            segments = None
            if len(wav) / 16000 > SINGLE_PASS_S:  # split at pauses, decoded with left context
                long = self.engine.transcribe_long(job["model"], wav, language=language)
                text, segments = long["text"], long["segments"]
            else:
                text = self.engine.transcribe(job["model"], wav, language=language)
            result = {"text": text, "model": job["model"], "device": self.engine.device,
                      "duration_s": job["duration_s"], "seconds": round(time.perf_counter() - start, 3)}
            if segments is not None:
                result["segments"] = segments
            (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            (folder / "upload.bin").unlink(missing_ok=True)
            self._update(job_id, status="done", finished=time.time())
        except Exception as e:
            if folder.exists():
                logger.exception("job %s failed", job_id)
                self._update(job_id, status="failed", finished=time.time(), error=f"{type(e).__name__}: {e}")
        finally:
            event = self.done_events.pop(job_id, None)
            if event is not None:
                event.set()
