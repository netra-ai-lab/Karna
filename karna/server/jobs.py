"""Jobs as plain directories, and the one inference worker thread that owns every model.

    <data_dir>/<job_id>/
        job.json      id, status, model, filename, duration, times, error
        upload.bin    the upload (deleted once processed)
        result.json   {"text": ..., "model": ..., "device": ..., "duration_s": ..., "seconds": ...}

Requests enqueue a job and return at once; the worker runs jobs one at a time, so one CPU or
GPU serves one job at a time and models are never shared across threads. Jobs queued or
running when the server stopped are queued again at start.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

JOB_ID = re.compile(r"^[0-9a-f]{12}$")


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

    def transcribe(self, name: str, wav) -> str:
        return self.load(name).transcribe(wav)


class JobStore:
    def __init__(self, root: Path, engine: Engine, keep: int = 200) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.keep = keep
        self.queue: queue.Queue[str | None] = queue.Queue()
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

    def create(self, data: bytes, filename: str, model: str, duration_s: float) -> dict:
        job_id = uuid.uuid4().hex[:12]
        folder = self.path(job_id)
        folder.mkdir()
        (folder / "upload.bin").write_bytes(data)
        job = {"id": job_id, "status": "queued", "model": model, "filename": filename,
               "duration_s": round(duration_s, 3), "created": time.time(), "finished": None, "error": None}
        with self.lock:
            self._write(job)
            self.done_events[job_id] = threading.Event()
        self.queue.put(job_id)
        self._prune()
        return job

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
                    self.queue.put(job["id"])
                else:
                    self._update(job["id"], status="failed", error="the server stopped while this job ran")
        self.thread = threading.Thread(target=self._work, args=(decode,), name="karna-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.queue.put(None)

    def _work(self, decode: Callable[[bytes], object]) -> None:
        try:
            self.engine.load(self.engine.default_model)
            self.engine.ready = True
        except Exception as e:  # reported by /v1/info; jobs then fail with the same message
            logger.exception("loading %s failed", self.engine.default_model)
            self.engine.error = f"{type(e).__name__}: {e}"
        while True:
            job_id = self.queue.get()
            if job_id is None:
                return
            self._run(job_id, decode)

    def _run(self, job_id: str, decode: Callable[[bytes], object]) -> None:
        folder = self.path(job_id)
        if not (folder / "job.json").exists():  # deleted while queued
            return
        try:
            job = self._update(job_id, status="running", started=time.time())
            start = time.perf_counter()
            wav = decode((folder / "upload.bin").read_bytes())
            text = self.engine.transcribe(job["model"], wav)
            result = {"text": text, "model": job["model"], "device": self.engine.device,
                      "duration_s": job["duration_s"], "seconds": round(time.perf_counter() - start, 3)}
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
