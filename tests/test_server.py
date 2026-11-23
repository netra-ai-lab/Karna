"""Every route, with a fake engine (no models)."""

import io
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
sf = pytest.importorskip("soundfile")

from fastapi.testclient import TestClient  # noqa: E402

from karna.server.app import create_app  # noqa: E402
from karna.server.jobs import Engine  # noqa: E402
from karna.server.settings import Settings  # noqa: E402


class FakeEngine(Engine):
    def load(self, name):
        if name == "karna-large" and getattr(self, "broken", False):
            raise RuntimeError("boom")
        self.loaded[name] = object()
        self.device = "cpu"
        return self.loaded[name]

    def transcribe(self, name, wav):
        self.load(name)
        return f"{name}: {len(wav)} samples"


def wav_bytes(seconds=1.0, rate=16000, fmt="WAV"):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * rate), np.float32), rate, format=fmt)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, model="karna-base", max_seconds=5)
    engine = FakeEngine("karna-base", "cpu")
    with TestClient(create_app(settings, engine)) as c:
        c.engine = engine
        yield c


def wait_done(client, job_id):
    for _ in range(100):
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_health_info_and_ui(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/v1/info").json()
    assert info["default_model"] == "karna-base" and info["max_seconds"] == 5
    assert {m["name"] for m in info["models"]} == {"karna-base", "karna-large"}
    assert "<title>Karna</title>" in client.get("/").text
    assert client.get("/static/app.css").status_code == 200


def test_transcribe_wait(client):
    r = client.post("/v1/transcribe?wait=true", files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")})
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "done" and job["result"]["text"] == "karna-base: 16000 samples"
    assert job["duration_s"] == 1.0


def test_queue_poll_export_delete(client):
    r = client.post("/v1/transcribe", files={"file": ("b.flac", wav_bytes(0.5, 8000, "FLAC"))}, data={"model": "karna-large"})
    assert r.status_code == 202
    job = wait_done(client, r.json()["id"])
    assert job["result"]["text"] == "karna-large: 8000 samples"  # resampled to 16 kHz
    assert [j["id"] for j in client.get("/v1/jobs").json()] == [job["id"]]
    txt = client.get(f"/v1/jobs/{job['id']}/export?format=txt")
    assert txt.text == "karna-large: 8000 samples\n" and 'filename="b.txt"' in txt.headers["content-disposition"]
    assert client.get(f"/v1/jobs/{job['id']}/export?format=json").json()["id"] == job["id"]
    assert client.get(f"/v1/jobs/{job['id']}/export?format=docx").status_code == 422
    assert client.delete(f"/v1/jobs/{job['id']}").status_code == 204
    assert client.get(f"/v1/jobs/{job['id']}").status_code == 404
    assert client.delete(f"/v1/jobs/{job['id']}").status_code == 404


def test_rejections(client):
    too_long = client.post("/v1/transcribe", files={"file": ("long.wav", wav_bytes(6.0))})
    assert too_long.status_code == 422 and "limit is 5 s" in too_long.json()["detail"]
    garbage = client.post("/v1/transcribe", files={"file": ("x.wav", b"not audio")})
    assert garbage.status_code == 422
    unknown = client.post("/v1/transcribe", files={"file": ("a.wav", wav_bytes())}, data={"model": "nope"})
    assert unknown.status_code == 422
    assert client.get("/v1/jobs/../../etc").status_code == 404
    assert client.get("/v1/jobs/0123456789ab").status_code == 404


def test_failed_job_reports_error(client):
    client.engine.broken = True
    r = client.post("/v1/transcribe?wait=true", files={"file": ("a.wav", wav_bytes())}, data={"model": "karna-large"})
    job = r.json()
    assert r.status_code == 202 and job["status"] == "failed" and "boom" in job["error"]


def test_api_key(tmp_path):
    settings = Settings(data_dir=tmp_path, api_key="s3cret")
    with TestClient(create_app(settings, FakeEngine("karna-base", "cpu"))) as c:
        assert c.get("/v1/info").json()["auth"] is True  # open, so the UI can ask for the key
        assert c.get("/v1/jobs").status_code == 401
        assert c.get("/v1/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/v1/jobs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/v1/jobs", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_jobs_survive_restart(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings, FakeEngine("karna-base", "cpu"))) as c:
        job = c.post("/v1/transcribe?wait=true", files={"file": ("a.wav", wav_bytes())}).json()
    with TestClient(create_app(settings, FakeEngine("karna-base", "cpu"))) as c:
        assert c.get(f"/v1/jobs/{job['id']}").json()["result"]["text"] == job["result"]["text"]
