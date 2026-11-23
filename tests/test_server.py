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

    def transcribe(self, name, wav, language="auto"):
        self.load(name)
        self.__dict__.setdefault("calls", []).append((name, len(wav), 0))
        self.__dict__.setdefault("languages", []).append(language)
        return f"{name}: {len(wav)} samples"

    def transcribe_tail(self, name, wav, skip, language="auto"):
        self.load(name)
        self.__dict__.setdefault("languages", []).append(language)
        self.__dict__.setdefault("calls", []).append((name, len(wav), skip))
        return f"{name}: {len(wav) - skip} samples after context"


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


def test_transcribe_language(client):
    r = client.post("/v1/transcribe?wait=true", files={"file": ("a.wav", wav_bytes())}, data={"language": "km"})
    assert r.json()["language"] == "km" and client.engine.languages[-1] == "km"
    r = client.post("/v1/transcribe", files={"file": ("a.wav", wav_bytes())}, data={"language": "fr"})
    assert r.status_code == 422 and "language" in r.json()["detail"]
    assert client.get("/v1/info").json()["languages"] == ["auto", "km", "en"]


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


# ---- live transcription (/v1/stream)

def pcm16(signal):
    return (np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes()


def speech_like(rate=16000):
    t = np.arange(int(1.5 * rate)) / rate
    burst = 0.3 * np.sin(2 * np.pi * 220 * t)
    gap = np.zeros(2 * rate)  # over 1.5 s, so the two bursts are two segments
    return np.concatenate([gap[:8000], burst, gap, burst, gap[:4000]]).astype(np.float32)


def stream(client, chunks, start=None):
    """Run one live session; returns every message the server sent."""
    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json(start or {"type": "start"})
        messages = [ws.receive_json()]
        assert messages[0]["type"] == "ready", messages
        for chunk in chunks:
            ws.send_bytes(chunk)
        ws.send_json({"type": "stop"})
        while messages[-1]["type"] not in ("done", "error"):
            messages.append(ws.receive_json())
    return messages


def test_stream_drafts_from_the_chosen_model(client):
    messages = stream(client, [pcm16(speech_like())], {"type": "start", "model": "karna-large"})
    assert messages[0]["partial_model"] == "karna-large"
    assert all(m["text"].startswith("karna-large: ") for m in messages if m["type"] in ("partial", "final") and m["text"])


def test_stream(client):
    data = pcm16(speech_like())
    chunks = [data[i:i + 3201] for i in range(0, len(data), 3201)]  # odd sizes split samples
    messages = stream(client, chunks, {"type": "start", "model": "karna-large", "language": "km",
                                       "partial_model": "karna-base"})
    assert messages[0]["model"] == "karna-large" and messages[0]["partial_model"] == "karna-base"
    assert messages[0]["language"] == "km" and set(client.engine.languages) == {"km"}
    finals = [m for m in messages if m["type"] == "final" and m["text"]]
    assert [f["segment"] for f in finals] == [0, 1]
    assert finals[0]["text"].startswith("karna-large: ") and finals[0]["start"] < finals[0]["end"] <= finals[1]["start"]
    # partials come from the requested draft model, finals from the chosen one
    partials = [m for m in messages if m["type"] == "partial"]
    assert partials and all(p["text"].startswith("karna-base: ") for p in partials)
    # the second utterance is decoded after 1 s of the first, which is cut from its text
    final_calls = [c for c in client.engine.calls if c[0] == "karna-large"]
    assert final_calls[0][2] == 0 and final_calls[1][2] == 8000 + 16000
    assert finals[1]["text"].endswith("after context")
    done = messages[-1]
    assert done["type"] == "done" and done["reason"] == "stop"
    assert done["text"] == " ".join(f["text"] for f in finals)
    job = done["job"]
    assert job["status"] == "done" and job["live"] and job["filename"].startswith("Live ") and job["language"] == "km"
    assert job["result"]["text"] == done["text"] and len(job["result"]["segments"]) == 2
    assert job["duration_s"] == pytest.approx(len(data) / 2 / 16000, abs=1e-3)
    assert client.get("/v1/jobs").json()[0]["id"] == job["id"]
    assert client.get(f"/v1/jobs/{job['id']}/export?format=txt").text == done["text"] + "\n"


def test_stream_silence_saves_nothing(client):
    messages = stream(client, [pcm16(np.zeros(16000, np.float32))])
    assert messages[-1] == {"type": "done", "text": "", "job": None, "reason": "stop"}
    assert client.get("/v1/jobs").json() == []


def test_stream_limit(tmp_path):
    settings = Settings(data_dir=tmp_path, max_stream_seconds=2)
    with TestClient(create_app(settings, FakeEngine("karna-base", "cpu"))) as c:
        with c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "start"})
            assert ws.receive_json()["max_seconds"] == 2
            ws.send_bytes(pcm16(speech_like()))
            msg = ws.receive_json()
            while msg["type"] != "done":
                msg = ws.receive_json()
        assert msg["reason"] == "limit" and msg["job"]["duration_s"] == 2.0


def test_stream_rejections(tmp_path):
    from starlette.websockets import WebSocketDisconnect

    settings = Settings(data_dir=tmp_path, api_key="s3cret")
    with TestClient(create_app(settings, FakeEngine("karna-base", "cpu"))) as c:
        with c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "start", "api_key": "wrong"})
            with pytest.raises(WebSocketDisconnect) as e:
                ws.receive_json()
        assert e.value.code == 1008
        with c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "start", "api_key": "s3cret", "model": "nope"})
            assert ws.receive_json()["type"] == "error"
        with c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "start", "api_key": "s3cret", "partial_model": "nope"})
            assert ws.receive_json()["type"] == "error"
        with c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "start", "api_key": "s3cret", "language": "fr"})
            assert ws.receive_json()["type"] == "error"
        with c.websocket_connect("/v1/stream", headers={"X-API-Key": "s3cret"}) as ws:
            ws.send_json({"type": "start"})
            assert ws.receive_json()["type"] == "ready"
        with c.websocket_connect("/v1/stream") as ws:
            ws.send_bytes(b"\x00\x00")
            with pytest.raises(WebSocketDisconnect) as e:
                ws.receive_json()
        assert e.value.code == 1003
        assert c.get("/v1/info").json()["streaming"] is True
