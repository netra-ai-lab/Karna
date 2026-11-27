"""Build-time check that the (pruned) image still runs everything the app does.

Runs offline against the baked-in models: every model on every golden clip (wav and flac
decoding, resampling, the ONNX graph, greedy decoding) must give the golden text, the baked-in
language model must load and decode (beam search, the default), and the
server must answer each route. It also checks that the heavy libraries stay out of the image.

    KARNA_BUNDLES=/models HF_HUB_OFFLINE=1 python smoke_test.py golden/
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

for module in ("torch", "cv2", "scipy", "transformers"):
    assert importlib.util.find_spec(module) is None, f"{module} is installed in the image"

import karna  # noqa: E402

golden_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "golden")
golden = json.loads((golden_dir / "golden.json").read_text(encoding="utf-8"))
bundles = Path(os.environ["KARNA_BUNDLES"])
models = [m for m in golden["models"] if (bundles / m).is_dir()]
assert models, f"no model folder in {bundles}"
for name in models:
    asr = karna.load_model(name, device="cpu")
    assert asr.recognizer.lm is not None, f"{name}: the language model did not load"
    for clip, want in golden["models"][name].items():
        got = asr.transcribe(golden_dir / "audio" / clip, decoder="greedy")
        assert got == want, f"{name} {clip}:\n  want {want}\n  got  {got}"
        assert asr.transcribe(golden_dir / "audio" / clip).strip(), f"{name} {clip}: empty beam output"
    print(f"{name}: {len(golden['models'][name])} golden clips identical (greedy), beam search runs")

from fastapi.testclient import TestClient  # noqa: E402

from karna.server.app import create_app  # noqa: E402
from karna.server.settings import Settings  # noqa: E402

clip = next(iter(golden["models"][models[0]]))
want = karna.load_model(models[0], device="cpu").transcribe(golden_dir / "audio" / clip)  # beam, as the server
with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(Settings(data_dir=Path(tmp), model=models[0]))) as c:
    assert c.get("/health").status_code == 200
    assert c.get("/").status_code == 200
    with open(golden_dir / "audio" / clip, "rb") as f:
        job = c.post("/v1/transcribe?wait=true", files={"file": (clip, f)}).json()
    assert job["status"] == "done" and job["result"]["text"] == want, job
    assert c.get(f"/v1/jobs/{job['id']}/export?format=txt").text == want + "\n"
    assert c.get("/v1/info").json()["models_ready"]
print("smoke test passed")
