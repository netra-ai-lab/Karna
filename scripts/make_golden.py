"""Write tests/golden/golden.json: each model's text for each clip in tests/golden/audio.

    KARNA_BUNDLES=<exported folders> python scripts/make_golden.py     # before uploading
    python scripts/make_golden.py                                       # the published models

Run it only when a model changed on purpose, after export.sh showed the ONNX folder gives the
PyTorch model's text; the golden file then guards the runtime, the pre/post-processing and the
Docker image against any later change in output.
"""

import json
from pathlib import Path

import karna
from karna.models import MODELS

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden"

clips = sorted(p for p in (GOLDEN / "audio").iterdir() if p.suffix in (".flac", ".wav"))
out = {"note": "written by scripts/make_golden.py; references in references.tsv", "models": {}}
for name in MODELS:
    asr = karna.load_model(name, device="cpu")
    out["models"][name] = dict(zip((p.name for p in clips), asr.transcribe([str(p) for p in clips])))
    print(name, out["models"][name])
(GOLDEN / "golden.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
