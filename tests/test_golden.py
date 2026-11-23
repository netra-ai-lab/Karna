"""The published models must give exactly the golden text (pytest --slow; downloads both models)."""

import json
from pathlib import Path

import pytest

import karna

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["karna-base", "karna-large"])
def test_golden(name):
    want = json.loads((GOLDEN / "golden.json").read_text(encoding="utf-8"))["models"][name]
    asr = karna.load_model(name, device="cpu")
    clips = sorted(want)
    assert dict(zip(clips, asr.transcribe([str(GOLDEN / "audio" / c) for c in clips]))) == want
