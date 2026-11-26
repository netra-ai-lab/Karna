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
    assert dict(zip(clips, asr.transcribe([str(GOLDEN / "audio" / c) for c in clips], decoder="greedy"))) == want


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["karna-base", "karna-large"])
def test_streaming_matches_offline(name):
    """Fed in 100 ms chunks, live transcription gives the offline text, or within 15 % of its characters
    (segments carry less surrounding audio than the whole file). Measured 2026-11-23: 4/8 identical,
    the rest 1-5 characters apart (6.5 % at most)."""
    from karna.audio import load_audio

    want = json.loads((GOLDEN / "golden.json").read_text(encoding="utf-8"))["models"][name]
    asr = karna.load_model(name, device="cpu")
    for clip, offline in sorted(want.items()):
        wav = load_audio(str(GOLDEN / "audio" / clip))
        live = asr.stream(decoder="greedy")
        events = [e for i in range(0, len(wav), 1600) for e in live.feed(wav[i:i + 1600])] + live.finish()
        assert any(e["type"] == "partial" for e in events)
        assert edit_distance(live.text, offline) <= 0.15 * len(offline), (clip, live.text, offline)
