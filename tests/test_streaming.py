"""Endpointing and partial / final events, with synthetic audio and a fake decoder (no model)."""

import numpy as np
import pytest

from karna.streaming import Endpointer, StreamingTranscriber

SR = 16000


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds, seed=0):
    return (0.001 * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)


def feed_in_chunks(target, signal, seed=0):
    """Random chunk sizes, as from a microphone; returns everything ``target.push``/``feed`` returned."""
    rng, out, i = np.random.default_rng(seed), [], 0
    push = target.push if isinstance(target, Endpointer) else target.feed
    while i < len(signal):
        n = int(rng.integers(50, 4000))
        out += push(signal[i:i + n])
        i += n
    return out


def seconds(seg):
    return len(seg.audio) / SR


def test_splits_at_pauses():
    signal = np.concatenate([quiet(0.5), tone(1.5), quiet(1.0), tone(2.0), quiet(1.0)])
    ep = Endpointer(min_segment_s=0)
    segs = feed_in_chunks(ep, signal) + ep.flush()
    assert [s.id for s in segs] == [0, 1] and all(s.keep for s in segs)
    # starts include the 300 ms pre-roll; each keeps 300 ms of trailing silence
    assert segs[0].start_s == pytest.approx(0.2, abs=0.05) and segs[0].end_s == pytest.approx(2.3, abs=0.06)
    assert segs[1].start_s == pytest.approx(2.7, abs=0.05) and segs[1].end_s == pytest.approx(5.3, abs=0.06)
    assert seconds(segs[0]) == pytest.approx(segs[0].end_s - segs[0].start_s, abs=1e-6)
    # the samples are the input's, in order
    start = int(round(segs[1].start_s * SR))
    np.testing.assert_array_equal(segs[1].audio, signal[start:start + len(segs[1].audio)])


def test_short_pauses_do_not_split_short_segments():
    # 0.8 s pauses: a 600 ms pause only closes a segment of 5 s or more; 1.5 s closes any
    words = [tone(1.0), quiet(0.8)] * 4 + [tone(2.0)]
    signal = np.concatenate([quiet(0.5)] + words + [quiet(2.0), tone(1.0), quiet(2.0)])
    ep = Endpointer()
    segs = feed_in_chunks(ep, signal) + ep.flush()
    assert len(segs) == 3 and all(s.keep for s in segs)
    # words end at 1.5, 3.3, 5.1, 6.9 and 9.7 s; at 5.1 s the segment (from 0.2 s) holds only 4.9 s,
    # so it closes at the next pause: 6.9 s plus the 300 ms tail
    assert segs[0].start_s == pytest.approx(0.2, abs=0.05) and segs[0].end_s == pytest.approx(7.2, abs=0.06)
    assert segs[1].start_s == pytest.approx(7.4, abs=0.05) and segs[1].end_s == pytest.approx(10.0, abs=0.06)
    assert segs[2].start_s == pytest.approx(11.4, abs=0.05)


def test_forced_cut_at_the_cap():
    # 10 s of speech whose quietest stretch is near 4 s: a 5 s cap cuts there
    speech = tone(10.0)
    speech[int(4.0 * SR):int(4.1 * SR)] *= 0.2
    ep = Endpointer(max_segment_s=5.0)
    segs = feed_in_chunks(ep, np.concatenate([quiet(0.3), speech, quiet(1.0)])) + ep.flush()
    assert all(seconds(s) <= 5.0 + 1e-6 for s in segs)
    assert len(segs) >= 2 and segs[0].end_s == pytest.approx(4.3, abs=0.1)
    # no samples lost or repeated across the cuts
    for a, b in zip(segs, segs[1:]):
        assert a.end_s == pytest.approx(b.start_s, abs=1e-6)


def test_clicks_are_dropped_and_flush_closes_the_open_segment():
    ep = Endpointer()
    segs = feed_in_chunks(ep, np.concatenate([quiet(0.5), tone(0.1), quiet(2.0), tone(1.0)]))
    assert [s.keep for s in segs] == [False]
    last = ep.flush()
    assert len(last) == 1 and last[0].keep and ep.open_id is None and ep.flush() == []


def test_adapts_to_a_noisy_room():
    rng = np.random.default_rng(1)
    noise = (0.02 * rng.standard_normal(int(6.5 * SR))).astype(np.float32)  # well above abs_floor
    signal = noise.copy()
    signal[int(3 * SR):int(5 * SR)] += tone(2.0)
    ep = Endpointer()
    segs = [s for s in feed_in_chunks(ep, signal) + ep.flush() if s.keep]
    assert len(segs) == 1 and segs[0].start_s == pytest.approx(2.7, abs=0.1)


def test_transcriber_events():
    calls = []

    def decode(wav):
        calls.append(len(wav))
        return f"{len(wav) / SR:.1f}s"

    live = StreamingTranscriber(decode, partial_interval_s=0.5, min_segment_s=0, pad_ms=0)
    signal = np.concatenate([quiet(0.5), tone(2.0), quiet(1.0), tone(0.1), quiet(1.0), tone(1.5)])
    events = feed_in_chunks(live, signal) + live.finish()
    finals = [e for e in events if e["type"] == "final"]
    partials = [e for e in events if e["type"] == "partial"]
    assert partials and all(p["segment"] in {f["segment"] for f in finals} for p in partials)
    # each segment's partials come before its final
    for f in finals:
        idx = events.index(f)
        assert all(events.index(p) < idx for p in partials if p["segment"] == f["segment"])
    kept = [f for f in finals if f["text"]]
    assert len(kept) == 2 and live.text == " ".join(f["text"] for f in kept)
    assert [f["text"] for f in live.finals] == [f["text"] for f in kept]


def test_karna_stream_uses_the_recognizer():
    from karna.asr import Karna

    class FakeRecognizer:
        sample_rate = SR

        def transcribe(self, wavs, batch_size=8, language="auto"):
            return [f"n={len(w)}" for w in wavs]

        def transcribe_tail(self, wav, skip, language="auto"):
            return f"n={len(wav) - skip}"

    live = Karna(FakeRecognizer()).stream()
    events = live.feed(np.concatenate([quiet(0.5), tone(1.0), quiet(1.0)])) + live.finish()
    assert [e["type"] for e in events][-1] == "final" and live.text.startswith("n=")


def test_decodes_are_padded_with_silence():
    seen = []
    live = StreamingTranscriber(lambda w: seen.append(w) or "x", pad_ms=500)
    live.feed(np.concatenate([quiet(0.5), tone(1.0)]))
    live.finish()
    final = seen[-1]
    assert not final[:SR // 2].any() and not final[-SR // 2:].any() and final[SR // 2:-SR // 2].any()


def test_speech_right_after_digital_silence():
    # a microphone that opens with zeros, then speech at once
    ep = Endpointer()
    segs = ep.push(np.concatenate([np.zeros(SR // 2, np.float32), tone(1.0), np.zeros(2 * SR, np.float32)])) + ep.flush()
    assert len(segs) == 1 and segs[0].keep and segs[0].start_s == pytest.approx(0.2, abs=0.05)



def test_later_utterances_are_decoded_after_the_end_of_the_previous_one():
    calls = []

    def decode_context(wav, skip):
        calls.append((wav, skip))
        return "second"

    live = StreamingTranscriber(lambda w: calls.append((w, 0)) or "first", decode_context=decode_context,
                                partial_interval_s=100, context_s=1.0, pad_ms=500)
    first = tone(2.0, amp=0.5)
    events = feed_in_chunks(live, np.concatenate([quiet(0.5), first, quiet(2.0), tone(1.5), quiet(2.0)])) + live.finish()
    assert [e["text"] for e in events if e["type"] == "final"] == ["first", "second"]
    wav, skip = calls[-1]
    assert skip == SR // 2 + SR                     # the padding, then 1 s from the end of the first utterance
    assert not wav[:SR // 2].any() and not wav[-SR // 2:].any()
    context = wav[SR // 2:skip]
    assert np.abs(context[:SR // 2]).max() > 0.45   # the louder first tone (the segment ends in 0.3 s of quiet)


def test_transcribe_long_splits_and_gives_context():
    from karna.asr import Karna

    calls = []

    class FakeRecognizer:
        sample_rate = SR

        def transcribe(self, wavs, batch_size=8, language="auto", skips=None):
            calls.append((len(wavs), skips))
            return [f"n={len(w) - (s or 0)}" for w, s in zip(wavs, skips or [0] * len(wavs))]

    signal = np.concatenate([quiet(0.5), tone(6.0), quiet(2.0), tone(6.0), quiet(2.0), tone(3.0), quiet(1.0)])
    out = Karna(FakeRecognizer()).transcribe_long(signal, batch_size=8)
    segs = out["segments"]
    assert len(segs) == 3 and calls[0][0] == 3  # one batch
    skips = calls[0][1]
    assert skips[0] == 0 and skips[1] == skips[2] == SR // 2 + SR  # 0.5 s padding + 1 s context
    assert all(s["end"] > s["start"] for s in segs) and segs[1]["start"] > segs[0]["end"]
    assert out["text"] == " ".join(s["text"] for s in segs)
