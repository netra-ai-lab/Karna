"""Live transcription: endpointed re-decoding (numpy only, no model code).

Karna is an offline CTC model. It uses full attention, normalises each utterance as a whole, and was
trained on 1-25 s clips, so it can't carry state from one chunk to the next. Live text comes from
this instead:

1. An energy VAD (``Endpointer``) splits the incoming audio into utterance segments at pauses
   (only at long pauses while a segment is under 5 s, so pieces keep enough context). A segment
   that grows past ``max_segment_s`` is cut at its quietest point near the end.
2. While a segment is open, all of it is decoded again every ``partial_interval_s`` of new audio and
   sent as a *partial*, which may still change.
3. When a segment closes, it is decoded once more as a *final*, which never changes. A final is the
   offline transcription of that segment with 0.5 s of silence on each side (see ``pad``).

On the golden clips, streamed text in 100 ms chunks matches the offline text for 4 of 8 (clip,
model) pairs; the others are 1-5 characters apart (tests/test_golden.py).

    asr = karna.load_model("karna-base")
    live = asr.stream()
    for chunk in chunks:                  # 16 kHz mono float32, any size
        for event in live.feed(chunk):
            print(event["type"], event["text"])
    live.finish()
    print(live.text)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

#: frames quieter than this (about -100 dBFS) are digital silence and never set the noise floor
DIGITAL_SILENCE = 1e-5
#: the first estimate of the room is capped (-40 dBFS), so speech right at the start is still speech
INITIAL_NOISE_MAX = 0.01


@dataclass
class Segment:
    id: int
    audio: np.ndarray
    start_s: float
    end_s: float
    #: False when the segment had too little speech to be worth decoding (a click, a cough)
    keep: bool = True


class Endpointer:
    """Splits a stream of 16 kHz float32 samples into utterance segments at pauses.

    The VAD compares the RMS of each frame with ``max(abs_floor, noise * ratio)``. ``noise`` starts
    at the first frame's level, follows quieter frames down at once and louder background up slowly,
    so it adapts to the room. ``abs_floor`` is low (about -66 dBFS) because the model normalises
    each utterance, so quiet recordings transcribe well and must not be missed.

    A segment closes at a pause of ``min_silence_ms`` once it holds ``min_segment_s`` of audio, at a
    pause of ``long_silence_ms`` whatever its length, and at ``max_segment_s`` (cut at the quietest
    frame of its last 2 s). Short segments lose context: cut at every brief pause, a 2 s piece of
    Khmer can come back in Latin script.
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, min_silence_ms: int = 600,
                 min_segment_s: float = 5.0, long_silence_ms: int = 1500, max_segment_s: float = 15.0,
                 preroll_ms: int = 300, tail_ms: int = 300, min_speech_ms: int = 300,
                 abs_floor: float = 5e-4, ratio: float = 4.0) -> None:
        self.sample_rate = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.frame_ms = frame_ms
        self.silence_frames = max(1, min_silence_ms // frame_ms)
        self.long_silence_frames = max(self.silence_frames, long_silence_ms // frame_ms)
        self.min_segment_frames = int(min_segment_s * 1000 // frame_ms)
        self.max_frames = max(2, int(max_segment_s * 1000 // frame_ms))
        self.search_frames = min(self.max_frames // 2, int(2000 // frame_ms))  # where a forced cut may land
        self.tail_frames = tail_ms // frame_ms
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.abs_floor = abs_floor
        self.ratio = ratio
        self.noise: float | None = None
        self.preroll: deque[np.ndarray] = deque(maxlen=max(0, preroll_ms // frame_ms))
        self.pending = np.zeros(0, np.float32)  # samples short of a whole frame
        self.frames: list[np.ndarray] = []      # the open segment
        self.speech: list[bool] = []
        self.rms: list[float] = []
        self.silence_run = 0
        self.seg_start = 0                      # first sample of the open segment
        self.position = 0                       # samples framed so far
        self.next_id = 0
        self.open_id: int | None = None

    # ---- the open segment

    @property
    def open_samples(self) -> int:
        return len(self.frames) * self.frame

    def open_audio(self) -> np.ndarray | None:
        return np.concatenate(self.frames) if self.frames else None

    # ---- input

    def push(self, samples: np.ndarray) -> list[Segment]:
        """Add samples; returns the segments that closed (in order)."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        data = np.concatenate([self.pending, samples]) if self.pending.size else samples
        whole = len(data) // self.frame * self.frame
        self.pending = data[whole:].copy()
        closed: list[Segment] = []
        for start in range(0, whole, self.frame):
            seg = self._frame(data[start:start + self.frame])
            if seg is not None:
                closed.append(seg)
        return closed

    def flush(self) -> list[Segment]:
        """End of stream: frame the leftover samples (zero-padded) and close the open segment."""
        closed: list[Segment] = []
        if self.pending.size:
            frame = np.concatenate([self.pending, np.zeros(self.frame - self.pending.size, np.float32)])
            self.pending = np.zeros(0, np.float32)
            seg = self._frame(frame)
            if seg is not None:
                closed.append(seg)
        if self.frames:  # keep at most tail_frames of trailing silence
            closed.append(self._close(len(self.frames) - self.silence_run + min(self.silence_run, self.tail_frames)))
        return closed

    # ---- internals

    def _frame(self, frame: np.ndarray) -> Segment | None:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms < DIGITAL_SILENCE:     # a muted or just-opened microphone, padding: not the room
            is_speech = False
        elif self.noise is None:      # the first real frame is taken as the room, unless it is loud
            self.noise = min(rms, INITIAL_NOISE_MAX)
            is_speech = rms > max(self.abs_floor, self.noise * self.ratio)
        else:
            is_speech = rms > max(self.abs_floor, self.noise * self.ratio)
            if rms < self.noise:      # follow quieter frames down within ~100 ms
                self.noise = 0.7 * self.noise + 0.3 * rms
            elif not is_speech:       # and louder background up within a second or so
                self.noise = 0.98 * self.noise + 0.02 * rms
            else:                     # creep up during speech (about 7 % a second), in case the room got louder
                self.noise *= 1.002
        start = self.position
        self.position += self.frame

        if not self.frames:
            if not is_speech:
                self.preroll.append(frame)
                return None
            self.open_id = self.next_id
            self.next_id += 1
            pre = list(self.preroll)
            self.preroll.clear()
            self.frames = pre + [frame]
            self.speech = [False] * len(pre) + [True]
            self.rms = [0.0] * len(pre) + [rms]
            self.seg_start = start - len(pre) * self.frame
            self.silence_run = 0
            return None

        self.frames.append(frame)
        self.speech.append(is_speech)
        self.rms.append(rms)
        self.silence_run = 0 if is_speech else self.silence_run + 1
        voiced = len(self.frames) - self.silence_run  # the segment without its trailing pause
        if (self.silence_run >= self.long_silence_frames
                or (self.silence_run >= self.silence_frames and voiced >= self.min_segment_frames)):
            return self._close(len(self.frames) - self.silence_run + self.tail_frames)
        if len(self.frames) >= self.max_frames:
            lo = len(self.frames) - self.search_frames
            cut = lo + int(np.argmin(self.rms[lo:]))
            return self._close(cut, carry=True)
        return None

    def _close(self, end: int, carry: bool = False) -> Segment:
        """Emit frames[:end]. With ``carry``, frames[end:] open the next segment (a forced cut);
        otherwise they become the pre-roll."""
        end = max(1, min(end, len(self.frames)))
        frames, speech = self.frames[:end], self.speech[:end]
        seg = Segment(id=self.open_id if self.open_id is not None else self.next_id,
                      audio=np.concatenate(frames),
                      start_s=self.seg_start / self.sample_rate,
                      end_s=(self.seg_start + end * self.frame) / self.sample_rate,
                      keep=sum(speech) >= self.min_speech_frames)
        rest, rest_speech, rest_rms = self.frames[end:], self.speech[end:], self.rms[end:]
        new_start = self.seg_start + end * self.frame
        self.frames, self.speech, self.rms = [], [], []
        self.open_id = None
        self.silence_run = 0
        if carry and rest:
            self.open_id = self.next_id
            self.next_id += 1
            self.frames, self.speech, self.rms = rest, rest_speech, rest_rms
            self.seg_start = new_start
            run = 0
            for s in reversed(rest_speech):
                if s:
                    break
                run += 1
            self.silence_run = run
        else:
            self.preroll.clear()
            self.preroll.extend(rest)
        return seg


class StreamingTranscriber:
    """Feeds audio to an ``Endpointer`` and decodes partials and finals synchronously.

    ``decode`` turns one 16 kHz float32 waveform into text (``Karna.stream()`` passes the model).
    Events are dicts:

        {"type": "partial", "segment": 3, "text": "..."}
        {"type": "final", "segment": 3, "text": "...", "start": 12.3, "end": 15.9}
    """

    def __init__(self, decode: Callable[[np.ndarray], str], partial_interval_s: float = 0.8,
                 sample_rate: int = 16000, pad_ms: int = 500, **endpointer_opts) -> None:
        self._decode = decode
        self.pad = np.zeros(int(sample_rate * pad_ms / 1000), np.float32)
        self.endpointer = Endpointer(sample_rate=sample_rate, **endpointer_opts)
        self.partial_samples = int(partial_interval_s * sample_rate)
        self.finals: list[dict] = []
        self._partial_at = 0             # open-segment length at the last partial
        self._partial_id: int | None = None

    def decode(self, audio: np.ndarray) -> str:
        return self._decode(pad(audio, self.pad))

    @property
    def text(self) -> str:
        return " ".join(f["text"] for f in self.finals if f["text"])

    def feed(self, samples: np.ndarray) -> list[dict]:
        events = [e for seg in self.endpointer.push(samples) for e in self._final(seg)]
        ep = self.endpointer
        if ep.open_id is not None:
            if ep.open_id != self._partial_id:
                self._partial_id, self._partial_at = ep.open_id, 0
            if ep.open_samples - self._partial_at >= self.partial_samples:
                audio = ep.open_audio()
                assert audio is not None
                self._partial_at = len(audio)
                events.append({"type": "partial", "segment": ep.open_id, "text": self.decode(audio)})
        return events

    def finish(self) -> list[dict]:
        return [e for seg in self.endpointer.flush() for e in self._final(seg)]

    def _final(self, seg: Segment) -> list[dict]:
        had_partial = self._partial_id == seg.id and self._partial_at > 0
        if self._partial_id == seg.id:
            self._partial_id, self._partial_at = None, 0
        if not seg.keep:  # clear a partial the UI may be showing for it
            return [final_event(seg, "")] if had_partial else []
        event = final_event(seg, self.decode(seg.audio))
        self.finals.append(event)
        return [event]


def pad(audio: np.ndarray, silence: np.ndarray) -> np.ndarray:
    """Digital silence on both sides of a segment before decoding. Measured on a quiet FLEURS clip,
    karna-large garbled the bare 7 s segment (Latin letters inside Khmer) and gave the offline
    text again with 0.5 s of zeros each side: normalisation then sees speech as offline clips do."""
    return np.concatenate([silence, audio, silence]) if silence.size else audio


def final_event(seg: Segment, text: str) -> dict:
    return {"type": "final", "segment": seg.id, "text": text,
            "start": round(seg.start_s, 3), "end": round(seg.end_s, 3)}
