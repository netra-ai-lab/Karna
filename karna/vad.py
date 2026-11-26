"""Speech detectors for the live ``Endpointer``: is this frame speech?

- ``energy`` (``Endpointer``'s built-in rule): the frame's RMS against an adaptive noise floor.
  No model; misses nothing quiet, but any loud noise (a fan, traffic, music) is "speech".
- ``silero``: Silero VAD v5.1.2 (MIT, 2.3 MB ONNX, downloaded on first use from the v5.1.2 tag
  of github.com/snakers4/silero-vad into ``KARNA_CACHE_DIR/vad`` and checked by sha256), a small recurrent
  network on 32 ms windows. It tells speech from noise, so segments close at pauses in noisy
  rooms instead of running to the forced cut.

A detector is any callable ``(frame, rms) -> bool`` with a ``reset()`` method and a ``frame``
attribute (the samples per frame it needs at 16 kHz).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SILERO_URL = "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"


def silero_path() -> Path:
    """The Silero VAD file: ``KARNA_VAD_MODEL`` if set, else a cached download (fetched once)."""
    if os.environ.get("KARNA_VAD_MODEL"):
        return Path(os.environ["KARNA_VAD_MODEL"])
    path = Path(os.environ.get("KARNA_CACHE_DIR") or Path.home() / ".cache" / "karna") / "vad" / "silero_vad-v5.1.2.onnx"
    if path.exists():
        return path
    import hashlib
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SILERO_URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SILERO_SHA256:
        raise RuntimeError(f"{SILERO_URL}: unexpected sha256; set KARNA_VAD_MODEL to a Silero VAD v5 file")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


class SileroVAD:
    """Silero VAD v5 on ONNX Runtime (CPU, one thread), fed 512-sample frames at 16 kHz.

    Speech starts when the probability reaches ``threshold`` and ends when it falls below
    ``threshold - 0.15`` (hysteresis, as in Silero's own ``get_speech_timestamps``).
    """

    frame = 512
    context = 64
    rate = 16000

    def __init__(self, path: str | Path | None = None, threshold: float = 0.5, agc: bool = True,
                 target: float = 0.05, max_gain: float = 100.0) -> None:
        from .inference.session import _ort

        ort = _ort()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        path = Path(path) if path else silero_path()
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.release = max(0.0, threshold - 0.15)
        self.agc, self.target, self.max_gain = agc, target, max_gain
        self._sr = np.array(self.rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros(self.context, np.float32)
        self._speaking = False
        self.last_prob = 0.0
        self._level: float | None = None  # ~1 s running RMS
        self._noise: float | None = None  # quietest recent level (follows down fast, up slowly)

    def prob(self, frame: np.ndarray) -> float:
        frame = np.asarray(frame, np.float32).reshape(-1)
        if frame.size != self.frame:
            raise ValueError(f"Silero VAD needs {self.frame}-sample frames, got {frame.size}")
        x = np.concatenate([self._context, frame])[None, :]
        out, self._state = self.session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = frame[-self.context:]
        self.last_prob = float(out[0, 0])
        return self.last_prob

    def gain(self, rms: float) -> float:
        """Silero is not level-invariant: after loud speech it scores speech 40 dB quieter as 0.
        So its input is brought to about ``target`` RMS over the last second, but never by more
        than keeps the background below ``target / 3``."""
        rms = max(rms, 1e-6)
        if self._level is None:
            self._level = self._noise = rms
        self._level = float(np.sqrt(0.94 * self._level ** 2 + 0.06 * rms ** 2))  # ~0.5 s at 32 ms frames
        self._noise = 0.7 * self._noise + 0.3 * rms if rms < self._noise else self._noise * 1.003
        cap = self.target / 3 / self._noise
        return float(np.clip(self.target / self._level, 1.0, min(self.max_gain, max(cap, 1.0))))

    def __call__(self, frame: np.ndarray, rms: float) -> bool:
        p = self.prob(frame * self.gain(rms) if self.agc else frame)
        if p >= self.threshold:
            self._speaking = True
        elif p < self.release:
            self._speaking = False
        return self._speaking


def make_vad(vad) -> SileroVAD | None:
    """``"energy"`` or None -> None (the Endpointer's built-in rule); ``"silero"`` -> SileroVAD;
    a detector object is returned as is."""
    if vad is None or vad == "energy":
        return None
    if vad == "silero":
        return SileroVAD()
    if callable(vad):
        return vad
    raise ValueError(f"vad must be 'energy', 'silero' or a detector, not {vad!r}")
