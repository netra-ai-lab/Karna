"""Audio input: any file soundfile can read, or an array, as a mono 16 kHz float32 numpy array."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

SAMPLE_RATE = 16000

#: what ``Karna.transcribe`` accepts per utterance: a path, file bytes, a 16 kHz waveform
#: (numpy array, or anything with ``.numpy()`` such as a torch tensor), or (waveform, sample_rate)
AudioInput = Union[str, Path, bytes, np.ndarray, tuple, Any]


def resample(wav: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return wav
    import soxr

    return soxr.resample(wav, sr, target).astype(np.float32, copy=False)


def _to_numpy(wav) -> np.ndarray:
    if hasattr(wav, "detach"):  # a torch tensor, without importing torch
        wav = wav.detach().cpu().numpy()
    elif not isinstance(wav, np.ndarray) and hasattr(wav, "numpy"):
        wav = wav.numpy()
    return np.asarray(wav, dtype=np.float32)


def load_audio(audio: AudioInput, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode / convert one input to a 1-D float32 array at ``sample_rate`` (channels averaged)."""
    if isinstance(audio, tuple):
        wav, sr = audio
    elif isinstance(audio, (str, Path, bytes)):
        import io

        import soundfile as sf

        src = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
        wav, sr = sf.read(src, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    else:
        wav, sr = audio, sample_rate
    wav = _to_numpy(wav)
    if wav.ndim == 2:  # (channels, samples) or (samples, channels): average the shorter axis
        wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[1] else 1)
    if wav.ndim != 1 or wav.size == 0:
        raise ValueError(f"expected a non-empty mono or stereo waveform, got shape {wav.shape}")
    return np.ascontiguousarray(resample(wav, int(sr), sample_rate), dtype=np.float32)
