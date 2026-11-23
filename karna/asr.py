"""The ``Karna`` speech recogniser: load a model, transcribe audio (greedy CTC, no language model)."""

from __future__ import annotations

from pathlib import Path

from .audio import AudioInput, load_audio
from .models import DEFAULT_MODEL, get
from .streaming import StreamingTranscriber


class Karna:
    """A Khmer + English CTC model on ONNX Runtime.

    >>> asr = Karna.from_pretrained("karna-base")
    >>> asr.transcribe("speech.wav")
    """

    def __init__(self, recognizer) -> None:
        self.recognizer = recognizer

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path = DEFAULT_MODEL, device: str | None = None) -> Karna:
        return cls(get(name_or_path, device))

    @property
    def device(self) -> str:
        return self.recognizer.device

    @property
    def name(self) -> str:
        return self.recognizer.name

    @property
    def num_parameters(self) -> int | None:
        return self.recognizer.num_parameters

    @property
    def sample_rate(self) -> int:
        return self.recognizer.sample_rate

    def transcribe(self, audio: AudioInput | list[AudioInput], batch_size: int = 8) -> str | list[str]:
        """Text for one input, or a list of texts for a list of inputs (same order).

        An input is a file path, the file's bytes, a 16 kHz waveform (numpy array or tensor), or a
        ``(waveform, sample_rate)`` tuple. Each file is decoded in one pass, so very long
        recordings should be split into utterances first.
        """
        single = not isinstance(audio, list)
        items = [audio] if single else audio
        wavs = [load_audio(a, self.sample_rate) for a in items]
        texts = self.recognizer.transcribe(wavs, batch_size=batch_size)
        return texts[0] if single else texts

    def stream(self, partial_interval_s: float = 0.8, **endpointer_opts) -> StreamingTranscriber:
        """Live transcription: feed 16 kHz mono float32 chunks, get partial and final text as you go.

        >>> live = asr.stream()
        >>> for chunk in chunks:
        ...     for event in live.feed(chunk):
        ...         print(event["type"], event["text"])
        >>> live.finish(); live.text

        Speech is split into utterances at pauses (``min_silence_ms``, default 600) and at
        ``max_segment_s`` (default 15); see ``karna.streaming`` for the other options.
        """
        return StreamingTranscriber(lambda wav: self.recognizer.transcribe([wav])[0],
                                    partial_interval_s=partial_interval_s, sample_rate=self.sample_rate,
                                    **endpointer_opts)


def load_model(name_or_path: str | Path = DEFAULT_MODEL, device: str | None = None) -> Karna:
    """``Karna.from_pretrained``: a short name (``karna-base``, ``karna-large``), a Hugging Face
    repo id, or a local ONNX model folder. ``device``: auto (default), cpu, cuda, tensorrt, openvino."""
    return Karna.from_pretrained(name_or_path, device)


def transcribe(audio: AudioInput | list[AudioInput], model: str | Path = DEFAULT_MODEL,
               device: str | None = None, batch_size: int = 8) -> str | list[str]:
    """Two lines to a result: ``karna.transcribe("speech.wav")``. The model is loaded once per process."""
    return load_model(model, device).transcribe(audio, batch_size=batch_size)
