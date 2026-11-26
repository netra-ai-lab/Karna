"""The ``Karna`` speech recogniser: load a model, transcribe audio (greedy CTC, or beam search with a character LM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import AudioInput, load_audio
from .models import DEFAULT_MODEL, get, lm_settings
from .streaming import Endpointer, StreamingTranscriber
from .text import clean_khmer

#: longer audio goes through ``transcribe_long`` in the CLI and the server (training used 1-25 s clips)
SINGLE_PASS_S = 30.0
#: the speech detector ``transcribe_long`` segments with (``karna.vad``)
LONG_VAD = "energy"


class Karna:
    """A Khmer + English CTC model on ONNX Runtime.

    >>> asr = Karna.from_pretrained("karna-base")
    >>> asr.transcribe("speech.wav")
    """

    def __init__(self, recognizer) -> None:
        self.recognizer = recognizer

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path = DEFAULT_MODEL, device: str | None = None) -> Karna:
        """A loaded model; with ``KARNA_LM`` set, its language model too (``models.lm_settings``)."""
        rec = get(name_or_path, device)
        if rec.lm is None and rec.beamable:
            found = lm_settings(name_or_path)
            if found is not None:
                rec.set_lm(found[0], **found[1])
        return cls(rec)

    def _decoder(self, decoder: str) -> str:
        """``auto``: beam search when a language model is set, else greedy."""
        if decoder != "auto":
            return decoder
        return "beam" if getattr(self.recognizer, "lm", None) is not None else "greedy"

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

    def set_lm(self, lm, **beam_options) -> None:
        """The character language model for ``decoder="beam"`` (a path to a ``.npz`` from
        ``scripts/build_char_lm.py``, a ``CharLM``, or None), and beam settings such as
        ``lm_weight``, ``insertion_bonus`` and ``beam``."""
        self.recognizer.set_lm(lm, **beam_options)

    def transcribe(self, audio: AudioInput | list[AudioInput], batch_size: int = 8, language: str = "auto",
                   clean: bool = True, decoder: str = "auto") -> str | list[str]:
        """Text for one input, or a list of texts for a list of inputs (same order).

        ``language``: ``auto`` (Khmer or English), ``km`` (Khmer: Latin letters are never output)
        or ``en``. ``clean`` repairs unwritable Khmer sequences (``karna.text.clean_khmer``).
        ``decoder``: ``greedy`` (the per-frame argmax), ``beam`` (CTC beam search, with the
        language model from ``set_lm`` or ``KARNA_LM``), or ``auto`` (default): beam when a
        language model is set, else greedy.

        An input is a file path, the file's bytes, a 16 kHz waveform (numpy array or tensor), or a
        ``(waveform, sample_rate)`` tuple. Each input is decoded in one pass; for recordings longer
        than about 30 s, use ``transcribe_long``.
        """
        single = not isinstance(audio, list)
        items = [audio] if single else audio
        wavs = [load_audio(a, self.sample_rate) for a in items]
        decoder = self._decoder(decoder)
        extra = {} if decoder == "greedy" else {"decoder": decoder}
        texts = self.recognizer.transcribe(wavs, batch_size=batch_size, language=language, **extra)
        if clean:
            texts = [clean_khmer(t) for t in texts]
        return texts[0] if single else texts

    def transcribe_long(self, audio: AudioInput, language: str = "auto", clean: bool = True,
                        decoder: str = "auto", batch_size: int = 8, vad=None, context_s: float = 1.0,
                        **endpointer_opts) -> dict:
        """A recording of any length: split at pauses (as live transcription does), each piece
        decoded with ``context_s`` of the one before it in view, in batches.

        Returns ``{"text": ..., "segments": [{"start": s, "end": s, "text": ...}, ...]}``.
        ``vad``: ``energy`` or ``silero`` (default ``LONG_VAD``); other keywords go to
        ``karna.streaming.Endpointer`` (e.g. ``max_segment_s``, default 15).
        """
        rate = self.sample_rate
        wav = load_audio(audio, rate)
        ep = Endpointer(sample_rate=rate, vad=vad or LONG_VAD, **endpointer_opts)
        segments = [s for s in [*ep.push(wav), *ep.flush()] if s.keep]
        pad_ = np.zeros(rate // 2, np.float32)
        ctx = int(context_s * rate)
        items, skips = [], []
        prev = None
        for seg in segments:
            left = prev.audio[-ctx:] if prev is not None and ctx > 0 else np.zeros(0, np.float32)
            items.append(np.concatenate([pad_, left, seg.audio, pad_]))
            skips.append(len(pad_) + len(left) if left.size else 0)
            prev = seg
        decoder = self._decoder(decoder)
        extra = {} if decoder == "greedy" else {"decoder": decoder}
        texts = self.recognizer.transcribe(items, batch_size=batch_size, language=language, skips=skips,
                                           **extra) if items else []
        if clean:
            texts = [clean_khmer(t) for t in texts]
        out = [{"start": round(s.start_s, 3), "end": round(s.end_s, 3), "text": t}
               for s, t in zip(segments, texts)]
        return {"text": " ".join(t["text"] for t in out if t["text"]), "segments": out}

    def stream(self, partial_interval_s: float = 0.8, language: str = "auto", clean: bool = True,
               decoder: str = "auto", **endpointer_opts) -> StreamingTranscriber:
        """Live transcription: feed 16 kHz mono float32 chunks, get partial and final text as you go.

        >>> live = asr.stream()
        >>> for chunk in chunks:
        ...     for event in live.feed(chunk):
        ...         print(event["type"], event["text"])
        >>> live.finish(); live.text

        ``language`` and ``clean`` are as for ``transcribe``; ``decoder`` applies to finals
        (partials are always greedy, as they are re-decoded often). Speech is split into utterances at
        pauses (``min_silence_ms``, default 600) and at ``max_segment_s`` (default 15); see
        ``karna.streaming`` for the other options.
        """
        post = clean_khmer if clean else (lambda text: text)
        rec = self.recognizer

        def decoders(kind: str):
            extra = {} if kind == "greedy" else {"decoder": kind}
            return (lambda wav: post(rec.transcribe([wav], language=language, **extra)[0]),
                    lambda wav, skip: post(rec.transcribe_tail(wav, skip, language=language, **extra)))

        partial, partial_context = decoders("greedy")
        final, final_context = decoders(self._decoder(decoder))
        return StreamingTranscriber(partial, decode_context=partial_context, final_decode=final,
                                    final_decode_context=final_context, partial_interval_s=partial_interval_s,
                                    sample_rate=self.sample_rate, **endpointer_opts)


def load_model(name_or_path: str | Path = DEFAULT_MODEL, device: str | None = None) -> Karna:
    """``Karna.from_pretrained``: a short name (``karna-base``, ``karna-large``), a Hugging Face
    repo id, or a local ONNX model folder. ``device``: auto (default), cpu, cuda, tensorrt, openvino."""
    return Karna.from_pretrained(name_or_path, device)


def transcribe(audio: AudioInput | list[AudioInput], model: str | Path = DEFAULT_MODEL,
               device: str | None = None, batch_size: int = 8, language: str = "auto") -> str | list[str]:
    """Two lines to a result: ``karna.transcribe("speech.wav")``. The model is loaded once per process."""
    return load_model(model, device).transcribe(audio, batch_size=batch_size, language=language)
