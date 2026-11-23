"""The Karna CTC recogniser on ONNX Runtime: waveforms in, text out (greedy CTC, no language model).

The graph (``training/karna_train/export_onnx.py``) takes a padded batch of normalised 16 kHz
audio and its lengths, and returns the best token per frame and the number of valid frames:

    audio       float32 [N, S]   zero-mean / unit-variance over each utterance's valid span, 0 after it
    audio_lens  int64   [N]
    ids         int64   [N, T]   argmax over the 9,812-character vocabulary per 20 ms frame
    frame_lens  int64   [N]

Everything else it needs (sample rate, normalisation, the conv front end, the blank id and the
vocabulary file) is read from the folder's ``netra-model.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..text import CharVocab
from .hub import load_sessions, model_file


class Recognizer:
    def __init__(self, source: str | Path, device: str | None = "auto") -> None:
        self.root, self.config, self.variant, self.sessions, self.device = load_sessions(source, device)
        self.source = str(source)
        components = self.config["components"]
        pre = components.get("preprocess", {})
        self.sample_rate = int(pre.get("sample_rate", 16000))
        self.eps = float(pre.get("eps", 1e-7))
        self.conv_layers = [tuple(layer) for layer in components["frontend"]["conv_layers"]]
        decoder = components.get("decoder", {})
        if decoder.get("type", "ctc-greedy") != "ctc-greedy":
            raise ValueError(f"unsupported decoder {decoder.get('type')!r} in {self.root}")
        self.vocab = CharVocab.load(model_file(self.root, components["tokenizer"]["vocab"]))
        self.session = self.sessions["model"]

    @property
    def name(self) -> str:
        return self.config.get("name", Path(self.root).name)

    @property
    def num_parameters(self) -> int | None:
        return self.config.get("metadata", {}).get("parameters")

    def frame_lengths(self, audio_lens: np.ndarray) -> np.ndarray:
        """Output frames per input length, from the conv front end (what the graph returns as ``frame_lens``)."""
        lens = np.asarray(audio_lens, dtype=np.int64)
        for _, kernel, stride in self.conv_layers:
            lens = np.maximum((lens - kernel) // stride + 1, 0)
        return lens

    def normalize(self, wav: np.ndarray) -> np.ndarray:
        """Zero mean, unit variance over the utterance (Omnilingual's ``apply_audio_normalization``)."""
        wav = wav.astype(np.float32, copy=False)
        std = max(float(wav.std()), self.eps)
        return ((wav - wav.mean()) / std).astype(np.float32, copy=False)

    def transcribe(self, wavs: list[np.ndarray], batch_size: int = 8) -> list[str]:
        """Text for each 1-D float32 waveform at ``sample_rate`` (same order).

        Longest first, so each batch pads as little as possible.
        """
        order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]), reverse=True)
        texts: list[str] = [""] * len(wavs)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            lens = np.array([len(wavs[i]) for i in idx], dtype=np.int64)
            batch = np.zeros((len(idx), int(lens.max())), dtype=np.float32)
            for row, i in enumerate(idx):
                batch[row, : lens[row]] = self.normalize(wavs[i])
            out = self.session.run({"audio": batch, "audio_lens": lens})
            for row, i in enumerate(idx):
                texts[i] = self.vocab.greedy(out["ids"][row, : int(out["frame_lens"][row])].tolist())
        return texts
