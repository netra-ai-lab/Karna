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
from .bias import INPUT as BIAS_INPUT
from .bias import biased_graph, language_bias
from .hub import load_sessions, model_file


class Recognizer:
    def __init__(self, source: str | Path, device: str | None = "auto") -> None:
        self.root, self.config, self.variant, self.sessions, self.device = load_sessions(
            source, device, prepare=lambda role, path: biased_graph(path, f"{Path(str(source)).name}-{role}"))
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
        #: False when the graph could not be given a token-bias input (no ``onnx`` package)
        self.biasable = BIAS_INPUT in {i.name for i in self.session.session.get_inputs()}
        self._biases: dict[str, np.ndarray] = {}

    def _inputs(self, audio: np.ndarray, lens: np.ndarray, language: str | None) -> dict:
        """Graph inputs; ``language=None`` decodes over the whole vocabulary (the plain argmax)."""
        inputs = {"audio": audio, "audio_lens": lens}
        if self.biasable:
            if language not in self._biases:
                self._biases[language] = (np.zeros(len(self.vocab), np.float32) if language is None
                                          else language_bias(self.vocab.pieces, language))
            inputs[BIAS_INPUT] = self._biases[language]
        elif language not in (None, "auto"):
            raise RuntimeError("language control needs the onnx package: pip install onnx")
        return inputs

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

    def transcribe_tail(self, wav: np.ndarray, skip: int, language: str | None = "auto") -> str:
        """Text for ``wav[skip:]`` only, decoded with ``wav[:skip]`` in view as context: the frames
        that cover the first ``skip`` samples are dropped before CTC collapsing. Live transcription
        passes the end of the previous utterance this way (``karna.streaming``)."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        out = self.session.run(self._inputs(self.normalize(wav)[None, :], np.array([len(wav)], np.int64), language))
        ids = out["ids"][0, : int(out["frame_lens"][0])]
        first = int(self.frame_lengths(np.array([skip]))[0]) if skip > 0 else 0
        return self.vocab.greedy(ids[first:].tolist())

    def transcribe(self, wavs: list[np.ndarray], batch_size: int = 8, language: str | None = "auto") -> list[str]:
        """Text for each 1-D float32 waveform at ``sample_rate`` (same order).

        ``language`` limits the characters the model may emit (``bias.language_bias``): ``auto``
        allows Khmer and ASCII, ``km`` Khmer without Latin letters, ``en`` ASCII only, and ``None``
        every character (the published graph's own argmax).

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
            out = self.session.run(self._inputs(batch, lens, language))
            for row, i in enumerate(idx):
                texts[i] = self.vocab.greedy(out["ids"][row, : int(out["frame_lens"][row])].tolist())
        return texts
