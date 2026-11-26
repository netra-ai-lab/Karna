"""The Karna CTC recogniser on ONNX Runtime: waveforms in, text out (greedy CTC, or beam search with a character LM).

The graph (``training/karna_train/export_onnx.py``) takes a padded batch of normalised 16 kHz
audio and its lengths, and returns the best token per frame and the number of valid frames:

    audio       float32 [N, S]   zero-mean / unit-variance over each utterance's valid span, 0 after it
    audio_lens  int64   [N]
    ids         int64   [N, T]   argmax over the 9,812-character vocabulary per 20 ms frame
    frame_lens  int64   [N]
    sub_logp    float32 [N, T, 1+K]  log-probabilities over the blank and the LM alphabet (added in
                                     the cached copy by ``bias.py``; for ``decoder="beam"``)

Everything else it needs (sample rate, normalisation, the conv front end, the blank id and the
vocabulary file) is read from the folder's ``netra-model.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..text import CharVocab
from .bias import INPUT as BIAS_INPUT
from .bias import LOGP, biased_graph, language_bias, lm_alphabet
from .hub import load_sessions, model_file, read_config

#: ``decoder`` values: ``greedy`` (the per-frame argmax) or ``beam`` (CTC beam search, with the
#: character language model when one is set: ``Recognizer.set_lm``)
DECODERS = ("greedy", "beam")


def _graph_vocab(path: Path) -> list[str] | None:
    """The vocabulary pieces of the model folder a graph file is in (for the ``sub_logp`` output)."""
    try:
        config = read_config(path.parent)
        return CharVocab.load(model_file(path.parent, config["components"]["tokenizer"]["vocab"])).pieces
    except (FileNotFoundError, KeyError, ValueError):
        return None


class Recognizer:
    def __init__(self, source: str | Path, device: str | None = "auto") -> None:
        self.root, self.config, self.variant, self.sessions, self.device = load_sessions(
            source, device,
            prepare=lambda role, path: biased_graph(path, f"{Path(str(source)).name}-{role}", _graph_vocab(path)))
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
        #: False when the graph has no ``sub_logp`` output: only greedy decoding is possible
        self.beamable = LOGP in self.session.output_names
        self._biases: dict[str, np.ndarray] = {}
        self.alphabet_ids = lm_alphabet(self.vocab.pieces)
        #: per ``sub_logp`` column (0 is the blank): is it a Khmer character
        self._khmer_cols = np.array([False] + [("ក" <= self.vocab.pieces[i] <= "៿" or "᧠" <= self.vocab.pieces[i] <= "᧿")
                                               for i in self.alphabet_ids])
        self.lm = None
        self._lm_symbols = None
        #: beam search settings (``beam.beam_search`` keywords); ``set_lm`` may change them
        self.beam_options: dict = {"beam": 8, "lm_weight": 0.0, "insertion_bonus": 0.0}

    def set_lm(self, lm, **beam_options) -> None:
        """Use ``lm`` (a ``CharLM``, a path to one, or None) for ``decoder="beam"``, with these
        ``beam_search`` settings (e.g. ``lm_weight``, ``insertion_bonus``, ``beam``)."""
        from .lm import CharLM, symbol_map

        if lm is not None and not isinstance(lm, CharLM):
            lm = CharLM(lm)
        self.lm = lm
        self._lm_symbols = symbol_map(lm, self.vocab.pieces, self.alphabet_ids) if lm is not None else None
        self.beam_options.update(beam_options)

    def _text(self, out: dict, row: int, first: int, decoder: str, options: dict | None) -> str:
        """Text of row ``row`` of a graph output, from frame ``first`` on."""
        n = int(out["frame_lens"][row])
        if decoder == "greedy":
            return self.vocab.greedy(out["ids"][row, first:n].tolist())
        if decoder != "beam":
            raise ValueError(f"decoder must be one of {DECODERS}, not {decoder!r}")
        if not self.beamable:
            raise RuntimeError("beam search needs the onnx package (pip install onnx) to prepare the graph")
        from .beam import beam_search

        opts = {**self.beam_options, **(options or {})}
        if "insertion_bonus_latin" in opts:  # Khmer columns take insertion_bonus, the rest this
            latin = opts.pop("insertion_bonus_latin")
            opts["insertion_bonus"] = np.where(self._khmer_cols, float(opts.get("insertion_bonus", 0.0)), latin)
        cols = beam_search(out[LOGP][row, first:n], self.lm, self._lm_symbols, **opts)
        return self.vocab.decode([self.alphabet_ids[c - 1] for c in cols])

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

    def transcribe_tail(self, wav: np.ndarray, skip: int, language: str | None = "auto",
                        decoder: str = "greedy", beam_options: dict | None = None) -> str:
        """Text for ``wav[skip:]`` only, decoded with ``wav[:skip]`` in view as context: the frames
        that cover the first ``skip`` samples are dropped before CTC collapsing. Live transcription
        passes the end of the previous utterance this way (``karna.streaming``)."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        out = self.session.run(self._inputs(self.normalize(wav)[None, :], np.array([len(wav)], np.int64), language))
        first = int(self.frame_lengths(np.array([skip]))[0]) if skip > 0 else 0
        return self._text(out, 0, first, decoder, beam_options)

    def transcribe(self, wavs: list[np.ndarray], batch_size: int = 8, language: str | None = "auto",
                   decoder: str = "greedy", beam_options: dict | None = None,
                   skips: list[int] | None = None) -> list[str]:
        """Text for each 1-D float32 waveform at ``sample_rate`` (same order).

        ``language`` limits the characters the model may emit (``bias.language_bias``): ``auto``
        allows Khmer and ASCII, ``km`` Khmer without Latin letters, ``en`` ASCII only, and ``None``
        every character (the published graph's own argmax).

        ``decoder``: ``greedy`` (default) or ``beam`` (CTC beam search, with the language model set
        by ``set_lm``; ``beam_options`` override ``beam_options`` for this call).

        ``skips``: per waveform, leading samples that are context only (as ``transcribe_tail``).

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
            firsts = (self.frame_lengths(np.array([skips[i] for i in idx], np.int64)) if skips is not None
                      else np.zeros(len(idx), np.int64))
            for row, i in enumerate(idx):
                first = int(firsts[row]) if skips is not None and skips[i] > 0 else 0
                texts[i] = self._text(out, row, first, decoder, beam_options)
        return texts
