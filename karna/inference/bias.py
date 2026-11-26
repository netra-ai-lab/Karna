"""Steer which characters the model may emit, by adding a per-token bias before its argmax.

Karna's vocabulary is Omnilingual's: 9,812 characters, of which 84 are Khmer and about 95 ASCII;
the rest (5,762 CJK, 1,222 Hangul, Ethiopic, accented Latin, ...) are never right for Khmer or
English speech. The published graph ends in ``ArgMax`` over the output layer's scores, so this
module derives a copy of the graph with one more input, ``token_bias`` (float32 [vocab]), added to
the scores right before that ``ArgMax``. A bias of -1e4 on a token means it is never chosen.

For beam search with a language model (``beam.py``), the copy has one more output, ``sub_logp``
(float32 [N, T, 1 + K]): log-probabilities over the blank and the K characters of the ``auto``
set (``lm_alphabet``), taken from the biased scores. The 9,812-wide scores still never leave
the session, and ``ids`` is unchanged, so greedy decoding gives the same text as before.

The copy is made once (it needs the ``onnx`` package) and cached under ``KARNA_CACHE_DIR``
(default ``~/.cache/karna/graphs``); the published files are not touched.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

INPUT = "token_bias"
#: log-probabilities over the blank and ``lm_alphabet`` (beam search input)
LOGP = "sub_logp"
BLOCK = -1e4
#: bump when the derived graph changes, so stale cached copies are not reused
GRAPH_VERSION = "lm1"
#: ``language`` values: which characters each one allows (the blank and special tokens always are)
LANGUAGES = ("auto", "km", "en")


def cache_dir() -> Path:
    return Path(os.environ.get("KARNA_CACHE_DIR") or Path.home() / ".cache" / "karna") / "graphs"


def biased_graph(path: Path, tag: str, pieces: list[str] | None = None) -> Path:
    """The cached biasable copy of the graph at ``path``, made on first use. With the vocabulary's
    ``pieces``, it also has the ``sub_logp`` output. Returns ``path`` itself when the copy can't be
    made (no ``onnx`` package, or no ``ArgMax`` in the graph)."""
    st = path.stat()
    version = f"{GRAPH_VERSION}-" if pieces is not None else ""
    target = cache_dir() / f"{tag}-{version}{st.st_size}-{st.st_mtime_ns}.onnx"
    if target.exists():
        return target
    try:
        import onnx
    except ImportError:
        logger.warning("the onnx package is not installed: language control (Khmer / English) is off")
        return path
    logger.info("preparing %s for language control (once; cached in %s)", path.name, target.parent)
    model = onnx.load(str(path))
    graph = model.graph
    argmax = next((n for n in graph.node if n.op_type == "ArgMax"), None)
    if argmax is None:
        logger.warning("%s has no ArgMax: language control is off", path)
        return path
    vocab = _vocab_size(model, argmax.input[0])
    graph.input.append(onnx.helper.make_tensor_value_info(INPUT, onnx.TensorProto.FLOAT, [vocab]))
    scores = argmax.input[0]
    add = onnx.helper.make_node("Add", [scores, INPUT], [scores + "_biased"], name="karna_token_bias")
    graph.node.insert(list(graph.node).index(argmax), add)
    argmax.input[0] = scores + "_biased"
    if pieces is not None:
        subset = onnx.numpy_helper.from_array(np.array(lm_subset(pieces), np.int64), "karna_lm_subset")
        graph.initializer.append(subset)
        gather = onnx.helper.make_node("Gather", [scores + "_biased", subset.name], [scores + "_subset"],
                                       axis=2, name="karna_lm_gather")
        logp = onnx.helper.make_node("LogSoftmax", [scores + "_subset"], [LOGP], axis=-1, name="karna_lm_logp")
        graph.node.extend([gather, logp])
        graph.output.append(onnx.helper.make_tensor_value_info(
            LOGP, onnx.TensorProto.FLOAT, ["batch", "frames", int(subset.dims[0])]))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    onnx.save(model, str(tmp))
    tmp.replace(target)
    return target


def _vocab_size(model, scores: str) -> int | str:
    """The last dimension of the scores: from the bias of the Add that makes them, else symbolic."""
    producer = next((n for n in model.graph.node if scores in n.output), None)
    sizes = {t.name: list(t.dims) for t in model.graph.initializer}
    for name in (producer.input if producer is not None else []):
        if len(sizes.get(name, [])) == 1:
            return int(sizes[name][0])
    return "vocab"


def language_bias(pieces: list[str], language: str) -> np.ndarray:
    """``token_bias`` for ``language``: 0 for allowed tokens, -1e4 for the rest.

    - ``auto``: Khmer, ASCII (letters, digits, punctuation) and the space
    - ``km``: Khmer, digits, ASCII punctuation and the space (no Latin letters)
    - ``en``: ASCII and the space
    """
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, not {language!r}")
    from ..text import SPACE_PIECE, SPECIAL

    bias = np.full(len(pieces), BLOCK, np.float32)
    for i, p in enumerate(pieces):
        if i in SPECIAL or p in (SPACE_PIECE, " "):
            bias[i] = 0
        elif len(p) == 1:
            khmer = "ក" <= p <= "៿" or "᧠" <= p <= "᧿"
            ascii_ = p.isascii() and p.isprintable()
            if language == "auto":
                ok = khmer or ascii_
            elif language == "km":
                ok = khmer or (ascii_ and not p.isalpha())
            else:
                ok = ascii_
            if ok:
                bias[i] = 0
    return bias


def lm_alphabet(pieces: list[str]) -> list[int]:
    """Vocabulary ids of the characters a language model scores: the ``auto`` set (Khmer, ASCII
    and the space) without the special tokens, in id order. LM symbol ``s`` (1..K) is
    ``lm_alphabet[s - 1]``; symbol 0 is the sentence boundary."""
    from ..text import SPECIAL

    allowed = language_bias(pieces, "auto")
    return [i for i in range(len(pieces)) if allowed[i] == 0 and i not in SPECIAL]


def lm_subset(pieces: list[str]) -> list[int]:
    """The vocabulary ids ``sub_logp`` covers: the blank, then ``lm_alphabet``. Column ``s`` of
    ``sub_logp`` is LM symbol ``s`` (column 0 is the blank)."""
    from ..text import BLANK

    return [BLANK, *lm_alphabet(pieces)]
