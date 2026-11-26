"""Beam search, the character LM, the ``sub_logp`` graph output, and the Silero detector (no Karna model)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from karna.inference.beam import beam_search
from karna.inference.lm import CharLM

ROOT = Path(__file__).resolve().parents[1]


def peaked(path, size=6, blank_prob=0.9):
    """[T, size] log-probabilities, each frame peaked on ``path[t]`` (0 is the blank)."""
    rest = (1 - blank_prob) / (size - 1)
    out = np.full((len(path), size), rest)
    out[np.arange(len(path)), path] = blank_prob
    return np.log(out).astype(np.float32)


def greedy(path):
    out, prev = [], None
    for c in path:
        if c != prev and c != 0:
            out.append(c)
        prev = c
    return out


@pytest.mark.parametrize("path", [[0, 1, 1, 0, 2, 2, 2, 0], [3, 3, 0, 3, 4, 0, 0, 5], [0, 0, 0], [1, 2, 1, 2, 0, 1]])
def test_beam_without_lm_is_greedy_on_peaked_input(path):
    assert beam_search(peaked(path), None) == greedy(path)


def test_blank_frames_are_skipped_but_repeats_still_split():
    lp = peaked([1, 0, 1], blank_prob=0.9995)
    assert beam_search(lp, None) == [1, 1]


def test_insertion_bonus_per_column():
    # an ambiguous frame: blank 0.5, character 1 0.5; a per-column bonus tips it either way
    lp = np.log(np.array([[0.5, 0.5, 1e-9]], np.float64)).astype(np.float32)
    assert beam_search(lp, None, insertion_bonus=np.array([0, 1.0, 0])) == [1]
    assert beam_search(lp, None, insertion_bonus=np.array([0, -1.0, 0])) == []


@pytest.fixture(scope="module")
def toy_lm(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_char_lm as b

    enc = b.Encoder()
    lines = ["the cat sat on the mat", "the dog sat on the log", "a cat and a dog", "ខ្ញុំទៅសាលារៀន"] * 3
    buf = bytearray(b"\0")
    for line in lines:
        buf += enc.encode(line)[0].encode("latin-1") + b"\0"
    built = b.build(np.frombuffer(bytes(buf), np.uint8), 4, [0, 0, 0, 0], lambda m: None)
    path = tmp_path_factory.mktemp("lm") / "toy.npz"
    b.save(path, built["tables"], 4, enc.alphabet, {})
    return CharLM(path), enc


def test_lm_distributions_sum_to_one(toy_lm):
    lm, enc = toy_lm
    state = lm.start()
    for ch in "the cat":
        assert np.exp(lm.scores(state)).sum() == pytest.approx(1.0, abs=2e-3)
        state = lm.advance(state, ord(enc.encode(ch)[0]) if ch != " " else enc.space)


def test_lm_prefers_seen_text(toy_lm):
    lm, enc = toy_lm

    def score(text):
        return lm.score_text([ord(c) for c in enc.encode(text)[0]])

    assert score("the cat sat on the mat") > score("mat the on sat cat the") + 10


def test_lm_steers_the_beam(toy_lm):
    lm, enc = toy_lm
    # columns: blank, then the LM's symbols; two characters equally likely in the middle frame
    cols = 1 + len(lm.alphabet)
    t, h, e, x = (ord(enc.encode(c)[0]) for c in "thex")
    lp = np.full((5, cols), np.log(1e-6), np.float32)
    lp[0, t] = lp[1, 0] = lp[3, 0] = 0
    lp[2, h] = lp[2, x] = np.log(0.5)
    lp[4, e] = 0
    symbols = np.arange(cols)
    symbols[0] = -1
    assert beam_search(lp, lm, symbols, lm_weight=1.0) == [t, h, e]


def test_sub_logp_output(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    import onnxruntime as ort

    from karna.inference.bias import LOGP, biased_graph, lm_subset

    pieces = ["<s>", "<pad>", "</s>", "<unk>", "▁", "a", "b", "ក", "中"]
    v = len(pieces)
    x = onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, "T", v])
    y = onnx.helper.make_tensor_value_info("ids", onnx.TensorProto.INT64, [1, "T"])
    bias = onnx.numpy_helper.from_array(np.zeros(v, np.float32), "out_bias")
    graph = onnx.helper.make_graph([onnx.helper.make_node("Add", ["scores", "out_bias"], ["logits"]),
                                    onnx.helper.make_node("ArgMax", ["logits"], ["ids"], axis=-1, keepdims=0)],
                                   "g", [x], [y], [bias])
    src = tmp_path / "model.onnx"
    # IR 8: loadable by every onnxruntime the package supports (a new onnx writes a newer IR by default)
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8), str(src))
    monkeypatch.setenv("KARNA_CACHE_DIR", str(tmp_path / "cache"))
    path = biased_graph(src, "toy", pieces)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    scores = np.random.default_rng(0).standard_normal((1, 3, v)).astype(np.float32)
    ids, logp = sess.run(["ids", LOGP], {"scores": scores, "token_bias": np.zeros(v, np.float32)})
    subset = lm_subset(pieces)
    assert subset == [0, 4, 5, 6, 7]  # blank, space, a, b, Khmer; not the specials or CJK
    assert np.array_equal(ids, scores.argmax(-1))
    want = scores[..., subset] - np.log(np.exp(scores[..., subset]).sum(-1, keepdims=True))
    assert np.allclose(logp, want, atol=1e-5)


def test_silero_vad():
    pytest.importorskip("onnxruntime")
    from karna.streaming import Endpointer
    from karna.vad import SileroVAD

    from karna.vad import silero_path

    try:
        silero_path()
    except OSError as error:  # no network and no cached copy
        pytest.skip(f"Silero VAD not available: {error}")
    vad = SileroVAD()
    quiet = (1e-3 * np.random.default_rng(0).standard_normal(16000)).astype(np.float32)
    probs = [vad.prob(quiet[i:i + 512]) for i in range(0, 16000 - 512, 512)]
    assert max(probs) < 0.5
    ep = Endpointer(vad="silero")
    assert ep.frame == 512 and ep.push(quiet) == [] and ep.open_id is None
    assert Endpointer(vad="energy").vad is None
