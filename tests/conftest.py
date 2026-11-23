"""Shared fixtures. The fast tests need no published model: ``tiny_bundle`` exports a tiny random
model to an ONNX model folder (needs torch + onnxscript, as in CI). ``--slow`` adds the tests
that download the published models and compare their text with tests/golden/."""

import json

import pytest


@pytest.fixture(autouse=True, scope="session")
def _graph_cache(tmp_path_factory):
    """Token-bias graphs of the tiny test models go to a temporary cache, not ~/.cache/karna (bias.py).
    With --slow the user's cache is kept, so the published models' copies are made once."""
    import os

    if os.environ.get("KARNA_CACHE_DIR") is None and not _slow:
        os.environ["KARNA_CACHE_DIR"] = str(tmp_path_factory.mktemp("karna-cache"))
    yield


_slow = False


def pytest_configure(config):
    global _slow
    _slow = config.getoption("--slow")


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run the tests that need the published models")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="needs --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_model():
    """(PrunableCTC, CharVocab): non-uniform layers like a pruned model, one head-less and one FFN-less
    layer, and the real 320x conv stride and 9,812-character vocabulary."""
    torch = pytest.importorskip("torch")
    from karna.text import CharVocab
    from karna_train.model import ArchConfig, PrunableCTC

    vocab = CharVocab.default()
    cfg = ArchConfig(conv_layers=[(16, 10, 5), (16, 4, 4), (16, 4, 4), (12, 2, 4)],
                     num_heads=[4, 0, 3], ffn_dims=[64, 16, 0], model_dim=32, head_dim=8,
                     pos_conv_kernel=4, pos_conv_groups=2, vocab_size=len(vocab))
    torch.manual_seed(0)
    model = PrunableCTC(cfg).eval()
    with torch.no_grad():  # a sharper head, so random audio decodes to varied, non-empty text
        model.final_proj.weight.mul_(20)
    return model, vocab


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory, tiny_model):
    """A model folder named karna-base inside a KARNA_BUNDLES-style directory."""
    pytest.importorskip("onnxscript")
    from karna_train.export_onnx import export_graph, model_config
    from karna_train.export_utils import write_config

    model, vocab = tiny_model
    root = tmp_path_factory.mktemp("bundles")
    out = root / "karna-base"
    out.mkdir()
    export_graph(model, out / "model.onnx")
    vocab.save(out / "vocab.json")
    write_config(out, model_config("karna-base", model, "tiny-test", "0.0.0"))
    assert json.loads((out / "netra-model.json").read_text())["files"]["model.onnx"]
    return out
