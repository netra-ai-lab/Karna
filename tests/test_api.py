import json
import subprocess
import sys

import numpy as np
import pytest

import karna
from karna.audio import load_audio
from karna.inference import Recognizer


def clips(seed=0, lengths=(16000, 5000, 11000, 40000)):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n).astype(np.float32) * 0.1 for n in lengths]


def test_onnx_matches_pytorch(tiny_model, tiny_bundle):
    from karna_train.export_onnx import torch_transcribe

    model, vocab = tiny_model
    rec = Recognizer(tiny_bundle, device="cpu")
    wavs = clips()
    expected = torch_transcribe(model, vocab, wavs)
    assert rec.transcribe(wavs, batch_size=4) == expected  # batched (padded) vs one at a time
    assert all(expected), "the test model should produce text, or the comparison proves nothing"


def test_frame_lengths_match_graph(tiny_bundle):
    rec = Recognizer(tiny_bundle, device="cpu")
    lens = np.array([16000, 12345, 801, 400], dtype=np.int64)
    out = rec.session.run({"audio": np.zeros((4, 16000), np.float32), "audio_lens": lens})
    assert np.array_equal(out["frame_lens"], rec.frame_lengths(lens))


def test_load_model_api(tiny_bundle, monkeypatch):
    monkeypatch.setenv("KARNA_BUNDLES", str(tiny_bundle.parent))
    asr = karna.load_model("karna-base", device="cpu")
    assert asr.device == "cpu" and asr.num_parameters > 0 and asr.sample_rate == 16000
    wavs = clips(1)
    batched = asr.transcribe(wavs, batch_size=3)
    assert batched == [asr.transcribe(w) for w in wavs]
    assert isinstance(asr.transcribe(wavs[0]), str)
    assert karna.transcribe(wavs[0], device="cpu") == batched[0]
    assert karna.load_model(tiny_bundle).transcribe(wavs[0]) == batched[0]  # a folder works too


def test_pytorch_folder_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="training/export.sh"):
        karna.load_model(tmp_path)


def test_unknown_short_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        karna.load_model("not-a-model")


def test_load_audio_files_bytes_and_rates(tmp_path):
    sf = pytest.importorskip("soundfile")
    stereo = np.stack([np.ones(8000), np.zeros(8000)], axis=1).astype(np.float32)
    sf.write(tmp_path / "a.wav", stereo, 8000)
    wav = load_audio(tmp_path / "a.wav")
    assert wav.dtype == np.float32 and wav.ndim == 1 and abs(len(wav) - 16000) <= 1
    assert abs(wav[100:-100].mean() - 0.5) < 1e-3
    assert np.array_equal(load_audio((tmp_path / "a.wav").read_bytes()), wav)
    assert len(load_audio((np.zeros(44100, dtype=np.float32), 44100))) == 16000
    with pytest.raises(ValueError):
        load_audio(np.zeros(0, dtype=np.float32))


def test_import_stays_light():
    code = ("import sys, karna; print(','.join(m for m in ('torch', 'onnxruntime', 'onnx', 'fastapi', 'karna_train') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == ""


def test_config_names_every_file(tiny_bundle):
    config = json.loads((tiny_bundle / "netra-model.json").read_text())
    assert set(config["files"]) == {"model.onnx", "vocab.json"}
    assert config["components"]["decoder"] == {"type": "ctc-greedy", "blank": 0}
