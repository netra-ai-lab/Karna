import json

import numpy as np
import pytest

from karna.cli import main


def test_models(capsys):
    main(["models"])
    out = capsys.readouterr().out
    assert "karna-base\t71M\thttps://huggingface.co/Netra-Lab/karna-base" in out
    assert "karna-large" in out


def test_transcribe_formats(tiny_bundle, tmp_path, monkeypatch, capsys):
    sf = pytest.importorskip("soundfile")
    monkeypatch.setenv("KARNA_BUNDLES", str(tiny_bundle.parent))
    rng = np.random.default_rng(0)
    paths = []
    for i, n in enumerate((16000, 24000)):
        paths.append(str(tmp_path / f"{i}.wav"))
        sf.write(paths[-1], rng.standard_normal(n).astype(np.float32) * 0.1, 16000)

    main(["transcribe", *paths, "--device", "cpu"])
    tsv = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[0] for line in tsv] == paths

    main(["transcribe", *paths, "--device", "cpu", "--format", "json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["model"] == "karna-base" and doc["device"] == "cpu"
    assert [r["text"] for r in doc["results"]] == [line.split("\t", 1)[1] for line in tsv]

    main(["transcribe", paths[0], "--device", "cpu", "--format", "txt"])
    assert capsys.readouterr().out.strip() == doc["results"][0]["text"]
