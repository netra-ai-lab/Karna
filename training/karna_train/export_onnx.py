"""Export a Karna model as the ONNX model folder the ``karna`` package runs, and check it.

    python -m karna_train.export_onnx SOURCE OUT [--name karna-base] [--check AUDIO]

SOURCE is anything ``karna_train.release.load_release`` reads: a PyTorch release folder
(``scripts/06_export_pruned.py``), a ``{"arch", "model"}`` checkpoint, a short name or a Hugging
Face repo id. OUT receives:

    model.onnx         audio [N,S] f32, audio_lens [N] i64 -> ids [N,T] i64, frame_lens [N] i64
    vocab.json         the character vocabulary
    netra-model.json   everything the package needs to run it, plus a sha256 per file

The folder is then run **through the package's own code** (``karna.inference.Recognizer``) and
compared with the PyTorch model: on mixed-length random batches, and on real audio with
``--check`` (a folder of audio files, or a text file with one path per line). The export
fails below 99% identical transcripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from karna.text import CharVocab

from .export_utils import parity_check, write_config
from .model import PrunableCTC, normalize_audio_
from .release import load_release

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".opus"}


class OnnxCTC(nn.Module):
    """``PrunableCTC.forward`` for inference, traceable: the padding mask is always built
    (forward skips it when nothing is padded, a data-dependent branch), and the graph ends in
    the per-frame argmax, so the runtime never copies [N, T, 9812] logits off the device."""

    def __init__(self, model: PrunableCTC) -> None:
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor, audio_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        frame_lens = m.frame_lengths(audio_lens)
        x = m.encoder_frontend.feature_extractor(audio)
        t = x.shape[1]
        frame_lens = frame_lens.clamp(max=t)
        pad = torch.arange(t, device=x.device)[None, :] >= frame_lens[:, None]
        x = m._frontend_rest(x, pad)
        attn_mask = (~pad)[:, None, None, :]
        for layer in m.encoder.layers:
            x = layer(x, attn_mask)
        ids = m.final_proj(m.encoder.layer_norm(x)).argmax(-1)
        return ids, frame_lens


def _plain(model: PrunableCTC) -> PrunableCTC:
    """A copy with the weight-norm parametrization folded into a plain weight (simpler graph).

    Rebuilt rather than deep-copied: a copy shares the parametrized class with the original,
    and removing the parametrization from one would break the other."""
    fresh = PrunableCTC(model.cfg)
    fresh.load_state_dict(model.state_dict())
    model = fresh.eval()
    conv = model.encoder_frontend.pos_conv
    if torch.nn.utils.parametrize.is_parametrized(conv, "weight"):
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    if any(True for _ in model.hard_concretes()):
        raise ValueError("the model still has Hard Concrete gates: export the pruned model (after prune())")
    return model


def export_graph(model: PrunableCTC, path: Path, opset: int = 18, exporter: str = "dynamo") -> None:
    wrapper = OnnxCTC(_plain(model)).eval()
    audio = torch.randn(2, 16000 * 2)
    lens = torch.tensor([16000 * 2, 16000], dtype=torch.int64)
    with torch.no_grad():
        if exporter == "dynamo":
            batch, samples = torch.export.Dim("batch", min=1), torch.export.Dim("samples", min=800)
            program = torch.onnx.export(
                wrapper, (audio, lens), dynamo=True, opset_version=opset, external_data=False, verbose=False,
                input_names=["audio", "audio_lens"], output_names=["ids", "frame_lens"],
                dynamic_shapes={"audio": {0: batch, 1: samples}, "audio_lens": {0: batch}},
            )
            program.save(str(path))
        else:
            torch.onnx.export(
                wrapper, (audio, lens), str(path), dynamo=False, opset_version=opset,
                input_names=["audio", "audio_lens"], output_names=["ids", "frame_lens"],
                dynamic_axes={"audio": {0: "batch", 1: "samples"}, "audio_lens": {0: "batch"},
                              "ids": {0: "batch", 1: "frames"}, "frame_lens": {0: "batch"}},
            )


def model_config(name: str, model: PrunableCTC, source: str, version: str) -> dict:
    cfg = model.cfg
    return {
        "name": name,
        "version": version,
        "task": "speech-recognition",
        "graphs": {"model": "model.onnx"},
        "components": {
            "preprocess": {"type": "waveform", "sample_rate": 16000, "channels": "mean",
                           "normalize": "utterance-mean-std", "eps": 1e-7},
            "frontend": {"conv_layers": [list(c) for c in cfg.conv_layers]},
            "decoder": {"type": "ctc-greedy", "blank": 0},
            "tokenizer": {"type": "vocab", "vocab": "vocab.json"},
        },
        "metadata": {
            "source_checkpoint": source,
            "parameters": sum(p.numel() for p in model.parameters()),
            "encoder_layers": cfg.num_layers,
            "attention_heads": sum(cfg.num_heads),
            "ffn_units": sum(cfg.ffn_dims),
            "languages": ["km", "en"],
            "license": "Apache-2.0",
        },
    }


def torch_transcribe(model: PrunableCTC, vocab: CharVocab, wavs: list[np.ndarray]) -> list[str]:
    """What the package did before ONNX, in fp32 on CPU, one utterance at a time."""
    texts = []
    with torch.no_grad():
        for wav in wavs:
            audio = torch.from_numpy(wav.copy())[None]
            lens = torch.tensor([audio.shape[1]])
            normalize_audio_(audio, lens)
            out = model(audio, lens)
            texts.append(vocab.greedy(out["logits"][0, : int(out["frame_lens"][0])].argmax(-1).tolist()))
    return texts


def check_inputs(spec: str | None) -> list[Path]:
    if not spec:
        return []
    path = Path(spec)
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    lines = [line.split("\t")[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [Path(line) if Path(line).is_absolute() else path.parent / line for line in lines if line]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="release folder, checkpoint, short name or Hugging Face id")
    ap.add_argument("out", type=Path, help="the model folder to write")
    ap.add_argument("--name", help="model name in netra-model.json (default: OUT's folder name)")
    ap.add_argument("--version", default="2.0.0", help="model version in netra-model.json")
    ap.add_argument("--check", help="real audio for the parity check: a folder, or a file with one path per line")
    ap.add_argument("--limit", type=int, default=200, help="at most this many --check files")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--exporter", choices=["dynamo", "torchscript"], default="dynamo")
    ap.add_argument("--min-identical", type=float, default=0.99)
    args = ap.parse_args(argv)

    model, vocab = load_release(args.source)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    export_graph(model, out / "model.onnx", opset=args.opset, exporter=args.exporter)
    vocab.save(out / "vocab.json")
    name = args.name or out.name
    write_config(out, model_config(name, model, str(args.source), args.version))
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"wrote {out} ({size:.0f} MB)")

    from karna.inference import Recognizer
    from karna.audio import load_audio

    rec = Recognizer(out, device="cpu")
    onnx_predict = lambda wavs: rec.transcribe(wavs)  # noqa: E731
    torch_predict = lambda wavs: torch_transcribe(model, vocab, wavs)  # noqa: E731

    # The frame counts the runtime computes must match the graph's.
    lens = np.array([16000 * 3, 12345, 801], dtype=np.int64)
    graph = rec.session.run({"audio": np.zeros((3, int(lens.max())), np.float32), "audio_lens": lens})
    if not np.array_equal(graph["frame_lens"], rec.frame_lengths(lens)):
        sys.exit(f"frame_lens differ: graph {graph['frame_lens']} vs config {rec.frame_lengths(lens)}")

    rng = np.random.default_rng(0)
    noise = [rng.standard_normal(n).astype(np.float32) * 0.1 for n in rng.integers(8000, 16000 * 12, 16)]
    score = parity_check(onnx_predict, torch_predict, noise, "random clips (mixed lengths, batched)")
    files = check_inputs(args.check)[: args.limit]
    if files:
        wavs = [load_audio(p) for p in files]
        score = min(score, parity_check(onnx_predict, torch_predict, wavs, f"{len(files)} files from {args.check}"))
    else:
        print("no --check audio: only random clips were compared; check real audio before publishing")
    if score < args.min_identical:
        sys.exit(f"parity {score:.1%} is below {args.min_identical:.0%}: not a usable export")
    print(f"ok: use it with  KARNA_BUNDLES={out.parent} karna transcribe ... --model {name}")


if __name__ == "__main__":
    main()
