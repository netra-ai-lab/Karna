"""PyTorch release folders and checkpoints: what the ``karna`` package ran before it moved to ONNX.

A release folder (``training/scripts/06_export_pruned.py``) holds ``config.json`` (the pruned
architecture), ``model.safetensors`` and ``vocab.json``; it sits at the root of each Hugging Face
model repo, for fine-tuning. A training checkpoint is a ``{"arch", "model", ...}`` ``.pt`` file.

    from karna_train.release import load_release
    model, vocab = load_release("karna-base")          # or "teacher-1b", a repo id, a folder, a .pt
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from karna.text import CharVocab

from .model import ArchConfig, PrunableCTC

#: PyTorch weights on Hugging Face: the released models (repo root) and the fine-tuned teachers
REPOS = {
    "karna-base": "Netra-Lab/karna-base",  # 71M (v2), pruned from omniASR CTC-300M
    "karna-large": "Netra-Lab/karna-large",  # 217M (v2), pruned from omniASR CTC-1B
    "teacher-300m": "Darayut/omniASR-CTC-300M-km-en",  # 325M, fine-tuned (v1 teacher), unpruned
    "teacher-1b": "Darayut/omniASR-CTC-1B-km-en",  # 975M, fine-tuned (v1 teacher), unpruned
}
RELEASE_FILES = ["config.json", "model.safetensors", "vocab.json"]


def resolve(name_or_path: str | Path) -> Path:
    """A local folder or file for ``name_or_path``; Hub repos are downloaded (and cached) first."""
    path = Path(name_or_path).expanduser()
    if path.exists():
        return path
    repo = REPOS.get(str(name_or_path), str(name_or_path))
    if "/" not in repo:
        raise ValueError(f"unknown model {name_or_path!r}: use one of {sorted(REPOS)}, a Hugging Face id or a path")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=RELEASE_FILES))


def load_weights(path: Path) -> tuple[dict, dict[str, torch.Tensor], Path | None]:
    """(architecture dict, state dict, vocab.json or None) from a release folder or a training checkpoint."""
    if path.is_dir():
        from safetensors.torch import load_file

        arch = json.loads((path / "config.json").read_text())
        vocab = path / "vocab.json"
        return arch, load_file(str(path / "model.safetensors")), vocab if vocab.exists() else None
    ck = torch.load(path, map_location="cpu", weights_only=False)  # {"arch", "model", ...} from training
    return ck["arch"], ck["model"], None


def load_release(name_or_path: str | Path) -> tuple[PrunableCTC, CharVocab]:
    """The model (eval mode, CPU) and its vocabulary."""
    arch, state, vocab_path = load_weights(resolve(name_or_path))
    model = PrunableCTC(ArchConfig.from_dict(arch))
    model.load_state_dict(state)
    vocab = CharVocab.load(vocab_path) if vocab_path else CharVocab.default()
    return model.eval(), vocab
