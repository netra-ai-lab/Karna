"""Where the models come from, and one loaded copy of each per process.

Each released model has its own Hugging Face repo; the ONNX model folder the package runs
is ``onnx/`` in it (the PyTorch weights at the repo root are for fine-tuning). The package
loads each repo at a pinned commit, ``revision`` below, so a model upload never changes
what an installed version runs.

``KARNA_LM=<file.npz>`` is the character language model for beam search (``scripts/build_char_lm.py``);
the beam settings tuned for each model are read from ``<file>.json`` next to it (``"beam"``).
``KARNA_LM=none`` turns it off.

``KARNA_BUNDLES=<dir>`` points at local model folders with the same names
(``<dir>/karna-base/netra-model.json``, e.g. from ``training/export.sh``); a model missing
from it falls back to Hugging Face. ``KARNA_DEVICE`` picks the device ("cpu", "cuda", ...;
unset: GPU if usable).
"""

from __future__ import annotations

import os
from pathlib import Path

from .inference.hub import CONFIG_NAME

#: short name -> Hugging Face repo, the ONNX folder in it, and the commit this version runs.
#: Move a revision only together with a package release, after uploading (docs/RELEASING.md).
MODELS = {
    "karna-base": {"repo": "Netra-Lab/karna-base", "folder": "onnx",
                   "revision": "563d3649971a8f554ec589e5e6d4297079e7173a",
                   "params": "71M", "about": "pruned from omniASR CTC-300M; fastest"},
    "karna-large": {"repo": "Netra-Lab/karna-large", "folder": "onnx",
                    "revision": "d124eca52b4911bd651b3804f775ca63855cc423",
                    "params": "217M", "about": "pruned from omniASR CTC-1B; most accurate, best English"},
}
DEFAULT_MODEL = "karna-base"

_loaded: dict = {}


def available_models() -> dict[str, str]:
    """Short name -> Hugging Face repo id."""
    return {name: m["repo"] for name, m in MODELS.items()}


def source(name_or_path: str | Path = DEFAULT_MODEL) -> str:
    """A model folder for a short name, a local folder, a Hugging Face repo id or an ``hf://`` source.

    Local folders win: ``KARNA_BUNDLES/<name>`` for a short name, or the path itself.
    """
    text = str(name_or_path)
    if text.startswith("hf://"):
        return text
    path = Path(text).expanduser()
    if path.exists():
        for folder in (path, path / "onnx"):
            if (folder / CONFIG_NAME).is_file():
                return str(folder)
        raise ValueError(
            f"{path} is not an ONNX model folder (no {CONFIG_NAME}). PyTorch release folders and "
            "training checkpoints run in training/ (karna_train.release.load_release); "
            "turn them into a model folder with training/export.sh."
        )
    if text in MODELS:
        bundles = os.environ.get("KARNA_BUNDLES")
        if bundles and (Path(bundles) / text / CONFIG_NAME).is_file():
            return str(Path(bundles) / text)
        m = MODELS[text]
        return f"hf://{m['repo']}/{m['folder']}@{m['revision']}"
    if text.count("/") == 1:  # another Hugging Face repo with the same layout, at its latest commit
        return f"hf://{text}/onnx"
    raise ValueError(f"unknown model {text!r}: use one of {sorted(MODELS)}, a Hugging Face id or a model folder")


def get(name_or_path: str | Path = DEFAULT_MODEL, device: str | None = None):
    """The loaded :class:`~karna.inference.Recognizer` for ``name_or_path`` (once per process and device)."""
    from .inference import Recognizer
    from .inference.session import resolve_device

    key = (source(name_or_path), resolve_device(device))
    if key not in _loaded:
        _loaded[key] = Recognizer(key[0], device=key[1])
    return _loaded[key]


def lm_settings(name_or_path: str | Path) -> tuple[Path, dict] | None:
    """The language model to use with a model, and its beam settings: from ``KARNA_LM``, else none.

    Settings come from the LM's manifest (``<lm>.json``, ``"beam"``: per model name, or
    ``"default"``); a model without tuned settings gets ``lm_weight`` 0.5.
    """
    import json

    env = os.environ.get("KARNA_LM", "").strip()
    if not env or env.lower() == "none":
        return None
    path = Path(env).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"KARNA_LM={env}: no such file")
    settings: dict = {"lm_weight": 0.5}
    manifest = path.with_suffix(".json")
    if manifest.is_file():
        beam = json.loads(manifest.read_text(encoding="utf-8")).get("beam", {})
        key = Path(str(name_or_path)).name
        settings = dict(beam.get(key) or beam.get("default") or settings)
    return path, settings
