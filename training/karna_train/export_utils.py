# From the Netra Lab kit (templates/repo), after Netra-OCR's
# training/netra_ocr_train/{model_folder.py,scripts/export_recognizer.py}.
# Generic: use write_config as is; call parity_check from the export script.
# See playbooks/INFERENCE.md §3, §5.
"""Model folders: write netra-model.json, and check an exported folder against PyTorch."""

import hashlib
import json
from pathlib import Path

from tqdm import tqdm

CONFIG_NAME = "netra-model.json"
FORMAT, FORMAT_VERSION = "netra-bundle", 1


def write_config(root: str | Path, config: dict) -> Path:
    """Write ``config`` into ``root``, filling in the format header and the sha256 of every file."""
    root = Path(root)
    out = {"format": FORMAT, "format_version": FORMAT_VERSION,
           **{k: v for k, v in config.items() if k not in ("format", "format_version", "files")}}
    out["files"] = {str(p.relative_to(root)): _sha256(p)
                    for p in sorted(root.rglob("*")) if p.is_file() and p.name != CONFIG_NAME}
    path = root / CONFIG_NAME
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parity_check(onnx_predict, torch_predict, inputs: list, label: str, device: str = "cpu",
                 batch_size: int = 16) -> float:
    """The share of inputs on which the exported folder (run through the
    package's own inference code, ``onnx_predict``) gives exactly the PyTorch
    checkpoint's output (``torch_predict``). Both take a list, return a list.

    The export script exits non-zero below 0.99.
    """
    expected, got = [], []
    for i in tqdm(range(0, len(inputs), batch_size), desc=f"checking on {device}", unit="batch",
                  dynamic_ncols=True, leave=False, disable=None):
        batch = inputs[i:i + batch_size]
        expected += torch_predict(batch)
        got += onnx_predict(batch)
    same = sum(a == b for a, b in zip(got, expected))
    print(f"identical to PyTorch on {same}/{len(inputs)} of {label}")
    for i, (want, have) in [(i, p) for i, p in enumerate(zip(expected, got)) if p[0] != p[1]][:5]:
        print(f"   #{i}\n     torch: {want}\n     onnx:  {have}")
    return same / max(1, len(inputs))
