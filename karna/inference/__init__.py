"""Karna's model inference on ONNX Runtime (no PyTorch).

Each model is a folder holding ``model.onnx``, ``vocab.json`` and ``netra-model.json`` (its
settings), published as ``onnx/`` in the Hugging Face repos ``Netra-Lab/karna-base`` and
``Netra-Lab/karna-large``. This package does everything around the graph: picking the device,
downloading only the files that device needs, and the pre- and post-processing.

    from karna.inference import Recognizer
    rec = Recognizer("hf://Netra-Lab/karna-base/onnx", device="auto")
    print(rec.transcribe([waveform_16k]))
"""

from .recognizer import Recognizer
from .session import available_devices, resolve_device

__all__ = ["Recognizer", "available_devices", "resolve_device"]
