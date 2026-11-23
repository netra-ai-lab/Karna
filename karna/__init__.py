"""Karna: compact Khmer and English speech recognition (structurally pruned Omnilingual ASR CTC),
on ONNX Runtime.

    import karna
    print(karna.transcribe("speech.wav"))           # karna-base, downloaded on first use

    asr = karna.load_model("karna-large")           # or a Hugging Face id, a model folder
    asr.transcribe(["a.wav", "b.flac"])

Package map:
    asr.py         Karna, load_model, transcribe: the high-level API
    audio.py       files, bytes and arrays -> mono 16 kHz float32
    text.py        the Omnilingual character vocabulary and greedy CTC decoding
    models.py      the released models, their Hugging Face repos and pinned revisions
    inference/     ONNX Runtime: devices (session.py), model folders (hub.py), the recogniser
    cli.py         the ``karna`` command: transcribe, models, download, serve
    server/        optional (``[server]`` extra): FastAPI REST API + web app
"""

from importlib.metadata import PackageNotFoundError, version

from .asr import Karna, load_model, transcribe
from .models import available_models

try:
    __version__ = version("netra-karna")
except PackageNotFoundError:  # a source checkout that isn't installed
    __version__ = "0.0.0"

__all__ = ["Karna", "load_model", "transcribe", "available_models", "__version__"]
