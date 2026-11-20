"""Karna: compact Khmer and English speech recognition (structurally pruned Omnilingual ASR CTC).

    import karna
    asr = karna.load_model("karna-base")          # or "karna-large", a Hugging Face id, a folder
    print(asr.transcribe("speech.wav"))
"""

from .asr import Karna, load_model, transcribe
from .hub import available_models

__version__ = "0.1.0"
__all__ = ["Karna", "load_model", "transcribe", "available_models", "__version__"]
