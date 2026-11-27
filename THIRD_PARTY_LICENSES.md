# Third-party licences

Karna is released under the Apache License 2.0 (see [LICENSE](LICENSE)). It builds on the following.

## Models

| Work | Licence | Used for |
| --- | --- | --- |
| [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) `omniASR_CTC_300M` and `omniASR_CTC_1B` (Meta) | Apache-2.0 | The starting weights and the character vocabulary. karna-base and karna-large are derived from them and released under Apache-2.0. |
| [karna-lm](https://huggingface.co/Netra-Lab/karna-lm) `karna-char8.npz` (Netra Lab) | CC BY-SA 4.0 (it is derived in part from the DDD transcripts, CC BY-SA 4.0, and LibriSpeech, CC BY 4.0) | The character language model for beam search, downloaded on first use and baked into the Docker image |
| [Silero VAD](https://github.com/snakers4/silero-vad) v5.1.2 `silero_vad.onnx` (Silero Team) | MIT | Not redistributed: downloaded unchanged from its v5.1.2 tag on first use of the optional `vad="silero"` detector |

## Code

| Work | Licence | Used for |
| --- | --- | --- |
| [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) Hard Concrete gates, from DPHuBERT and FLOP | MIT | `training/karna_train/hardconcrete.py` (training only; not in the `netra-karna` package) |

The `netra-karna` package depends on numpy, soundfile (libsndfile), soxr, huggingface_hub, onnx and ONNX Runtime, and with the `server` extra on FastAPI and Uvicorn. They are installed by pip under their own licences and aren't redistributed here. The Docker image contains them, each with its licence in its installed package metadata.

## Data

The models were trained on the following. None of it is redistributed in this repository, except for the short test clips in `tests/golden/`.

| Dataset | Licence |
| --- | --- |
| [DDD Khmer speech dataset](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset) (Digital Divide Data) | CC BY-SA 4.0 |
| [LibriSpeech](https://www.openslr.org/12) | CC BY 4.0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google) | CC BY 4.0 |

The four clips in `tests/golden/audio/` come from FLEURS (km) and LibriSpeech, under CC BY 4.0; [tests/golden/README.md](tests/golden/README.md) gives the rows.

The language model for beam search, [karna-lm](https://huggingface.co/Netra-Lab/karna-lm), was built by `scripts/build_char_lm.py` from Netra Lab's Khmer and Khmer–English text corpora, the DDD transcripts and the LibriSpeech transcripts; its card and manifest list them. The text isn't redistributed.

## Fonts

The web app (`karna/server/static/fonts/`) serves these fonts, each with its licence text alongside:

| Font | Licence |
| --- | --- |
| [Figtree](https://github.com/erikdkennedy/figtree) | SIL Open Font License 1.1 (`OFL-Figtree.txt`) |
| [Kantumruy Pro](https://github.com/googlefonts/kantumruy-pro) | SIL Open Font License 1.1 (`OFL-KantumruyPro.txt`) |
