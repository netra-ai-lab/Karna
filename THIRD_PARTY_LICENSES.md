# Third-party licences

Karna is released under the Apache License 2.0 (see [LICENSE](LICENSE)). It builds on the following.

## Models

| Work | Licence | Used for |
| --- | --- | --- |
| [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) `omniASR_CTC_300M` and `omniASR_CTC_1B` (Meta) | Apache-2.0 | The starting weights and the character vocabulary. karna-base and karna-large are derived from them and released under Apache-2.0. |

## Code

| Work | Licence | Used for |
| --- | --- | --- |
| [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) Hard Concrete gates, from DPHuBERT and FLOP | MIT | `training/karna_train/hardconcrete.py` (training only; not in the `netra-karna` package) |

The `netra-karna` package depends on numpy, soundfile (libsndfile), soxr, huggingface_hub and ONNX Runtime, and with the `server` extra on FastAPI and Uvicorn. They are installed by pip under their own licences and aren't redistributed here. The Docker image contains them, each with its licence in its installed package metadata.

## Data

The models were trained on the following. None of it is redistributed in this repository, except for the short test clips in `tests/golden/`.

| Dataset | Licence |
| --- | --- |
| [DDD Khmer speech dataset](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset) (Digital Divide Data) | CC BY-SA 4.0 |
| [LibriSpeech](https://www.openslr.org/12) | CC BY 4.0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google) | CC BY 4.0 |

The four clips in `tests/golden/audio/` come from FLEURS (km) and LibriSpeech, under CC BY 4.0; [tests/golden/README.md](tests/golden/README.md) gives the rows.

## Fonts

The web app (`karna/server/static/fonts/`) serves these fonts, each with its licence text alongside:

| Font | Licence |
| --- | --- |
| [Figtree](https://github.com/erikdkennedy/figtree) | SIL Open Font License 1.1 (`OFL-Figtree.txt`) |
| [Kantumruy Pro](https://github.com/googlefonts/kantumruy-pro) | SIL Open Font License 1.1 (`OFL-KantumruyPro.txt`) |
