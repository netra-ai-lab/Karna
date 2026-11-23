# Golden clips

`golden.json` holds each released model's exact text for the clips in `audio/`, written by
`scripts/make_golden.py`. `pytest --slow` and the Docker build's smoke test compare against it.
`references.tsv` has the human transcripts.

| File | Source | Licence |
| --- | --- | --- |
| `fleurs-km-validation-0000.flac`, `-0012.flac` | [FLEURS](https://huggingface.co/datasets/google/fleurs) km_kh validation, rows 0 and 12 | CC BY 4.0 |
| `librispeech-dummy-0014.flac`, `-0019.flac` | [LibriSpeech](https://www.openslr.org/12) via `hf-internal-testing/librispeech_asr_dummy`, rows 14 and 19 | CC BY 4.0 |

Converted to 16-bit FLAC; otherwise unchanged.
