---
license: apache-2.0
language:
- km
- en
library_name: pytorch
pipeline_tag: automatic-speech-recognition
tags:
- omnilingual-asr
- ctc
- wav2vec2
- structured-pruning
- khmer
---

# karna-base

A **71M-parameter** Khmer + English CTC speech recogniser, made by structured pruning of
Meta's Omnilingual ASR `omniASR_CTC_300M` to **about 80% encoder sparsity**. It is the Base size of a pair,
named after DiariZen's Base-s80 / Large-s80 pruned WavLM models. The other size is
[Netra-Lab/karna-large](https://huggingface.co/Netra-Lab/karna-large).

**Code:** [github.com/netra-ai-lab/Karna](https://github.com/netra-ai-lab/Karna), with the `karna` inference package and the full training recipe.

## Results

Test sets; greedy CTC decoding, no language model. CER for Khmer, WER for English, in %. Lower is better.

| Model | Params | GMAC / s audio | DDD (km) | FLEURS-km | FLEURS-en | LibriSpeech test-clean | LibriSpeech test-other |
|---|---|---|---|---|---|---|---|
| `omniASR_CTC_300M` (stock) | 325M | 18.3 | 9.65 | 16.26 | – | 6.53 | 14.45 |
| Fine-tuned 300M (teacher) | 325M | 18.3 | 1.47 | 9.95 | 18.78 | 5.57 | 13.20 |
| **karna-base** | **71M** | **5.8** | **1.94** | **15.20** | **29.66** | **10.96** | **23.53** |

- **Khmer:** on DDD it is within 0.5 points of its unpruned fine-tuned teacher, and its error is a fifth of the stock model's.
- **English:** it keeps much less of its teacher's accuracy, with about 1.5–2× the teacher's WER, and is behind the stock model.
- **Version:** this is v2. The first release (73M; DDD 2.60, test-clean 11.15) is at revision [`5cadb20f7e`](https://huggingface.co/Netra-Lab/karna-base/tree/5cadb20f7e); v2 improves every test set.
- **In-domain test sets:** FLEURS-km and FLEURS-en train are in the training mix, so their test sets are in-domain.

<p align="center"><img src="https://huggingface.co/Netra-Lab/karna-base/resolve/main/assets/inference_time.png" width="60%" alt="Inference time for 60 s of audio at 0% and 80% encoder sparsity, for the base and large models"></p>

*Inference time* for one 60 s utterance (batch 1, bf16, median of 20 runs), at 0% sparsity (the fine-tuned teacher) and about 80% (Karna).

## Architecture

<p align="center"><img src="https://huggingface.co/Netra-Lab/karna-base/resolve/main/assets/architecture.png" width="45%" alt="Karna's architecture, reading upwards from 16 kHz audio to the CTC head, with one transformer layer opened up"></p>

*The model*, reading upwards, with one transformer layer opened up: 16 kHz audio, a feature extractor of 7 convolutional layers giving one frame every 20 ms, a convolutional position encoder added back to the frames, 48 pre-LayerNorm transformer layers of width 1,280, a final LayerNorm and a CTC head over 9,812 characters. Every attention head and every FFN unit has its own Hard Concrete gate z. The cells in the opened layer are layer 12 of karna-large: 11 of 16 heads and 3,788 of 5,120 FFN units kept; which ones are kept is schematic.

<p align="center"><img src="https://huggingface.co/Netra-Lab/karna-base/resolve/main/assets/pruning.png" width="100%" alt="The fine-tuned omniASR CTC-1B before structured pruning and karna-large after: CNN and CTC head kept, attention heads and FFN units removed"></p>

*What is pruned*, shown for karna-large: whole attention heads (MHA) and FFN units are removed from every transformer layer, and the CNN front end is kept. Each cell is a quarter of a layer's heads or units.

- **Base:** a wav2vec2-style CTC encoder with Omnilingual's 9,812-entry character vocabulary, which covers Khmer and Latin script.
- **Pruned parts:** per-layer attention heads and FFN units, so every layer has its own width. `config.json` lists the exact `num_heads` and `ffn_dims`; the encoder has 61M parameters, 19% of the 300M's 315M.
- **Unchanged parts:** the CNN feature extractor and the output layer.
- **Loading:** the layer widths are non-uniform, so fairseq2 / `omnilingual-asr` cannot load it. Use the `karna` package (see Usage) or the code bundled in this repo.

## How it was made

The recipe follows *Fine-tune Before Structured Pruning* (arXiv 2505.24111), with the pruning method of
DPHuBERT (arXiv 2305.17651) as used in DiariZen.

1. **Fine-tune** (62k steps, about one pass over the mix). Train `omniASR_CTC_300M` with CTC on the training mix. The result is the teacher. The first release's teacher, fine-tuned for 6k steps, is [omniASR-CTC-300M-km-en](https://huggingface.co/Darayut/omniASR-CTC-300M-km-en).
2. **Prune** (12k steps). Hard Concrete L0 gates sit on the attention heads and FFN units. An augmented-Lagrangian constraint ramps the target sparsity to 80% over the first 6k steps. The pruned model is distilled to the teacher with an L1 − cosine loss on hidden states at layers 6, 12, 18 and 24.
3. **Cut.** Remove the pruned units for good.
4. **Distil** (5k steps). Keep the mask frozen and continue the layer distillation.
5. **Recover** (24k steps). Train with CTC plus a frame-level KL divergence to the teacher's output distribution (temperature 1, weight 1).

**Training mix**, 50/50 by language:
- Khmer: [DDD](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset) train and FLEURS-km train.
- English: LibriSpeech train-clean-100 + 360 + other-500 (961 h) and FLEURS-en train.
- FLEURS is fixed at 20% of each language's audio.

## Usage

```bash
pip install git+https://github.com/netra-ai-lab/Karna
```

```python
import karna

asr = karna.load_model("karna-base")         # downloads this repo's weights once
print(asr.transcribe("audio.wav"))
print(asr.transcribe(["a.wav", "b.flac"]))
```

```bash
karna transcribe audio.wav --model karna-base
```

Without installing the package, the code bundled in this repo also works:

```bash
huggingface-cli download Netra-Lab/karna-base --local-dir karna-base
pip install torch torchaudio soundfile safetensors
python karna-base/inference.py audio.wav
```

Audio of any sample rate is accepted; it is mixed to mono and resampled to 16 kHz. Output text is in the
Omnilingual character set: lowercase, with no Khmer punctuation.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | weights |
| `config.json` | architecture, including per-layer `num_heads` and `ffn_dims` |
| `vocab.json` | the Omnilingual character vocabulary, in id order |
| `omniprune/` | model and text code (`model.py`, `hardconcrete.py`, `text.py`) |
| `inference.py` | load and transcribe |

The released files reproduce the training checkpoint's logits exactly: the maximum difference is 0.

## License and attribution

- **Weights:** derived from Meta's Omnilingual ASR `omniASR_CTC_300M` (Apache 2.0) and released under Apache 2.0.
- **`omniprune/hardconcrete.py`:** ported from DiariZen (MIT), itself derived from DPHuBERT and FLOP.
