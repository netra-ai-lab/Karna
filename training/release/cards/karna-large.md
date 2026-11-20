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

# karna-large

A **204M-parameter** Khmer + English CTC speech recogniser, made by structured pruning of
Meta's Omnilingual ASR `omniASR_CTC_1B` to **80% encoder sparsity**. It is the Large size of a pair,
named after DiariZen's Base-s80 / Large-s80 pruned WavLM models. The other size is
[Netra-Lab/karna-base](https://huggingface.co/Netra-Lab/karna-base).

**Code:** [github.com/netra-ai-lab/Karna](https://github.com/netra-ai-lab/Karna), with the `karna` inference package and the full training recipe.

## Results

Test sets; greedy CTC decoding, no language model. CER for Khmer, WER for English, in %. Lower is better.

| Model | Params | GMAC / s audio | DDD (km) | FLEURS-km | FLEURS-en | LibriSpeech test-clean | LibriSpeech test-other |
|---|---|---|---|---|---|---|---|
| `omniASR_CTC_1B` (stock) | 975M | 50.3 | 5.95 | 11.26 | 12.38 | 4.40 | 8.87 |
| [omniASR-CTC-1B-km-en](https://huggingface.co/Darayut/omniASR-CTC-1B-km-en) (fine-tuned teacher) | 975M | 50.3 | 2.47 | 9.44 | 11.94 | 3.61 | 8.21 |
| **karna-large** | **204M** | **12.3** | **1.77** | **11.40** | **19.25** | **6.07** | **16.73** |

- **Khmer:** on DDD it is more accurate than both its unpruned fine-tuned teacher and the stock model.
- **English:** it keeps much less of its teacher's accuracy, with about 1.5–2× the teacher's WER.
- **In-domain test sets:** FLEURS-km and FLEURS-en train are in the training mix, so their test sets are in-domain.

<p align="center"><img src="https://huggingface.co/Netra-Lab/karna-large/resolve/main/assets/inference_time.png" width="60%" alt="Inference time for 60 s of audio at 0% and 80% encoder sparsity, for the base and large models"></p>

*Inference time* for one 60 s utterance (batch 1, bf16, median of 20 runs), at 0% sparsity (the fine-tuned teacher) and 80% (Karna).

## Architecture

<p align="center"><img src="https://huggingface.co/Netra-Lab/karna-large/resolve/main/assets/architecture.png" width="45%" alt="Karna's architecture, reading upwards from 16 kHz audio to the CTC head, with one transformer layer opened up"></p>

*The model*, reading upwards, with one transformer layer opened up: 16 kHz audio, a feature extractor of 7 convolutional layers giving one frame every 20 ms, a convolutional position encoder added back to the frames, 48 pre-LayerNorm transformer layers of width 1,280, a final LayerNorm and a CTC head over 9,812 characters. Every attention head and every FFN unit has its own Hard Concrete gate z. The cells in the opened layer are layer 12 of karna-large: 8 of 16 heads and 3,355 of 5,120 FFN units kept; which ones are kept is schematic.

<p align="center"><img src="https://huggingface.co/Netra-Lab/karna-large/resolve/main/assets/pruning.png" width="100%" alt="The fine-tuned omniASR CTC-1B before structured pruning and karna-large after: CNN and CTC head kept, attention heads and FFN units removed"></p>

*What is pruned*, shown for karna-large: whole attention heads (MHA) and FFN units are removed from every transformer layer, and the CNN front end is kept. Each cell is a quarter of a layer's heads or units.

- **Base:** a wav2vec2-style CTC encoder with Omnilingual's 9,812-entry character vocabulary, which covers Khmer and Latin script.
- **Pruned parts:** per-layer attention heads and FFN units, so every layer has its own width. `config.json` lists the exact `num_heads` and `ffn_dims`; the encoder has 191M parameters.
- **Unchanged parts:** the CNN feature extractor and the output layer.
- **Loading:** the layer widths are non-uniform, so fairseq2 / `omnilingual-asr` cannot load it. Use the `karna` package (see Usage) or the code bundled in this repo.

## How it was made

The recipe follows *Fine-tune Before Structured Pruning* (arXiv 2505.24111), with the pruning method of
DPHuBERT (arXiv 2305.17651) as used in DiariZen.

1. **Fine-tune.** Train `omniASR_CTC_1B` with CTC on the training mix. The result is the teacher, [omniASR-CTC-1B-km-en](https://huggingface.co/Darayut/omniASR-CTC-1B-km-en).
2. **Prune** (12k steps). Hard Concrete L0 gates sit on the attention heads and FFN units. An augmented-Lagrangian constraint ramps the target sparsity to 80%. The pruned model is distilled to the teacher with an L1 − cosine loss on hidden states at layers 12, 24, 36 and 48.
3. **Cut.** Remove the pruned units for good.
4. **Distil** (5k steps). Keep the mask frozen and continue the layer distillation.
5. **Recover** (16k steps). Train with CTC plus a frame-level KL divergence to the teacher's output distribution (temperature 1, weight 1).

**Training mix**, 50/50 by language:
- Khmer: [DDD](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset) train and FLEURS-km train.
- English: LibriSpeech train-clean-100 + 360 and FLEURS-en train.
- FLEURS is fixed at 20% of each language's audio.

## Usage

```bash
pip install git+https://github.com/netra-ai-lab/Karna
```

```python
import karna

asr = karna.load_model("karna-large")         # downloads this repo's weights once
print(asr.transcribe("audio.wav"))
print(asr.transcribe(["a.wav", "b.flac"]))
```

```bash
karna transcribe audio.wav --model karna-large
```

Without installing the package, the code bundled in this repo also works:

```bash
huggingface-cli download Netra-Lab/karna-large --local-dir karna-large
pip install torch torchaudio soundfile safetensors
python karna-large/inference.py audio.wav
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

- **Weights:** derived from Meta's Omnilingual ASR `omniASR_CTC_1B` (Apache 2.0) and released under Apache 2.0.
- **`omniprune/hardconcrete.py`:** ported from DiariZen (MIT), itself derived from DPHuBERT and FLOP.
