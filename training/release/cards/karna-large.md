---
license: apache-2.0
language:
- km
- en
library_name: onnx
pipeline_tag: automatic-speech-recognition
base_model: facebook/omniASR-CTC-1B
datasets:
- Digital-Divide-Data/khmer-speech-dataset
- google/fleurs
- openslr/librispeech_asr
metrics:
- cer
- wer
tags:
- speech-recognition
- khmer
- english
- ctc
- wav2vec2
- omnilingual-asr
- structured-pruning
- onnx
- netra-lab
---

<p align="center">
  <img src="https://huggingface.co/Netra-Lab/karna-large/resolve/main/assets/header.png" width="260" alt="Netra Lab" />
</p>

<hr>

<p align="center">
  <a href="https://github.com/netra-ai-lab/Karna"><b>GitHub</b></a> |
  <a href="https://github.com/netra-ai-lab/Karna/pkgs/container/karna"><b>Docker Image</b></a> |
  <a href="https://pypi.org/project/netra-karna/"><b>PyPI</b></a> |
  <a href="https://huggingface.co/Netra-Lab/karna-base"><b>Karna-base</b></a>
</p>

# karna-large

**Khmer and English speech in, text out.** karna-large is a **217M-parameter** CTC speech recogniser, made by structured pruning of
Meta's Omnilingual ASR `omniASR_CTC_1B` to **about 80% encoder sparsity**. It runs on ONNX Runtime with the `netra-karna` package, on a CPU, without PyTorch.
It is the Large size of a pair, named after DiariZen's Base-s80 / Large-s80 pruned WavLM models; the other size is
[Netra-Lab/karna-base](https://huggingface.co/Netra-Lab/karna-base).

## Results

Test sets; greedy CTC decoding, no language model. CER for Khmer, WER for English, in %. Lower is better.

| Model | Params | GMAC / s audio | DDD (km) | FLEURS-km | FLEURS-en | LibriSpeech test-clean | LibriSpeech test-other |
|---|---|---|---|---|---|---|---|
| `omniASR_CTC_1B` (stock) | 975M | 50.3 | 5.95 | 11.26 | 12.38 | 4.40 | 8.87 |
| Fine-tuned 1B (teacher) | 975M | 50.3 | 1.22 | 7.77 | 12.01 | 3.54 | 7.96 |
| **karna-large** | **217M** | **12.9** | **1.33** | **10.20** | **18.15** | **5.44** | **12.98** |

- **Khmer:** on DDD it is within 0.11 points of its unpruned fine-tuned teacher, and its error is less than a quarter of the stock model's.
- **English:** it keeps much less of its teacher's accuracy, with about 1.5–2× the teacher's WER, and is behind the stock model.
- **Version:** this is v2. The first release (204M; DDD 1.77, test-clean 6.07) is at revision [`593366493e`](https://huggingface.co/Netra-Lab/karna-large/tree/593366493e); v2 improves every test set.
- **In-domain test sets:** FLEURS-km and FLEURS-en train are in the training mix, so their test sets are in-domain.

<p align="center"><img src="https://huggingface.co/Netra-Lab/karna-large/resolve/main/assets/inference_time.png" width="60%" alt="Inference time for 60 s of audio at 0% and 80% encoder sparsity, for the base and large models"></p>

*Inference time* for one 60 s utterance (batch 1, bf16, median of 20 runs), at 0% sparsity (the fine-tuned teacher) and about 80% (Karna).

## Architecture

<p align="center"><img src="https://huggingface.co/Netra-Lab/karna-large/resolve/main/assets/architecture.png" width="45%" alt="Karna's architecture, reading upwards from 16 kHz audio to the CTC head, with one transformer layer opened up"></p>

*The model*, reading upwards, with one transformer layer opened up: 16 kHz audio, a feature extractor of 7 convolutional layers giving one frame every 20 ms, a convolutional position encoder added back to the frames, 48 pre-LayerNorm transformer layers of width 1,280, a final LayerNorm and a CTC head over 9,812 characters. Every attention head and every FFN unit has its own Hard Concrete gate z. The cells in the opened layer are layer 12 of karna-large: 11 of 16 heads and 3,788 of 5,120 FFN units kept; which ones are kept is schematic.

<p align="center"><img src="https://huggingface.co/Netra-Lab/karna-large/resolve/main/assets/pruning.png" width="100%" alt="The fine-tuned omniASR CTC-1B before structured pruning and karna-large after: CNN and CTC head kept, attention heads and FFN units removed"></p>

*What is pruned*, shown for karna-large: whole attention heads (MHA) and FFN units are removed from every transformer layer, and the CNN front end is kept. Each cell is a quarter of a layer's heads or units.

- **Base:** a wav2vec2-style CTC encoder with Omnilingual's 9,812-entry character vocabulary, which covers Khmer and Latin script.
- **Pruned parts:** per-layer attention heads and FFN units, so every layer has its own width. `config.json` lists the exact `num_heads` and `ffn_dims`; the encoder has 205M parameters, 21% of the 1B's 962M.
- **Unchanged parts:** the CNN feature extractor and the output layer.
- **Loading:** the layer widths are non-uniform, so fairseq2 / `omnilingual-asr` cannot load it. Use the `karna` package (ONNX, see Usage) or the PyTorch code bundled in this repo.

## How it was made

The recipe follows *Fine-tune Before Structured Pruning* (arXiv 2505.24111), with the pruning method of
DPHuBERT (arXiv 2305.17651) as used in DiariZen.

1. **Fine-tune** (62k steps, about one pass over the mix). Train `omniASR_CTC_1B` with CTC on the training mix. The result is the teacher. The first release's teacher, fine-tuned for 6k steps, is [omniASR-CTC-1B-km-en](https://huggingface.co/Darayut/omniASR-CTC-1B-km-en).
2. **Prune** (12k steps). Hard Concrete L0 gates sit on the attention heads and FFN units. An augmented-Lagrangian constraint ramps the target sparsity to 80% over the first 6k steps. The pruned model is distilled to the teacher with an L1 − cosine loss on hidden states at layers 12, 24, 36 and 48.
3. **Cut.** Remove the pruned units for good.
4. **Distil** (5k steps). Keep the mask frozen and continue the layer distillation.
5. **Recover** (24k steps). Train with CTC plus a frame-level KL divergence to the teacher's output distribution (temperature 1, weight 1).

**Training mix**, 50/50 by language:
- Khmer: [DDD](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset) train and FLEURS-km train.
- English: LibriSpeech train-clean-100 + 360 + other-500 (961 h) and FLEURS-en train.
- FLEURS is fixed at 20% of each language's audio.

## Try the app

The Docker image bundles both Karna models with a web app and a REST API (`POST /v1/transcribe`), and runs offline. Drop an audio file on the page or record from the microphone.

```bash
docker run -d -p 8000:8000 -v karna-data:/data ghcr.io/netra-ai-lab/karna
# then open http://localhost:8000
```

## Usage

The `karna` package runs the ONNX model in `onnx/` on ONNX Runtime, with no PyTorch:

```bash
pip install "netra-karna[cpu]"         # or [gpu] for CUDA
```

```python
import karna

asr = karna.load_model("karna-large")         # downloads onnx/ from this repo once (871 MB)
print(asr.transcribe("audio.wav"))
print(asr.transcribe(["a.wav", "b.flac"]))
```

```bash
karna transcribe audio.wav --model karna-large
karna serve                              # web app and REST API at http://localhost:8000 ([server] extra)
```

Audio of any sample rate is accepted; it is mixed to mono and resampled to 16 kHz. Output text is in the
Omnilingual character set: lowercase, with no Khmer punctuation.

To fine-tune or prune further, the PyTorch weights at the repo root load with the training package in the
GitHub repo (`training/`, `karna_train.release.load_release`). Without it, the code bundled here also works:

```bash
huggingface-cli download Netra-Lab/karna-large --local-dir karna-large
pip install torch torchaudio soundfile safetensors
python karna-large/inference.py audio.wav
```

## Files

| File | Contents |
|---|---|
| `onnx/model.onnx` | the model as an ONNX graph (fp32, opset 18): normalised 16 kHz audio and lengths in, per-frame token ids (argmax of the CTC logits) and frame lengths out |
| `onnx/vocab.json` | the Omnilingual character vocabulary, in id order |
| `onnx/netra-model.json` | preprocessing, decoding and metadata, and the sha256 of each file |
| `model.safetensors` | PyTorch weights, for fine-tuning |
| `config.json` | architecture, including per-layer `num_heads` and `ffn_dims` |
| `vocab.json` | the same vocabulary |
| `omniprune/` | PyTorch model and text code (`model.py`, `hardconcrete.py`, `text.py`) |
| `inference.py` | load and transcribe with PyTorch |
| `assets/` | the figures on this card |

The PyTorch files reproduce the training checkpoint's logits exactly (maximum difference 0). The ONNX model gives
transcripts identical to PyTorch on 40 of 40 real clips (20 Khmer, 20 English) and 16 of 16 random batches of mixed lengths.

## Learn more

| | |
|---|---|
| [README](https://github.com/netra-ai-lab/Karna#readme) | The project overview: data, architecture, evaluation, install, usage and limitations |
| [Usage](https://github.com/netra-ai-lab/Karna/blob/master/docs/USAGE.md) | The Python API, CLI flags, REST routes, environment variables and devices |
| [Results](https://github.com/netra-ai-lab/Karna/blob/master/docs/RESULTS.md) | Every test and dev number, the ablations, the pruning run step by step |
| [Design](https://github.com/netra-ai-lab/Karna/blob/master/docs/DESIGN.md) | The pruning recipe and each decision behind it |
| [Training](https://github.com/netra-ai-lab/Karna/blob/master/docs/TRAINING.md) | The full recipe, fine-tuning on your own data, evaluation and export |
| [Issues](https://github.com/netra-ai-lab/Karna/issues) | Bug reports and questions |

## License and attribution

- **Weights:** derived from Meta's Omnilingual ASR `omniASR_CTC_1B` (Apache 2.0) and released under Apache 2.0.
- **`omniprune/hardconcrete.py`:** ported from DiariZen (MIT), itself derived from DPHuBERT and FLOP.
- **Training data:** DDD (CC BY-SA 4.0), FLEURS (CC BY 4.0) and LibriSpeech (CC BY 4.0); check their terms before commercial use.
