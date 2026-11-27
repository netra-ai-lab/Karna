---
license: cc-by-sa-4.0
language:
- km
- en
library_name: karna
tags:
- language-model
- n-gram
- character-level
- kneser-ney
- ctc-beam-search
- speech-recognition
- khmer
- english
- netra-lab
---

<p align="center">
  <img src="https://huggingface.co/Netra-Lab/karna-lm/resolve/main/assets/header.png" width="260" alt="Netra Lab" />
</p>

<hr>

<p align="center">
  <a href="https://github.com/netra-ai-lab/Karna"><b>GitHub</b></a> |
  <a href="https://pypi.org/project/netra-karna/"><b>PyPI</b></a> |
  <a href="https://huggingface.co/Netra-Lab/karna-base"><b>Karna-base</b></a> |
  <a href="https://huggingface.co/Netra-Lab/karna-large"><b>Karna-large</b></a>
</p>

# karna-lm

**A character language model of Khmer and English, for beam-search decoding of the Karna speech recognisers.** It is an interpolated modified Kneser-Ney character 8-gram, stored as numpy arrays (65 MB) and read by the `netra-karna` package with no other dependency. Fused into CTC prefix beam search, it lowers the error of [karna-base](https://huggingface.co/Netra-Lab/karna-base) and [karna-large](https://huggingface.co/Netra-Lab/karna-large) on read, noisy, far-field and live speech.

## Results

CER for Khmer, WER for English, in %, lower is better; greedy decoding against beam search with this model. The sets are real-world proxies (`scripts/realworld_eval.py` in the GitHub repo): other recordings than the training data, and FLEURS-km / LibriSpeech test-other with background noise, reverberation, a far-field room, or a telephone channel. The beam settings were tuned on separate dev sets only.

| Set | karna-base greedy | karna-base + LM | karna-large greedy | karna-large + LM |
|---|---:|---:|---:|---:|
| OpenSLR 42 Khmer (test half), CER | 11.3 | **8.5** | 9.1 | **8.2** |
| kmcorpus (spontaneous Khmer), CER | 39.7 | **38.7** | 37.0 | **36.9** |
| FLEURS-km, CER | 15.5 | **13.0** | 10.5 | **9.2** |
| FLEURS-km + noise, CER | 29.7 | **25.6** | 19.1 | **17.2** |
| FLEURS-km far-field, CER | 60.7 | **56.1** | 47.6 | **46.1** |
| LibriSpeech test-other, WER | 23.3 | **17.4** | 13.3 | **11.0** |
| test-other + noise, WER | 40.9 | **33.0** | 24.9 | **20.6** |
| test-other telephone, WER | 44.6 | **36.7** | 26.1 | **21.8** |

- **Live transcription** (finals decoded with the beam, together with a more sensitive endpointer in `netra-karna` 1.2): karna-large on streamed FLEURS-km went from 10.7 to 9.4 CER (far-field 57.5 to 45.8), and on noisy English from 21.4 to 14.6 WER.
- **Cost:** 3–8 ms of one CPU core per second of audio, against about 145 ms for karna-base's network on a laptop CPU.
- **Where it doesn't help:** Common Voice Khmer (12 clips) went from 3.3 to 3.6 CER with karna-large.
- **Perplexity:** 2.06 bits per character on FLEURS-km dev, 1.99 on DDD dev, 2.80 on OpenSLR 42, 2.48 on kmcorpus, 1.88 on LibriSpeech dev-clean and 2.04 on FLEURS-en dev.

## Usage

`netra-karna` 1.3 and later download this model on first load and use it for every transcription; `KARNA_LM=none` turns it off.

```bash
pip install "netra-karna[cpu]"
```

```python
import karna

asr = karna.load_model("karna-large")          # also fetches karna-lm once (65 MB)
asr.transcribe("audio.wav")                    # beam search with the LM
asr.transcribe("audio.wav", decoder="greedy")  # without it
```

With `netra-karna` 1.2, download it and point `KARNA_LM` at it:

```bash
huggingface-cli download Netra-Lab/karna-lm --local-dir karna-lm
KARNA_LM=karna-lm/karna-char8.npz karna transcribe audio.wav --model karna-large
```

The beam settings tuned for each Karna model are in `karna-char8.json` (`"beam"`) and are read from there.

## How it was made

`scripts/build_char_lm.py` in the GitHub repo:

1. **Text:** 124 M Khmer and 52 M English characters, sampled to a quota per source and deduplicated (sources below).
2. **Normalisation:** the characters the Karna models can write, lowercased; one byte per character.
3. **Held-out text removed:** every line sharing a 20-character span with an evaluation sentence (FLEURS, DDD dev and test, OpenSLR 42, kmcorpus, Common Voice, LibriSpeech dev and test, the golden test clips). This dropped 22 k lines of the parallel corpus, which contains FLoRes sentences.
4. **Counts:** an 8-gram with interpolated modified Kneser-Ney smoothing; n-grams of orders 4–8 seen at most 1–4 times are pruned. 13.9 M n-grams are kept. Built in 134 s.

| Source | Language | Characters used |
|---|---|---:|
| Khmer web-text corpus | Khmer | 70.0 M |
| Khmer ASR text corpus | Khmer | 23.6 M |
| Khmer side of a Khmer–English parallel corpus | Khmer | 20.0 M |
| [DDD Khmer speech dataset](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset) train transcripts | Khmer | 9.1 M |
| [LibriSpeech](https://www.openslr.org/12) 960 h transcripts | English | 35.0 M |
| English side of the parallel corpus | English | 16.2 M |

## Files

| File | Contents |
|---|---|
| `karna-char8.npz` | the model: per order, sorted uint64 n-gram keys with float16 log-probabilities and backoff weights, and a JSON header |
| `karna-char8.json` | the build manifest (sources, counts, discounts, perplexities, sha256) and the tuned beam settings per Karna model |
| `assets/` | the figure on this card |

## Learn more

| | |
|---|---|
| [Usage](https://github.com/netra-ai-lab/Karna/blob/master/docs/USAGE.md#decoding) | Decoding options, `KARNA_LM`, building and tuning your own model |
| [Results](https://github.com/netra-ai-lab/Karna/blob/master/docs/RESULTS.md) | The real-world proxies in full (§8) |
| [Issues](https://github.com/netra-ai-lab/Karna/issues) | Bug reports and questions |

## License

CC BY-SA 4.0. The model is derived from text including the DDD transcripts (CC BY-SA 4.0, whose ShareAlike terms carry over) and LibriSpeech (CC BY 4.0).
