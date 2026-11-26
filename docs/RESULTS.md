# Results

Sections 1–7: greedy CTC decoding with no language model. §8 measures inference away from the test sets, including beam search with a language model. CER is used for Khmer and WER for English, all in %, and lower is better. Compute is GMAC per second of audio. Every number comes from `training/runs/eval/*.json` and `training/runs/*/*/metrics.jsonl`.

## 1. Test sets

The released models are v2 ([DESIGN.md](DESIGN.md) §4): a full-pass Stage 1, 961 h of LibriSpeech, a slower sparsity ramp and a 24k-step Stage 3. The teachers are the v2 Stage 1 models.

| Model | Params | GMAC/s | DDD | FLEURS-km | FLEURS-en | test-clean | test-other |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| omniASR CTC-1B (stock) | 975M | 50.3 | 5.95 | 11.26 | 12.38 | 4.40 | 8.87 |
| Fine-tuned 1B (teacher) | 975M | 50.3 | 1.22 | 7.77 | 12.01 | 3.54 | 7.96 |
| **karna-large** | **217M** | **12.9** | **1.33** | **10.20** | **18.15** | **5.44** | **12.98** |
| omniASR CTC-300M (stock) | 325M | 18.3 | 9.65 | 16.26 | – | 6.53 | 14.45 |
| Fine-tuned 300M (teacher) | 325M | 18.3 | 1.47 | 9.95 | 18.78 | 5.57 | 13.20 |
| **karna-base** | **71M** | **5.8** | **1.94** | **15.20** | **29.66** | **10.96** | **23.53** |

**Test sets:**

| Set | Utterances | Notes |
| :--- | ---: | :--- |
| DDD | 1,993 | held out from DDD training |
| FLEURS-km test | 771 | |
| FLEURS-en test | 647 | |
| LibriSpeech test-clean | 2,620 | |
| LibriSpeech test-other | 2,939 | |

FLEURS train is in the training mix, so the two FLEURS test sets are in-domain. The stock 300M was not scored on FLEURS-en.

## 2. First release against v2

| Model | Params | GMAC/s | DDD | FLEURS-km | FLEURS-en | test-clean | test-other |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fine-tuned 1B, v1 (6k steps) | 975M | 50.3 | 2.47 | 9.44 | 11.94 | 3.61 | 8.21 |
| Fine-tuned 1B, v2 (62k steps) | 975M | 50.3 | 1.22 | 7.77 | 12.01 | 3.54 | 7.96 |
| karna-large v1 | 204M | 12.3 | 1.77 | 11.40 | 19.25 | 6.07 | 16.73 |
| **karna-large v2** | **217M** | **12.9** | **1.33** | **10.20** | **18.15** | **5.44** | **12.98** |
| Fine-tuned 300M, v1 (6k steps) | 325M | 18.3 | 3.68 | 12.64 | 19.51 | 5.89 | 13.86 |
| Fine-tuned 300M, v2 (62k steps) | 325M | 18.3 | 1.47 | 9.95 | 18.78 | 5.57 | 13.20 |
| karna-base v1 | 73M | 5.8 | 2.60 | 16.39 | 30.24 | 11.15 | 27.36 |
| **karna-base v2** | **71M** | **5.8** | **1.94** | **15.20** | **29.66** | **10.96** | **23.53** |

- **Every set improves** for both sizes. The largest gains are on DDD and on test-other.
- **The teacher is the main change.** A full pass over the data halves the teachers' DDD error. v1's pruned models beat their under-trained teachers on DDD; v2's stay within half a point of much stronger ones (0.11 for large, 0.47 for base).
- **English stays the weak side,** at 1.5–2× the teacher's WER, behind the stock models.
- **Size.** karna-large v2 ends at 78.7% encoder sparsity (217M), against 80.2% (204M) for v1; see §4.
- **The v1 weights** stay in the history of the Hugging Face repos.

## 3. Ablations on the 1B

| Run | What changes | DDD | FLEURS-km | FLEURS-en | test-clean | test-other |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| A | prune the stock 1B, no Stage 1, no FLEURS | 2.20 | 16.36 | 28.07 | 6.56 | 18.26 |
| B1 | fine-tune first, Stage 3 CTC only (8k) | 2.14 | 12.51 | 21.16 | 7.07 | 18.51 |
| B2 | + logit KD in Stage 3 (8k) | 2.09 | 11.90 | 19.62 | 6.47 | 17.42 |
| B3 (karna-large v1) | + Stage 3 for 16k steps | 1.77 | 11.40 | 19.25 | 6.07 | 16.73 |
| **v2** (karna-large) | + 62k-step Stage 1, 961 h LibriSpeech, ramp over 50% of Stage 2, Stage 3 for 24k steps | **1.33** | **10.20** | **18.15** | **5.44** | **12.98** |

A → B changes two things at once: the teacher (stock vs fine-tuned) and the data (FLEURS added). B3 → v2 changes four at once. Neither set of effects is separated.

## 4. Stage 2: the pruning run

Dev sets:
- DDD dev and LibriSpeech dev-clean, the first 400 utterances each;
- FLEURS-km dev, 326 utterances;
- FLEURS-en dev, 394 utterances.

**v2** (target ramps over the first 6,000 steps):

| Step | Large sparsity | Large DDD | Large Libri | Base sparsity | Base DDD | Base Libri |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 26.3% | 4.30 | 8.94 | 26.2% | 4.41 | 12.26 |
| 4,000 | 52.6% | 5.46 | 10.60 | 52.8% | 6.04 | 13.82 |
| 6,000 | 78.8% | 9.78 | 21.26 | 77.0% | 13.85 | 32.07 |
| 8,000 | 82.2% | 9.24 | 21.08 | 83.6% | 14.99 | 33.56 |
| 10,000 | 80.5% | 6.77 | 15.00 | 78.0% | 8.73 | 22.18 |
| 12,000 | 78.7% | 5.96 | 13.52 | 80.6% | 9.21 | 22.23 |

**v1** (target ramps over the first 3,000 steps):

| Step | Large sparsity | Large DDD | Large Libri | Base sparsity | Base DDD | Base Libri |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 51.8% | 7.87 | 12.37 | 51.8% | 9.80 | 18.38 |
| 4,000 | 83.1% | 11.95 | 30.32 | 83.0% | 20.03 | 44.97 |
| 6,000 | 73.2% | 7.53 | 14.18 | 76.9% | 13.42 | 27.42 |
| 8,000 | 80.7% | 7.81 | 16.27 | 80.6% | 12.14 | 24.38 |
| 10,000 | 80.3% | 6.85 | 12.58 | 80.1% | 11.43 | 22.23 |
| 12,000 | 80.2% | 7.06 | 12.97 | 80.1% | 9.98 | 20.01 |

- **Constraint behaviour.** In both versions the Lagrangian overshoots the target once it is reached, then oscillates around 80%. The slower v2 ramp moved the overshoot later (step 8,000) but did not remove it, and karna-large v2 was cut at 78.7%.
- **Experiment A** (stock 1B as init and teacher) ended Stage 2 at 10.71% DDD dev CER, against 7.06% when pruning from the v1 fine-tuned model.

## 5. Dev error through the chain

**v2:**

| Point | Large DDD | Large Libri | Large FLEURS-km | Large FLEURS-en | Base DDD | Base Libri | Base FLEURS-km | Base FLEURS-en |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock | 7.94 | 4.35 | 10.69 | 14.47 | 10.98 | 6.96 | 15.61 | 21.47 |
| After Stage 1 (teacher) | 1.65 | 3.53 | 7.25 | 13.84 | 1.94 | 5.93 | 9.55 | 20.29 |
| After Stage 2 (pruned) | 5.96 | 13.52 | 16.49 | 30.86 | 9.21 | 22.23 | 25.04 | 45.85 |
| After Stage 2b | 4.21 | 7.76 | 12.39 | 23.41 | 7.05 | 15.74 | 20.86 | 37.21 |
| After Stage 3 (released) | 1.96 | 5.69 | 9.76 | 19.64 | 2.70 | 11.47 | 14.90 | 31.91 |

**v1:**

| Point | Large DDD | Large Libri | Large FLEURS-km | Large FLEURS-en | Base DDD | Base Libri | Base FLEURS-km | Base FLEURS-en |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock | 7.94 | 4.35 | 10.69 | 14.47 | 10.98 | 6.96 | 15.61 | 21.47 |
| After Stage 1 (teacher) | 3.40 | 3.95 | 8.83 | 13.68 | 4.64 | 5.98 | 12.33 | 20.86 |
| After Stage 2 (pruned) | 7.06 | 12.97 | 18.29 | 33.96 | 9.98 | 20.01 | 25.99 | 44.01 |
| After Stage 2b | 5.60 | 8.46 | 14.64 | 25.53 | 8.61 | 15.49 | 23.42 | 37.91 |
| After Stage 3 (released) | 2.40 | 6.60 | 10.71 | 21.15 | 3.47 | 11.73 | 15.88 | 32.15 |

- **Recovery.** Stages 2b and 3 recover most of what pruning cost. On DDD dev, v2 ends 0.3 (large) and 0.8 (base) points above its teacher; v1 ended below its weaker teacher.
- **English.** It recovers less: the released models' English error is roughly 1.5–2× their teachers'.

## 6. Training data

| Source | Language | Train size | Share of training audio |
| :--- | :--- | :--- | :--- |
| [DDD](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset) | Khmer | 1,058 h, 4,423 parquet shards | 80% of Khmer |
| FLEURS km_kh train | Khmer | 7.1 h, 1,675 clips | 20% of Khmer (repeated) |
| LibriSpeech train-clean-100 + 360 + other-500 | English | 961 h (v1: 464 h, clean only) | 80% of English |
| FLEURS en_us train | English | 7.5 h, 2,602 clips | 20% of English (repeated) |

The two languages are mixed 50/50 by audio seconds.

## 7. karna-base v2, first attempt: output distillation (dropped)

The first v2 chain used two teachers:
- the 300M fine-tune for hidden states;
- the fine-tuned 1B for outputs, distilled in every stage (Stage 1 included);

and in Stage 2 it also distilled 13 layers instead of 5. Stage 1 ran 6k steps.

Its Stage 1 teacher reached 4.61 / 6.56 / 12.41 / 21.42 on dev (DDD / Libri / FLEURS-km / FLEURS-en). That is about the same as v1 on Khmer and worse on English (v1: 4.64 / 5.98 / 12.33 / 20.86). In Stage 2, English collapsed early: Libri dev WER reached 52.8% at step 500 and 60.2% at step 2,000. The run was stopped and replaced by these short tests, all started from that Stage 1 teacher and scored on the dev sets:

| Test | Change | Step | DDD | Libri | FLEURS-km | FLEURS-en |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| v1 Stage 2 | reference (v1 teacher) | 500 | 7.70 | 11.17 | 18.14 | 29.94 |
| first v2 | 1B output KD, 13 layers, 960 h | 500 | 13.04 | 52.80 | 26.25 | 68.21 |
| B | same, but 464 h English | 500 | 11.62 | 23.75 | 23.72 | 43.01 |
| C | output KD from the 300M itself | 500 | 14.02 | 19.43 | 26.77 | 42.12 |
| A | no output KD, 13 layers | 500 | 8.95 | 13.96 | 19.37 | 32.88 |
| E | no output KD, 5 layers | 500 | **7.21** | 14.12 | **17.51** | 33.03 |
| v1 Stage 3 | reference (300M teacher) | 1,000 | **6.44** | **15.21** | **21.65** | **38.19** |
| D | Stage 3 with the 1B as teacher (kd 2, 960 h), from v1's Stage 2b | 1,000 | 7.02 | 19.44 | 22.17 | 41.49 |

- **Output KD hurts Stage 2 whatever the teacher.** Compare A with C and with the first v2.
- **The new data makes the 1B's KD worse,** but is harmless without it. Compare B with the first v2, and A with v1.
- **13 distillation layers do not beat 5** (A vs E).
- **The 1B is a worse recovery teacher than the 300M,** on Khmer as well, where the 1B is trained (D).

**Spike timing.** On LibriSpeech dev, only 45% of the v1 300M teacher's non-blank frames coincide with the 1B's (KL 2.7 nats per frame). After the first v2 Stage 1 with 1B KD, 78% coincide (KL 0.24).

## 8. Real-world proxies (inference only, 2026-11-26)

Sections 1–7 are in-domain: read speech, recorded clean, from the training sources. Microphone use is worse, and live use worse still. No real recordings with references were available, so `scripts/realworld_eval.py` builds proxies, all fixed by a seed:
- **Natural sets:** other recordings than the training data. These are OpenSLR 42 (crowd-recorded Khmer, split into halves: tune on `slr42_dev`, report `slr42_test`), 600 clips of kmcorpus (spontaneous, colloquial Khmer with code-switching) and Common Voice Khmer test (12 clips).
- **Simulated sets:** FLEURS-km test and LibriSpeech test-other (200 clips each) under five conditions:
  - *noise:* MUSAN / ESC-50 background at 5–15 dB SNR;
  - *reverb:* a simulated room and a 25 dB background;
  - *far:* a room and 5–10 dB, like a laptop across the table;
  - *phone:* 8 kHz mu-law with a 300 Hz high-pass and 15 dB.
- **Live:** 40 utterances of a simulated set, joined by 0.3–2 s gaps of the same background, fed in 100 ms chunks through `Karna.stream()`.

Settings were tuned only on dev sets: `slr42_dev`, `kmcorpus_dev` (400 other kmcorpus clips), FLEURS-km dev and LibriSpeech dev-clean under clean, noise and far. Every number below is on the test proxies, on CUDA (RTX 3060). On the CPU, karna-base matched CUDA to 0.1 points on the sets checked (kmcorpus, cv_km, FLEURS-km clean); the full CPU run was not finished. JSON is in `training/runs/eval/realworld/`; tables come from `scripts/realworld_report.py`.

### Beam search with a character language model

The new `decoder="beam"` is CTC prefix beam search, beam 8, with shallow fusion of a character 8-gram (interpolated modified Kneser-Ney, pruned, pure numpy). The LM was built by `scripts/build_char_lm.py` from text on the build machine (124 M Khmer and 52 M English characters): the Khmer web corpus, the ASR text corpus, the Khmer side of the MT corpus and the DDD training transcripts, plus LibriSpeech-960 and the English side of the MT corpus.

The build removed every line sharing a 20-character span with an evaluation sentence, which dropped 22 k Khmer MT lines (FLEURS sentences come from FLoRes, which the MT corpus contains). The result is 65 MB, at 2.06 bits per character on FLEURS-km dev and 1.88 on LibriSpeech dev-clean.

| Set | | karna-base greedy | karna-base beam + LM | karna-large greedy | karna-large beam + LM |
| :--- | :--- | ---: | ---: | ---: | ---: |
| slr42_test | CER | 11.3 | **8.5** | 9.1 | **8.2** |
| kmcorpus | CER | 39.7 | **38.7** | 37.0 | **36.9** |
| cv_km (12 clips) | CER | 16.3 | **7.7** | **3.3** | 3.6 |
| fleurs_km clean | CER | 15.5 | **13.0** | 10.5 | **9.2** |
| fleurs_km noise | CER | 29.7 | **25.6** | 19.1 | **17.2** |
| fleurs_km reverb | CER | 41.6 | **37.6** | 29.0 | **27.0** |
| fleurs_km far | CER | 60.7 | **56.1** | 47.6 | **46.1** |
| fleurs_km phone | CER | 29.8 | **25.6** | 19.0 | **17.3** |
| libri_other clean | WER | 23.3 | **17.4** | 13.3 | **11.0** |
| libri_other noise | WER | 40.9 | **33.0** | 24.9 | **20.6** |
| libri_other reverb | WER | 47.3 | **39.4** | 30.6 | **26.2** |
| libri_other far | WER | 65.1 | **58.1** | 49.8 | **44.7** |
| libri_other phone | WER | 44.6 | **36.7** | 26.1 | **21.8** |

- **Settings** (`lm_weight`, Khmer insertion bonus, ASCII and space bonus): karna-base 0.5 / 2.5 / 0.5, karna-large 0.5 / 1.0 / 0.5. Khmer takes a larger bonus because it is written with more characters per second of speech.
  - The bonus is the risky knob. With LM weight 0.5, bonus 2.0 took karna-large's kmcorpus CER from 37.0 to 42.0, through insertions in long, uncertain stretches of spontaneous speech. The read-speech dev sets never showed this.
  - With `kmcorpus_dev` added, the rule became the lowest mean error among settings no worse than greedy on any dev set. That rule picked bonus 1.0 for karna-large.
- **Cost:** 3–8 ms of one CPU core per second of audio, so about 0.2–0.5 s per minute against 8.7 s (karna-base) for the graph on the reference laptop. Beam search reads the graph's new `sub_logp` output, a log-softmax over the blank and the 127 writable characters, so the 9,812-wide scores still never leave the session. Its effect on graph speed was not measured separately.
- **Unchanged:** greedy text, and `decoder="auto"` without an LM, so all golden transcripts are identical.

### Live transcription

The live finals use beam search; partials stay greedy so they stay fast. The energy endpointer's `ratio` went from 4 to 3: a frame is speech when it is 9.5 dB above the noise floor, not 12 dB. All other endpointer settings are unchanged.

| Live set | | karna-base before | karna-base now | karna-large before | karna-large now |
| :--- | :--- | ---: | ---: | ---: | ---: |
| fleurs_km clean | CER | 16.3 | **14.1** | 10.7 | **9.4** |
| fleurs_km noise | CER | 33.5 | **29.9** | 23.9 | **22.3** |
| fleurs_km far | CER | 65.2 | **53.6** | 57.5 | **45.8** |
| libri_other clean | WER | 24.5 | **18.6** | 13.1 | **10.3** |
| libri_other noise | WER | 35.3 | **26.2** | 21.4 | **14.6** |

How the two changes add up, live, for karna-large (greedy, ratio 4 → greedy, ratio 3 → beam, ratio 3):

| | fleurs_km clean | noise | far | libri_other clean | noise |
| :--- | ---: | ---: | ---: | ---: | ---: |
| greedy, ratio 4 | 10.7 | 23.9 | 57.5 | 13.1 | 21.4 |
| greedy, ratio 3 | 10.4 | 24.1 | 47.5 | 12.9 | 18.2 |
| beam + LM, ratio 3 | **9.4** | **22.3** | **45.8** | **10.3** | **14.6** |

- **Ratio 3 fixes far speech.**
  - At ratio 4, 35–38 % of the far-field speech was in no segment, or in one whose final came back empty. At ratio 3 it is 5 %.
  - Mid-utterance cuts dropped from 69 to 53 (far), and from 35 to 21 for noisy English.
  - Clean streams are unchanged: 44 cuts became 43, and 10 % of the speech still has no text.
  - An offline sweep of the segmentation alone: speech missed in far went from 17.0 % (ratio 4) to 5.1 % (3) and 1.8 % (2.5), and in noisy English from 8.6 % to 1.9 % (3). Ratio 2.5 was not run live.
- **Decode cost** went from 0.074–0.088 to 0.079–0.090 s per second of audio for karna-large (0.043–0.055 → 0.048–0.054 for karna-base). That is more segments with text, plus the beam.
- **Drafts change a little more:** a partial's CER against its final rose by up to 5 points (mostly 1–2; one set fell by 2), because drafts are greedy and finals use the beam.

### Tried and not kept

| | Result |
| :--- | :--- |
| Silero VAD v5 instead of the energy rule | Kept only as an option, `vad="silero"`. It tells speech from noise, so it covered much less of the noisy gaps than the energy rule (noise: 0 % of gaps against 64 %; far: 0 % against 92 %). But it scored much Khmer speech low: it missed 21 % of clean FLEURS-km speech against 10 % for energy, 26 % against 3 % in noise and 44 % against 17 % far. Thresholds of 0.3–0.5 all missed 21–44 %, and lower thresholds never released. It also isn't level-invariant: without gain control it missed quiet utterances after loud ones, and live CER on clean FLEURS-km was 34.8 against 16.3. A 1 s automatic gain stage in front of it (`karna/vad.py`) is kept for the option. |
| A smaller LM (half the text quota, 42 MB, 2.11 bits per character) | Slightly worse on the dev sets than the full one. A pruned variant (more count pruning, 49 MB) was worse again (2.16 bits per character). |
| Beam 16 | karna-base dev: 0–3 % relative better than beam 8, at 1.5× the cost (15.5 against 10.5 ms per second of audio). |
| ORT's offline transformer fusion (`onnxruntime.transformers.optimizer`) | It finds only Gelu, LayerNorm and SkipLayerNorm, which `ORT_ENABLE_ALL` already fuses at load. karna-base CPU, batch 4 × 10 s: 1.38 s against 1.42 s, inside the run-to-run spread (1.29–1.89 s). |
| fp16 | Not tried. GPU fp32 already runs at RTF 0.005–0.009 (below); on CPU, fp16 storage would only shrink the download. |

### Speed on a GPU

Golden clips, RTX 3060 12 GB, onnxruntime-gpu 1.32, `scripts/benchmark.py --device cuda --repeat 3`: karna-base 0.31 s per minute of audio (RTF 0.005), karna-large 0.54 s (RTF 0.009); model load 0.5 s and 1.1 s. The CPU parity check also passes on CUDA: the golden transcripts are 4/4 identical for both models.
