# How the pipeline works

[← back to the README](../README.md)

What runs, in what order, what feeds what, and which parts are trained. If you
only want the commands, the [README quickstart](../README.md#quickstart) has
them; this is the map behind them.

## The one fact that explains everything else

openWakeWord trains **only a small DNN head on a frozen front end.** Everything
upstream of the head is pretrained, fixed, and never learns:

```
audio 16 kHz ─► melspectrogram.onnx ─► embedding_model.onnx ─► 16×96 ─► DNN head ─► score
               └────────────── frozen, downloaded by setup_env.sh ────┘   └ the only
                                                                            trained part ┘
```

`16×96` is 16 frames of a 96-dimensional embedding (`input_shape=(16, 96)` in
`train.py`), roughly two seconds of audio. Two consequences worth internalising:

- **Training is cheap; feature extraction is not.** The head is ~100k parameters.
  Nearly all wall clock goes into turning audio into 16×96 features.
- **Anything the embedding discards is unrecoverable by training.** This is the
  whole subject of [the front-end probe][exp], which found the embedding is the
  leading suspect for this project's loud-music failures.

[exp]: https://github.com/Dethon/openwakeword-training/tree/experiments

## One-time setup

**`setup_env.sh`** creates `.venv`, clones the two upstream repos, and installs
dependencies in a specific order that matters: torch from the cu121 index
*first*, so the CUDA build wins over any transitive CPU pin, then
`onnxruntime-gpu` (never `onnxruntime` — they provide the same module). It then
downloads the two frozen extractors and the Piper TTS voice.

The clones are gitignored and both need local changes; see
[`patches/README.md`](../patches/README.md).

**`02_download_data.py`** fetches ~33 GB across four independent corpora:

| | feeds |
|---|---|
| `mit_rirs/` | room impulse responses, used in augmentation |
| `audioset_16k/`, `fma/` | background noise and music, used in augmentation |
| `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | 17 GB of **precomputed** negative features |
| `validation_set_features.npy` | held-out negatives for measuring false-positive rate while training |

The last two are already in embedding space. They go straight into the training
loop and never pass through augmentation or feature extraction — which is why
they are `.npy` rather than audio.

`downloads/` holds the raw archives and is deletable once this finishes.

**`03_make_config.py`** reads openWakeWord's example config, applies
`--word/--examples/--steps/--penalty`, and writes `my_model.yaml`. Every later
stage reads that YAML, selected by the `CONFIG` environment variable.

## Training: three stages, three processes

`04_train.sh` runs each stage as a separate process so each frees its GPU memory
before the next starts. Any stage can be run alone (`./04_train.sh augment`), and
all of them resume.

### ① `clips` — synthesise audio

```
piper TTS ────────────────────► positive_train/   (n_samples,     e.g. 50,000)
                                positive_test/    (n_samples_val, e.g.  5,000)
generate_adversarial_texts ───► negative_train/, negative_test/
```

The negatives are not random speech: `generate_adversarial_texts` builds phrases
with **phoneme overlap** against your target phrase, so the head learns to reject
near-misses rather than merely "not-the-wake-word".

`verify_clips.py` runs first and prunes clips left empty by an interrupted run —
`train.py` sizes each generation batch by counting files on disk, so a zero-byte
clip would otherwise be counted as done and break augmentation later.

### ② `augment` — the expensive one

Every clip is augmented `augmentation_rounds` times, then turned into features:

```
for each clip × augmentation_rounds:
    audiomentations:        SevenBandParametricEQ, TanhDistortion
    torch-audiomentations:  PitchShift, BandStopFilter, AddColoredNoise,
                            AddBackgroundNoise (at a drawn SNR), Gain
    applied directly:       RIR convolution with a randomly chosen impulse
                            response (not a torch-audiomentations transform)
        │
        ▼
    melspectrogram.onnx ─► embedding_model.onnx
        │
        ▼
    positive_features_train.npy   negative_features_train.npy
    positive_features_test.npy    negative_features_test.npy
```

Clips are padded to `total_length`, which `train.py` **computes** rather than
reads from the config: it samples 50 generated clips, takes the median duration
rounded to the nearest 1,000 samples, adds 12,000 samples (750 ms) of buffer,
then floors the result at 32,000 samples (2 s) and snaps to 32,000 if it lands
within 4,000 of it. Clips longer than `total_length` are truncated, which is why
`trim_clips.py --training` reports how many would be.

`background_mode: per_example` makes every clip draw its own background and SNR;
`per_batch` draws once per batch. That choice, and the SNR bounds, are the main
levers in [the v6 recipe][exp].

This stage is where nearly all the wall clock goes — augmentation *and* feature
extraction for every clip. Measured on the v7 workload: 268,480 clip-augmentations
in 70.7 minutes, **63 clip-augmentations/s** with GPU feature extraction active.

### ③ `train` — no audio at all

```
mmap_batch_generator draws a FIXED NUMBER OF ROWS PER CLASS per step:
    ACAV100M_sample 1024 │ adversarial_negative 50 │ positive 50 │ media_negatives 128
        │
        ▼
    DNN head (layer_size, n_blocks)
        │
        ├─ negative weight ramps linearly 1 → max_negative_weight across the run
        ├─ validates false-positive rate against validation_set_features.npy
        └─► .onnx
```

`batch_n_per_class` is the composition knob: it fixes the class mix per step
rather than sampling from a pooled set, so adding a new negative class does not
dilute the others unless you say so. The linear negative-weight ramp is how the
false-activation penalty is applied — early steps learn to detect, later steps
learn to shut up.

This stage only shuffles precomputed feature rows, so 30,000 steps takes minutes.

## Injecting real recordings

Real recordings enter **between ① and ②**, so augmentation treats them as ordinary
positives. It is why v7's positive set is 74,240 files rather than 50,000:

```
① clips ──► 50,000 synthetic
              + trim_clips.py --training ─► duplicate ×202 ─► REAL_*        21,412
              + trim_clips.py --training ─► duplicate ×202 ─► REAL_MUSIC_*   2,828
            ──────────────────────────────────────────────────────────────  74,240
② augment
```

The ×202 duplication exists so each copy draws its **own** random background and
RIR in stage ②; with `augmentation_rounds: 2` that is 404 distinct variants per
real recording.

`04_train.sh` performs this automatically for any config declaring
`real_clip_sources` (`tools/data/inject_real_clips.py`), and asserts
`expect_positive_train_total` before augmentation starts. A config without the key
is unaffected. It used to be an unscripted manual step, which is why v6, v7 and v8
could not be rebuilt from their own configs — running `04_train.sh` against them
produced a 50,000-positive model with no error, and no warning that it was not
the model the config named.

## Export

`05_export_tflite.sh` converts `.onnx` → `.tflite` (plus a float16 variant) using
`onnx2tf` in a **separate venv**, because its TensorFlow dependency conflicts with
the `numpy<2` pin the training environment needs. `train.py` also ends by calling
its own converter, which imports the long-dead `onnx_tf` and fails — that happens
*after* the `.onnx` is written, and `04_train.sh` recognises that specific failure
and continues.

## Evaluation is a separate pipeline

Nothing above measures quality. That is `tools/eval/`, running against the
`.tflite`:

```
real recordings ($OWW_DATA_ROOT) ─┐
eval_media/{music,tv}             ─┼─► embed (cached) ─► head ─► peak score per file
synthetic SNR sweep               ─┘                            ──► results *.json
```

`eval_music.py --out` picks where the JSON lands. The reports and raw scores from
this project's own runs are on the [`experiments` branch][exp].

It reuses the same frozen front end, with `embed_cache.py` persisting embeddings
across runs — which is why re-scoring the 24 h false-activation corpus takes 87 s
warm against 284 s before.

At runtime the deployed model adds two more knobs that are not part of training:
a score `threshold` and a `trigger_level` (how many consecutive hot frames are
required). Both were tuned against a 24 h false-activation corpus; that analysis
is on the [`experiments` branch][exp].

## Reproducibility

**Seeding** is applied immediately after the config is loaded, before any stage
branches, so `seed: <int>` covers whichever stage the process runs. It is
**opt-in**: the default is unseeded, matching how every model in this repo was
actually trained.

Measured, not assumed: a seed pins stage ② byte-for-byte. It does **not**
guarantee bit-identical stage ③ on GPU, because cuDNN/cuBLAS kernel selection can
still vary. Both halves of that were measured; see the reproduction write-up on
the [`experiments` branch][exp].

**Resumability.** Every stage skips work already done. Stage ② used to gate on a
single feature file and would silently skip with three of four missing; that is
fixed by `patches/openwakeword-augment-resume-gate.patch`.

## Wall clock, on an RTX 4090

| stage | time |
|---|---|
| ① clips (50,000 + adversarial) | ~25 min |
| ② augment (268,480 clip-augmentations) | ~70 min |
| ③ train (30,000 steps) | a few min |
| export | a few min |

Plus ~1 h once for `02_download_data.py`.
