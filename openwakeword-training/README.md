# openwakeword-training

A local, reproducible pipeline for training [openWakeWord](https://github.com/dscripka/openWakeWord)
custom wake-word models — a port of the upstream
[automatic training Colab notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb)
that runs on your own GPU, plus an evaluation harness for measuring what you got.

The upstream notebook no longer runs as written: AudioSet moved, FMA's loader is
dead, the built-in tflite converter imports a package that no longer exists, and
`augmentation_rounds > 1` silently truncates your training data back to one
round. Those are fixed here and
[documented](docs/differences-from-notebook.md), with every change to the
vendored upstream repos exported as a reviewable patch.

> **Status:** a personal research repo, published because the local port seemed
> worth sharing. It is not a library and there is no API. The default config
> trains one specific Spanish-accented wake phrase (`ouh_ke-i_nahbu`, a phonetic
> spelling of "ok nabu") until you change it.

## Quickstart

Needs Linux, an NVIDIA GPU, [`uv`](https://github.com/astral-sh/uv), and about
35 GB of disk.

```bash
./setup_env.sh                          # venv + upstream repos + base models  (~5 min)
./.venv/bin/python 01_test_voice.py     # listen to the wake word              (~1 min)
./.venv/bin/python 02_download_data.py  # all training data                    (~1 h, 33 GB)
./.venv/bin/python 03_make_config.py    # write my_model.yaml
./04_train.sh                           # generate -> augment -> train
./05_export_tflite.sh                   # .onnx -> .tflite
```

Models land in `my_custom_model/<wake_word>.{onnx,tflite}`.

Every step is re-runnable and skips work already done, so an interrupted download
or a failed stage resumes by just running it again. `04_train.sh` also takes a
single stage name (`clips`, `augment`, `train`).

> **This used to have an exception**, and it bit during a real run. The augment
> stage writes four feature files but upstream's skip check tested for
> `positive_features_train.npy` alone, so an interrupted run was treated as
> finished forever and the missing files surfaced much later as a
> `FileNotFoundError` in the train stage. Fixed by
> `patches/openwakeword-augment-resume-gate.patch`, which requires all four and
> says which were missing when it regenerates.

**Training a different wake word.** `03_make_config.py` takes `--word`,
`--examples`, `--steps` and `--penalty`; `02_download_data.py` takes
`--audioset-shards`, `--music-hours` and `--only`. Re-run `03_make_config.py`
after changing anything, since `04_train.sh` reads the YAML.

**Reproducible runs.** Add `seed: <int>` to the training config to pin python,
numpy and torch. The default is unseeded. A seed pins the augment stage
byte-for-byte; it does *not* guarantee a bit-identical training stage on GPU,
because cuDNN/cuBLAS kernel selection can still vary.

**Adding real recordings.** Synthetic TTS positives only get you so far. A config
can declare `real_clip_sources`, and `04_train.sh` will trim your own recordings
and duplicate them into the positive set between clip generation and
augmentation, so every copy draws its own background and impulse response:

```yaml
real_clip_sources:
  - dir: "${OWW_DATA_ROOT}/train"
    prefix: REAL
    copies: 202
expect_positive_train_total: 71412   # asserted before augmentation starts
```

See [`tools/data/inject_real_clips.py`](tools/data/inject_real_clips.py) and
[How the pipeline works](docs/pipeline.md#injecting-real-recordings).

**Paths.** Everything resolves relative to the repo, so a fresh clone works
anywhere. The one external input is a corpus of your own recordings, which lives
outside the checkout — set `OWW_DATA_ROOT` if yours is not at
`~/openwakeword-train-data`. See [`repo_paths.py`](repo_paths.py). Run scripts
from the repo root.

For what these stages actually do — and which part of the model is trained at
all — see [How the pipeline works](docs/pipeline.md).

## The research record

This repo grew out of nine generations of models trained for one household wake
word. That whole record — evaluation reports, raw scores, benchmark data,
per-generation configs and write-ups — lives on the
[**`experiments` branch**](https://github.com/Dethon/openwakeword-training/tree/experiments),
so `master` stays a pipeline rather than a lab notebook. It is worth a look if
you are about to run a similar experiment yourself:

| | result |
|---|---|
| The headline problem | recordings where **the music is louder than the wake word** score ~0.001 against a 0.5 threshold, in every generation |
| Harder augmentation + media negatives (v6) | not a clean win; gains did not transfer to real recordings |
| Real over-music clips in training (v7) | best model, still 0/7 on the loud clips |
| Milder augmentation for already-noisy clips (v8) | **refuted**, cleanly |
| Is the frozen front end the ceiling? | **leading suspect** — the wake word reaches the melspectrogram (AUC 0.82–0.95) and is largely lost by the embedding (0.61–0.64) |
| 24 h independent false-activation baseline | the deployed model is *not* zero-FA anywhere: 0.460/h, cut 64% by trigger-level 2 |

Two methodological notes worth more than any single number:

- **That front-end probe's first pass was wrong on all four of its pillars**, and
  the audit that reversed it is written up in full on that branch. The protocol's
  entire dynamic range turned out to be ~0.12 AUC, which made its headline 0.963
  reading uninterpretable. Three cheap controls would have caught it and none
  were run first.
- **Most of those results are underpowered, and say so.** n = 3 held-out loud
  clips; n = 7 loud clips from one speaker in one session.

The recordings behind all of it are of identifiable people, are not
redistributable, and are **not** in this repo on any branch.

## Layout

```
setup_env.sh            environment, upstream repos, pretrained feature extractors
01_test_voice.py        single TTS sample -> test_generation.wav
02_download_data.py     RIRs, AudioSet, FMA, precomputed features
03_make_config.py       -> my_model.yaml
04_train.sh             the three train.py stages -> logs/
05_export_tflite.sh     onnx2tf conversion in .venv-onnx2tf
resume_all.sh           run every remaining stage to completion, unattended
verify_clips.py         prune clips left empty by an interrupted run

repo_paths.py           REPO_ROOT / DATA_ROOT resolution
media_utils.py          audio and embedding helpers shared across the repo
compare_models.py       tflite model loading and scoring
embed_cache.py          self-validating persisted embedding cache
speech_fx.py            Lombard-effect and vocal-tract-length speech transforms

tools/data/             corpus and training-material builders
  inject_real_clips.py  adds a config's real recordings to positive_train
  make_eval_media.py    music/TV eval corpus, kept disjoint from training
  make_media_negatives.py, mine_hard_negatives.py, make_tv_mix.py
tools/recordings/       capturing, labelling, trimming and splitting real recordings
tools/eval/             eval_music.py (SNR sweep + FP streaming), score_clips_tflite.py
tests/                  equivalence and correctness guards (see tests/README.md)

my_model.yaml           the training config every stage reads
data/                   recording metadata: labels, splits, ground truth (no audio)
patches/                required changes to the vendored upstream repos
docs/                   pipeline reference and the notebook diff
```

Created at runtime and not tracked: `downloads/` (~33 GB of raw archives,
deletable once `02_download_data.py` finishes), `audioset_16k/`, `fma/`,
`mit_rirs/`, `tv_mix/`, `eval_media/`, `my_custom_model*/`, `logs/`, and the
`.npy` feature files in the repo root.

## The vendored upstream repos

`setup_env.sh` clones [openWakeWord](https://github.com/dscripka/openWakeWord)
and [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
into the repo. Both are gitignored, and **both need local changes to work** —
those are exported as patches in [`patches/`](patches/README.md), one file per
upstream commit, each with an explanation of why it is load-bearing.

Two are correctness fixes worth knowing about even outside this project:

- `openwakeword-v6-n-total-rounds.patch` — with `augmentation_rounds > 1`,
  upstream silently truncates the feature file back to one round's worth. No
  error, just undersized training data.
- `openwakeword-feature-extraction-performance.patch` — unpatched
  `device="gpu"` computes *different* melspectrograms than the CPU path every
  existing model was trained on, because that model is batch-sensitive.

## Documentation

| | |
|---|---|
| [How the pipeline works](docs/pipeline.md) | what runs, in what order, and which part is actually trained |
| [Differences from the notebook](docs/differences-from-notebook.md) | why this is not just the Colab notebook in a shell script |
| [`patches/README.md`](patches/README.md) | every vendored-repo change, and why |
| [`tests/README.md`](tests/README.md) | what each test proves and what it needs to run |
| [`experiments` branch](https://github.com/Dethon/openwakeword-training/tree/experiments) | the full research record: results, raw scores, benchmarks, per-generation configs |

## Data licensing

The training data downloaded by `02_download_data.py` carries a mix of licenses
and usage restrictions, so **models trained with it should be treated as
non-commercial, personal use only.** That is a property of the data, not of this
repo's code — see [NOTICE.md](NOTICE.md).

## License

[MIT](LICENSE) for the code in this repository. Third-party attributions and the
data caveat are in [NOTICE.md](NOTICE.md).
