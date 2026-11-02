# Tests

These are **equivalence and correctness guards**, not a unit-test suite. Almost
every one exists because a specific change had to be proven not to alter what the
pipeline computes — a patch to the vendored `openwakeword`, a performance
rewrite, or a cache that must never serve a stale value.

They are plain scripts: each exits non-zero and prints `PASS ...` on success.
There is no pytest dependency.

```bash
./tests/run_tests.sh          # everything that can run here
./tests/run_tests.sh --light  # only the tests with no heavy prerequisites
./.venv/bin/python tests/test_media_utils.py    # or run one directly
```

Run them from the repo root.

## What each test needs

Most tests import the vendored `openwakeword` and need the training venv, so
run them with `./.venv/bin/python`. A few need real audio or a trained model and
will skip (or fail loudly) without it.

| test | needs | proves |
|---|---|---|
| `test_media_utils.py` | numpy only | mixing, segment picking and windowing helpers |
| `test_speech_fx.py` | numpy, scipy | Lombard/VTLP transforms and pause insertion |
| `test_make_tv_mix.py` | soundfile | `make_tv_mix.py` resume is RNG-safe (a real past bug) |
| `test_inject_real_clips.py` | `.venv` | real-clip injection: subset filter, prefix overlap, idempotency |
| `test_augment_resume_gate.py` | `.venv` | an interrupted augment run is regenerated, not skipped |
| `test_seeding.py` | `.venv` | `seed:` pins the augment stage; no seed still varies |
| `test_augment_patch.py` | `.venv` (torch, openwakeword) | the configurable-SNR / per-example-background patch |
| `test_noisy_clip_augmentation.py` | `.venv` | the `noisy_clip_pattern` exemption applies to the right clips |
| `test_background_cache.py` | `.venv` | the background-noise memmap cache matches per-file reads |
| `test_feature_batching.py` | `.venv` | `compute_features_from_generator` does not depend on `FEATURE_BATCH_SIZE` |
| `test_positive_class_labels.py` | `.venv` | extra positive classes really carry label 1 |
| `test_train_eval_modes.py` | `.venv` | the `.train()` / `.eval()` patch |
| `test_piper_generate_samples.py` | `.venv` + piper-sample-generator | the three piper patches |
| `test_clip_loader_equivalence.py` | `.venv` + audio corpora | `_load_audio` is bit-identical to `torchaudio.load` |
| `test_embed_cache.py` | `.venv` + audio | the embedding cache is exact and **refuses** anything it cannot prove |
| `test_eval_path_equivalence.py` | `.venv-onnx2tf` + audio | the two eval-path speedups change no score |

`test_embed_cache.py` is the one worth reading if you touch `embed_cache.py`: it
mutates a *copy* of the embedding code and asserts the cache detects it and
recomputes, so the guarantee is proven rather than asserted.

## Prerequisites

- `./setup_env.sh` creates `.venv` (training) and clones the upstream repos.
- `05_export_tflite.sh` creates `.venv-onnx2tf`, needed for anything touching
  tflite.
- Tests that read real recordings resolve them through `OWW_DATA_ROOT`
  (see [`repo_paths.py`](../repo_paths.py)); those are not in this repo.
- Tests that read generated corpora (`fma/`, `tv_mix/`, `my_custom_model*/`)
  need the pipeline to have been run.

Two further guards — `test_batch_assembly.py` (the GPU upcast path is
bit-identical to `np.vstack`) and `test_v9r_real_music_class.py` (an extra
positive class carries label 1 in real batches) — need this project's own
per-generation configs and feature files, so they live on the
[`experiments` branch](https://github.com/Dethon/openwakeword-training/tree/experiments)
alongside them.

Scratch directories named `tests/_tmp_*` are created and reused by some tests;
they are gitignored.
