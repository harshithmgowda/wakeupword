# patches/

`openwakeword/` and `piper-sample-generator/` are vendored git clones
(https://github.com/dscripka/openwakeword and
https://github.com/rhasspy/piper-sample-generator), gitignored by this outer repo. Any
change made inside either clone must also be exported here as a `.patch` file, or it
disappears on a fresh `git clone` of this repo (the clone would have to be re-cloned
from upstream and the local fixes redone by hand).

Each file below is `git -C <clone> diff <commit>~1 <commit>` for the local commit
named, generated with `git -C <clone> log` to find the commit and the same `diff`
command to export it — never `git diff HEAD~1`, which would also capture whatever
unrelated uncommitted changes happen to be sitting in the clone's working tree at
export time.

> **A note on the citations below.** Several patches justify themselves with
> measurements, and cite where those measurements live: `perf_audit/` benchmark
> scripts, `eval_results/` JSON, per-generation configs such as
> `my_model_v7.yaml`, and `tests/test_batch_assembly.py`. None of those are on
> `master` — they are on the
> [`experiments` branch](https://github.com/Dethon/openwakeword-training/tree/experiments),
> which holds this project's full research record. The paths are left as written
> so the evidence stays traceable; `git show experiments:<path>` will fetch any
> of them without switching branches. The patches themselves are complete here.

## Patches to `openwakeword/`

1. **openwakeword-v6-augmentation.patch** — commit `5e7a5e0`, "feat: configurable
   background-noise SNR/mode for augment_clips". Adds `background_min_snr_db`,
   `background_max_snr_db`, and `background_mode` params to `augment_clips` in
   `openwakeword/data.py` (previously hardcoded), and passes them through from
   `openwakeword/train.py`. Needed for the music/TV-mix robustness augmentation config
   used on this branch.

2. **openwakeword-v6-n-total-rounds.patch** — commit `6ab682b`, "fix: n_total not
   scaled by augmentation_rounds in compute_features_from_generator". Fixes the four
   `compute_features_from_generator(...)` calls in `openwakeword/train.py` to pass
   `n_total=len(os.listdir(...)) * config["augmentation_rounds"]` instead of the
   unmultiplied directory count. `compute_features_from_generator` uses `n_total` both
   to size the output memmap and as its early-stop condition, so with
   `augmentation_rounds > 1` the unpatched code silently truncated every feature file
   back down to exactly one round's worth — no error, just quietly wrong (undersized)
   training data. **Load-bearing for any config with `augmentation_rounds != 1`**,
   which includes the v6 config on this branch.

3. **openwakeword-local-runtime-fixes.patch** — commit `bdceed6`, "fix: local runtime
   fixes for DataLoader deadlock, X_val_fp memory, and best_val_fp visibility". Three
   changes to `openwakeword/train.py`, all found running the v6 pipeline on this
   machine rather than upstream:
   - **`DataLoader(num_workers=0)`** instead of upstream's `n_cpus` (12 here). A
     12-worker run deadlocked mid-epoch — all processes parked in
     `futex_wait_queue`, workers gone, GPU idle. This isn't just a flaky race to
     work around: `IterDataset` wraps one plain Python generator with no sharding,
     so with `num_workers > 0` every forked worker gets its own copy of the *same*
     iterator state and would emit the SAME batches, not a partition of them.
     `num_workers=0` is arguably the *correct* setting on the merits — the
     multi-worker path is broken by construction until `IterDataset` gets real
     sharding, not merely unstable. An `OWW_DATALOADER_WORKERS` env var is kept as
     an escape hatch for anyone who wants to try the upstream behavior anyway.
   - **`X_val_fp` built via `np.lib.stride_tricks.sliding_window_view`** instead of
     upstream's list comprehension of ~481,345 single-sample slices. The list
     comprehension materializes every slice before `np.array()` stacks them, so
     peak memory is roughly 2x the final ~2.96 GB array; `sliding_window_view` is a
     zero-copy view and `np.ascontiguousarray` pays for exactly one copy. Same
     values, same shape.
   - **Log `best_val_fp`** (the actual number), not just whether it crossed the
     threshold, at both negative-weight escalation points. Upstream logs only
     "Increasing weight on negative examples...", never the value driving that
     branch, which made it impossible to choose `target_fp_per_hour` from evidence
     instead of guesswork.

   **Load-bearing for `04_train.sh` to complete at all** on this machine: without
   the `num_workers=0` fix, `CONFIG=my_model_v6.yaml ./04_train.sh` deadlocks (GPU
   idle, all worker processes parked) rather than erroring, so it doesn't fail
   loudly — it just never finishes. Without the `X_val_fp` fix, the same run is at
   real risk of OOM-killing partway through validation on machines sized for the
   ~2.96 GB steady-state rather than the ~2x transient peak.

4. **openwakeword-noisy-clip-augmentation.patch** — commit `23a7716`, "feat: milder
   background augmentation for already-noisy clips". Adds `noisy_clip_pattern`,
   `noisy_background_min_snr_db` and `noisy_background_max_snr_db` to `augment_clips`
   in `openwakeword/data.py`, plus the three matching `config.get` keys in
   `openwakeword/train.py`'s `aug_kwargs` (so all four `augment_clips` calls get them).

   Why: v7's positive set includes 14 *real* recordings of the wake phrase spoken
   over music (duplicated 202x each as `REAL_MUSIC_*.wav`). Four of them
   (`speaker_b_34/35/38/40`) already have the music louder than the phrase, and the
   augment stage then mixes **another** background over them at
   `background_min_snr_db`..`background_max_snr_db` (-12..+10 dB in
   `my_model_v7.yaml`, `AddBackgroundNoise` p=0.9). The phrase is buried past
   recovery before the model ever sees it, so those 404 variants each act as noise
   labelled "positive" — v7 scores its own training clips 0.0008-0.0021 on them.
   This patch lets those clips keep background *variation* without being buried:
   `noisy_clip_pattern: "REAL_MUSIC_*"` routes them to a mild SNR range instead.

   How: `torch_audiomentations` applies a transform across the whole batch tensor,
   so exempting individual examples inside one batch isn't expressible. Instead
   `clip_paths` is partitioned — matching vs non-matching, `fnmatch` against each
   clip's **file name**, not its full path — and each partition is batched and
   augmented with its own `AddBackgroundNoise` range. Yielded batches are therefore
   no longer in `clip_paths` order once a pattern is set; that is fine, because the
   outputs go straight into an unordered feature pool. The batch-level `Compose` is
   now built by a small helper (`build_batch_augmentations`) so both ranges share one
   transform list rather than the previous copy-pasted if/else.

   **Backwards compatible by construction**: the default `noisy_clip_pattern=""`
   matches no clips, the noisy partition stays empty, and no second `Compose` is
   built — the surviving code path is the original one. Verified byte-for-byte:
   with all default augmentation probabilities and a fixed seed, pre-patch
   (`bdceed6`) and post-patch output arrays are identical (`np.array_equal` True).
   `tests/test_noisy_clip_augmentation.py` covers the exemption, the
   uneven-partition case (nothing dropped), and this no-op default.

5. **openwakeword-feature-extraction-performance.patch** — commit `82a4219`, "perf:
   GPU-correct feature extraction, background cache, validation loader". Touches
   `openwakeword/train.py`, `openwakeword/utils.py` and `openwakeword/data.py`.

   Why: the augment stage took **2 h 48 m of a 3 h 03 m run** while the GPU sat at a
   ~1–2% duty cycle (`logs/augment_clips.log`: four `Computing features` passes
   totalling 2 h 38 m at ~1.05 s per 32-clip batch). Four independent causes, each
   measured rather than guessed:

   - **`ncpu` was forced to 1 *because* torch saw CUDA.** The two upstream arguments
     were `device="gpu" if torch.cuda.is_available() else "cpu"` and
     `ncpu=n_cpus if not torch.cuda.is_available() else 1`, which assume "torch sees
     CUDA" implies "onnxruntime will use the GPU". They are independent: the pinned
     `onnxruntime==1.22.1` was the CPU-only wheel, so requesting
     `CUDAExecutionProvider` only warned and fell back to CPU — and because torch
     *did* see CUDA, the thread count was pinned to 1. Every feature in every model
     up to v8 was therefore computed single-threaded on a 24-core machine.
     `select_feature_device()` now probes onnxruntime instead, and
     `compute_features_from_generator` re-picks the thread count if the session it
     receives is not the one it asked for (the CUDA EP is a separate `.so` that fails
     to load unless cuBLAS/cuDNN are already resident — see the note below).
     Measured, one 32-clip batch: 520 ms at 1 thread, 90 ms at 8, flat above that.

   - **`melspectrogram.onnx` is batch-sensitive, and upstream's GPU path batched it.**
     Its ONNX output dims are named `('time', 1, <dim>, 32)` — it was authored for
     `batch_size=1`. Feeding it a stacked batch silently returns different numbers
     instead of failing: on the **CPU provider alone**, a fixed clip's output differs
     between batch=1 and batch≥2 by max 4.02 / mean 2.41 on values of magnitude ~42,
     identically for batch 2, 4 and 8. `_get_melspectrogram_batch` took the batched
     path whenever the provider was CUDA and mapped per clip otherwise, so
     `device="gpu"` computed **different — wrong — melspectrograms** than
     `device="cpu"`: 3.3% mean difference in the resulting embeddings, trained
     wake-word scores moved by up to 0.11, and 2 of 128 detections flipped at
     threshold 0.5. It is now always mapped per clip, which makes the GPU path match
     the CPU path every model in this repo was trained on — max score delta
     **0.0000**, zero decision flips. The ThreadPool is used on GPU too, where it
     hides per-call launch latency (32.5 ms at 1 thread vs 16.3 ms at 8).
     The CUDA provider is additionally pinned to `use_tf32=0`: TF32 is on by default
     for cuBLAS/cuDNN on Ampere and newer, and cost a further 6.9e-4 relative drift
     against the CPU provider for no measurable speed benefit (264.5 vs 262.4 ms).

     **This is the load-bearing correctness change in this patch.** Without it,
     installing `onnxruntime-gpu` silently changes the training data.

   - **`AddBackgroundNoise` re-read background audio from disk on every call**, which
     in `mode="per_example"` is once per clip: ~64 file operations per batch scattered
     over a 7398-file / 3.89 GB pool, ~507 ms of every ~1077 ms batch, against ~20 ms
     for every other augmentation combined. `build_background_cache` pre-decodes the
     pool into one `int16` memmap of `total_length` segments and
     `CachedAddBackgroundNoise` slices it (~5.6 ms, a ~90x reduction). Only
     `random_background` is overridden, so the SNR draw, RMS normalisation and mixing
     are all inherited unchanged. A per-source-file index preserves upstream's
     weighting — it draws a *file* uniformly then an offset inside it, so sampling
     segments uniformly instead would silently re-weight the pool toward longer files
     (a 30 s track yields 15 segments, a 10 s clip only 5).

   - **The false-positive validation set was rebuilt from scratch 60 times.**
     `DataLoader(TensorDataset(x, y), batch_size=len(y))` makes the default fetcher
     call `__getitem__` once per row and then `default_collate`-stack the results, so
     the 481,329-row / 2.96 GB set cost **8.35 s per iteration** — and it is
     re-iterated at every validation step (20 per training sequence × 3 sequences).
     That was ~9.3 min of a ~15 min training stage, and it is exactly the sawtooth in
     `logs/train_model.log` (95 it/s collapsing to 6 it/s at steps 164/175/186,
     319/329/339, 481/490/499 … against `val_steps` of 158, 316, 474 …).
     `TensorBatches` slices the contiguous tensor instead and holds it on-device:
     **0.005 s**, with a documented fallback to host memory if the allocation fails.

   Plus two overlap changes: `prefetch_generator` runs `augment_clips` on one
   background thread so it overlaps feature extraction (both release the GIL), and
   the training batch generator gets the same treatment. A *thread*, not the forked
   workers patch 3 had to disable — a thread shares the single generator rather than
   duplicating its iterator state, so batches are still produced exactly once, in
   order, and the deadlock cannot recur. `OWW_DATALOADER_WORKERS > 0` skips it.

   **Backwards compatible by construction**: `background_cache_path` defaults to `""`
   (upstream file-reading path untouched), and the melspec change makes the GPU path
   agree with the CPU path rather than altering the CPU path — verified byte-for-byte
   through `augment_clips` (`max int16 delta = 0`).
   `tests/test_background_cache.py` covers exactness of the cached samples,
   bit-identical output against upstream across 8 seeds, file-uniform sampling,
   refusal of a cache cut at the wrong `total_length`, and the no-op default.

   Escape hatches: `OWW_FEATURE_DEVICE=cpu|gpu`, `OWW_FEATURE_THREADS=N`,
   `OWW_PREFETCH=0`.

   Measured end to end on the real pipeline, identical work both ways:
   **1173 ms → 155 ms per 32-clip batch (7.54x)**, projecting the 2.63 h feature
   stage to ~0.35 h.

   **Depends on `onnxruntime-gpu`** for the GPU path (`setup_env.sh` installs
   `onnxruntime-gpu==1.23.2` in place of `onnxruntime`; only one may be installed,
   they provide the same `onnxruntime` module). No separate CUDA toolkit is needed —
   the CUDA EP dlopens the cuBLAS/cuDNN from the `nvidia-*` wheels torch already
   depends on, **which is why torch must be imported before onnxruntime**
   (`train.py` does). If it fails to load anyway, the patch logs the fallback and
   uses a properly threaded CPU path rather than silently dropping to one core.

6. **openwakeword-phase0-trainer-corrections.patch** — commit `8cbe50f`, "fix: phase 0
   trainer corrections -- train/eval modes, positive classes, ONNX input name". Touches
   `openwakeword/train.py` only. Three independent defects, all found reviewing the
   trainer before the v9 phase-4 architecture work, and all documented as verified in
   `docs/specs/2026-07-30-v9-roadmap.md`'s corrections table. **None of them
   changes what today's head trains to** — each blocks a change the roadmap needs.

   - **`.train()` / `.eval()` were never called anywhere** (zero grep hits upstream).
     Harmless for the shipped LayerNorm/Linear heads, which behave identically in both
     modes, and silently wrong for anything with BatchNorm or Dropout: the three
     validation blocks would normalise with the *validation batch's own* statistics and
     would additionally *mutate* the running estimates they exist to measure. Now
     `.train()` before the training loop, `.eval()` around the validation blocks,
     `.train()` after; plus `.eval()` on `average_models`' output and on `auto_train`'s
     merged model (`self.model` is left in training mode by `train_model`, and the merged
     model is validated and then exported).

     **A second BatchNorm blocker turned up while testing this, which `.eval()` does not
     fix:** `average_models` ran `averaged_model_dict[key] /= len(models)` over *every*
     state_dict entry, and BatchNorm registers `num_batches_tracked` as int64, so it
     raises `RuntimeError: result type Float can't be cast to the desired output type
     Long`. Checkpoint merging would therefore have **hard-failed at the end of the first
     BatchNorm training run**. Only floating-point entries are averaged now; integer
     counters are carried over from the deepcopied base model, which is what
     `torch.optim.swa_utils.AveragedModel` does with buffers. Provably a no-op for
     float-only state_dicts (every entry is floating point ⇒ `averageable` is every key).

   - **Label assignment hardcoded `positive -> 1`** and every other feature-data class
     `-> 0`, and `config["feature_data_files"]["positive"]` was only assigned *after* that
     loop, so the loop had no way to recognise a second positive class. Any extra
     positive class in `feature_data_files` was therefore trained as a **negative**, with
     no error and no warning — just a worse model. `make_label_transforms()` replaces the
     loop: a class is positive if its key is `positive`, starts with `positive_`, or is
     named in the config's new optional `positive_classes` list (a name in that list that
     matches no class logs a warning, since that is the same silent-typo failure). **This
     is the mechanism phase 3 needs** to give the real over-music recordings their own
     class with their own `batch_n_per_class` share, instead of 202× duplication.

   - **`torch.onnx.export` was called without `input_names`**, so the graph input was
     named after the first traced op (`onnx::Flatten_0` for the current Flatten-first
     head) and would change with any topology change — while `05_export_tflite.sh`
     hardcoded that name in onnx2tf's `-kat` flag, and onnx2tf **ignores an unknown
     `-kat` name**. Verified in this tree: passing a nonexistent name exits 0 and the
     input axes come out transposed, `[1,16,96] → [1,96,16]`, i.e. a model that loads,
     runs, and never activates. All three export calls now pass `input_names=["input"]`,
     and the script reads the name out of the ONNX and asserts the axis order survived.

   **Backwards compatible, verified rather than argued:** re-exporting the committed v7
   ONNX through the fixed script reproduces the committed
   `my_custom_model_v7/ouh_ke-i_nahbu.tflite` **byte for byte**
   (md5 `d660a0e760b70b3f6a0973fda0015719`) and scores bit-identically on every window of
   three clips (`speaker_a_close_01` 0.969546, `speaker_a_close_04` 0.989070, `speaker_b_36` 0.001423).
   `tests/test_train_eval_modes.py` covers the mode transitions, that validation leaves
   BatchNorm's running statistics untouched, and that averaging is still an exact
   elementwise mean; `tests/test_positive_class_labels.py` covers the label map, that
   v7's own class set is labelled exactly as before, and that a real batch built through
   `mmap_batch_generator` carries label 1 for a second positive class.

7. **openwakeword-pipeline-performance.patch** — commit `68cd81f`, "perf: augment stage
   2.42x -- soundfile clip loader + feature batch decoupled from augmentation batch".
   Touches `openwakeword/data.py`, `openwakeword/utils.py` and `openwakeword/train.py`.
   Three independent changes, **all bit-identical on real data**, measured end to end on
   9600 real v7 positive clips at `augmentation_batch_size: 32`, `device="gpu"`,
   `ncpu=8`, idle box, two runs each: ~~**39.19 / 39.47 s → 16.45 / 16.19 s = 2.42x**~~ —
   **CORRECTED 2026-07-30: 9600 clips was a small slice (v7's real `positive_train` pool
   is 74,240) and overstated the multiplier.** Re-measured at PRODUCTION SCALE — 70,000
   real clips, idle box, two full runs each, "before" reconstructed by pointing
   `_load_audio` back at `torchaudio.load` (the original bench script's `torchaudio.load`
   monkeypatch had gone stale — see patch 8): **241.22 / 242.15 s → 114.88 / 118.80 s =
   2.10x / 2.04x, ~2.07x average.** Still a real, substantial speedup, just not 2.42x.
   `perf_audit/bench_augment_e2e_v2.py` in the outer repo.

   - **`_load_audio()` replaces `torchaudio.load` in `augment_clips`** (the per-clip read
     and the per-batch RIR read). torchaudio 2.5's dispatcher re-resolves the backend and
     re-probes the file on every call, while the decode is ~0.1 ms — the same failure
     mode as the tflite thread-count bug, per-call library overhead swamping the work.
     500 real PCM_16 clips, best of 3, warm cache: **2.101 → 0.115 ms/clip (18.3x)**.
     `torch_audiomentations` already forces the soundfile backend at import, so this is
     the same libsndfile underneath: `torch.equal` holds on **367 real files across 9
     corpus pools** (v7 positives, `positives_es`, `positives_lombard`, `tv_mix`,
     `tv_mix_speech`, `fma`, `audioset_16k`, `mit_rirs`, the eval recordings). The
     torchaudio fallback is kept on purpose. The `sr` shadowing on the RIR line is
     upstream's and is deliberately **preserved, not fixed** — it is latent here because
     all 270 `mit_rirs` files are 16 kHz mono PCM_16.
   - **`FEATURE_BATCH_SIZE` (256) decouples feature extraction from augmentation.**
     Upstream set `batch_size = audio_data.shape[0]`, so `augmentation_batch_size` also
     pinned the embedding batch to 32. Raising *that* instead would be a data change —
     `build_batch_augmentations` uses `mode="per_batch"` and draws one RIR per batch, so
     128 clips would share one pitch shift and one RIR. Buffering already-augmented
     batches costs 16 MB and changes nothing. Proven on 256 real clips straight out of
     `augment_clips`: `np.array_equal` True for melspectrogram **and** embedding at
     batch_size 32/64/128/256 and ncpu 1/8/16, on **both** `CUDAExecutionProvider` and
     `CPUExecutionProvider`, `max_abs_diff` 0.0 everywhere. Feature step (GPU, ncpu=8):
     29.67 → 19.01 ms per 32 clips.
   - **`mmap_batch_generator(stack=False)` + `train._assemble_batch`** do the
     float16→float32 widening on the GPU rather than inside `np.vstack` on one CPU core
     (1152 of a v9r-shaped batch's 1264 rows are float16; 1148 of a production v7
     batch's 1252). Batch assembly alone, GPU otherwise idle, 400 steps after
     40 warm-up: **4.300 → 0.970 ms/step (4.43x)**.
     **REVERTED 2026-07-30 — patch 8 turns this back off in production.** This
     commit's own docstring already carried an honest caveat that the isolated
     4.43x didn't show up as a training-stage win, reporting a real
     `Model.train_model` run as roughly neutral (1.01x, 8.63 → 8.51 ms/step). An
     independent measurement review flagged that number as unpinned to a saved
     script and inconsistent with the isolated 4.43x, and a same-day re-measurement
     on the PRODUCTION config (`my_model_v7.yaml`'s real `batch_n_per_class`, the
     real 17.28 GB ACAV100M memmap, one continuous generator per variant so the
     ACAV100M page cache goes cold→warm exactly like a real 30,000-step run)
     resolved it the other way: **`stack=False` is 1.23x SLOWER, not faster**
     (8.55 → 10.54 ms/step; 1.22x cold / 1.29x warm) — see patch 8 below. The
     isolated micro-benchmark didn't have anything else queued on the GPU; in the
     real loop the four separate per-class `.to(device)` calls queue behind the
     previous step's still-draining CUDA work (backward/optimiser/the per-step
     `.cpu()` syncs), and that cost outweighs what moving the upcast off the CPU
     saves. `train.py`'s `__main__` is back on `stack=True`. `_assemble_batch` and
     the `stack` parameter are kept (not deleted) — both remain proven
     bit-identical and become load-bearing the moment `train_model`'s own
     ~5 ms/step of device syncs (`loss.detach().cpu().numpy()`,
     `self.recall(...).cpu()`, the per-step weight tensor) are addressed, which
     would change which side of this comparison wins.

   Guards: `tests/test_clip_loader_equivalence.py`, `tests/test_feature_batching.py`
   (which replays the *same* captured augmented buffers through both paths, since the
   augmentation is unseeded, and checks values, shape, row count and row order),
   `tests/test_batch_assembly.py` (120 real batches `torch.equal`, promotion rule matches
   `np.vstack` including the all-float16 case, and float16→float32 verified exact for all
   63,488 finite half values). A 3000-step **real** training run both ways from an
   identical seeded init produced all 10 `state_dict` tensors `torch.equal`.

8. **openwakeword-training-loop-regression-fix.patch** — commit `a435454`, "fix: revert
   item 7 training-loop regression + correct augment multiplier". Touches
   `openwakeword/data.py`, `openwakeword/train.py` and `openwakeword/utils.py` (comments
   only in the last). Two corrections from an independent measurement review of patch 7:

   - **Reverts patch 7's `stack=False` batch-assembly change in production.** Measured
     on the PRODUCTION config (`my_model_v7.yaml`'s real `batch_n_per_class`, the real
     17.28 GB ACAV100M memmap) through the real `Model.train_model` loop, one continuous
     generator per variant so the ACAV100M page cache goes cold→warm exactly like a real
     30,000-step run: `stack=False` (the GPU-side widen) is **1.23x SLOWER** than
     `stack=True` (upstream's `np.vstack`) — 8.55 → 10.54 ms/step (1.22x cold / 1.29x
     warm). `train.py`'s `__main__` is back on `stack=True`. `_assemble_batch` and the
     `stack` parameter are **kept, not deleted** — both remain proven bit-identical
     (`tests/test_batch_assembly.py`, unchanged) and become load-bearing the moment
     `train_model`'s own ~5 ms/step of device syncs are fixed. Full numbers, the second
     independent measurement that confirmed it, and the bounded look at a per-DTYPE-group
     alternative that was dropped (reorders rows relative to labels for v7's specific
     class/dtype ordering) are all in `_assemble_batch`'s docstring and the commit
     message.
   - **Corrects `FEATURE_BATCH_SIZE`'s docstring** with the production-scale augment
     multiplier from the entry above (~2.07x, not 2.42x) — no behaviour change.

   Depends on patch 7 (edits the same functions/comments patch 7 introduced), so must be
   applied after it.

   Verified: applying all eight `openwakeword/` patches in order to upstream `368c037`
   reproduces the clone's `a435454` tree exactly (scratch `git worktree`, never the live
   checkout) — resulting tree `9e7b4bf8038e9551961189e4f6fd7af2447eebb1` ==
   `git -C openwakeword rev-parse a435454^{tree}`.

## Patches to `piper-sample-generator/`

These are indexed separately because they are based on a different upstream clone
(`piper-sample-generator`, pre-patch commit `213d4d5`) and have no ordering
relationship with the `openwakeword/` patches above.

P1. **piper-sample-generator-tts-performance.patch** — commit `f90f0a7`, "perf+fix:
   delete dead np.trim_zeros (3.0x), vectorise remove_silence (6.1-6.4x), make
   auto_reduce_batch_size actually work". Three changes to `generate_samples.py`:
   - **Deletes a dead `audio_data = np.trim_zeros(...)`** whose result the very next
     statement overwrote, re-reading the same *untrimmed* source array. numpy 1.26's
     `trim_zeros` is a Python element loop and every clip is right-padded to
     `y_lengths.max()`, so it walked `T_max(batch) − own_length` zeros per clip:
     **9.107 ms/clip at batch 1, 27.154 at batch 25, 151.300 at batch 100**
     (15.13 s per 100-clip batch).
   - **Vectorises `remove_silence()`'s output assembly** — `np.concatenate` of the
     accepted frames instead of `x_new.extend(x[i:i+step].tolist())` per 30 ms frame.
     Identical `vad.is_speech` calls on identical byte slices, so identical decisions:
     **6.13x** on 60 `positives_es` clips (0.660 → 0.108 ms/clip) and **6.38x** on 60
     `tv_mix_speech` clips (3.002 → 0.470 ms/clip), `np.array_equal` on every clip.
   - **Makes `auto_reduce_batch_size` work.** It never has, for two reasons. (a) On
     WSL2 the NVIDIA driver's sysmem-fallback satisfies an oversized `cudaMalloc` out
     of *host* memory rather than failing, so `except torch.cuda.OutOfMemoryError`
     never fires — measured, three identical batch-100 calls reserved 16.35 / 24.83 /
     **33.10 GB on a 24 GB card** with `num_ooms=0`, taking 0.891 / 2.330 / **398.892
     s**. A new `cuda_memory_fraction=0.9` argument caps the allocator so it raises
     instead (pass `None` to restore the old behaviour). (b) The retry then crashed:
     it shrank the audio batch but left `clip_indexes_by_batch` at full length, so the
     following loop indexed `phoneme_samples` past its end. Verified by reverting just
     the truncation line: `IndexError: index 4 is out of bounds for dimension 0 with
     size 4`.

   The first two changes are **output-preserving**: 200 clips generated with
   `torch.manual_seed(0)` hash to the same `sha256 0129750bf839ab28` before and after
   (stock was verified reproducible across two runs first, so the A/B means
   something), while wall clock went **11.247 / 10.183 s → 3.466 / 3.367 s (3.0x)**.
   The third is a behaviour change *in the OOM path only* and is not
   output-preserving there — a reduced batch changes every clip's gain, because
   `audio_float_to_int16` normalises by `np.max(np.abs(audio))` over the whole
   `(batch, 1, T)` array. That is what `auto_reduce_batch_size` asks for by name;
   this patch only makes it reachable instead of silently thrashing.

   Verified: applying this patch to upstream `213d4d5` reproduces the
   `piper-sample-generator` clone's `f90f0a7` tree exactly (checked in a scratch
   `git worktree`, never the live checkout). `tests/test_piper_generate_samples.py`
   guards all three.

## Applying to a fresh clone

All **eight** `openwakeword/` patches were verified to apply cleanly, in order,
from the pre-patch upstream commit (`368c037`, the parent of `5e7a5e0`) —
applying all eight reproduces the `openwakeword` clone's `a435454` tree exactly,
byte for byte (verified in a scratch `git worktree`, never the live
`openwakeword/` checkout; resulting tree hash
`9e7b4bf8038e9551961189e4f6fd7af2447eebb1` matches
`git -C openwakeword rev-parse a435454^{tree}` exactly). The equivalent check at
the seven-patch state gave `41655c58a0746716ae0da5058ba3d811dbd8dab5` against
`68cd81f^{tree}`, at the six-patch state
`9118deddc57c7b221dcfb857b5b2f2afea3ce2b8` against `8cbe50f^{tree}`, and at the
five-patch state `b55b871af90277de2cb6b509627f4013b30f1bb6` against
`82a4219^{tree}`.

The single `piper-sample-generator/` patch is verified the same way against its
own upstream commit `213d4d5`; the two clones are independent and their patch
sets have no ordering relationship.

Patch 7 depends on patch 5: it edits the `compute_features_from_generator` body
that patch 5 rewrote (the `prefetch_generator` wrapper and the `ncpu`-carrying
`embed_clips` calls), so it must be applied after it. The first two edit
overlapping files (both touch `openwakeword/train.py`, in nearby but
non-overlapping line ranges) but not overlapping hunks, and `git apply` patches
by context rather than by blob hash, so order between them is not load-bearing.
The third patch also touches `train.py`, in yet other line ranges, so the same
reasoning applies to it. The fourth touches both `train.py` (inside the
`aug_kwargs` dict introduced by patch 1, so it *does* depend on patch 1) and
`data.py` (rewriting the block patch 1 created there, likewise dependent).
The fifth also depends on patch 1: it adds a key to the same `aug_kwargs` dict
and a branch inside the `build_batch_augmentations` helper patch 4 created, and
it rewrites the four `compute_features_from_generator` call sites patch 2 edited.
The sixth touches only `train.py`, and in line ranges none of the others touch —
`export_to_onnx`, `average_models`, `auto_train`'s merge, `export_model`, the
`train_model` loop, and the `__main__` label-transform block — but it *does*
depend on patch 3, which rewrote nearby lines inside `train_model`'s validation
section, so keep it last. **Patch 8 depends on patch 7**: it edits the exact
comments/logic patch 7 introduced in `data.py`, `train.py` and `utils.py`, so it
must be applied last. Chronologically patches 1 and 2 landed in that order
(`5e7a5e0` before `6ab682b`), patch 3 was written on top of both, patch 4 next,
patch 5 after that, patch 6 next, patch 7 after that, and patch 8 last, so apply
them in that order:

```
cd openwakeword
git apply ../patches/openwakeword-v6-augmentation.patch
git apply ../patches/openwakeword-v6-n-total-rounds.patch
git apply ../patches/openwakeword-local-runtime-fixes.patch
git apply ../patches/openwakeword-noisy-clip-augmentation.patch
git apply ../patches/openwakeword-feature-extraction-performance.patch
git apply ../patches/openwakeword-phase0-trainer-corrections.patch
git apply ../patches/openwakeword-pipeline-performance.patch
git apply ../patches/openwakeword-training-loop-regression-fix.patch
```

(The previous revision of this file's apply-sequence code block omitted patch 7
itself — a pre-existing documentation gap, fixed here alongside adding patch 8.
All eight were re-verified together in the scratch-worktree check above.)

A clean clone that skips patch 3 will either hang indefinitely (DataLoader
deadlock) or risk OOM (X_val_fp peak memory) on `CONFIG=my_model_v6.yaml
./04_train.sh` — it is not optional for reproducing the tree that actually
trained v6.

Skipping patch 5 is safe but slow: the run still completes, just at roughly
7.5x the augment-stage wall clock. What is *not* safe is installing
`onnxruntime-gpu` (as `setup_env.sh` now does) without patch 5 — the GPU path in
unpatched `_get_melspectrogram_batch` computes different melspectrograms from the
CPU path that trained v1–v8, silently and with no error. Either take both or
neither.

Skipping patch 6 reproduces v1–v8 exactly (it is a no-op for the head topology
and class set every model in this repo used), but a clone without it cannot train
a model with a second positive class, cannot use BatchNorm anywhere (validation
would corrupt its running statistics, and `average_models` would raise at the
final checkpoint merge), and its `.onnx` will carry a topology-dependent input
name that `05_export_tflite.sh` now reads rather than assumes. Take it.

9. **openwakeword-augment-resume-gate.patch** — commit `2d8dcdf`, "fix: augment stage
   must not treat a partial feature set as complete". One hunk in
   `openwakeword/train.py`. The augment stage writes **four** feature files
   (positive/negative x train/test), largest first, but the resume gate tested only
   `positive_features_train.npy`. An augment run interrupted after the first file landed
   was therefore considered finished forever: every later run logged `Openwakeword
   features already exist, skipping data augmentation and feature generation`, exited 0,
   and did nothing — and the three missing files surfaced only in the *train* stage, as
   `FileNotFoundError: positive_features_test.npy`.

   This is not hypothetical: it cost a full augment run during the v7 reproduction on
   2026-07-30 (`eval_results/v7_reproduction.md`). It is also invisible to a size check,
   because the memmaps are sized upfront from `n_total` — a half-written file has exactly
   the size a complete one would. The gate now requires all four files and logs which
   were missing when it regenerates. **Take it**: without it, an interrupted augment
   silently yields an undersized or unusable training set.

   Guarded by `tests/test_augment_resume_gate.py`, which drives the real
   `train.py --augment_clips` in a subprocess: a clean run writes all four, then three
   are deleted and the run repeated. Verified non-vacuous — with the gate reverted to
   upstream the test fails with the three files still missing.

10. **openwakeword-optional-seeding.patch** — commit `f27a127`, "feat: optional seed
    config key for reproducible runs". One hunk plus an `import random` in
    `openwakeword/train.py`. Nothing in the pipeline was seeded, so clip generation,
    augmentation draws and the training loop varied run to run and two runs of one config
    were two samples rather than one result — which is why the v7 reproduction could only
    be judged through the eval harness, never by comparing artifacts.

    `seed: <int>` in the training config now seeds python, numpy and torch (CPU + all
    CUDA devices), covering `augment_clips`' numpy draws, torch-audiomentations' torch
    draws, weight initialisation and batch sampling.

    **The default stays unseeded, deliberately.** Every model this repo has shipped was
    trained without a seed; seeding by default would change what an existing config
    produces and would quietly invalidate the description of how v1–v9 were built. It
    also does **not** by itself guarantee bit-identical GPU *training* — cuDNN/cuBLAS
    kernel selection and atomics can still vary. What was measured is the augment stage:
    same seed → byte-identical feature files, different seed → different, no seed → still
    varying (`tests/test_seeding.py`, all three end-to-end through the real
    `train.py --augment_clips`).

## Notes

- These `.patch` files are marked `-text -diff` in `.gitattributes` because their
  content is a diff of CRLF-native upstream source; without that, git's autocrlf
  normalization on commit would rewrite the embedded CRLFs and the patch would stop
  applying against the real (CRLF) source files. Keep any new patch added here under
  the same `patches/*.patch` glob, or add an explicit line if it needs a different
  name pattern.
- If you add another patch, update this file and re-verify the full apply sequence
  against a clean checkout of the commit the first patch's diff is based on.
