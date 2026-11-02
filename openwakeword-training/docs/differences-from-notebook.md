# What differs from the upstream Colab notebook, and why

This project began as a local port of openWakeWord's [automatic training Colab notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb).
Everything below is a deliberate divergence from it.

[← back to the README](../README.md)

---

**AudioSet moved.** The notebook downloads `bal_train09.tar` and reads flac
files out of it. Those URLs now 404 — the HF dataset was restructured into
`data/bal_train/NN.parquet` with the audio embedded as flac bytes. `02` reads
the parquet shards directly.

**FMA's loader is dead.** The notebook streams `rudraml/fma` via
`datasets==2.14.6`. That's a legacy script-dataset which modern `datasets`
refuses to load, and the 2.14.6 pin drags in an incompatible
`fsspec`/`huggingface_hub` combination. Since that loader only wraps the
upstream `fma_small.zip`, `02` fetches the zip directly.

Together those two changes remove the need for `datasets` entirely —
`openwakeword` itself never imports it, only the notebook's download cells did.

**More background data.** The notebook grabs one AudioSet tar and 1 hour of
music, which is sized for its 1,000-example demo. Defaults here are 10 shards
(~14 h of noise) and 10 h of music, to give 50k examples and a
false-activation penalty of 1,000 enough negative variety to work against.

**`setuptools<81`.** `webrtcvad` and `pronouncing` both import `pkg_resources`,
which setuptools removed in 81.

**tflite conversion is a separate venv.** `train.py` ends by calling its
built-in converter, which imports the long-dead `onnx_tf` and raises
`ModuleNotFoundError`. This happens *after* the `.onnx` is written, so
`04_train.sh` recognises that specific failure and continues. `05` then does
the real conversion with `onnx2tf`, whose TensorFlow dependency conflicts with
the `numpy<2` pin the training environment needs — hence `.venv-onnx2tf`.

**GPU batch sizes.** `tts_batch_size` 50 → 100 and `augmentation_batch_size`
16 → 32, since sample generation and augmentation dominate wall-clock at 50k
examples and this runs on a local GPU rather than a Colab CPU box. Lower them
if you hit VRAM limits.

**Feature extraction runs on the GPU** (`onnxruntime-gpu`, installed by
`setup_env.sh` in place of the CPU wheel). This matters more than it sounds: with
the CPU-only wheel, `train.py` pinned its thread count to 1 *because* torch could
see CUDA, so every feature up to v8 was computed single-threaded on 24 cores and
the augment stage took 2 h 48 m of a 3 h 03 m run at a ~1–2% GPU duty cycle. The
augment stage is now ~7.5x faster end to end (1173 ms → 155 ms per 32-clip batch),
and the training stage no longer spends ~9 of its ~15 minutes re-collating the
same 2.96 GB validation tensor 60 times.

Two things to know about it. `torch` must be imported before `onnxruntime`, since
the CUDA execution provider dlopens the cuBLAS/cuDNN that torch's `nvidia-*`
wheels supply — `train.py` does, and if the provider fails to load anyway the code
logs it and falls back to a *properly threaded* CPU path. And the GPU path is only
correct with `patches/openwakeword-feature-extraction-performance.patch` applied:
upstream fed `melspectrogram.onnx` a whole batch when the provider was CUDA, but
that model is batch-sensitive, so unpatched `device="gpu"` silently computes
different melspectrograms than the CPU path every existing model was trained on.
Patched, GPU and CPU features agree to fp32 rounding (max wake-word score delta
0.0000, zero decision flips). Escape hatches: `OWW_FEATURE_DEVICE=cpu|gpu`,
`OWW_FEATURE_THREADS=N`, `OWW_PREFETCH=0`.

**Background-noise cache.** `background_cache_path` in the config points at a
memmap of pre-decoded fixed-length background segments, built automatically on
first use (~89 s, 3.8 GB for this pool). `AddBackgroundNoise` in `per_example`
mode otherwise re-reads a background file per clip — ~507 ms of every ~1077 ms
batch. Delete the `.npy` and its `.index.npz` sidecar to force a rebuild; both are
gitignored.
