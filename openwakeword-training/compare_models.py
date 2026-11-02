#!/usr/bin/env python
"""Compare the trained model against the stock Home Assistant ok_nabu model.

Both models take the same (1, 16, 96) openWakeWord feature input, so they can
be run head-to-head on identical data.

Metrics mirror train.py's definitions exactly, so the numbers here line up with
what training reported:
  * recall / accuracy: threshold 0.5 over positive_features_test +
    negative_features_test
  * false positives per hour: predictions above 0.5 across stride-1 windows of
    validation_set_features.npy, divided by 11.3 hours (train.py's val_set_hrs)

    ./.venv-onnx2tf/bin/python compare_models.py
    ./.venv-onnx2tf/bin/python compare_models.py --clips ~/openwakeword-train-data/eval

`--clips` scores a directory of real recordings: each clip is scored with a
sliding window and counted as detected if its peak score clears the threshold.
That is the metric that actually reflects deployment, because the synthetic
test set shares a TTS distribution with the training data.

Point it at eval/ -- those 27 clips were held out of v5's training. Scoring
train/ instead measures recall on data the model was fitted to, and unused/
is worse still: rec_trimmed/ and rec_trimmed_25/ look like held-out sets but
are re-cuts of clips that are in train/.
"""
import argparse
import os
from pathlib import Path

import numpy as np

from media_utils import EMBED_NCPU

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = Path(__file__).parent.resolve()
VAL_SET_HRS = 11.3  # train.py's constant for validation_set_features.npy
THRESHOLD = 0.5


def _config_defaults():
    """Best-effort --ours/--features-dir defaults sourced from $CONFIG, the
    same env var 04_train.sh / 05_export_tflite.sh key off of, so e.g.
    `CONFIG=my_other_model.yaml ./compare_models.py` targets the model that
    config actually built without spelling out --ours/--features-dir by
    hand every time. Falls back to the original my_custom_model/ defaults
    (this script's behavior before CONFIG existed, and still what most
    eval scripts invoke explicitly via --ours/--features-dir
    anyway) whenever $CONFIG is unset, missing, or unparsable, so nothing
    changes for existing callers that don't set it.
    """
    default_tflite = "my_custom_model/ouh_ke-i_nahbu.tflite"
    default_features = "my_custom_model/ouh_ke-i_nahbu"
    config_path = os.environ.get("CONFIG")
    if not config_path:
        return default_tflite, default_features
    try:
        import yaml
        cfg = yaml.safe_load(open(config_path))
        model_name = cfg["model_name"]
        output_dir = cfg["output_dir"]
        return f"{output_dir}/{model_name}.tflite", f"{output_dir}/{model_name}"
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return default_tflite, default_features


#: Threads for heads whose export accepts a resized batch dim (stock models).
#: Only useful there; see load() for why the un-batchable path must be 1.
BATCHED_HEAD_THREADS = 8


def load(path):
    """Load a wake-word head, with a thread count chosen by whether it batches.

    num_threads=1 IS NOT A CONSERVATIVE DEFAULT -- IT IS THE FAST PATH.  Our
    onnx2tf exports bake a fixed batch dim, so `_can_batch()` is False and
    `predict()` issues one `invoke()` per 16x96 window.  On a ~100k-parameter
    graph the per-invoke thread-pool fan-out/join dominates completely: the
    work per call is microseconds and the synchronisation is hundreds of them.

    CORRECTED 2026-07-30 -- the original numbers here (9.82 us/window at
    num_threads=1, "251x" at os.cpu_count()) were an overstated headline: an
    independent re-measurement, then reproduced here, found the real nt=1
    cost is closer to 21 us/window -- off by ~2x -- and the 9.82 us figure was
    also inconsistent with this same commit's own end-to-end number (it
    predicts 9.9 s to classify the 24 h corpus's 1,006,521 windows; that run
    actually measured 18.5-21.6 s, which matches ~20.7 us/window, not 9.82).
    The fix itself (num_threads=1) is unaffected and still large -- only the
    multiplier was wrong.

    Measured on 21,029 REAL windows (60 real files drawn from the 23.924 h FA
    baseline corpus's embedding cache, one pass each after a 200-window
    warm-up, best of 2 reps, .venv-onnx2tf, idle box),
    my_custom_model_v7/ouh_ke-i_nahbu.tflite:

        num_threads=1     20.72 us/window   0.436 s
        num_threads=8    635.10 us/window  13.356 s    (30.6x slower)
        num_threads=24  2212.38 us/window  46.524 s   (106.8x slower)

    os.cpu_count() is 24 on this box, so the previous default was the ~107x
    case (not 251x). Output is unaffected: np.array_equal(nt=1, nt=8) and
    np.array_equal(nt=1, nt=24) are both True over all 21,029 windows.

    For heads that DO batch (stock_models/ok_nabu_v0.1.tflite), predict() feeds
    4096 windows per invoke and threading pays a little, so those get
    BATCHED_HEAD_THREADS:

        num_threads=1  6.05 us/window     num_threads=8  3.64 us/window
    """
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(path), num_threads=1)
    interp.allocate_tensors()
    if _can_batch(interp, interp.get_input_details()[0]["index"]):
        interp = Interpreter(model_path=str(path),
                             num_threads=BATCHED_HEAD_THREADS)
        interp.allocate_tensors()
    return interp


def _can_batch(interp, idx):
    """onnx2tf bakes a fixed batch dim into our export, so batching is model
    dependent. Results are identical either way -- only speed differs."""
    try:
        interp.resize_tensor_input(idx, (2, 16, 96))
        interp.allocate_tensors()
        interp.resize_tensor_input(idx, (1, 16, 96))
        interp.allocate_tensors()
        return True
    except Exception:
        interp.resize_tensor_input(idx, (1, 16, 96))
        interp.allocate_tensors()
        return False


def predict(interp, x, batch=4096):
    """Score an (N, 16, 96) array."""
    idx = interp.get_input_details()[0]["index"]
    scores = np.empty(len(x), dtype=np.float32)

    if not _can_batch(interp, idx):
        batch = 1
    for i in range(0, len(x), batch):
        chunk = np.ascontiguousarray(x[i:i + batch], dtype=np.float32)
        if chunk.shape[0] != interp.get_input_details()[0]["shape"][0]:
            interp.resize_tensor_input(idx, chunk.shape)
            interp.allocate_tensors()
        interp.set_tensor(interp.get_input_details()[0]["index"], chunk)
        interp.invoke()
        scores[i:i + batch] = interp.get_tensor(
            interp.get_output_details()[0]["index"]).reshape(-1)
    return scores


def eval_features(interp, root):
    pos = np.load(root / "positive_features_test.npy")
    neg = np.load(root / "negative_features_test.npy")
    p_pos, p_neg = predict(interp, pos), predict(interp, neg)

    tp = int((p_pos >= THRESHOLD).sum())
    fp = int((p_neg >= THRESHOLD).sum())
    recall = tp / len(p_pos)
    accuracy = (tp + (len(p_neg) - fp)) / (len(p_pos) + len(p_neg))
    return {"recall": recall, "accuracy": accuracy,
            "adv_neg_fp_rate": fp / len(p_neg),
            "median_pos_score": float(np.median(p_pos))}


def eval_false_positives(interp, path):
    feats = np.load(path)  # (N, 96)
    windows = np.lib.stride_tricks.sliding_window_view(feats, 16, axis=0)
    windows = np.ascontiguousarray(windows.transpose(0, 2, 1))
    scores = predict(interp, windows)
    n_fp = int((scores >= THRESHOLD).sum())
    return {"fp_total": n_fp, "fp_per_hour": n_fp / VAL_SET_HRS,
            "windows": len(windows)}


def eval_clips(interp, clip_dir, lead_in=0.0):
    """Peak sliding-window score per real recording.

    lead_in (seconds) prepends room tone taken from the clip's own tail. This
    matters: detection slides a ~2.0s window, so its earliest right edge is at
    1.96s. A recording whose utterance ends before that -- anything captured
    without a pause before speaking -- offers no window in the geometry the
    model was trained on, and scores near zero for reasons that have nothing to
    do with the model. Live audio always supplies that lead-in, so emulating it
    is what makes a finite file comparable to deployment.

    Tile the TAIL, not the head: the head of these clips often contains the
    speech onset, which contaminates the lead-in with the wake word itself.
    """
    import openwakeword.utils

    feat = openwakeword.utils.AudioFeatures(device="cpu")
    import soundfile as sf

    files = sorted(p for p in Path(clip_dir).rglob("*")
                   if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3", ".m4a"})
    if not files:
        return None

    peaks, per_file = [], []
    for f in files:
        audio, sr = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import scipy.signal
            g = np.gcd(int(sr), 16000)
            audio = scipy.signal.resample_poly(audio, 16000 // g, int(sr) // g)
        if lead_in > 0:
            n = int(lead_in * 16000)
            tail = audio[-n:] if len(audio) >= n else audio
            tone = np.tile(tail, int(np.ceil(n / max(1, len(tail)))) + 1)[:n]
            audio = np.concatenate([tone.astype(np.float32), audio])
        audio16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

        # ncpu is embed_clips' ThreadPool width over independent per-window ORT
        # calls -- bit-identical at any value (verified max_abs_diff 0.0 for
        # K in {1,4,8,12,16,24} on real clips). It is NOT AudioFeatures(ncpu=),
        # which sets onnxruntime thread counts and does change the numbers.
        emb = feat.embed_clips(audio16[None, :], batch_size=64,
                               ncpu=EMBED_NCPU)[0]  # (frames, 96)
        if emb.shape[0] < 16:
            per_file.append((f.name, float("nan")))
            continue
        w = np.lib.stride_tricks.sliding_window_view(emb, 16, axis=0)
        w = np.ascontiguousarray(w.transpose(0, 2, 1))
        peak = float(predict(interp, w).max())
        peaks.append(peak)
        per_file.append((f.name, peak))

    peaks = np.array(peaks)
    return {"n_clips": len(peaks),
            "detected": int((peaks >= THRESHOLD).sum()),
            "detection_rate": float((peaks >= THRESHOLD).mean()),
            "median_peak": float(np.median(peaks)),
            "per_file": per_file}


def main():
    default_ours, default_features_dir = _config_defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=default_ours)
    ap.add_argument("--stock", default="stock_models/ok_nabu_v0.1.tflite")
    ap.add_argument("--features-dir", default=default_features_dir)
    ap.add_argument("--fp-set", default="validation_set_features.npy")
    ap.add_argument("--clips", default=None,
                    help="directory of real recordings of the wake word")
    args = ap.parse_args()

    models = {"ours (ouh_ke-i_nahbu)": args.ours, "stock ok_nabu_v0.1": args.stock}
    results = {}
    for name, path in models.items():
        interp = load(path)
        r = {}
        r.update(eval_features(interp, Path(args.features_dir)))
        r.update(eval_false_positives(interp, args.fp_set))
        if args.clips:
            clips = eval_clips(load(path), args.clips)
            if clips:
                r["clips"] = clips
        results[name] = r

    w = 26
    print(f"\n{'metric':<{w}}" + "".join(f"{n:>26}" for n in results))
    print("-" * (w + 26 * len(results)))
    rows = [("recall (synthetic)", "recall", "{:.3f}"),
            ("accuracy (synthetic)", "accuracy", "{:.3f}"),
            ("adversarial-neg FP rate", "adv_neg_fp_rate", "{:.3f}"),
            ("median positive score", "median_pos_score", "{:.3f}"),
            ("false positives/hour", "fp_per_hour", "{:.3f}"),
            ("  (raw FP count)", "fp_total", "{:d}")]
    for label, key, fmt in rows:
        print(f"{label:<{w}}" + "".join(fmt.format(r[key]).rjust(26)
                                        for r in results.values()))

    if any("clips" in r for r in results.values()):
        print(f"\n{'REAL RECORDINGS':<{w}}")
        for label, key, fmt in [("clips tested", "n_clips", "{:d}"),
                                ("detected", "detected", "{:d}"),
                                ("detection rate", "detection_rate", "{:.3f}"),
                                ("median peak score", "median_peak", "{:.3f}")]:
            print(f"{label:<{w}}" + "".join(
                fmt.format(r["clips"][key]).rjust(26) for r in results.values()))

        names = [n for n in results]
        print(f"\n{'per-clip peak scores':<{w}}" + "".join(n[:24].rjust(26) for n in names))
        per = {n: dict(results[n]["clips"]["per_file"]) for n in names}
        for fname in sorted(per[names[0]]):
            print("  " + fname[:w - 2].ljust(w - 2) + "".join(
                f"{per[n][fname]:.3f}".rjust(26) for n in names))


if __name__ == "__main__":
    main()
