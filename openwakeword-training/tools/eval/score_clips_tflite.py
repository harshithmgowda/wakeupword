#!/usr/bin/env python
"""Score a directory of short clips with a deployed openWakeWord .tflite head.

    ./.venv-onnx2tf/bin/python tools/eval/score_clips_tflite.py \
        --model my_custom_model_v7/ouh_ke-i_nahbu.tflite \
        --dir positives_es --limit 400 --out scores.json

WHY THE PADDING MATTERS.  Detection slides a 16-frame embedding window, whose
earliest right edge is at 1.96 s of audio.  A 1.2 s TTS clip therefore yields
*zero* windows and scores NaN for reasons that have nothing to do with the
model (compare_models.eval_clips documents the same trap for real recordings).
Synthetic positives are generated in digital silence, so the honest "quiet"
analogue of deployment is to surround the clip with a very low-level noise floor
rather than exact zeros -- exact zeros are not a room, and the melspectrogram
log-compresses them into a region no real microphone ever produces.

Clips in one batch are padded to a common length with more of the same floor;
extra trailing windows contain no speech, so the per-clip peak is unaffected.
"""
import sys
import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

# python puts the *script's* directory on sys.path, not the cwd, so the
# repo-root modules below have to be found explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from media_utils import EMBED_NCPU

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SR = 16000
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def load_head(path):
    """num_threads=1 is the FAST path, not a conservative one -- see
    compare_models.load for the full measurement (CORRECTED 2026-07-30 -- the
    original 9.82 us/window / 251x figures were overstated by ~2x).  Our
    onnx2tf exports bake a fixed batch dim, so predict() below falls into its
    per-window loop, where the tflite thread-pool fan-out/join per invoke()
    costs far more than the ~100k-parameter graph itself.  Measured on 21,029
    real windows: 20.72 us/window at num_threads=1 vs 2212.38 at
    num_threads=os.cpu_count() (24 here) -- ~107x -- with
    np.array_equal(nt=1, nt=24) True throughout."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(path), num_threads=1)
    interp.allocate_tensors()
    return interp


def predict(interp, x):
    """x: (n, 16, 96) float32 -> (n,) scores. The onnx2tf export bakes in a
    batch dim of 1, so feed one window at a time unless resizing works."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = np.asarray(x, dtype=np.float32)
    try:
        interp.resize_tensor_input(inp["index"], (x.shape[0], 16, 96))
        interp.allocate_tensors()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        res = interp.get_tensor(out["index"]).reshape(-1)
        interp.resize_tensor_input(inp["index"], (1, 16, 96))
        interp.allocate_tensors()
        return res
    except (ValueError, RuntimeError):
        interp.resize_tensor_input(inp["index"], (1, 16, 96))
        interp.allocate_tensors()
        res = np.empty(x.shape[0], dtype=np.float32)
        for i in range(x.shape[0]):
            interp.set_tensor(inp["index"], x[i: i + 1])
            interp.invoke()
            res[i] = interp.get_tensor(out["index"]).reshape(-1)[0]
        return res


def read16k(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import scipy.signal

        g = np.gcd(int(sr), SR)
        audio = scipy.signal.resample_poly(audio, SR // g, int(sr) // g)
    return audio.astype(np.float32)


def pad_quiet(audio, rng, lead_s=1.2, tail_s=0.6, floor_dbfs=-62.0):
    """Surround the clip with a low-level noise floor (see module docstring)."""
    amp = 10.0 ** (floor_dbfs / 20.0)
    lead = rng.normal(0.0, amp, int(lead_s * SR)).astype(np.float32)
    tail = rng.normal(0.0, amp, int(tail_s * SR)).astype(np.float32)
    return np.concatenate([lead, audio, tail])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", action="append", required=True,
                    help="clip directory (repeatable); each becomes one group")
    ap.add_argument("--label", action="append", default=None,
                    help="group label per --dir (default: directory name)")
    ap.add_argument("--limit", type=int, default=None, help="clips per group")
    ap.add_argument("--file-list", default=None,
                    help="restrict to these file names (one per line)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--lead-s", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels = args.label or [Path(d).name for d in args.dir]
    if len(labels) != len(args.dir):
        raise SystemExit("--label count must match --dir count")

    keep = None
    if args.file_list:
        keep = {l.strip() for l in open(args.file_list) if l.strip()}

    import openwakeword.utils

    feat = openwakeword.utils.AudioFeatures(device="cpu")
    interp = load_head(args.model)
    rng = np.random.default_rng(args.seed)

    results = {"model": args.model, "threshold": args.threshold,
               "lead_s": args.lead_s, "groups": {}}
    for label, d in zip(labels, args.dir):
        files = sorted(p for p in Path(d).iterdir()
                       if p.suffix.lower() in AUDIO_EXT)
        if keep is not None:
            files = [p for p in files if p.name in keep]
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"{label}: no clips, skipping")
            continue

        peaks, names = [], []
        for i in range(0, len(files), args.batch):
            chunk = files[i: i + args.batch]
            padded = [pad_quiet(read16k(p), rng, lead_s=args.lead_s) for p in chunk]
            # Round up to a whole melspectrogram hop (160 samples).
            # openWakeWord's _get_melspectrogram_batch preallocates
            # int(ceil(n/160 - 3)) frames and assigns the model's real output into
            # it; for lengths that are not a multiple of 160 that prediction is off
            # by one and the assignment raises a broadcast error. Verified: every
            # multiple of 160 predicts exactly, non-multiples are where it breaks.
            n = max(len(a) for a in padded)
            n = int(np.ceil(n / 160.0) * 160)
            amp = 10.0 ** (-62.0 / 20.0)
            block = np.stack([
                np.concatenate([a, rng.normal(0, amp, n - len(a)).astype(np.float32)])
                if len(a) < n else a for a in padded
            ])
            block16 = (np.clip(block, -1, 1) * 32767).astype(np.int16)
            # ncpu = embed_clips' ThreadPool width only (bit-identical at any
            # value); NOT AudioFeatures(ncpu=), which changes the numbers.
            embs = feat.embed_clips(block16, batch_size=64, ncpu=EMBED_NCPU)
            for p, emb in zip(chunk, embs):
                if emb.shape[0] < 16:
                    peaks.append(float("nan"))
                else:
                    w = np.lib.stride_tricks.sliding_window_view(emb, 16, axis=0)
                    w = np.ascontiguousarray(w.transpose(0, 2, 1))
                    peaks.append(float(predict(interp, w).max()))
                names.append(p.name)
            print(f"  {label}: {min(i + args.batch, len(files))}/{len(files)}",
                  flush=True)

        arr = np.asarray(peaks, dtype=float)
        ok = arr[np.isfinite(arr)]
        results["groups"][label] = {
            "dir": str(d),
            "n": int(len(arr)),
            "n_nan": int(np.isnan(arr).sum()),
            "detected": int((ok >= args.threshold).sum()),
            "detection_rate": float((ok >= args.threshold).mean()) if len(ok) else None,
            "mean_peak": float(ok.mean()) if len(ok) else None,
            "median_peak": float(np.median(ok)) if len(ok) else None,
            "p10_peak": float(np.percentile(ok, 10)) if len(ok) else None,
            "p90_peak": float(np.percentile(ok, 90)) if len(ok) else None,
            "per_file": dict(zip(names, peaks)),
        }
        g = results["groups"][label]
        print(f"{label}: n={g['n']} det={g['detected']} ({g['detection_rate']:.3f}) "
              f"median={g['median_peak']:.4f} mean={g['mean_peak']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
